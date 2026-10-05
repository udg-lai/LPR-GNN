from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .checkpoints import load_checkpoint, load_model_state, model_state_dict
from .data import RcpspGraphDataset, batch_graph_count
from .losses import multilabel_top1_loss
from .metrics import MetricAccumulator
from .model import RcpspGNN
from .instance_evaluation import load_instance_splits, evaluate_instances, final_instance_evaluation


# Load YAML settings and resolve project-relative paths.
# Parameters: path: input/output path
# Returns: dict.
def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The configuration file must contain a YAML mapping.")
    return config


# Seed Python, NumPy and PyTorch random generators.
# Parameters: seed: random seed
# Returns: None; updates state or writes output.
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Select CPU or CUDA from the requested setting.
# Parameters: requested: requested device name
# Returns: torch.device.
def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


# Build a graph loader with the epoch and variant policy.
# Parameters: config: project settings; split: dataset split name; repeat: repeat dataset stream
#   epoch: epoch index; deterministic: disable evaluation shuffling; variant_policy: augmentation filter
# Returns: DataLoader.
def make_loader(
    config: Mapping,
    split: str,
    repeat: bool,
    *,
    epoch: int | None = None,
    deterministic: bool = False,
    variant_policy: str | None = None,
) -> DataLoader:
    data_config = config["data"]
    training_config = config["training"]
    directory = data_config[f"{split}_dir"]
    is_training = split == "train"
    num_workers = int(data_config.get("num_workers", 0))

    seed = int(config.get("seed", 42))
    if is_training and epoch is not None and not deterministic:
        # Rebuilding the training loader with seed_base + epoch guarantees a
        # reproducible, different file/graph order at every epoch boundary.
        seed += int(epoch)

    variant_config = data_config.get("variants", {})
    if not isinstance(variant_config, Mapping):
        raise ValueError("data.variants must be a mapping.")
    default_variant_policy = "all" if is_training else "original"
    if variant_policy is None:
        variant_policy = str(variant_config.get(split, default_variant_policy))
    else:
        # Final reporting evaluates an explicit, fixed variant matrix and must
        # not inherit the per-split policy used during training.
        variant_policy = str(variant_policy).strip().lower()
    legacy_variant_group_size = int(variant_config.get("legacy_group_size", 3))

    dataset = RcpspGraphDataset(
        directory,
        repeat=repeat,
        shuffle_files=is_training
        and not deterministic
        and bool(data_config.get("shuffle", True)),
        shuffle_buffer=(
            int(data_config.get("shuffle_buffer", 0))
            if is_training and not deterministic
            else 0
        ),
        seed=seed,
        variant_policy=variant_policy,
        legacy_variant_group_size=legacy_variant_group_size,
    )

    kwargs = dict(
        dataset=dataset,
        batch_size=int(training_config["batch_size"]),
        num_workers=num_workers,
        pin_memory=bool(data_config.get("pin_memory", True)) and torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(data_config.get("prefetch_factor", 2))
    return DataLoader(**kwargs)


# Construct the configured priority network.
# Parameters: config: project settings
# Returns: RcpspGNN.
def build_model(config: Mapping) -> RcpspGNN:
    model_config = config["model"]
    return RcpspGNN(
        state_dim=int(model_config.get("state_dim", 32)),
        num_iterations=int(model_config.get("num_iterations", 8)),
        readout_hidden=int(model_config.get("readout_hidden", 32)),
        readout_dropout=float(model_config.get("readout_dropout", 0.15)),
    )


# Set learning rate from the number of processed graphs.
# Parameters: optimizer: training optimizer; initial_lr: initial learning rate
#   decay_rate: learning-rate decay factor; decay_steps_graphs: graphs per decay interval
#   graphs_seen: processed graph count
# Returns: float.
def set_exponential_lr(
    optimizer: torch.optim.Optimizer,
    *,
    initial_lr: float,
    decay_rate: float,
    decay_steps_graphs: int,
    graphs_seen: int,
) -> float:
    learning_rate = float(initial_lr) * float(decay_rate) ** (
        float(graphs_seen) / max(int(decay_steps_graphs), 1)
    )
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


# Extract configured loss coefficients.
# Parameters: config: project settings
# Returns: Dict[str, float].
def _loss_kwargs(config: Mapping) -> Dict[str, float]:
    loss_config = config["loss"]
    return {
        "bce_weight": float(loss_config.get("bce_weight", 1.0)),
        "top1_weight": float(loss_config.get("top1_weight", 1.0)),
        "pos_weight": float(loss_config.get("pos_weight", 1.0)),
    }


# Extract Border labels, segment offsets and graph membership.
# Parameters: batch: batched graphs
# Returns: labels, Border offsets, graph indices and graph count.
def _batch_targets(batch: HeteroData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    border = batch["border"]
    return border.y, border.batch, border.ptr, batch_graph_count(batch)


# Evaluate graph-level loss and selection metrics.
# Parameters: model: priority network; loader: graph data loader; device: compute device
#   config: project settings; max_graphs: evaluation graph limit
# Returns: Dict[str, float].
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Mapping,
    max_graphs: int,
) -> Dict[str, float]:
    model.eval()
    loss_sums = torch.zeros(3, dtype=torch.float64, device=device)
    graph_total = 0
    metrics = MetricAccumulator(
        threshold_logit=float(config["loss"].get("threshold_logit", 0.0))
    )

    with torch.inference_mode():
        for batch in loader:
            if max_graphs > 0 and graph_total >= max_graphs:
                break
            batch = batch.to(device, non_blocking=device.type == "cuda")
            labels, border_batch, border_ptr, graph_count = _batch_targets(batch)
            logits = model(batch)
            losses = multilabel_top1_loss(
                logits,
                labels,
                border_batch,
                graph_count,
                **_loss_kwargs(config),
            )
            loss_sums += torch.stack(
                (losses.total.detach(), losses.bce.detach(), losses.top1.detach())
            ).to(torch.float64) * graph_count
            graph_total += graph_count
            metrics.update(logits, labels, border_ptr)

    if graph_total == 0:
        raise RuntimeError("Validation dataset yielded no valid graph.")

    result = metrics.compute()
    loss_sum, bce_sum, top1_loss_sum = (
        float(value) for value in loss_sums.cpu().tolist()
    )
    result.update(
        {
            "loss": loss_sum / graph_total,
            "bce_loss": bce_sum / graph_total,
            "top1_loss": top1_loss_sum / graph_total,
            "graphs": float(graph_total),
        }
    )
    return result


# Compute performance and detailed diagnostics in one dataset pass.
# Parameters: model: priority network; loader: graph data loader; device: compute device
#   config: project settings
# Returns: compact metrics and full diagnostics.
def evaluate_final_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Mapping,
) -> tuple[Dict[str, float], Dict[str, float | int | str]]:
    """Compute performance and detailed diagnostics in one dataset pass."""
    from .diagnostics import FullSelectionDiagnostics

    threshold_logit = float(config["loss"].get("threshold_logit", 0.0))
    model.eval()
    loss_sums = torch.zeros(3, dtype=torch.float64, device=device)
    graph_total = 0
    performance_accumulator = MetricAccumulator(threshold_logit=threshold_logit)
    selection_accumulator = FullSelectionDiagnostics(
        threshold_logit=threshold_logit
    )

    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            labels, border_batch, border_ptr, graph_count = _batch_targets(batch)
            logits = model(batch)
            losses = multilabel_top1_loss(
                logits,
                labels,
                border_batch,
                graph_count,
                **_loss_kwargs(config),
            )
            loss_sums += torch.stack(
                (losses.total.detach(), losses.bce.detach(), losses.top1.detach())
            ).to(torch.float64) * graph_count
            graph_total += graph_count
            performance_accumulator.update(logits, labels, border_ptr)
            selection_accumulator.update(logits, labels, border_ptr)

    if graph_total == 0:
        raise RuntimeError("Final evaluation dataset yielded no valid graph.")

    performance = performance_accumulator.compute()
    loss_sum, bce_sum, top1_loss_sum = (
        float(value) for value in loss_sums.cpu().tolist()
    )
    performance.update(
        {
            "loss": loss_sum / graph_total,
            "bce_loss": bce_sum / graph_total,
            "top1_loss": top1_loss_sum / graph_total,
            "graphs": float(graph_total),
        }
    )
    return performance, selection_accumulator.compute()


# Recompute and write both fixed five-set final evaluation tables.
# Parameters: model: priority network; config: project settings; device: compute device
#   output_dir: output directory
# Returns: diagnostics for the five evaluation datasets.
def generate_final_selection_diagnostics(
    model: nn.Module,
    config: Mapping,
    device: torch.device,
    output_dir: str | Path,
) -> Dict[str, Dict[str, float | int | str]]:
    """Recompute and write both fixed five-set final evaluation tables."""
    from .diagnostics import (
        write_full_selection_diagnostics,
        write_performance_metrics_table,
    )

    diagnostics: Dict[str, Dict[str, float | int | str]] = {}
    performance_metrics: Dict[str, Dict[str, float]] = {}
    evaluation_specs = (
        ("train_all", "Train", "train", "all"),
        ("validation_all", "Valid. all", "validation", "all"),
        ("validation_original", "Valid. original", "validation", "original"),
        ("test_all", "Test all", "test", "all"),
        ("test_original", "Test original", "test", "original"),
    )
    for key, label, split, variant_policy in evaluation_specs:
        print(f"Final deterministic diagnostics: {label}")
        loader = make_loader(
            config,
            split,
            repeat=False,
            deterministic=True,
            variant_policy=variant_policy,
        )
        performance_metrics[key], diagnostics[key] = evaluate_final_dataset(
            model, loader, device, config
        )

    caption = (
        "Activity-selection diagnostics of the selected checkpoint, "
        "recomputed deterministically after training for train/all, "
        "validation/all, validation/original, test/all, and test/original."
    )
    output_dir = Path(output_dir)
    write_full_selection_diagnostics(
        diagnostics,
        output_dir / "full_selection_diagnostics.tex",
        caption=caption,
        columns=tuple((key, label) for key, label, _, _ in evaluation_specs),
    )
    write_performance_metrics_table(
        performance_metrics,
        output_dir / "train_validation_test_metrics.tex",
        rows=tuple((key, label.replace("Valid.", "Validation")) for key, label, _, _ in evaluation_specs),
        caption=(
            "Performance of the selected checkpoint, recomputed deterministically "
            "for train/all, validation/all, validation/original, test/all, and "
            "test/original."
        ),
    )
    return diagnostics


# Append an epoch record to training history.
# Parameters: path: input/output path; row: history record
# Returns: None; updates state or writes output.
def _append_history(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# Train and select weights by mean validation rollout makespan.
# Parameters: config_path: YAML settings path; resume: optional resume checkpoint
# Returns: None; updates state or writes output.
def train(config_path: str | Path, resume: str | Path | None = None) -> None:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    project_root = config_path.parent
    config['_project_root'] = str(project_root)

    for key in ("train_dir", "validation_dir", "test_dir"):
        value = Path(config["data"][key])
        if not value.is_absolute():
            config["data"][key] = str((project_root / value).resolve())
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config['output_dir'] = str(output_dir)
    splits, split_hash = load_instance_splits(config)
    resume = resume or config['training'].get('resume_checkpoint')
    if resume is None and any((output_dir / name).exists() for name in ('history.csv', 'best.pt', 'last.pt')):
        raise FileExistsError('Existing training run: choose a new output_dir or explicitly resume.')

    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = resolve_device(str(config.get("device", "auto")))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    model = build_model(config).to(device)
    if bool(config["training"].get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model, dynamic=True)  # type: ignore[assignment]

    optimizer_config = config["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_config["initial_learning_rate"]),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
    )

    start_epoch = 0
    graphs_seen = 0
    best_makespan = math.inf
    if resume is not None:
        resume_path = Path(resume)
        if not resume_path.is_absolute():
            resume_path = (project_root / resume_path).resolve()
        checkpoint = load_checkpoint(resume_path, device)
        if checkpoint.get('selection_metric') != 'validation_mean_makespan' or checkpoint.get('split_manifest_sha256') != split_hash:
            raise ValueError('Resume requires a makespan-selected checkpoint with the same instance split.')
        load_model_state(model, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        graphs_seen = int(checkpoint.get("graphs_seen", 0))
        best_makespan = float(checkpoint['best_mean_makespan'])

    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    accumulation_steps = max(
        1, int(config["training"].get("gradient_accumulation_steps", 1))
    )
    max_grad_norm = float(config["training"].get("max_grad_norm", 0.0))
    graphs_per_epoch = int(config["training"]["graphs_per_epoch"])
    batch_size = int(config["training"]["batch_size"])
    consume_full_dataset = graphs_per_epoch <= 0
    steps_per_epoch = (
        None if consume_full_dataset else math.ceil(graphs_per_epoch / batch_size)
    )
    epochs = int(config["training"]["epochs"])
    validation_frequency = int(config["training"].get("validation_frequency", 1))
    if validation_frequency < 1:
        raise ValueError('validation_frequency must be positive.')
    progress_log_interval = max(
        1, int(config["training"].get("progress_log_interval", 100))
    )

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyG real graph batch size: {batch_size}")
    if consume_full_dataset:
        print("Graphs per epoch: all selected training graphs (one complete pass)")
    else:
        print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    history_path = output_dir / "history.csv"
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, epochs):
        # A fresh iterable dataset starts each epoch at a newly shuffled order.
        # Validation/test loaders keep the base seed and never shuffle.
        train_loader = make_loader(
            config, "train", repeat=not consume_full_dataset, epoch=epoch
        )
        model.train()
        metrics = MetricAccumulator(
            threshold_logit=float(config["loss"].get("threshold_logit", 0.0))
        )
        loss_sums = torch.zeros(3, dtype=torch.float64, device=device)
        epoch_graphs = 0
        start_time = time.perf_counter()

        if consume_full_dataset:
            progress_source = train_loader
        else:
            train_iterator = iter(train_loader)
            progress_source = range(steps_per_epoch)  # type: ignore[arg-type]
        progress = tqdm(progress_source, desc=f"Epoch {epoch + 1}/{epochs}")
        processed_steps = 0
        for step, progress_item in enumerate(progress):
            batch = progress_item if consume_full_dataset else next(train_iterator)
            batch = batch.to(device, non_blocking=device.type == "cuda")
            labels, border_batch, border_ptr, graph_count = _batch_targets(batch)

            learning_rate = set_exponential_lr(
                optimizer,
                initial_lr=float(optimizer_config["initial_learning_rate"]),
                decay_rate=float(optimizer_config.get("decay_rate", 1.0)),
                decay_steps_graphs=int(optimizer_config.get("decay_steps_graphs", 1)),
                graphs_seen=graphs_seen,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16,
                enabled=amp_enabled,
            ):
                logits = model(batch)
                losses = multilabel_top1_loss(
                    logits,
                    labels,
                    border_batch,
                    graph_count,
                    **_loss_kwargs(config),
                )
                scaled_loss = losses.total / accumulation_steps

            scaler.scale(scaled_loss).backward()
            should_step = ((step + 1) % accumulation_steps == 0) or (
                steps_per_epoch is not None and step + 1 == steps_per_epoch
            )
            if should_step:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            graphs_seen += graph_count
            epoch_graphs += graph_count
            processed_steps = step + 1
            loss_sums += torch.stack(
                (losses.total.detach(), losses.bce.detach(), losses.top1.detach())
            ).to(torch.float64) * graph_count
            metrics.update(logits, labels, border_ptr)

            is_last_capped_batch = (
                steps_per_epoch is not None and step + 1 == steps_per_epoch
            )
            if (step + 1) % progress_log_interval == 0 or is_last_capped_batch:
                current_metrics = metrics.compute()
                progress.set_postfix(
                    loss=f"{float(loss_sums[0].item()) / epoch_graphs:.4f}",
                    top1=f"{current_metrics['top1_any_positive']:.3f}",
                    lr=f"{learning_rate:.2e}",
                )

        if epoch_graphs == 0:
            raise RuntimeError("Training dataset yielded no valid graph.")

        # With an exhaustible full-dataset loader, the last accumulation group
        # may contain fewer batches than gradient_accumulation_steps.
        if consume_full_dataset and processed_steps % accumulation_steps != 0:
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_metrics = metrics.compute()
        loss_sum, bce_sum, top1_loss_sum = (
            float(value) for value in loss_sums.cpu().tolist()
        )
        train_metrics.update(
            {
                "loss": loss_sum / epoch_graphs,
                "bce_loss": bce_sum / epoch_graphs,
                "top1_loss": top1_loss_sum / epoch_graphs,
                "graphs": float(epoch_graphs),
            }
        )
        progress.set_postfix(
            loss=f"{train_metrics['loss']:.4f}",
            top1=f"{train_metrics['top1_any_positive']:.3f}",
            lr=f"{learning_rate:.2e}",
        )

        validation_metrics = dict(instances=0, mean_makespan=math.nan, seconds=0.0)
        if (epoch + 1) % validation_frequency == 0 or epoch + 1 == epochs:
            validation_metrics = evaluate_instances(
                model, splits['validation'], device, config,
                output_dir / 'validation_rollouts' / f'epoch_{epoch + 1:04d}.json',
            )

        elapsed = time.perf_counter() - start_time
        row: Dict[str, object] = {
            "epoch": epoch + 1,
            "seconds": round(elapsed, 3),
            "graphs_seen": graphs_seen,
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in validation_metrics.items()})
        _append_history(history_path, row)

        validation_makespan = validation_metrics['mean_makespan']
        is_best = validation_makespan < best_makespan
        if is_best:
            best_makespan = validation_makespan

        checkpoint = {
            "epoch": epoch,
            "graphs_seen": graphs_seen,
            "best_mean_makespan": best_makespan,
            "selection_metric": "validation_mean_makespan",
            "split_manifest_sha256": split_hash,
            "model_state": model_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
        }
        torch.save(checkpoint, output_dir / "last.pt")

        if is_best:
            torch.save(checkpoint, output_dir / "best.pt")

        print(f"Epoch {epoch + 1}: train={train_metrics}")
        if validation_metrics:
            print(f"Epoch {epoch + 1}: validation={validation_metrics}")
        if config['training'].get('plot_history', True):
            # Plot only recorded metrics; no extra rollout or model evaluation.
            try:
                import matplotlib.pyplot as plt
                from .training_plots import plot_training_history
                figure = plot_training_history(history_path, output_dir / 'training_evolution.png')
                if figure is not None:
                    plt.close(figure)
            except Exception as exc:
                print(f'Warning: progress plot unavailable: {exc}')

    if True:  # Always test the selected checkpoint after training.
        best_checkpoint_path = output_dir / "best.pt"
        if not best_checkpoint_path.exists():
            raise RuntimeError(
                "Cannot generate final diagnostics because outputs/best.pt is missing."
            )
        best_checkpoint = load_checkpoint(best_checkpoint_path, device)
        load_model_state(model, best_checkpoint)
        summary = final_instance_evaluation(model, config, device, output_dir, splits)
        print('Final RCPSP evaluation:', summary)


# Run training from command-line arguments.
# Parameters: none.
# Returns: None; updates state or writes output.
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train the RCPSP graph model.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume")
    args = parser.parse_args()
    train(args.config, resume=args.resume)


if __name__ == "__main__":
    main()
