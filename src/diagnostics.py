from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn


def _first_segment_argmax(logits: Tensor, graph_index: Tensor, graph_count: int) -> Tensor:
    """Return the first maximum node index in every graph segment."""
    graph_maximum = logits.new_full((graph_count,), -torch.inf)
    graph_maximum.scatter_reduce_(
        0, graph_index, logits, reduce="amax", include_self=True
    )
    positions = torch.arange(logits.numel(), device=logits.device)
    candidates = torch.where(
        logits == graph_maximum.index_select(0, graph_index),
        positions,
        torch.full_like(positions, logits.numel()),
    )
    result = torch.full(
        (graph_count,),
        logits.numel(),
        dtype=torch.long,
        device=logits.device,
    )
    result.scatter_reduce_(
        0, graph_index, candidates, reduce="amin", include_self=True
    )
    return result


def _binary_auc(scores: Tensor, labels: Tensor) -> float:
    """Compute exact ROC AUC from average ranks, including tied scores."""
    scores = scores.detach().float().cpu().reshape(-1)
    labels = labels.detach().bool().cpu().reshape(-1)
    positive_count = int(labels.sum().item())
    negative_count = labels.numel() - positive_count
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    order = torch.argsort(scores, stable=True)
    sorted_scores = scores.index_select(0, order)
    sorted_labels = labels.index_select(0, order)
    _, tie_counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    tie_ends = torch.cumsum(tie_counts, dim=0).to(torch.float64)
    tie_starts = tie_ends - tie_counts.to(torch.float64) + 1.0
    average_ranks = torch.repeat_interleave((tie_starts + tie_ends) / 2.0, tie_counts)
    positive_rank_sum = average_ranks[sorted_labels].sum().item()
    return float(
        (positive_rank_sum - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


@dataclass
class FullSelectionDiagnostics:
    threshold_logit: float = 0.0
    activity_nodes: int = 0
    positive_nodes: int = 0
    negative_nodes: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    decision_graphs: int = 0
    graphs_without_positives: int = 0
    multiple_target_graphs: int = 0
    correct_top1_decisions: int = 0
    target_logit_sum: float = 0.0
    target_probability_sum: float = 0.0
    target_node_count: int = 0
    predicted_logit_sum: float = 0.0
    predicted_probability_sum: float = 0.0
    _auc_scores: list[Tensor] = field(default_factory=list)
    _auc_labels: list[Tensor] = field(default_factory=list)
    _target_ranks: list[Tensor] = field(default_factory=list)
    _border_sizes: list[Tensor] = field(default_factory=list)

    def update(self, logits: Tensor, labels: Tensor, border_ptr: Tensor) -> None:
        logits = logits.detach().float().reshape(-1)
        labels = labels.detach().reshape(-1) > 0.5
        border_ptr = border_ptr.detach().to(device=logits.device, dtype=torch.long)
        graph_count = border_ptr.numel() - 1
        border_sizes = border_ptr[1:] - border_ptr[:-1]
        graph_index = torch.repeat_interleave(
            torch.arange(graph_count, device=logits.device), border_sizes
        )
        predictions = logits >= float(self.threshold_logit)

        node_counts = torch.stack(
            (
                labels.sum(),
                (~labels).sum(),
                (predictions & labels).sum(),
                ((~predictions) & (~labels)).sum(),
                (predictions & (~labels)).sum(),
                ((~predictions) & labels).sum(),
            )
        ).cpu().tolist()
        positive, negative, tp, tn, fp, fn = (int(value) for value in node_counts)
        self.activity_nodes += logits.numel()
        self.positive_nodes += positive
        self.negative_nodes += negative
        self.true_positives += tp
        self.true_negatives += tn
        self.false_positives += fp
        self.false_negatives += fn

        positive_per_graph = torch.bincount(
            graph_index, weights=labels.to(torch.float32), minlength=graph_count
        ).to(torch.long)
        top_indices = _first_segment_argmax(logits, graph_index, graph_count)
        valid_graph = positive_per_graph > 0
        top1_correct = labels.index_select(0, top_indices) & valid_graph

        best_target_logit = logits.new_full((graph_count,), -torch.inf)
        if torch.any(labels):
            best_target_logit.scatter_reduce_(
                0,
                graph_index[labels],
                logits[labels],
                reduce="amax",
                include_self=True,
            )
        higher_than_target = logits > best_target_logit.index_select(0, graph_index)
        higher_count = torch.bincount(
            graph_index,
            weights=higher_than_target.to(torch.float32),
            minlength=graph_count,
        ).to(torch.long)
        target_ranks = (higher_count[valid_graph] + 1).cpu()

        graph_counts = torch.stack(
            (
                (~valid_graph).sum(),
                (positive_per_graph > 1).sum(),
                top1_correct.sum(),
            )
        ).cpu().tolist()
        no_positive, multiple_target, correct_top1 = (
            int(value) for value in graph_counts
        )
        self.decision_graphs += graph_count
        self.graphs_without_positives += no_positive
        self.multiple_target_graphs += multiple_target
        self.correct_top1_decisions += correct_top1

        target_logits = logits[labels]
        predicted_logits = logits.index_select(0, top_indices)
        score_sums = torch.stack(
            (
                target_logits.sum(dtype=torch.float64),
                torch.sigmoid(target_logits).sum(dtype=torch.float64),
                predicted_logits.sum(dtype=torch.float64),
                torch.sigmoid(predicted_logits).sum(dtype=torch.float64),
            )
        ).cpu().tolist()
        self.target_logit_sum += float(score_sums[0])
        self.target_probability_sum += float(score_sums[1])
        self.predicted_logit_sum += float(score_sums[2])
        self.predicted_probability_sum += float(score_sums[3])
        self.target_node_count += target_logits.numel()

        self._auc_scores.append(logits.cpu())
        self._auc_labels.append(labels.cpu())
        self._target_ranks.append(target_ranks)
        self._border_sizes.append(border_sizes.cpu())

    def compute(self) -> Dict[str, float | int | str]:
        eps = 1e-12
        tp = self.true_positives
        tn = self.true_negatives
        fp = self.false_positives
        fn = self.false_negatives
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        specificity = tn / (tn + fp + eps)
        valid_graphs = self.decision_graphs - self.graphs_without_positives

        ranks = torch.cat(self._target_ranks) if self._target_ranks else torch.empty(0)
        sizes = torch.cat(self._border_sizes) if self._border_sizes else torch.empty(0)
        auc_scores = torch.cat(self._auc_scores) if self._auc_scores else torch.empty(0)
        auc_labels = torch.cat(self._auc_labels) if self._auc_labels else torch.empty(0)

        return {
            "activity_nodes": self.activity_nodes,
            "positive_nodes": self.positive_nodes,
            "negative_nodes": self.negative_nodes,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "node_accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
            "node_precision": precision,
            "node_recall": recall,
            "node_specificity": specificity,
            "node_f1": 2.0 * precision * recall / (precision + recall + eps),
            "balanced_accuracy": (recall + specificity) / 2.0,
            "node_auc": _binary_auc(auc_scores, auc_labels),
            "decision_graphs": self.decision_graphs,
            "graphs_without_positives": self.graphs_without_positives,
            "multiple_target_graphs": self.multiple_target_graphs,
            "multiple_target_percent": 100.0
            * self.multiple_target_graphs
            / max(self.decision_graphs, 1),
            "correct_top1_decisions": self.correct_top1_decisions,
            "top1_valid_accuracy": self.correct_top1_decisions
            / max(valid_graphs, 1),
            "mean_target_rank": float(ranks.float().mean().item()) if ranks.numel() else float("nan"),
            "median_target_rank": float(ranks.float().median().item()) if ranks.numel() else float("nan"),
            "mean_reciprocal_rank": float((1.0 / ranks.float()).mean().item())
            if ranks.numel()
            else float("nan"),
            "mean_border_size": float(sizes.float().mean().item()) if sizes.numel() else float("nan"),
            "border_size_range": f"{int(sizes.min().item())}--{int(sizes.max().item())}"
            if sizes.numel()
            else "--",
            "mean_target_probability": self.target_probability_sum
            / max(self.target_node_count, 1),
            "mean_predicted_probability": self.predicted_probability_sum
            / max(self.decision_graphs, 1),
            "mean_target_logit": self.target_logit_sum / max(self.target_node_count, 1),
            "mean_predicted_logit": self.predicted_logit_sum
            / max(self.decision_graphs, 1),
        }


def evaluate_full_selection_diagnostics(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    *,
    threshold_logit: float,
) -> Dict[str, float | int | str]:
    model.eval()
    accumulator = FullSelectionDiagnostics(threshold_logit=threshold_logit)
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            logits = model(batch)
            accumulator.update(logits, batch["border"].y, batch["border"].ptr)
    if accumulator.decision_graphs == 0:
        raise RuntimeError("Diagnostic dataset yielded no valid graph.")
    return accumulator.compute()


DIAGNOSTIC_ROWS = (
    ("Activity nodes", "activity_nodes", "count"),
    ("Positive nodes", "positive_nodes", "count"),
    ("Negative nodes", "negative_nodes", "count"),
    ("True positives", "true_positives", "count"),
    ("True negatives", "true_negatives", "count"),
    ("False positives", "false_positives", "count"),
    ("False negatives", "false_negatives", "count"),
    ("Node accuracy", "node_accuracy", "float4"),
    ("Node precision", "node_precision", "float4"),
    ("Node recall", "node_recall", "float4"),
    ("Node specificity", "node_specificity", "float4"),
    ("Node F1", "node_f1", "float4"),
    ("Balanced accuracy", "balanced_accuracy", "float4"),
    ("Node AUC", "node_auc", "float4"),
    ("__MIDRULE__", "", ""),
    ("Decision graphs", "decision_graphs", "count"),
    ("Graphs without positives", "graphs_without_positives", "count"),
    ("Multiple-target graphs", "multiple_target_graphs", "count"),
    ("Multiple-target graphs (\\%)", "multiple_target_percent", "float2"),
    ("Correct top-1 decisions", "correct_top1_decisions", "count"),
    ("Top-1 valid accuracy", "top1_valid_accuracy", "float4"),
    ("Mean target rank", "mean_target_rank", "float4"),
    ("Median target rank", "median_target_rank", "general"),
    ("Mean reciprocal rank", "mean_reciprocal_rank", "float4"),
    ("Mean Border size", "mean_border_size", "float3"),
    ("Border size range", "border_size_range", "text"),
    ("Mean target probability", "mean_target_probability", "float4"),
    ("Mean predicted probability", "mean_predicted_probability", "float4"),
    ("Mean target logit", "mean_target_logit", "float4"),
    ("Mean predicted logit", "mean_predicted_logit", "float4"),
)


def _format_value(value: object, style: str) -> str:
    if style == "count":
        return f"{int(value):,}"
    if style == "float2":
        return f"{float(value):.2f}"
    if style == "float3":
        return f"{float(value):.3f}"
    if style == "float4":
        return f"{float(value):.4f}"
    if style == "general":
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:.4f}"
    return str(value)


PERFORMANCE_ROWS = (
    ("Graphs", "graphs", "count"),
    ("Loss", "loss", "float4"),
    ("BCE loss", "bce_loss", "float4"),
    ("Top-1 loss", "top1_loss", "float4"),
    ("Top-1", "top1_any_positive", "float4"),
    ("Accuracy", "binary_accuracy", "float4"),
    ("Precision", "precision", "float4"),
    ("Recall", "recall", "float4"),
    ("F1", "f1", "float4"),
)


def write_performance_metrics_table(
    metrics: Mapping[str, Mapping[str, object]],
    latex_path: str | Path,
    csv_path: str | Path | None = None,
    *,
    rows: Sequence[tuple[str, str]],
    caption: str,
) -> None:
    """Write the compact five-row final performance table."""
    row_specs = tuple(rows)
    if not row_specs:
        raise ValueError("At least one performance row is required.")
    missing = [key for key, _ in row_specs if key not in metrics]
    if missing:
        raise ValueError(f"Missing performance datasets: {missing}")

    headers = [label for label, _, _ in PERFORMANCE_ROWS]
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{0.90}",
        f"\\caption{{{caption}}}",
        r"\label{tab:train-validation-test}",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        "Dataset & " + " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    csv_rows = []
    for key, label in row_specs:
        values = [
            _format_value(metrics[key][metric_key], style)
            for _, metric_key, style in PERFORMANCE_ROWS
        ]
        latex_lines.append(f"{label} & {' & '.join(values)} " + r"\\")
        csv_rows.append([label, *values])
    latex_lines.extend(
        [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    )

    latex_path = Path(latex_path)
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    latex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("Dataset", *headers))
            writer.writerows(csv_rows)


def write_full_selection_diagnostics(
    diagnostics: Mapping[str, Mapping[str, object]],
    latex_path: str | Path,
    csv_path: str | Path | None = None,
    *,
    caption: str,
    columns: Sequence[tuple[str, str]] | None = None,
) -> None:
    column_specs = tuple(
        columns
        or (
            ("train", "Train"),
            ("validation", "Valid."),
            ("test", "Test"),
        )
    )
    if not column_specs:
        raise ValueError("At least one diagnostic column is required.")
    column_keys = tuple(key for key, _ in column_specs)
    missing = [key for key in column_keys if key not in diagnostics]
    if missing:
        raise ValueError(f"Missing diagnostic splits: {missing}")

    latex_lines = [
        r"\begin{table}[!t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{2.5pt}",
        r"  \renewcommand{\arraystretch}{0.88}",
        rf"  \begin{{tabular}}{{@{{}}l{'r' * len(column_specs)}@{{}}}}",
        r"    \toprule",
        "    Metric & " + " & ".join(label for _, label in column_specs) + r" \\",
        r"    \midrule",
    ]
    csv_rows = []
    for label, key, style in DIAGNOSTIC_ROWS:
        if label == "__MIDRULE__":
            latex_lines.append(r"    \midrule")
            continue
        values = [
            _format_value(diagnostics[column_key][key], style)
            for column_key in column_keys
        ]
        latex_lines.append(f"    {label} & {' & '.join(values)} " + r"\\")
        csv_rows.append([label.replace("\\%", "%"), *values])
    latex_lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            f"  \\caption{{{caption}}}",
            r"  \label{tab:full-selection-diagnostics}",
            r"\end{table}",
        ]
    )

    latex_path = Path(latex_path)
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    latex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("Metric", *(label for _, label in column_specs)))
            writer.writerows(csv_rows)
