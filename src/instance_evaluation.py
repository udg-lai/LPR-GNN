"""Complete RCPSP rollouts, never labelled validation/test graphs."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from .normalization import GraphNormalizer
from .rcpsp_instance import parse_mrcpsp
from .rcpsp_schedule import calculate_makespans_batched


# Read the configured instance memberships without prevalidating their contents.
def load_instance_splits(config):
    root = Path(config['_project_root'])
    manifest_path = root / config['instance_evaluation']['manifest']
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    return {
        split: sorted(
            (root / entry['path'] for entry in manifest['splits'][split]),
            key=lambda p: p.as_posix().casefold(),
        )
        for split in ('train', 'validation', 'test')
    }


# Arithmetic mean over ALL instances; any failed rollout aborts evaluation.
# Parameters: model: priority network; paths: instance paths; device: compute device; config: project settings
#   output_path: optional output destination
# Returns: instance count, mean makespan and elapsed seconds.
def evaluate_instances(model, paths, device, config, output_path=None):
    """Arithmetic mean over ALL instances; any failed rollout aborts evaluation.

    Same parser, normalizer, serial horizon and batched scheduler as
    makespan_predictor. No optimum/label is used. Timing is amortized per batch.
    """
    paths = sorted(map(Path, paths), key=lambda p: p.as_posix().casefold())
    if not paths:
        raise ValueError('Cannot evaluate an empty instance set.')
    cfg = config['inference']
    batch_size = int(cfg['rollout_batch_size'])
    workers = int(cfg['graph_workers'])
    if batch_size < 1 or workers < 1:
        raise ValueError('Rollout batch size and workers must be positive.')
    normalizer = GraphNormalizer(
        temporal_reference=float(config['normalization']['temporal_reference'])
    )
    was_training = model.training
    model.eval()
    records = []
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for offset in range(0, len(paths), batch_size):
                batch_paths = paths[offset:offset + batch_size]
                instances = [parse_mrcpsp(path) for path in batch_paths]
                horizons = [sum(int(d[0]) for d in inst.durations) + 1 for inst in instances]
                batch_started = time.perf_counter()
                results = calculate_makespans_batched(
                    model, instances, horizons, device, graph_transform=normalizer,
                    graph_workers=workers, include_scores=False,
                )
                if len(results) != len(batch_paths):
                    raise RuntimeError('Scheduler returned an incomplete batch.')
                seconds = (time.perf_counter() - batch_started) / len(batch_paths)
                for path, result in zip(batch_paths, results):
                    if not math.isfinite(result.makespan) or result.makespan < 0:
                        raise ValueError(f'Invalid makespan: {path}')
                    records.append(dict(instance=path.name, makespan=result.makespan,
                                        amortized_seconds=seconds, solution=result.solution))
                print(f'RCPSP rollouts: {len(records)}/{len(paths)}', flush=True)
    finally:
        model.train(was_training)
    summary = dict(instances=len(records),
                   mean_makespan=sum(r['makespan'] for r in records) / len(records),
                   seconds=time.perf_counter() - started)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(dict(summary=summary, results=records), indent=2), encoding='utf-8')
    return summary


# Evaluate held-out instances and write their makespan table.
# Parameters: model: priority network; config: project settings; device: compute device
#   output_dir: output directory; splits: instance paths by partition
# Returns: validation and test rollout summaries.
def final_instance_evaluation(model, config, device, output_dir, splits):
    summaries = {
        split: evaluate_instances(model, splits[split], device, config,
                                  Path(output_dir) / f'{split}_makespans_best.json')
        for split in ('validation', 'test')
    }
    lines = [r'\begin{table}[!t]', r'\centering', r'\begin{tabular}{lrr}',
             r'\hline', r'Split & Instances & Mean makespan \\', r'\hline']
    for split, values in summaries.items():
        lines.append(f"{split.title()} & {values['instances']} & {values['mean_makespan']:.4f} " + r'\\')
    lines += [r'\hline', r'\end{tabular}',
              r'\caption{Complete RCPSP rollouts of the checkpoint selected by minimum validation mean makespan. Test is evaluated only after training.}',
              r'\label{tab:instance-makespan-evaluation}', r'\end{table}']
    (Path(output_dir) / 'validation_test_makespans.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summaries
