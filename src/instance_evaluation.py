"""Complete RCPSP rollouts, never labelled validation/test graphs."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import torch

from .normalization import GraphNormalizer
from .rcpsp_instance import parse_mrcpsp
from .rcpsp_schedule import calculate_makespans_batched


# Compute a file SHA-256 digest.
# Parameters: path: input/output path
# Returns: SHA-256 hex string.
def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Verify instance integrity, split separation and graph membership.
# Parameters: config: project settings
# Returns: split paths and manifest SHA-256.
def load_instance_splits(config):
    root = Path(config['_project_root'])
    manifest_path = root / config['instance_evaluation']['manifest']
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    splits = {}
    identities, hashes = {}, {}
    for split in ('train', 'validation', 'test'):
        entries = manifest['splits'][split]
        if not entries:
            raise ValueError(f'Empty instance split: {split}')
        paths = []
        for entry in entries:
            path = (root / entry['path']).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError(f'Missing or external instance: {path}')
            digest = file_hash(path)
            if digest != entry['sha256']:
                raise ValueError(f'Instance changed since split creation: {path}')
            key = path.stem.casefold()
            if key in identities or digest in hashes:
                raise ValueError(f'Duplicate/overlapping instance: {path}')
            identities[key] = split
            hashes[digest] = split
            paths.append(path)
        splits[split] = sorted(paths, key=lambda p: p.as_posix().casefold())
    # Protect against accidentally adding validation/test instances to train.
    graph_keys = set()
    for path in Path(config['data']['train_dir']).iterdir():
        name = path.name.lower()
        if name.endswith('.json'):
            graph_keys.add(name[:-5])
        elif name.endswith('.tar.gz'):
            graph_keys.add(name[:-7])
    expected = {p.stem.casefold() for p in splits['train']}
    if graph_keys != expected:
        raise ValueError('Training graph membership differs from the frozen instance split.')
    return splits, file_hash(manifest_path)


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
