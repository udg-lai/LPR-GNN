"""Prepare the published partitions from checksum-verified public archives."""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
import urllib.request
import zipfile

from src.rcpsp_instance import parse_mrcpsp

ROOT = Path(__file__).resolve().parent


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def canonical_rcp(blob):
    """Canonical Patterson rows: integer tokens, single spaces, LF, final LF."""
    tokens = [int(token) for token in blob.decode('utf-8-sig').split()]
    if len(tokens) < 2:
        raise ValueError('Invalid Patterson header')
    activities, resources = tokens[:2]
    if activities < 2 or resources < 0:
        raise ValueError('Invalid Patterson dimensions')
    rows = [[activities, resources], tokens[2:2 + resources]]
    offset = 2 + resources
    for _ in range(activities):
        if len(tokens) < offset + resources + 2:
            raise ValueError('Truncated Patterson activity')
        count = tokens[offset + resources + 1]
        stop = offset + resources + 2 + count
        rows.append(tokens[offset:stop])
        offset = stop
    if offset != len(tokens) or len(rows[1]) != resources or any(x < 0 for row in rows for x in row):
        raise ValueError('Invalid resource capacities or negative integer')
    for row in rows[2:]:
        if len(row) < resources + 2 or len(row) != resources + 2 + row[resources + 1]:
            raise ValueError('Invalid successor count')
        if any(x < 1 or x > activities for x in row[resources + 2:]):
            raise ValueError('Invalid successor identifier')
    return ('\n'.join(' '.join(map(str, row)) for row in rows) + '\n').encode()


def sm_to_rcp(blob):
    """Preserve PSPLIB activity order, durations, demands and precedences."""
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'instance.sm'
        path.write_bytes(blob)
        instance = parse_mrcpsp(path)
    rows = [[instance.n_activities, instance.n_resources], instance.capacities]
    for a in range(instance.n_activities):
        successors = [s + 1 for s in instance.successors[a]]
        rows.append([instance.durations[a][0]] +
                    [instance.demands[a][r][0] for r in range(instance.n_resources)] +
                    [len(successors)] + successors)
    return canonical_rcp(('\n'.join(' '.join(map(str, row)) for row in rows)).encode())


def transform(blob, method):
    if method == 'native_sm':
        return blob
    if method == 'psplib_sm_to_rcp_v1':
        return sm_to_rcp(blob)
    if method == 'canonical_rcp_v1':
        return canonical_rcp(blob)
    raise ValueError(f'Unknown conversion: {method}')


def destination(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or path.suffix.lower() not in ('.sm', '.rcp'):
        raise ValueError(f'Unsafe instance destination: {relative}')
    return path


def read_archives(recipe, cache):
    archives = {}
    for name, source in recipe['archives'].items():
        path = cache / source['filename']
        if not path.exists():
            cache.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(source['url'], headers={'User-Agent': 'LPR-GNN-input-preparation'})
            print(f'Downloading {name}', flush=True)
            with urllib.request.urlopen(request, timeout=120) as response:
                blob = response.read()
            if digest(blob) != source['sha256']:
                raise ValueError(f'Upstream archive checksum mismatch: {name}')
            path.write_bytes(blob)
        blob = path.read_bytes()
        if digest(blob) != source['sha256']:
            raise ValueError(f'Cached archive checksum mismatch: {name}')
        archives[name] = zipfile.ZipFile(io.BytesIO(blob))
    # Read exact tar members without extracting arbitrary upstream paths.
    archives['rcplib_all'] = tarfile.open(fileobj=io.BytesIO(archives['rcplib'].read('ALL/ALL.tar')))
    return archives


def source_bytes(archives, entry):
    archive = archives[entry['archive']]
    if isinstance(archive, tarfile.TarFile):
        member = archive.getmember(entry['member'])
        if not member.isfile():
            raise ValueError('Source member is not a regular file')
        return archive.extractfile(member).read()
    return archive.read(entry['member'])


def validate_training_references(prepared):
    """Check every bundled reference schedule against the public instance."""
    paths_by_name = {name.casefold(): name for name in prepared}
    with tempfile.TemporaryDirectory() as temporary:
        for row in csv.DictReader((ROOT / 'source_data/training_sources.csv').open(encoding='utf-8'), delimiter=';'):
            relative = paths_by_name[('source_data/' + row['file']).casefold()]
            blob = prepared[relative]
            path = Path(temporary) / ('instance.' + row['file_type'])
            path.write_bytes(blob)
            instance = parse_mrcpsp(path)
            sequence = ast.literal_eval(row['sequence'])
            starts = dict(sequence)
            if len(sequence) != instance.n_activities or set(starts) != set(range(instance.n_activities)):
                raise ValueError(f'Invalid reference activity identifiers: {relative}')
            if any(t < 0 for t in starts.values()):
                raise ValueError(f'Negative reference start time: {relative}')
            for a, successors in enumerate(instance.successors):
                if any(starts[b] < starts[a] + instance.durations[a][0] for b in successors):
                    raise ValueError(f'Reference precedence violation: {relative}')
            for r in range(instance.n_resources):
                events = collections.defaultdict(int)
                for a in range(instance.n_activities):
                    duration, demand = instance.durations[a][0], instance.demands[a][r][0]
                    if duration:
                        events[starts[a]] += demand
                        events[starts[a] + duration] -= demand
                use = 0
                for _, delta in sorted(events.items()):
                    use += delta
                    if use > instance.capacities[r]:
                        raise ValueError(f'Reference resource violation: {relative}')


def prepare(cache):
    recipe = json.loads((ROOT / 'instance_sources.json').read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / 'instance_splits.json').read_text(encoding='utf-8'))
    entries = [entry for split in ('train', 'validation', 'test') for entry in manifest['splits'][split]]
    if set(recipe['instances']) != {entry['path'] for entry in entries}:
        raise ValueError('Source recipe and split manifest differ')
    archives = read_archives(recipe, cache)
    prepared = {}
    try:
        for entry in entries:
            source = recipe['instances'][entry['path']]
            blob = source_bytes(archives, source)
            if digest(blob) != source['source_sha256']:
                raise ValueError(f'Source member checksum mismatch: {entry["path"]}')
            converted = transform(blob, source['conversion'])
            if digest(converted) != entry['sha256']:
                raise ValueError(f'Prepared instance checksum mismatch: {entry["path"]}')
            path = destination(entry['path'])
            if path.exists() and path.read_bytes() != converted:
                raise ValueError(f'Refusing to overwrite different existing input: {path}')
            prepared[entry['path']] = converted
        validate_training_references(prepared)
        # Only place inputs after all checks have passed. Existing equal files are kept.
        for relative, blob in prepared.items():
            path = destination(relative)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(blob)
        for split in ('train', 'validation', 'test'):
            print(f'{split}: {len(manifest["splits"][split])} verified instances')
        print('All 1,106 training reference schedules pass precedence and resource checks.')
    finally:
        for archive in archives.values():
            archive.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'outputs/input_archives',
                        help='Archive cache; filenames and SHA-256 are checked before use.')
    args = parser.parse_args()
    prepare(args.cache_dir.resolve())
