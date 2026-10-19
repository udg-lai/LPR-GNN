"""Prepare the published partitions from public archives."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import tarfile
import tempfile
import urllib.request
import zipfile

from src.rcpsp_instance import parse_mrcpsp

ROOT = Path(__file__).resolve().parent


def canonical_rcp(blob):
    """Canonical Patterson rows: integer tokens, single spaces, LF, final LF."""
    tokens = [int(token) for token in blob.decode('utf-8-sig').split()]
    if len(tokens) < 2:
        raise ValueError('Invalid Patterson header')
    activities, resources = tokens[:2]
    rows = [[activities, resources], tokens[2:2 + resources]]
    offset = 2 + resources
    for _ in range(activities):
        if len(tokens) < offset + resources + 2:
            raise ValueError('Truncated Patterson activity')
        count = tokens[offset + resources + 1]
        stop = offset + resources + 2 + count
        rows.append(tokens[offset:stop])
        offset = stop
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
            path.write_bytes(blob)
        blob = path.read_bytes()
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


def prepare(cache):
    recipe = json.loads((ROOT / 'instance_sources.json').read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / 'instance_splits.json').read_text(encoding='utf-8'))
    entries = [entry for split in ('train', 'validation', 'test') for entry in manifest['splits'][split]]
    archives = read_archives(recipe, cache)
    prepared = {}
    try:
        for entry in entries:
            source = recipe['instances'][entry['path']]
            blob = source_bytes(archives, source)
            converted = transform(blob, source['conversion'])
            prepared[entry['path']] = converted
        for relative, blob in prepared.items():
            path = destination(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        for split in ('train', 'validation', 'test'):
            print(f'{split}: {len(manifest["splits"][split])} prepared instances')
    finally:
        for archive in archives.values():
            archive.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'outputs/input_archives',
                        help='Directory for downloaded archives.')
    args = parser.parse_args()
    prepare(args.cache_dir.resolve())
