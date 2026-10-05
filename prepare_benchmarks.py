"""Download and prepare the seven public benchmark families (no split changes)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import urllib.request

from prepare_instances import ROOT, canonical_rcp, digest, read_archives
from src.rcpsp_instance import parse_mrcpsp


def pack_to_rcp(blob):
    """Convert MiniZinc PACK data, adding only zero-duration source/sink nodes."""
    text = re.sub(r'%[^\n]*', '', blob.decode('utf-8'))

    def field(name):
        match = re.search(r'\b' + name + r'\s*=\s*(.*?);', text, flags=re.S)
        if not match:
            raise ValueError(f'Missing PACK field: {name}')
        return match.group(1)

    def integers(value):
        return [int(x) for x in re.findall(r'-?\d+', value)]

    resources, activities = int(field('n_res')), int(field('n_tasks'))
    capacities, durations = integers(field('rc')), integers(field('d'))
    demands = [integers(row) for row in field('rr').strip(' []').strip('|').split('|')]
    successors = [integers(row) for row in re.findall(r'\{([^}]*)\}', field('suc'))]
    if (len(capacities) != resources or len(durations) != activities or
            len(demands) != resources or any(len(row) != activities for row in demands) or
            len(successors) != activities):
        raise ValueError('Inconsistent PACK dimensions')
    if any(s < 1 or s > activities for row in successors for s in row):
        raise ValueError('Invalid PACK successor ID')
    # MiniZinc contains real activities only; the SSGS expects dummy endpoints.
    incoming = {s for row in successors for s in row}
    sources = [a + 1 for a in range(1, activities + 1) if a not in incoming]
    rows = [[activities + 2, resources], capacities, [0] * (resources + 1) + [len(sources)] + sources]
    for a in range(activities):
        target_ids = [s + 1 for s in successors[a]] or [activities + 2]
        rows.append([durations[a]] + [demands[r][a] for r in range(resources)] + [len(target_ids)] + target_ids)
    rows.append([0] * (resources + 2))
    return canonical_rcp(('\n'.join(' '.join(map(str, row)) for row in rows)).encode())


def prepare(cache):
    recipe = json.loads((ROOT / 'benchmark_sources.json').read_text(encoding='utf-8'))
    archives = read_archives(recipe, cache)
    records = {}

    def place(family, name, blob):
        if PurePosixPath(name).name != name:
            raise ValueError('Unsafe benchmark filename')
        path = ROOT / 'rcpsp' / family / name
        if path.exists() and path.read_bytes() != blob:
            raise ValueError(f'Refusing to overwrite a different instance: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(blob)
        parse_mrcpsp(path)
        records.setdefault(family, {})[name] = digest(blob)

    try:
        for family in ('j30', 'j60', 'j90', 'j120'):
            archive = archives[family]
            for member in archive.namelist():
                if member.lower().endswith('.sm') and '__MACOSX' not in member:
                    place(family, PurePosixPath(member).name, archive.read(member))
            print(f'{family}: {len(records[family])}', flush=True)
        for family in ('RG30', 'RG300'):
            archive = archives['rcplib_all']
            for member in archive.getmembers():
                if member.isfile() and member.name.startswith('original/' + family + '_'):
                    place(family, PurePosixPath(member.name).name, canonical_rcp(archive.extractfile(member).read()))
            print(f'{family}: {len(records[family])}', flush=True)
        for source in recipe['pack']['files']:
            path = cache / 'PACK' / source['name']
            if not path.exists():
                request = urllib.request.Request(source['url'], headers={'User-Agent': 'LPR-GNN-benchmark-preparation'})
                with urllib.request.urlopen(request, timeout=120) as response:
                    blob = response.read()
                if digest(blob) != source['sha256']:
                    raise ValueError(f'PACK source checksum changed: {source["name"]}')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(blob)
            blob = path.read_bytes()
            if digest(blob) != source['sha256']:
                raise ValueError('Cached PACK checksum mismatch')
            converted = pack_to_rcp(blob)
            if digest(converted) != source['prepared_sha256']:
                raise ValueError('Converted PACK checksum mismatch')
            place('PACK', Path(source['name']).stem + '.rcp', converted)
        print(f'PACK: {len(records["PACK"])}', flush=True)
        for family, expected in recipe['expected_counts'].items():
            if len(records[family]) != expected:
                raise ValueError(f'Incomplete benchmark family: {family}')
        report = ROOT / 'outputs/benchmark_inputs.json'
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
    finally:
        for archive in archives.values():
            archive.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'outputs/input_archives')
    args = parser.parse_args()
    prepare(args.cache_dir.resolve())
