"""Download and prepare the six public benchmark families (no split changes)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

from prepare_instances import ROOT, canonical_rcp, read_archives


def prepare(cache):
    recipe = json.loads((ROOT / 'benchmark_sources.json').read_text(encoding='utf-8'))
    archives = read_archives(recipe, cache)
    records = {}

    def place(family, name, blob):
        if PurePosixPath(name).name != name:
            raise ValueError('Unsafe benchmark filename')
        path = ROOT / 'rcpsp' / family / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        records.setdefault(family, []).append(name)

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
