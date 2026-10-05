# External datasets and directory layout

Raw RCPSP instances are not included in this repository. Download them separately
and respect the providers' terms. The software license does not license these data.
All paths below are relative to the cloned repository root.

## Download sources

| Family | External source | Benchmark directory |
| --- | --- | --- |
| J30 | [PSPLIB](https://www.om-db.wi.tum.de/psplib/), single-mode J30 | `rcpsp/j30/` |
| J60 | [PSPLIB](https://www.om-db.wi.tum.de/psplib/), single-mode J60 | `rcpsp/j60/` |
| J90 | [PSPLIB](https://www.om-db.wi.tum.de/psplib/), single-mode J90 | `rcpsp/j90/` |
| J120 | [PSPLIB](https://www.om-db.wi.tum.de/psplib/), single-mode J120 | `rcpsp/j120/` |
| PACK | Carlier and Neron's PACK suite; historical [LAAS collection](http://www.laas.fr/~okone/) (see availability note below) | `rcpsp/PACK/` |
| RG30 | [RCPLIB releases](https://github.com/MarioVanhoucke/RCPLIB-Resource-Constrained-Project-Library/releases/latest), original RG30 | `rcpsp/RG30/` |
| RG300 | [RCPLIB releases](https://github.com/MarioVanhoucke/RCPLIB-Resource-Constrained-Project-Library/releases/latest), original RG300 | `rcpsp/RG300/` |

For PSPLIB, use the Download data section and the single-mode `.sm` archives,
not multi-mode `.mm` instances. See its [download instructions](https://www.om-db.wi.tum.de/psplib/download.php).
For RG30/RG300, select the original datasets, not resource-modified versions;
the [OR&S dataset page](https://www.projectmanagement.ugent.be/research/project_scheduling/rcpsp)
provides background and points to RCPLIB. Extract archives first, then copy the
instance files into the family directories above.

PACK availability: the historical LAAS address is cited in the benchmark
literature, but a working current direct download was not verified when preparing
this distribution. Obtain the original PACK suite from its authors or an existing
author-provided copy; do not substitute a different dataset. This entry is a
provenance pointer, not a confirmed downloadable archive.

## Benchmark inference

`makespan_predictor.ipynb` defaults to `INSTANCE_DIR = PROJECT_ROOT / "rcpsp"`
and `RECURSIVE = True`. It discovers `.sm` and `.rcp` files beneath this directory.
Keep each family in the directory listed above; do not leave extra benchmark copies
under `rcpsp/`, since the recursive scan can evaluate them as well.

The parser chooses its format by suffix. Keep native PSPLIB files as `.sm`.
Renaming a `.sm` file to `.RCP` does **not** convert it to Patterson format.
Use the supplied `paper_results/best.pt` for inference as described in
`paper_results/README.md`, or train to produce the configured `best.pt`.
Reference bounds are optional; configure `BOUNDS_PATH` in the predictor to obtain gaps.

## Training, validation and test

Benchmark storage alone does not prepare the training split. The frozen
`instance_splits.json` specifies exact paths and SHA-256 hashes for:

| Purpose | Required location | Number of instances |
| --- | --- | ---: |
| Training J30 | `source_data/j30/<manifest filename>` | 436 |
| Training J60 | `source_data/j60/<manifest filename>` | 334 |
| Training J90 | `source_data/j90/<manifest filename>` | 336 |
| Validation J30/J60/J90 | `data/validation/<manifest filename>` | 122 |
| Test RG30 | `data/test/<manifest filename>.rcp` | 180 |

1. Run `python prepare_instances.py` using the project environment. It downloads
   the pinned public archives and places instances at the exact manifest paths.
   An existing file with different contents is rejected, not overwritten.
2. `instance_sources.json` records each upstream archive, member, source hash and
   conversion. Archive hashes and prepared-file hashes are checked independently.
   Archives are cached under `outputs/input_archives/`. To reuse archives downloaded
   from the same public sources, use `--cache-dir <archive-directory>`; the same
   checksum checks apply. Upstream changes fail explicitly.
3. Native `.sm` members retain their bytes. Required `.RCP` files are converted
   from PSPLIB with the supplied parser, preserving zero-based activity identity,
   durations, demands, capacities and precedence relations. RG30 test inputs come
   from `original/RG30_<id>.rcp` in `ALL/ALL.tar` in RCPLIB v2.4. Patterson output
   uses integer tokens, single spaces, LF and a final newline. Wrapped successor
   lists are interpreted by their declared counts. No instance data are redistributed.
4. The published memberships and `source_data/training_sources.csv` are unchanged.
   The manifest now pins these reproducibly prepared public files, not unprovided
   project-specific byte representations. All 1,106 training reference schedules
   are checked for activity coverage, precedence and resource feasibility before
   placement. This is an input-preparation repair, not evidence that a retraining
   run reproduces the archived metrics exactly. Published inference weights and
   result tables are unchanged. Old project checkpoint resumes tied to a different
   split-manifest hash are intentionally incompatible; do not bypass that check.
5. Run `create_graphs_training.ipynb` to populate `data/train/`, then run
   `training.ipynb`. Validation and test are rollouts on raw instances; do not
   generate validation/test graph datasets. Finally run `makespan_predictor.ipynb`.

To inspect the required destinations in PowerShell, from the repository root:

```powershell
$manifest = Get-Content -LiteralPath .\instance_splits.json -Raw | ConvertFrom-Json
foreach ($splitName in @('train', 'validation', 'test')) {
    foreach ($entry in $manifest.splits.$splitName) {
        $present = Test-Path -LiteralPath $entry.path -PathType Leaf
        $matches = $false
        if ($present) {
            $matches = (Get-FileHash -LiteralPath $entry.path -Algorithm SHA256).Hash -eq $entry.sha256
        }
        [pscustomobject]@{ Split = $splitName; Path = $entry.path; Exists = $present; HashMatches = $matches }
    }
}
```

Missing instances and hash mismatches intentionally stop training. A fresh clone
does not run end to end until these external inputs have been prepared. Downloaded
`.sm`/`.rcp` files are ignored by Git to avoid accidental redistribution.
