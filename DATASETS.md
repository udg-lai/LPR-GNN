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
No pretrained model is bundled: train first to produce the configured `best.pt`.
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

1. Read each entry in `instance_splits.json` and copy its instance to that exact
   relative path. Validation and test files go directly in their directories,
   without additional family subdirectories. Preserve the frozen memberships.
2. Training graph generation also reads `source_data/training_sources.csv`.
   Preserve its `file`, `file_type`, and `sequence` columns and their associations
   with the training instances. These are reference activity orders, not raw instances.
3. Preserve the format and extension of each manifest entry. Training uses 770
   Patterson-format `.RCP` instances and 336 native `.sm` instances; validation
   uses 87 `.RCP` and 35 `.sm` instances. Public PSPLIB downloads normally use `.sm`,
   so downloading and renaming them is insufficient for the `.RCP` entries.
   For exact reproduction, supply the byte-identical copies used to create the
   manifest (for example, from your original project). A semantic conversion alone
   may still differ in whitespace and fail SHA-256 checks. Native `.sm` entries
   likewise must match their stored hashes, including headers and line endings.
4. Do not remove or replace hashes to hide a mismatch. An independently prepared
   corpus requires deliberate conversion, matching reference activity numbering,
   an updated CSV and manifest, and validation of the split. It is not the frozen
   corpus merely because the projects have similar names.
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
