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
| PACK | Carlier and Neron's PACK suite; public [MiniZinc benchmark mirror](https://github.com/MiniZinc/minizinc-benchmarks/tree/master/rcpsp/data_pack) | `rcpsp/PACK/` |
| RG30 | [RCPLIB releases](https://github.com/MarioVanhoucke/RCPLIB-Resource-Constrained-Project-Library/releases/latest), original RG30 | `rcpsp/RG30/` |
| RG300 | [RCPLIB releases](https://github.com/MarioVanhoucke/RCPLIB-Resource-Constrained-Project-Library/releases/latest), original RG300 | `rcpsp/RG300/` |

For PSPLIB, use the Download data section and the single-mode `.sm` archives,
not multi-mode `.mm` instances. See its [download instructions](https://www.om-db.wi.tum.de/psplib/download.php).
For RG30/RG300, select the original datasets, not resource-modified versions;
the [OR&S dataset page](https://www.projectmanagement.ugent.be/research/project_scheduling/rcpsp)
provides background and points to RCPLIB. Extract archives first, then copy the
instance files into the family directories above.

PACK is obtained from the MiniZinc benchmark mirror, whose README attributes it
to Carlier and Neron (2003), DOI `10.1016/S0377-2217(02)00763-4`. The source commit
is recorded in `benchmark_sources.json`. The MiniZinc
data contain real activities only: conversion adds a zero-duration, zero-demand
dummy source and sink, preserving durations, demands and precedence constraints.
It does not use local copies from an earlier project.

## Benchmark inference

Run `python prepare_benchmarks.py` with the installed project environment to
prepare all seven families: J30/J60/J90 (480 each), J120 (600), RG30 (1,800),
RG300 (480) and PACK (55). It uses pinned public archives and PACK source files,
writes parser-compatible inputs under `rcpsp/`.
Its archive cache defaults to `outputs/input_archives/`; `--cache-dir` can point
to an existing cache. Preparation writes the selected inputs to their destination paths, replacing existing files.
RG instances without renewable resources remain valid and are preserved.
This command does not change the training, validation or test partitions.

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
`instance_splits.json` specifies paths for:

| Purpose | Required location | Number of instances |
| --- | --- | ---: |
| Training J30 | `source_data/j30/<manifest filename>` | 436 |
| Training J60 | `source_data/j60/<manifest filename>` | 334 |
| Training J90 | `source_data/j90/<manifest filename>` | 336 |
| Validation J30/J60/J90 | `data/validation/<manifest filename>` | 122 |
| Test RG30 | `data/test/<manifest filename>.rcp` | 180 |

1. Run `python prepare_instances.py` using the project environment. It downloads
   public archives and places instances at the manifest paths, replacing existing files.
2. `instance_sources.json` records each upstream archive, member and conversion.
   Archives are cached under `outputs/input_archives/`; use `--cache-dir` to reuse
   an existing archive directory.
3. Native `.sm` members retain their bytes. Required `.RCP` files are converted
   from PSPLIB. Patterson output uses integer tokens, single spaces, LF and a final
   newline. Wrapped successor lists are read using their declared counts.
4. `instance_splits.json` supplies training, validation and test memberships.
   Inputs and reference schedules are read when used, without a separate validation pass.
5. Run `create_graphs_training.ipynb` to populate `data/train/`, then run
   `training.ipynb`. Validation and test use rollouts on raw instances. Finally run
   `makespan_predictor.ipynb`.

Downloaded `.sm`/`.rcp` files are ignored by Git.
