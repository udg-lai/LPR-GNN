# LPR-GNN

Source distribution of the GRU LPR-GNN base model: 16 message-passing
iterations, explicit residual resource-time profiles and 41,514 trainable parameters.
The GNN selects an eligible activity; the serial schedule generation scheme (SSGS)
enforces precedence and resource feasibility. Forced single-activity decisions bypass
the GNN and are excluded from the generated training graphs.

## Files excluded from the distribution

The distribution contains source code, configuration, minimal notebooks, split
metadata and training reference schedules. Raw benchmark instances are not bundled.
The following files and directories are excluded by `.gitignore`:

- Downloaded RCPSP instances: `.rcp` and `.sm` files, regardless of letter case.

- Generated results and other runtime outputs: `outputs/`.
- Generated training graphs: `data/train/*`, except `data/train/.gitkeep`.
- Model checkpoints and serialized objects: `*.pt`, `*.pth`, `*.pkl`.
- Excel workbooks, logs and compressed archives: `*.xlsx`, `*.log`, `*.zip`, `*.tar.gz`.
- Virtual environments named `.venv/` or `venv/`.
- Python caches, compiled files and notebook checkpoints: `__pycache__/`,
  `*.py[cod]`, `.pytest_cache/`, `.ipynb_checkpoints/`.
- Environment files that may contain credentials: `.env`, `.env.*`.
- Local editor settings: `.vscode/`, `.idea/`.

The `paper_results/` archive contains LPR-GNN base-model results, handcrafted
priority-rule results and Wheatley rollouts in `other_priority_rules_results.csv`,
plus validation-selected LPR-GNN inference weights. See
[paper_results/README.md](paper_results/README.md) for methods and timing definitions.
The checkpoint exclusion above
has a narrow exception for `paper_results/best.pt`. Saved notebook execution
outputs are not bundled.

## Installation

Clone the repository using an account with access, then enter its directory:

```powershell
git clone https://github.com/josepsuy/LPR-GNN.git
cd LPR-GNN
```

The reference environment uses **Python 3.11.15**, recorded in `.python-version`.
Install that interpreter before creating the environment: `requirements.txt`
installs packages, not Python itself. On Windows, from the repository root:

Python 3.11.15 has no python.org Windows installer. One option is the
[Astral uv managed interpreter](https://docs.astral.sh/uv/guides/install-python/):
`uv python install 3.11.15`, then `uv venv --python 3.11.15 PTR-GNN`.
Use the environment's Python executable for the installation and kernel commands
below; the `py -3.11` commands apply only if that launcher selects exactly 3.11.15.

```powershell
py -3.11 -c "import sys; assert sys.version_info[:3] == (3, 11, 15), sys.version"
py -3.11 -m venv PTR-GNN
.\PTR-GNN\Scripts\python.exe -m pip install --upgrade pip
.\PTR-GNN\Scripts\python.exe -m pip install -r requirements.txt
.\PTR-GNN\Scripts\python.exe -m ipykernel install --user --name PTR-GNN --display-name "Python (PTR-GNN)"
.\PTR-GNN\Scripts\python.exe -m pip check
.\PTR-GNN\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Direct dependencies are pinned to the versions installed in the base environment;
this is not a lockfile of all transitive dependencies. PyTorch is pinned to
`2.5.1+cu121` (CUDA 12.1). CUDA execution also requires a compatible NVIDIA GPU
and driver. The CUDA version reported by `nvidia-smi` may be newer than the
runtime bundled with PyTorch; installing a separate CUDA toolkit is not required
for the supplied wheel. Select the `Python (PTR-GNN)` kernel in VS Code/Jupyter.
Set `device: cpu` in `config.yaml` for CPU execution; no GPU is required to run
the scheduler. All stored project paths are relative to this repository.

On another supported platform, create a virtual environment with Python 3.11.15
and install the appropriate PyTorch build if the pinned CUDA wheel is unavailable.
Such a build is a platform adaptation, not the exact reference environment.

## Minimal workflow

First download and prepare the raw instances following [DATASETS.md](DATASETS.md).
That guide lists the external sources, the required directories, and the frozen
split's format and filename conventions. Cloning alone does not supply
the instances required for graph generation, validation or test.

From the repository root, run `python prepare_instances.py` with the installed
project environment. It downloads public PSPLIB and RCPLIB v2.4
archives, converts the required Patterson inputs deterministically,
and places the three fixed partitions. No local
copies from an earlier project are required. The source recipe is
`instance_sources.json`; partition memberships are in `instance_splits.json`.

For benchmark inference, `python prepare_benchmarks.py` separately prepares
J30, J60, J90, J120, RG30, RG300 and PACK under `rcpsp/`, using
`benchmark_sources.json`. This does not change the training partitions.

Start Jupyter from the repository root and run these notebooks in order:

1. `create_graphs_training.ipynb`: generate training graphs from raw instances and
   their reference schedules, restricted to the frozen training split.
2. `training.ipynb`: train, select the checkpoint with the lowest mean validation
   makespan, and evaluate it on test instances after training.
3. `makespan_predictor.ipynb`: evaluate a trained checkpoint on a directory of
   RCPSP instances (recursive by default), and generate an Excel workbook.

The third notebook can use the `best.pt` produced by the second or the supplied
`paper_results/best.pt`, following [paper_results/README.md](paper_results/README.md).
Reference bounds are optional and are not bundled; by default
the predictor reports makespans and times without gaps. To compute gaps, supply a
compatible reference file and set its relative path in `BOUNDS_PATH`.

Configure training in `config.yaml`. Defaults preserve the base model, 12 data workers,
24 epochs, all available training graphs per epoch, and no checkpoint resume.
Adjust worker count or batch size for smaller machines. Training order is shuffled;
validation/test use deterministic inference over raw RCPSP instances, not augmented
validation/test graphs. Training augmentation includes resource permutation and duration scaling.
Each decision graph produces three variants: original, resource-permuted and
duration-scaled (the two augmentations are not combined). Training uses all
401,778 archived graphs, matching the reported base checkpoint. Existing graph
files are preserved unless regeneration is explicitly requested. The generation seed follows the YAML seed.

## Data and split

- `rcpsp/`: place externally obtained benchmark instances here, grouped by family.
- `source_data/j30/`, `source_data/j60/`, `source_data/j90/`: place only the frozen
  training instances here. The bundled `source_data/training_sources.csv` contains
  reference schedules needed as supervision; it does not contain raw instances.
- `data/validation/`, `data/test/`: place the original held-out instances here,
  directly in each directory, as specified in `instance_splits.json`.
- Keep each manifest entry's format: training/validation include both `.RCP`
  and `.sm` files; test uses `.rcp`. Do not change extensions to match a family name.
- `instance_splits.json`: fixed memberships of reproducibly
  prepared public inputs (1,106 train, 122 validation, 180 test instances).
  Do not resplit individual graphs. `instance_sources.json` pins archives, source
  members and conversions; `prepare_instances.py` prepares these inputs.
- `data/train/`: initially empty; populated by graph generation.
- `outputs/`: created only when running the code; excluded from Git.

See [DATASETS.md](DATASETS.md) for download sources and placement instructions.
The project distribution excludes raw instances, generated graphs, saved notebook
outputs, credentials, machine-specific configuration and runtime outputs outside
the explicitly included `paper_results/` archive.

## Graph schema

Generated graphs use English resource feature names such as `resources_needed`,
`resources_needed_other`, and `resources_available`. The parser and normalizer
also accept the previous spellings through `src/graph_schema.py`; feature values,
tensor order, relations, and model parameter names are unchanged. Conflicting
values supplied under both spellings raise an error. The training CSV uses
`file`, `file_type`, and `sequence` columns.

## Project distribution and data rights

Benchmark instances must be obtained separately from their external providers,
following the sources listed in [DATASETS.md](DATASETS.md) and their applicable
terms. No third-party instance files are redistributed in the current source tree.
The MIT License covers the original
project software and its documentation; it does not grant a new license to
third-party datasets or third-party components, which retain their own terms.

## License

The original project software is released under the [MIT License](LICENSE).
Copyright (c) 2026 Josep Suy and Mateu Villaret.

## Citation

If you use LPR-GNN in scientific work, please cite the associated manuscript:

> Josep Suy and Mateu Villaret. *Learning a State-Dependent Priority Rule for
> Serial Schedule Generation with Graph Neural Networks*. Manuscript.

Machine-readable metadata and the preferred article citation are provided in
[CITATION.cff](CITATION.cff). Publication details and a DOI will be added when
available. This citation request is a scholarly acknowledgement, not an
additional condition of the MIT License.
