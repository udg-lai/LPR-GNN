# Paper results

Results for *LPR-GNN*, *classical priority rules* and the *Wheatley* neural baseline.
The supplied inference weights belong only to the 16-iteration shared-GRU
*LPR-GNN* model (41,514 parameters), not to *Wheatley*.

## Contents

- `lpr_gnn_results.csv`: per-instance makespans and measured times exported from
  the base model's current `makespan_results.xlsx`.
- `other_priority_rules_results.csv`: individual runs of *classical priority rules*
  and *Wheatley* rollouts, including makespans and recorded times. All original
  classical-rule runs and repetitions are preserved without averaging. *Wheatley*
  contributes one deterministic greedy rollout per instance.
- `best.pt`: the validation-selected model's state dictionary. All weight tensors
  are unchanged. Optimizer state and machine-specific configuration are omitted.
- `manifest.json`: source filenames, record counts and
  checkpoint-selection metadata for *LPR-GNN* and *Wheatley*.

The combined CSV contains 147,850 classical priority-rule runs and 4,375 *Wheatley*
rollouts. Select `rule=wheatley` to extract *Wheatley* results. Its
`method_type=neural_RL` distinguishes it from `method_type=handcrafted`.
The latter is a retained CSV identifier for *classical priority rules*;
the descriptive terminology follows the article.

CSV headers are in English. `family` identifies each benchmark family; `PACK`
corresponds to input filenames beginning with `pac`. Included families are J30,
J60, J90, J120, PACK, RG30 and RG300. Absolute input paths and long schedule/selection sequences are not
included. The source workbooks are left unchanged.

The result files also contain J30/J60/J90 records. Do not treat these as unseen
benchmarks: these families include instances used for training and validation.
The article's schedule-level benchmark families are J120, PACK, RG30 and RG300.
Blank gap fields mean no reference was available. A negative reference value is
the original missing-reference sentinel, not an optimal makespan.

For *classical priority rules*, the `gap_to_best_*` fields use the source workbook's best recorded classical-rule
makespan. They are not automatically gaps to certified optima or literature
best-known values. Recompute article gaps using the corresponding reference values.
`method_type` distinguishes *classical priority rules* from the neural reinforcement-learning
baseline. *Wheatley* results use the validation-selected checkpoint at cycle 45,
evaluated on CPU. `feasible=True` records a successful feasibility check for each
*Wheatley* rollout; blank classical-rule values mean that this flag was not present in the
export, not that classical-rule schedules are infeasible. *Wheatley*'s gap fields are blank:
no literature-reference or classical-rule-best gap has been invented.

Timing columns retain the original measurement definitions. `time_scope` records
their boundaries. *Wheatley*'s `total_seconds` includes instance/environment
preparation and the rollout; `parse_seconds` and `schedule_seconds` are blank
because separate measurements are unavailable. It excludes one-off checkpoint
loading and the subsequent feasibility audit. Batched *LPR-GNN* times and the other
methods' times do not necessarily have identical timing boundaries.

## Article comparison

The article uses J120 (600), RG30 (1,800), RG300 (480) and PACK (55),
for 2,935 instances. The 22 *classical priority rules* are:
SPT, LPT, MIS, MTS, LRPW, GRPW, LRW, GRW, EST, EFT, LST, LFT, ESTD,
EFTD, LMSLK, GMSLK, LRWC, GRWC, LCRWC, GCRWC, LCRWCS and GCRWCS.
Their CSV identifiers are `spt`, `lpt`, `mis`, `mts`, `lrpw`, `grpw`,
`lrw`, `grw`, `est`, `eft`, `lst`, `lft`, `dest`, `deft`, `mslk`,
`mmslk`, `lrwc`, `grwc`, `lcrwc`, `gcrwc`, `lcrwcs` and `gcrwcs`.
Other archived rules and repetitions are not part of this 22-rule selection.
The mean runtime for the classical rules averages these 22 rules, rather than
summing the cost of running them all; the pooled mean weights instances equally.

The source distribution supplies the shared-GRU *LPR-GNN* weights only.
The article's other trained variants and eleven-model oracle portfolio are
not additional model checkpoints bundled in this archive.

## Use the supplied weights

In the checkpoint-path cell of `makespan_predictor.ipynb`, set:

```python
CHECKPOINT_PATH = PROJECT_ROOT / "paper_results" / "best.pt"
```

Keep the base architecture in `config.yaml`. The supplied state dictionary is
accepted by `load_model_state`. It is for inference, not resuming optimizer state.
Obtain raw instances separately as described in `../DATASETS.md`.
