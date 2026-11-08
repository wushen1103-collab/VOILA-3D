# TaskMetric-LCB Reproducibility

## Current and Historical Methods

The current study uses one frozen endpoint-level permission decision per
scaffold partition and matched expert seed. The earlier R9 molecule-level router,
budgets, and fixed-partition diagnostics are historical experiments. They are
kept under their existing names and are not inputs to TaskMetric-LCB.

The current evaluation has six main endpoints and four additional QM9 endpoints,
not ten independent molecular datasets. Their exact membership is recorded in
`configs/taskmetric_protocol.json`. Model seeds 0--4 are nested within each of
the ten scaffold assignment seeds 1--10.

## Prediction-Level Reproduction

Install the pinned environment from `environment.yml`, then run from the
repository root:

```bash
python -m unittest discover -s tests -v
python scripts/reproduce_taskmetric.py
python scripts/analyze_taskmetric_reference.py
```

The first script reads `results/reference/taskmetric_20261108/paired_predictions.zip`,
which contains 500 compressed NumPy records with validation/test targets,
paired expert predictions, and original molecule-row indices. It recomputes
endpoint gains and the 5,000-resample bootstrap bounds, checks them against the
frozen reference states, and calculates all seven selection policies. It uses
validation targets only to determine permission. Test targets score the frozen
decisions. `--limit 2` provides a small smoke test without changing the full
protocol.

The second script uses compact aggregate states to regenerate equal-endpoint
risk-benefit tables and paired 10,000-replicate partition-cluster intervals.
Each bootstrap draw preserves all five model seeds. The endpoint set remains
fixed, and undefined positive/negative denominators are omitted per metric.
Native AUROC and MAE changes are not averaged across endpoints.

## Rebuild Features and Refit Experts

The prediction archive reproduces the reported selection results without
regenerating conformers or fitting models. A full computational rerun starts
with the published molecular identities and target values:

```bash
python scripts/prepare_taskmetric_features.py --jobs 4
```

This creates the feature caches expected by the original experiment code.
The manifest fixes row ordering, including the 2,000 QM9 molecules, rather
than resampling a fresh QM9 subset. Generated feature arrays, models, and raw
download files are excluded from Git. Regeneration requires the pinned RDKit
version; conformer timeouts can depend on the execution platform.

For each split seed 1--10, refit the fixed paired experts and all six original
selectors. The following command illustrates split 1; repeat it for each split:

```bash
python experiments/run_task_selector_comparison.py \
  --tasks BBBP ESOL FreeSolv QM9_GAP QM9_MU QM9_R2 QM9_A QM9_B QM9_C QM9_ALPHA \
  --conditions same_seed_true3d seed_null_2d permuted_3d_placebo \
  --split-seed 1 --split-protocol scaffold_all --seeds 0 1 2 3 4 \
  --bootstrap-iters 5000 --relative-thresholds 0.005 0.01 \
  --model-threads 8 --xgb-device cuda --out-dir results/taskmetric_refit/split1
```

Run the standard paired-test comparator on the same expert specification:

```bash
python experiments/run_paired_statistical_selector.py \
  --tasks BBBP ESOL FreeSolv QM9_GAP QM9_MU QM9_R2 QM9_A QM9_B QM9_C QM9_ALPHA \
  --reference results/reference/taskmetric_20261108/expert_states.csv.gz \
  --threads 8 --device cuda --out-dir results/paired_test_refit
```

The paired-test runner checks all four reference/augmented validation/test
metrics against the frozen state before accepting a result. It uses a paired
t test of absolute-error reductions for regression and paired DeLong covariance
for AUROC, including ties. Its one-sided alpha is fixed at 0.025, matching the
bootstrap's lower-tail probability. No threshold or test family is selected
from the test outcomes.

## Files in the Compact Release

| File | Content |
| --- | --- |
| `expert_states.csv.gz` | 500 original true-3D fitted-pair states and bootstrap evidence |
| `paired_predictions.zip` | Validation/test targets, predictions, and indices for those 500 states |
| `all_selector_states.csv.gz` | Original 9,000 policy records across true-3D, seed-null, and permuted-3D conditions |
| `paired_test_rows.csv` | 500 matched conventional paired-test decisions |
| `molecule_manifest.csv.gz` | Canonical molecular identities, row order, and targets |
| `partition_membership.csv.gz` | Complete train/validation/test allocation for every endpoint and split |
| `verification.json` | Refit metric parity and decision-agreement checks |
| `SHA256SUMS.txt` | Hash identities of the released reference inputs |

`results/taskmetric_reproduced/` and `results/taskmetric_reference_analysis/`
are generated outputs. They are not read by the permission rule.
