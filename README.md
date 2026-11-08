# VOILA-3D

VOILA-3D evaluates when independent validation evidence supports using a
3D-augmented molecular expert. The current method, **TaskMetric-LCB**, compares
paired validation AUROC or MAE, computes a 2.5th-percentile bootstrap bound,
and freezes one expert-permission decision for every test molecule in an
endpoint/partition/model-seed state. Matched true-3D, seed-null, and permuted-3D
comparisons separate structural information from fitting variation.

The current protocol covers ten endpoints from four molecular data sources,
ten scaffold-randomized partitions, and five paired model seeds per partition.
QM9 endpoints share molecular identities and are not independent datasets.
Protection from negative transfer is evaluated jointly with retained positive
gain. The preceding molecule-level R9 experiments remain as historical diagnostics.

## Current TaskMetric-LCB Reproduction

The compact release includes paired validation/test predictions for all 500
true-3D states, exact molecule rows and split membership, the original aggregate
control results, and fixed protocol settings. No GPU is needed for prediction-level
reproduction:

```bash
python -m unittest discover -s tests -v
python scripts/reproduce_taskmetric.py
python scripts/analyze_taskmetric_reference.py
```

`TASKMETRIC_REPRODUCIBILITY.md` separates prediction-level verification from
conformer regeneration and complete expert refitting. The protocol is recorded in
`configs/taskmetric_protocol.json`. No molecule-level router or acquisition budget
enters the current permission rule.

This repository contains the experiment source used for the paper, compact
reference outputs, exact protocol settings, and verification utilities. Raw
datasets, feature matrices, model checkpoints, and environments remain excluded.
The compact current-method prediction archive is included for exact decision-level
reproduction.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `voila3d/` | Data loading, molecular features, metrics, and routing primitives |
| `experiments/` | Expert, router, ablation, robustness, and baseline experiments |
| `scripts/reproduce_paper.sh` | Staged reproduction entry point |
| `configs/paper_protocol.yaml` | Fixed splits, seeds, budgets, and policy settings |
| `configs/taskmetric_protocol.json` | Current ten-partition endpoint-permission specification |
| `scripts/reproduce_taskmetric.py` | Current paired-prediction decision reproduction |
| `results/reference/` | Compact, five-seed reference tables used for verification |
| `scripts/verify_reference_results.py` | Integrity and data-input checks |
| `tests/` | Network-free unit tests for core behavior |

## Environment

The primary experiments used Python 3.10.20 and the package versions pinned in
`environment.yml`.

```bash
conda env create -f environment.yml
conda activate voila3d
python -m unittest discover -s tests -v
python scripts/verify_reference_results.py
```

`requirements.txt` provides an equivalent pip-oriented specification. Chemprop
1.6.1 baselines use a separate environment because their NumPy and
scikit-learn constraints differ; see `requirements-chemprop.txt`.

## Historical Router Smoke Test

The smoke stage downloads ESOL and BACE automatically, generates one conformer,
and runs one seed on a small subset:

```bash
bash scripts/reproduce_paper.sh smoke
```

Use `XGB_DEVICE=cuda` to request a CUDA-enabled XGBoost build, or retain the
default `XGB_DEVICE=cpu` for a hardware-independent run.

## Historical Molecule-Level Experiments

The main stages are:

```bash
bash scripts/reproduce_paper.sh core
bash scripts/reproduce_paper.sh robustness
bash scripts/reproduce_paper.sh partitions
bash scripts/reproduce_paper.sh qm9
bash scripts/reproduce_paper.sh baselines
bash scripts/reproduce_paper.sh audits
```

`bash scripts/reproduce_paper.sh all` executes these stages in order. Full
reproduction includes conformer generation and five-seed model fitting, so its
runtime depends strongly on CPU parallelism and XGBoost acceleration. Resource
limits are controlled with `CONFORMER_JOBS`, `MODEL_THREADS`, and
`ROUTER_JOBS`.

The operational R9 policy and the fold-separated calibration analysis are distinct:

- `run_router_gated_selection.py` evaluates the operational Utility-LCB policy.
- `run_router_nested_calibration.py` evaluates permission using a calibration
  fold excluded from router and gate selection.

Detailed stage-to-output mapping is provided in `REPRODUCIBILITY.md`.

The `partitions` stage refits the paired experts and R7--R9 policies on five
additional randomized scaffold-disjoint assignments. It reuses only the
split-invariant molecular feature arrays from the primary run; model fitting,
OOF utility estimation, router selection, and test evaluation are repeated for
every assignment.

## Data and Results

`DATASETS.md` records the public download URLs and SHA-256 identities of the raw
inputs. Data are downloaded to `data/raw/` and are ignored by Git.

The repository includes aggregate tables and the compact current-method paired
prediction archive under `results/reference/`. The verification command checks
the released artifacts against `results/reference/SHA256SUMS.txt`. Generated
outputs elsewhere under `results/` are ignored by Git.

## License

The code is released under the MIT License. Public datasets remain subject to
their upstream terms.
