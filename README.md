# VOILA-3D

VOILA-3D is a reliability-aware system for adaptive 2D-to-2D+3D prediction
substitution in molecular property modeling. It learns sample-level paired
utility from out-of-fold predictions, ranks candidate molecules, and applies a
Utility-LCB permission rule that retains the complete 2D prediction when the
available evidence does not support using its augmented replacement.

This repository contains the experiment source used for the paper, compact
reference outputs, exact protocol settings, and verification utilities. Raw
datasets, feature caches, predictions, model checkpoints, and environments are
excluded to keep the repository small.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `voila3d/` | Data loading, molecular features, metrics, and routing primitives |
| `experiments/` | Expert, router, ablation, robustness, and baseline experiments |
| `scripts/reproduce_paper.sh` | Staged reproduction entry point |
| `configs/paper_protocol.yaml` | Fixed splits, seeds, budgets, and policy settings |
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

## Quick Smoke Test

The smoke stage downloads ESOL and BACE automatically, generates one conformer,
and runs one seed on a small subset:

```bash
bash scripts/reproduce_paper.sh smoke
```

Use `XGB_DEVICE=cuda` to request a CUDA-enabled XGBoost build, or retain the
default `XGB_DEVICE=cpu` for a hardware-independent run.

## Paper Experiments

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

The repository includes only compact aggregate tables under
`results/reference/`. The verification command checks every reference table
against `results/reference/SHA256SUMS.txt`. Generated outputs elsewhere under
`results/` are ignored by Git.

## License

The code is released under the MIT License. Public datasets remain subject to
their upstream terms.
