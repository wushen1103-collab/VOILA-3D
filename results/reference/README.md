# Reference Results

This directory contains compact CSV outputs from the five-seed experiments.
The original experiment directory names are retained so every table can be
traced to its generating script. Molecule-level predictions, feature arrays,
models, and logs are omitted.

Run the integrity check from the repository root:

```bash
python scripts/verify_reference_results.py
```

The most direct entry points are:

| Evidence | Directory or file |
| --- | --- |
| Operational R9 | `if_router_gated_selection_lcb_v1_5seed/` |
| Fold-separated audit | `if_router_nested_calibration_lcb_5seed/` |
| Objective sweep | `if_router_objective_sweep_v1_5seed/` |
| Normalized BIG | `if_router_lcb_final/` |
| Fusion and robustness | `if_full_audit/` and `jcim_full_audit/` |
| QM9 stress test | `qm9_conformation_stress_k10_full_5seed/` and `qm9_r9_lcb_guardrail_5seed/` |
| Same-protocol prediction | `comparisons/` and validation-ensemble directories |
