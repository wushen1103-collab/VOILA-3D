#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
XGB_DEVICE="${XGB_DEVICE:-cpu}"
CONFORMER_JOBS="${CONFORMER_JOBS:-8}"
MODEL_THREADS="${MODEL_THREADS:-8}"
ROUTER_JOBS="${ROUTER_JOBS:-8}"
STAGE="${1:-smoke}"

SEEDS=(0 1 2 3 4)
BUDGETS=(0 5 10 20 40 60 80 100)
PRIMARY_TASKS=(ESOL FreeSolv Lipophilicity BACE BBBP HIV)

run_if_missing() {
  local marker="$1"
  shift
  if [[ -f "$marker" ]]; then
    printf 'Skipping completed output: %s\n' "$marker"
    return
  fi
  printf 'Running: %q ' "$@"
  printf '\n'
  "$@"
}

run_smoke() {
  run_if_missing results/smoke/run_metadata.json \
    "$PYTHON" experiments/fast_screen.py \
    --tasks ESOL BACE \
    --split scaffold_balanced \
    --seeds 0 \
    --budgets 0 20 100 \
    --max-mols 500 \
    --conformers 1 \
    --feature2d-set ecfp_desc \
    --feature3d-set usr \
    --conformer-jobs "$CONFORMER_JOBS" \
    --model-threads "$MODEL_THREADS" \
    --xgb-device "$XGB_DEVICE" \
    --router-label-source oof \
    --router-folds 3 \
    --router-feature-set ecfp_desc \
    --out-dir results/smoke
}

run_core() {
  run_if_missing results/if_oof_usr_rdkit2d_combo_cls_5seed/run_metadata.json \
    "$PYTHON" experiments/fast_screen.py \
    --tasks BACE BBBP HIV \
    --split scaffold_balanced \
    --seeds "${SEEDS[@]}" \
    --budgets "${BUDGETS[@]}" \
    --max-mols 12000 \
    --conformers 10 \
    --feature2d-set rdkit2d_combo \
    --feature3d-set usr \
    --conformer-jobs "$CONFORMER_JOBS" \
    --parallel-tasks 1 \
    --model-threads "$MODEL_THREADS" \
    --xgb-device "$XGB_DEVICE" \
    --router-feature-set ecfp_desc \
    --classification-benefit auc_contrib \
    --router-label-source oof \
    --router-folds 5 \
    --out-dir results/if_oof_usr_rdkit2d_combo_cls_5seed

  run_if_missing results/if_oof_scalar_rdkit2d_combo_reg_5seed/run_metadata.json \
    "$PYTHON" experiments/fast_screen.py \
    --tasks ESOL FreeSolv Lipophilicity \
    --split scaffold_balanced \
    --seeds "${SEEDS[@]}" \
    --budgets "${BUDGETS[@]}" \
    --max-mols 12000 \
    --conformers 10 \
    --feature2d-set rdkit2d_combo \
    --feature3d-set rdkit_scalar \
    --conformer-jobs "$CONFORMER_JOBS" \
    --parallel-tasks 1 \
    --model-threads "$MODEL_THREADS" \
    --xgb-device "$XGB_DEVICE" \
    --router-feature-set ecfp_desc \
    --router-label-source oof \
    --router-folds 5 \
    --out-dir results/if_oof_scalar_rdkit2d_combo_reg_5seed

  run_if_missing results/if_router_objective_sweep_v1_5seed/run_metadata.json \
    "$PYTHON" experiments/run_router_objective_sweep.py \
    --oof-results-dir results/if_oof_usr_rdkit2d_combo_cls_5seed \
    --oof-results-dir results/if_oof_scalar_rdkit2d_combo_reg_5seed \
    --budgets "${BUDGETS[@]}" \
    --main-budget 20 \
    --router-estimators 128 \
    --router-n-jobs "$ROUTER_JOBS" \
    --pair-samples 30000 \
    --pair-margin-frac 0.25 \
    --bootstrap-iters 2000 \
    --out-dir results/if_router_objective_sweep_v1_5seed

  run_if_missing results/if_router_gated_selection_lcb_v1_5seed/run_metadata.json \
    "$PYTHON" experiments/run_router_gated_selection.py \
    --oof-results-dir results/if_oof_usr_rdkit2d_combo_cls_5seed \
    --oof-results-dir results/if_oof_scalar_rdkit2d_combo_reg_5seed \
    --budgets "${BUDGETS[@]}" \
    --main-budget 20 \
    --router-estimators 64 \
    --router-n-jobs "$ROUTER_JOBS" \
    --pair-samples 5000 \
    --pair-margin-frac 0.25 \
    --bootstrap-iters 1000 \
    --allow-safe-abstain \
    --safe-rule utility_lcb \
    --safe-bootstrap-iters 500 \
    --safe-min-selected 8 \
    --safe-utility-lcb-threshold 0.0 \
    --min-oof-big-to-call 0.0 \
    --out-dir results/if_router_gated_selection_lcb_v1_5seed

  run_if_missing results/if_router_nested_calibration_lcb_5seed/run_metadata.json \
    "$PYTHON" experiments/run_router_nested_calibration.py \
    --oof-results-dir results/if_oof_usr_rdkit2d_combo_cls_5seed \
    --oof-results-dir results/if_oof_scalar_rdkit2d_combo_reg_5seed \
    --out-dir results/if_router_nested_calibration_lcb_5seed \
    --router-estimators 48 \
    --router-n-jobs "$ROUTER_JOBS" \
    --pair-samples 3000 \
    --safe-bootstrap-iters 400 \
    --bootstrap-iters 800
}

run_usr_cache() {
  run_if_missing results/fast_screen_xgb_k10_auc_oof_ecfp_router_5seed_cls/run_metadata.json \
    "$PYTHON" experiments/fast_screen.py \
    --tasks "${PRIMARY_TASKS[@]}" \
    --split scaffold_balanced \
    --seeds "${SEEDS[@]}" \
    --budgets "${BUDGETS[@]}" \
    --max-mols 12000 \
    --conformers 10 \
    --feature2d-set rdkit2d_combo \
    --feature3d-set usr \
    --conformer-jobs "$CONFORMER_JOBS" \
    --model-threads "$MODEL_THREADS" \
    --xgb-device "$XGB_DEVICE" \
    --router-label-source oof \
    --router-folds 5 \
    --router-feature-set ecfp_desc \
    --classification-benefit auc_contrib \
    --out-dir results/fast_screen_xgb_k10_auc_oof_ecfp_router_5seed_cls
}

run_robustness() {
  run_core
  run_usr_cache

  run_if_missing results/if_fusion_baselines_usr_5seed/run_metadata.json \
    "$PYTHON" experiments/run_if_fusion_baselines.py \
    --tasks "${PRIMARY_TASKS[@]}" \
    --seeds "${SEEDS[@]}" \
    --split scaffold_balanced \
    --max-mols 12000 \
    --feature2d-set rdkit2d_combo \
    --feature3d-set usr \
    --feature3d-source-dir results/fast_screen_xgb_k10_auc_oof_ecfp_router_5seed_cls \
    --combo-jobs "$CONFORMER_JOBS" \
    --threads "$MODEL_THREADS" \
    --xgb-device "$XGB_DEVICE" \
    --out-dir results/if_fusion_baselines_usr_5seed

  run_if_missing results/if_noisy3d_stress_usr_5seed/run_metadata.json \
    "$PYTHON" experiments/run_if_noisy3d_stress.py \
    --tasks "${PRIMARY_TASKS[@]}" \
    --seeds "${SEEDS[@]}" \
    --noise-levels 0 0.1 0.25 0.5 1.0 \
    --budgets "${BUDGETS[@]}" \
    --split scaffold_balanced \
    --max-mols 12000 \
    --feature2d-set rdkit2d_combo \
    --feature3d-set usr \
    --feature3d-source-dir results/fast_screen_xgb_k10_auc_oof_ecfp_router_5seed_cls \
    --combo-jobs "$CONFORMER_JOBS" \
    --threads "$MODEL_THREADS" \
    --xgb-device "$XGB_DEVICE" \
    --out-dir results/if_noisy3d_stress_usr_5seed

  run_if_missing results/if_router_input_ablation_usr_full_cached_5seed/run_metadata.json \
    "$PYTHON" experiments/run_if_router_ablation.py \
    --oof-results-dir results/if_oof_usr_rdkit2d_combo_cls_5seed \
    --oof-results-dir results/if_oof_scalar_rdkit2d_combo_reg_5seed \
    --budgets "${BUDGETS[@]}" \
    --router-n-jobs "$ROUTER_JOBS" \
    --out-dir results/if_router_input_ablation_usr_full_cached_5seed
}

run_qm9() {
  run_if_missing results/qm9_conformation_stress_k10_full_5seed/run_metadata.json \
    "$PYTHON" experiments/fast_screen.py \
    --tasks QM9_MU QM9_R2 \
    --split random \
    --seeds "${SEEDS[@]}" \
    --budgets "${BUDGETS[@]}" \
    --max-mols 2000 \
    --conformers 10 \
    --feature2d-set rdkit2d_combo \
    --feature3d-set rdkit_full \
    --conformer-jobs "$CONFORMER_JOBS" \
    --parallel-tasks 1 \
    --model-threads "$MODEL_THREADS" \
    --xgb-device "$XGB_DEVICE" \
    --router-label-source oof \
    --router-folds 5 \
    --router-feature-set ecfp_desc \
    --out-dir results/qm9_conformation_stress_k10_full_5seed

  run_if_missing results/qm9_r9_lcb_guardrail_5seed/run_metadata.json \
    "$PYTHON" experiments/run_router_gated_selection.py \
    --oof-results-dir results/qm9_conformation_stress_k10_full_5seed \
    --out-dir results/qm9_r9_lcb_guardrail_5seed \
    --tasks QM9_MU QM9_R2 \
    --seeds "${SEEDS[@]}" \
    --budgets "${BUDGETS[@]}" \
    --main-budget 20 \
    --router-estimators 64 \
    --router-n-jobs "$ROUTER_JOBS" \
    --pair-samples 5000 \
    --bootstrap-iters 800 \
    --allow-safe-abstain \
    --safe-rule utility_lcb \
    --safe-bootstrap-iters 500 \
    --safe-min-selected 8 \
    --safe-utility-lcb-threshold 0.0
}

run_baselines() {
  run_usr_cache
  run_if_missing results/baseline_matrix_scaffold_balanced_5seed/run_metadata.json \
    "$PYTHON" experiments/run_baseline_matrix.py \
    --tasks "${PRIMARY_TASKS[@]}" \
    --split scaffold_balanced \
    --seeds "${SEEDS[@]}" \
    --max-mols 12000 \
    --threads "$MODEL_THREADS" \
    --feature2d-set rdkit2d_combo \
    --combo-jobs "$CONFORMER_JOBS" \
    --xgb-device "$XGB_DEVICE" \
    --out-dir results/baseline_matrix_scaffold_balanced_5seed

  run_if_missing results/label_efficiency_k10_usr_5seed/run_metadata.json \
    "$PYTHON" experiments/run_label_efficiency.py \
    --tasks "${PRIMARY_TASKS[@]}" \
    --seeds "${SEEDS[@]}" \
    --split scaffold_balanced \
    --conformers 10 \
    --feature3d-set usr \
    --feature2d-set rdkit2d_combo \
    --feature-cache-dir results/fast_screen_xgb_k10_auc_oof_ecfp_router_5seed_cls/feature_cache \
    --model-threads "$MODEL_THREADS" \
    --xgb-device "$XGB_DEVICE" \
    --combo-jobs "$CONFORMER_JOBS" \
    --out-dir results/label_efficiency_k10_usr_5seed
}

run_audits() {
  run_core
  run_usr_cache
  run_baselines

  run_if_missing results/if_router_lcb_final/normalized_big_macro_summary.csv \
    "$PYTHON" scripts/audit_normalized_big.py \
    --results-dir results/if_router_gated_selection_lcb_v1_5seed \
    --report-dir results/if_router_lcb_final \
    --main-budget 20 \
    --bootstrap-iters 5000 \
    --bootstrap-seed 3409

  run_if_missing results/fast_screen_scaffold_balanced/run_metadata.json \
    "$PYTHON" experiments/fast_screen.py \
    --tasks "${PRIMARY_TASKS[@]}" \
    --split scaffold_balanced \
    --seeds "${SEEDS[@]}" \
    --budgets "${BUDGETS[@]}" \
    --max-mols 12000 \
    --conformers 1 \
    --feature2d-set ecfp_desc \
    --feature3d-set usr \
    --conformer-jobs "$CONFORMER_JOBS" \
    --model-threads "$MODEL_THREADS" \
    --xgb-device "$XGB_DEVICE" \
    --router-label-source oof \
    --router-folds 5 \
    --out-dir results/fast_screen_scaffold_balanced

  run_if_missing results/jcim_full_audit/experiment_status_checklist.csv \
    "$PYTHON" scripts/audit_jcim_full.py \
    --repo-root . \
    --lcb-dir results/if_router_gated_selection_lcb_v1_5seed \
    --cache-dir results/fast_screen_xgb_k10_auc_oof_ecfp_router_5seed_cls \
    --k1-dir results/fast_screen_scaffold_balanced \
    --k10-dir results/fast_screen_xgb_k10_auc_oof_ecfp_router_5seed_cls \
    --baseline-dir results/baseline_matrix_scaffold_balanced_5seed \
    --out-dir results/jcim_full_audit
}

case "$STAGE" in
  smoke) run_smoke ;;
  core) run_core ;;
  robustness) run_robustness ;;
  qm9) run_qm9 ;;
  baselines) run_baselines ;;
  audits) run_audits ;;
  all)
    run_core
    run_robustness
    run_qm9
    run_baselines
    run_audits
    ;;
  *)
    printf 'Unknown stage: %s\n' "$STAGE" >&2
    printf 'Choose one of: smoke, core, robustness, qm9, baselines, audits, all\n' >&2
    exit 2
    ;;
esac
