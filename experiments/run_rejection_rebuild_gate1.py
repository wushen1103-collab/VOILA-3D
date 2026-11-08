from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from experiments.fast_screen import _fit_xgb, _make_router_features, _predict
from voila3d.data import assign_splits, scaffold_for_smiles
from voila3d.features import build_2d_feature_matrix
from voila3d.metrics import classification_metrics, regression_metrics
from voila3d.routing import train_voi_router, true_benefit


TASK_CACHE = {
    "BBBP": {
        "root": "results/if_oof_usr_rdkit2d_combo_cls_5seed/feature_cache",
        "prefix": "BBBP_scaffold_balanced_max12000_k10",
        "task_type": "classification",
        "classification_mode": "auc_contrib",
    },
    "ESOL": {
        "root": "results/if_oof_scalar_rdkit2d_combo_reg_5seed/feature_cache",
        "prefix": "ESOL_scaffold_balanced_max12000_k10_rdkit_scalar",
        "task_type": "regression",
        "classification_mode": "logloss",
    },
    "FreeSolv": {
        "root": "results/if_oof_scalar_rdkit2d_combo_reg_5seed/feature_cache",
        "prefix": "FreeSolv_scaffold_balanced_max12000_k10_rdkit_scalar",
        "task_type": "regression",
        "classification_mode": "logloss",
    },
    "QM9_R2": {
        "root": "results/qm9_conformation_stress_k10_full_5seed/feature_cache",
        "prefix": "QM9_R2_random_max2000_k10_rdkit_full",
        "task_type": "regression",
        "classification_mode": "logloss",
    },
    "QM9_MU": {
        "root": "results/qm9_conformation_stress_k10_full_5seed/feature_cache",
        "prefix": "QM9_MU_random_max2000_k10_rdkit_full",
        "task_type": "regression",
        "classification_mode": "logloss",
    },
    "QM9_GAP": {
        "root": "results/qm9_gap_r2_smoke_k10_full_val/feature_cache",
        "prefix": "QM9_GAP_random_max2000_k10_rdkit_full",
        "task_type": "regression",
        "classification_mode": "logloss",
    },
    "QM9_A": {
        "root": "results/rejection_rebuild_qm9_geometry_endpoints/feature_cache",
        "prefix": "QM9_A_random_max2000_k10_rdkit_full",
        "task_type": "regression",
        "classification_mode": "logloss",
    },
    "QM9_B": {
        "root": "results/rejection_rebuild_qm9_geometry_endpoints/feature_cache",
        "prefix": "QM9_B_random_max2000_k10_rdkit_full",
        "task_type": "regression",
        "classification_mode": "logloss",
    },
    "QM9_C": {
        "root": "results/rejection_rebuild_qm9_geometry_endpoints/feature_cache",
        "prefix": "QM9_C_random_max2000_k10_rdkit_full",
        "task_type": "regression",
        "classification_mode": "logloss",
    },
    "QM9_ALPHA": {
        "root": "results/rejection_rebuild_qm9_geometry_endpoints/feature_cache",
        "prefix": "QM9_ALPHA_random_max2000_k10_rdkit_full",
        "task_type": "regression",
        "classification_mode": "logloss",
    },
}

CONDITIONS = (
    "original_true3d",
    "same_seed_true3d",
    "seed_null_2d",
    "permuted_3d_placebo",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", choices=sorted(TASK_CACHE), default=sorted(TASK_CACHE))
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--budget", type=float, default=20.0)
    p.add_argument(
        "--split-seed",
        type=int,
        default=-1,
        help="Use cached primary split at -1; otherwise regenerate a seeded scaffold-randomized split (random for QM9_R2).",
    )
    p.add_argument("--oof-folds", type=int, default=5)
    p.add_argument("--bootstrap-iters", type=int, default=2000)
    p.add_argument("--random-repeats", type=int, default=200)
    p.add_argument("--model-threads", type=int, default=16)
    p.add_argument("--xgb-device", choices=["cpu", "cuda", "auto"], default="cuda")
    p.add_argument("--out-dir", default="results/rejection_rebuild_gate1")
    return p.parse_args()


@lru_cache(maxsize=None)
def _load_task(task: str, split_seed: int) -> dict:
    cfg = TASK_CACHE[task]
    root = Path(cfg["root"])
    prefix = cfg["prefix"]
    meta = pd.read_csv(root / f"{prefix}_meta.csv")
    x_ecfp = np.load(root / f"{prefix}_ecfp.npy", mmap_mode="r")
    x_desc = np.load(root / f"{prefix}_desc.npy", mmap_mode="r")
    x3d = np.load(root / f"{prefix}_x3d.npy", mmap_mode="r")
    if not (len(meta) == len(x_ecfp) == len(x_desc) == len(x3d)):
        raise ValueError(f"Feature length mismatch for {task}")
    x2d = build_2d_feature_matrix(
        meta["canonical_smiles"].astype(str).tolist(),
        np.asarray(x_ecfp),
        np.asarray(x_desc),
        feature2d_set="rdkit2d_combo",
        jobs=min(16, max(1, (os.cpu_count() or 1) // 4)),
    )
    if split_seed >= 0:
        split_kind = "random" if task.startswith("QM9_") else "scaffold_randomized"
        split = assign_splits(
            meta,
            split_kind,
            seed=split_seed,
            frac_train=0.8,
            frac_val=0.1,
            task_type=cfg["task_type"],
        ).astype(str)
    else:
        split = meta["split"].astype(str).to_numpy()
    return {
        **cfg,
        "meta": meta,
        "x2d": x2d,
        "x_ecfp": np.asarray(x_ecfp, dtype=np.float32),
        "x_desc": np.asarray(x_desc, dtype=np.float32),
        "x3d": np.asarray(x3d, dtype=np.float32),
        "y": meta["y"].to_numpy(),
        "train": split == "train",
        "val": split == "val",
        "test": split == "test",
    }


def _condition_features(data: dict, condition: str, task: str) -> tuple[np.ndarray, np.ndarray, int]:
    x2d = data["x2d"]
    x3d = data["x3d"]
    if condition == "seed_null_2d":
        return x2d, x2d, 1000
    if condition == "permuted_3d_placebo":
        rng = np.random.default_rng(340_000 + sum(map(ord, task)))
        perm = np.arange(len(x3d))
        for mask_name in ("train", "val", "test"):
            idx = np.flatnonzero(data[mask_name])
            perm[idx] = rng.permutation(idx)
        return x2d, np.hstack([x2d, x3d[perm]]).astype(np.float32), 0
    seed_offset = 1000 if condition == "original_true3d" else 0
    return x2d, np.hstack([x2d, x3d]).astype(np.float32), seed_offset


def _metric(task_type: str, y: np.ndarray, pred: np.ndarray) -> tuple[str, float]:
    if task_type == "classification":
        return "AUROC", classification_metrics(y, pred)["AUROC"]
    return "MAE", regression_metrics(y, pred)["MAE"]


def _metric_gain(task_type: str, baseline: float, candidate: float) -> float:
    return candidate - baseline if task_type == "classification" else baseline - candidate


def _top_mask(scores: np.ndarray, budget: float) -> np.ndarray:
    n = len(scores)
    k = max(0, min(n, int(math.ceil(n * budget / 100.0))))
    out = np.zeros(n, dtype=bool)
    if k:
        out[np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="stable")[:k]] = True
    return out


def _top_k_mask(scores: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros(len(scores), dtype=bool)
    if k > 0:
        out[np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="stable")[: min(k, len(scores))]] = True
    return out


def _bootstrap_lcb(values: np.ndarray, iters: int, seed: int) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 8:
        return float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(iters, dtype=float)
    for i in range(iters):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.quantile(means, 0.025))


def _oof_predictions(
    task_type: str,
    x_ref: np.ndarray,
    x_alt: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    groups: np.ndarray,
    seed: int,
    alt_seed_offset: int,
    folds: int,
    threads: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.asarray(train_idx, dtype=int)
    pred_ref = np.full(len(idx), np.nan, dtype=float)
    pred_alt = np.full(len(idx), np.nan, dtype=float)
    fold_ids = np.full(len(idx), -1, dtype=int)
    n_groups = len(np.unique(groups))
    n_splits = min(folds, n_groups)
    if task_type == "classification":
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + 1701)
        split_iter = splitter.split(idx, y[idx], groups)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(idx, y[idx], groups)
    for fold_id, (fit_rel, hold_rel) in enumerate(split_iter):
        fit_idx, hold_idx = idx[fit_rel], idx[hold_rel]
        ref, _ = _fit_xgb(task_type, x_ref[fit_idx], y[fit_idx], seed + 2000 + fold_id, threads, device)
        alt, _ = _fit_xgb(
            task_type,
            x_alt[fit_idx],
            y[fit_idx],
            seed + 2000 + fold_id + alt_seed_offset,
            threads,
            device,
        )
        pred_ref[hold_rel] = _predict(ref, task_type, x_ref[hold_idx])
        pred_alt[hold_rel] = _predict(alt, task_type, x_alt[hold_idx])
        fold_ids[hold_rel] = fold_id
    return pred_ref, pred_alt, fold_ids


def _diagnostics(utility: np.ndarray, scores: np.ndarray, selected: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(utility) & np.isfinite(scores)
    sign = utility[finite] > 0
    sign_auc = float(roc_auc_score(sign, scores[finite])) if len(np.unique(sign)) == 2 else float("nan")
    sign_ap = float(average_precision_score(sign, scores[finite])) if sign.any() else float("nan")
    rho = spearmanr(utility[finite], scores[finite]).statistic if finite.sum() > 2 else float("nan")
    vals = utility[selected]
    oracle_positive = np.maximum(utility, 0.0).sum()
    captured = np.maximum(vals, 0.0).sum() / oracle_positive if oracle_positive > 0 else float("nan")
    return {
        "utility_sign_auroc": sign_auc,
        "utility_sign_auprc": sign_ap,
        "utility_spearman": float(rho),
        "selected_n": int(selected.sum()),
        "selected_mean_utility": float(vals.mean()) if len(vals) else 0.0,
        "selected_sum_utility": float(vals.sum()) if len(vals) else 0.0,
        "selected_harm_fraction": float((vals < 0).mean()) if len(vals) else 0.0,
        "selected_beneficial_fraction": float((vals > 0).mean()) if len(vals) else 0.0,
        "captured_positive_oracle": float(captured),
    }


def _route_row(
    task: str,
    task_type: str,
    condition: str,
    seed: int,
    policy: str,
    y_test: np.ndarray,
    pred_ref: np.ndarray,
    pred_alt: np.ndarray,
    utility: np.ndarray,
    scores: np.ndarray,
    selected: np.ndarray,
    permission: bool | None,
    cal_lcb: float | None,
    random_repeat: int | None = None,
) -> dict:
    routed = pred_ref.copy()
    routed[selected] = pred_alt[selected]
    metric_name, base_value = _metric(task_type, y_test, pred_ref)
    _, value = _metric(task_type, y_test, routed)
    row = {
        "task": task,
        "task_type": task_type,
        "condition": condition,
        "seed": seed,
        "policy": policy,
        "metric": metric_name,
        "baseline_value": base_value,
        "value": value,
        "metric_gain": _metric_gain(task_type, base_value, value),
        "call_rate": float(selected.mean() * 100.0),
        "permission": permission,
        "calibration_lcb": cal_lcb,
        "random_repeat": random_repeat,
    }
    row.update(_diagnostics(utility, scores, selected))
    return row


def run_one(task: str, condition: str, seed: int, args: argparse.Namespace) -> tuple[list[dict], dict, pd.DataFrame]:
    started = time.perf_counter()
    data = _load_task(task, args.split_seed)
    task_type = data["task_type"]
    mode = data["classification_mode"]
    x_ref, x_alt, alt_offset = _condition_features(data, condition, task)
    y = data["y"]
    train_idx = np.flatnonzero(data["train"])
    val_idx = np.flatnonzero(data["val"])
    test_idx = np.flatnonzero(data["test"])
    scaffolds = np.asarray([scaffold_for_smiles(s) for s in data["meta"].iloc[train_idx]["canonical_smiles"]])

    pred_ref_oof, pred_alt_oof, fold_ids = _oof_predictions(
        task_type,
        x_ref,
        x_alt,
        y,
        train_idx,
        scaffolds,
        seed,
        alt_offset,
        args.oof_folds,
        args.model_threads,
        args.xgb_device,
    )
    ok = np.isfinite(pred_ref_oof) & np.isfinite(pred_alt_oof)
    oof_idx = train_idx[ok]
    u_oof = true_benefit(task_type, y[oof_idx], pred_ref_oof[ok], pred_alt_oof[ok], classification_mode=mode)
    router_x_oof = _make_router_features(
        data["x_ecfp"][oof_idx],
        data["x_desc"][oof_idx],
        pred_ref_oof[ok],
        task_type,
        "ecfp_desc",
        uncertainty_reference=pred_ref_oof[ok],
    )
    router = train_voi_router(router_x_oof, u_oof, seed=seed)

    ref, _ = _fit_xgb(task_type, x_ref[train_idx], y[train_idx], seed, args.model_threads, args.xgb_device)
    alt, _ = _fit_xgb(task_type, x_alt[train_idx], y[train_idx], seed + alt_offset, args.model_threads, args.xgb_device)
    pred_ref_val = _predict(ref, task_type, x_ref[val_idx])
    pred_alt_val = _predict(alt, task_type, x_alt[val_idx])
    pred_ref_test = _predict(ref, task_type, x_ref[test_idx])
    pred_alt_test = _predict(alt, task_type, x_alt[test_idx])
    u_val = true_benefit(task_type, y[val_idx], pred_ref_val, pred_alt_val, classification_mode=mode)
    u_test = true_benefit(task_type, y[test_idx], pred_ref_test, pred_alt_test, classification_mode=mode)
    router_x_val = _make_router_features(
        data["x_ecfp"][val_idx],
        data["x_desc"][val_idx],
        pred_ref_val,
        task_type,
        "ecfp_desc",
        uncertainty_reference=pred_ref_oof[ok],
    )
    router_x_test = _make_router_features(
        data["x_ecfp"][test_idx],
        data["x_desc"][test_idx],
        pred_ref_test,
        task_type,
        "ecfp_desc",
        uncertainty_reference=pred_ref_oof[ok],
    )
    score_val = router.predict(router_x_val)
    score_test = router.predict(router_x_test)

    cal_selected = _top_mask(score_val, args.budget)
    cal_lcb = _bootstrap_lcb(u_val[cal_selected], args.bootstrap_iters, seed + 9100)
    permission = bool(np.isfinite(cal_lcb) and cal_lcb > 0.0)
    task_lcb = _bootstrap_lcb(u_val, args.bootstrap_iters, seed + 9200)
    task_permission = bool(np.isfinite(task_lcb) and task_lcb > 0.0)
    r9_selected = _top_mask(score_test, args.budget) if permission else np.zeros(len(test_idx), dtype=bool)
    simple_selected = _top_mask(score_test, args.budget) & (score_test > 0)
    task_select = np.ones(len(test_idx), dtype=bool) if float(np.mean(u_val)) > 0 else np.zeros(len(test_idx), dtype=bool)
    task_lcb_selected = np.ones(len(test_idx), dtype=bool) if task_permission else np.zeros(len(test_idx), dtype=bool)
    oracle_selected = _top_mask(u_test, args.budget)

    rows = []
    zero = np.zeros(len(test_idx), dtype=bool)
    rows.append(_route_row(task, task_type, condition, seed, "2D_reference", y[test_idx], pred_ref_test, pred_alt_test, u_test, score_test, zero, None, None))
    rows.append(_route_row(task, task_type, condition, seed, "100%_alternative", y[test_idx], pred_ref_test, pred_alt_test, u_test, score_test, np.ones(len(test_idx), dtype=bool), None, None))
    rows.append(_route_row(task, task_type, condition, seed, "TaskSelect", y[test_idx], pred_ref_test, pred_alt_test, u_test, score_test, task_select, bool(task_select.any()), None))
    rows.append(_route_row(task, task_type, condition, seed, "TaskLCB", y[test_idx], pred_ref_test, pred_alt_test, u_test, score_test, task_lcb_selected, task_permission, task_lcb))
    rows.append(_route_row(task, task_type, condition, seed, "SimpleUtility", y[test_idx], pred_ref_test, pred_alt_test, u_test, score_test, simple_selected, None, None))
    rows.append(_route_row(task, task_type, condition, seed, "R9_HoldoutLCB", y[test_idx], pred_ref_test, pred_alt_test, u_test, score_test, r9_selected, permission, cal_lcb))
    rows.append(_route_row(task, task_type, condition, seed, "Oracle@20", y[test_idx], pred_ref_test, pred_alt_test, u_test, u_test, oracle_selected, True, None))

    fixed_random = np.random.default_rng(seed + 7000)
    k_budget = int(_top_mask(score_test, args.budget).sum())
    for repeat in range(args.random_repeats):
        selected = np.zeros(len(test_idx), dtype=bool)
        if k_budget:
            selected[fixed_random.choice(len(test_idx), size=k_budget, replace=False)] = True
        rows.append(_route_row(task, task_type, condition, seed, "Random@20", y[test_idx], pred_ref_test, pred_alt_test, u_test, score_test, selected, None, None, repeat))

    matched_rng = np.random.default_rng(seed + 8000)
    k_r9 = int(r9_selected.sum())
    for repeat in range(args.random_repeats):
        selected = np.zeros(len(test_idx), dtype=bool)
        if k_r9:
            selected[matched_rng.choice(len(test_idx), size=k_r9, replace=False)] = True
        rows.append(_route_row(task, task_type, condition, seed, "Random_call_matched_R9", y[test_idx], pred_ref_test, pred_alt_test, u_test, score_test, selected, None, None, repeat))

    for row in rows:
        row["split_seed"] = int(args.split_seed)

    metric_name, ref_value = _metric(task_type, y[test_idx], pred_ref_test)
    _, alt_value = _metric(task_type, y[test_idx], pred_alt_test)
    summary = {
        "task": task,
        "task_type": task_type,
        "condition": condition,
        "seed": seed,
        "split_seed": int(args.split_seed),
        "metric": metric_name,
        "reference_value": ref_value,
        "alternative_value": alt_value,
        "alternative_gain": _metric_gain(task_type, ref_value, alt_value),
        "oof_utility_mean": float(np.mean(u_oof)),
        "oof_utility_std": float(np.std(u_oof)),
        "oof_positive_rate": float(np.mean(u_oof > 0)),
        "validation_utility_mean": float(np.mean(u_val)),
        "validation_lcb_at_budget": cal_lcb,
        "validation_task_lcb": task_lcb,
        "permission": permission,
        "task_permission": task_permission,
        "elapsed_sec": time.perf_counter() - started,
    }
    pred = pd.DataFrame(
        {
            "task": task,
            "condition": condition,
            "seed": seed,
            "split_seed": int(args.split_seed),
            "mol_id": data["meta"].iloc[test_idx]["mol_id"].to_numpy(),
            "y": y[test_idx],
            "pred_reference": pred_ref_test,
            "pred_alternative": pred_alt_test,
            "utility": u_test,
            "router_score": score_test,
            "r9_selected": r9_selected,
        }
    )
    return rows, summary, pred


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(exist_ok=True)
    metadata = {
        "started_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tasks": args.tasks,
        "conditions": args.conditions,
        "seeds": args.seeds,
        "budget": args.budget,
        "split_seed": int(args.split_seed),
        "oof_protocol": "train-only scaffold-grouped five-fold OOF",
        "permission_protocol": "independent validation partition bootstrap LCB",
        "test_protocol": "untouched until final evaluation",
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    all_rows: list[dict] = []
    all_summaries: list[dict] = []
    for task in args.tasks:
        for condition in args.conditions:
            for seed in args.seeds:
                print(f"[start] task={task} condition={condition} seed={seed}", flush=True)
                rows, summary, pred = run_one(task, condition, seed, args)
                all_rows.extend(rows)
                all_summaries.append(summary)
                pred.to_csv(out_dir / "predictions" / f"{task}_{condition}_seed{seed}.csv", index=False)
                pd.DataFrame(all_rows).to_csv(out_dir / "policy_results_long.csv", index=False)
                pd.DataFrame(all_summaries).to_csv(out_dir / "expert_pair_summary.csv", index=False)
                print(f"[done] task={task} condition={condition} seed={seed} sec={summary['elapsed_sec']:.1f}", flush=True)
    print(f"[complete] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
