from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments.fast_screen import _fit_xgb, _predict
from experiments.run_rejection_rebuild_gate1 import (
    CONDITIONS,
    TASK_CACHE,
    _condition_features,
    _load_task,
    _metric,
    _metric_gain,
)
from voila3d.metrics import regression_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", choices=sorted(TASK_CACHE), default=sorted(TASK_CACHE))
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=["same_seed_true3d", "seed_null_2d", "permuted_3d_placebo"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--split-seed", type=int, default=-1)
    p.add_argument("--budget", type=float, default=20.0)
    p.add_argument("--bootstrap-iters", type=int, default=5000)
    p.add_argument("--random-repeats", type=int, default=500)
    p.add_argument("--model-threads", type=int, default=16)
    p.add_argument("--xgb-device", choices=["cpu", "cuda", "auto"], default="cuda")
    p.add_argument("--out-dir", default="results/rejection_rebuild_task_metric_permission")
    return p.parse_args()


def _endpoint_gain(task_type: str, y: np.ndarray, ref: np.ndarray, alt: np.ndarray) -> float:
    _, ref_value = _metric(task_type, y, ref)
    _, alt_value = _metric(task_type, y, alt)
    return _metric_gain(task_type, ref_value, alt_value)


def _bootstrap_endpoint_lcb(
    task_type: str,
    y: np.ndarray,
    ref: np.ndarray,
    alt: np.ndarray,
    iters: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    boot = np.empty(iters, dtype=float)
    if task_type == "classification":
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        if not len(pos) or not len(neg):
            return np.nan, np.nan, np.nan
        for i in range(iters):
            idx = np.concatenate(
                [rng.choice(pos, size=len(pos), replace=True), rng.choice(neg, size=len(neg), replace=True)]
            )
            boot[i] = roc_auc_score(y[idx], alt[idx]) - roc_auc_score(y[idx], ref[idx])
    else:
        n = len(y)
        for i in range(iters):
            idx = rng.integers(0, n, size=n)
            ref_mae = regression_metrics(y[idx], ref[idx])["MAE"]
            alt_mae = regression_metrics(y[idx], alt[idx])["MAE"]
            boot[i] = ref_mae - alt_mae
    return float(np.mean(boot)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _random_budget_gain(
    task_type: str,
    y: np.ndarray,
    ref: np.ndarray,
    alt: np.ndarray,
    budget: float,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    n = len(y)
    k = min(n, max(0, int(np.ceil(n * budget / 100.0))))
    rng = np.random.default_rng(seed)
    gains = np.empty(repeats, dtype=float)
    for i in range(repeats):
        selected = rng.choice(n, size=k, replace=False) if k else np.empty(0, dtype=int)
        routed = ref.copy()
        routed[selected] = alt[selected]
        gains[i] = _endpoint_gain(task_type, y, ref, routed)
    return float(gains.mean()), float(gains.std(ddof=1))


def run_one(task: str, condition: str, seed: int, args: argparse.Namespace) -> dict:
    data = _load_task(task, args.split_seed)
    task_type = data["task_type"]
    x_ref, x_alt, alt_offset = _condition_features(data, condition, task)
    y = data["y"]
    train_idx = np.flatnonzero(data["train"])
    val_idx = np.flatnonzero(data["val"])
    test_idx = np.flatnonzero(data["test"])

    ref, _ = _fit_xgb(task_type, x_ref[train_idx], y[train_idx], seed, args.model_threads, args.xgb_device)
    alt, _ = _fit_xgb(task_type, x_alt[train_idx], y[train_idx], seed + alt_offset, args.model_threads, args.xgb_device)
    ref_val = _predict(ref, task_type, x_ref[val_idx])
    alt_val = _predict(alt, task_type, x_alt[val_idx])
    ref_test = _predict(ref, task_type, x_ref[test_idx])
    alt_test = _predict(alt, task_type, x_alt[test_idx])

    val_gain = _endpoint_gain(task_type, y[val_idx], ref_val, alt_val)
    boot_mean, lcb, ucb = _bootstrap_endpoint_lcb(
        task_type,
        y[val_idx],
        ref_val,
        alt_val,
        args.bootstrap_iters,
        seed + 44_000,
    )
    test_gain = _endpoint_gain(task_type, y[test_idx], ref_test, alt_test)
    random_gain, random_sd = _random_budget_gain(
        task_type,
        y[test_idx],
        ref_test,
        alt_test,
        args.budget,
        args.random_repeats,
        seed + 45_000,
    )
    point_permission = bool(val_gain > 0)
    lcb_permission = bool(np.isfinite(lcb) and lcb > 0)
    return {
        "task": task,
        "task_type": task_type,
        "condition": condition,
        "seed": seed,
        "split_seed": args.split_seed,
        "metric": _metric(task_type, y[test_idx], ref_test)[0],
        "validation_endpoint_gain": val_gain,
        "validation_bootstrap_mean": boot_mean,
        "validation_endpoint_lcb": lcb,
        "validation_endpoint_ucb": ucb,
        "point_permission": point_permission,
        "lcb_permission": lcb_permission,
        "full_alternative_test_gain": test_gain,
        "task_point_policy_gain": test_gain if point_permission else 0.0,
        "task_lcb_policy_gain": test_gain if lcb_permission else 0.0,
        "task_point_call_rate": 100.0 if point_permission else 0.0,
        "task_lcb_call_rate": 100.0 if lcb_permission else 0.0,
        "task_point_random_budget_gain": random_gain if point_permission else 0.0,
        "task_lcb_random_budget_gain": random_gain if lcb_permission else 0.0,
        "random_budget_gain": random_gain,
        "random_budget_gain_sd": random_sd,
        "budget": args.budget,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "argv": sys.argv,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "protocol": "train-fit expert pair; independent validation endpoint bootstrap; untouched test evaluation",
        "classification_statistic": "paired AUROC difference with class-stratified molecule bootstrap",
        "regression_statistic": "paired MAE reduction with molecule bootstrap",
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rows = []
    for task in args.tasks:
        for condition in args.conditions:
            for seed in args.seeds:
                print(f"[start] {task} {condition} seed={seed}", flush=True)
                rows.append(run_one(task, condition, seed, args))
                pd.DataFrame(rows).to_csv(out_dir / "task_metric_permission.csv", index=False)
                print(f"[done] {task} {condition} seed={seed}", flush=True)
    print(f"[complete] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
