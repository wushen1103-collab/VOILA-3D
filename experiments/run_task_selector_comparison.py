from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.fast_screen import _fit_xgb, _predict
from experiments.run_rejection_rebuild_gate1 import (
    CONDITIONS,
    TASK_CACHE,
    _condition_features,
    _load_task,
    _metric,
    _metric_gain,
)
from experiments.run_task_metric_permission import _bootstrap_endpoint_lcb
from voila3d.data import assign_splits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", choices=sorted(TASK_CACHE), required=True)
    p.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=["same_seed_true3d", "seed_null_2d", "permuted_3d_placebo"],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--split-seed", type=int, required=True)
    p.add_argument(
        "--split-protocol",
        choices=["default", "scaffold_all"],
        default="default",
        help="default preserves the historical QM9 random split; scaffold_all applies scaffold-randomized splitting to every task.",
    )
    p.add_argument("--bootstrap-iters", type=int, default=5000)
    p.add_argument("--relative-thresholds", nargs="+", type=float, default=[0.005, 0.01])
    p.add_argument("--model-threads", type=int, default=16)
    p.add_argument("--xgb-device", choices=["cpu", "cuda", "auto"], default="cuda")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def _endpoint_values(
    task_type: str,
    y: np.ndarray,
    ref: np.ndarray,
    alt: np.ndarray,
) -> tuple[str, float, float, float]:
    metric, ref_value = _metric(task_type, y, ref)
    _, alt_value = _metric(task_type, y, alt)
    gain = _metric_gain(task_type, ref_value, alt_value)
    return metric, float(ref_value), float(alt_value), float(gain)


def _policy_row(
    common: dict,
    policy: str,
    permission: bool,
    test_gain: float,
) -> dict:
    oracle_gain = max(0.0, test_gain)
    policy_gain = test_gain if permission else 0.0
    return {
        **common,
        "policy": policy,
        "permission": bool(permission),
        "policy_test_gain": float(policy_gain),
        "oracle_task_gain": float(oracle_gain),
        "decision_correct": bool(permission == (test_gain > 0.0)),
        "false_activation": bool(permission and test_gain <= 0.0),
        "false_rejection": bool((not permission) and test_gain > 0.0),
        "negative_transfer": float(max(0.0, -policy_gain)),
        "missed_positive_gain": float(test_gain if (not permission and test_gain > 0.0) else 0.0),
        "regret": float(oracle_gain - policy_gain),
    }


def run_one(task: str, condition: str, seed: int, args: argparse.Namespace) -> list[dict]:
    data = _load_task(task, args.split_seed)
    if args.split_protocol == "scaffold_all" and task.startswith("QM9_"):
        split = assign_splits(
            data["meta"],
            "scaffold_randomized",
            seed=args.split_seed,
            frac_train=0.8,
            frac_val=0.1,
            task_type=data["task_type"],
        ).astype(str)
        data = {
            **data,
            "train": split == "train",
            "val": split == "val",
            "test": split == "test",
        }
    task_type = data["task_type"]
    x_ref, x_alt, alt_offset = _condition_features(data, condition, task)
    y = data["y"]
    train_idx = np.flatnonzero(data["train"])
    val_idx = np.flatnonzero(data["val"])
    test_idx = np.flatnonzero(data["test"])

    ref, _ = _fit_xgb(
        task_type,
        x_ref[train_idx],
        y[train_idx],
        seed,
        args.model_threads,
        args.xgb_device,
    )
    alt, _ = _fit_xgb(
        task_type,
        x_alt[train_idx],
        y[train_idx],
        seed + alt_offset,
        args.model_threads,
        args.xgb_device,
    )
    ref_val = _predict(ref, task_type, x_ref[val_idx])
    alt_val = _predict(alt, task_type, x_alt[val_idx])
    ref_test = _predict(ref, task_type, x_ref[test_idx])
    alt_test = _predict(alt, task_type, x_alt[test_idx])

    metric, val_ref, val_alt, val_gain = _endpoint_values(
        task_type, y[val_idx], ref_val, alt_val
    )
    _, test_ref, test_alt, test_gain = _endpoint_values(
        task_type, y[test_idx], ref_test, alt_test
    )
    boot_mean, lcb, ucb = _bootstrap_endpoint_lcb(
        task_type,
        y[val_idx],
        ref_val,
        alt_val,
        args.bootstrap_iters,
        seed + 44_000,
    )
    relative_val_gain = val_gain / max(abs(val_ref), 1e-12)

    common = {
        "task": task,
        "task_type": task_type,
        "condition": condition,
        "split_seed": args.split_seed,
        "seed": seed,
        "metric": metric,
        "validation_ref_value": val_ref,
        "validation_alt_value": val_alt,
        "validation_gain": val_gain,
        "validation_relative_gain": relative_val_gain,
        "validation_bootstrap_mean": boot_mean,
        "validation_lcb": lcb,
        "validation_ucb": ucb,
        "test_ref_value": test_ref,
        "test_alt_value": test_alt,
        "full_alternative_test_gain": test_gain,
    }

    rows = [
        _policy_row(common, "Always-2D", False, test_gain),
        _policy_row(common, "Always-2D+3D", True, test_gain),
        _policy_row(common, "TaskSelect", val_gain > 0.0, test_gain),
        _policy_row(common, "TaskMetric-LCB", bool(np.isfinite(lcb) and lcb > 0.0), test_gain),
    ]
    for threshold in args.relative_thresholds:
        label = f"ThresholdSelect-{threshold * 100:g}pct"
        rows.append(_policy_row(common, label, relative_val_gain > threshold, test_gain))
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "argv": sys.argv,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "protocol": "train-fit expert pair; independent validation selection; untouched test evaluation",
        "split_protocol": args.split_protocol,
        "threshold_definition": "validation endpoint gain divided by absolute reference validation metric",
        "thresholds": args.relative_thresholds,
        "lcb_definition": "2.5% percentile of paired endpoint-metric bootstrap on validation",
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    rows: list[dict] = []
    for task in args.tasks:
        for condition in args.conditions:
            for seed in args.seeds:
                print(
                    f"[start] task={task} condition={condition} split={args.split_seed} seed={seed}",
                    flush=True,
                )
                rows.extend(run_one(task, condition, seed, args))
                pd.DataFrame(rows).to_csv(out_dir / "selector_comparison.csv", index=False)
                print(
                    f"[done] task={task} condition={condition} split={args.split_seed} seed={seed}",
                    flush=True,
                )
    print(f"[complete] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
