from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.fast_screen import _fit_xgb, _make_router_features, _predict
from voila3d.features import build_2d_feature_matrix
from voila3d.metrics import classification_metrics, per_sample_loss, primary_metric, regression_metrics
from voila3d.routing import evaluate_routing_curves, train_voi_router, true_benefit
from voila3d.utils import ensure_dir, now_iso, set_reproducible


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["ESOL", "FreeSolv", "Lipophilicity", "BBBP", "BACE", "HIV"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--fractions", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0])
    p.add_argument("--budgets", nargs="+", type=float, default=[0, 5, 10, 20, 40, 100])
    p.add_argument("--split", default="scaffold_balanced")
    p.add_argument("--max-mols", type=int, default=12000)
    p.add_argument("--conformers", type=int, default=10)
    p.add_argument("--feature3d-set", choices=["usr", "rdkit_scalar", "rdkit_rich", "rdkit_full"], default="usr")
    p.add_argument("--feature2d-set", choices=["ecfp_desc", "rdkit2d_combo"], default="ecfp_desc")
    p.add_argument("--feature-cache-dir", default="results/fast_screen_xgb_k10_auc_ecfp_router/feature_cache")
    p.add_argument("--out-dir", default="results/label_efficiency_k10_usr_5seed")
    p.add_argument("--model-threads", type=int, default=16)
    p.add_argument("--xgb-device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--router-feature-set", choices=["desc", "ecfp_desc"], default="ecfp_desc")
    p.add_argument("--classification-benefit", choices=["logloss", "margin", "auc_contrib"], default="auc_contrib")
    p.add_argument("--combo-jobs", type=int, default=16)
    return p.parse_args()


def _cache_key(task: str, split: str, max_mols: int | None, conformers: int, feature3d_set: str) -> str:
    suffix = "" if feature3d_set == "usr" else f"_{feature3d_set}"
    return f"{task}_{split}_max{max_mols or 'all'}_k{conformers}{suffix}"


def _load_cached_features(args: argparse.Namespace, task: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    cache = Path(args.feature_cache_dir)
    key = _cache_key(task, args.split, args.max_mols, args.conformers, args.feature3d_set)
    paths = {
        "meta": cache / f"{key}_meta.csv",
        "ecfp": cache / f"{key}_ecfp.npy",
        "desc": cache / f"{key}_desc.npy",
        "x3d": cache / f"{key}_x3d.npy",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing cached feature files for {task}: {missing}")
    meta = pd.read_csv(paths["meta"])
    return meta, np.load(paths["ecfp"]), np.load(paths["desc"]), np.load(paths["x3d"])


def _task_type(meta: pd.DataFrame) -> str:
    y = meta["y"].to_numpy()
    uniq = sorted(set(float(v) for v in np.unique(y)))
    if len(uniq) <= 2 and set(uniq).issubset({0.0, 1.0}):
        return "classification"
    return "regression"


def _sample_train_indices(y: np.ndarray, train_idx: np.ndarray, task_type: str, frac: float, seed: int) -> np.ndarray:
    if frac >= 0.999:
        return train_idx.copy()
    rng = np.random.default_rng(seed + int(round(frac * 10000)))
    n_take = max(2, int(round(len(train_idx) * frac)))
    if task_type == "classification":
        parts = []
        for cls in [0, 1]:
            cls_idx = train_idx[y[train_idx].astype(int) == cls]
            if len(cls_idx) == 0:
                continue
            take = max(1, int(round(len(cls_idx) * frac)))
            take = min(take, len(cls_idx))
            parts.append(rng.choice(cls_idx, size=take, replace=False))
        out = np.concatenate(parts) if parts else rng.choice(train_idx, size=n_take, replace=False)
        return np.sort(out)
    n_take = min(n_take, len(train_idx))
    return np.sort(rng.choice(train_idx, size=n_take, replace=False))


def _evaluate_all(task_type: str, y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if task_type == "classification":
        return classification_metrics(y, pred)
    return regression_metrics(y, pred)


def run_task(task: str, args: argparse.Namespace) -> tuple[list[dict], list[pd.DataFrame]]:
    meta, x_ecfp, x_desc, x3d = _load_cached_features(args, task)
    x2d = build_2d_feature_matrix(
        meta["canonical_smiles"].tolist(),
        x_ecfp,
        x_desc,
        feature2d_set=args.feature2d_set,
        jobs=args.combo_jobs,
    )
    x3d_aug = np.hstack([x2d, x3d]).astype(np.float32)
    y = meta["y"].to_numpy()
    split = meta["split"].to_numpy()
    task_type = _task_type(meta)
    train_idx = np.flatnonzero(split == "train")
    val_idx = np.flatnonzero(split == "val")
    test_idx = np.flatnonzero(split == "test")
    metric_rows: list[dict] = []
    route_frames: list[pd.DataFrame] = []

    for seed in args.seeds:
        set_reproducible(seed)
        for frac in args.fractions:
            fit_idx = _sample_train_indices(y, train_idx, task_type, float(frac), seed)
            if task_type == "classification" and len(np.unique(y[fit_idx])) < 2:
                continue
            m2d, device2d = _fit_xgb(task_type, x2d[fit_idx], y[fit_idx], seed, args.model_threads, args.xgb_device)
            m3d, device3d = _fit_xgb(task_type, x3d_aug[fit_idx], y[fit_idx], seed + 1000, args.model_threads, args.xgb_device)
            pred2d_val = _predict(m2d, task_type, x2d[val_idx])
            pred3d_val = _predict(m3d, task_type, x3d_aug[val_idx])
            pred2d_test = _predict(m2d, task_type, x2d[test_idx])
            pred3d_test = _predict(m3d, task_type, x3d_aug[test_idx])

            for method, pred in [("all_2d", pred2d_test), ("all_3d_aug", pred3d_test)]:
                row = {
                    "scenario": "label_efficiency_cold_start",
                    "task": task,
                    "task_type": task_type,
                    "seed": seed,
                    "train_fraction": float(frac),
                    "method": method,
                    "primary_metric": primary_metric(task_type),
                    "n_fit": int(len(fit_idx)),
                    "n_train_total": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "n_test": int(len(test_idx)),
                    "xgb_device_2d": device2d,
                    "xgb_device_3d": device3d,
                }
                row.update(_evaluate_all(task_type, y[test_idx], pred))
                metric_rows.append(row)

            benefit_val = true_benefit(
                task_type,
                y[val_idx],
                pred2d_val,
                pred3d_val,
                classification_mode=args.classification_benefit,
            )
            router_x_val = _make_router_features(x_ecfp[val_idx], x_desc[val_idx], pred2d_val, task_type, args.router_feature_set)
            router = train_voi_router(router_x_val, benefit_val, seed=seed)
            router_scores = router.predict(
                _make_router_features(x_ecfp[test_idx], x_desc[test_idx], pred2d_test, task_type, args.router_feature_set)
            )
            curves = evaluate_routing_curves(
                task,
                task_type,
                y[test_idx],
                pred2d_test,
                pred3d_test,
                {"voi_router": router_scores, "oracle": true_benefit(task_type, y[test_idx], pred2d_test, pred3d_test, classification_mode=args.classification_benefit)},
                args.budgets,
                seed,
            )
            curves.insert(0, "scenario", "label_efficiency_cold_start")
            curves.insert(4, "train_fraction", float(frac))
            curves.insert(5, "n_fit", int(len(fit_idx)))
            route_frames.append(curves)

    return metric_rows, route_frames


def _mean_std(vals: pd.Series) -> str:
    vals = pd.to_numeric(vals, errors="coerce")
    return f"{vals.mean():.6f} +/- {vals.std(ddof=0):.6f}"


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, task_type, frac, method), g in metrics.groupby(["task", "task_type", "train_fraction", "method"]):
        pmet = primary_metric(task_type)
        row = {
            "scenario": "label_efficiency_cold_start",
            "task": task,
            "task_type": task_type,
            "train_fraction": frac,
            "method": method,
            "primary_metric": pmet,
            "primary_mean": float(g[pmet].mean()),
            "primary_std": float(g[pmet].std(ddof=0)),
            "primary_mean_std": _mean_std(g[pmet]),
            "n_seeds": int(g["seed"].nunique()),
            "n_fit_mean": float(g["n_fit"].mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["task", "train_fraction", "method"])


def summarize_routes(routes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, task_type, frac, router, budget), g in routes.groupby(["task", "task_type", "train_fraction", "router", "budget"]):
        pmet = primary_metric(task_type)
        rows.append(
            {
                "scenario": "label_efficiency_cold_start",
                "task": task,
                "task_type": task_type,
                "train_fraction": frac,
                "router": router,
                "budget": budget,
                "primary_metric": pmet,
                "primary_mean": float(g["primary_value"].mean()),
                "primary_std": float(g["primary_value"].std(ddof=0)),
                "primary_mean_std": _mean_std(g["primary_value"]),
                "call_rate_mean": float(g["call_rate"].mean()),
                "n_seeds": int(g["seed"].nunique()),
                "n_fit_mean": float(g["n_fit"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["task", "train_fraction", "router", "budget"])


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    metadata = {
        "started_at": now_iso(),
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tasks": args.tasks,
        "seeds": args.seeds,
        "fractions": args.fractions,
        "feature_cache_dir": args.feature_cache_dir,
        "feature2d_set": args.feature2d_set,
        "feature3d_set": args.feature3d_set,
        "classification_benefit": args.classification_benefit,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metric_rows: list[dict] = []
    route_frames: list[pd.DataFrame] = []
    for task in args.tasks:
        print(f"[label-eff] {task}", flush=True)
        m, r = run_task(task, args)
        metric_rows.extend(m)
        route_frames.extend(r)

    metrics = pd.DataFrame(metric_rows)
    routes = pd.concat(route_frames, ignore_index=True) if route_frames else pd.DataFrame()
    metrics.to_csv(out_dir / "metrics_long.csv", index=False)
    if not routes.empty:
        routes.to_csv(out_dir / "routing_curves.csv", index=False)
    summarize_metrics(metrics).to_csv(out_dir / "method_summary.csv", index=False)
    if not routes.empty:
        summarize_routes(routes).to_csv(out_dir / "routing_summary.csv", index=False)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
