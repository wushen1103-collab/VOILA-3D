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

from experiments.fast_screen import _feature_cache_paths, _fit_xgb, _make_router_features, _predict
from voila3d.data import DATASETS, assign_splits, load_dataset
from voila3d.features import build_2d_feature_matrix
from voila3d.metrics import classification_metrics, primary_metric, regression_metrics
from voila3d.routing import evaluate_routing_curves, train_voi_router, true_benefit, uncertainty_scores
from voila3d.utils import ensure_dir, now_iso, set_reproducible


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["ESOL", "FreeSolv", "Lipophilicity", "BBBP", "BACE", "HIV"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--noise-levels", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5, 1.0])
    p.add_argument("--budgets", nargs="+", type=float, default=[0, 5, 10, 20, 40, 60, 80, 100])
    p.add_argument("--split", choices=["scaffold", "scaffold_balanced", "random"], default="scaffold_balanced")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--max-mols", type=int, default=12000)
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--out-dir", default="results/if_noisy3d_stress_usr_5seed")
    p.add_argument("--feature2d-set", choices=["ecfp_desc", "rdkit2d_combo"], default="rdkit2d_combo")
    p.add_argument("--feature3d-set", choices=["usr", "rdkit_scalar", "rdkit_rich", "rdkit_full"], default="usr")
    p.add_argument("--feature3d-source-dir", default="results/fast_screen_xgb_k10_auc_oof_ecfp_router_5seed_cls")
    p.add_argument("--conformers", type=int, default=10)
    p.add_argument("--combo-jobs", type=int, default=10)
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--xgb-device", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--router-feature-set", choices=["desc", "ecfp_desc"], default="ecfp_desc")
    p.add_argument("--classification-benefit", choices=["logloss", "margin", "auc_contrib"], default="auc_contrib")
    return p.parse_args()


def _load_cached_features(task: str, df: pd.DataFrame, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = _feature_cache_paths(
        Path(args.feature3d_source_dir),
        task,
        args.split,
        args.max_mols,
        args.conformers,
        args.feature3d_set,
    )
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing cached 3D feature files: " + "; ".join(missing))
    meta = pd.read_csv(paths["meta"])
    if meta["mol_id"].astype(str).tolist() != df["mol_id"].astype(str).tolist():
        raise ValueError(f"3D cache order mismatch for {task}")
    return np.load(paths["ecfp"]), np.load(paths["desc"]), np.load(paths["x3d"])


def _evaluate(task_type: str, y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if task_type == "classification":
        return classification_metrics(y_true, np.clip(pred, 1e-6, 1 - 1e-6))
    return regression_metrics(y_true, pred)


def _metric_row(
    task: str,
    task_type: str,
    seed: int,
    noise_level: float,
    method: str,
    y_true: np.ndarray,
    pred: np.ndarray,
    call_rate: float,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "scenario": "noisy_3d_feature_stress",
        "task": task,
        "task_type": task_type,
        "seed": seed,
        "noise_level": float(noise_level),
        "method": method,
        "primary_metric": primary_metric(task_type),
        "call_rate": float(call_rate),
    }
    row.update(_evaluate(task_type, y_true, pred))
    return row


def _select_top(scores: np.ndarray, budget: float) -> np.ndarray:
    n = len(scores)
    k = int(round(n * budget / 100.0))
    out = np.zeros(n, dtype=bool)
    if k <= 0:
        return out
    if k >= n:
        out[:] = True
        return out
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    out[idx] = True
    return out


def _positive_top(scores: np.ndarray, budget: float, threshold: float = 0.0) -> np.ndarray:
    selected = _select_top(scores, budget)
    selected &= np.asarray(scores, dtype=float) > threshold
    return selected


def _routed(pred2d: np.ndarray, pred3d: np.ndarray, selected: np.ndarray) -> np.ndarray:
    pred = pred2d.copy()
    pred[selected] = pred3d[selected]
    return pred


def _summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, task_type, noise, method), g in metrics.groupby(["task", "task_type", "noise_level", "method"]):
        pmet = primary_metric(task_type)
        vals = pd.to_numeric(g[pmet], errors="coerce")
        rows.append(
            {
                "scenario": "noisy_3d_feature_stress",
                "task": task,
                "task_type": task_type,
                "noise_level": float(noise),
                "method": method,
                "primary_metric": pmet,
                "primary_mean": float(vals.mean()),
                "primary_std": float(vals.std(ddof=0)),
                "primary_mean_std": f"{vals.mean():.6f} +/- {vals.std(ddof=0):.6f}",
                "call_rate_mean": float(g["call_rate"].mean()),
                "n_seeds": int(g["seed"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["task", "noise_level", "method"])


def _summarize_routes(routes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, task_type, noise, router, budget), g in routes.groupby(["task", "task_type", "noise_level", "router", "budget"]):
        pmet = primary_metric(task_type)
        vals = pd.to_numeric(g["primary_value"], errors="coerce")
        rows.append(
            {
                "scenario": "noisy_3d_feature_stress",
                "task": task,
                "task_type": task_type,
                "noise_level": float(noise),
                "router": router,
                "budget": float(budget),
                "primary_metric": pmet,
                "primary_mean": float(vals.mean()),
                "primary_std": float(vals.std(ddof=0)),
                "primary_mean_std": f"{vals.mean():.6f} +/- {vals.std(ddof=0):.6f}",
                "call_rate_mean": float(g["call_rate"].mean()),
                "n_seeds": int(g["seed"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["task", "noise_level", "router", "budget"])


def run_task(task: str, args: argparse.Namespace) -> tuple[list[dict], list[pd.DataFrame], list[pd.DataFrame]]:
    df, spec, _ = load_dataset(task, args.data_dir, args.max_mols, seed=args.split_seed)
    df["split"] = assign_splits(df, args.split, seed=args.split_seed, task_type=spec.task_type)
    x_ecfp, x_desc, x3d = _load_cached_features(task, df, args)
    x2d = build_2d_feature_matrix(
        df["canonical_smiles"].tolist(),
        x_ecfp,
        x_desc,
        feature2d_set=args.feature2d_set,
        jobs=args.combo_jobs,
    )
    y = df["y"].to_numpy()
    split = df["split"].to_numpy()
    train = split == "train"
    val = split == "val"
    test = split == "test"
    scale = np.nanstd(x3d[train], axis=0).astype(np.float32)
    scale[~np.isfinite(scale)] = 0.0
    scale = np.maximum(scale, 1e-6)
    metric_rows: list[dict] = []
    route_frames: list[pd.DataFrame] = []
    pred_frames: list[pd.DataFrame] = []

    for seed in args.seeds:
        set_reproducible(seed)
        if spec.task_type == "classification" and len(np.unique(y[train])) < 2:
            continue
        m2d, _ = _fit_xgb(spec.task_type, x2d[train], y[train], seed, args.threads, args.xgb_device)
        pred2d_val = _predict(m2d, spec.task_type, x2d[val])
        pred2d_test = _predict(m2d, spec.task_type, x2d[test])

        for noise_level in args.noise_levels:
            metric_rows.append(_metric_row(task, spec.task_type, seed, noise_level, "all_2d", y[test], pred2d_test, 0.0))
            rng = np.random.default_rng(seed * 1000003 + int(round(noise_level * 10000)))
            def noisy(block: np.ndarray) -> np.ndarray:
                if noise_level <= 0:
                    return block.astype(np.float32, copy=True)
                return (block + rng.normal(0.0, noise_level, size=block.shape).astype(np.float32) * scale).astype(np.float32)

            x3d_train = noisy(x3d[train])
            x3d_val = noisy(x3d[val])
            x3d_test = noisy(x3d[test])
            x3d_aug_train = np.hstack([x2d[train], x3d_train]).astype(np.float32)
            x3d_aug_val = np.hstack([x2d[val], x3d_val]).astype(np.float32)
            x3d_aug_test = np.hstack([x2d[test], x3d_test]).astype(np.float32)
            m3d, _ = _fit_xgb(spec.task_type, x3d_aug_train, y[train], seed + 1000 + int(round(noise_level * 1000)), args.threads, args.xgb_device)
            pred3d_val = _predict(m3d, spec.task_type, x3d_aug_val)
            pred3d_test = _predict(m3d, spec.task_type, x3d_aug_test)
            benefit_val = true_benefit(
                spec.task_type,
                y[val],
                pred2d_val,
                pred3d_val,
                classification_mode=args.classification_benefit,
            )
            router = train_voi_router(
                _make_router_features(x_ecfp[val], x_desc[val], pred2d_val, spec.task_type, args.router_feature_set),
                benefit_val,
                seed=seed,
            )
            scores = router.predict(_make_router_features(x_ecfp[test], x_desc[test], pred2d_test, spec.task_type, args.router_feature_set))
            random_scores = np.random.default_rng(seed).normal(size=test.sum())
            unc = uncertainty_scores(spec.task_type, pred2d_test)
            flex = x_desc[test, 2] + 0.05 * x_desc[test, 1]
            oracle = true_benefit(
                spec.task_type,
                y[test],
                pred2d_test,
                pred3d_test,
                classification_mode=args.classification_benefit,
            )
            curves = evaluate_routing_curves(
                task,
                spec.task_type,
                y[test],
                pred2d_test,
                pred3d_test,
                {
                    "voi_router_fixed_budget": scores,
                    "random": random_scores,
                    "uncertainty": unc,
                    "flexibility": flex,
                    "oracle": oracle,
                },
                args.budgets,
                seed,
            )
            curves.insert(0, "scenario", "noisy_3d_feature_stress")
            curves.insert(4, "noise_level", float(noise_level))
            positive_rows = []
            reliability_rows = []
            score_penalty = float(noise_level * np.nanstd(scores))
            for budget in args.budgets:
                pos_sel = _positive_top(scores, budget, 0.0)
                rel_sel = _positive_top(scores - score_penalty, budget, 0.0)
                for name, selected in [
                    ("voi_positive_threshold", pos_sel),
                    ("voi_reliability_penalized_positive", rel_sel),
                ]:
                    pred = _routed(pred2d_test, pred3d_test, selected)
                    mets = _evaluate(spec.task_type, y[test], pred)
                    row = {
                        "scenario": "noisy_3d_feature_stress",
                        "task": task,
                        "task_type": spec.task_type,
                        "seed": seed,
                        "noise_level": float(noise_level),
                        "router": name,
                        "budget": float(budget),
                        "call_rate": float(selected.mean() * 100.0),
                        "primary_metric": primary_metric(spec.task_type),
                        "primary_value": float(mets[primary_metric(spec.task_type)]),
                    }
                    row.update({f"metric_{k}": v for k, v in mets.items()})
                    positive_rows.append(row)
                    if name == "voi_reliability_penalized_positive":
                        reliability_rows.append(row)
            if positive_rows:
                curves = pd.concat([curves, pd.DataFrame(positive_rows)], ignore_index=True)
            route_frames.append(curves)
            metric_rows.append(_metric_row(task, spec.task_type, seed, noise_level, "all_3d_noisy_aug", y[test], pred3d_test, 100.0))
            sel20 = _select_top(scores, 20.0)
            metric_rows.append(
                _metric_row(task, spec.task_type, seed, noise_level, "voi_router_fixed_budget_20", y[test], _routed(pred2d_test, pred3d_test, sel20), float(sel20.mean() * 100.0))
            )
            pos20 = _positive_top(scores, 20.0, 0.0)
            metric_rows.append(
                _metric_row(task, spec.task_type, seed, noise_level, "voi_positive_threshold_20", y[test], _routed(pred2d_test, pred3d_test, pos20), float(pos20.mean() * 100.0))
            )
            rel20 = _positive_top(scores - score_penalty, 20.0, 0.0)
            metric_rows.append(
                _metric_row(task, spec.task_type, seed, noise_level, "voi_reliability_penalized_positive_20", y[test], _routed(pred2d_test, pred3d_test, rel20), float(rel20.mean() * 100.0))
            )
            pred_frames.append(
                pd.DataFrame(
                    {
                        "task": task,
                        "seed": seed,
                        "noise_level": float(noise_level),
                        "mol_id": df.loc[test, "mol_id"].to_numpy(),
                        "y": y[test],
                        "pred2d": pred2d_test,
                        "pred3d_noisy": pred3d_test,
                        "voi_score": scores,
                        "voi_score_reliability_penalized": scores - score_penalty,
                        "uncertainty_score": unc,
                        "oracle_benefit": oracle,
                    }
                )
            )

    return metric_rows, route_frames, pred_frames


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    ensure_dir(out_dir / "predictions")
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "started_at": now_iso(),
                "argv": sys.argv,
                "python": sys.version,
                "platform": platform.platform(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "tasks": args.tasks,
                "seeds": args.seeds,
                "noise_levels": args.noise_levels,
                "protocol_note": "Feature-level 3D corruption stress test: 3D descriptors are perturbed before fitting the 3D expert and validation utility router; test labels are never used for routing.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    invalid = [task for task in args.tasks if task not in DATASETS]
    if invalid:
        raise ValueError(f"Unknown tasks: {invalid}")
    metric_rows: list[dict] = []
    route_frames: list[pd.DataFrame] = []
    for task in args.tasks:
        print(f"[noisy3d] {task}", flush=True)
        m_rows, r_frames, p_frames = run_task(task, args)
        metric_rows.extend(m_rows)
        route_frames.extend(r_frames)
        if p_frames:
            pd.concat(p_frames, ignore_index=True).to_csv(out_dir / "predictions" / f"{task}_noisy3d_predictions.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    routes = pd.concat(route_frames, ignore_index=True) if route_frames else pd.DataFrame()
    metrics.to_csv(out_dir / "metrics_long.csv", index=False)
    _summarize_metrics(metrics).to_csv(out_dir / "method_summary.csv", index=False)
    if not routes.empty:
        routes.to_csv(out_dir / "routing_curves.csv", index=False)
        _summarize_routes(routes).to_csv(out_dir / "routing_summary.csv", index=False)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
