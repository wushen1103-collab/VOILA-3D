from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.fast_screen import _feature_cache_paths, _fit_xgb, _predict
from experiments.run_validation_ensemble import (
    _collect_chemprop_predictions,
    _ensemble_selection,
    _eval_row,
    _fit_member_predictions,
    _summarize,
    _weighted_average,
    _write_split_csvs,
)
from voila3d.data import DATASETS, assign_splits, load_dataset
from voila3d.features import build_2d_feature_matrix
from voila3d.metrics import higher_is_better, primary_metric
from voila3d.utils import ensure_dir, now_iso, set_reproducible


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["BACE", "BBBP", "HIV"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--split", choices=["scaffold", "scaffold_balanced", "random"], default="scaffold_balanced")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--max-mols", type=int, default=12000)
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--out-dir", default="results/validation_ensemble_with_3d_5seed")
    p.add_argument("--feature2d-set", choices=["ecfp_desc", "rdkit2d_combo"], default="rdkit2d_combo")
    p.add_argument("--feature3d-set", choices=["usr", "rdkit_scalar", "rdkit_rich", "rdkit_full"], default="usr")
    p.add_argument("--feature3d-source-dir", default="results/fast_screen_xgb_k10_auc_ecfp_router")
    p.add_argument("--conformers", type=int, default=10)
    p.add_argument("--combo-jobs", type=int, default=8)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--xgb-device", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--skip-svm-above", type=int, default=15000)
    p.add_argument("--skip-krr-above", type=int, default=2500)
    p.add_argument("--chemprop-root", action="append", default=[])
    p.add_argument("--chemprop-bin", default=".chemprop_venv/bin/chemprop_predict")
    p.add_argument("--chemprop-gpu", type=int, default=0)
    p.add_argument("--cuda-visible-devices", default=None)
    p.add_argument("--chemprop-no-cuda", action="store_true")
    p.add_argument("--chemprop-batch-size", type=int, default=256)
    p.add_argument("--chemprop-workers", type=int, default=4)
    p.add_argument("--ensemble-rounds", type=int, default=50)
    p.add_argument("--overwrite-preds", action="store_true")
    return p.parse_args()


def _load_cached_3d_features(task: str, df: pd.DataFrame, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_paths = _feature_cache_paths(
        Path(args.feature3d_source_dir),
        task,
        args.split,
        args.max_mols,
        args.conformers,
        args.feature3d_set,
    )
    missing = [str(path) for path in cache_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing 3D feature cache. Run fast_screen first or pass --feature3d-source-dir. Missing: "
            + "; ".join(missing)
        )
    meta = pd.read_csv(cache_paths["meta"])
    expected_ids = df["mol_id"].astype(str).tolist()
    cached_ids = meta["mol_id"].astype(str).tolist()
    if expected_ids != cached_ids:
        raise ValueError(f"3D cache order mismatch for {task}; refusing to align by position.")
    x_ecfp = np.load(cache_paths["ecfp"])
    x_desc = np.load(cache_paths["desc"])
    x3d = np.load(cache_paths["x3d"])
    return x_ecfp, x_desc, x3d


def _fit_3d_member_predictions(task: str, task_type: str, df: pd.DataFrame, args: argparse.Namespace) -> dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]:
    x_ecfp, x_desc, x3d = _load_cached_3d_features(task, df, args)
    x2d = build_2d_feature_matrix(
        df["canonical_smiles"].tolist(),
        x_ecfp,
        x_desc,
        feature2d_set=args.feature2d_set,
        jobs=args.combo_jobs,
    )
    x3d_aug = np.hstack([x2d, x3d]).astype(np.float32)
    y = df["y"].to_numpy()
    split = df["split"].to_numpy()
    train = split == "train"
    val = split == "val"
    test = split == "test"

    by_seed: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for seed in args.seeds:
        set_reproducible(seed)
        if task_type == "classification" and len(np.unique(y[train])) < 2:
            continue
        m2d, _ = _fit_xgb(task_type, x2d[train], y[train], seed, args.threads, args.xgb_device)
        m3d, _ = _fit_xgb(task_type, x3d_aug[train], y[train], seed + 1000, args.threads, args.xgb_device)
        by_seed[seed] = {
            "voila_xgb_all_2d": (_predict(m2d, task_type, x2d[val]), _predict(m2d, task_type, x2d[test])),
            "voila_xgb_all_3d_aug": (_predict(m3d, task_type, x3d_aug[val]), _predict(m3d, task_type, x3d_aug[test])),
        }
    return by_seed


def run_task(task: str, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    df, spec, _ = load_dataset(task, args.data_dir, args.max_mols, seed=args.split_seed)
    df["split"] = assign_splits(df, args.split, seed=args.split_seed, task_type=spec.task_type)
    split_paths = _write_split_csvs(task, df, Path(args.out_dir))
    y = df["y"].to_numpy()
    split = df["split"].to_numpy()
    y_val = y[split == "val"]
    y_test = y[split == "test"]

    classical = _fit_member_predictions(task, spec.task_type, df, args)
    three_d = _fit_3d_member_predictions(task, spec.task_type, df, args)
    metric_rows: list[dict] = []
    selection_rows: list[dict] = []

    for seed in args.seeds:
        members = dict(classical.get(seed, {}))
        members.update(three_d.get(seed, {}))
        members.update(_collect_chemprop_predictions(task, seed, split_paths, args, Path(args.out_dir)))
        if not members:
            continue
        val_preds = {name: vt[0] for name, vt in members.items()}
        test_preds = {name: vt[1] for name, vt in members.items()}
        for name, pred in test_preds.items():
            metric_rows.append(_eval_row(task, spec.task_type, seed, name, y_test, pred))

        val_scores = {name: _eval_primary(spec.task_type, y_val, pred) for name, pred in val_preds.items()}
        best_single = sorted(
            val_scores,
            key=lambda name: val_scores[name],
            reverse=higher_is_better(primary_metric(spec.task_type)),
        )[0]
        metric_rows.append(_eval_row(task, spec.task_type, seed, "val_best_single", y_test, test_preds[best_single]))
        selection_rows.append(
            {
                "task": task,
                "seed": seed,
                "selector": "val_best_single",
                "members": best_single,
                "weights_json": json.dumps({best_single: 1.0}, sort_keys=True),
                "val_primary": float(val_scores[best_single]),
                "n_members_considered": len(members),
            }
        )

        weights, val_primary = _ensemble_selection(spec.task_type, y_val, val_preds, args.ensemble_rounds)
        ensemble_pred = _weighted_average(test_preds, weights)
        if spec.task_type == "classification":
            ensemble_pred = np.clip(ensemble_pred, 1e-6, 1 - 1e-6)
        metric_rows.append(_eval_row(task, spec.task_type, seed, "val_greedy_ensemble", y_test, ensemble_pred))
        selection_rows.append(
            {
                "task": task,
                "seed": seed,
                "selector": "val_greedy_ensemble",
                "members": ";".join(sorted(weights)),
                "weights_json": json.dumps(weights, sort_keys=True),
                "val_primary": float(val_primary),
                "n_members_considered": len(members),
            }
        )

    return metric_rows, selection_rows


def _eval_primary(task_type: str, y: np.ndarray, pred: np.ndarray) -> float:
    return float(_eval_row("_", task_type, 0, "_", y, pred)[primary_metric(task_type)])


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "started_at": now_iso(),
                "argv": sys.argv,
                "python": sys.version,
                "platform": platform.platform(),
                "tasks": args.tasks,
                "seeds": args.seeds,
                "split": args.split,
                "split_seed": args.split_seed,
                "max_mols": args.max_mols,
                "feature2d_set": args.feature2d_set,
                "feature3d_set": args.feature3d_set,
                "feature3d_source_dir": args.feature3d_source_dir,
                "chemprop_root": args.chemprop_root,
                "protocol_note": "Validation-only selection over classical, Chemprop, and cached VOILA/XGB 2D/3D members; no test labels used for selection.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    invalid = [task for task in args.tasks if task not in DATASETS]
    if invalid:
        raise ValueError(f"Unknown tasks: {invalid}")

    metric_rows: list[dict] = []
    selection_rows: list[dict] = []
    for task in args.tasks:
        print(f"[validation-ensemble-with-3d] {task}", flush=True)
        m_rows, s_rows = run_task(task, args)
        metric_rows.extend(m_rows)
        selection_rows.extend(s_rows)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "metrics_long.csv", index=False)
    _summarize(metrics).to_csv(out_dir / "method_summary.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(out_dir / "selection_by_seed.csv", index=False)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
