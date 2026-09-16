from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.fast_screen import _feature_cache_paths, _fit_xgb, _make_router_features, _predict
from voila3d.data import DATASETS, assign_splits, load_dataset
from voila3d.features import build_2d_feature_matrix
from voila3d.metrics import classification_metrics, higher_is_better, primary_metric, regression_metrics
from voila3d.routing import train_voi_router, true_benefit, uncertainty_scores
from voila3d.utils import ensure_dir, now_iso, set_reproducible


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["ESOL", "FreeSolv", "Lipophilicity", "BBBP", "BACE", "HIV"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--split", choices=["scaffold", "scaffold_balanced", "random"], default="scaffold_balanced")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--max-mols", type=int, default=12000)
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--out-dir", default="results/if_fusion_baselines_usr_5seed")
    p.add_argument("--feature2d-set", choices=["ecfp_desc", "rdkit2d_combo"], default="rdkit2d_combo")
    p.add_argument("--feature3d-set", choices=["usr", "rdkit_scalar", "rdkit_rich", "rdkit_full"], default="usr")
    p.add_argument("--feature3d-source-dir", default="results/fast_screen_xgb_k10_auc_ecfp_router")
    p.add_argument("--conformers", type=int, default=10)
    p.add_argument("--combo-jobs", type=int, default=12)
    p.add_argument("--threads", type=int, default=12)
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
    expected_ids = df["mol_id"].astype(str).tolist()
    cached_ids = meta["mol_id"].astype(str).tolist()
    if expected_ids != cached_ids:
        raise ValueError(f"3D cache order mismatch for {task}; refusing position-based alignment.")
    return np.load(paths["ecfp"]), np.load(paths["desc"]), np.load(paths["x3d"])


def _evaluate(task_type: str, y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if task_type == "classification":
        return classification_metrics(y_true, np.clip(pred, 1e-6, 1 - 1e-6))
    return regression_metrics(y_true, pred)


def _primary_value(task_type: str, y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(_evaluate(task_type, y_true, pred)[primary_metric(task_type)])


def _best_weight(task_type: str, y_val: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> tuple[float, float]:
    metric = primary_metric(task_type)
    hib = higher_is_better(metric)
    best_w = 0.0
    best_score = -np.inf if hib else np.inf
    for w in np.linspace(0.0, 1.0, 21):
        pred = (1.0 - w) * pred_a + w * pred_b
        if task_type == "classification":
            pred = np.clip(pred, 1e-6, 1 - 1e-6)
        score = _primary_value(task_type, y_val, pred)
        if (hib and score > best_score) or ((not hib) and score < best_score):
            best_score = score
            best_w = float(w)
    return best_w, float(best_score)


def _fit_meta_model(task_type: str, x_val: np.ndarray, y_val: np.ndarray, seed: int):
    if task_type == "classification":
        if len(np.unique(y_val)) < 2:
            return None
        model = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed,
        )
    else:
        model = RidgeCV(alphas=np.logspace(-4, 4, 17))
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)
    pipe.fit(x_val, y_val)
    return pipe


def _meta_predict(model, task_type: str, x: np.ndarray) -> np.ndarray:
    if model is None:
        return np.full(x.shape[0], np.nan, dtype=float)
    if task_type == "classification":
        return model.predict_proba(x)[:, 1].astype(float)
    return model.predict(x).astype(float)


def _row(
    task: str,
    task_type: str,
    seed: int,
    method: str,
    y_test: np.ndarray,
    pred: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    call_rate: float | None = None,
    calibration_note: str = "",
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "task": task,
        "task_type": task_type,
        "seed": seed,
        "method": method,
        "primary_metric": primary_metric(task_type),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "call_rate": float(call_rate) if call_rate is not None else np.nan,
        "calibration_note": calibration_note,
    }
    row.update(_evaluate(task_type, y_test, pred))
    return row


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["AUROC", "AUPRC", "Accuracy", "LogLoss", "MAE", "RMSE", "R2"]
    rows = []
    for (task, task_type, method), g in metrics.groupby(["task", "task_type", "method"], sort=True):
        pmet = primary_metric(task_type)
        row = {
            "task": task,
            "task_type": task_type,
            "method": method,
            "primary_metric": pmet,
            "primary_mean": float(g[pmet].mean()),
            "primary_std": float(g[pmet].std(ddof=0)),
            "primary_mean_std": f"{g[pmet].mean():.6f} +/- {g[pmet].std(ddof=0):.6f}",
            "n_seeds": int(g["seed"].nunique()),
            "call_rate_mean": float(g["call_rate"].mean()) if "call_rate" in g else np.nan,
            "n_train": int(g["n_train"].iloc[0]),
            "n_val": int(g["n_val"].iloc[0]),
            "n_test": int(g["n_test"].iloc[0]),
        }
        for col in metric_cols:
            if col in g:
                vals = pd.to_numeric(g[col], errors="coerce")
                if vals.notna().any():
                    row[f"{col}_mean_std"] = f"{vals.mean():.6f} +/- {vals.std(ddof=0):.6f}"
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["task", "method"])


def run_task(task: str, args: argparse.Namespace) -> tuple[list[dict], list[pd.DataFrame], list[dict]]:
    df, spec, _ = load_dataset(task, args.data_dir, args.max_mols, seed=args.split_seed)
    df["split"] = assign_splits(df, args.split, seed=args.split_seed, task_type=spec.task_type)
    x_ecfp, x_desc, x3d = _load_cached_features(task, df, args)
    feature_start = time.perf_counter()
    x2d = build_2d_feature_matrix(
        df["canonical_smiles"].tolist(),
        x_ecfp,
        x_desc,
        feature2d_set=args.feature2d_set,
        jobs=args.combo_jobs,
    )
    feature_build_sec = time.perf_counter() - feature_start
    x3d_aug = np.hstack([x2d, x3d]).astype(np.float32)
    x3d_aug_drop = np.hstack([x2d, np.zeros_like(x3d)]).astype(np.float32)
    y = df["y"].to_numpy()
    split = df["split"].to_numpy()
    train = split == "train"
    val = split == "val"
    test = split == "test"
    y_val = y[val]
    y_test = y[test]
    n_train, n_val, n_test = int(train.sum()), int(val.sum()), int(test.sum())
    metric_rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    calibration_rows: list[dict] = []

    for seed in args.seeds:
        set_reproducible(seed)
        seed_start = time.perf_counter()
        if spec.task_type == "classification" and len(np.unique(y[train])) < 2:
            continue

        m2d, dev2d = _fit_xgb(spec.task_type, x2d[train], y[train], seed, args.threads, args.xgb_device)
        m3d_only, dev3d_only = _fit_xgb(spec.task_type, x3d[train], y[train], seed + 1000, args.threads, args.xgb_device)
        mearly, dev_early = _fit_xgb(spec.task_type, x3d_aug[train], y[train], seed + 2000, args.threads, args.xgb_device)

        dropout_x = np.vstack([x3d_aug[train], x3d_aug_drop[train]])
        dropout_y = np.concatenate([y[train], y[train]])
        mdrop, dev_drop = _fit_xgb(spec.task_type, dropout_x, dropout_y, seed + 3000, args.threads, args.xgb_device)

        pred2d_val = _predict(m2d, spec.task_type, x2d[val])
        pred2d_test = _predict(m2d, spec.task_type, x2d[test])
        pred3d_val = _predict(m3d_only, spec.task_type, x3d[val])
        pred3d_test = _predict(m3d_only, spec.task_type, x3d[test])
        pred_early_val = _predict(mearly, spec.task_type, x3d_aug[val])
        pred_early_test = _predict(mearly, spec.task_type, x3d_aug[test])
        pred_drop_test = _predict(mdrop, spec.task_type, x3d_aug[test])

        late_mean_val = 0.5 * pred2d_val + 0.5 * pred3d_val
        late_mean_test = 0.5 * pred2d_test + 0.5 * pred3d_test
        late_w, late_val_primary = _best_weight(spec.task_type, y_val, pred2d_val, pred3d_val)
        late_weighted_test = (1.0 - late_w) * pred2d_test + late_w * pred3d_test
        early_w, early_val_primary = _best_weight(spec.task_type, y_val, pred2d_val, pred_early_val)
        early_weighted_test = (1.0 - early_w) * pred2d_test + early_w * pred_early_test

        benefit_val = true_benefit(
            spec.task_type,
            y_val,
            pred2d_val,
            pred_early_val,
            classification_mode=args.classification_benefit,
        )
        gate_x_val = _make_router_features(x_ecfp[val], x_desc[val], pred2d_val, spec.task_type, args.router_feature_set)
        gate_x_test = _make_router_features(x_ecfp[test], x_desc[test], pred2d_test, spec.task_type, args.router_feature_set)
        gate = train_voi_router(gate_x_val, benefit_val, seed=seed)
        gate_score = gate.predict(gate_x_test)
        gate_selected = gate_score > 0.0
        gated_pred = pred2d_test.copy()
        gated_pred[gate_selected] = pred_early_test[gate_selected]

        moe_val = np.column_stack(
            [
                pred2d_val,
                pred3d_val,
                pred_early_val,
                late_mean_val,
                uncertainty_scores(spec.task_type, pred2d_val),
                np.abs(pred_early_val - pred2d_val),
                x_desc[val],
            ]
        )
        moe_test = np.column_stack(
            [
                pred2d_test,
                pred3d_test,
                pred_early_test,
                late_mean_test,
                uncertainty_scores(spec.task_type, pred2d_test),
                np.abs(pred_early_test - pred2d_test),
                x_desc[test],
            ]
        )
        meta_model = _fit_meta_model(spec.task_type, moe_val, y_val, seed)
        moe_pred = _meta_predict(meta_model, spec.task_type, moe_test)
        if not np.isfinite(moe_pred).all():
            moe_pred = early_weighted_test.copy()

        if spec.task_type == "classification":
            for arr in [
                late_mean_test,
                late_weighted_test,
                early_weighted_test,
                gated_pred,
                moe_pred,
            ]:
                arr[:] = np.clip(arr, 1e-6, 1 - 1e-6)

        method_preds = {
            "2d_only": pred2d_test,
            "3d_only": pred3d_test,
            "early_concat_fusion": pred_early_test,
            "late_mean_fusion": late_mean_test,
            "late_val_weighted_2d_3d": late_weighted_test,
            "late_val_weighted_2d_early": early_weighted_test,
            "gated_val_utility_fusion": gated_pred,
            "moe_stacking_val_calibrated": moe_pred,
            "modality_dropout_early_fusion": pred_drop_test,
        }
        for method, pred in method_preds.items():
            call_rate = float(gate_selected.mean() * 100.0) if method == "gated_val_utility_fusion" else np.nan
            note = "validation utility gate; no test labels used" if method == "gated_val_utility_fusion" else ""
            metric_rows.append(_row(task, spec.task_type, seed, method, y_test, pred, n_train, n_val, n_test, call_rate, note))

        calibration_rows.append(
            {
                "task": task,
                "task_type": spec.task_type,
                "seed": seed,
                "late_weight_3d_only": float(late_w),
                "late_val_primary": float(late_val_primary),
                "late_weight_early_fusion": float(early_w),
                "early_val_primary": float(early_val_primary),
                "gated_call_rate": float(gate_selected.mean() * 100.0),
                "feature_build_sec": float(feature_build_sec),
                "seed_total_sec": float(time.perf_counter() - seed_start),
                "xgb_device_2d": dev2d,
                "xgb_device_3d_only": dev3d_only,
                "xgb_device_early": dev_early,
                "xgb_device_dropout": dev_drop,
            }
        )
        pred_frames.append(
            pd.DataFrame(
                {
                    "task": task,
                    "seed": seed,
                    "mol_id": df.loc[test, "mol_id"].to_numpy(),
                    "y": y_test,
                    "pred_2d_only": pred2d_test,
                    "pred_3d_only": pred3d_test,
                    "pred_early_concat_fusion": pred_early_test,
                    "pred_late_mean_fusion": late_mean_test,
                    "pred_late_val_weighted_2d_3d": late_weighted_test,
                    "pred_late_val_weighted_2d_early": early_weighted_test,
                    "pred_gated_val_utility_fusion": gated_pred,
                    "pred_moe_stacking_val_calibrated": moe_pred,
                    "pred_modality_dropout_early_fusion": pred_drop_test,
                    "gate_score": gate_score,
                    "gate_selected": gate_selected,
                    "late_weight_3d_only": late_w,
                    "late_weight_early_fusion": early_w,
                }
            )
        )

    return metric_rows, pred_frames, calibration_rows


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    ensure_dir(out_dir / "predictions")
    metadata = {
        "started_at": now_iso(),
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tasks": args.tasks,
        "seeds": args.seeds,
        "split": args.split,
        "split_seed": args.split_seed,
        "max_mols": args.max_mols,
        "feature2d_set": args.feature2d_set,
        "feature3d_set": args.feature3d_set,
        "feature3d_source_dir": args.feature3d_source_dir,
        "protocol_note": "Train on train split; validation split tunes fusion weights, utility gate, and MoE stacker; test split is held out.",
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    invalid = [task for task in args.tasks if task not in DATASETS]
    if invalid:
        raise ValueError(f"Unknown tasks: {invalid}")

    metric_rows: list[dict] = []
    pred_frames_by_task: dict[str, list[pd.DataFrame]] = {}
    calibration_rows: list[dict] = []
    for task in args.tasks:
        print(f"[if-fusion] {task}", flush=True)
        rows, frames, cal = run_task(task, args)
        metric_rows.extend(rows)
        pred_frames_by_task[task] = frames
        calibration_rows.extend(cal)
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out_dir / "predictions" / f"{task}_fusion_predictions.csv", index=False)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "metrics_long.csv", index=False)
    _summarize(metrics).to_csv(out_dir / "method_summary.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(out_dir / "fusion_calibration_by_seed.csv", index=False)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
