from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover - environment fallback
    XGBClassifier = None
    XGBRegressor = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voila3d.data import DATASETS, assign_splits, dataset_card, load_dataset
from voila3d.features import build_2d_feature_matrix, build_feature_tables, descriptor_frame
from voila3d.metrics import classification_metrics, per_sample_loss, primary_metric, regression_metrics
from voila3d.routing import evaluate_routing_curves, make_router_features, train_voi_router, true_benefit, uncertainty_scores
from voila3d.utils import ensure_dir, now_iso, set_reproducible


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["ESOL", "FreeSolv", "Lipophilicity", "BBBP", "BACE", "HIV"])
    p.add_argument(
        "--split",
        choices=["scaffold", "scaffold_balanced", "scaffold_randomized", "random"],
        default="scaffold_balanced",
    )
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--budgets", nargs="+", type=float, default=[0, 5, 10, 20, 40, 60, 80, 100])
    p.add_argument("--max-mols", type=int, default=None)
    p.add_argument("--conformers", type=int, default=1)
    p.add_argument("--feature2d-set", choices=["ecfp_desc", "rdkit2d_combo"], default="ecfp_desc")
    p.add_argument("--feature3d-set", choices=["usr", "rdkit_scalar", "rdkit_rich", "rdkit_full"], default="usr")
    p.add_argument("--conformer-jobs", type=int, default=16)
    p.add_argument("--parallel-tasks", type=int, default=1)
    p.add_argument("--model-threads", type=int, default=16)
    p.add_argument("--xgb-device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--router-label-source", choices=["val", "oof"], default="val")
    p.add_argument("--router-folds", type=int, default=5)
    p.add_argument("--router-feature-set", choices=["desc", "ecfp_desc"], default="desc")
    p.add_argument("--classification-benefit", choices=["logloss", "margin", "auc_contrib"], default="logloss")
    p.add_argument("--out-dir", default="results/fast_screen")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--feature-cache-source-dir", default=None)
    p.add_argument("--feature-cache-source-split", default=None)
    p.add_argument("--match-source-split-fractions", action="store_true")
    return p.parse_args()


def _gpu_available() -> bool:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
        return "GPU" in out
    except Exception:
        return False


def _xgb_params(task_type: str, seed: int, threads: int, device: str) -> dict:
    base = dict(
        n_estimators=700,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=threads,
        tree_method="hist",
        verbosity=0,
    )
    if device == "cuda" or (device == "auto" and _gpu_available()):
        base["device"] = "cuda"
    if task_type == "classification":
        base.update(objective="binary:logistic", eval_metric="logloss")
    else:
        base.update(objective="reg:squarederror", eval_metric="mae")
    return base


def _fit_xgb(task_type: str, x_train: np.ndarray, y_train: np.ndarray, seed: int, threads: int, device: str):
    if XGBClassifier is None or XGBRegressor is None:
        cls = HistGradientBoostingClassifier if task_type == "classification" else HistGradientBoostingRegressor
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            cls(
                max_iter=350,
                learning_rate=0.04,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                random_state=seed,
            ),
        )
        model.fit(x_train, y_train)
        return model, "sklearn_hgb"

    cls = XGBClassifier if task_type == "classification" else XGBRegressor
    params = _xgb_params(task_type, seed, threads, device)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(with_mean=False), cls(**params))
    try:
        model.fit(x_train, y_train)
        return model, params.get("device", "cpu")
    except Exception as exc:
        if params.get("device") == "cuda":
            params.pop("device", None)
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(with_mean=False), cls(**params))
            model.fit(x_train, y_train)
            print(f"[warn] CUDA XGBoost failed ({type(exc).__name__}); retried on CPU", flush=True)
            return model, "cpu"
        raise


def _predict(model, task_type: str, x: np.ndarray) -> np.ndarray:
    if task_type == "classification":
        return model.predict_proba(x)[:, 1].astype(float)
    return model.predict(x).astype(float)


def _make_router_features(
    x_ecfp: np.ndarray,
    x_desc: np.ndarray,
    pred2d: np.ndarray,
    task_type: str,
    feature_set: str,
    uncertainty_reference: np.ndarray | None = None,
) -> np.ndarray:
    base = make_router_features(x_desc, pred2d, task_type, uncertainty_reference)
    if feature_set == "desc":
        return base
    if feature_set == "ecfp_desc":
        return np.hstack([x_ecfp, base]).astype(np.float32)
    raise ValueError(f"Unknown router feature set: {feature_set}")


def _fit_oof_router_labels(
    task_type: str,
    x2d: np.ndarray,
    x3d_aug: np.ndarray,
    x_ecfp: np.ndarray,
    x_desc: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    seed: int,
    folds: int,
    threads: int,
    device: str,
    feature_set: str,
    classification_benefit: str,
) -> dict[str, np.ndarray] | None:
    pool_idx = np.flatnonzero(train | val)
    n_pool = len(pool_idx)
    n_splits = min(int(folds), n_pool)
    if n_splits < 2:
        return None

    if task_type == "classification":
        counts = np.bincount(y[pool_idx].astype(int), minlength=2)
        n_splits = min(n_splits, int(counts.min()))
        if n_splits < 2:
            return None
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 1701)
        splits = splitter.split(pool_idx, y[pool_idx])
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed + 1701)
        splits = splitter.split(pool_idx)

    pred2d_oof = np.full(n_pool, np.nan, dtype=float)
    pred3d_oof = np.full(n_pool, np.nan, dtype=float)
    fold_id_oof = np.full(n_pool, -1, dtype=int)
    for fold_id, (fit_rel, hold_rel) in enumerate(splits):
        fit_idx = pool_idx[fit_rel]
        hold_idx = pool_idx[hold_rel]
        if task_type == "classification" and len(np.unique(y[fit_idx])) < 2:
            continue
        m2d, _ = _fit_xgb(task_type, x2d[fit_idx], y[fit_idx], seed + 2000 + fold_id, threads, device)
        m3d, _ = _fit_xgb(task_type, x3d_aug[fit_idx], y[fit_idx], seed + 3000 + fold_id, threads, device)
        pred2d_oof[hold_rel] = _predict(m2d, task_type, x2d[hold_idx])
        pred3d_oof[hold_rel] = _predict(m3d, task_type, x3d_aug[hold_idx])
        fold_id_oof[hold_rel] = fold_id

    ok = np.isfinite(pred2d_oof) & np.isfinite(pred3d_oof)
    if int(ok.sum()) < max(4, n_splits):
        return None
    idx = pool_idx[ok]
    benefit = true_benefit(task_type, y[idx], pred2d_oof[ok], pred3d_oof[ok], classification_mode=classification_benefit)
    return {
        "index": idx,
        "x_router": _make_router_features(
            x_ecfp[idx],
            x_desc[idx],
            pred2d_oof[ok],
            task_type,
            feature_set,
            uncertainty_reference=pred2d_oof[ok],
        ),
        "benefit": benefit,
        "pred2d": pred2d_oof[ok],
        "pred3d": pred3d_oof[ok],
        "fold_id": fold_id_oof[ok],
    }


def _feature_cache_paths(out_dir: Path, task: str, split: str, max_mols: int | None, conformers: int, feature3d_set: str) -> dict[str, Path]:
    suffix = "" if feature3d_set == "usr" else f"_{feature3d_set}"
    key = f"{task}_{split}_max{max_mols or 'all'}_k{conformers}{suffix}"
    cache = ensure_dir(out_dir / "feature_cache")
    return {
        "meta": cache / f"{key}_meta.csv",
        "ecfp": cache / f"{key}_ecfp.npy",
        "desc": cache / f"{key}_desc.npy",
        "x3d": cache / f"{key}_x3d.npy",
        "manifest": cache / f"{key}_manifest.csv",
    }


def _load_or_build_features(
    df: pd.DataFrame,
    out_dir: Path,
    task: str,
    split: str,
    max_mols: int | None,
    conformers: int,
    feature3d_set: str,
    conformer_jobs: int,
    seed: int,
    feature_cache_source_dir: str | None = None,
    feature_cache_source_split: str | None = None,
):
    paths = _feature_cache_paths(out_dir, task, split, max_mols, conformers, feature3d_set)
    if all(p.exists() for p in paths.values()):
        meta = pd.read_csv(paths["meta"])
        x_ecfp = np.load(paths["ecfp"])
        x_desc = np.load(paths["desc"])
        x3d = np.load(paths["x3d"])
        manifest = pd.read_csv(paths["manifest"])
        return meta, x_ecfp, x_desc, x3d, manifest
    if feature_cache_source_dir:
        source_paths = _feature_cache_paths(
            Path(feature_cache_source_dir),
            task,
            feature_cache_source_split or split,
            max_mols,
            conformers,
            feature3d_set,
        )
        missing = [
            str(source_paths[key])
            for key in ["meta", "ecfp", "desc", "x3d", "manifest"]
            if not source_paths[key].exists()
        ]
        if missing:
            raise FileNotFoundError("Missing reusable feature cache files: " + "; ".join(missing))
        source_meta = pd.read_csv(source_paths["meta"])
        if source_meta["mol_id"].astype(str).tolist() != df["mol_id"].astype(str).tolist():
            raise ValueError(f"Feature cache molecule order mismatch for {task}")
        meta = df[["mol_id", "canonical_smiles", "y", "split"]].copy()
        meta.to_csv(paths["meta"], index=False)
        for key in ["ecfp", "desc", "x3d", "manifest"]:
            try:
                os.link(source_paths[key], paths[key])
            except OSError:
                shutil.copy2(source_paths[key], paths[key])
        return (
            meta,
            np.load(paths["ecfp"]),
            np.load(paths["desc"]),
            np.load(paths["x3d"]),
            pd.read_csv(paths["manifest"]),
        )
    x_ecfp, x_desc, x3d, manifest = build_feature_tables(
        df,
        n_bits=2048,
        n_conformers=conformers,
        conformer_jobs=conformer_jobs,
        seed=seed,
        feature3d_set=feature3d_set,
    )
    meta = df[["mol_id", "canonical_smiles", "y", "split"]].copy()
    meta.to_csv(paths["meta"], index=False)
    np.save(paths["ecfp"], x_ecfp)
    np.save(paths["desc"], x_desc)
    np.save(paths["x3d"], x3d)
    manifest.to_csv(paths["manifest"], index=False)
    return meta, x_ecfp, x_desc, x3d, manifest


def _read_nonempty_csvs(paths: list[Path]) -> list[pd.DataFrame]:
    frames = []
    for path in paths:
        if not path.exists() or path.stat().st_size <= 1:
            continue
        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            continue
    return frames


def run_task(task: str, args: argparse.Namespace) -> dict[str, str | int | float]:
    out_dir = ensure_dir(args.out_dir)
    ensure_dir(out_dir / "dataset_cards")
    ensure_dir(out_dir / "predictions")
    set_reproducible(min(args.seeds))

    df, spec, raw_path = load_dataset(task, args.data_dir, args.max_mols, seed=0)
    frac_train, frac_val = 0.8, 0.1
    if args.match_source_split_fractions:
        if not args.feature_cache_source_dir:
            raise ValueError("--match-source-split-fractions requires --feature-cache-source-dir")
        source_meta_path = _feature_cache_paths(
            Path(args.feature_cache_source_dir),
            task,
            args.feature_cache_source_split or args.split,
            args.max_mols,
            args.conformers,
            args.feature3d_set,
        )["meta"]
        source_meta = pd.read_csv(source_meta_path)
        frac_train = float((source_meta["split"] == "train").mean())
        frac_val = float((source_meta["split"] == "val").mean())
    df["split"] = assign_splits(
        df,
        args.split,
        seed=args.split_seed,
        frac_train=frac_train,
        frac_val=frac_val,
        task_type=spec.task_type,
    )
    feature_paths = _feature_cache_paths(out_dir, task, args.split, args.max_mols, args.conformers, args.feature3d_set)
    feature_cache_hit = all(p.exists() for p in feature_paths.values())
    feature_start = time.perf_counter()
    meta, x_ecfp, x_desc, x3d, manifest = _load_or_build_features(
        df,
        out_dir,
        task,
        args.split,
        args.max_mols,
        args.conformers,
        args.feature3d_set,
        args.conformer_jobs,
        seed=0,
        feature_cache_source_dir=args.feature_cache_source_dir,
        feature_cache_source_split=args.feature_cache_source_split,
    )
    feature_elapsed_sec = time.perf_counter() - feature_start
    desc_df = descriptor_frame(df, x_desc)
    df = df.merge(desc_df, on="mol_id", how="left")
    conf_success = float(manifest["success"].mean()) if len(manifest) else float("nan")
    conf_time = float(manifest["elapsed_sec"].mean()) if len(manifest) else float("nan")
    dataset_card(df, spec, raw_path, args.split, conf_success, conf_time, out_dir / "dataset_cards" / f"{task}.yaml")

    x2d = build_2d_feature_matrix(
        df["canonical_smiles"].tolist(),
        x_ecfp,
        x_desc,
        feature2d_set=args.feature2d_set,
        jobs=args.conformer_jobs,
    )
    x3d_aug = np.hstack([x2d, x3d]).astype(np.float32)
    y = df["y"].to_numpy()
    split = df["split"].to_numpy()

    metric_rows = []
    route_rows = []
    pred_rows = []
    label_rows = []
    timing_rows = []

    for seed in args.seeds:
        seed_start = time.perf_counter()
        train = split == "train"
        val = split == "val"
        test = split == "test"
        if spec.task_type == "classification" and (len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2):
            print(f"[warn] {task} seed={seed}: insufficient class diversity; skipping", flush=True)
            continue

        fit2d_start = time.perf_counter()
        m2d, device2d = _fit_xgb(spec.task_type, x2d[train], y[train], seed, args.model_threads, args.xgb_device)
        fit2d_sec = time.perf_counter() - fit2d_start
        fit3d_start = time.perf_counter()
        m3d, device3d = _fit_xgb(spec.task_type, x3d_aug[train], y[train], seed + 1000, args.model_threads, args.xgb_device)
        fit3d_sec = time.perf_counter() - fit3d_start
        predict_start = time.perf_counter()
        pred2d_val = _predict(m2d, spec.task_type, x2d[val])
        pred3d_val = _predict(m3d, spec.task_type, x3d_aug[val])
        pred2d_train = _predict(m2d, spec.task_type, x2d[train])
        pred2d_test = _predict(m2d, spec.task_type, x2d[test])
        pred3d_test = _predict(m3d, spec.task_type, x3d_aug[test])
        predict_sec = time.perf_counter() - predict_start

        for method, pred in [("all_2d", pred2d_test), ("all_3d_aug", pred3d_test)]:
            mets = regression_metrics(y[test], pred) if spec.task_type == "regression" else classification_metrics(y[test], pred)
            row = {
                "task": task,
                "task_type": spec.task_type,
                "seed": seed,
                "method": method,
                "split": args.split,
                "n_train": int(train.sum()),
                "n_val": int(val.sum()),
                "n_test": int(test.sum()),
                "xgb_device_2d": device2d,
                "xgb_device_3d": device3d,
            }
            row.update(mets)
            metric_rows.append(row)

        benefit_val = true_benefit(spec.task_type, y[val], pred2d_val, pred3d_val, classification_mode=args.classification_benefit)
        router_x_val = _make_router_features(
            x_ecfp[val],
            x_desc[val],
            pred2d_val,
            spec.task_type,
            args.router_feature_set,
            uncertainty_reference=pred2d_train,
        )
        router_x_train = router_x_val
        router_benefit_train = benefit_val
        router_label_source = "val"
        uncertainty_reference = pred2d_train
        oof_labels = None
        oof_label_sec = 0.0
        if args.router_label_source == "oof":
            oof_start = time.perf_counter()
            oof_labels = _fit_oof_router_labels(
                spec.task_type,
                x2d,
                x3d_aug,
                x_ecfp,
                x_desc,
                y,
                train,
                val,
                seed,
                args.router_folds,
                args.model_threads,
                args.xgb_device,
                args.router_feature_set,
                args.classification_benefit,
            )
            oof_label_sec = time.perf_counter() - oof_start
            if oof_labels is not None:
                router_x_train = oof_labels["x_router"]
                router_benefit_train = oof_labels["benefit"]
                router_label_source = f"oof{args.router_folds}"
                uncertainty_reference = oof_labels["pred2d"]
            else:
                print(f"[warn] {task} seed={seed}: OOF router labels unavailable; falling back to val", flush=True)
        router_fit_start = time.perf_counter()
        router = train_voi_router(router_x_train, router_benefit_train, seed=seed)
        router_fit_sec = time.perf_counter() - router_fit_start
        route_eval_start = time.perf_counter()
        router_scores = router.predict(
            _make_router_features(
                x_ecfp[test],
                x_desc[test],
                pred2d_test,
                spec.task_type,
                args.router_feature_set,
                uncertainty_reference=uncertainty_reference,
            )
        )
        random_scores = np.random.default_rng(seed).normal(size=test.sum())
        uncertainty = uncertainty_scores(spec.task_type, pred2d_test, uncertainty_reference)
        flex = x_desc[test, 2] + 0.05 * x_desc[test, 1]
        oracle = true_benefit(spec.task_type, y[test], pred2d_test, pred3d_test, classification_mode=args.classification_benefit)
        curves = evaluate_routing_curves(
            task,
            spec.task_type,
            y[test],
            pred2d_test,
            pred3d_test,
            {
                "voi_router": router_scores,
                "random": random_scores,
                "uncertainty": uncertainty,
                "flexibility": flex,
                "oracle": oracle,
            },
            args.budgets,
            seed,
        )
        route_rows.append(curves)
        route_eval_sec = time.perf_counter() - route_eval_start

        test_ids = df.loc[test, "mol_id"].to_numpy()
        pred_rows.append(pd.DataFrame({
            "task": task,
            "seed": seed,
            "mol_id": test_ids,
            "y": y[test],
            "pred2d": pred2d_test,
            "pred3d_aug": pred3d_test,
            "voi_score": router_scores,
            "router_label_source": router_label_source,
            "router_feature_set": args.router_feature_set,
            "classification_benefit": args.classification_benefit,
            "uncertainty_score": uncertainty,
            "flexibility_score": flex,
            "oracle_benefit": oracle,
        }))
        if oof_labels is not None:
            idx = oof_labels["index"].astype(int)
            label_rows.append(pd.DataFrame({
                "task": task,
                "seed": seed,
                "mol_id": df.iloc[idx]["mol_id"].to_numpy(),
                "split": df.iloc[idx]["split"].to_numpy(),
                "label_source": router_label_source,
                "feature_set": args.router_feature_set,
                "classification_benefit": args.classification_benefit,
                "fold_id": oof_labels["fold_id"],
                "benefit": oof_labels["benefit"],
                "pred2d_label": oof_labels["pred2d"],
                "pred3d_label": oof_labels["pred3d"],
                "loss_2d": per_sample_loss(spec.task_type, y[idx], oof_labels["pred2d"]),
                "loss_3d": per_sample_loss(spec.task_type, y[idx], oof_labels["pred3d"]),
            }))
        else:
            label_rows.append(pd.DataFrame({
                "task": task,
                "seed": seed,
                "mol_id": df.loc[val, "mol_id"].to_numpy(),
                "split": df.loc[val, "split"].to_numpy(),
                "label_source": router_label_source,
                "feature_set": args.router_feature_set,
                "classification_benefit": args.classification_benefit,
                "fold_id": -1,
                "benefit": benefit_val,
                "pred2d_label": pred2d_val,
                "pred3d_label": pred3d_val,
                "loss_2d": per_sample_loss(spec.task_type, y[val], pred2d_val),
                "loss_3d": per_sample_loss(spec.task_type, y[val], pred3d_val),
            }))
        timing_rows.append(
            {
                "task": task,
                "task_type": spec.task_type,
                "seed": seed,
                "split": args.split,
                "n_train": int(train.sum()),
                "n_val": int(val.sum()),
                "n_test": int(test.sum()),
                "feature_cache_hit": bool(feature_cache_hit),
                "feature_elapsed_sec": float(feature_elapsed_sec),
                "conformer_success_rate": conf_success,
                "mean_conformer_time_sec": conf_time,
                "fit2d_sec": float(fit2d_sec),
                "fit3d_aug_sec": float(fit3d_sec),
                "predict_val_test_sec": float(predict_sec),
                "oof_label_sec": float(oof_label_sec),
                "router_fit_sec": float(router_fit_sec),
                "route_eval_sec": float(route_eval_sec),
                "seed_total_sec": float(time.perf_counter() - seed_start),
                "xgb_device_2d": device2d,
                "xgb_device_3d": device3d,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "model_threads": int(args.model_threads),
                "conformer_jobs": int(args.conformer_jobs),
                "split_seed": int(args.split_seed),
            }
        )

    pd.DataFrame(metric_rows).to_csv(out_dir / f"metrics_{task}.csv", index=False)
    if route_rows:
        pd.concat(route_rows, ignore_index=True).to_csv(out_dir / f"routing_{task}.csv", index=False)
    if pred_rows:
        pd.concat(pred_rows, ignore_index=True).to_csv(out_dir / "predictions" / f"{task}_predictions.csv", index=False)
    if label_rows:
        labels = pd.concat(label_rows, ignore_index=True)
        try:
            labels.to_parquet(out_dir / f"{task}_voi_labels.parquet", index=False)
        except ImportError:
            labels.to_csv(out_dir / f"{task}_voi_labels.csv", index=False)
    if timing_rows:
        pd.DataFrame(timing_rows).to_csv(out_dir / f"timing_{task}.csv", index=False)
    manifest.insert(0, "task", task)
    manifest.to_csv(out_dir / f"conformer_manifest_{task}.csv", index=False)
    return {
        "task": task,
        "n": int(len(df)),
        "conformer_success_rate": conf_success,
        "mean_conformer_time_sec": conf_time,
        "status": "done",
    }


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    ensure_dir(args.data_dir)
    metadata = {
        "started_at": now_iso(),
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tasks": args.tasks,
        "split": args.split,
        "split_seed": args.split_seed,
        "seeds": args.seeds,
        "max_mols": args.max_mols,
        "conformers": args.conformers,
        "feature2d_set": args.feature2d_set,
        "feature3d_set": args.feature3d_set,
        "router_label_source": args.router_label_source,
        "router_folds": args.router_folds,
        "router_feature_set": args.router_feature_set,
        "classification_benefit": args.classification_benefit,
        "regression_uncertainty_reference": "same-seed non-test OOF predictions when OOF labels are available",
        "feature_cache_source_dir": args.feature_cache_source_dir,
        "feature_cache_source_split": args.feature_cache_source_split,
        "match_source_split_fractions": args.match_source_split_fractions,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    invalid = [t for t in args.tasks if t not in DATASETS]
    if invalid:
        raise ValueError(f"Unknown tasks: {invalid}. Known: {sorted(DATASETS)}")

    summaries = Parallel(n_jobs=args.parallel_tasks, backend="loky")(
        delayed(run_task)(task, args) for task in args.tasks
    )
    pd.DataFrame(summaries).to_csv(out_dir / "run_summary.csv", index=False)
    metric_files = sorted(out_dir.glob("metrics_*.csv"))
    route_files = sorted(out_dir.glob("routing_*.csv"))
    manifest_files = sorted(out_dir.glob("conformer_manifest_*.csv"))
    timing_files = sorted(out_dir.glob("timing_*.csv"))
    metric_frames = _read_nonempty_csvs(metric_files)
    route_frames = _read_nonempty_csvs(route_files)
    manifest_frames = _read_nonempty_csvs(manifest_files)
    timing_frames = _read_nonempty_csvs(timing_files)
    if metric_frames:
        pd.concat(metric_frames, ignore_index=True).to_csv(out_dir / "metrics_long.csv", index=False)
    if route_frames:
        pd.concat(route_frames, ignore_index=True).to_csv(out_dir / "routing_curves.csv", index=False)
    if manifest_frames:
        pd.concat(manifest_frames, ignore_index=True).to_csv(out_dir / "conformer_manifest.csv", index=False)
    if timing_frames:
        pd.concat(timing_frames, ignore_index=True).to_csv(out_dir / "timing_long.csv", index=False)
    print(f"[done] outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
