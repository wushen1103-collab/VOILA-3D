from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.svm import SVC, SVR

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voila3d.data import DATASETS, assign_splits, load_dataset
from voila3d.features import build_2d_feature_matrix, cheap_descriptors, ecfp_bits
from voila3d.metrics import classification_metrics, regression_metrics
from voila3d.utils import ensure_dir, now_iso, set_reproducible


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["ESOL", "FreeSolv", "Lipophilicity", "BBBP", "BACE", "HIV"])
    p.add_argument("--split", choices=["scaffold", "scaffold_balanced", "random"], default="scaffold_balanced")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--out-dir", default="results/baseline_matrix_scaffold_balanced_5seed")
    p.add_argument("--max-mols", type=int, default=None)
    p.add_argument("--threads", type=int, default=24)
    p.add_argument("--feature2d-set", choices=["ecfp_desc", "rdkit2d_combo"], default="ecfp_desc")
    p.add_argument("--combo-jobs", type=int, default=16)
    p.add_argument("--xgb-device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--skip-svm-above", type=int, default=15000)
    p.add_argument("--skip-krr-above", type=int, default=2500)
    return p.parse_args()


def _gpu_available() -> bool:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
        return "GPU" in out
    except Exception:
        return False


def _xgb_params(task_type: str, seed: int, threads: int, device: str, y_train: np.ndarray) -> dict:
    base = dict(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.035,
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
        pos = float(np.sum(y_train == 1))
        neg = float(np.sum(y_train == 0))
        base.update(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=(neg / max(pos, 1.0)),
        )
    else:
        base.update(objective="reg:squarederror", eval_metric="mae")
    return base


def _features(df: pd.DataFrame, feature2d_set: str, jobs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smiles = df["canonical_smiles"].tolist()
    x_ecfp = np.vstack([ecfp_bits(s) for s in smiles]).astype(np.float32)
    x_desc = np.vstack([cheap_descriptors(s) for s in smiles]).astype(np.float32)
    x2d = build_2d_feature_matrix(smiles, x_ecfp, x_desc, feature2d_set=feature2d_set, jobs=jobs)
    return x_ecfp, x_desc, x2d


def _classification_from_scores(y_true: np.ndarray, score: np.ndarray, probability_like: bool) -> dict[str, float]:
    out: dict[str, float] = {
        "Accuracy": float(accuracy_score(y_true, (score >= (0.5 if probability_like else 0.0)).astype(int))),
    }
    if len(set(int(x) for x in np.unique(y_true))) == 2:
        out["AUROC"] = float(roc_auc_score(y_true, score))
        out["AUPRC"] = float(average_precision_score(y_true, score))
    else:
        out["AUROC"] = float("nan")
        out["AUPRC"] = float("nan")
    if probability_like:
        out["LogLoss"] = float(log_loss(y_true, np.clip(score, 1e-6, 1 - 1e-6), labels=[0, 1]))
    else:
        out["LogLoss"] = float("nan")
    return out


def _predict_classification(model, x: np.ndarray) -> tuple[np.ndarray, bool]:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1].astype(float), True
    if hasattr(model, "decision_function"):
        return model.decision_function(x).astype(float), False
    pred = model.predict(x).astype(float)
    return pred, bool(np.nanmin(pred) >= 0.0 and np.nanmax(pred) <= 1.0)


def _method_specs(task_type: str, threads: int, seed: int, n_train: int, xgb_device: str, y_train: np.ndarray) -> list[tuple[str, str, object]]:
    if task_type == "classification":
        specs: list[tuple[str, str, object]] = [
            (
                "LR_ECFPDesc",
                "classic-linear",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    MaxAbsScaler(),
                    LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", n_jobs=threads, random_state=seed),
                ),
            ),
            (
                "RF_ECFPDesc",
                "classic-ensemble",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    RandomForestClassifier(
                        n_estimators=300,
                        max_features="sqrt",
                        min_samples_leaf=1,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=threads,
                    ),
                ),
            ),
            (
                "ExtraTrees_ECFPDesc",
                "classic-ensemble",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    ExtraTreesClassifier(
                        n_estimators=500,
                        max_features="sqrt",
                        min_samples_leaf=1,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=threads,
                    ),
                ),
            ),
            (
                "SVM_RBF_Desc",
                "classic-kernel",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    SVC(C=10.0, gamma="scale", class_weight="balanced", cache_size=1200),
                ),
            ),
        ]
        if XGBClassifier is not None:
            specs.append(
                (
                    "XGB_ECFPDesc",
                    "descriptor-boosting",
                    make_pipeline(
                        SimpleImputer(strategy="median"),
                        StandardScaler(with_mean=False),
                        XGBClassifier(**_xgb_params(task_type, seed, threads, xgb_device, y_train)),
                    ),
                )
            )
        return specs

    specs = [
        (
            "Ridge_ECFPDesc",
            "classic-linear",
            make_pipeline(SimpleImputer(strategy="median"), MaxAbsScaler(), Ridge(alpha=1.0, random_state=seed)),
        ),
        (
            "RF_ECFPDesc",
            "classic-ensemble",
            make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestRegressor(
                    n_estimators=350,
                    max_features="sqrt",
                    min_samples_leaf=1,
                    random_state=seed,
                    n_jobs=threads,
                ),
            ),
        ),
        (
            "ExtraTrees_ECFPDesc",
            "classic-ensemble",
            make_pipeline(
                SimpleImputer(strategy="median"),
                ExtraTreesRegressor(
                    n_estimators=500,
                    max_features="sqrt",
                    min_samples_leaf=1,
                    random_state=seed,
                    n_jobs=threads,
                ),
            ),
        ),
        (
            "SVR_RBF_Desc",
            "classic-kernel",
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVR(C=30.0, gamma=0.1, epsilon=0.05)),
        ),
    ]
    if n_train <= 2500:
        specs.append(
            (
                "KRR_RBF_Desc",
                "classic-kernel",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KernelRidge(alpha=0.03, kernel="rbf", gamma=0.1)),
            )
        )
    if XGBRegressor is not None:
        specs.append(
            (
                "XGB_ECFPDesc",
                "descriptor-boosting",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(with_mean=False),
                    XGBRegressor(**_xgb_params(task_type, seed, threads, xgb_device, y_train)),
                ),
            )
        )
    return specs


def _select_x(method_name: str, x_desc: np.ndarray, x2d: np.ndarray) -> np.ndarray:
    return x_desc if method_name.endswith("_Desc") else x2d


def run_task(task: str, args: argparse.Namespace) -> list[dict]:
    df, spec, _ = load_dataset(task, args.data_dir, args.max_mols, seed=min(args.seeds))
    df["split"] = assign_splits(df, args.split, seed=min(args.seeds), task_type=spec.task_type)
    _, x_desc, x2d = _features(df, args.feature2d_set, args.combo_jobs)
    y = df["y"].to_numpy()
    split = df["split"].to_numpy()
    train = split == "train"
    test = split == "test"
    rows = []
    for seed in args.seeds:
        set_reproducible(seed)
        specs = _method_specs(spec.task_type, args.threads, seed, int(train.sum()), args.xgb_device, y[train])
        for method, group, model in specs:
            if method == "SVM_RBF_Desc" and int(train.sum()) > args.skip_svm_above:
                continue
            if method == "KRR_RBF_Desc" and int(train.sum()) > args.skip_krr_above:
                continue
            x = _select_x(method, x_desc, x2d)
            if spec.task_type == "classification" and len(np.unique(y[train])) < 2:
                continue
            model.fit(x[train], y[train])
            row = {
                "task": task,
                "task_type": spec.task_type,
                "seed": seed,
                "method": method,
                "method_group": group,
                "split": args.split,
                "source": "ours_rerun_same_split",
                "feature_set": "cheap_desc" if method.endswith("_Desc") else args.feature2d_set,
                "n_train": int(train.sum()),
                "n_val": int((split == "val").sum()),
                "n_test": int(test.sum()),
            }
            if spec.task_type == "classification":
                score, probability_like = _predict_classification(model, x[test])
                row.update(_classification_from_scores(y[test], score, probability_like))
            else:
                pred = model.predict(x[test]).astype(float)
                row.update(regression_metrics(y[test], pred))
            rows.append(row)
    return rows


def _mean_std(vals: pd.Series) -> str:
    vals = pd.to_numeric(vals, errors="coerce")
    return f"{vals.mean():.6f} +/- {vals.std(ddof=0):.6f}"


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [c for c in ["AUROC", "AUPRC", "Accuracy", "LogLoss", "MAE", "RMSE", "R2"] if c in metrics.columns]
    for (task, task_type, method, group), g in metrics.groupby(["task", "task_type", "method", "method_group"]):
        primary = "AUROC" if task_type == "classification" else "MAE"
        row = {
            "task": task,
            "task_type": task_type,
            "method_group": group,
            "method": method,
            "primary_metric": primary,
            "primary_mean": float(g[primary].mean()),
            "primary_std": float(g[primary].std(ddof=0)),
            "n_seeds": int(g["seed"].nunique()),
            "source": g["source"].iloc[0],
            "split": g["split"].iloc[0],
            "n_train": int(g["n_train"].iloc[0]),
            "n_val": int(g["n_val"].iloc[0]),
            "n_test": int(g["n_test"].iloc[0]),
        }
        for col in metric_cols:
            if col in g:
                row[f"{col}_mean_std"] = _mean_std(g[col])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["task", "method_group", "method"])


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
        "split": args.split,
        "seeds": args.seeds,
        "feature2d_set": args.feature2d_set,
        "threads": args.threads,
        "xgb_device": args.xgb_device,
        "protocol_note": "Fixed split generated with min(seeds); mean/std reflects model randomness under the same split.",
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    invalid = [t for t in args.tasks if t not in DATASETS]
    if invalid:
        raise ValueError(f"Unknown tasks: {invalid}. Known: {sorted(DATASETS)}")

    all_rows = []
    for task in args.tasks:
        print(f"[baseline] {task}", flush=True)
        all_rows.extend(run_task(task, args))
    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(out_dir / "metrics_long.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(out_dir / "method_summary.csv", index=False)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
