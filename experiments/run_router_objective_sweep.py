from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.fast_screen import _feature_cache_paths
from experiments.run_if_router_ablation import _support_features
from voila3d.features import DESCRIPTOR_NAMES
from voila3d.metrics import classification_metrics, per_sample_loss, primary_metric, regression_metrics
from voila3d.routing import evaluate_routing_curves, route_predictions, uncertainty_scores
from voila3d.utils import ensure_dir, now_iso


TASK_TYPE = {
    "BACE": "classification",
    "BBBP": "classification",
    "HIV": "classification",
    "ESOL": "regression",
    "FreeSolv": "regression",
    "Lipophilicity": "regression",
    "QM9_MU": "regression",
    "QM9_R2": "regression",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--oof-results-dir", action="append", required=True)
    p.add_argument("--out-dir", default="results/if_router_objective_sweep_v1")
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--budgets", nargs="+", type=float, default=[0, 5, 10, 20, 40, 60, 80, 100])
    p.add_argument("--main-budget", type=float, default=20.0)
    p.add_argument("--router-estimators", type=int, default=128)
    p.add_argument("--router-n-jobs", type=int, default=8)
    p.add_argument("--pair-samples", type=int, default=30000)
    p.add_argument("--pair-margin-frac", type=float, default=0.25)
    p.add_argument("--bootstrap-iters", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=1701)
    return p.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_labels(results_dir: Path, task: str) -> pd.DataFrame:
    for path in [results_dir / f"{task}_voi_labels.parquet", results_dir / f"{task}_voi_labels.csv"]:
        if path.exists():
            return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    raise FileNotFoundError(f"Missing OOF utility labels for {task} in {results_dir}")


def _load_cache(results_dir: Path, task: str, run_meta: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    paths = _feature_cache_paths(
        results_dir,
        task,
        run_meta.get("split", "scaffold_balanced"),
        run_meta.get("max_mols", 12000),
        run_meta.get("conformers", 10),
        run_meta.get("feature3d_set", "usr"),
    )
    missing = [str(paths[k]) for k in ["meta", "ecfp", "desc", "x3d"] if not paths[k].exists()]
    if missing:
        raise FileNotFoundError("Missing feature cache files: " + "; ".join(missing))
    manifest_path = results_dir / f"conformer_manifest_{task}.csv"
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    return pd.read_csv(paths["meta"]), np.load(paths["ecfp"]), np.load(paths["desc"]), np.load(paths["x3d"]), manifest


def _utility_from_predictions(task_type: str, y: np.ndarray, pred2d: np.ndarray, pred3d: np.ndarray) -> np.ndarray:
    return per_sample_loss(task_type, y, pred2d) - per_sample_loss(task_type, y, pred3d)


def _standardize_by_train(train_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    train_scores = np.asarray(train_scores, dtype=float)
    scores = np.asarray(scores, dtype=float)
    mu = np.nanmean(train_scores)
    sd = np.nanstd(train_scores)
    if not np.isfinite(sd) or sd < 1e-12:
        sd = 1.0
    out = (scores - mu) / sd
    out[~np.isfinite(out)] = 0.0
    return out


def _metric_value(task_type: str, y: np.ndarray, pred: np.ndarray) -> float:
    metric = primary_metric(task_type)
    mets = regression_metrics(y, pred) if task_type == "regression" else classification_metrics(y, np.clip(pred, 1e-6, 1 - 1e-6))
    return float(mets[metric])


def _positive_performance(task_type: str, value: float) -> float:
    return -float(value) if primary_metric(task_type) == "MAE" else float(value)


def _ndcg_at_budget(utility: np.ndarray, scores: np.ndarray, budget: float) -> float:
    rel = np.maximum(np.asarray(utility, dtype=float), 0.0)
    if float(np.nansum(rel)) <= 0:
        return float("nan")
    k = max(1, int(round(len(rel) * budget / 100.0)))
    try:
        return float(ndcg_score(rel[None, :], np.asarray(scores, dtype=float)[None, :], k=k))
    except Exception:
        return float("nan")


def _precision_at_budget(utility: np.ndarray, scores: np.ndarray, budget: float, tau: float = 0.0) -> float:
    scores = np.asarray(scores, dtype=float)
    beneficial = np.asarray(utility, dtype=float) > tau
    k = max(1, int(round(len(scores) * budget / 100.0)))
    idx = np.argpartition(-scores, kth=min(k - 1, len(scores) - 1))[:k]
    return float(beneficial[idx].mean())


def _auc_safe(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y.astype(int))) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y.astype(int), scores))
    except Exception:
        return float("nan")


def _ap_safe(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y.astype(int))) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y.astype(int), scores))
    except Exception:
        return float("nan")


def _spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    try:
        val = spearmanr(x, y, nan_policy="omit").correlation
        return float(val) if np.isfinite(val) else float("nan")
    except Exception:
        return float("nan")


def _budget_integrated_gain(curves: pd.DataFrame, task_type: str) -> pd.DataFrame:
    rows = []
    for (task, seed, router), g in curves.groupby(["task", "seed", "router"]):
        g = g.sort_values("budget")
        base = g[np.isclose(g["budget"], 0.0)]
        if base.empty:
            continue
        p0 = _positive_performance(task_type, float(base["primary_value"].iloc[0]))
        xs = g["budget"].to_numpy(float) / 100.0
        ys = np.asarray([_positive_performance(task_type, v) - p0 for v in g["primary_value"].to_numpy(float)], dtype=float)
        rows.append(
            {
                "task": task,
                "task_type": task_type,
                "seed": int(seed),
                "router": router,
                "BIG": float(np.trapezoid(ys, xs)),
                "p0": float(p0),
            }
        )
    return pd.DataFrame(rows)


def _summarize(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for col in value_cols:
            vals = pd.to_numeric(g[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=0))
            row[f"{col}_mean_std"] = f"{vals.mean():.6f} +/- {vals.std(ddof=0):.6f}"
        row["n_seeds"] = int(g["seed"].nunique()) if "seed" in g else int(len(g))
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_big_delta(big: pd.DataFrame, final_router: str, baselines: list[str], iters: int, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    pivot = big.pivot_table(index=["task", "seed"], columns="router", values="BIG", aggfunc="mean")
    if final_router not in pivot.columns:
        return pd.DataFrame()
    for base in baselines:
        if base not in pivot.columns:
            continue
        paired = pivot[[final_router, base]].dropna()
        if paired.empty:
            continue
        delta = (paired[final_router] - paired[base]).to_numpy(float)
        obs = float(delta.mean())
        boots = []
        for _ in range(iters):
            idx = rng.integers(0, len(delta), size=len(delta))
            boots.append(float(delta[idx].mean()))
        rows.append(
            {
                "final_router": final_router,
                "baseline": base,
                "mean_delta_BIG": obs,
                "ci95_low": float(np.percentile(boots, 2.5)),
                "ci95_high": float(np.percentile(boots, 97.5)),
                "n_task_seed_pairs": int(len(delta)),
                "n_positive_pairs": int((delta > 0).sum()),
                "n_bootstrap": int(iters),
            }
        )
    return pd.DataFrame(rows)


def _forest_reg(n_estimators: int, n_jobs: int, seed: int):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=seed,
            n_jobs=n_jobs,
        ),
    )


def _forest_clf(n_estimators: int, n_jobs: int, seed: int):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=n_jobs,
        ),
    )


@dataclass
class LinearPairwiseRanker:
    imputer: SimpleImputer
    scaler: StandardScaler
    clf: SGDClassifier

    def score(self, x: np.ndarray) -> np.ndarray:
        z = self.scaler.transform(self.imputer.transform(x))
        return z @ self.clf.coef_.ravel()


def _train_pairwise_ranker(
    x: np.ndarray,
    utility: np.ndarray,
    seed: int,
    pair_samples: int,
    margin: float,
    weighted: bool,
) -> LinearPairwiseRanker:
    rng = np.random.default_rng(seed + 991)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    z = scaler.fit_transform(imputer.fit_transform(x)).astype(np.float32)
    n = len(utility)
    diffs = []
    labels = []
    weights = []
    tries = 0
    max_tries = max(pair_samples * 20, 1000)
    while len(labels) < pair_samples and tries < max_tries:
        tries += 1
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        du = float(utility[i] - utility[j])
        if abs(du) <= margin:
            continue
        diffs.append(z[i] - z[j])
        labels.append(1 if du > 0 else 0)
        weights.append(abs(du))
    if len(np.unique(labels)) < 2:
        # Fall back to a loose margin if the task has tiny utility spread.
        return _train_pairwise_ranker(x, utility, seed, pair_samples, 0.0, weighted) if margin > 0 else _degenerate_ranker(x)
    x_pair = np.vstack(diffs).astype(np.float32)
    y_pair = np.asarray(labels, dtype=int)
    sample_weight = np.asarray(weights, dtype=float)
    if weighted and np.nanmean(sample_weight) > 0:
        sample_weight = sample_weight / np.nanmean(sample_weight)
    else:
        sample_weight = None
    clf = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=1e-4,
        l1_ratio=0.05,
        max_iter=2000,
        tol=1e-4,
        random_state=seed,
    )
    clf.fit(x_pair, y_pair, sample_weight=sample_weight)
    return LinearPairwiseRanker(imputer=imputer, scaler=scaler, clf=clf)


def _degenerate_ranker(x: np.ndarray) -> LinearPairwiseRanker:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    z = scaler.fit_transform(imputer.fit_transform(x))
    clf = SGDClassifier(loss="log_loss")
    clf.classes_ = np.asarray([0, 1])
    clf.coef_ = np.zeros((1, z.shape[1]), dtype=float)
    clf.intercept_ = np.zeros(1, dtype=float)
    return LinearPairwiseRanker(imputer=imputer, scaler=scaler, clf=clf)


def _proba_positive(model, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = list(getattr(model[-1], "classes_", []))
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(len(x), dtype=float)


def _class_score(model, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = list(getattr(model[-1], "classes_", []))
    pos = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(x), dtype=float)
    neg = proba[:, classes.index(-1)] if -1 in classes else np.zeros(len(x), dtype=float)
    return pos - neg


def _manifest_features(meta: pd.DataFrame, manifest: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    if manifest.empty or "mol_id" not in manifest:
        return np.zeros((len(idx), 5), dtype=np.float32)
    cols = ["success", "elapsed_sec", "energy_std", "rmsd_mean"]
    table = manifest.drop_duplicates("mol_id").set_index("mol_id")
    rows = []
    for mol_id in meta.iloc[idx]["mol_id"].astype(str):
        if mol_id not in table.index:
            rows.append([0.0, np.nan, np.nan, np.nan, 1.0])
            continue
        r = table.loc[mol_id]
        success = float(bool(r.get("success", False)))
        rows.append(
            [
                success,
                float(r.get("elapsed_sec", np.nan)),
                float(r.get("energy_std", np.nan)),
                float(r.get("rmsd_mean", np.nan)),
                1.0 - success,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _feature_groups(
    meta: pd.DataFrame,
    x_ecfp: np.ndarray,
    x_desc: np.ndarray,
    manifest: pd.DataFrame,
    idx: np.ndarray,
    pred2d: np.ndarray,
    task_type: str,
    support: np.ndarray,
) -> dict[str, np.ndarray]:
    unc = uncertainty_scores(task_type, pred2d)[:, None].astype(np.float32)
    pred = pred2d[:, None].astype(np.float32)
    desc = x_desc[idx].astype(np.float32)
    geom_cols = [DESCRIPTOR_NAMES.index(c) for c in ["mol_wt", "heavy_atoms", "rotatable_bonds", "tpsa", "rings", "fraction_csp3", "bertz_ct"]]
    geom = np.hstack([desc[:, geom_cols], _manifest_features(meta, manifest, idx)]).astype(np.float32)
    support = support.astype(np.float32)
    model_meta = np.hstack([pred, unc]).astype(np.float32)
    return {
        "chemistry": desc,
        "chemistry_uncertainty": np.hstack([desc, model_meta]).astype(np.float32),
        "chemistry_uncertainty_geometry": np.hstack([desc, model_meta, geom]).astype(np.float32),
        "chemistry_uncertainty_support": np.hstack([desc, model_meta, support]).astype(np.float32),
        "full_mechanism": np.hstack([x_ecfp[idx].astype(np.float32), desc, model_meta, geom, support]).astype(np.float32),
    }


def _eval_dev_big(task: str, task_type: str, y: np.ndarray, pred2d: np.ndarray, pred3d: np.ndarray, score: np.ndarray, budgets: list[float], seed: int) -> float:
    curves = evaluate_routing_curves(task, task_type, y, pred2d, pred3d, {"candidate": score}, budgets, seed)
    return float(_budget_integrated_gain(curves, task_type)["BIG"].iloc[0])


def _train_and_score(
    objective: str,
    x_train: np.ndarray,
    u_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    margin: float,
) -> tuple[np.ndarray, dict]:
    info: dict[str, float | str | int] = {}
    if objective == "R0_raw_utility":
        model = _forest_reg(args.router_estimators, args.router_n_jobs, seed)
        model.fit(x_train, u_train)
        return model.predict(x_test), info
    if objective == "R1_binary":
        y = (u_train > 0.0).astype(int)
        if len(np.unique(y)) < 2:
            return np.full(len(x_test), float(y[0]) if len(y) else 0.0), info
        model = _forest_clf(args.router_estimators, args.router_n_jobs, seed)
        model.fit(x_train, y)
        return _proba_positive(model, x_test), info
    if objective == "R2_margin_binary":
        keep = np.abs(u_train) > margin
        y = (u_train[keep] > margin).astype(int)
        info["margin"] = float(margin)
        info["neutral_rate"] = float(1.0 - keep.mean())
        if int(keep.sum()) < 16 or len(np.unique(y)) < 2:
            return np.zeros(len(x_test), dtype=float), info
        model = _forest_clf(args.router_estimators, args.router_n_jobs, seed)
        model.fit(x_train[keep], y)
        return _proba_positive(model, x_test), info
    if objective == "R3_three_class":
        y = np.zeros(len(u_train), dtype=int)
        y[u_train > margin] = 1
        y[u_train < -margin] = -1
        info["margin"] = float(margin)
        if len(np.unique(y)) < 2:
            return np.zeros(len(x_test), dtype=float), info
        model = _forest_clf(args.router_estimators, args.router_n_jobs, seed)
        model.fit(x_train, y)
        return _class_score(model, x_test), info
    if objective == "R4_pairwise_margin_weighted":
        ranker = _train_pairwise_ranker(
            x_train,
            u_train,
            seed=seed,
            pair_samples=args.pair_samples,
            margin=margin,
            weighted=True,
        )
        info["margin"] = float(margin)
        return ranker.score(x_test), info
    if objective == "R5_rank_regression":
        order = pd.Series(u_train).rank(method="average", pct=True).to_numpy(float)
        model = _forest_reg(args.router_estimators, args.router_n_jobs, seed)
        model.fit(x_train, order)
        return model.predict(x_test), info
    raise ValueError(f"Unknown objective {objective}")


def run_task(results_dir: Path, task: str, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_meta = _read_json(results_dir / "run_metadata.json")
    labels = _load_labels(results_dir, task)
    preds = pd.read_csv(results_dir / "predictions" / f"{task}_predictions.csv")
    meta, x_ecfp, x_desc, _x3d, manifest = _load_cache(results_dir, task, run_meta)
    task_type = TASK_TYPE[task]
    idx_by_mol = {str(m): i for i, m in enumerate(meta["mol_id"].astype(str))}
    seeds = sorted(preds["seed"].astype(int).unique())
    if args.seeds:
        seeds = [s for s in seeds if s in set(args.seeds)]
    route_frames = []
    diag_rows = []
    feature_diag_rows = []
    score_frames = []
    support_cache: dict[bytes, np.ndarray] = {}

    def support_for(idx: np.ndarray) -> np.ndarray:
        key = np.asarray(idx, dtype=np.int64).tobytes()
        if key not in support_cache:
            support_cache[key] = _support_features(meta, x_ecfp, idx)
        return support_cache[key]

    objectives = [
        "R0_raw_utility",
        "R1_binary",
        "R2_margin_binary",
        "R3_three_class",
        "R4_pairwise_margin_weighted",
        "R5_rank_regression",
    ]
    for seed in seeds:
        print(f"[objective] {task} seed={seed}", flush=True)
        glabel = labels[labels["seed"].astype(int) == seed].copy()
        gpred = preds[preds["seed"].astype(int) == seed].copy()
        if glabel.empty or gpred.empty:
            continue
        label_idx = np.asarray([idx_by_mol[str(m)] for m in glabel["mol_id"].astype(str)], dtype=int)
        test_idx = np.asarray([idx_by_mol[str(m)] for m in gpred["mol_id"].astype(str)], dtype=int)
        y_train = meta.iloc[label_idx]["y"].to_numpy(float)
        y_test = gpred["y"].to_numpy(float)
        pred2d_train = glabel["pred2d_label"].to_numpy(float)
        pred3d_train = glabel["pred3d_label"].to_numpy(float)
        pred2d_test = gpred["pred2d"].to_numpy(float)
        pred3d_test = gpred["pred3d_aug"].to_numpy(float)
        u_train = _utility_from_predictions(task_type, y_train, pred2d_train, pred3d_train)
        u_test = _utility_from_predictions(task_type, y_test, pred2d_test, pred3d_test)
        sigma = float(np.nanstd(u_train))
        margin = max(1e-12, args.pair_margin_frac * sigma)
        train_groups = _feature_groups(meta, x_ecfp, x_desc, manifest, label_idx, pred2d_train, task_type, support_for(label_idx))
        test_groups = _feature_groups(meta, x_ecfp, x_desc, manifest, test_idx, pred2d_test, task_type, support_for(test_idx))
        x_train = train_groups["full_mechanism"]
        x_test = test_groups["full_mechanism"]

        base_scores = {
            "random": np.random.default_rng(seed).normal(size=len(gpred)),
            "uncertainty": gpred["uncertainty_score"].to_numpy(float),
            "flexibility": gpred["flexibility_score"].to_numpy(float),
            "old_voi": gpred["voi_score"].to_numpy(float),
            "oracle": u_test,
        }
        score_cols = dict(base_scores)
        objective_info = {}
        train_score_cache = {}
        for obj in objectives:
            score, info = _train_and_score(obj, x_train, u_train, x_test, seed, args, margin)
            train_score, _ = _train_and_score(obj, x_train, u_train, x_train, seed, args, margin)
            score_cols[obj] = score
            train_score_cache[obj] = train_score
            objective_info[obj] = info

        # Reliability-aware final router: margin-weighted rank utility minus learned harm risk.
        risk_model = _forest_reg(args.router_estimators, args.router_n_jobs, seed + 9917)
        risk_target = np.maximum(-u_train, 0.0)
        risk_model.fit(x_train, risk_target)
        risk_train = risk_model.predict(x_train)
        risk_test = risk_model.predict(x_test)
        utility_train = train_score_cache["R4_pairwise_margin_weighted"]
        utility_test = score_cols["R4_pairwise_margin_weighted"]
        lam_grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
        best_lam = 0.0
        best_ndcg = -np.inf
        utility_train_z = _standardize_by_train(utility_train, utility_train)
        risk_train_z = _standardize_by_train(risk_train, risk_train)
        for lam in lam_grid:
            cand = utility_train_z - lam * risk_train_z
            val = _ndcg_at_budget(u_train, cand, args.main_budget)
            if np.isfinite(val) and val > best_ndcg:
                best_ndcg = float(val)
                best_lam = float(lam)
        score_cols["R6_rank_reliability"] = _standardize_by_train(utility_train, utility_test) - best_lam * _standardize_by_train(risk_train, risk_test)
        objective_info["R6_rank_reliability"] = {"lambda": best_lam, "selected_by": f"oof_ndcg@{args.main_budget:g}"}

        curves = evaluate_routing_curves(task, task_type, y_test, pred2d_test, pred3d_test, score_cols, args.budgets, seed)
        route_frames.append(curves)
        for router, scores in score_cols.items():
            beneficial = u_test > 0.0
            row = {
                "task": task,
                "task_type": task_type,
                "seed": seed,
                "router": router,
                "utility_definition": "logloss_improvement" if task_type == "classification" else "absolute_error_improvement",
                "main_budget": float(args.main_budget),
                "beneficial_rate": float(beneficial.mean()),
                "beneficial_auc": _auc_safe(beneficial, scores),
                "beneficial_ap": _ap_safe(beneficial, scores),
                "precision_at_10": _precision_at_budget(u_test, scores, 10.0),
                "precision_at_20": _precision_at_budget(u_test, scores, 20.0),
                "ndcg_at_10": _ndcg_at_budget(u_test, scores, 10.0),
                "ndcg_at_20": _ndcg_at_budget(u_test, scores, 20.0),
                "utility_spearman": _spearman_safe(u_test, scores),
                "utility_std_train": sigma,
            }
            row.update(objective_info.get(router, {}))
            diag_rows.append(row)
        for group_name in ["chemistry", "chemistry_uncertainty", "chemistry_uncertainty_geometry", "chemistry_uncertainty_support", "full_mechanism"]:
            score, _ = _train_and_score("R4_pairwise_margin_weighted", train_groups[group_name], u_train, test_groups[group_name], seed, args, margin)
            feature_diag_rows.append(
                {
                    "task": task,
                    "task_type": task_type,
                    "seed": seed,
                    "feature_group": group_name,
                    "objective": "R4_pairwise_margin_weighted",
                    "beneficial_auc": _auc_safe(u_test > 0, score),
                    "precision_at_20": _precision_at_budget(u_test, score, 20.0),
                    "ndcg_at_20": _ndcg_at_budget(u_test, score, 20.0),
                    "utility_spearman": _spearman_safe(u_test, score),
                }
            )
        score_frames.append(
            pd.DataFrame(
                {
                    "task": task,
                    "seed": seed,
                    "mol_id": gpred["mol_id"].to_numpy(),
                    "y": y_test,
                    "pred2d": pred2d_test,
                    "pred3d_aug": pred3d_test,
                    "clean_utility": u_test,
                    **{f"score_{k}": v for k, v in score_cols.items()},
                }
            )
        )
    return (
        pd.concat(route_frames, ignore_index=True) if route_frames else pd.DataFrame(),
        pd.DataFrame(diag_rows),
        pd.DataFrame(feature_diag_rows),
        pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame(),
    )


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    ensure_dir(out_dir / "predictions")
    metadata = {
        "started_at": now_iso(),
        "argv": sys.argv,
        "oof_results_dir": args.oof_results_dir,
        "budgets": args.budgets,
        "main_budget": args.main_budget,
        "router_estimators": args.router_estimators,
        "router_n_jobs": args.router_n_jobs,
        "pair_samples": args.pair_samples,
        "pair_margin_frac": args.pair_margin_frac,
        "protocol_note": "Router objective sweep reuses fixed 2D/2D+3D experts and OOF counterfactual labels. Classification router utility is log-loss improvement; task evaluation remains AUROC.",
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    all_curves = []
    all_diag = []
    all_feature_diag = []
    all_big = []
    for path in args.oof_results_dir:
        results_dir = Path(path)
        meta = _read_json(results_dir / "run_metadata.json")
        tasks = list(meta.get("tasks", []))
        if args.tasks:
            tasks = [t for t in tasks if t in set(args.tasks)]
        for task in tasks:
            print(f"[objective] source={results_dir.name} task={task}", flush=True)
            curves, diag, feature_diag, scores = run_task(results_dir, task, args)
            if not curves.empty:
                curves["source_dir"] = str(results_dir)
                curves.to_csv(out_dir / f"routing_curves_{task}.csv", index=False)
                all_curves.append(curves)
                big = _budget_integrated_gain(curves, TASK_TYPE[task])
                big["source_dir"] = str(results_dir)
                big.to_csv(out_dir / f"budget_integrated_gain_{task}.csv", index=False)
                all_big.append(big)
            if not diag.empty:
                diag["source_dir"] = str(results_dir)
                all_diag.append(diag)
            if not feature_diag.empty:
                feature_diag["source_dir"] = str(results_dir)
                all_feature_diag.append(feature_diag)
            if not scores.empty:
                scores.to_csv(out_dir / "predictions" / f"{task}_router_objective_scores.csv", index=False)
    curves = pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()
    diag = pd.concat(all_diag, ignore_index=True) if all_diag else pd.DataFrame()
    feature_diag = pd.concat(all_feature_diag, ignore_index=True) if all_feature_diag else pd.DataFrame()
    big = pd.concat(all_big, ignore_index=True) if all_big else pd.DataFrame()
    if not curves.empty:
        curves.to_csv(out_dir / "routing_curves.csv", index=False)
        _summarize(curves, ["task", "task_type", "router", "budget"], ["primary_value", "call_rate"]).to_csv(
            out_dir / "routing_summary.csv", index=False
        )
    if not diag.empty:
        diag.to_csv(out_dir / "router_diagnostics_long.csv", index=False)
        _summarize(
            diag,
            ["task", "task_type", "router"],
            ["beneficial_auc", "beneficial_ap", "precision_at_10", "precision_at_20", "ndcg_at_10", "ndcg_at_20", "utility_spearman"],
        ).to_csv(out_dir / "router_diagnostics_summary.csv", index=False)
    if not feature_diag.empty:
        feature_diag.to_csv(out_dir / "feature_group_diagnostics_long.csv", index=False)
        _summarize(feature_diag, ["task", "task_type", "feature_group"], ["beneficial_auc", "precision_at_20", "ndcg_at_20", "utility_spearman"]).to_csv(
            out_dir / "feature_group_diagnostics_summary.csv", index=False
        )
    if not big.empty:
        big.to_csv(out_dir / "budget_integrated_gain_long.csv", index=False)
        _summarize(big, ["task", "task_type", "router"], ["BIG"]).to_csv(out_dir / "budget_integrated_gain_summary.csv", index=False)
        _bootstrap_big_delta(
            big,
            "R6_rank_reliability",
            ["random", "uncertainty", "flexibility", "old_voi", "R0_raw_utility", "R2_margin_binary", "R4_pairwise_margin_weighted"],
            args.bootstrap_iters,
            args.bootstrap_seed,
        ).to_csv(out_dir / "big_paired_bootstrap_ci.csv", index=False)
    print(f"[objective] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
