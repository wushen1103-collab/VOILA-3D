from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.fast_screen import _feature_cache_paths
from voila3d.data import scaffold_for_smiles
from voila3d.metrics import primary_metric
from voila3d.routing import evaluate_routing_curves, true_benefit, uncertainty_scores
from voila3d.utils import ensure_dir, now_iso


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--oof-results-dir", action="append", required=True)
    p.add_argument("--out-dir", default="results/if_router_input_ablation_5seed")
    p.add_argument("--budgets", nargs="+", type=float, default=[0, 5, 10, 20, 40, 60, 80, 100])
    p.add_argument("--min-labels", type=int, default=32)
    p.add_argument("--router-estimators", type=int, default=400)
    p.add_argument("--router-n-jobs", type=int, default=8)
    p.add_argument("--router-max-features", default="1.0")
    return p.parse_args()


def _parse_max_features(value: str) -> str | float:
    if value in {"sqrt", "log2", "None"}:
        return None if value == "None" else value
    return float(value)


def _train_ablation_router(x_router: np.ndarray, benefit: np.ndarray, seed: int, n_estimators: int, n_jobs: int, max_features: str | float | None):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=4,
        random_state=seed,
        n_jobs=n_jobs,
        max_features=max_features,
    )
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)
    pipe.fit(x_router, benefit)
    return pipe


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_labels(results_dir: Path, task: str) -> pd.DataFrame:
    parquet = results_dir / f"{task}_voi_labels.parquet"
    csv = results_dir / f"{task}_voi_labels.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing labels for {task} in {results_dir}")


def _load_cache(results_dir: Path, task: str, meta: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    paths = _feature_cache_paths(
        results_dir,
        task,
        meta.get("split", "scaffold_balanced"),
        meta.get("max_mols", 12000),
        meta.get("conformers", 10),
        meta.get("feature3d_set", "usr"),
    )
    missing = [str(p) for p in [paths["meta"], paths["ecfp"], paths["desc"]] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing feature cache files: " + "; ".join(missing))
    return pd.read_csv(paths["meta"]), np.load(paths["ecfp"]), np.load(paths["desc"])


def _safe_sparse_nearest_tanimoto(x_query: np.ndarray, x_ref: np.ndarray, query_global_idx: np.ndarray, ref_global_idx: np.ndarray) -> np.ndarray:
    try:
        from scipy import sparse
    except Exception:
        return np.full(x_query.shape[0], np.nan, dtype=float)

    ref = sparse.csr_matrix(x_ref.astype(bool))
    ref_sum = np.asarray(ref.sum(axis=1)).ravel().astype(float)
    out = np.full(x_query.shape[0], np.nan, dtype=float)
    ref_pos_by_global = {int(idx): pos for pos, idx in enumerate(ref_global_idx)}
    chunk = 512
    for start in range(0, x_query.shape[0], chunk):
        end = min(x_query.shape[0], start + chunk)
        q = sparse.csr_matrix(x_query[start:end].astype(bool))
        q_sum = np.asarray(q.sum(axis=1)).ravel().astype(float)
        inter = (q @ ref.T).toarray().astype(float)
        denom = q_sum[:, None] + ref_sum[None, :] - inter
        sim = np.divide(inter, denom, out=np.zeros_like(inter), where=denom > 0)
        for local, global_idx in enumerate(query_global_idx[start:end]):
            ref_pos = ref_pos_by_global.get(int(global_idx))
            if ref_pos is not None:
                sim[local, ref_pos] = -np.inf
        best = np.max(sim, axis=1)
        best[~np.isfinite(best)] = np.nan
        out[start:end] = best
    return out


def _support_features(meta: pd.DataFrame, x_ecfp: np.ndarray, query_idx: np.ndarray) -> np.ndarray:
    scaffolds = meta["canonical_smiles"].map(scaffold_for_smiles).fillna("").astype(str)
    train_idx = np.flatnonzero(meta["split"].to_numpy() == "train")
    train_scaffolds = scaffolds.iloc[train_idx]
    counts = train_scaffolds.value_counts().to_dict()
    support = np.asarray([np.log1p(counts.get(scaffolds.iloc[int(i)], 0)) for i in query_idx], dtype=np.float32)
    nn_sim = _safe_sparse_nearest_tanimoto(x_ecfp[query_idx], x_ecfp[train_idx], query_idx, train_idx)
    ood = 1.0 - nn_sim
    return np.column_stack([support, ood]).astype(np.float32)


def _feature_sets(
    x_ecfp: np.ndarray,
    x_desc: np.ndarray,
    pred2d: np.ndarray,
    task_type: str,
    support_ood: np.ndarray,
) -> dict[str, np.ndarray]:
    pred = pred2d[:, None].astype(np.float32)
    unc = uncertainty_scores(task_type, pred2d)[:, None].astype(np.float32)
    return {
        "ecfp_only": x_ecfp.astype(np.float32),
        "uncertainty_only": np.hstack([pred, unc]).astype(np.float32),
        "ecfp_uncertainty": np.hstack([x_ecfp, pred, unc]).astype(np.float32),
        "uncertainty_support_ood": np.hstack([pred, unc, support_ood]).astype(np.float32),
        "desc_uncertainty_support": np.hstack([x_desc, pred, unc, support_ood]).astype(np.float32),
        "full_ecfp_desc_uncertainty_support": np.hstack([x_ecfp, x_desc, pred, unc, support_ood]).astype(np.float32),
    }


def _summarize(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, task_type, feature_set, budget), g in curves.groupby(["task", "task_type", "router_feature_set", "budget"]):
        pmet = primary_metric(task_type)
        vals = pd.to_numeric(g["primary_value"], errors="coerce")
        rows.append(
            {
                "task": task,
                "task_type": task_type,
                "router_feature_set": feature_set,
                "budget": float(budget),
                "primary_metric": pmet,
                "primary_mean": float(vals.mean()),
                "primary_std": float(vals.std(ddof=0)),
                "primary_mean_std": f"{vals.mean():.6f} +/- {vals.std(ddof=0):.6f}",
                "call_rate_mean": float(g["call_rate"].mean()),
                "n_seeds": int(g["seed"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["task", "router_feature_set", "budget"])


def run_dir(
    results_dir: Path,
    out_pred_dir: Path,
    budgets: list[float],
    min_labels: int,
    router_estimators: int,
    router_n_jobs: int,
    router_max_features: str | float | None,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[dict]]:
    meta_json = _load_json(results_dir / "run_metadata.json")
    tasks = meta_json.get("tasks", [])
    curve_frames: list[pd.DataFrame] = []
    pred_frames: list[pd.DataFrame] = []
    label_rows: list[dict] = []
    for task in tasks:
        print(f"[ablation] {results_dir.name}/{task}", flush=True)
        pred_path = results_dir / "predictions" / f"{task}_predictions.csv"
        if not pred_path.exists():
            continue
        meta, x_ecfp, x_desc = _load_cache(results_dir, task, meta_json)
        labels = _load_labels(results_dir, task)
        preds = pd.read_csv(pred_path)
        idx_by_mol = {mol_id: idx for idx, mol_id in enumerate(meta["mol_id"].astype(str))}
        task_type = str(preds["task"].iloc[0]) if "task_type" not in preds.columns else str(preds["task_type"].iloc[0])
        if task in {"ESOL", "FreeSolv", "Lipophilicity"}:
            task_type = "regression"
        elif task in {"BBBP", "BACE", "HIV"}:
            task_type = "classification"
        y_test_by_seed = {}
        support_cache: dict[bytes, np.ndarray] = {}

        def support_for(query_idx: np.ndarray) -> np.ndarray:
            query_idx = np.asarray(query_idx, dtype=np.int64)
            key = query_idx.tobytes()
            cached = support_cache.get(key)
            if cached is None:
                cached = _support_features(meta, x_ecfp, query_idx)
                support_cache[key] = cached
            return cached

        for seed, gpred in preds.groupby("seed"):
            seed = int(seed)
            print(f"[ablation] {task} seed={seed}", flush=True)
            glabel = labels[labels["seed"].astype(int) == seed].copy()
            if "pred2d_label" not in glabel or len(glabel) < min_labels:
                continue
            train_idx = np.asarray([idx_by_mol[str(m)] for m in glabel["mol_id"].astype(str)], dtype=int)
            test_idx = np.asarray([idx_by_mol[str(m)] for m in gpred["mol_id"].astype(str)], dtype=int)
            train_support = support_for(train_idx)
            test_support = support_for(test_idx)
            train_sets = _feature_sets(
                x_ecfp[train_idx],
                x_desc[train_idx],
                glabel["pred2d_label"].to_numpy(float),
                task_type,
                train_support,
            )
            test_sets = _feature_sets(
                x_ecfp[test_idx],
                x_desc[test_idx],
                gpred["pred2d"].to_numpy(float),
                task_type,
                test_support,
            )
            y = gpred["y"].to_numpy(float)
            pred2d = gpred["pred2d"].to_numpy(float)
            pred3d = gpred["pred3d_aug"].to_numpy(float)
            benefit = glabel["benefit"].to_numpy(float)
            score_cols = {
                "random": np.random.default_rng(seed).normal(size=len(gpred)),
                "uncertainty": gpred["uncertainty_score"].to_numpy(float),
                "oracle": true_benefit(
                    task_type,
                    y,
                    pred2d,
                    pred3d,
                    classification_mode=str(gpred["classification_benefit"].iloc[0]) if task_type == "classification" else "logloss",
                ),
            }
            pred_out = {
                "task": task,
                "seed": seed,
                "mol_id": gpred["mol_id"].to_numpy(),
                "y": y,
                "pred2d": pred2d,
                "pred3d_aug": pred3d,
            }
            for name in train_sets:
                router = _train_ablation_router(
                    train_sets[name],
                    benefit,
                    seed=seed,
                    n_estimators=router_estimators,
                    n_jobs=router_n_jobs,
                    max_features=router_max_features,
                )
                score_cols[f"ablate_{name}"] = router.predict(test_sets[name])
                pred_out[f"score_{name}"] = score_cols[f"ablate_{name}"]
            curves = evaluate_routing_curves(task, task_type, y, pred2d, pred3d, score_cols, budgets, seed)
            curves["router_feature_set"] = curves["router"].str.replace("ablate_", "", regex=False)
            curves.loc[curves["router"].isin(["random", "uncertainty", "oracle"]), "router_feature_set"] = curves["router"]
            curve_frames.append(curves)
            pred_frames.append(pd.DataFrame(pred_out))
            label_rows.append(
                {
                    "source_dir": str(results_dir),
                    "task": task,
                    "seed": seed,
                    "task_type": task_type,
                    "n_router_labels": int(len(glabel)),
                    "beneficial_rate": float((benefit > 0).mean()),
                    "label_source": str(glabel["label_source"].iloc[0]) if "label_source" in glabel else "",
                }
            )
        task_preds = [p for p in pred_frames if not p.empty and str(p["task"].iloc[0]) == task]
        if task_preds:
            pd.concat(task_preds, ignore_index=True).to_csv(out_pred_dir / f"{task}_router_ablation_predictions.csv", index=False)
    return curve_frames, pred_frames, label_rows


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    pred_dir = ensure_dir(out_dir / "predictions")
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "started_at": now_iso(),
                "argv": sys.argv,
                "oof_results_dir": args.oof_results_dir,
                "budgets": args.budgets,
                "router_estimators": args.router_estimators,
                "router_n_jobs": args.router_n_jobs,
                "router_max_features": args.router_max_features,
                "protocol_note": "Router input ablation retrains only the acquisition model on OOF counterfactual utility labels; 2D/3D experts and test split are held fixed.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    curve_frames: list[pd.DataFrame] = []
    label_rows: list[dict] = []
    router_max_features = _parse_max_features(str(args.router_max_features))
    for path in args.oof_results_dir:
        curves, _, labels = run_dir(
            Path(path),
            pred_dir,
            args.budgets,
            args.min_labels,
            args.router_estimators,
            args.router_n_jobs,
            router_max_features,
        )
        curve_frames.extend(curves)
        label_rows.extend(labels)
    if curve_frames:
        curves = pd.concat(curve_frames, ignore_index=True)
        curves.to_csv(out_dir / "routing_curves.csv", index=False)
        _summarize(curves).to_csv(out_dir / "routing_summary.csv", index=False)
    pd.DataFrame(label_rows).to_csv(out_dir / "label_summary.csv", index=False)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
