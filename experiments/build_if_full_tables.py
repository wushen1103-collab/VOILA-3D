from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.fast_screen import _feature_cache_paths
from voila3d.data import scaffold_for_smiles
from voila3d.features import DESCRIPTOR_NAMES
from voila3d.metrics import classification_metrics, higher_is_better, primary_metric, regression_metrics
from voila3d.routing import route_predictions, true_benefit
from voila3d.utils import ensure_dir, now_iso


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--oof-results-dir", action="append", required=True)
    p.add_argument("--fusion-dir", default="results/if_fusion_baselines_usr_5seed")
    p.add_argument("--router-ablation-dir", default="results/if_router_input_ablation_5seed")
    p.add_argument("--noisy3d-dir", default="results/if_noisy3d_stress_usr_5seed")
    p.add_argument("--strict-sota-csv", default="results/comparisons/final_candidate_strict_sota_aware_5seed.csv")
    p.add_argument("--external-method-matrix", default="results/review_audit/modern_external_method_matrix.csv")
    p.add_argument("--special-budget-csv", default="results/review_audit/special_missing_modality_budget.csv")
    p.add_argument("--label-efficiency-csv", default="results/review_audit/special_label_efficiency.csv")
    p.add_argument("--resource-monitor-csv", default="logs/if_resource_monitor.csv")
    p.add_argument("--out-dir", default="results/if_full_audit")
    p.add_argument("--main-budget", type=float, default=20.0)
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=1701)
    return p.parse_args()


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def _read_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "(empty)"
    out = df.copy()
    if max_rows is not None and len(out) > max_rows:
        out = out.head(max_rows)
    try:
        return out.to_markdown(index=False)
    except Exception:
        return out.to_csv(index=False)


def _positive_delta(metric: str, reference: float, candidate: float) -> float:
    return candidate - reference if higher_is_better(metric) else reference - candidate


def _eval(task_type: str, y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if task_type == "classification":
        return classification_metrics(y, np.clip(pred, 1e-6, 1 - 1e-6))
    return regression_metrics(y, pred)


def _select_top(scores: np.ndarray, budget: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    k = int(round(n * budget / 100.0))
    selected = np.zeros(n, dtype=bool)
    if k <= 0:
        return selected
    if k >= n:
        selected[:] = True
        return selected
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    selected[idx] = True
    return selected


def _load_oof_summaries(paths: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    route_summary_frames = []
    route_curve_frames = []
    metric_frames = []
    metas = []
    for item in paths:
        d = Path(item)
        meta = _read_json(d / "run_metadata.json")
        meta["results_dir"] = str(d)
        metas.append(meta)
        summary = _read_csv(d / "routing_summary.csv")
        curves = _read_csv(d / "routing_curves.csv")
        metrics = _read_csv(d / "metrics_long.csv")
        if summary.empty and not curves.empty:
            summary = _summarize_route_curves(curves)
        for frame in (summary, curves, metrics):
            if not frame.empty:
                frame["source_dir"] = str(d)
        if not summary.empty:
            route_summary_frames.append(summary)
        if not curves.empty:
            route_curve_frames.append(curves)
        if not metrics.empty:
            metric_frames.append(metrics)
    return (
        pd.concat(route_summary_frames, ignore_index=True) if route_summary_frames else pd.DataFrame(),
        pd.concat(route_curve_frames, ignore_index=True) if route_curve_frames else pd.DataFrame(),
        pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame(),
        metas,
    )


def _summarize_route_curves(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, task_type, router, budget), g in curves.groupby(["task", "task_type", "router", "budget"]):
        vals = pd.to_numeric(g["primary_value"], errors="coerce")
        rows.append(
            {
                "task": task,
                "task_type": task_type,
                "router": router,
                "budget": float(budget),
                "primary_metric": primary_metric(str(task_type)),
                "primary_mean": float(vals.mean()),
                "primary_std": float(vals.std(ddof=0)),
                "call_rate_mean": float(pd.to_numeric(g["call_rate"], errors="coerce").mean()),
                "n_seeds": int(g["seed"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def build_acquisition_tables(route_summary: pd.DataFrame, main_budget: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if route_summary.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    budget_curve = route_summary.sort_values(["task", "router", "budget"]).copy()
    rows = []
    recovery_rows = []
    for task, gtask in route_summary.groupby("task"):
        metric = str(gtask["primary_metric"].iloc[0])
        hib = higher_is_better(metric)
        base = gtask[(gtask["budget"] == 0.0) & (gtask["router"] == "voi_router")]
        if base.empty:
            base = gtask[gtask["budget"] == 0.0].head(1)
        base_val = float(base["primary_mean"].iloc[0]) if not base.empty else np.nan
        g20 = gtask[np.isclose(gtask["budget"].astype(float), main_budget)].copy()
        if g20.empty:
            continue
        random_val = float(g20[g20["router"] == "random"]["primary_mean"].iloc[0]) if (g20["router"] == "random").any() else np.nan
        uncert_val = (
            float(g20[g20["router"] == "uncertainty"]["primary_mean"].iloc[0])
            if (g20["router"] == "uncertainty").any()
            else np.nan
        )
        oracle_val = float(g20[g20["router"] == "oracle"]["primary_mean"].iloc[0]) if (g20["router"] == "oracle").any() else np.nan
        oracle_gain = _positive_delta(metric, base_val, oracle_val) if np.isfinite(base_val) and np.isfinite(oracle_val) else np.nan
        for _, row in g20.iterrows():
            val = float(row["primary_mean"])
            delta_base = _positive_delta(metric, base_val, val) if np.isfinite(base_val) else np.nan
            delta_rand = _positive_delta(metric, random_val, val) if np.isfinite(random_val) else np.nan
            delta_unc = _positive_delta(metric, uncert_val, val) if np.isfinite(uncert_val) else np.nan
            recovery = delta_base / oracle_gain if oracle_gain and np.isfinite(oracle_gain) and abs(oracle_gain) > 1e-12 else np.nan
            rows.append(
                {
                    "task": task,
                    "task_type": row["task_type"],
                    "router": row["router"],
                    "budget": float(row["budget"]),
                    "primary_metric": metric,
                    "primary_mean": val,
                    "primary_std": float(row["primary_std"]),
                    "primary_mean_std": f"{val:.6f} +/- {float(row['primary_std']):.6f}",
                    "delta_vs_2d_positive": delta_base,
                    "delta_vs_random_positive": delta_rand,
                    "delta_vs_uncertainty_positive": delta_unc,
                    "oracle_recovery_fraction": recovery,
                    "call_rate_mean": float(row["call_rate_mean"]),
                    "n_seeds": int(row["n_seeds"]),
                }
            )
        recovery_rows.append(
            {
                "task": task,
                "primary_metric": metric,
                "higher_is_better": bool(hib),
                "budget": main_budget,
                "all_2d": base_val,
                "oracle": oracle_val,
                "oracle_gain_positive": oracle_gain,
                "voi_router": float(g20[g20["router"] == "voi_router"]["primary_mean"].iloc[0])
                if (g20["router"] == "voi_router").any()
                else np.nan,
                "random": random_val,
                "uncertainty": uncert_val,
            }
        )
    acquisition20 = pd.DataFrame(rows).sort_values(["task", "router"]) if rows else pd.DataFrame()
    oracle_recovery = pd.DataFrame(recovery_rows).sort_values("task") if recovery_rows else pd.DataFrame()
    if not oracle_recovery.empty:
        oracle_recovery["voi_oracle_recovery_fraction"] = [
            _positive_delta(str(r.primary_metric), float(r.all_2d), float(r.voi_router)) / float(r.oracle_gain_positive)
            if np.isfinite(float(r.oracle_gain_positive)) and abs(float(r.oracle_gain_positive)) > 1e-12
            else np.nan
            for r in oracle_recovery.itertuples(index=False)
        ]
    return budget_curve, acquisition20, oracle_recovery


def _routed_prediction(pred2d: np.ndarray, pred3d: np.ndarray, scores: np.ndarray, budget: float) -> np.ndarray:
    pred, _ = route_predictions(pred2d, pred3d, scores, budget)
    return pred


def _metric_delta(task_type: str, y: np.ndarray, pred_ref: np.ndarray, pred_cand: np.ndarray) -> float:
    metric = primary_metric(task_type)
    ref = _eval(task_type, y, pred_ref)[metric]
    cand = _eval(task_type, y, pred_cand)[metric]
    return _positive_delta(metric, float(ref), float(cand))


def _bootstrap_delta(
    groups: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    iters: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    if not groups:
        return np.nan, np.nan, np.nan
    obs = np.nanmean([_metric_delta(task_type, y, pred_ref, pred_cand) for y, pred_ref, pred_cand, task_type in groups])
    boots = []
    for _ in range(iters):
        vals = []
        picked = rng.integers(0, len(groups), size=len(groups))
        for pos in picked:
            y, pred_ref, pred_cand, task_type = groups[int(pos)]
            n = len(y)
            idx = rng.integers(0, n, size=n)
            try:
                vals.append(_metric_delta(task_type, y[idx], pred_ref[idx], pred_cand[idx]))
            except Exception:
                continue
        if vals:
            boots.append(float(np.nanmean(vals)))
    if not boots:
        return float(obs), np.nan, np.nan
    return float(obs), float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def build_prediction_level_tables(paths: list[str], main_budget: float, boot_iters: int, boot_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    diag_rows = []
    boot_rows = []
    for item in paths:
        d = Path(item)
        meta = _read_json(d / "run_metadata.json")
        for task in meta.get("tasks", []):
            pred_path = d / "predictions" / f"{task}_predictions.csv"
            if not pred_path.exists():
                continue
            preds = pd.read_csv(pred_path)
            if task in {"ESOL", "FreeSolv", "Lipophilicity"}:
                task_type = "regression"
            else:
                task_type = "classification"
            random_groups = []
            uncertainty_groups = []
            flexibility_groups = []
            for seed, g in preds.groupby("seed"):
                seed = int(seed)
                y = g["y"].to_numpy(float)
                pred2d = g["pred2d"].to_numpy(float)
                pred3d = g["pred3d_aug"].to_numpy(float)
                voi = g["voi_score"].to_numpy(float)
                oracle = g["oracle_benefit"].to_numpy(float)
                beneficial = oracle > 0
                try:
                    router_auc = roc_auc_score(beneficial.astype(int), voi) if len(np.unique(beneficial)) == 2 else np.nan
                except Exception:
                    router_auc = np.nan
                top20 = _select_top(voi, main_budget)
                precision20 = float(beneficial[top20].mean()) if top20.any() else np.nan
                positive_rate = float(beneficial.mean())
                corr = pd.Series(voi).corr(pd.Series(oracle), method="spearman")
                pred_voi = _routed_prediction(pred2d, pred3d, voi, main_budget)
                pred_random = _routed_prediction(pred2d, pred3d, np.random.default_rng(seed).normal(size=len(g)), main_budget)
                pred_unc = _routed_prediction(pred2d, pred3d, g["uncertainty_score"].to_numpy(float), main_budget)
                pred_flex = _routed_prediction(pred2d, pred3d, g["flexibility_score"].to_numpy(float), main_budget)
                random_groups.append((y, pred_random, pred_voi, task_type))
                uncertainty_groups.append((y, pred_unc, pred_voi, task_type))
                flexibility_groups.append((y, pred_flex, pred_voi, task_type))
                diag_rows.append(
                    {
                        "task": task,
                        "task_type": task_type,
                        "seed": seed,
                        "budget": main_budget,
                        "router_auc_for_beneficial_3d": router_auc,
                        "precision_at_budget": precision20,
                        "beneficial_rate": positive_rate,
                        "precision_lift_vs_base_rate": precision20 / positive_rate if positive_rate > 0 else np.nan,
                        "utility_spearman": float(corr) if corr is not None and np.isfinite(corr) else np.nan,
                        "delta_vs_random_positive": _metric_delta(task_type, y, pred_random, pred_voi),
                        "delta_vs_uncertainty_positive": _metric_delta(task_type, y, pred_unc, pred_voi),
                        "delta_vs_flexibility_positive": _metric_delta(task_type, y, pred_flex, pred_voi),
                        "source_dir": str(d),
                    }
                )
            for baseline, groups in [
                ("random", random_groups),
                ("uncertainty", uncertainty_groups),
                ("flexibility", flexibility_groups),
            ]:
                stable_offset = int(hashlib.md5(f"{task}:{baseline}".encode("utf-8")).hexdigest()[:8], 16) % 100000
                obs, lo, hi = _bootstrap_delta(groups, boot_iters, boot_seed + stable_offset)
                boot_rows.append(
                    {
                        "task": task,
                        "task_type": task_type,
                        "budget": main_budget,
                        "comparison": f"voi_router_minus_{baseline}",
                        "positive_delta_mean": obs,
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "n_bootstrap": int(boot_iters),
                        "source_dir": str(d),
                    }
                )
    diag = pd.DataFrame(diag_rows)
    if not diag.empty:
        summary = []
        for (task, task_type), g in diag.groupby(["task", "task_type"]):
            row = {
                "task": task,
                "task_type": task_type,
                "budget": main_budget,
                "n_seeds": int(g["seed"].nunique()),
            }
            for col in [
                "router_auc_for_beneficial_3d",
                "precision_at_budget",
                "beneficial_rate",
                "precision_lift_vs_base_rate",
                "utility_spearman",
                "delta_vs_random_positive",
                "delta_vs_uncertainty_positive",
                "delta_vs_flexibility_positive",
            ]:
                vals = pd.to_numeric(g[col], errors="coerce")
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_std"] = float(vals.std(ddof=0))
            summary.append(row)
        diag = pd.DataFrame(summary).sort_values("task")
    return diag, pd.DataFrame(boot_rows).sort_values(["task", "comparison"]) if boot_rows else pd.DataFrame()


def _load_cache_for_dir(results_dir: Path, task: str, run_meta: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    paths = _feature_cache_paths(
        results_dir,
        task,
        run_meta.get("split", "scaffold_balanced"),
        run_meta.get("max_mols", 12000),
        run_meta.get("conformers", 10),
        run_meta.get("feature3d_set", "usr"),
    )
    if not paths["meta"].exists() or not paths["ecfp"].exists() or not paths["desc"].exists():
        raise FileNotFoundError(f"Missing cache files for {task} in {results_dir}")
    return pd.read_csv(paths["meta"]), np.load(paths["ecfp"]), np.load(paths["desc"])


def _nearest_tanimoto(query: np.ndarray, ref: np.ndarray, query_idx: np.ndarray, ref_idx: np.ndarray) -> np.ndarray:
    try:
        from scipy import sparse
    except Exception:
        return np.full(query.shape[0], np.nan)
    ref_bool = sparse.csr_matrix(ref.astype(bool))
    query_bool = sparse.csr_matrix(query.astype(bool))
    ref_sum = np.asarray(ref_bool.sum(axis=1)).ravel().astype(float)
    q_sum = np.asarray(query_bool.sum(axis=1)).ravel().astype(float)
    ref_pos = {int(idx): pos for pos, idx in enumerate(ref_idx)}
    out = np.full(query.shape[0], np.nan, dtype=float)
    for start in range(0, query.shape[0], 512):
        end = min(query.shape[0], start + 512)
        inter = (query_bool[start:end] @ ref_bool.T).toarray().astype(float)
        denom = q_sum[start:end, None] + ref_sum[None, :] - inter
        sim = np.divide(inter, denom, out=np.zeros_like(inter), where=denom > 0)
        for local, global_idx in enumerate(query_idx[start:end]):
            pos = ref_pos.get(int(global_idx))
            if pos is not None:
                sim[local, pos] = -np.inf
        best = np.max(sim, axis=1)
        best[~np.isfinite(best)] = np.nan
        out[start:end] = best
    return out


def build_chemical_space_table(paths: list[str], main_budget: float) -> pd.DataFrame:
    rows = []
    for item in paths:
        d = Path(item)
        run_meta = _read_json(d / "run_metadata.json")
        for task in run_meta.get("tasks", []):
            pred_path = d / "predictions" / f"{task}_predictions.csv"
            if not pred_path.exists():
                continue
            meta, x_ecfp, x_desc = _load_cache_for_dir(d, task, run_meta)
            idx_by_mol = {mol_id: i for i, mol_id in enumerate(meta["mol_id"].astype(str))}
            train_idx = np.flatnonzero(meta["split"].to_numpy() == "train")
            scaffold_counts = meta.iloc[train_idx]["canonical_smiles"].map(scaffold_for_smiles).value_counts().to_dict()
            preds = pd.read_csv(pred_path)
            nearest_cache: dict[bytes, np.ndarray] = {}
            for seed, g in preds.groupby("seed"):
                idx = np.asarray([idx_by_mol[str(m)] for m in g["mol_id"].astype(str)], dtype=int)
                idx_key = idx.astype(np.int64).tobytes()
                nn_sim = nearest_cache.get(idx_key)
                if nn_sim is None:
                    nn_sim = _nearest_tanimoto(x_ecfp[idx], x_ecfp[train_idx], idx, train_idx)
                    nearest_cache[idx_key] = nn_sim
                ood = 1.0 - nn_sim
                scaffold_support = np.asarray(
                    [math.log1p(scaffold_counts.get(scaffold_for_smiles(meta["canonical_smiles"].iloc[int(i)]), 0)) for i in idx],
                    dtype=float,
                )
                selected = _select_top(g["voi_score"].to_numpy(float), main_budget)
                candidates: dict[str, np.ndarray] = {
                    "uncertainty_score": g["uncertainty_score"].to_numpy(float),
                    "flexibility_score": g["flexibility_score"].to_numpy(float),
                    "oracle_benefit": g["oracle_benefit"].to_numpy(float),
                    "ood_distance": ood,
                    "nearest_train_tanimoto": nn_sim,
                    "log_scaffold_train_count": scaffold_support,
                }
                for pos, name in enumerate(DESCRIPTOR_NAMES):
                    candidates[name] = x_desc[idx, pos].astype(float)
                for feature, values in candidates.items():
                    score = g["voi_score"].to_numpy(float)
                    corr = pd.Series(score).corr(pd.Series(values), method="spearman")
                    rows.append(
                        {
                            "task": task,
                            "seed": int(seed),
                            "budget": main_budget,
                            "feature": feature,
                            "spearman_with_voi_score": float(corr) if corr is not None and np.isfinite(corr) else np.nan,
                            "selected_mean": float(np.nanmean(values[selected])) if selected.any() else np.nan,
                            "not_selected_mean": float(np.nanmean(values[~selected])) if (~selected).any() else np.nan,
                            "selected_minus_not_selected": float(np.nanmean(values[selected]) - np.nanmean(values[~selected]))
                            if selected.any() and (~selected).any()
                            else np.nan,
                        }
                    )
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    summary = []
    for (task, feature), g in raw.groupby(["task", "feature"]):
        summary.append(
            {
                "task": task,
                "feature": feature,
                "budget": main_budget,
                "spearman_mean": float(g["spearman_with_voi_score"].mean()),
                "spearman_std": float(g["spearman_with_voi_score"].std(ddof=0)),
                "selected_minus_not_selected_mean": float(g["selected_minus_not_selected"].mean()),
                "selected_mean": float(g["selected_mean"].mean()),
                "not_selected_mean": float(g["not_selected_mean"].mean()),
                "n_seeds": int(g["seed"].nunique()),
            }
        )
    return pd.DataFrame(summary).sort_values(["task", "feature"])


def build_cost_summary(paths: list[str], resource_monitor: str | Path) -> pd.DataFrame:
    rows = []
    for item in paths:
        d = Path(item)
        timing = _read_csv(d / "timing_long.csv")
        manifest = _read_csv(d / "conformer_manifest.csv")
        if timing.empty and manifest.empty:
            continue
        tasks = sorted(set(timing.get("task", pd.Series(dtype=str)).dropna()).union(set(manifest.get("task", pd.Series(dtype=str)).dropna())))
        for task in tasks:
            t = timing[timing["task"] == task] if not timing.empty and "task" in timing else pd.DataFrame()
            m = manifest[manifest["task"] == task] if not manifest.empty and "task" in manifest else pd.DataFrame()
            rows.append(
                {
                    "source_dir": str(d),
                    "task": task,
                    "n_timing_seeds": int(t["seed"].nunique()) if not t.empty and "seed" in t else 0,
                    "feature_cache_hit": bool(t["feature_cache_hit"].all()) if not t.empty and "feature_cache_hit" in t else np.nan,
                    "mean_feature_elapsed_sec": float(t["feature_elapsed_sec"].mean()) if "feature_elapsed_sec" in t else np.nan,
                    "mean_fit2d_sec": float(t["fit2d_sec"].mean()) if "fit2d_sec" in t else np.nan,
                    "mean_fit3d_aug_sec": float(t["fit3d_aug_sec"].mean()) if "fit3d_aug_sec" in t else np.nan,
                    "mean_oof_label_sec": float(t["oof_label_sec"].mean()) if "oof_label_sec" in t else np.nan,
                    "mean_router_fit_sec": float(t["router_fit_sec"].mean()) if "router_fit_sec" in t else np.nan,
                    "mean_seed_total_sec": float(t["seed_total_sec"].mean()) if "seed_total_sec" in t else np.nan,
                    "conformer_success_rate": float(m["success"].astype(bool).mean()) if not m.empty and "success" in m else np.nan,
                    "mean_conformer_elapsed_sec": float(m["elapsed_sec"].mean()) if not m.empty and "elapsed_sec" in m else np.nan,
                    "p95_conformer_elapsed_sec": float(m["elapsed_sec"].quantile(0.95)) if not m.empty and "elapsed_sec" in m else np.nan,
                    "failed_conformer_rate": float((~m["success"].astype(bool)).mean()) if not m.empty and "success" in m else np.nan,
                }
            )
    cost = pd.DataFrame(rows)
    monitor = _read_csv(resource_monitor)
    if not cost.empty and not monitor.empty:
        gpu = monitor[monitor["kind"].astype(str) == "gpu"]
        proc = monitor[monitor["kind"].astype(str) == "process"]
        cost.attrs["resource_summary"] = {
            "max_gpu_memory_mb": float(pd.to_numeric(gpu.get("gpu_memory_mb"), errors="coerce").max()) if not gpu.empty else np.nan,
            "max_gpu_util_pct": float(pd.to_numeric(gpu.get("gpu_util_pct"), errors="coerce").max()) if not gpu.empty else np.nan,
            "max_process_cpu_pct": float(pd.to_numeric(proc.get("pcpu"), errors="coerce").max()) if not proc.empty else np.nan,
            "max_process_rss_gb": float(pd.to_numeric(proc.get("rss_kb"), errors="coerce").max() / 1024.0 / 1024.0) if not proc.empty else np.nan,
        }
    return cost


def build_fusion_tables(fusion_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = _read_csv(Path(fusion_dir) / "method_summary.csv")
    if summary.empty:
        return summary, pd.DataFrame()
    best_rows = []
    for task, g in summary.groupby("task"):
        metric = str(g["primary_metric"].iloc[0])
        idx = g["primary_mean"].idxmax() if higher_is_better(metric) else g["primary_mean"].idxmin()
        best = g.loc[idx]
        base = g[g["method"] == "2d_only"]
        base_val = float(base["primary_mean"].iloc[0]) if not base.empty else np.nan
        best_rows.append(
            {
                "task": task,
                "task_type": best["task_type"],
                "primary_metric": metric,
                "best_fusion_method": best["method"],
                "best_fusion_mean": float(best["primary_mean"]),
                "best_fusion_std": float(best["primary_std"]),
                "best_fusion_mean_std": best.get("primary_mean_std", ""),
                "delta_vs_2d_positive": _positive_delta(metric, base_val, float(best["primary_mean"])) if np.isfinite(base_val) else np.nan,
                "n_seeds": int(best["n_seeds"]),
            }
        )
    return summary.sort_values(["task", "method"]), pd.DataFrame(best_rows).sort_values("task")


def build_noisy3d_tables(noisy_dir: str | Path, main_budget: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    noisy_dir = Path(noisy_dir)
    summary = _read_csv(noisy_dir / "method_summary.csv")
    routing = _read_csv(noisy_dir / "routing_summary.csv")
    if summary.empty:
        return summary, pd.DataFrame()
    rows = []
    methods = [
        "all_2d",
        "all_3d_noisy_aug",
        "voi_router_fixed_budget_20",
        "voi_positive_threshold_20",
        "voi_reliability_penalized_positive_20",
    ]
    for (task, noise), g in summary.groupby(["task", "noise_level"]):
        base = g[g["method"] == "all_2d"]
        if base.empty:
            continue
        task_type = str(base["task_type"].iloc[0])
        metric = str(base["primary_metric"].iloc[0])
        base_val = float(base["primary_mean"].iloc[0])
        for method in methods:
            sub = g[g["method"] == method]
            if sub.empty:
                continue
            val = float(sub["primary_mean"].iloc[0])
            rows.append(
                {
                    "task": task,
                    "task_type": task_type,
                    "noise_level": float(noise),
                    "method": method,
                    "primary_metric": metric,
                    "primary_mean": val,
                    "primary_std": float(sub["primary_std"].iloc[0]),
                    "primary_mean_std": sub["primary_mean_std"].iloc[0],
                    "call_rate_mean": float(sub["call_rate_mean"].iloc[0]),
                    "delta_vs_2d_positive": _positive_delta(metric, base_val, val),
                    "n_seeds": int(sub["n_seeds"].iloc[0]),
                }
            )
    compact = pd.DataFrame(rows).sort_values(["task", "noise_level", "method"]) if rows else pd.DataFrame()
    if not routing.empty:
        routing = routing[np.isclose(pd.to_numeric(routing["budget"], errors="coerce"), main_budget)].copy()
    return compact, routing


def build_checklist(
    route_summary: pd.DataFrame,
    fusion_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    external_matrix: pd.DataFrame,
    strict_sota: pd.DataFrame,
    special_budget: pd.DataFrame,
    label_eff: pd.DataFrame,
    router_ablation: pd.DataFrame,
    noisy3d: pd.DataFrame,
    metas: list[dict],
) -> pd.DataFrame:
    tasks = sorted(set(route_summary.get("task", pd.Series(dtype=str))).union(set(strict_sota.get("task", pd.Series(dtype=str)))))
    routers = set(route_summary.get("router", pd.Series(dtype=str)).astype(str))
    budgets = set(float(x) for x in route_summary.get("budget", pd.Series(dtype=float)).dropna().unique())
    all_oof = all(str(meta.get("router_label_source")) == "oof" for meta in metas) if metas else False
    seed_ok = bool((route_summary.get("n_seeds", pd.Series(dtype=int)) >= 5).all()) if not route_summary.empty else False
    method_count = int(external_matrix["method"].nunique()) if "method" in external_matrix else len(external_matrix)
    required_budgets = {0.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0}
    rows = [
        {
            "requirement": "Sample-level learned 3D acquisition router trained from OOF counterfactual utility",
            "status": "PASS" if all_oof and len(tasks) >= 6 else "PARTIAL",
            "evidence": f"{len(tasks)} tasks; router_label_source={sorted(set(str(m.get('router_label_source')) for m in metas))}",
        },
        {
            "requirement": "Full cost-performance budget curve",
            "status": "PASS" if required_budgets.issubset(budgets) else "PARTIAL",
            "evidence": f"budgets={sorted(budgets)}",
        },
        {
            "requirement": "Acquisition baselines: Random, uncertainty, structure/flexibility, Oracle",
            "status": "PASS" if {"random", "uncertainty", "flexibility", "oracle", "voi_router"}.issubset(routers) else "PARTIAL",
            "evidence": ", ".join(sorted(routers)),
        },
        {
            "requirement": "2D/3D/fusion/MoE/modality-dropout baselines",
            "status": "PASS" if not fusion_summary.empty and fusion_summary["method"].nunique() >= 8 else "PARTIAL",
            "evidence": f"{fusion_summary['method'].nunique() if not fusion_summary.empty else 0} fusion-family methods",
        },
        {
            "requirement": "Real cost and resource accounting",
            "status": "PASS" if not cost_summary.empty else "PARTIAL",
            "evidence": f"{len(cost_summary)} task-source cost rows",
        },
        {
            "requirement": "Low-label/cold-start and missing-modality special scenarios",
            "status": "PASS" if not special_budget.empty and not label_eff.empty else "PARTIAL",
            "evidence": f"missing_modality_rows={len(special_budget)}, label_efficiency_rows={len(label_eff)}",
        },
        {
            "requirement": "Corrupted/unreliable 3D stress scenario",
            "status": "PASS" if not noisy3d.empty and noisy3d.get("noise_level", pd.Series(dtype=float)).nunique() >= 3 else "PARTIAL",
            "evidence": f"noisy3d_rows={len(noisy3d)}, noise_levels={sorted(noisy3d.get('noise_level', pd.Series(dtype=float)).dropna().unique().tolist()) if not noisy3d.empty else []}",
        },
        {
            "requirement": "Router input ablation",
            "status": "PASS" if not router_ablation.empty and router_ablation.get("router_feature_set", pd.Series()).nunique() >= 6 else "PARTIAL",
            "evidence": f"{router_ablation.get('router_feature_set', pd.Series(dtype=str)).nunique()} router feature sets",
        },
        {
            "requirement": "Mean +/- std over 5 seeds",
            "status": "PASS" if seed_ok else "PARTIAL",
            "evidence": "All acquisition summary rows have n_seeds>=5" if seed_ok else "Some rows have fewer than 5 seeds or are missing.",
        },
        {
            "requirement": "Modern SOTA comparison with 8-12 external methods and provenance",
            "status": "PASS" if method_count >= 8 and not strict_sota.empty else "PARTIAL",
            "evidence": f"external_methods={method_count}, strict_sota_rows={len(strict_sota)}",
        },
        {
            "requirement": "At least 2-4 public datasets under a fair split/protocol",
            "status": "PASS" if len(tasks) >= 4 else "PARTIAL",
            "evidence": f"datasets/tasks covered={', '.join(tasks)}",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    route_summary, route_curves, metrics, metas = _load_oof_summaries(args.oof_results_dir)
    budget_curve, acquisition20, oracle_recovery = build_acquisition_tables(route_summary, args.main_budget)
    router_diag, bootstrap = build_prediction_level_tables(
        args.oof_results_dir, args.main_budget, args.bootstrap_iters, args.bootstrap_seed
    )
    chemical = build_chemical_space_table(args.oof_results_dir, args.main_budget)
    cost = build_cost_summary(args.oof_results_dir, args.resource_monitor_csv)
    fusion_summary, best_fusion = build_fusion_tables(args.fusion_dir)
    noisy3d_summary, noisy3d_routing20 = build_noisy3d_tables(args.noisy3d_dir, args.main_budget)
    ablation_summary = _read_csv(Path(args.router_ablation_dir) / "routing_summary.csv")
    strict_sota = _read_csv(args.strict_sota_csv)
    external_matrix = _read_csv(args.external_method_matrix)
    special_budget = _read_csv(args.special_budget_csv)
    label_eff = _read_csv(args.label_efficiency_csv)
    checklist = build_checklist(
        route_summary,
        fusion_summary,
        cost,
        external_matrix,
        strict_sota,
        special_budget,
        label_eff,
        ablation_summary,
        noisy3d_summary,
        metas,
    )

    outputs = {
        "acquisition_budget_curve.csv": budget_curve,
        "acquisition_baseline_20pct.csv": acquisition20,
        "oracle_recovery_20pct.csv": oracle_recovery,
        "router_diagnostics_20pct.csv": router_diag,
        "paired_bootstrap_ci_20pct.csv": bootstrap,
        "router_chemical_space_summary.csv": chemical,
        "cost_resource_summary.csv": cost,
        "fusion_baseline_summary.csv": fusion_summary,
        "best_fusion_by_task.csv": best_fusion,
        "noisy3d_stress_summary.csv": noisy3d_summary,
        "noisy3d_routing_20pct.csv": noisy3d_routing20,
        "router_input_ablation_summary.csv": ablation_summary,
        "if_requirement_checklist.csv": checklist,
    }
    for name, df in outputs.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(out_dir / name, index=False)
    resource_summary = getattr(cost, "attrs", {}).get("resource_summary", {})
    (out_dir / "resource_monitor_summary.json").write_text(json.dumps(resource_summary, indent=2), encoding="utf-8")

    lines = [
        "# IF Full Experiment Audit",
        "",
        f"Generated at: {now_iso()}",
        "",
        "## Requirement Checklist",
        "",
        _markdown(checklist),
        "",
        f"## Acquisition Baselines at {args.main_budget:g}% 3D Budget",
        "",
        _markdown(acquisition20),
        "",
        "## Router Diagnostics",
        "",
        _markdown(router_diag),
        "",
        "## Paired Bootstrap CI",
        "",
        _markdown(bootstrap),
        "",
        "## Best Fusion Baseline by Task",
        "",
        _markdown(best_fusion),
        "",
        "## Noisy 3D Stress",
        "",
        _markdown(noisy3d_summary),
        "",
        "## Strict SOTA Candidate",
        "",
        _markdown(strict_sota),
        "",
        "## Cost Summary",
        "",
        _markdown(cost),
        "",
        "## Resource Monitor Summary",
        "",
        "```json",
        json.dumps(resource_summary, indent=2),
        "```",
        "",
        "## Files",
        "",
        "\n".join(f"- `{name}`" for name in outputs),
        "",
    ]
    (out_dir / "IF_FULL_EXPERIMENT_PACKAGE.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[if-full] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
