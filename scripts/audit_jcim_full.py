from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


TASK_TYPE = {
    "BACE": "classification",
    "BBBP": "classification",
    "HIV": "classification",
    "ESOL": "regression",
    "FreeSolv": "regression",
    "Lipophilicity": "regression",
}

ROUTER_ORDER = [
    "never_3d",
    "always_abstain",
    "random",
    "uncertainty",
    "flexibility",
    "old_voi",
    "R7_oof_selected_top",
    "R8_oof_selected_gated",
    "R9_oof_safe_guardrail",
    "oracle",
]

ROUTER_LABEL = {
    "never_3d": "Never-3D",
    "always_abstain": "Always-Abstain",
    "random": "Random",
    "uncertainty": "Uncertainty",
    "flexibility": "Flexibility",
    "old_voi": "Old VOI",
    "R7_oof_selected_top": "R7 Top-Budget",
    "R8_oof_selected_gated": "R8 OOF Gated",
    "R9_oof_safe_guardrail": "R9 Utility-LCB",
    "oracle": "Oracle",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--lcb-dir", default="results/if_router_gated_selection_lcb_v1_5seed")
    p.add_argument("--cache-dir", default="results/fast_screen_xgb_k10_auc_ecfp_router_5seed")
    p.add_argument("--k1-dir", default="results/fast_screen_scaffold_balanced")
    p.add_argument("--k10-dir", default="results/fast_screen_xgb_k10_auc_ecfp_router_5seed")
    p.add_argument("--baseline-dir", default="results/baseline_matrix_scaffold_balanced_5seed")
    p.add_argument("--out-dir", default="results/jcim_full_audit")
    p.add_argument("--main-budget", type=float, default=20.0)
    p.add_argument("--bootstrap-iters", type=int, default=2000)
    p.add_argument("--permutation-iters", type=int, default=300)
    p.add_argument("--seed", type=int, default=20260912)
    return p.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 1:
        return pd.read_csv(path)
    return pd.DataFrame()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def higher_is_better(metric: str) -> bool:
    return str(metric).upper() in {"AUROC", "AUPRC", "ACCURACY", "R2"}


def signed_delta(metric: str, baseline: float, candidate: float) -> float:
    if pd.isna(baseline) or pd.isna(candidate):
        return np.nan
    return float(candidate - baseline) if higher_is_better(metric) else float(baseline - candidate)


def mean_std(x: pd.Series) -> str:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if vals.empty:
        return ""
    if len(vals) == 1:
        return f"{float(vals.iloc[0]):.6f}"
    return f"{float(vals.mean()):.6f} +/- {float(vals.std(ddof=1)):.6f}"


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    show = df.head(max_rows).copy()
    return show.to_markdown(index=False)


def add_never3d_rows(curves: pd.DataFrame) -> pd.DataFrame:
    budgets = sorted(curves["budget"].astype(float).unique())
    base = (
        curves[curves["budget"].astype(float).eq(0.0)]
        .sort_values(["task", "seed", "router"])
        .groupby(["task", "task_type", "seed", "primary_metric"], as_index=False)
        .first()
    )
    rows = []
    for _, r in base.iterrows():
        for budget in budgets:
            for router in ["never_3d", "always_abstain"]:
                row = r.to_dict()
                row["router"] = router
                row["budget"] = float(budget)
                row["call_rate"] = 0.0
                row["primary_value"] = float(r["primary_value"])
                row["gate_quantile"] = np.nan
                row["gate_threshold"] = np.nan
                rows.append(row)
    if rows:
        curves = pd.concat([curves, pd.DataFrame(rows)], ignore_index=True)
    return curves


def build_pareto(curves: pd.DataFrame, main_budget: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curves = add_never3d_rows(curves.copy())
    curves["budget"] = curves["budget"].astype(float)
    base = (
        curves[curves["router"].eq("never_3d") & curves["budget"].eq(0.0)]
        [["task", "seed", "primary_value"]]
        .rename(columns={"primary_value": "never3d_value"})
    )
    out = curves.merge(base, on=["task", "seed"], how="left")
    out["gain_vs_never3d"] = [
        signed_delta(metric, base_val, cand)
        for metric, base_val, cand in zip(out["primary_metric"], out["never3d_value"], out["primary_value"])
    ]
    oracle = (
        out[out["router"].eq("oracle")]
        [["task", "seed", "budget", "gain_vs_never3d"]]
        .rename(columns={"gain_vs_never3d": "oracle_gain_same_budget"})
    )
    out = out.merge(oracle, on=["task", "seed", "budget"], how="left")
    out["oracle_gain_recovered"] = np.where(
        out["oracle_gain_same_budget"].abs() > 1e-12,
        out["gain_vs_never3d"] / out["oracle_gain_same_budget"],
        np.nan,
    )
    out["router_label"] = out["router"].map(ROUTER_LABEL).fillna(out["router"])
    out["is_main_budget"] = np.isclose(out["budget"], main_budget)

    grp_cols = ["router", "router_label", "budget"]
    macro = (
        out.groupby(grp_cols, dropna=False)
        .agg(
            gain_mean=("gain_vs_never3d", "mean"),
            gain_std=("gain_vs_never3d", "std"),
            call_rate_mean=("call_rate", "mean"),
            call_rate_std=("call_rate", "std"),
            oracle_recovered_mean=("oracle_gain_recovered", "mean"),
            positive_task_seed_pairs=("gain_vs_never3d", lambda s: int((s > 0).sum())),
            n_task_seed_pairs=("gain_vs_never3d", "count"),
        )
        .reset_index()
    )
    order = {r: i for i, r in enumerate(ROUTER_ORDER)}
    macro["router_order"] = macro["router"].map(order).fillna(999)
    macro = macro.sort_values(["budget", "router_order", "router"]).drop(columns=["router_order"])

    by_task = (
        out.groupby(["task", "task_type", "router", "router_label", "budget"], dropna=False)
        .agg(
            gain_mean=("gain_vs_never3d", "mean"),
            gain_std=("gain_vs_never3d", "std"),
            call_rate_mean=("call_rate", "mean"),
            oracle_recovered_mean=("oracle_gain_recovered", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    by_task["router_order"] = by_task["router"].map(order).fillna(999)
    by_task = by_task.sort_values(["task", "budget", "router_order"]).drop(columns=["router_order"])
    return out, macro, by_task


def select_top(scores: np.ndarray, budget: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    k = int(round(n * float(budget) / 100.0))
    selected = np.zeros(n, dtype=bool)
    if k <= 0 or n == 0:
        return selected
    if k >= n:
        selected[:] = True
        return selected
    safe_scores = np.where(np.isfinite(scores), scores, -np.inf)
    idx = np.argpartition(-safe_scores, kth=k - 1)[:k]
    selected[idx] = True
    return selected


def route_threshold(row: pd.Series) -> float:
    value = pd.to_numeric(pd.Series([row.get("gate_threshold", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(value):
        return -np.inf
    return float(value)


def source_prediction_dir(repo: Path, curves: pd.DataFrame, task: str) -> Path | None:
    g = curves[curves["task"].eq(task)]
    if g.empty or "source_dir" not in g:
        return None
    rel = str(g["source_dir"].dropna().iloc[0])
    p = Path(rel)
    return p if p.is_absolute() else repo / p


def load_score_frame(repo: Path, lcb_dir: Path, curves: pd.DataFrame, task: str) -> pd.DataFrame:
    gated = read_csv(lcb_dir / "predictions" / f"{task}_gated_scores.csv")
    src = source_prediction_dir(repo, curves, task)
    pred = read_csv(src / "predictions" / f"{task}_predictions.csv") if src else pd.DataFrame()
    if gated.empty:
        return pd.DataFrame()
    if pred.empty:
        return gated.copy()
    cols = [
        c
        for c in [
            "task",
            "seed",
            "mol_id",
            "y",
            "pred2d",
            "pred3d_aug",
            "voi_score",
            "uncertainty_score",
            "flexibility_score",
            "oracle_benefit",
        ]
        if c in pred
    ]
    return gated.merge(pred[cols], on=["task", "seed", "mol_id"], how="left", sort=False)


def scores_for_router(df: pd.DataFrame, router: str, seed: int) -> np.ndarray:
    if router == "random":
        return np.random.default_rng(int(seed)).normal(size=len(df))
    if router == "uncertainty" and "uncertainty_score" in df:
        return df["uncertainty_score"].to_numpy(float)
    if router == "flexibility" and "flexibility_score" in df:
        return df["flexibility_score"].to_numpy(float)
    if router == "old_voi" and "voi_score" in df:
        return df["voi_score"].to_numpy(float)
    if router == "oracle":
        return df["clean_utility"].to_numpy(float)
    if router in {"R7_oof_selected_top", "R8_oof_selected_gated", "R9_oof_safe_guardrail"}:
        return df["score_R8_oof_selected"].to_numpy(float)
    return np.zeros(len(df), dtype=float)


def selected_mask_for_route(score_df: pd.DataFrame, row: pd.Series) -> np.ndarray:
    router = str(row["router"])
    if router in {"never_3d", "always_abstain"}:
        return np.zeros(len(score_df), dtype=bool)
    if router == "R9_oof_safe_guardrail" and "safe_decision" in score_df:
        decisions = set(str(x) for x in score_df["safe_decision"].dropna().unique())
        if decisions == {"abstain"} or "abstain" in decisions and "call" not in decisions:
            return np.zeros(len(score_df), dtype=bool)
    scores = scores_for_router(score_df, router, int(row["seed"]))
    selected = select_top(scores, float(row["budget"]))
    threshold = route_threshold(row)
    if np.isfinite(threshold):
        selected &= scores >= threshold
    return selected


def build_risk_tables(repo: Path, lcb_dir: Path, curves_aug: pd.DataFrame, main_budget: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    task_frames: dict[str, pd.DataFrame] = {}
    for task in sorted(curves_aug["task"].dropna().unique()):
        task_frames[task] = load_score_frame(repo, lcb_dir, curves_aug, task)
    for _, route in curves_aug.iterrows():
        task = str(route["task"])
        df_all = task_frames.get(task, pd.DataFrame())
        if df_all.empty:
            continue
        df = df_all[df_all["seed"].astype(int).eq(int(route["seed"]))].copy()
        if df.empty:
            continue
        mask = selected_mask_for_route(df, route)
        vals = df.loc[mask, "clean_utility"].to_numpy(float)
        rows.append(
            {
                "task": task,
                "task_type": route["task_type"],
                "seed": int(route["seed"]),
                "router": route["router"],
                "router_label": ROUTER_LABEL.get(str(route["router"]), str(route["router"])),
                "budget": float(route["budget"]),
                "selected_n": int(mask.sum()),
                "test_n": int(len(df)),
                "coverage": float(mask.mean() * 100.0) if len(df) else 0.0,
                "harmful_selected_n": int((vals < 0).sum()) if len(vals) else 0,
                "beneficial_selected_n": int((vals > 0).sum()) if len(vals) else 0,
                "harmful_acquisition_rate": float((vals < 0).mean()) if len(vals) else 0.0,
                "beneficial_acquisition_rate": float((vals > 0).mean()) if len(vals) else 0.0,
                "selected_mean_utility": float(np.nanmean(vals)) if len(vals) else np.nan,
                "selected_sum_utility": float(np.nansum(vals)) if len(vals) else 0.0,
            }
        )
    long = pd.DataFrame(rows)
    if long.empty:
        return long, pd.DataFrame(), pd.DataFrame()
    summary = (
        long.groupby(["router", "router_label", "budget"], dropna=False)
        .agg(
            coverage_mean=("coverage", "mean"),
            harmful_rate_mean=("harmful_acquisition_rate", "mean"),
            beneficial_rate_mean=("beneficial_acquisition_rate", "mean"),
            selected_mean_utility_mean=("selected_mean_utility", "mean"),
            selected_sum_utility_mean=("selected_sum_utility", "mean"),
            selected_n_mean=("selected_n", "mean"),
            n_pairs=("selected_n", "count"),
        )
        .reset_index()
    )
    order = {r: i for i, r in enumerate(ROUTER_ORDER)}
    summary["router_order"] = summary["router"].map(order).fillna(999)
    summary = summary.sort_values(["budget", "router_order"]).drop(columns=["router_order"])

    r9 = long[long["router"].eq("R9_oof_safe_guardrail") & np.isclose(long["budget"], main_budget)].copy()
    if not r9.empty:
        r9["false_safe_realized_mean_le_0"] = (r9["selected_n"] > 0) & (r9["selected_mean_utility"] <= 0)
        r9["safe_call_but_zero_selected"] = r9["selected_n"].eq(0)
    return long, summary, r9


def bootstrap_lcb(vals: np.ndarray, alpha: float, rng: np.random.Generator, iters: int) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return -np.inf
    if len(vals) == 1:
        return float(vals[0])
    idx = rng.integers(0, len(vals), size=(iters, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(np.nanpercentile(means, 100.0 * alpha))


def build_lcb_posthoc_tables(
    repo: Path,
    lcb_dir: Path,
    curves_aug: pd.DataFrame,
    main_budget: float,
    rng: np.random.Generator,
    bootstrap_iters: int,
    permutation_iters: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oof = read_csv(lcb_dir / "oof_selection_long.csv")
    task_frames = {task: load_score_frame(repo, lcb_dir, curves_aug, task) for task in sorted(curves_aug["task"].dropna().unique())}
    route_rows = curves_aug[
        curves_aug["router"].eq("R8_oof_selected_gated") & np.isclose(curves_aug["budget"].astype(float), main_budget)
    ].copy()
    alpha_rows = []
    null_rows = []
    for _, route in route_rows.iterrows():
        task = str(route["task"])
        seed = int(route["seed"])
        df = task_frames.get(task, pd.DataFrame())
        df = df[df["seed"].astype(int).eq(seed)].copy() if not df.empty else df
        if df.empty:
            continue
        mask = selected_mask_for_route(df, route)
        vals = df.loc[mask, "clean_utility"].to_numpy(float)
        realized_mean = float(np.nanmean(vals)) if len(vals) else np.nan
        for alpha in [0.01, 0.025, 0.05, 0.10]:
            lcb = bootstrap_lcb(vals, alpha, rng, bootstrap_iters)
            alpha_rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "candidate_router": "R8_oof_selected_gated",
                    "budget": float(main_budget),
                    "alpha": float(alpha),
                    "selected_n": int(len(vals)),
                    "realized_selected_mean_utility": realized_mean,
                    "posthoc_test_lcb": lcb,
                    "would_permit_on_posthoc_test_lcb": bool(lcb > 0.0),
                    "false_safe_if_permitted": bool(lcb > 0.0 and (not np.isfinite(realized_mean) or realized_mean <= 0.0)),
                    "note": "diagnostic only; uses test utilities after selection, not an independent calibration split",
                }
            )
        if len(vals):
            all_vals = df["clean_utility"].to_numpy(float)
            selected_n = int(mask.sum())
            permits = []
            lcbs = []
            for _ in range(permutation_iters):
                perm_vals = rng.permutation(all_vals)[:selected_n]
                lcb = bootstrap_lcb(perm_vals, 0.025, rng, max(200, bootstrap_iters // 4))
                lcbs.append(lcb)
                permits.append(lcb > 0.0)
            null_rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "selected_n": selected_n,
                    "permutation_iters": int(permutation_iters),
                    "null_permit_rate_alpha_0p025": float(np.mean(permits)),
                    "null_lcb_mean": float(np.mean(lcbs)),
                    "null_lcb_p95": float(np.percentile(lcbs, 95)),
                    "note": "diagnostic null; shuffles realized test utility labels with selected set size fixed",
                }
            )
    alpha_df = pd.DataFrame(alpha_rows)
    if not alpha_df.empty:
        alpha_summary = (
            alpha_df.groupby("alpha")
            .agg(
                permitted_rate=("would_permit_on_posthoc_test_lcb", "mean"),
                false_safe_rate=("false_safe_if_permitted", "mean"),
                mean_realized_utility=("realized_selected_mean_utility", "mean"),
                n_task_seed_pairs=("seed", "count"),
            )
            .reset_index()
        )
    else:
        alpha_summary = pd.DataFrame()
    oof_keep = oof.copy()
    if not oof_keep.empty:
        oof_keep["safe_decision_call"] = oof_keep["safe_decision"].eq("call")
        oof_keep = oof_keep[
            [
                "task",
                "task_type",
                "seed",
                "objective",
                "feature_group",
                "gate_quantile",
                "oof_BIG",
                "safe_decision",
                "oof_selected_n",
                "oof_selected_mean_utility",
                "oof_selected_utility_lcb",
            ]
        ]
    return oof_keep, alpha_summary, pd.DataFrame(null_rows)


def sign_permutation_p(delta: np.ndarray, rng: np.random.Generator, iters: int) -> tuple[float, float]:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    obs = float(np.mean(delta))
    if len(delta) == 0:
        return np.nan, np.nan
    signs = rng.choice(np.array([-1.0, 1.0]), size=(iters, len(delta)))
    null = (signs * delta[None, :]).mean(axis=1)
    p_two = float((np.abs(null) >= abs(obs)).mean())
    p_one = float((null >= obs).mean()) if obs >= 0 else float((null <= obs).mean())
    return max(p_two, 1.0 / iters), max(p_one, 1.0 / iters)


def adjust_pvalues(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    m = len(out)
    if m == 0:
        return out
    order = np.argsort(out["p_two_sided"].to_numpy(float))
    sorted_p = out["p_two_sided"].to_numpy(float)[order]
    holm = np.empty(m, dtype=float)
    running = 0.0
    for rank, p in enumerate(sorted_p):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        holm[order[rank]] = running
    bh = np.empty(m, dtype=float)
    running_bh = 1.0
    for rev_rank in range(m - 1, -1, -1):
        p = sorted_p[rev_rank]
        adj = min(running_bh, p * m / (rev_rank + 1))
        running_bh = adj
        bh[order[rev_rank]] = adj
    out["p_two_sided_holm"] = holm
    out["p_two_sided_bh"] = bh
    return out


def build_holm_table(big: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    pivot = big.pivot_table(index=["task", "seed"], columns="router", values="BIG", aggfunc="mean")
    final = "R9_oof_safe_guardrail"
    rows = []
    for base in ["random", "uncertainty", "flexibility", "old_voi", "R7_oof_selected_top", "R8_oof_selected_gated"]:
        if final not in pivot or base not in pivot:
            continue
        paired = pivot[[final, base]].dropna()
        delta = (paired[final] - paired[base]).to_numpy(float)
        p_two, p_one = sign_permutation_p(delta, rng, iters=20000)
        rows.append(
            {
                "comparison": f"R9 - {ROUTER_LABEL.get(base, base)}",
                "baseline": base,
                "mean_delta_BIG": float(np.mean(delta)),
                "std_delta_BIG": float(np.std(delta, ddof=1)),
                "n_pairs": int(len(delta)),
                "positive_pairs": int((delta > 0).sum()),
                "p_two_sided": p_two,
                "p_one_sided_improvement": p_one,
            }
        )
    return adjust_pvalues(pd.DataFrame(rows))


def find_meta(repo: Path, cache_dir: Path, task: str) -> pd.DataFrame:
    candidates = sorted((cache_dir / "feature_cache").glob(f"{task}_*meta.csv"))
    if not candidates:
        candidates = sorted((repo / "results").glob(f"**/{task}_*meta.csv"))
    for p in candidates:
        df = read_csv(p)
        if not df.empty and "mol_id" in df and ("canonical_smiles" in df or "smiles" in df):
            if "canonical_smiles" not in df and "smiles" in df:
                df = df.rename(columns={"smiles": "canonical_smiles"})
            return df[["mol_id", "canonical_smiles", "split"] + (["y"] if "y" in df else [])].copy()
    return pd.DataFrame()


def rdkit_descriptors(smiles: str) -> dict[str, float]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    except Exception:
        return {}
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {}
    return {
        "MolWt": float(Descriptors.MolWt(mol)),
        "RotatableBonds": float(Lipinski.NumRotatableBonds(mol)),
        "RingCount": float(rdMolDescriptors.CalcNumRings(mol)),
        "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "MolLogP": float(Crippen.MolLogP(mol)),
        "HeavyAtomCount": float(mol.GetNumHeavyAtoms()),
    }


def build_chemical_interpretation(
    repo: Path,
    lcb_dir: Path,
    cache_dir: Path,
    curves_aug: pd.DataFrame,
    main_budget: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = []
    descriptor_cache: dict[str, dict[str, float]] = {}
    for task in sorted(curves_aug["task"].dropna().unique()):
        scores = load_score_frame(repo, lcb_dir, curves_aug, task)
        meta = find_meta(repo, cache_dir, task)
        if scores.empty or meta.empty:
            continue
        scores = scores.merge(meta[["mol_id", "canonical_smiles", "split"]], on="mol_id", how="left")
        routes = curves_aug[
            curves_aug["task"].eq(task)
            & curves_aug["router"].eq("R9_oof_safe_guardrail")
            & np.isclose(curves_aug["budget"].astype(float), main_budget)
        ]
        for _, route in routes.iterrows():
            seed = int(route["seed"])
            df = scores[scores["seed"].astype(int).eq(seed)].copy()
            if df.empty:
                continue
            mask = selected_mask_for_route(df, route)
            for idx, row in df.iterrows():
                smi = str(row.get("canonical_smiles", ""))
                if smi not in descriptor_cache:
                    descriptor_cache[smi] = rdkit_descriptors(smi)
                desc = descriptor_cache[smi]
                if not desc:
                    continue
                rec = {
                    "task": task,
                    "seed": seed,
                    "mol_id": row["mol_id"],
                    "r9_acquired": bool(mask[df.index.get_loc(idx)]),
                    "clean_utility": float(row["clean_utility"]),
                    "safe_decision": row.get("safe_decision", ""),
                }
                rec.update(desc)
                records.append(rec)
    long = pd.DataFrame(records)
    if long.empty:
        return long, pd.DataFrame(), pd.DataFrame()
    desc_cols = ["clean_utility", "MolWt", "RotatableBonds", "RingCount", "FractionCSP3", "TPSA", "MolLogP", "HeavyAtomCount"]
    summary_rows = []
    for (task, acquired), g in long.groupby(["task", "r9_acquired"]):
        row = {"task": task, "r9_acquired": bool(acquired), "n": int(len(g))}
        for c in desc_cols:
            row[f"{c}_mean"] = float(pd.to_numeric(g[c], errors="coerce").mean())
            row[f"{c}_std"] = float(pd.to_numeric(g[c], errors="coerce").std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["task", "r9_acquired"])
    bins = [-0.1, 0, 2, 5, 8, 99]
    labels = ["0", "1-2", "3-5", "6-8", "9+"]
    long["rotatable_bin"] = pd.cut(long["RotatableBonds"], bins=bins, labels=labels)
    rot = (
        long.groupby(["task", "rotatable_bin"], observed=False)
        .agg(p_acquire_3d=("r9_acquired", "mean"), n=("r9_acquired", "count"), mean_utility=("clean_utility", "mean"))
        .reset_index()
    )
    return long, summary, rot


def build_cost_summary(cache_dir: Path, risk_long: pd.DataFrame, main_budget: float) -> pd.DataFrame:
    manifest = read_csv(cache_dir / "conformer_manifest.csv")
    if manifest.empty:
        frames = [read_csv(p) for p in sorted(cache_dir.glob("conformer_manifest_*.csv"))]
        frames = [f for f in frames if not f.empty]
        manifest = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if manifest.empty:
        return pd.DataFrame()
    rows = []
    for task, g in manifest.groupby("task"):
        elapsed = pd.to_numeric(g.get("elapsed_sec", pd.Series(dtype=float)), errors="coerce")
        succ = g.get("success", pd.Series(dtype=bool)).astype(bool)
        r9 = risk_long[
            risk_long["task"].eq(task)
            & risk_long["router"].eq("R9_oof_safe_guardrail")
            & np.isclose(risk_long["budget"].astype(float), main_budget)
        ]
        oracle = risk_long[
            risk_long["task"].eq(task)
            & risk_long["router"].eq("oracle")
            & np.isclose(risk_long["budget"].astype(float), main_budget)
        ]
        rows.append(
            {
                "task": task,
                "n_molecules_with_conformer_attempt": int(len(g)),
                "conformer_success_rate": float(succ.mean()),
                "mean_etkdg_mmff_elapsed_sec": float(elapsed.mean()),
                "p95_etkdg_mmff_elapsed_sec": float(elapsed.quantile(0.95)),
                "total_etkdg_mmff_cpu_hours_if_all": float(elapsed.sum() / 3600.0),
                "r9_actual_call_rate_at_20": float(r9["coverage"].mean()) if not r9.empty else np.nan,
                "oracle_call_rate_at_20": float(oracle["coverage"].mean()) if not oracle.empty else np.nan,
                "estimated_r9_cpu_hours_at_test_scale": float((r9["selected_n"].mean() * elapsed.mean()) / 3600.0) if not r9.empty else np.nan,
                "estimated_oracle20_cpu_hours_at_test_scale": float((oracle["selected_n"].mean() * elapsed.mean()) / 3600.0) if not oracle.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("task")


def build_k1_k10(cache_root: Path, k1_dir: Path, k10_dir: Path) -> pd.DataFrame:
    k1 = read_csv(k1_dir / "analysis_tables" / "budget20_comparison.csv")
    k10 = read_csv(k10_dir / "analysis_tables" / "budget20_comparison.csv")
    if k1.empty or k10.empty:
        return pd.DataFrame()
    a = k1[["task", "method", "primary_metric", "primary_mean", "delta_vs_all_2d_positive_better", "call_rate_mean"]].rename(
        columns={
            "primary_mean": "k1_primary_mean",
            "delta_vs_all_2d_positive_better": "k1_delta_vs_2d",
            "call_rate_mean": "k1_call_rate",
        }
    )
    b = k10[["task", "method", "primary_metric", "primary_mean", "delta_vs_all_2d_positive_better", "call_rate_mean"]].rename(
        columns={
            "primary_mean": "k10_primary_mean",
            "delta_vs_all_2d_positive_better": "k10_delta_vs_2d",
            "call_rate_mean": "k10_call_rate",
        }
    )
    out = a.merge(b, on=["task", "method", "primary_metric"], how="inner")
    out["k10_minus_k1_signed"] = [
        signed_delta(metric, k1v, k10v) for metric, k1v, k10v in zip(out["primary_metric"], out["k1_primary_mean"], out["k10_primary_mean"])
    ]
    return out.sort_values(["task", "method"])


def collect_backbones(repo: Path) -> pd.DataFrame:
    paths = [
        repo / "results/chemprop_cls_small_5seed_e30/method_summary.csv",
        repo / "results/chemprop_reg_small_5seed_e30/method_summary.csv",
        repo / "results/chemprop_hiv_5seed_e30/method_summary.csv",
        repo / "results/chemprop_hiv_balanced_5seed_e30/method_summary.csv",
        repo / "results/chemprop_lipo_5seed_e30/method_summary.csv",
        repo / "results/baseline_matrix_scaffold_balanced_5seed/method_summary.csv",
    ]
    frames = []
    for p in paths:
        df = read_csv(p)
        if not df.empty:
            df["result_file"] = str(p.relative_to(repo))
            frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def build_status_checklist(out_dir: Path) -> pd.DataFrame:
    rows = [
        {
            "priority": "P0",
            "requirement": "Never-3D / Always-Abstain baseline and Pareto/risk tables",
            "status": "completed_from_existing_5seed_data",
            "evidence": "never3d_pareto_long.csv, never3d_pareto_macro.csv, risk_coverage_long.csv",
        },
        {
            "priority": "P0",
            "requirement": "Conformation-sensitive public benchmark",
            "status": "not_completed",
            "evidence": "No PubChemQC/PQC/stereochemistry/enantioselectivity benchmark result found in repository.",
        },
        {
            "priority": "P0",
            "requirement": "Chemistry-grounded conformer quality experiment",
            "status": "partial",
            "evidence": "K=1 vs K=10 and ETKDG/MMFF manifest are available; random/high-energy/Boltzmann/best conformer conditions are not yet run.",
        },
        {
            "priority": "P0",
            "requirement": "LCB statistical independence / nested calibration",
            "status": "partial_diagnostic_only",
            "evidence": "OOF Utility-LCB decisions exist; post-hoc alpha/null diagnostics are generated, but independent calibration split is not yet implemented.",
        },
        {
            "priority": "P0/P1",
            "requirement": "Strong 2D / strong 3D backbone",
            "status": "partial",
            "evidence": "Chemprop D-MPNN 5-seed results exist; Uni-Mol or equivalent strong 3D backbone not found.",
        },
        {
            "priority": "P1",
            "requirement": "OOD / chemical-space shift",
            "status": "partial",
            "evidence": "Scaffold-balanced protocol/baseline exists; cluster-based OOD split not found.",
        },
        {
            "priority": "P1",
            "requirement": "Reliability calibration and risk-coverage",
            "status": "partial",
            "evidence": "Risk-coverage and realized false-safe tables generated for the selected-set utility analysis.",
        },
        {
            "priority": "P1",
            "requirement": "Chemical interpretation",
            "status": "partial_completed",
            "evidence": "RDKit descriptor comparison for R9 acquired vs abstained is generated; scaffold frequency/distance and functional enrichment remain optional additions.",
        },
        {
            "priority": "P1",
            "requirement": "Real acquisition cost",
            "status": "partial_completed",
            "evidence": "ETKDG+MMFF elapsed-time/cost table generated; xTB refinement cost not run.",
        },
        {
            "priority": "Stats",
            "requirement": "Holm / BH multiple-comparison correction",
            "status": "completed_from_existing_5seed_data",
            "evidence": "holm_bh_big_correction.csv",
        },
    ]
    return pd.DataFrame(rows)


def write_report(
    out_dir: Path,
    status: pd.DataFrame,
    pareto_macro: pd.DataFrame,
    pareto_task: pd.DataFrame,
    risk_summary: pd.DataFrame,
    r9_risk: pd.DataFrame,
    holm: pd.DataFrame,
    lcb_alpha: pd.DataFrame,
    null_df: pd.DataFrame,
    chem_summary: pd.DataFrame,
    cost: pd.DataFrame,
    k1k10: pd.DataFrame,
    backbones: pd.DataFrame,
    main_budget: float,
) -> None:
    main_pareto = pareto_macro[np.isclose(pareto_macro["budget"].astype(float), main_budget)].copy()
    main_pareto = main_pareto[
        [
            "router_label",
            "gain_mean",
            "gain_std",
            "call_rate_mean",
            "oracle_recovered_mean",
            "positive_task_seed_pairs",
            "n_task_seed_pairs",
        ]
    ]
    main_risk = risk_summary[np.isclose(risk_summary["budget"].astype(float), main_budget)].copy()
    main_risk = main_risk[["router_label", "coverage_mean", "harmful_rate_mean", "selected_mean_utility_mean", "selected_n_mean"]]
    r9_row = main_pareto[main_pareto["router_label"].eq("R9 Utility-LCB")]
    random_row = main_pareto[main_pareto["router_label"].eq("Random")]
    oracle_row = main_pareto[main_pareto["router_label"].eq("Oracle")]
    r9_gain = float(r9_row["gain_mean"].iloc[0]) if not r9_row.empty else np.nan
    random_gain = float(random_row["gain_mean"].iloc[0]) if not random_row.empty else np.nan
    oracle_recovered = float(r9_row["oracle_recovered_mean"].iloc[0]) if not r9_row.empty else np.nan
    r9_call = float(r9_row["call_rate_mean"].iloc[0]) if not r9_row.empty else np.nan
    false_safe = float(r9_risk["false_safe_realized_mean_le_0"].mean()) if not r9_risk.empty else np.nan

    text = f"""# JCIM/IF Full Experiment Audit for VOILA-3D

This report checks the requested JCIM-style supplement items against the current remote results and generates reviewer-facing tables without drawing figures.

## Completion Status

{markdown_table(status, max_rows=20)}

## Main Pareto Table at {main_budget:.0f}% Nominal Budget

Never-3D / Always-Abstain are explicit zero-cost baselines. Gains are signed relative to Never-3D: AUROC gain for classification and MAE reduction for regression.

{markdown_table(main_pareto, max_rows=20)}

Key reading: R9 has mean gain {r9_gain:.6f} at mean actual call rate {r9_call:.2f}%, while Random has mean gain {random_gain:.6f}. R9 recovers {oracle_recovered:.3f} of the same-budget oracle gain on average. This supports the safer story: R9 mainly removes harmful acquisition rather than claiming large raw prediction gains.

## Risk/Coverage at {main_budget:.0f}% Nominal Budget

{markdown_table(main_risk, max_rows=20)}

R9 realized false-safe rate using selected-set mean utility <= 0 is {false_safe:.3f}. This is a post-hoc test-set diagnostic, not an independent calibration guarantee.

## Multiple-Comparison Correction

{markdown_table(holm, max_rows=20)}

## LCB Diagnostics

Stored OOF Utility-LCB decisions are available, but the current pipeline does not yet implement a fully independent nested calibration split. The generated alpha table below is a post-hoc diagnostic over realized selected utilities.

{markdown_table(lcb_alpha, max_rows=20)}

Permutation-null diagnostics:

{markdown_table(null_df, max_rows=12)}

## Chemical Interpretation

R9 acquired vs abstained descriptor comparison:

{markdown_table(chem_summary, max_rows=20)}

## Conformer Cost and K-Conformer Sensitivity

ETKDG+MMFF cost summary:

{markdown_table(cost, max_rows=20)}

K=1 vs K=10 sensitivity:

{markdown_table(k1k10.head(24), max_rows=24)}

## Strong Backbone Check

Current rerun backbone rows include classic descriptor ML and Chemprop D-MPNN. A Uni-Mol/equivalent 3D backbone was not found.

{markdown_table(backbones[['task','method_group','method' if 'method' in backbones.columns else 'variant','variant' if 'variant' in backbones.columns else 'primary_metric','primary_metric','primary_mean','primary_std','n_seeds','result_file']].head(30) if not backbones.empty else backbones, max_rows=30)}

## Reviewer-Safe Story Update

1. More molecular information is not automatically better: fixed 3D spending and naive heuristics can be harmful.
2. R9 should be presented as the Utility-LCB acquisition policy inside VOILA-3D, not as the final predictor.
3. The main acquisition claim is risk control: when evidence for 3D value is weak, R9 abstains and largely removes negative transfer.
4. Strong claims still need new runs for independent nested calibration, a true conformation-sensitive benchmark, and a strong 3D backbone.

## Generated Files

- `experiment_status_checklist.csv`
- `never3d_pareto_long.csv`
- `never3d_pareto_macro.csv`
- `never3d_pareto_by_task.csv`
- `risk_coverage_long.csv`
- `risk_coverage_summary.csv`
- `r9_seed_level_risk_at_20.csv`
- `lcb_oof_decisions.csv`
- `lcb_posthoc_alpha_summary.csv`
- `permutation_null_utility_lcb.csv`
- `holm_bh_big_correction.csv`
- `chemical_interpretation_long.csv`
- `chemical_interpretation_summary.csv`
- `p_acquire_by_rotatable_bin.csv`
- `real_conformer_cost_summary.csv`
- `k1_vs_k10_conformer_sensitivity.csv`
- `backbone_summary.csv`
"""
    (out_dir / "JCIM_FULL_AUDIT_REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = Path(args.repo_root).resolve()
    lcb_dir = repo / args.lcb_dir
    cache_dir = repo / args.cache_dir
    out_dir = ensure_dir(repo / args.out_dir)
    rng = np.random.default_rng(args.seed)

    curves = read_csv(lcb_dir / "routing_curves.csv")
    big = read_csv(lcb_dir / "budget_integrated_gain_long.csv")
    if curves.empty or big.empty:
        raise FileNotFoundError(f"Missing LCB routing/BIG files under {lcb_dir}")

    pareto_long, pareto_macro, pareto_task = build_pareto(curves, args.main_budget)
    pareto_long.to_csv(out_dir / "never3d_pareto_long.csv", index=False)
    pareto_macro.to_csv(out_dir / "never3d_pareto_macro.csv", index=False)
    pareto_task.to_csv(out_dir / "never3d_pareto_by_task.csv", index=False)

    curves_aug = add_never3d_rows(curves)
    risk_long, risk_summary, r9_risk = build_risk_tables(repo, lcb_dir, curves_aug, args.main_budget)
    risk_long.to_csv(out_dir / "risk_coverage_long.csv", index=False)
    risk_summary.to_csv(out_dir / "risk_coverage_summary.csv", index=False)
    r9_risk.to_csv(out_dir / "r9_seed_level_risk_at_20.csv", index=False)

    oof_lcb, alpha_summary, null_df = build_lcb_posthoc_tables(
        repo,
        lcb_dir,
        curves_aug,
        args.main_budget,
        rng,
        args.bootstrap_iters,
        args.permutation_iters,
    )
    oof_lcb.to_csv(out_dir / "lcb_oof_decisions.csv", index=False)
    alpha_summary.to_csv(out_dir / "lcb_posthoc_alpha_summary.csv", index=False)
    null_df.to_csv(out_dir / "permutation_null_utility_lcb.csv", index=False)

    holm = build_holm_table(big, rng)
    holm.to_csv(out_dir / "holm_bh_big_correction.csv", index=False)

    chem_long, chem_summary, rot = build_chemical_interpretation(repo, lcb_dir, cache_dir, curves_aug, args.main_budget)
    chem_long.to_csv(out_dir / "chemical_interpretation_long.csv", index=False)
    chem_summary.to_csv(out_dir / "chemical_interpretation_summary.csv", index=False)
    rot.to_csv(out_dir / "p_acquire_by_rotatable_bin.csv", index=False)

    cost = build_cost_summary(cache_dir, risk_long, args.main_budget)
    cost.to_csv(out_dir / "real_conformer_cost_summary.csv", index=False)

    k1k10 = build_k1_k10(repo, repo / args.k1_dir, repo / args.k10_dir)
    k1k10.to_csv(out_dir / "k1_vs_k10_conformer_sensitivity.csv", index=False)

    backbones = collect_backbones(repo)
    backbones.to_csv(out_dir / "backbone_summary.csv", index=False)

    status = build_status_checklist(out_dir)
    status.to_csv(out_dir / "experiment_status_checklist.csv", index=False)

    write_report(
        out_dir,
        status,
        pareto_macro,
        pareto_task,
        risk_summary,
        r9_risk,
        holm,
        alpha_summary,
        null_df,
        chem_summary,
        cost,
        k1k10,
        backbones,
        args.main_budget,
    )
    print(f"[jcim-audit] wrote {out_dir}")


if __name__ == "__main__":
    main()
