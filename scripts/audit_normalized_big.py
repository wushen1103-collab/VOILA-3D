from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FINAL_ROUTER = "R9_oof_safe_guardrail"
BASELINES = [
    "random",
    "uncertainty",
    "flexibility",
    "old_voi",
    "R7_oof_selected_top",
    "R8_oof_selected_gated",
]
ROUTER_LABELS = {
    "oracle": "Oracle",
    "R9_oof_safe_guardrail": "R9 Utility-LCB Guardrail",
    "R8_oof_selected_gated": "R8 OOF Selected Gated",
    "R7_oof_selected_top": "R7 Top-Budget",
    "old_voi": "Old VOI",
    "uncertainty": "Uncertainty",
    "random": "Random",
    "flexibility": "Flexibility",
}


def _fmt(x: float, digits: int = 6) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{digits}f}"


def _mean_std(vals: pd.Series, digits: int = 6) -> str:
    vals = pd.to_numeric(vals, errors="coerce")
    return f"{_fmt(float(vals.mean()), digits)} +/- {_fmt(float(vals.std(ddof=0)), digits)}"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_normalized_big(big_long: pd.DataFrame, eps: float) -> pd.DataFrame:
    pivot = big_long.pivot_table(index=["task", "task_type", "seed"], columns="router", values="BIG", aggfunc="mean")
    if "oracle" not in pivot:
        raise ValueError("Missing oracle BIG values; cannot normalize by oracle acquisition scale.")
    rows = []
    for idx, row in pivot.iterrows():
        task, task_type, seed = idx
        oracle = float(row["oracle"])
        scale = abs(oracle) + eps
        for router, big in row.dropna().items():
            rows.append(
                {
                    "task": task,
                    "task_type": task_type,
                    "seed": int(seed),
                    "router": router,
                    "BIG": float(big),
                    "oracle_BIG": oracle,
                    "oracle_abs_scale": scale,
                    "norm_BIG_by_oracle": float(big) / scale,
                }
            )
    return pd.DataFrame(rows)


def summarize_macro(norm_long: pd.DataFrame) -> pd.DataFrame:
    seed_level = (
        norm_long.groupby(["router", "task", "task_type"], as_index=False)
        .agg(
            BIG_mean=("BIG", "mean"),
            BIG_std=("BIG", lambda x: float(pd.Series(x).std(ddof=0))),
            norm_BIG_mean=("norm_BIG_by_oracle", "mean"),
            norm_BIG_std=("norm_BIG_by_oracle", lambda x: float(pd.Series(x).std(ddof=0))),
            n_seeds=("seed", "nunique"),
        )
    )
    rows = []
    for router, g in seed_level.groupby("router"):
        rows.append(
            {
                "router": router,
                "router_label": ROUTER_LABELS.get(router, router),
                "macro_raw_BIG_task_mean": float(g["BIG_mean"].mean()),
                "macro_norm_BIG_task_mean": float(g["norm_BIG_mean"].mean()),
                "macro_norm_BIG_task_std": float(g["norm_BIG_mean"].std(ddof=0)),
                "n_tasks": int(g["task"].nunique()),
                "n_seeds_per_task_min": int(g["n_seeds"].min()),
                "n_seeds_per_task_max": int(g["n_seeds"].max()),
            }
        )
    out = pd.DataFrame(rows)
    out["macro_norm_rank"] = out["macro_norm_BIG_task_mean"].rank(ascending=False, method="min").astype(int)
    return out.sort_values(["macro_norm_rank", "router_label"]).reset_index(drop=True)


def task_level_delta(norm_long: pd.DataFrame, final_router: str, baselines: list[str]) -> pd.DataFrame:
    pivot = norm_long.pivot_table(
        index=["task", "task_type", "seed"],
        columns="router",
        values=["BIG", "norm_BIG_by_oracle"],
        aggfunc="mean",
    )
    rows = []
    for base in baselines:
        if ("BIG", final_router) not in pivot or ("BIG", base) not in pivot:
            continue
        df = pd.DataFrame(
            {
                "BIG_final": pivot[("BIG", final_router)],
                "BIG_baseline": pivot[("BIG", base)],
                "norm_final": pivot[("norm_BIG_by_oracle", final_router)],
                "norm_baseline": pivot[("norm_BIG_by_oracle", base)],
            }
        ).dropna()
        df["raw_delta"] = df["BIG_final"] - df["BIG_baseline"]
        df["norm_delta"] = df["norm_final"] - df["norm_baseline"]
        for (task, task_type), g in df.reset_index().groupby(["task", "task_type"]):
            rows.append(
                {
                    "final_router": final_router,
                    "baseline": base,
                    "task": task,
                    "task_type": task_type,
                    "raw_delta_mean": float(g["raw_delta"].mean()),
                    "raw_delta_std": float(g["raw_delta"].std(ddof=0)),
                    "norm_delta_mean": float(g["norm_delta"].mean()),
                    "norm_delta_std": float(g["norm_delta"].std(ddof=0)),
                    "n_seeds": int(g["seed"].nunique()),
                    "n_positive_raw_seed_pairs": int((g["raw_delta"] > 0).sum()),
                    "n_positive_norm_seed_pairs": int((g["norm_delta"] > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def stratified_bootstrap_delta(
    norm_long: pd.DataFrame,
    final_router: str,
    baselines: list[str],
    iters: int,
    seed: int,
) -> pd.DataFrame:
    pivot = norm_long.pivot_table(
        index=["task", "task_type", "seed"],
        columns="router",
        values=["BIG", "norm_BIG_by_oracle"],
        aggfunc="mean",
    )
    rng = np.random.default_rng(seed)
    rows = []
    for base in baselines:
        raw_final = ("BIG", final_router)
        raw_base = ("BIG", base)
        norm_final = ("norm_BIG_by_oracle", final_router)
        norm_base = ("norm_BIG_by_oracle", base)
        if raw_final not in pivot or raw_base not in pivot or norm_final not in pivot or norm_base not in pivot:
            continue
        df = pd.DataFrame(
            {
                "raw_final": pivot[raw_final],
                "raw_base": pivot[raw_base],
                "norm_final": pivot[norm_final],
                "norm_base": pivot[norm_base],
            }
        ).dropna().reset_index()
        df["raw_delta"] = df["raw_final"] - df["raw_base"]
        df["norm_delta"] = df["norm_final"] - df["norm_base"]
        tasks = sorted(df["task"].unique())
        if not tasks:
            continue
        task_means = df.groupby("task", as_index=False).agg(raw_delta=("raw_delta", "mean"), norm_delta=("norm_delta", "mean"))
        task_pos_raw = int((task_means["raw_delta"] > 0).sum())
        task_pos_norm = int((task_means["norm_delta"] > 0).sum())
        raw_by_task = []
        norm_by_task = []
        for task in tasks:
            g = df[df["task"] == task]
            raw_vals = g["raw_delta"].to_numpy(float)
            norm_vals = g["norm_delta"].to_numpy(float)
            seed_idx = rng.integers(0, len(raw_vals), size=(iters, len(raw_vals)))
            raw_by_task.append(raw_vals[seed_idx].mean(axis=1))
            norm_by_task.append(norm_vals[seed_idx].mean(axis=1))
        raw_by_task_arr = np.vstack(raw_by_task)
        norm_by_task_arr = np.vstack(norm_by_task)
        task_idx = rng.integers(0, len(tasks), size=(iters, len(tasks)))
        iter_idx = np.arange(iters)[:, None]
        raw_boot = raw_by_task_arr.T[iter_idx, task_idx].mean(axis=1)
        norm_boot = norm_by_task_arr.T[iter_idx, task_idx].mean(axis=1)
        rows.append(
            {
                "final_router": final_router,
                "baseline": base,
                "macro_raw_task_mean_delta": float(task_means["raw_delta"].mean()),
                "macro_raw_task_ci95_low": float(np.percentile(raw_boot, 2.5)),
                "macro_raw_task_ci95_high": float(np.percentile(raw_boot, 97.5)),
                "macro_norm_task_mean_delta": float(task_means["norm_delta"].mean()),
                "macro_norm_task_ci95_low": float(np.percentile(norm_boot, 2.5)),
                "macro_norm_task_ci95_high": float(np.percentile(norm_boot, 97.5)),
                "n_tasks": int(len(tasks)),
                "n_tasks_positive_raw_mean": task_pos_raw,
                "n_tasks_positive_norm_mean": task_pos_norm,
                "n_bootstrap": int(iters),
            }
        )
    return pd.DataFrame(rows)


def budget_rounding_audit(routing_summary: pd.DataFrame, main_budget: float) -> pd.DataFrame:
    df = routing_summary[np.isclose(routing_summary["budget"].astype(float), main_budget)].copy()
    if df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "task": r["task"],
                "router": r["router"],
                "budget_upper_bound_percent": float(main_budget),
                "actual_call_rate_mean": float(r["call_rate_mean"]),
                "actual_minus_budget_percent": float(r["call_rate_mean"] - main_budget),
                "note": "rounding from selecting an integer number of molecules" if float(r["call_rate_mean"]) > main_budget else "",
            }
        )
    return pd.DataFrame(rows)


def safe_decision_audit(oof_long: pd.DataFrame, routing_summary: pd.DataFrame, main_budget: float) -> pd.DataFrame:
    counts = (
        oof_long.groupby(["task", "safe_decision"], as_index=False)
        .size()
        .pivot(index="task", columns="safe_decision", values="size")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    for col in ["call", "abstain"]:
        if col not in counts:
            counts[col] = 0
    call_rate = routing_summary[
        (routing_summary["router"] == FINAL_ROUTER) & np.isclose(routing_summary["budget"].astype(float), main_budget)
    ][["task", "call_rate_mean", "call_rate_std"]]
    out = counts.merge(call_rate, on="task", how="left")
    out["interpretation"] = np.where(
        (out["call"] > 0) & np.isclose(out["call_rate_mean"].fillna(0.0), 0.0),
        "LCB allowed at least one seed, but the test-time gated/top-budget intersection selected no molecules at the reported budget.",
        "",
    )
    return out[["task", "call", "abstain", "call_rate_mean", "call_rate_std", "interpretation"]]


def markdown_table(df: pd.DataFrame, columns: list[str], headers: list[str] | None = None) -> str:
    if headers is None:
        headers = columns
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(_fmt(val))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_revised_story(
    out_path: Path,
    norm_summary: pd.DataFrame,
    task_delta: pd.DataFrame,
    strat_ci: pd.DataFrame,
    safe_audit: pd.DataFrame,
    rounding_audit: pd.DataFrame,
    final_tables_path: Path,
) -> None:
    r9_summary = norm_summary[norm_summary["router"] == FINAL_ROUTER].iloc[0]
    ci_random = strat_ci[strat_ci["baseline"] == "random"].iloc[0]
    task_random = task_delta[task_delta["baseline"] == "random"].copy()
    task_random = task_random.sort_values("task")
    rounding_hits = rounding_audit[rounding_audit["actual_minus_budget_percent"] > 0.0].copy()
    lines = []
    lines.append("# VOILA-3D IF Revision: Identity, Normalized BIG, and Claim Audit\n")
    lines.append("Generated from existing 5-seed router outputs. No model was retrained for this audit.\n")
    lines.append("## 1. Method Identity Is Now Fixed\n")
    lines.append(
        "VOILA-3D should be written as the full adaptive information-acquisition system, not as a single final predictor. "
        "The R9 Utility-LCB Guardrail is the final acquisition policy inside VOILA-3D. It decides whether the selected 3D information is reliable enough to acquire under a budget upper bound.\n"
    )
    lines.append(
        "Therefore the paper should separate two result families: prediction performance and acquisition performance. "
        "The strict same-protocol final performance table reports the expert-system prediction quality. R9 should be evaluated in the acquisition tables using BIG, call rate, safe-decision counts, and task-level deltas.\n"
    )
    lines.append("Do not claim: `R9 achieves SOTA on all six tasks.`\n")
    lines.append(
        "Recommended claim: VOILA-3D reaches competitive/same-protocol expert-system prediction performance while the R9 acquisition policy largely eliminates harmful 3D acquisition compared with forced-budget and heuristic routers.\n"
    )
    lines.append("## 2. BIG Scale Audit\n")
    lines.append(
        "The original BIG integrates the change in the task primary metric along the budget curve after aligning direction: AUROC gain for classification and MAE reduction for regression. "
        "This is appropriate within a task, but pooled cross-task significance can be scale-sensitive. In particular, the R9-vs-random raw macro delta is strongly affected by FreeSolv.\n"
    )
    lines.append(
        "For reviewer-facing evidence, use task-level BIG and normalized BIG. Here normalized BIG is defined per task and seed as `BIG_norm = BIG / (|BIG_oracle| + eps)`, with `eps=1e-8`. "
        "Macro values are computed by averaging seed means within each task and then averaging equally across tasks.\n"
    )
    lines.append("### Normalized Macro BIG\n")
    show = norm_summary[["router_label", "macro_raw_BIG_task_mean", "macro_norm_BIG_task_mean", "macro_norm_BIG_task_std", "macro_norm_rank"]].copy()
    lines.append(markdown_table(show, show.columns.tolist(), ["Router", "Macro Raw BIG", "Macro Norm BIG", "Norm Task Std", "Rank"]))
    lines.append("\n")
    lines.append(
        f"R9 normalized macro BIG is {_fmt(float(r9_summary['macro_norm_BIG_task_mean']))}, ranked "
        f"{int(r9_summary['macro_norm_rank'])} among the evaluated routers. Its value being close to zero is expected: the guardrail is designed to avoid unreliable acquisition, not to force positive gains on every task.\n"
    )
    lines.append("### Task-Level R9 vs Random\n")
    lines.append(markdown_table(task_random[["task", "raw_delta_mean", "norm_delta_mean", "n_positive_raw_seed_pairs"]], ["task", "raw_delta_mean", "norm_delta_mean", "n_positive_raw_seed_pairs"], ["Task", "Raw Delta", "Norm Delta", "Positive Seeds"]))
    lines.append("\n")
    lines.append(
        f"Task-stratified bootstrap for R9 vs random gives raw macro delta {_fmt(float(ci_random['macro_raw_task_mean_delta']))} "
        f"with 95% CI [{_fmt(float(ci_random['macro_raw_task_ci95_low']))}, {_fmt(float(ci_random['macro_raw_task_ci95_high']))}], "
        f"and normalized macro delta {_fmt(float(ci_random['macro_norm_task_mean_delta']))} "
        f"with 95% CI [{_fmt(float(ci_random['macro_norm_task_ci95_low']))}, {_fmt(float(ci_random['macro_norm_task_ci95_high']))}]. "
        f"The direction is positive on {int(ci_random['n_tasks_positive_norm_mean'])}/{int(ci_random['n_tasks'])} task-level normalized means.\n"
    )
    lines.append("## 3. Numerical Consistency Fixes\n")
    lines.append("- R8 FreeSolv BIG should be reported as `-0.037071 +/- 0.043192`, not the older residual value `-0.032788`.\n")
    lines.append(
        "- BBBP can have `Call seeds = 1` but `0.00%` reported R9 call rate at the 20% budget summary because the LCB decision is seed-level permission, whereas the final test-time action is the intersection of top-budget selection and the learned gate threshold. In the stored curve, this intersection selected no BBBP molecules at the 20% budget point.\n"
    )
    if not rounding_hits.empty:
        tasks = ", ".join(sorted(rounding_hits["task"].unique()))
        lines.append(
            f"- Some nominal 20% budget rows are slightly above 20% ({tasks}) because selection is over an integer number of molecules: `round(n * budget / 100) / n`. State that budgets are nominal upper bounds implemented by integer molecule counts.\n"
        )
    lines.append("### Safe-Decision / Call-Rate Audit\n")
    lines.append(markdown_table(safe_audit, ["task", "call", "abstain", "call_rate_mean", "interpretation"], ["Task", "Call Seeds", "Abstain Seeds", "R9 Call Rate @20%", "Interpretation"]))
    lines.append("\n")
    lines.append("## 4. Final Claim Should Be About Avoiding Harmful Acquisition\n")
    lines.append(
        "The central IF story should be: more molecular information is not necessarily better. 3D information is costly, conditionally useful, noisy, and sometimes harmful. "
        "VOILA-3D reframes 3D fusion as reliability-aware information acquisition: cheap 2D evidence -> OOF counterfactual utility -> sample-level ranking -> budget upper bound -> OOF utility-LCB reliability test.\n"
    )
    lines.append(
        "The strongest empirical conclusion is not that R9 extracts large positive 3D gains everywhere. The stronger and better-supported claim is that naive 3D spending often causes negative transfer, while R9 learns to abstain when the evidence for 3D value is insufficient.\n"
    )
    lines.append("## 5. Recommended Results Order\n")
    lines.append("1. 3D is conditionally valuable rather than universally beneficial.\n")
    lines.append("2. Ranking better molecules is insufficient: R0-R6 objective sweep.\n")
    lines.append("3. Validation gating turns a fixed budget into an upper bound: R7 -> R8.\n")
    lines.append("4. Utility-LCB guardrail eliminates unreliable acquisitions: R8 -> R9.\n")
    lines.append("5. VOILA adapts modality usage across tasks: call rate and safe-decision counts.\n")
    lines.append("6. Robustness and fusion benchmark: external methods, MoE, gated fusion, missing/noisy modality, budget curves.\n")
    lines.append("7. Same-protocol predictive performance: expert-system prediction quality, not R9-as-SOTA.\n")
    lines.append("\n")
    lines.append(f"Full numeric tables remain in `{final_tables_path.name}`; this file is the corrected claim/statistics companion.\n")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def update_final_tables(
    out_path: Path,
    norm_summary: pd.DataFrame,
    task_delta: pd.DataFrame,
    strat_ci: pd.DataFrame,
    safe_audit: pd.DataFrame,
    rounding_audit: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# Reviewer-Safe Additional Numeric Tables\n")
    lines.append("These tables supplement `FINAL_NUMERIC_TABLES.md` and correct the cross-task interpretation of BIG.\n")
    lines.append("## Table A. Normalized Macro BIG\n")
    show = norm_summary[["router_label", "macro_raw_BIG_task_mean", "macro_norm_BIG_task_mean", "macro_norm_BIG_task_std", "macro_norm_rank"]].copy()
    lines.append(markdown_table(show, show.columns.tolist(), ["Router", "Macro Raw BIG", "Macro Norm BIG", "Norm Task Std", "Rank"]))
    lines.append("\n## Table B. R9 vs Baselines, Task-Stratified Bootstrap\n")
    show_ci = strat_ci[
        [
            "baseline",
            "macro_raw_task_mean_delta",
            "macro_raw_task_ci95_low",
            "macro_raw_task_ci95_high",
            "macro_norm_task_mean_delta",
            "macro_norm_task_ci95_low",
            "macro_norm_task_ci95_high",
            "n_tasks_positive_norm_mean",
        ]
    ].copy()
    lines.append(markdown_table(show_ci, show_ci.columns.tolist(), ["Baseline", "Raw Delta", "Raw CI Low", "Raw CI High", "Norm Delta", "Norm CI Low", "Norm CI High", "Positive Tasks"]))
    lines.append("\n## Table C. R9 vs Random, Task-Level Deltas\n")
    show_task = task_delta[task_delta["baseline"] == "random"][
        ["task", "raw_delta_mean", "raw_delta_std", "norm_delta_mean", "norm_delta_std", "n_positive_norm_seed_pairs"]
    ].sort_values("task")
    lines.append(markdown_table(show_task, show_task.columns.tolist(), ["Task", "Raw Delta", "Raw Std", "Norm Delta", "Norm Std", "Positive Seeds"]))
    lines.append("\n## Table D. Safe-Decision and Call-Rate Consistency\n")
    lines.append(markdown_table(safe_audit, ["task", "call", "abstain", "call_rate_mean", "interpretation"], ["Task", "Call Seeds", "Abstain Seeds", "R9 Call Rate @20%", "Interpretation"]))
    lines.append("\n## Table E. Budget Rounding Audit at Nominal 20%\n")
    show_round = rounding_audit[rounding_audit["actual_minus_budget_percent"] > 0.0][
        ["task", "router", "actual_call_rate_mean", "actual_minus_budget_percent", "note"]
    ].sort_values(["task", "router"])
    if show_round.empty:
        lines.append("No nominal 20% call-rate row exceeds 20%.\n")
    else:
        lines.append(markdown_table(show_round, show_round.columns.tolist(), ["Task", "Router", "Actual Call Rate", "Above 20%", "Note"]))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("results/if_router_lcb_final/results/if_router_gated_selection_lcb_v1_5seed"))
    p.add_argument("--report-dir", type=Path, default=Path("results/if_router_lcb_final"))
    p.add_argument("--main-budget", type=float, default=20.0)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--bootstrap-iters", type=int, default=5000)
    p.add_argument("--bootstrap-seed", type=int, default=3409)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    big_long = _read_csv(results_dir / "budget_integrated_gain_long.csv")
    routing_summary = _read_csv(results_dir / "routing_summary.csv")
    oof_long = _read_csv(results_dir / "oof_selection_long.csv")

    norm_long = build_normalized_big(big_long, eps=args.eps)
    norm_summary = summarize_macro(norm_long)
    task_delta = task_level_delta(norm_long, FINAL_ROUTER, BASELINES)
    strat_ci = stratified_bootstrap_delta(norm_long, FINAL_ROUTER, BASELINES, args.bootstrap_iters, args.bootstrap_seed)
    safe_audit = safe_decision_audit(oof_long, routing_summary, args.main_budget)
    rounding_audit = budget_rounding_audit(routing_summary, args.main_budget)

    norm_long.to_csv(report_dir / "normalized_big_long.csv", index=False)
    norm_summary.to_csv(report_dir / "normalized_big_macro_summary.csv", index=False)
    task_delta.to_csv(report_dir / "r9_task_level_delta_audit.csv", index=False)
    strat_ci.to_csv(report_dir / "r9_task_stratified_bootstrap_ci.csv", index=False)
    safe_audit.to_csv(report_dir / "r9_safe_decision_callrate_audit.csv", index=False)
    rounding_audit.to_csv(report_dir / "budget_rounding_audit_20pct.csv", index=False)

    write_revised_story(
        report_dir / "IF_REVIEWER_SAFE_R9_REVISION_CN.md",
        norm_summary,
        task_delta,
        strat_ci,
        safe_audit,
        rounding_audit,
        report_dir / "FINAL_NUMERIC_TABLES.md",
    )
    update_final_tables(
        report_dir / "REVIEWER_SAFE_ADDITIONAL_NUMERIC_TABLES.md",
        norm_summary,
        task_delta,
        strat_ci,
        safe_audit,
        rounding_audit,
    )

    print(f"Wrote reviewer-safe audit files to {report_dir}")


if __name__ == "__main__":
    main()
