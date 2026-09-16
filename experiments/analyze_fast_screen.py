from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from voila3d.metrics import higher_is_better


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/fast_screen_scaffold_balanced")
    p.add_argument("--budget", type=float, default=20.0)
    p.add_argument("--primary-router", default="voi_router")
    p.add_argument("--oracle-router", default="oracle")
    return p.parse_args()


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        text = df.copy()
        for col in text.columns:
            if pd.api.types.is_float_dtype(text[col]):
                text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
            else:
                text[col] = text[col].astype(str)
        widths = {
            col: max(len(str(col)), int(text[col].map(lambda x: len(str(x))).max()))
            for col in text.columns
        }
        header = "| " + " | ".join(str(col).ljust(widths[col]) for col in text.columns) + " |"
        sep = "| " + " | ".join("-" * widths[col] for col in text.columns) + " |"
        rows = [
            "| " + " | ".join(str(row[col]).ljust(widths[col]) for col in text.columns) + " |"
            for _, row in text.iterrows()
        ]
        return "\n".join([header, sep, *rows])


def positive_delta(metric: str, reference: float, candidate: float) -> float:
    if higher_is_better(metric):
        return candidate - reference
    return reference - candidate


def build_budget_table(results_dir: Path, budget: float) -> pd.DataFrame:
    task_summary = pd.read_csv(results_dir / "task_summary.csv")
    routing = pd.read_csv(results_dir / "routing_summary.csv")
    rows = []
    for _, task_row in task_summary.iterrows():
        task = task_row["task"]
        metric = task_row["primary_metric"]
        all_2d = float(task_row["all_2d"])
        all_3d = float(task_row["all_3d_aug"])
        rows.append({
            "task": task,
            "method": "all_2d",
            "budget": 0.0,
            "primary_metric": metric,
            "primary_mean": all_2d,
            "delta_vs_all_2d_positive_better": 0.0,
            "call_rate_mean": 0.0,
        })
        rows.append({
            "task": task,
            "method": "all_3d_aug",
            "budget": 100.0,
            "primary_metric": metric,
            "primary_mean": all_3d,
            "delta_vs_all_2d_positive_better": positive_delta(metric, all_2d, all_3d),
            "call_rate_mean": 100.0,
        })
        sub = routing[(routing["task"] == task) & (routing["budget"] == budget)]
        for _, r in sub.iterrows():
            rows.append({
                "task": task,
                "method": r["router"],
                "budget": budget,
                "primary_metric": metric,
                "primary_mean": float(r["primary_mean"]),
                "primary_std": float(r["primary_std"]),
                "delta_vs_all_2d_positive_better": positive_delta(metric, all_2d, float(r["primary_mean"])),
                "call_rate_mean": float(r["call_rate_mean"]),
            })
    return pd.DataFrame(rows)


def build_best_router_table(results_dir: Path, primary_router: str) -> pd.DataFrame:
    routing = pd.read_csv(results_dir / "routing_summary.csv")
    oracle_like = routing["router"].astype(str).str.contains("oracle", case=False, regex=False)
    routing = routing[~oracle_like].copy()
    rows = []
    for (task, budget), g in routing.groupby(["task", "budget"]):
        metric = g["primary_metric"].iloc[0]
        idx = g["primary_mean"].idxmax() if higher_is_better(metric) else g["primary_mean"].idxmin()
        best = g.loc[idx]
        primary = g[g["router"] == primary_router]
        rows.append({
            "task": task,
            "budget": budget,
            "primary_metric": metric,
            "best_nonoracle_router": best["router"],
            "best_nonoracle_value": float(best["primary_mean"]),
            "primary_router": primary_router,
            "primary_router_value": float(primary["primary_mean"].iloc[0]) if not primary.empty else np.nan,
            "primary_router_is_best_nonoracle": bool(best["router"] == primary_router),
        })
    return pd.DataFrame(rows)


def build_conformer_qc(results_dir: Path) -> pd.DataFrame:
    manifest = pd.read_csv(results_dir / "conformer_manifest.csv")
    rows = []
    for task, g in manifest.groupby("task"):
        failures = g.loc[~g["success"].astype(bool), "failure_reason"].fillna("unknown")
        top_failure = failures.value_counts().index[0] if len(failures) else ""
        rows.append({
            "task": task,
            "n_molecules": int(len(g)),
            "conformer_success_rate": float(g["success"].astype(bool).mean()),
            "energy_nan_rate": float(g["energy_min"].isna().mean()),
            "mean_elapsed_sec": float(g["elapsed_sec"].mean()),
            "p95_elapsed_sec": float(g["elapsed_sec"].quantile(0.95)),
            "top_failure_reason": top_failure,
            "main_table_allowed_by_5pct_failure_rule": bool((1.0 - g["success"].astype(bool).mean()) <= 0.05),
        })
    return pd.DataFrame(rows).sort_values("task")


def build_pareto_auc(results_dir: Path) -> pd.DataFrame:
    routing = pd.read_csv(results_dir / "routing_summary.csv")
    rows = []
    for (task, router), g in routing.groupby(["task", "router"]):
        metric = g["primary_metric"].iloc[0]
        gg = g.sort_values("budget")
        y = gg["primary_mean"].to_numpy(float)
        if not higher_is_better(metric):
            y = -y
        auc = np.trapezoid(y, gg["budget"].to_numpy(float) / 100.0)
        rows.append({
            "task": task,
            "router": router,
            "primary_metric": metric,
            "signed_pareto_auc_higher_better": float(auc),
        })
    return pd.DataFrame(rows)


def write_analysis_md(
    results_dir: Path,
    tables_dir: Path,
    budget_table: pd.DataFrame,
    qc: pd.DataFrame,
    best_router: pd.DataFrame,
    budget: float,
    primary_router: str,
    oracle_router: str,
) -> None:
    task_summary = pd.read_csv(results_dir / "task_summary.csv")
    helpful = task_summary[task_summary["screen_category"] == "3D-helpful"]["task"].tolist()
    neutral = task_summary[task_summary["screen_category"] == "3D-neutral"]["task"].tolist()
    harmful = task_summary[task_summary["screen_category"] == "3D-harmful/unstable"]["task"].tolist()
    b = budget_table[budget_table["budget"] == budget]
    primary = b[b["method"] == primary_router]
    oracle = b[b["method"] == oracle_router]
    primary_wins = int((primary["delta_vs_all_2d_positive_better"] > 0).sum())
    oracle_wins = int((oracle["delta_vs_all_2d_positive_better"] > 0).sum())
    qc_flags = qc[(1.0 - qc["conformer_success_rate"] > 0.05) | (qc["energy_nan_rate"] > 0.05)]
    primary_best_rate = float(best_router["primary_router_is_best_nonoracle"].mean()) if len(best_router) else float("nan")

    lines = [
        "# Fast Screen Analysis",
        "",
        "## Raw Task Categories",
        "",
        markdown_table(task_summary),
        "",
        f"## Budget {budget:g} Comparison",
        "",
        markdown_table(budget_table[budget_table["budget"].isin([0.0, 20.0, 100.0])]),
        "",
        "## Conformer QC",
        "",
        markdown_table(qc),
        "",
        "## Key Findings",
        "",
        f"1. 3D-helpful tasks: {', '.join(helpful) if helpful else 'none'}; neutral: {', '.join(neutral) if neutral else 'none'}; harmful/unstable: {', '.join(harmful) if harmful else 'none'}.",
        f"2. At {budget:g}% 3D budget, `{primary_router}` beats all-2D on {primary_wins}/{len(primary)} tasks; `{oracle_router}` beats all-2D on {oracle_wins}/{len(oracle)} tasks.",
        f"3. Across task-budget pairs, `{primary_router}` is the best non-oracle router in {primary_best_rate:.1%} of cases.",
        "4. Lightweight RDKit USR/USRCAT 3D is a screening surrogate, not the final 3D expert. Negative all-3D results should trigger stronger 3D backbones and K-conformer sensitivity rather than killing the main idea.",
        "",
        "## Flags",
        "",
        markdown_table(qc_flags) if not qc_flags.empty else "No task violates the 5% conformer failure rule by the current `success` flag. Energy-NaN rate is tracked separately because force-field coverage can fail even when embedding succeeds.",
        "",
        "## Next Experiments",
        "",
        "1. Run K=10 conformer sensitivity on HIV, BBBP, BACE, ESOL, and FreeSolv to test whether ensemble 3D reduces noisy all-3D behavior.",
        "2. Install or enable XGBoost and rerun the fast screen to align the 2D baseline with the protocol.",
        "3. For HIV and any K=10-improved task, add a true 3D-lite expert (SchNet/PaiNN) before attempting Uni-Mol-scale baselines.",
        "4. For tasks where oracle helps but learned VOI does not, train router labels with cross-fitting rather than a single validation split.",
        "",
    ]
    (tables_dir / "ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    tables_dir = results_dir / "analysis_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    budget_table = build_budget_table(results_dir, args.budget)
    best_router = build_best_router_table(results_dir, args.primary_router)
    conformer_qc = build_conformer_qc(results_dir)
    pareto_auc = build_pareto_auc(results_dir)

    budget_table.to_csv(tables_dir / f"budget{int(args.budget)}_comparison.csv", index=False)
    best_router.to_csv(tables_dir / "best_router_by_budget.csv", index=False)
    conformer_qc.to_csv(tables_dir / "conformer_qc.csv", index=False)
    pareto_auc.to_csv(tables_dir / "pareto_auc.csv", index=False)

    (tables_dir / f"budget{int(args.budget)}_comparison.md").write_text(markdown_table(budget_table), encoding="utf-8")
    (tables_dir / "conformer_qc.md").write_text(markdown_table(conformer_qc), encoding="utf-8")
    (tables_dir / "pareto_auc.md").write_text(markdown_table(pareto_auc), encoding="utf-8")
    write_analysis_md(
        results_dir,
        tables_dir,
        budget_table,
        conformer_qc,
        best_router,
        args.budget,
        args.primary_router,
        args.oracle_router,
    )
    print(f"[analysis] wrote tables under {tables_dir}")


if __name__ == "__main__":
    main()
