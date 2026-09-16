from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from voila3d.metrics import higher_is_better, primary_metric


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/fast_screen")
    return p.parse_args()


def _mean_std(vals: pd.Series) -> str:
    vals = pd.to_numeric(vals, errors="coerce")
    return f"{vals.mean():.4f} +/- {vals.std(ddof=0):.4f}"


def _markdown_table(df: pd.DataFrame) -> str:
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


def summarize_methods(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [c for c in ["MAE", "RMSE", "R2", "AUROC", "AUPRC", "Accuracy", "LogLoss"] if c in metrics.columns]
    for (task, task_type, method), g in metrics.groupby(["task", "task_type", "method"]):
        pmet = primary_metric(task_type)
        row = {
            "task": task,
            "task_type": task_type,
            "method": method,
            "primary_metric": pmet,
            "primary_mean": float(g[pmet].mean()),
            "primary_std": float(g[pmet].std(ddof=0)),
            "n_seeds": int(g["seed"].nunique()),
            "n_train": int(g["n_train"].iloc[0]),
            "n_val": int(g["n_val"].iloc[0]),
            "n_test": int(g["n_test"].iloc[0]),
        }
        for c in metric_cols:
            row[f"{c}_mean_std"] = _mean_std(g[c])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_tasks(method_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, g in method_summary.groupby("task"):
        task_type = g["task_type"].iloc[0]
        pmet = g["primary_metric"].iloc[0]
        base = g[g["method"] == "all_2d"]
        aug = g[g["method"] == "all_3d_aug"]
        if base.empty or aug.empty:
            continue
        b = float(base["primary_mean"].iloc[0])
        a = float(aug["primary_mean"].iloc[0])
        if higher_is_better(pmet):
            delta = a - b
            rel = delta / (abs(b) + 1e-12)
            category = "3D-helpful" if delta > 0.01 else ("3D-harmful/unstable" if delta < -0.01 else "3D-neutral")
        else:
            delta = b - a
            rel = delta / (abs(b) + 1e-12)
            category = "3D-helpful" if rel > 0.02 else ("3D-harmful/unstable" if rel < -0.02 else "3D-neutral")
        rows.append({
            "task": task,
            "task_type": task_type,
            "primary_metric": pmet,
            "all_2d": b,
            "all_3d_aug": a,
            "delta_positive_means_3d_better": delta,
            "relative_delta": rel,
            "screen_category": category,
            "n_train": int(base["n_train"].iloc[0]),
            "n_val": int(base["n_val"].iloc[0]),
            "n_test": int(base["n_test"].iloc[0]),
        })
    return pd.DataFrame(rows).sort_values(["screen_category", "task"])


def summarize_routing(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, task_type, router, budget), g in curves.groupby(["task", "task_type", "router", "budget"]):
        pmet = primary_metric(task_type)
        rows.append({
            "task": task,
            "task_type": task_type,
            "router": router,
            "budget": budget,
            "primary_metric": pmet,
            "primary_mean": float(g["primary_value"].mean()),
            "primary_std": float(g["primary_value"].std(ddof=0)),
            "call_rate_mean": float(g["call_rate"].mean()),
            "n_seeds": int(g["seed"].nunique()),
        })
    return pd.DataFrame(rows)


def write_experiment_log(results_dir: Path, task_summary: pd.DataFrame, routing_summary: pd.DataFrame) -> None:
    lines = []
    lines.append("# EXPERIMENT LOG")
    lines.append("")
    lines.append("## Fast Screen")
    lines.append("")
    lines.append("Purpose: identify whether lightweight RDKit 3D descriptors provide heterogeneous value and whether a VOI router beats simple same-budget routing baselines.")
    lines.append("")
    if not task_summary.empty:
        lines.append("### Task Categories")
        lines.append("")
        lines.append(_markdown_table(task_summary))
        lines.append("")
    if not routing_summary.empty:
        b20 = routing_summary[routing_summary["budget"].isin([20.0])]
        lines.append("### Budget 20 Routing Summary")
        lines.append("")
        lines.append(_markdown_table(b20))
        lines.append("")
    lines.append("### Notes")
    lines.append("")
    lines.append("- `all_3d_aug` is ECFP + cheap 2D descriptors + RDKit USR/USRCAT descriptors, not a final neural 3D expert.")
    lines.append("- `oracle` routing uses test labels and is reported only as an upper bound.")
    lines.append("- No plots were generated.")
    (results_dir.parent.parent / "EXPERIMENT_LOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    metrics = pd.read_csv(results_dir / "metrics_long.csv")
    method_summary = summarize_methods(metrics)
    method_summary.to_csv(results_dir / "method_summary.csv", index=False)
    task_summary = summarize_tasks(method_summary)
    task_summary.to_csv(results_dir / "task_summary.csv", index=False)
    routing_summary = pd.DataFrame()
    if (results_dir / "routing_curves.csv").exists():
        curves = pd.read_csv(results_dir / "routing_curves.csv")
        routing_summary = summarize_routing(curves)
        routing_summary.to_csv(results_dir / "routing_summary.csv", index=False)
    write_experiment_log(results_dir, task_summary, routing_summary)
    print("[summary] wrote method_summary.csv, task_summary.csv, routing_summary.csv, EXPERIMENT_LOG.md")


if __name__ == "__main__":
    main()
