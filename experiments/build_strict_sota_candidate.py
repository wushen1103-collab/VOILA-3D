from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from voila3d.metrics import higher_is_better


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-aware", default="results/comparisons/final_candidate_baseline_aware_budget20_5seed.csv")
    p.add_argument("--modern-summary", action="append", default=[])
    p.add_argument("--min-seeds", type=int, default=5)
    p.add_argument("--out-csv", default="results/comparisons/final_candidate_strict_sota_aware_5seed.csv")
    p.add_argument("--out-md", default="results/comparisons/final_candidate_strict_sota_aware_5seed.md")
    p.add_argument("--pool-csv", default="results/comparisons/strict_sota_candidate_pool_5seed.csv")
    return p.parse_args()


def _positive_delta(metric: str, reference: float, candidate: float) -> float:
    return candidate - reference if higher_is_better(metric) else reference - candidate


def _best(g: pd.DataFrame) -> pd.Series:
    metric = str(g["primary_metric"].iloc[0])
    idx = g["primary_mean"].idxmax() if higher_is_better(metric) else g["primary_mean"].idxmin()
    return g.loc[idx]


def _baseline_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "task": row["task"],
                "task_type": row["task_type"],
                "method": row["selected_source"],
                "method_group": row["selected_route_group"],
                "primary_metric": row["primary_metric"],
                "primary_mean": float(row["selected20"]),
                "primary_std": float(row["selected20_std"]),
                "n_seeds": int(row["n_seeds"]),
                "source": "ours_rerun_same_split",
                "call_rate": float(row["call_rate"]),
                "selection_family": "baseline_aware_start",
                "selection_note": row["audit_action"],
                "baseline_aware_mean": float(row["selected20"]),
                "baseline_aware_std": float(row["selected20_std"]),
            }
        )
    return pd.DataFrame(rows)


def _modern_rows(path: Path, min_seeds: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "method" not in df.columns and "variant" in df.columns:
        df = df.rename(columns={"variant": "method"})
        df["method"] = "Chemprop_" + df["method"].astype(str)
    rows = []
    for _, row in df.iterrows():
        n_seeds = int(row.get("n_seeds", 0))
        source = str(row.get("source", "ours_rerun_same_split"))
        if n_seeds < min_seeds or source != "ours_rerun_same_split":
            continue
        rows.append(
            {
                "task": row["task"],
                "task_type": row["task_type"],
                "method": row["method"],
                "method_group": row.get("method_group", "modern rerun baseline"),
                "primary_metric": row["primary_metric"],
                "primary_mean": float(row["primary_mean"]),
                "primary_std": float(row.get("primary_std", 0.0)),
                "n_seeds": n_seeds,
                "source": source,
                "call_rate": float(row.get("call_rate", 0.0)),
                "selection_family": "modern_same_split_rerun",
                "selection_note": f"from {path}",
            }
        )
    return pd.DataFrame(rows)


def _markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_csv(index=False)


def main() -> None:
    args = parse_args()
    baseline = _baseline_rows(Path(args.baseline_aware))
    modern_frames = [_modern_rows(Path(path), args.min_seeds) for path in args.modern_summary if Path(path).exists()]
    pool = pd.concat([baseline, *[m for m in modern_frames if not m.empty]], ignore_index=True)
    baseline_ref = baseline.set_index("task")
    pool.to_csv(args.pool_csv, index=False)

    rows = []
    for task, g in pool.groupby("task"):
        selected = _best(g)
        base = baseline_ref.loc[task]
        rows.append(
            {
                "variant": "strict_sota_aware_5seed",
                "task": task,
                "task_type": selected["task_type"],
                "primary_metric": selected["primary_metric"],
                "selected": float(selected["primary_mean"]),
                "selected_std": float(selected["primary_std"]),
                "selected_mean_std": f"{float(selected['primary_mean']):.6f} +/- {float(selected['primary_std']):.6f}",
                "selected_method": selected["method"],
                "selected_route_group": selected["method_group"],
                "selected_family": selected["selection_family"],
                "source": selected["source"],
                "call_rate": float(selected["call_rate"]),
                "n_seeds": int(selected["n_seeds"]),
                "baseline_aware": float(base["primary_mean"]),
                "baseline_aware_std": float(base["primary_std"]),
                "delta_vs_baseline_aware_positive": _positive_delta(
                    str(selected["primary_metric"]), float(base["primary_mean"]), float(selected["primary_mean"])
                ),
                "audit_action": "upgrade_to_modern_same_split_rerun"
                if selected["selection_family"] != "baseline_aware_start"
                else "keep_baseline_aware_selection",
            }
        )
    out = pd.DataFrame(rows).sort_values("task")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    lines = [
        "# Strict SOTA-Aware 5-Seed Candidate",
        "",
        f"Selection pool: `{args.pool_csv}`.",
        "",
        "Only same-split reruns with at least the configured seed count are eligible. Original-paper values are intentionally excluded from this selector.",
        "",
        _markdown(out),
        "",
    ]
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"[strict-sota-aware] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
