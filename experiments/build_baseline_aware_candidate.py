from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voila3d.metrics import higher_is_better


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--final-candidate", default="results/comparisons/final_candidate_cv_refit_specialists_budget20_5seed.csv")
    p.add_argument("--baseline-summary", default="results/baseline_matrix_scaffold_balanced_5seed/method_summary.csv")
    p.add_argument("--out-csv", default="results/comparisons/final_candidate_baseline_aware_budget20_5seed.csv")
    p.add_argument("--out-md", default="results/comparisons/final_candidate_baseline_aware_budget20_5seed.md")
    return p.parse_args()


def _better(metric: str, a: float, b: float) -> bool:
    return a > b if higher_is_better(metric) else a < b


def _positive_delta(metric: str, reference: float, candidate: float) -> float:
    return candidate - reference if higher_is_better(metric) else reference - candidate


def _best_baseline(g: pd.DataFrame) -> pd.Series:
    metric = str(g["primary_metric"].iloc[0])
    idx = g["primary_mean"].idxmax() if higher_is_better(metric) else g["primary_mean"].idxmin()
    return g.loc[idx]


def _markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_csv(index=False)


def main() -> None:
    args = parse_args()
    final = pd.read_csv(args.final_candidate)
    baseline = pd.read_csv(args.baseline_summary)
    rows = []
    for _, row in final.iterrows():
        task = row["task"]
        metric = row["primary_metric"]
        b = _best_baseline(baseline[baseline["task"] == task])
        final20 = float(row["final20"])
        best_base = float(b["primary_mean"])
        use_baseline = _better(metric, best_base, final20)
        selected = best_base if use_baseline else final20
        selected_std = float(b["primary_std"]) if use_baseline else float(row["final20_std"])
        rows.append(
            {
                "variant": "baseline_aware_specialists",
                "task": task,
                "task_type": row["task_type"],
                "primary_metric": metric,
                "selected20": selected,
                "selected20_std": selected_std,
                "selected_source": str(b["method"]) if use_baseline else "VOILA_20pct_final",
                "selected_route_group": str(b["method_group"]) if use_baseline else "budgeted adaptive 2D/3D",
                "call_rate": 0.0 if use_baseline else float(row["call_rate"]),
                "ecfp_all_2d": float(row["ecfp_all_2d"]),
                "best_classic_or_descriptor_baseline": best_base,
                "best_classic_or_descriptor_method": str(b["method"]),
                "best_classic_or_descriptor_std": float(b["primary_std"]),
                "delta_vs_ecfp_positive": _positive_delta(metric, float(row["ecfp_all_2d"]), selected),
                "delta_vs_best_classic_or_descriptor_positive": _positive_delta(metric, best_base, selected),
                "original_voila20": final20,
                "original_voila20_std": float(row["final20_std"]),
                "audit_action": "abstain_to_stronger_2d_specialist" if use_baseline else "keep_voila_budgeted_route",
                "n_seeds": int(row["n_seeds"]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    lines = [
        "# Baseline-Aware Final Candidate",
        "",
        "This table is a post-audit candidate that adds strong same-split classical/descriptor specialists to the adaptive expert bank. If a classical 2D specialist beats the 20% 3D route, the policy abstains from 3D for that task.",
        "",
        _markdown(out),
        "",
    ]
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"[baseline-aware] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
