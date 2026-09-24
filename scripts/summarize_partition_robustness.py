from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


POLICIES = {
    "random": "Random",
    "R7_oof_selected_top": "R7",
    "R8_oof_selected_gated": "R8",
    "R9_oof_safe_guardrail": "R9",
}
PRIMARY_METRIC = {
    "BACE": "AUROC",
    "BBBP": "AUROC",
    "ESOL": "MAE",
    "FreeSolv": "MAE",
    "HIV": "AUROC",
    "Lipophilicity": "MAE",
}
SPLIT_SEEDS = tuple(range(1, 6))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate five randomized scaffold-partition VOILA-3D runs."
    )
    parser.add_argument("--root", type=Path, default=Path("results/partition_robustness"))
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def population_summary(frame: pd.DataFrame, groups: list[str], value: str) -> pd.DataFrame:
    return (
        frame.groupby(groups)[value]
        .agg(mean="mean", sd=lambda x: x.std(ddof=0), n_partitions="count")
        .reset_index()
    )


def main() -> None:
    args = parse_args()
    root = args.root
    out_dir = args.out_dir or root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    big_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    for split_seed in SPLIT_SEEDS:
        gated = root / f"split{split_seed}_gated"
        big = pd.read_csv(require(gated / "budget_integrated_gain_long.csv"))
        curves = pd.read_csv(require(gated / "routing_curves.csv"))
        big["split_seed"] = split_seed
        curves["split_seed"] = split_seed
        big_frames.append(big)
        curve_frames.append(curves)
        for branch in ("cls", "reg"):
            metrics = pd.read_csv(
                require(root / f"split{split_seed}_{branch}" / "metrics_long.csv")
            )
            metrics["split_seed"] = split_seed
            metric_frames.append(metrics)

    big = pd.concat(big_frames, ignore_index=True)
    curves = pd.concat(curve_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    observed_tasks = set(big["task"])
    if observed_tasks != set(PRIMARY_METRIC):
        raise ValueError(f"Unexpected task set: {sorted(observed_tasks)}")
    if set(big["seed"].astype(int)) != {0, 1, 2, 3, 4}:
        raise ValueError("Every partition must contain model seeds 0--4.")

    expert_rows: list[dict[str, object]] = []
    for (split_seed, task, method), group in metrics.groupby(
        ["split_seed", "task", "method"]
    ):
        if method not in {"all_2d", "all_3d_aug"}:
            continue
        metric = PRIMARY_METRIC[task]
        expert_rows.append(
            {
                "split_seed": split_seed,
                "task": task,
                "method": method,
                "metric": metric,
                "partition_mean": group[metric].mean(),
            }
        )
    expert_partition = pd.DataFrame(expert_rows)
    expert_summary = population_summary(
        expert_partition, ["task", "method", "metric"], "partition_mean"
    )
    expert_summary.to_csv(out_dir / "expert_summary.csv", index=False)

    policy_big = big[big["router"].isin(POLICIES)].copy()
    policy_big["policy"] = policy_big["router"].map(POLICIES)
    big_partition = (
        policy_big.groupby(["split_seed", "task", "task_type", "policy"], as_index=False)[
            "BIG"
        ]
        .mean()
        .rename(columns={"BIG": "partition_mean_BIG"})
    )
    big_summary = population_summary(
        big_partition, ["task", "task_type", "policy"], "partition_mean_BIG"
    )
    big_summary.to_csv(out_dir / "policy_big_summary.csv", index=False)

    calls = curves[
        np.isclose(curves["budget"], 20.0) & curves["router"].isin(POLICIES)
    ].copy()
    calls["policy"] = calls["router"].map(POLICIES)
    call_partition = (
        calls.groupby(["split_seed", "task", "task_type", "policy"], as_index=False)[
            "call_rate"
        ]
        .mean()
        .rename(columns={"call_rate": "partition_mean_call_rate"})
    )
    call_summary = population_summary(
        call_partition, ["task", "task_type", "policy"], "partition_mean_call_rate"
    )
    call_summary.to_csv(out_dir / "policy_call_summary.csv", index=False)

    oracle = big.loc[
        big["router"] == "oracle", ["split_seed", "task", "seed", "BIG"]
    ].rename(columns={"BIG": "oracle_BIG"})
    macro_source = policy_big.merge(oracle, on=["split_seed", "task", "seed"], how="left")
    macro_source["normalized_BIG"] = macro_source["BIG"] / (
        macro_source["oracle_BIG"].abs() + 1e-8
    )
    task_partition = macro_source.groupby(
        ["split_seed", "task", "policy"], as_index=False
    )[["BIG", "normalized_BIG"]].mean()
    macro_partition = (
        task_partition.groupby(["split_seed", "policy"], as_index=False)[
            ["BIG", "normalized_BIG"]
        ]
        .mean()
        .rename(columns={"BIG": "macro_BIG", "normalized_BIG": "macro_normalized_BIG"})
    )
    macro_calls = (
        call_partition.groupby(["split_seed", "policy"], as_index=False)[
            "partition_mean_call_rate"
        ]
        .mean()
        .rename(columns={"partition_mean_call_rate": "macro_call_rate"})
    )
    macro_partition = macro_partition.merge(macro_calls, on=["split_seed", "policy"])
    macro_partition.to_csv(out_dir / "macro_by_partition.csv", index=False)
    macro_summary = (
        macro_partition.groupby("policy")
        .agg(
            macro_BIG_mean=("macro_BIG", "mean"),
            macro_BIG_sd=("macro_BIG", lambda x: x.std(ddof=0)),
            macro_normalized_BIG_mean=("macro_normalized_BIG", "mean"),
            macro_normalized_BIG_sd=("macro_normalized_BIG", lambda x: x.std(ddof=0)),
            macro_call_rate_mean=("macro_call_rate", "mean"),
            macro_call_rate_sd=("macro_call_rate", lambda x: x.std(ddof=0)),
        )
        .reset_index()
    )
    macro_summary.to_csv(out_dir / "macro_summary.csv", index=False)

    paired = task_partition[task_partition["policy"] == "R9"].merge(
        task_partition[task_partition["policy"] == "Random"],
        on=["split_seed", "task"],
        suffixes=("_R9", "_Random"),
    )
    paired["delta_R9_minus_random"] = paired["BIG_R9"] - paired["BIG_Random"]
    paired.to_csv(out_dir / "r9_vs_random_by_task_partition.csv", index=False)

    r9 = macro_partition[macro_partition["policy"] == "R9"]
    random = macro_partition[macro_partition["policy"] == "Random"]
    report = [
        "# Scaffold-partition robustness summary",
        "",
        "Each partition estimate averages five model seeds; dispersion is across split seeds 1--5.",
        "",
        f"- R9 macro BIG: {r9['macro_BIG'].mean():.6f} +/- {r9['macro_BIG'].std(ddof=0):.6f}.",
        f"- Random macro BIG: {random['macro_BIG'].mean():.6f} +/- {random['macro_BIG'].std(ddof=0):.6f}.",
        f"- R9 exceeded random in {(paired['delta_R9_minus_random'] > 0).sum()}/{len(paired)} task-partition comparisons.",
        f"- R9 macro call rate: {r9['macro_call_rate'].mean():.2f}% +/- {r9['macro_call_rate'].std(ddof=0):.2f}%.",
    ]
    (out_dir / "PARTITION_ROBUSTNESS.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(f"Wrote partition summaries to {out_dir}")


if __name__ == "__main__":
    main()
