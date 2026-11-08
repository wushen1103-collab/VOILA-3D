from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--cluster-bootstrap-iters", type=int, default=10000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = [Path(value) for value in args.root]
    files = []
    for root in roots:
        files.extend(sorted(root.rglob("selector_comparison.csv")))
    if not files:
        raise FileNotFoundError(f"No selector_comparison.csv under {roots}")
    all_rows = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    if args.tasks:
        all_rows = all_rows[all_rows["task"].isin(args.tasks)].copy()
        if all_rows.empty:
            raise ValueError(f"No rows found for requested tasks: {args.tasks}")
    key = ["task", "condition", "split_seed", "seed", "policy"]
    all_rows = all_rows.sort_values(key).drop_duplicates(key, keep="last")
    all_rows.to_csv(out_dir / "selector_all_rows.csv", index=False)

    seed_summary = (
        all_rows.groupby(["task", "condition", "policy"], as_index=False)
        .agg(
            n_pairs=("seed", "size"),
            n_splits=("split_seed", "nunique"),
            permission_rate=("permission", "mean"),
            decision_accuracy=("decision_correct", "mean"),
            false_activations=("false_activation", "sum"),
            false_rejections=("false_rejection", "sum"),
            full_alt_gain_mean=("full_alternative_test_gain", "mean"),
            policy_gain_mean=("policy_test_gain", "mean"),
            policy_gain_sd=("policy_test_gain", "std"),
            negative_transfer_mean=("negative_transfer", "mean"),
            missed_positive_mean=("missed_positive_gain", "mean"),
            regret_mean=("regret", "mean"),
        )
    )
    seed_summary.to_csv(out_dir / "selector_seed_pair_summary.csv", index=False)

    split_rows = (
        all_rows.groupby(["task", "condition", "split_seed", "policy"], as_index=False)
        .agg(
            permission_rate=("permission", "mean"),
            decision_accuracy=("decision_correct", "mean"),
            false_activation_rate=("false_activation", "mean"),
            false_rejection_rate=("false_rejection", "mean"),
            full_alt_gain=("full_alternative_test_gain", "mean"),
            policy_gain=("policy_test_gain", "mean"),
            negative_transfer=("negative_transfer", "mean"),
            missed_positive=("missed_positive_gain", "mean"),
            regret=("regret", "mean"),
        )
    )
    split_rows.to_csv(out_dir / "selector_split_level.csv", index=False)
    split_summary = (
        split_rows.groupby(["task", "condition", "policy"], as_index=False)
        .agg(
            n_splits=("split_seed", "nunique"),
            permission_rate_mean=("permission_rate", "mean"),
            decision_accuracy_mean=("decision_accuracy", "mean"),
            false_activation_rate_mean=("false_activation_rate", "mean"),
            false_rejection_rate_mean=("false_rejection_rate", "mean"),
            policy_gain_mean=("policy_gain", "mean"),
            policy_gain_sd_across_splits=("policy_gain", "std"),
            negative_transfer_mean=("negative_transfer", "mean"),
            missed_positive_mean=("missed_positive", "mean"),
            regret_mean=("regret", "mean"),
        )
    )
    split_summary.to_csv(out_dir / "selector_split_summary.csv", index=False)

    true3d = split_rows[split_rows["condition"] == "same_seed_true3d"].copy()
    macro = (
        true3d.groupby(["split_seed", "policy"], as_index=False)
        .agg(
            tasks=("task", "nunique"),
            permission_rate=("permission_rate", "mean"),
            decision_accuracy=("decision_accuracy", "mean"),
            false_activation_rate=("false_activation_rate", "mean"),
            false_rejection_rate=("false_rejection_rate", "mean"),
            negative_transfer=("negative_transfer", "mean"),
            regret=("regret", "mean"),
        )
    )
    macro.to_csv(out_dir / "selector_true3d_macro_by_split.csv", index=False)
    macro_summary = (
        macro.groupby("policy", as_index=False)
        .agg(
            n_splits=("split_seed", "nunique"),
            tasks_min=("tasks", "min"),
            permission_rate=("permission_rate", "mean"),
            decision_accuracy=("decision_accuracy", "mean"),
            false_activation_rate=("false_activation_rate", "mean"),
            false_rejection_rate=("false_rejection_rate", "mean"),
            negative_transfer=("negative_transfer", "mean"),
            regret=("regret", "mean"),
        )
    )
    macro_summary.to_csv(out_dir / "selector_true3d_macro_summary.csv", index=False)

    controls = all_rows[all_rows["condition"].isin(["seed_null_2d", "permuted_3d_placebo"])]
    control_summary = (
        controls.groupby(["condition", "policy"], as_index=False)
        .agg(
            n_pairs=("seed", "size"),
            permission_rate=("permission", "mean"),
            false_activations=("false_activation", "sum"),
            false_activation_rate=("false_activation", "mean"),
            policy_gain_mean=("policy_test_gain", "mean"),
            negative_transfer_mean=("negative_transfer", "mean"),
            regret_mean=("regret", "mean"),
        )
    )
    control_summary.to_csv(out_dir / "selector_control_summary.csv", index=False)

    def risk_benefit(group: pd.DataFrame) -> pd.Series:
        gain = group["full_alternative_test_gain"].to_numpy(float)
        permit = group["permission"].astype(bool).to_numpy()
        positive_available = np.maximum(gain, 0.0).sum()
        negative_available = np.maximum(-gain, 0.0).sum()
        positive_captured = np.maximum(gain[permit], 0.0).sum()
        negative_incurred = np.maximum(-gain[permit], 0.0).sum()
        return pd.Series(
            {
                "positive_gain_retention": (
                    positive_captured / positive_available if positive_available > 0 else np.nan
                ),
                "negative_transfer_avoidance": (
                    1.0 - negative_incurred / negative_available if negative_available > 0 else np.nan
                ),
                "net_policy_gain": group["policy_test_gain"].sum(),
                "total_regret": group["regret"].sum(),
            }
        )

    true3d_seed = all_rows[all_rows["condition"] == "same_seed_true3d"]
    risk_rows = []
    for (task, policy), group in true3d_seed.groupby(["task", "policy"]):
        values = risk_benefit(group).to_dict()
        risk_rows.append({"task": task, "policy": policy, **values})
    task_risk_benefit = pd.DataFrame(risk_rows)
    task_risk_benefit.to_csv(out_dir / "selector_true3d_task_risk_benefit.csv", index=False)
    risk_benefit_macro = (
        task_risk_benefit.groupby("policy", as_index=False)
        .agg(
            tasks=("task", "nunique"),
            positive_gain_retention_macro=("positive_gain_retention", "mean"),
            negative_transfer_avoidance_macro=("negative_transfer_avoidance", "mean"),
            task_net_gain_positive_fraction=("net_policy_gain", lambda x: float((x > 0).mean())),
            task_regret_mean=("total_regret", "mean"),
        )
    )
    risk_benefit_macro.to_csv(out_dir / "selector_true3d_risk_benefit_macro.csv", index=False)

    comparison_policies = [
        "TaskSelect",
        "ThresholdSelect-0.5pct",
        "ThresholdSelect-1pct",
        "TaskMetric-LCB",
    ]

    def macro_metrics(frame: pd.DataFrame, policy: str) -> dict[str, float]:
        policy_frame = frame[frame["policy"] == policy]
        task_values = []
        for _, group in policy_frame.groupby("task"):
            gain = group["full_alternative_test_gain"].to_numpy(float)
            permit = group["permission"].astype(bool).to_numpy()
            pos = np.maximum(gain, 0.0)
            neg = np.maximum(-gain, 0.0)
            task_values.append(
                {
                    "false_activation_rate": float(group["false_activation"].mean()),
                    "false_rejection_rate": float(group["false_rejection"].mean()),
                    "positive_gain_retention": (
                        float(np.maximum(gain[permit], 0.0).sum() / pos.sum())
                        if pos.sum() > 0
                        else np.nan
                    ),
                    "negative_transfer_avoidance": (
                        float(1.0 - np.maximum(-gain[permit], 0.0).sum() / neg.sum())
                        if neg.sum() > 0
                        else np.nan
                    ),
                }
            )
        values = pd.DataFrame(task_values)
        return {name: float(values[name].mean()) for name in values.columns}

    observed = {policy: macro_metrics(true3d_seed, policy) for policy in comparison_policies}
    rng = np.random.default_rng(20261108)
    boot_records = []
    task_groups = {task: group for task, group in true3d_seed.groupby("task")}
    for iteration in range(args.cluster_bootstrap_iters):
        sampled_parts = []
        for task, group in task_groups.items():
            split_values = np.sort(group["split_seed"].unique())
            sampled = rng.choice(split_values, size=len(split_values), replace=True)
            for draw, split_seed in enumerate(sampled):
                part = group[group["split_seed"] == split_seed].copy()
                part["bootstrap_split"] = draw
                sampled_parts.append(part)
        sampled_frame = pd.concat(sampled_parts, ignore_index=True)
        for policy in comparison_policies:
            metrics = macro_metrics(sampled_frame, policy)
            boot_records.append({"iteration": iteration, "policy": policy, **metrics})
    boot = pd.DataFrame(boot_records)
    ci_rows = []
    for policy in comparison_policies:
        current = boot[boot["policy"] == policy]
        for metric, point in observed[policy].items():
            ci_rows.append(
                {
                    "policy": policy,
                    "metric": metric,
                    "estimate": point,
                    "ci_low": float(current[metric].quantile(0.025)),
                    "ci_high": float(current[metric].quantile(0.975)),
                }
            )
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(out_dir / "selector_cluster_bootstrap_ci.csv", index=False)

    delta_rows = []
    baseline = boot[boot["policy"] == "TaskSelect"].set_index("iteration")
    for policy in comparison_policies:
        if policy == "TaskSelect":
            continue
        current = boot[boot["policy"] == policy].set_index("iteration")
        for metric in observed[policy]:
            delta = current[metric] - baseline[metric]
            delta_rows.append(
                {
                    "policy": policy,
                    "reference": "TaskSelect",
                    "metric": metric,
                    "estimate_delta": observed[policy][metric] - observed["TaskSelect"][metric],
                    "ci_low": float(delta.quantile(0.025)),
                    "ci_high": float(delta.quantile(0.975)),
                }
            )
    delta_ci = pd.DataFrame(delta_rows)
    delta_ci.to_csv(out_dir / "selector_cluster_bootstrap_delta_vs_taskselect.csv", index=False)

    report = [
        "# Task-Level Selector Comparison",
        "",
        f"Input files: {len(files)}",
        f"Unique task-condition-split-seed-policy rows: {len(all_rows)}",
        "",
        "## True-3D macro decision summary",
        "",
        macro_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Null/placebo activation summary",
        "",
        control_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Equal-task risk-benefit summary",
        "",
        risk_benefit_macro.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Split-cluster bootstrap deltas versus TaskSelect",
        "",
        delta_ci.to_markdown(index=False, floatfmt=".6f"),
        "",
        "The scaffold/random partition is the uncertainty unit. Model seeds are paired replicates within each partition and are not counted as independent chemical-space partitions.",
    ]
    (out_dir / "TASK_SELECTOR_COMPARISON_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
