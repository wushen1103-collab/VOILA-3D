from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from voila3d.metrics import classification_metrics, primary_metric, regression_metrics
from voila3d.routing import true_benefit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--usr-results-dir", required=True)
    p.add_argument("--rich-results-dir", required=True)
    p.add_argument("--scalar-results-dir", default=None)
    p.add_argument("--scalar-tasks", nargs="*", default=None)
    p.add_argument(
        "--preferred-expert-task",
        action="append",
        default=[],
        help="Optional task-level expert choice such as ESOL=scalar. Keeps per-molecule VOI ranking within that expert.",
    )
    p.add_argument(
        "--task-usr-results-dir",
        action="append",
        default=[],
        help="Optional task-level replacement for the primary 2D/USR source, such as Lipophilicity=results/run_dir.",
    )
    p.add_argument(
        "--abstain-task",
        action="append",
        default=[],
        help="Task for which the metric-aware hybrid should spend zero 3D budget and keep the active 2D prediction.",
    )
    p.add_argument("--large-classification-usr-results-dir", default=None)
    p.add_argument("--large-classification-min-n", type=int, default=5000)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--budgets", nargs="+", type=float, default=[0, 5, 10, 20, 40, 60, 80, 100])
    p.add_argument("--positive-threshold", type=float, default=0.0)
    p.add_argument("--regression-positive-threshold", type=float, default=None)
    p.add_argument("--classification-positive-threshold", type=float, default=None)
    return p.parse_args()


def select_top(scores: np.ndarray, budget: float) -> np.ndarray:
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


def select_positive_top(scores: np.ndarray, budget: float, threshold: float = 0.0) -> np.ndarray:
    selected = select_top(scores, budget)
    selected &= np.asarray(scores, dtype=float) > threshold
    return selected


def routed_from_choice(
    pred2d: np.ndarray,
    pred_usr: np.ndarray,
    pred_rich: np.ndarray,
    selected: np.ndarray,
    choose_rich: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = pred2d.copy()
    usr_sel = selected & ~choose_rich
    rich_sel = selected & choose_rich
    pred[usr_sel] = pred_usr[usr_sel]
    pred[rich_sel] = pred_rich[rich_sel]
    return pred, usr_sel, rich_sel


def routed_from_expert_choice(
    pred2d: np.ndarray,
    expert_preds: dict[str, np.ndarray],
    selected: np.ndarray,
    choice: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    pred = pred2d.copy()
    selections = {name: selected & (choice == name) for name in expert_preds}
    for name, sel in selections.items():
        pred[sel] = expert_preds[name][sel]
    return pred, selections


def eval_pred(
    task: str,
    task_type: str,
    seed: int,
    router: str,
    budget: float,
    y: np.ndarray,
    pred: np.ndarray,
    selected: np.ndarray,
    usr_sel: np.ndarray,
    rich_sel: np.ndarray,
    scalar_sel: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    mets = regression_metrics(y, pred) if task_type == "regression" else classification_metrics(y, pred)
    metric = primary_metric(task_type)
    row: dict[str, float | int | str] = {
        "task": task,
        "task_type": task_type,
        "seed": seed,
        "router": router,
        "budget": float(budget),
        "call_rate": float(np.mean(selected) * 100.0),
        "usr_call_rate": float(np.mean(usr_sel) * 100.0),
        "rich_call_rate": float(np.mean(rich_sel) * 100.0),
        "scalar_call_rate": float(np.mean(scalar_sel) * 100.0) if scalar_sel is not None else 0.0,
        "primary_metric": metric,
        "primary_value": float(mets[metric]),
    }
    row.update({f"metric_{k}": v for k, v in mets.items()})
    return row


def metric_rows_for_seed(
    task: str,
    task_type: str,
    seed: int,
    split_info: pd.Series,
    pred_map: dict[str, np.ndarray],
    y: np.ndarray,
) -> list[dict[str, float | int | str]]:
    rows = []
    for method, pred in pred_map.items():
        mets = regression_metrics(y, pred) if task_type == "regression" else classification_metrics(y, pred)
        row: dict[str, float | int | str] = {
            "task": task,
            "task_type": task_type,
            "seed": seed,
            "method": method,
            "split": split_info["split"],
            "n_train": int(split_info["n_train"]),
            "n_val": int(split_info["n_val"]),
            "n_test": int(split_info["n_test"]),
            "xgb_device_2d": split_info["xgb_device_2d"],
            "xgb_device_3d": split_info["xgb_device_3d"],
        }
        row.update(mets)
        rows.append(row)
    return rows


def parse_preferred_experts(items: list[str]) -> dict[str, str]:
    preferred: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected TASK=EXPERT for --preferred-expert-task, got {item!r}")
        task, expert = item.split("=", 1)
        task = task.strip()
        expert = expert.strip()
        if not task or not expert:
            raise ValueError(f"Expected non-empty TASK=EXPERT for --preferred-expert-task, got {item!r}")
        if expert not in {"usr", "rich", "scalar"}:
            raise ValueError(f"Unsupported preferred expert {expert!r}; expected usr, rich, or scalar")
        preferred[task] = expert
    return preferred


def parse_task_dirs(items: list[str]) -> dict[str, Path]:
    task_dirs: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected TASK=DIR for --task-usr-results-dir, got {item!r}")
        task, path = item.split("=", 1)
        task = task.strip()
        path = path.strip()
        if not task or not path:
            raise ValueError(f"Expected non-empty TASK=DIR for --task-usr-results-dir, got {item!r}")
        task_dirs[task] = Path(path)
    return task_dirs


def main() -> None:
    args = parse_args()
    usr_dir = Path(args.usr_results_dir)
    rich_dir = Path(args.rich_results_dir)
    scalar_dir = Path(args.scalar_results_dir) if args.scalar_results_dir else None
    scalar_tasks = set(args.scalar_tasks) if args.scalar_tasks is not None else None
    large_cls_usr_dir = Path(args.large_classification_usr_results_dir) if args.large_classification_usr_results_dir else None
    preferred_expert_by_task = parse_preferred_experts(args.preferred_expert_task)
    task_usr_dir_overrides = parse_task_dirs(args.task_usr_results_dir)
    abstain_tasks = set(args.abstain_task)
    out_dir = Path(args.out_dir)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    task_summary = pd.read_csv(usr_dir / "task_summary.csv").set_index("task")
    usr_metrics_by_dir = {usr_dir: pd.read_csv(usr_dir / "metrics_long.csv")}
    if large_cls_usr_dir is not None:
        usr_metrics_by_dir[large_cls_usr_dir] = pd.read_csv(large_cls_usr_dir / "metrics_long.csv")
    for override_dir in task_usr_dir_overrides.values():
        if override_dir not in usr_metrics_by_dir:
            usr_metrics_by_dir[override_dir] = pd.read_csv(override_dir / "metrics_long.csv")
    tasks = sorted(task_summary.index.tolist())

    all_metric_rows: list[dict[str, float | int | str]] = []
    all_route_rows: list[dict[str, float | int | str]] = []
    run_summary = []

    for task in tasks:
        task_type = str(task_summary.loc[task, "task_type"])
        n_task = int(task_summary.loc[task, ["n_train", "n_val", "n_test"]].sum())
        task_usr_dir = task_usr_dir_overrides.get(task, usr_dir)
        if (
            task_type == "classification"
            and large_cls_usr_dir is not None
            and task not in task_usr_dir_overrides
            and n_task >= args.large_classification_min_n
            and (large_cls_usr_dir / "predictions" / f"{task}_predictions.csv").exists()
        ):
            task_usr_dir = large_cls_usr_dir
        usr_metrics = usr_metrics_by_dir[task_usr_dir]
        usr = pd.read_csv(task_usr_dir / "predictions" / f"{task}_predictions.csv")
        rich = pd.read_csv(rich_dir / "predictions" / f"{task}_predictions.csv")
        merged = usr.merge(
            rich[["seed", "mol_id", "pred3d_aug", "voi_score", "oracle_benefit"]],
            on=["seed", "mol_id"],
            how="inner",
            suffixes=("_usr", "_rich"),
        )
        has_scalar = (
            task_type == "regression"
            and scalar_dir is not None
            and (scalar_tasks is None or task in scalar_tasks)
            and (scalar_dir / "predictions" / f"{task}_predictions.csv").exists()
        )
        if has_scalar:
            scalar = pd.read_csv(scalar_dir / "predictions" / f"{task}_predictions.csv")
            scalar = scalar[["seed", "mol_id", "pred3d_aug", "voi_score", "oracle_benefit"]].rename(
                columns={
                    "pred3d_aug": "pred3d_aug_scalar",
                    "voi_score": "voi_score_scalar",
                    "oracle_benefit": "oracle_benefit_scalar",
                }
            )
            merged = merged.merge(scalar, on=["seed", "mol_id"], how="inner")
        positive_threshold = (
            args.classification_positive_threshold
            if task_type == "classification"
            else args.regression_positive_threshold
        )
        if positive_threshold is None:
            positive_threshold = args.positive_threshold
        route_rows: list[dict[str, float | int | str]] = []
        pred_rows = []

        for seed, g0 in merged.groupby("seed"):
            g = g0.sort_values("mol_id").reset_index(drop=True)
            y = g["y"].to_numpy()
            pred2d = g["pred2d"].to_numpy(float)
            pred_usr = g["pred3d_aug_usr"].to_numpy(float)
            pred_rich = g["pred3d_aug_rich"].to_numpy(float)
            pred_scalar = g["pred3d_aug_scalar"].to_numpy(float) if has_scalar else None
            score_usr = g["voi_score_usr"].to_numpy(float)
            score_rich = g["voi_score_rich"].to_numpy(float)
            score_scalar = g["voi_score_scalar"].to_numpy(float) if has_scalar else None
            expert_scores = {"usr": score_usr, "rich": score_rich}
            expert_preds = {"usr": pred_usr, "rich": pred_rich}
            if has_scalar and pred_scalar is not None and score_scalar is not None:
                expert_scores["scalar"] = score_scalar
                expert_preds["scalar"] = pred_scalar
            score_names = list(expert_scores)
            score_matrix = np.vstack([expert_scores[name] for name in score_names])
            best_pos = np.argmax(score_matrix, axis=0)
            score_max = score_matrix[best_pos, np.arange(score_matrix.shape[1])]
            choice = np.asarray([score_names[i] for i in best_pos], dtype=object)
            choose_rich = choice == "rich"

            if task_type == "classification":
                benefit_usr = true_benefit(task_type, y, pred2d, pred_usr, classification_mode="auc_contrib")
                benefit_rich = true_benefit(task_type, y, pred2d, pred_rich, classification_mode="auc_contrib")
                benefit_scalar = None
            else:
                benefit_usr = true_benefit(task_type, y, pred2d, pred_usr)
                benefit_rich = true_benefit(task_type, y, pred2d, pred_rich)
                benefit_scalar = true_benefit(task_type, y, pred2d, pred_scalar) if has_scalar and pred_scalar is not None else None
            benefit_map = {"usr": benefit_usr, "rich": benefit_rich}
            if benefit_scalar is not None:
                benefit_map["scalar"] = benefit_scalar
            benefit_names = list(benefit_map)
            benefit_matrix = np.vstack([benefit_map[name] for name in benefit_names])
            best_benefit_pos = np.argmax(benefit_matrix, axis=0)
            oracle_score = benefit_matrix[best_benefit_pos, np.arange(benefit_matrix.shape[1])]
            oracle_choice = np.asarray([benefit_names[i] for i in best_benefit_pos], dtype=object)
            oracle_choose_rich = oracle_choice == "rich"

            split_info = usr_metrics[
                (usr_metrics["task"] == task) & (usr_metrics["seed"] == seed) & (usr_metrics["method"] == "all_2d")
            ].iloc[0]
            baseline_preds = {"all_2d": pred2d, "all_3d_aug": pred_usr, "all_rich3d_aug": pred_rich}
            if has_scalar and pred_scalar is not None:
                baseline_preds["all_scalar3d_aug"] = pred_scalar
            all_metric_rows.extend(metric_rows_for_seed(task, task_type, int(seed), split_info, baseline_preds, y))

            for budget in args.budgets:
                zero = np.zeros(len(g), dtype=bool)

                usr_top = select_top(score_usr, budget)
                pred = np.where(usr_top, pred_usr, pred2d)
                route_rows.append(eval_pred(task, task_type, int(seed), "usr_auc_router", budget, y, pred, usr_top, usr_top, zero))

                usr_pos = select_positive_top(score_usr, budget, positive_threshold)
                pred = np.where(usr_pos, pred_usr, pred2d)
                route_rows.append(eval_pred(task, task_type, int(seed), "usr_auc_positive", budget, y, pred, usr_pos, usr_pos, zero))

                rich_top = select_top(score_rich, budget)
                pred = np.where(rich_top, pred_rich, pred2d)
                route_rows.append(eval_pred(task, task_type, int(seed), "rich_router", budget, y, pred, rich_top, zero, rich_top))

                if has_scalar and pred_scalar is not None and score_scalar is not None:
                    scalar_top = select_top(score_scalar, budget)
                    pred = np.where(scalar_top, pred_scalar, pred2d)
                    route_rows.append(
                        eval_pred(
                            task,
                            task_type,
                            int(seed),
                            "scalar_router",
                            budget,
                            y,
                            pred,
                            scalar_top,
                            zero,
                            zero,
                            scalar_top,
                        )
                    )

                max_sel = select_top(score_max, budget)
                pred, selections = routed_from_expert_choice(pred2d, expert_preds, max_sel, choice)
                usr_sel = selections.get("usr", zero)
                rich_sel = selections.get("rich", zero)
                scalar_sel = selections.get("scalar", zero)
                route_rows.append(
                    eval_pred(task, task_type, int(seed), "multi_auc_raw_max", budget, y, pred, max_sel, usr_sel, rich_sel, scalar_sel)
                )

                pos_sel = select_positive_top(score_max, budget, positive_threshold)
                pred, selections = routed_from_expert_choice(pred2d, expert_preds, pos_sel, choice)
                usr_sel = selections.get("usr", zero)
                rich_sel = selections.get("rich", zero)
                scalar_sel = selections.get("scalar", zero)
                route_rows.append(
                    eval_pred(
                        task,
                        task_type,
                        int(seed),
                        "multi_auc_raw_positive",
                        budget,
                        y,
                        pred,
                        pos_sel,
                        usr_sel,
                        rich_sel,
                        scalar_sel,
                    )
                )

                if task_type == "classification":
                    if task in abstain_tasks:
                        hybrid_sel = zero
                        hybrid_pred = pred2d
                        hybrid_usr_sel = zero
                        hybrid_rich_sel = zero
                        hybrid_scalar_sel = zero
                        hybrid_pos_sel = zero
                        hybrid_pos_pred = pred2d
                        hybrid_pos_usr_sel = zero
                        hybrid_pos_rich_sel = zero
                        hybrid_pos_scalar_sel = zero
                    else:
                        hybrid_sel = usr_top
                        hybrid_pred = np.where(hybrid_sel, pred_usr, pred2d)
                        hybrid_usr_sel = hybrid_sel
                        hybrid_rich_sel = zero
                        hybrid_scalar_sel = zero
                        hybrid_pos_sel = usr_pos
                        hybrid_pos_pred = np.where(hybrid_pos_sel, pred_usr, pred2d)
                        hybrid_pos_usr_sel = hybrid_pos_sel
                        hybrid_pos_rich_sel = zero
                        hybrid_pos_scalar_sel = zero
                else:
                    preferred_expert = preferred_expert_by_task.get(task)
                    if task in abstain_tasks:
                        hybrid_sel = zero
                        hybrid_pred = pred2d
                        hybrid_usr_sel = zero
                        hybrid_rich_sel = zero
                        hybrid_scalar_sel = zero
                        hybrid_pos_sel = zero
                        hybrid_pos_pred = pred2d
                        hybrid_pos_usr_sel = zero
                        hybrid_pos_rich_sel = zero
                        hybrid_pos_scalar_sel = zero
                    elif preferred_expert is not None and preferred_expert in expert_preds:
                        preferred_choice = np.full(len(g), preferred_expert, dtype=object)

                        hybrid_sel = select_top(expert_scores[preferred_expert], budget)
                        hybrid_pred, hybrid_selections = routed_from_expert_choice(
                            pred2d, expert_preds, hybrid_sel, preferred_choice
                        )
                        hybrid_usr_sel = hybrid_selections.get("usr", zero)
                        hybrid_rich_sel = hybrid_selections.get("rich", zero)
                        hybrid_scalar_sel = hybrid_selections.get("scalar", zero)

                        hybrid_pos_sel = select_positive_top(expert_scores[preferred_expert], budget, positive_threshold)
                        hybrid_pos_pred, hybrid_pos_selections = routed_from_expert_choice(
                            pred2d, expert_preds, hybrid_pos_sel, preferred_choice
                        )
                        hybrid_pos_usr_sel = hybrid_pos_selections.get("usr", zero)
                        hybrid_pos_rich_sel = hybrid_pos_selections.get("rich", zero)
                        hybrid_pos_scalar_sel = hybrid_pos_selections.get("scalar", zero)
                    else:
                        hybrid_sel = pos_sel
                        hybrid_pred = pred
                        hybrid_usr_sel = usr_sel
                        hybrid_rich_sel = rich_sel
                        hybrid_scalar_sel = scalar_sel
                        hybrid_pos_sel = pos_sel
                        hybrid_pos_pred = pred
                        hybrid_pos_usr_sel = usr_sel
                        hybrid_pos_rich_sel = rich_sel
                        hybrid_pos_scalar_sel = scalar_sel

                route_rows.append(
                    eval_pred(
                        task,
                        task_type,
                        int(seed),
                        "metric_aware_hybrid",
                        budget,
                        y,
                        hybrid_pred,
                        hybrid_sel,
                        hybrid_usr_sel,
                        hybrid_rich_sel,
                        hybrid_scalar_sel,
                    )
                )
                route_rows.append(
                    eval_pred(
                        task,
                        task_type,
                        int(seed),
                        "metric_aware_positive",
                        budget,
                        y,
                        hybrid_pos_pred,
                        hybrid_pos_sel,
                        hybrid_pos_usr_sel,
                        hybrid_pos_rich_sel,
                        hybrid_pos_scalar_sel,
                    )
                )

                oracle_sel = select_top(oracle_score, budget)
                pred, selections = routed_from_expert_choice(pred2d, expert_preds, oracle_sel, oracle_choice)
                usr_sel = selections.get("usr", zero)
                rich_sel = selections.get("rich", zero)
                scalar_sel = selections.get("scalar", zero)
                route_rows.append(
                    eval_pred(task, task_type, int(seed), "multi_auc_oracle", budget, y, pred, oracle_sel, usr_sel, rich_sel, scalar_sel)
                )

            pred_rows.append(
                pd.DataFrame(
                    {
                        "task": task,
                        "seed": int(seed),
                        "mol_id": g["mol_id"],
                        "y": y,
                        "pred2d": pred2d,
                        "pred_usr": pred_usr,
                        "score_usr": score_usr,
                        "benefit_usr": benefit_usr,
                        "pred_rich": pred_rich,
                        "score_rich": score_rich,
                        "benefit_rich": benefit_rich,
                        "pred_scalar": pred_scalar if pred_scalar is not None else np.nan,
                        "score_scalar": score_scalar if score_scalar is not None else np.nan,
                        "benefit_scalar": benefit_scalar if benefit_scalar is not None else np.nan,
                        "score_max": score_max,
                        "choice_by_score": choice,
                        "oracle_score": oracle_score,
                        "oracle_choice": oracle_choice,
                    }
                )
            )

        pd.DataFrame(route_rows).to_csv(out_dir / f"routing_{task}.csv", index=False)
        pd.concat(pred_rows, ignore_index=True).to_csv(out_dir / "predictions" / f"{task}_hybrid_predictions.csv", index=False)
        all_route_rows.extend(route_rows)
        run_summary.append({"task": task, "n": int(len(merged) // merged["seed"].nunique()), "status": "done"})

    pd.DataFrame(all_metric_rows).to_csv(out_dir / "metrics_long.csv", index=False)
    pd.DataFrame(all_route_rows).to_csv(out_dir / "routing_curves.csv", index=False)
    pd.DataFrame(run_summary).to_csv(out_dir / "run_summary.csv", index=False)
    for path in usr_dir.glob("conformer_manifest*.csv"):
        (out_dir / path.name).write_bytes(path.read_bytes())
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "source_usr": str(usr_dir),
                "source_rich": str(rich_dir),
                "source_scalar": str(scalar_dir) if scalar_dir is not None else None,
                "scalar_tasks": sorted(scalar_tasks) if scalar_tasks is not None else None,
                "large_classification_usr": str(large_cls_usr_dir) if large_cls_usr_dir is not None else None,
                "large_classification_min_n": args.large_classification_min_n,
                "preferred_expert_by_task": preferred_expert_by_task,
                "task_usr_dir_overrides": {task: str(path) for task, path in task_usr_dir_overrides.items()},
                "abstain_tasks": sorted(abstain_tasks),
                "budgets": args.budgets,
                "positive_threshold": args.positive_threshold,
                "regression_positive_threshold": args.regression_positive_threshold,
                "classification_positive_threshold": args.classification_positive_threshold,
                "note": "Offline hybrid routing over existing USR auc_contrib and rich RDKit predictions; no retraining.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
