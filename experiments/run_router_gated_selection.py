from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_router_objective_sweep import (
    TASK_TYPE,
    _ap_safe,
    _auc_safe,
    _budget_integrated_gain,
    _feature_groups,
    _load_cache,
    _load_labels,
    _metric_value,
    _ndcg_at_budget,
    _positive_performance,
    _precision_at_budget,
    _read_json,
    _spearman_safe,
    _summarize,
    _train_and_score,
    _utility_from_predictions,
)
from experiments.run_if_router_ablation import _support_features
from voila3d.metrics import classification_metrics, primary_metric, regression_metrics
from voila3d.utils import ensure_dir, now_iso


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--oof-results-dir", action="append", required=True)
    p.add_argument("--out-dir", default="results/if_router_gated_selection_v1_5seed")
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--budgets", nargs="+", type=float, default=[0, 5, 10, 20, 40, 60, 80, 100])
    p.add_argument("--main-budget", type=float, default=20.0)
    p.add_argument("--router-estimators", type=int, default=64)
    p.add_argument("--router-n-jobs", type=int, default=6)
    p.add_argument("--pair-samples", type=int, default=5000)
    p.add_argument("--pair-margin-frac", type=float, default=0.25)
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=2718)
    p.add_argument("--allow-safe-abstain", action="store_true")
    p.add_argument("--min-oof-big-to-call", type=float, default=float("-inf"))
    p.add_argument("--safe-rule", choices=["oof_big", "utility_lcb"], default="oof_big")
    p.add_argument("--safe-bootstrap-iters", type=int, default=500)
    p.add_argument("--safe-min-selected", type=int, default=8)
    p.add_argument("--safe-utility-lcb-threshold", type=float, default=0.0)
    return p.parse_args()


def _select_top(scores: np.ndarray, budget: float) -> np.ndarray:
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


def _threshold_from_quantile(scores: np.ndarray, q: float | None) -> float:
    if q is None:
        return -np.inf
    val = float(np.nanquantile(scores, q))
    return val if np.isfinite(val) else np.inf


def _standardize_from_train(train_scores: np.ndarray, test_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_scores = np.asarray(train_scores, dtype=float)
    test_scores = np.asarray(test_scores, dtype=float)
    mu = float(np.nanmean(train_scores))
    sd = float(np.nanstd(train_scores))
    if not np.isfinite(sd) or sd < 1e-12:
        sd = 1.0
    return (train_scores - mu) / sd, (test_scores - mu) / sd


def _evaluate_gated_curves(
    task: str,
    task_type: str,
    y: np.ndarray,
    pred2d: np.ndarray,
    pred3d: np.ndarray,
    router_scores: dict[str, tuple[np.ndarray, float | None, float | None]],
    budgets: list[float],
    seed: int,
) -> pd.DataFrame:
    rows = []
    metric = primary_metric(task_type)
    for name, (scores, q, fixed_threshold) in router_scores.items():
        threshold = -np.inf if fixed_threshold is None else float(fixed_threshold)
        for budget in budgets:
            selected = _select_top(scores, budget) & (scores >= threshold)
            routed = pred2d.copy()
            routed[selected] = pred3d[selected]
            mets = regression_metrics(y, routed) if task_type == "regression" else classification_metrics(y, np.clip(routed, 1e-6, 1 - 1e-6))
            row = {
                "task": task,
                "task_type": task_type,
                "seed": seed,
                "router": name,
                "budget": float(budget),
                "gate_quantile": "" if q is None else float(q),
                "gate_threshold": float(threshold) if np.isfinite(threshold) else "",
                "call_rate": float(selected.mean() * 100.0),
                "primary_metric": metric,
                "primary_value": float(mets[metric]),
            }
            row.update({f"metric_{k}": v for k, v in mets.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def _candidate_oof_scores(
    objective: str,
    feature_group: str,
    groups: dict[str, np.ndarray],
    utility: np.ndarray,
    fold_id: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    margin: float,
) -> np.ndarray | None:
    scores = np.full(len(utility), np.nan, dtype=float)
    folds = [f for f in sorted(set(int(x) for x in fold_id)) if f >= 0]
    if len(folds) < 2:
        return None
    for fold in folds:
        hold = fold_id.astype(int) == fold
        fit = ~hold
        if int(hold.sum()) == 0 or int(fit.sum()) < 16:
            continue
        try:
            score, _ = _train_and_score(
                objective,
                groups[feature_group][fit],
                utility[fit],
                groups[feature_group][hold],
                seed + 1009 * (fold + 1),
                args,
                margin,
            )
        except Exception:
            continue
        scores[hold] = score
    return scores if np.isfinite(scores).sum() >= max(16, len(utility) // 2) else None


def _oof_big_for_scores(
    task: str,
    task_type: str,
    y: np.ndarray,
    pred2d: np.ndarray,
    pred3d: np.ndarray,
    scores: np.ndarray,
    q: float | None,
    budgets: list[float],
    seed: int,
) -> float:
    curves = _evaluate_gated_curves(
        task,
        task_type,
        y,
        pred2d,
        pred3d,
        {"candidate": (scores, q, _threshold_from_quantile(scores, q))},
        budgets,
        seed,
    )
    return float(_budget_integrated_gain(curves, task_type)["BIG"].iloc[0])


def _selected_utility_lcb(
    utility: np.ndarray,
    scores: np.ndarray,
    q: float | None,
    budget: float,
    seed: int,
    iters: int,
    min_selected: int,
) -> tuple[int, float, float]:
    threshold = _threshold_from_quantile(scores, q)
    selected = _select_top(scores, budget) & (scores >= threshold)
    vals = np.asarray(utility, dtype=float)[selected]
    if len(vals) == 0:
        return 0, float("nan"), float("-inf")
    mean = float(np.nanmean(vals))
    if len(vals) < min_selected:
        return int(len(vals)), mean, float("-inf")
    rng = np.random.default_rng(seed)
    boots = [float(np.nanmean(vals[rng.integers(0, len(vals), size=len(vals))])) for _ in range(iters)]
    return int(len(vals)), mean, float(np.nanpercentile(boots, 2.5))


def _bootstrap_big_delta(big: pd.DataFrame, final_router: str, baselines: list[str], iters: int, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    pivot = big.pivot_table(index=["task", "seed"], columns="router", values="BIG", aggfunc="mean")
    if final_router not in pivot:
        return pd.DataFrame()
    for base in baselines:
        if base not in pivot:
            continue
        paired = pivot[[final_router, base]].dropna()
        delta = (paired[final_router] - paired[base]).to_numpy(float)
        if len(delta) == 0:
            continue
        boots = [float(delta[rng.integers(0, len(delta), size=len(delta))].mean()) for _ in range(iters)]
        rows.append(
            {
                "final_router": final_router,
                "baseline": base,
                "mean_delta_BIG": float(delta.mean()),
                "ci95_low": float(np.percentile(boots, 2.5)),
                "ci95_high": float(np.percentile(boots, 97.5)),
                "n_task_seed_pairs": int(len(delta)),
                "n_positive_pairs": int((delta > 0).sum()),
                "n_bootstrap": int(iters),
            }
        )
    return pd.DataFrame(rows)


def run_task(results_dir: Path, task: str, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta_json = _read_json(results_dir / "run_metadata.json")
    labels = _load_labels(results_dir, task)
    preds = pd.read_csv(results_dir / "predictions" / f"{task}_predictions.csv")
    meta, x_ecfp, x_desc, _x3d, manifest = _load_cache(results_dir, task, meta_json)
    idx_by_mol = {str(m): i for i, m in enumerate(meta["mol_id"].astype(str))}
    task_type = TASK_TYPE[task]
    seeds = sorted(preds["seed"].astype(int).unique())
    if args.seeds:
        seeds = [s for s in seeds if s in set(args.seeds)]
    objectives = ["R0_raw_utility", "R1_binary", "R2_margin_binary", "R3_three_class", "R4_pairwise_margin_weighted", "R5_rank_regression"]
    feature_groups = ["chemistry", "chemistry_uncertainty", "full_mechanism"]
    gate_grid: list[float | None] = [None, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    route_frames = []
    diag_rows = []
    selection_rows = []
    score_frames = []
    support_cache: dict[bytes, np.ndarray] = {}

    def support_for(idx: np.ndarray) -> np.ndarray:
        key = np.asarray(idx, dtype=np.int64).tobytes()
        if key not in support_cache:
            support_cache[key] = _support_features(meta, x_ecfp, idx)
        return support_cache[key]

    for seed in seeds:
        print(f"[gated] {task} seed={seed}", flush=True)
        glabel = labels[labels["seed"].astype(int) == seed].copy()
        gpred = preds[preds["seed"].astype(int) == seed].copy()
        label_idx = np.asarray([idx_by_mol[str(m)] for m in glabel["mol_id"].astype(str)], dtype=int)
        test_idx = np.asarray([idx_by_mol[str(m)] for m in gpred["mol_id"].astype(str)], dtype=int)
        y_label = meta.iloc[label_idx]["y"].to_numpy(float)
        y_test = gpred["y"].to_numpy(float)
        pred2d_label = glabel["pred2d_label"].to_numpy(float)
        pred3d_label = glabel["pred3d_label"].to_numpy(float)
        pred2d_test = gpred["pred2d"].to_numpy(float)
        pred3d_test = gpred["pred3d_aug"].to_numpy(float)
        u_label = _utility_from_predictions(task_type, y_label, pred2d_label, pred3d_label)
        u_test = _utility_from_predictions(task_type, y_test, pred2d_test, pred3d_test)
        margin = max(1e-12, args.pair_margin_frac * float(np.nanstd(u_label)))
        label_groups = _feature_groups(meta, x_ecfp, x_desc, manifest, label_idx, pred2d_label, task_type, support_for(label_idx))
        test_groups = _feature_groups(meta, x_ecfp, x_desc, manifest, test_idx, pred2d_test, task_type, support_for(test_idx))
        fold_id = glabel["fold_id"].to_numpy(int) if "fold_id" in glabel else np.arange(len(glabel)) % 5

        best = None
        best_cv_scores = None
        for objective in objectives:
            for feature_group in feature_groups:
                cv_scores = _candidate_oof_scores(objective, feature_group, label_groups, u_label, fold_id, seed, args, margin)
                if cv_scores is None:
                    continue
                ok = np.isfinite(cv_scores)
                if ok.mean() < 0.8:
                    continue
                fill = np.nanmedian(cv_scores[ok])
                cv_scores = np.where(np.isfinite(cv_scores), cv_scores, fill)
                for q in gate_grid:
                    big = _oof_big_for_scores(task, task_type, y_label, pred2d_label, pred3d_label, cv_scores, q, args.budgets, seed)
                    ndcg20 = _ndcg_at_budget(u_label, cv_scores, args.main_budget)
                    item = {
                        "objective": objective,
                        "feature_group": feature_group,
                        "gate_quantile": q,
                        "oof_BIG": big,
                        "oof_ndcg20": ndcg20,
                    }
                    if best is None or (big, ndcg20) > (best["oof_BIG"], best["oof_ndcg20"]):
                        best = item
                        best_cv_scores = cv_scores.copy()
        if best is None:
            best = {"objective": "R2_margin_binary", "feature_group": "full_mechanism", "gate_quantile": None, "oof_BIG": np.nan, "oof_ndcg20": np.nan}
        if best_cv_scores is None:
            best_cv_scores = np.zeros(len(u_label), dtype=float)
        q = best["gate_quantile"]
        q_float = None if pd.isna(q) else float(q)
        oof_selected_n, oof_selected_mean_utility, oof_selected_utility_lcb = _selected_utility_lcb(
            u_label,
            best_cv_scores,
            q_float,
            args.main_budget,
            seed + 7919,
            args.safe_bootstrap_iters,
            args.safe_min_selected,
        )
        unsafe_oof_big = not np.isfinite(float(best["oof_BIG"])) or float(best["oof_BIG"]) <= float(args.min_oof_big_to_call)
        unsafe_utility_lcb = (
            args.safe_rule == "utility_lcb"
            and (
                oof_selected_n < int(args.safe_min_selected)
                or (not np.isfinite(oof_selected_utility_lcb))
                or oof_selected_utility_lcb <= float(args.safe_utility_lcb_threshold)
            )
        )
        safe_abstain = bool(args.allow_safe_abstain and (unsafe_oof_big or unsafe_utility_lcb))
        selection_rows.append(
            {
                "task": task,
                "task_type": task_type,
                "seed": seed,
                **best,
                "safe_decision": "abstain" if safe_abstain else "call",
                "min_oof_big_to_call": float(args.min_oof_big_to_call),
                "safe_rule": str(args.safe_rule),
                "oof_selected_n": int(oof_selected_n),
                "oof_selected_mean_utility": float(oof_selected_mean_utility),
                "oof_selected_utility_lcb": float(oof_selected_utility_lcb),
            }
        )

        raw_train_scores, _ = _train_and_score(
            str(best["objective"]),
            label_groups[str(best["feature_group"])],
            u_label,
            label_groups[str(best["feature_group"])],
            seed,
            args,
            margin,
        )
        raw_test_scores, _ = _train_and_score(
            str(best["objective"]),
            label_groups[str(best["feature_group"])],
            u_label,
            test_groups[str(best["feature_group"])],
            seed,
            args,
            margin,
        )
        train_scores, test_scores = _standardize_from_train(raw_train_scores, raw_test_scores)
        train_threshold = _threshold_from_quantile(train_scores, q_float)
        baseline_scores: dict[str, tuple[np.ndarray, float | None, float | None]] = {
            "random": (np.random.default_rng(seed).normal(size=len(gpred)), None, None),
            "uncertainty": (gpred["uncertainty_score"].to_numpy(float), None, None),
            "flexibility": (gpred["flexibility_score"].to_numpy(float), None, None),
            "old_voi": (gpred["voi_score"].to_numpy(float), None, None),
            "oracle": (u_test, None, None),
            "R7_oof_selected_top": (test_scores, None, None),
            "R8_oof_selected_gated": (test_scores, q_float, train_threshold),
            "R9_oof_safe_guardrail": (test_scores, None if safe_abstain else q_float, np.inf if safe_abstain else train_threshold),
        }
        curves = _evaluate_gated_curves(task, task_type, y_test, pred2d_test, pred3d_test, baseline_scores, args.budgets, seed)
        route_frames.append(curves)

        for router, (scores, q_eval, _threshold) in baseline_scores.items():
            beneficial = u_test > 0
            diag_rows.append(
                {
                    "task": task,
                    "task_type": task_type,
                    "seed": seed,
                    "router": router,
                    "main_budget": float(args.main_budget),
                    "beneficial_rate": float(beneficial.mean()),
                    "beneficial_auc": _auc_safe(beneficial, scores),
                    "beneficial_ap": _ap_safe(beneficial, scores),
                    "precision_at_20": _precision_at_budget(u_test, scores, args.main_budget),
                    "ndcg_at_20": _ndcg_at_budget(u_test, scores, args.main_budget),
                    "utility_spearman": _spearman_safe(u_test, scores),
                    "gate_quantile": "" if q_eval is None else float(q_eval),
                }
            )
        score_frames.append(
            pd.DataFrame(
                {
                    "task": task,
                    "seed": seed,
                    "mol_id": gpred["mol_id"].to_numpy(),
                    "clean_utility": u_test,
                    "selected_objective": str(best["objective"]),
                    "selected_feature_group": str(best["feature_group"]),
                    "selected_gate_quantile": "" if q_float is None else float(q_float),
                    "safe_decision": "abstain" if safe_abstain else "call",
                    "min_oof_big_to_call": float(args.min_oof_big_to_call),
                    "safe_rule": str(args.safe_rule),
                    "oof_selected_n": int(oof_selected_n),
                    "oof_selected_mean_utility": float(oof_selected_mean_utility),
                    "oof_selected_utility_lcb": float(oof_selected_utility_lcb),
                    "score_R8_oof_selected": test_scores,
                }
            )
        )
    return (
        pd.concat(route_frames, ignore_index=True) if route_frames else pd.DataFrame(),
        pd.DataFrame(diag_rows),
        pd.DataFrame(selection_rows),
        pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame(),
    )


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    ensure_dir(out_dir / "predictions")
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "started_at": now_iso(),
                "argv": sys.argv,
                "protocol_note": "OOF-internal selection of router objective, feature group, and acquisition gate quantile. Budget is treated as an upper bound; gate thresholds are selected without test labels.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    all_curves = []
    all_diag = []
    all_sel = []
    all_big = []
    for path in args.oof_results_dir:
        results_dir = Path(path)
        meta = _read_json(results_dir / "run_metadata.json")
        tasks = list(meta.get("tasks", []))
        if args.tasks:
            tasks = [t for t in tasks if t in set(args.tasks)]
        for task in tasks:
            print(f"[gated] source={results_dir.name} task={task}", flush=True)
            curves, diag, selections, scores = run_task(results_dir, task, args)
            if not curves.empty:
                curves["source_dir"] = str(results_dir)
                curves.to_csv(out_dir / f"routing_curves_{task}.csv", index=False)
                all_curves.append(curves)
                big = _budget_integrated_gain(curves, TASK_TYPE[task])
                big["source_dir"] = str(results_dir)
                big.to_csv(out_dir / f"budget_integrated_gain_{task}.csv", index=False)
                all_big.append(big)
            if not diag.empty:
                diag["source_dir"] = str(results_dir)
                all_diag.append(diag)
            if not selections.empty:
                selections["source_dir"] = str(results_dir)
                all_sel.append(selections)
            if not scores.empty:
                scores.to_csv(out_dir / "predictions" / f"{task}_gated_scores.csv", index=False)
    curves = pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()
    diag = pd.concat(all_diag, ignore_index=True) if all_diag else pd.DataFrame()
    sel = pd.concat(all_sel, ignore_index=True) if all_sel else pd.DataFrame()
    big = pd.concat(all_big, ignore_index=True) if all_big else pd.DataFrame()
    if not curves.empty:
        curves.to_csv(out_dir / "routing_curves.csv", index=False)
        _summarize(curves, ["task", "task_type", "router", "budget"], ["primary_value", "call_rate"]).to_csv(out_dir / "routing_summary.csv", index=False)
    if not diag.empty:
        diag.to_csv(out_dir / "router_diagnostics_long.csv", index=False)
        _summarize(diag, ["task", "task_type", "router"], ["beneficial_auc", "beneficial_ap", "precision_at_20", "ndcg_at_20", "utility_spearman"]).to_csv(
            out_dir / "router_diagnostics_summary.csv", index=False
        )
    if not sel.empty:
        sel.to_csv(out_dir / "oof_selection_long.csv", index=False)
        sel.groupby(["task", "objective", "feature_group", "gate_quantile"], dropna=False).size().reset_index(name="n_seed_selected").to_csv(
            out_dir / "oof_selection_summary.csv", index=False
        )
    if not big.empty:
        big.to_csv(out_dir / "budget_integrated_gain_long.csv", index=False)
        _summarize(big, ["task", "task_type", "router"], ["BIG"]).to_csv(out_dir / "budget_integrated_gain_summary.csv", index=False)
        ci_frames = []
        for final_router in ["R8_oof_selected_gated", "R9_oof_safe_guardrail"]:
            ci = _bootstrap_big_delta(big, final_router, ["random", "uncertainty", "flexibility", "old_voi", "R7_oof_selected_top", "R8_oof_selected_gated"], args.bootstrap_iters, args.bootstrap_seed)
            if not ci.empty:
                ci_frames.append(ci)
        (pd.concat(ci_frames, ignore_index=True) if ci_frames else pd.DataFrame()).to_csv(out_dir / "big_paired_bootstrap_ci.csv", index=False)
    print(f"[gated] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
