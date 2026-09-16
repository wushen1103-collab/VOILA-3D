from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import classification_metrics, higher_is_better, per_sample_loss, primary_metric, regression_metrics


def uncertainty_scores(task_type: str, pred2d: np.ndarray) -> np.ndarray:
    if task_type == "classification":
        return 1.0 - np.maximum(pred2d, 1.0 - pred2d)
    z = np.abs((pred2d - np.nanmean(pred2d)) / (np.nanstd(pred2d) + 1e-8))
    return z


def make_router_features(x_desc: np.ndarray, pred2d: np.ndarray, task_type: str) -> np.ndarray:
    unc = uncertainty_scores(task_type, pred2d)[:, None]
    pred = pred2d[:, None]
    return np.hstack([x_desc, pred, unc]).astype(np.float32)


def train_voi_router(x_router: np.ndarray, benefit: np.ndarray, seed: int = 0):
    model = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=4,
        random_state=seed,
        n_jobs=8,
    )
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)
    pipe.fit(x_router, benefit)
    return pipe


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


def route_predictions(pred2d: np.ndarray, pred3d: np.ndarray, scores: np.ndarray, budget: float) -> tuple[np.ndarray, np.ndarray]:
    selected = _select_top(scores, budget)
    routed = pred2d.copy()
    routed[selected] = pred3d[selected]
    return routed, selected


def evaluate_routing_curves(
    task: str,
    task_type: str,
    y_true: np.ndarray,
    pred2d: np.ndarray,
    pred3d: np.ndarray,
    router_scores: dict[str, np.ndarray],
    budgets: list[float],
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    metric = primary_metric(task_type)
    for router_name, scores in router_scores.items():
        for budget in budgets:
            routed, selected = route_predictions(pred2d, pred3d, scores, budget)
            mets = regression_metrics(y_true, routed) if task_type == "regression" else classification_metrics(y_true, routed)
            row = {
                "task": task,
                "task_type": task_type,
                "seed": seed,
                "router": router_name,
                "budget": float(budget),
                "call_rate": float(selected.mean() * 100.0),
                "primary_metric": metric,
                "primary_value": float(mets.get(metric, np.nan)),
            }
            row.update({f"metric_{k}": v for k, v in mets.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_budget_gain(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, router, budget), g in curves.groupby(["task", "router", "budget"]):
        metric = g["primary_metric"].iloc[0]
        vals = g["primary_value"].to_numpy(float)
        rows.append({
            "task": task,
            "router": router,
            "budget": budget,
            "primary_metric": metric,
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
            "higher_is_better": higher_is_better(metric),
            "n": int(np.isfinite(vals).sum()),
        })
    return pd.DataFrame(rows)


def _binary_auc_contribution(y_true: np.ndarray, pred2d: np.ndarray, pred3d: np.ndarray) -> np.ndarray:
    y = y_true.astype(int)
    pos = y == 1
    neg = y == 0
    if int(pos.sum()) == 0 or int(neg.sum()) == 0:
        return np.zeros(len(y), dtype=float)

    pos_scores = np.sort(pred2d[pos].astype(float))
    neg_scores = np.sort(pred2d[neg].astype(float))

    def _pos_auc_terms(scores: np.ndarray) -> np.ndarray:
        left = np.searchsorted(neg_scores, scores, side="left")
        right = np.searchsorted(neg_scores, scores, side="right")
        return (left + 0.5 * (right - left)) / float(len(neg_scores))

    def _neg_auc_terms(scores: np.ndarray) -> np.ndarray:
        left = np.searchsorted(pos_scores, scores, side="left")
        right = np.searchsorted(pos_scores, scores, side="right")
        greater = len(pos_scores) - right
        return (greater + 0.5 * (right - left)) / float(len(pos_scores))

    benefit = np.zeros(len(y), dtype=float)
    benefit[pos] = (_pos_auc_terms(pred3d[pos]) - _pos_auc_terms(pred2d[pos])) / float(pos.sum())
    benefit[neg] = (_neg_auc_terms(pred3d[neg]) - _neg_auc_terms(pred2d[neg])) / float(neg.sum())
    return benefit


def true_benefit(
    task_type: str,
    y_true: np.ndarray,
    pred2d: np.ndarray,
    pred3d: np.ndarray,
    cost: np.ndarray | None = None,
    lam: float = 0.0,
    classification_mode: str = "logloss",
) -> np.ndarray:
    if task_type == "classification":
        if classification_mode == "margin":
            y_signed = 2.0 * y_true.astype(float) - 1.0
            benefit = y_signed * (pred3d - pred2d)
        elif classification_mode == "auc_contrib":
            benefit = _binary_auc_contribution(y_true, pred2d, pred3d)
        else:
            loss2d = per_sample_loss(task_type, y_true, pred2d)
            loss3d = per_sample_loss(task_type, y_true, pred3d)
            benefit = loss2d - loss3d
    else:
        loss2d = per_sample_loss(task_type, y_true, pred2d)
        loss3d = per_sample_loss(task_type, y_true, pred3d)
        benefit = loss2d - loss3d
    if cost is not None:
        cost = (cost - np.nanmean(cost)) / (np.nanstd(cost) + 1e-8)
        benefit = benefit - lam * cost
    return benefit
