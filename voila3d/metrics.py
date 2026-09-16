from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, pred)
    return {
        "MAE": float(mean_absolute_error(y_true, pred)),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, pred)),
    }


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    labels_present = set(int(x) for x in np.unique(y_true))
    out = {
        "Accuracy": float(accuracy_score(y_true, (proba >= 0.5).astype(int))),
        "LogLoss": float(log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1])),
    }
    if labels_present == {0, 1}:
        out["AUROC"] = float(roc_auc_score(y_true, proba))
        out["AUPRC"] = float(average_precision_score(y_true, proba))
    else:
        out["AUROC"] = float("nan")
        out["AUPRC"] = float("nan")
    return out


def per_sample_loss(task_type: str, y_true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if task_type == "regression":
        return np.abs(y_true - pred)
    p = np.clip(pred, 1e-6, 1 - 1e-6)
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def primary_metric(task_type: str) -> str:
    return "MAE" if task_type == "regression" else "AUROC"


def higher_is_better(metric: str) -> bool:
    return metric.upper() in {"AUROC", "AUPRC", "R2", "ACCURACY"}
