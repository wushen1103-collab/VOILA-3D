"""Endpoint-aligned permission using held-out paired predictions only."""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def endpoint_gain(task_type, y, reference, augmented):
    if task_type == 'classification':
        return float(roc_auc_score(y, augmented) - roc_auc_score(y, reference))
    return float(np.mean(np.abs(y - reference)) - np.mean(np.abs(y - augmented)))


def bootstrap_lcb(task_type, y, reference, augmented, seed, iterations=5000):
    """Exact resampling order of the manuscript implementation; paired metrics."""
    y, reference, augmented = [np.asarray(a) for a in (y, reference, augmented)]
    rng = np.random.default_rng(seed)
    if task_type == 'classification':
        positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        if not len(positive) or not len(negative):
            return float('nan')
        h = []
        for prediction in (reference, augmented):
            difference = prediction[positive, None] - prediction[None, negative]
            h.append((difference > 0).astype(float) + 0.5 * (difference == 0))
        pos_weights = np.empty((iterations, len(positive)))
        neg_weights = np.empty((iterations, len(negative)))
        for i in range(iterations):
            # Each choice draws the same positions as choosing the original indices.
            pos_weights[i] = np.bincount(rng.choice(len(positive), len(positive), replace=True),
                                        minlength=len(positive))
            neg_weights[i] = np.bincount(rng.choice(len(negative), len(negative), replace=True),
                                        minlength=len(negative))
        values = np.sum((pos_weights @ (h[1] - h[0])) * neg_weights, axis=1)
        values /= len(positive) * len(negative)
    else:
        indices = rng.integers(0, len(y), size=(iterations, len(y)))
        reference_error = np.abs(y - reference)
        augmented_error = np.abs(y - augmented)
        values = reference_error[indices].mean(axis=1) - augmented_error[indices].mean(axis=1)
    return float(np.quantile(values, 0.025))


def paired_test(task_type, y, reference, augmented):
    """One-sided paired t (MAE) or paired DeLong (AUROC), including ties."""
    y, reference, augmented = [np.asarray(a) for a in (y, reference, augmented)]
    gain = endpoint_gain(task_type, y, reference, augmented)
    if task_type == 'classification':
        positive, negative = y == 1, y == 0
        if min(positive.sum(), negative.sum()) < 2:
            return {'gain': gain, 'lower_bound': np.nan, 'pvalue': 1.0, 'permission': False}
        placements = []
        for prediction in (reference, augmented):
            difference = prediction[positive, None] - prediction[None, negative]
            h = (difference > 0).astype(float) + 0.5 * (difference == 0)
            placements.append((h.mean(axis=1), h.mean(axis=0)))
        a = placements[1][0] - placements[0][0]
        b = placements[1][1] - placements[0][1]
        se = np.sqrt(max(0, a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
        distribution = stats.norm
    else:
        difference = np.abs(y - reference) - np.abs(y - augmented)
        if len(difference) < 2:
            return {'gain': gain, 'lower_bound': np.nan, 'pvalue': 1.0, 'permission': False}
        se = difference.std(ddof=1) / np.sqrt(len(difference))
        distribution = stats.t(df=len(difference) - 1)
    pvalue = float(distribution.sf(gain / se)) if se > 0 else (0.0 if gain > 0 else 1.0)
    lower = float(gain - distribution.ppf(0.975) * se)
    return {'gain': gain, 'lower_bound': lower, 'pvalue': pvalue,
            'permission': bool(np.isfinite(lower) and gain > 0 and pvalue < 0.025)}


def risk_benefit(gains, permissions):
    gains, permissions = np.asarray(gains), np.asarray(permissions, dtype=bool)
    positive, negative = np.maximum(gains, 0), np.maximum(-gains, 0)
    return {'activation': float(permissions.mean()),
            'false_activation': float((permissions & (gains <= 0)).mean()),
            'false_rejection': float((~permissions & (gains > 0)).mean()),
            'gain_retention': float(positive[permissions].sum() / positive.sum()) if positive.sum() else np.nan,
            'harm_avoidance': float(1 - negative[permissions].sum() / negative.sum()) if negative.sum() else np.nan}
