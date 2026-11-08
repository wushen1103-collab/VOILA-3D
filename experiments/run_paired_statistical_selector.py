"""Matched held-out paired t / DeLong selection with frozen alpha=0.025."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.fast_screen import _fit_xgb, _predict
from experiments.run_rejection_rebuild_gate1 import _load_task, _metric, _metric_gain
from voila3d.data import assign_splits


def paired_test(task_type, y, ref, alt):
    if task_type == 'classification':
        positive, negative = y == 1, y == 0
        if min(positive.sum(), negative.sum()) < 2:
            return np.nan, np.nan, 1.0
        placements = []
        for pred in [ref, alt]:
            dif = pred[positive, None] - pred[None, negative]
            h = (dif > 0).astype(float) + 0.5 * (dif == 0)
            placements.append((h.mean(axis=1), h.mean(axis=0)))
        pos_diff = placements[1][0] - placements[0][0]
        neg_diff = placements[1][1] - placements[0][1]
        gain = float(pos_diff.mean())
        variance = np.var(pos_diff, ddof=1) / positive.sum() + np.var(neg_diff, ddof=1) / negative.sum()
        se = float(np.sqrt(max(variance, 0.0)))
        critical = stats.norm.ppf(0.975)
        distribution = stats.norm
    else:
        dif = np.abs(y - ref) - np.abs(y - alt)
        gain = float(dif.mean())
        if len(dif) < 2:
            return gain, np.nan, 1.0
        se = float(dif.std(ddof=1) / np.sqrt(len(dif)))
        critical = stats.t.ppf(0.975, len(dif) - 1)
        distribution = stats.t(df=len(dif) - 1)
    if se == 0:
        return gain, gain, 0.0 if gain > 0 else 1.0
    p = float(distribution.sf(gain / se))
    return gain, float(gain - critical * se), p


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tasks', nargs='+', required=True)
    p.add_argument('--splits', nargs='+', type=int, default=list(range(1, 11)))
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(5)))
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--reference', required=True)
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    expected = pd.read_csv(args.reference).set_index(['task', 'split_seed', 'seed'])
    rows = []
    for task in args.tasks:
        data = _load_task(task, -1)
        task_type, y = data['task_type'], data['y']
        ref_features = data['x2d']
        alt_features = np.hstack([data['x2d'], data['x3d']]).astype(np.float32)
        for split_seed in args.splits:
            split = assign_splits(data['meta'], 'scaffold_randomized', seed=split_seed,
                                  frac_train=0.8, frac_val=0.1, task_type=task_type).astype(str)
            train, val, test = [np.flatnonzero(split == role) for role in ['train', 'val', 'test']]
            for seed in args.seeds:
                start = time.monotonic()
                cache = out / f'{task}_split{split_seed}_seed{seed}.npz'
                if cache.exists():
                    with np.load(cache) as z:
                        rv, av, rt, at = [z[k] for k in ['ref_val', 'alt_val', 'ref_test', 'alt_test']]
                else:
                    ref, ref_device = _fit_xgb(task_type, ref_features[train], y[train], seed, args.threads, args.device)
                    alt, alt_device = _fit_xgb(task_type, alt_features[train], y[train], seed, args.threads, args.device)
                    rv, av = _predict(ref, task_type, ref_features[val]), _predict(alt, task_type, alt_features[val])
                    rt, at = _predict(ref, task_type, ref_features[test]), _predict(alt, task_type, alt_features[test])
                    np.savez_compressed(cache, y_val=y[val], y_test=y[test], val_indices=val, test_indices=test,
                                        ref_val=rv, alt_val=av, ref_test=rt, alt_test=at)
                refval, altval = _metric(task_type, y[val], rv)[1], _metric(task_type, y[val], av)[1]
                reftest, alttest = _metric(task_type, y[test], rt)[1], _metric(task_type, y[test], at)[1]
                baseline = expected.loc[(task, split_seed, seed)]
                measured = np.array([refval, altval, reftest, alttest])
                original = baseline[['validation_ref_value', 'validation_alt_value',
                                     'test_ref_value', 'test_alt_value']].to_numpy(float)
                error = float(np.max(np.abs(measured - original)))
                if not np.allclose(measured, original, rtol=1e-8, atol=1e-10):
                    raise ValueError(f'Expert mismatch for {task}/{split_seed}/{seed}: {error}')
                gain, lcb, pvalue = paired_test(task_type, y[val], rv, av)
                endpoint_gain = _metric_gain(task_type, refval, altval)
                assert np.isclose(gain, endpoint_gain, rtol=1e-8, atol=1e-10)
                permit = bool(np.isfinite(lcb) and gain > 0 and pvalue < 0.025)
                test_gain = float(baseline['full_alternative_test_gain'])
                rows.append(dict(task=task, task_type=task_type, condition='same_seed_true3d',
                                 split_seed=split_seed, seed=seed, policy='PairedTest-0.025',
                                 test='paired_DeLong' if task_type == 'classification' else 'paired_t',
                                 alpha=0.025, validation_gain=gain, validation_lower_bound=lcb,
                                 pvalue=pvalue, permission=permit, full_alternative_test_gain=test_gain,
                                 policy_test_gain=test_gain if permit else 0.0,
                                 false_activation=bool(permit and test_gain <= 0),
                                 false_rejection=bool(not permit and test_gain > 0),
                                 expert_metric_max_abs_difference=error,
                                 lcb_permission=bool(baseline['permission']),
                                 negative_transfer=max(0.0, -(test_gain if permit else 0.0)),
                                 regret=max(0.0, test_gain) - (test_gain if permit else 0.0)))
                pd.DataFrame(rows).to_csv(out / 'paired_test_rows.csv', index=False)
                print(json.dumps({'task': task, 'split': split_seed, 'seed': seed,
                                  'permission': permit, 'pvalue': pvalue, 'expert_parity': error,
                                  'seconds': round(time.monotonic() - start, 2)}), flush=True)
        _load_task.cache_clear()
    print('COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
