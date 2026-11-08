"""Equal-endpoint risk-benefit summaries and paired partition-cluster intervals."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voila3d.permission import risk_benefit


def analyze(reference_dir, out, iterations=10000):
    out.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(reference_dir / 'all_selector_states.csv.gz')
    additional = pd.read_csv(reference_dir / 'paired_test_rows.csv')
    rows = pd.concat([original[original.condition == 'same_seed_true3d'], additional], ignore_index=True)
    assert len(rows) == 3500
    assert not rows.duplicated(['task', 'split_seed', 'seed', 'policy']).any()
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs/taskmetric_protocol.json').read_text())
    summaries, intervals, contrasts, endpoint_rows = [], [], [], []
    for group_name, tasks in [('Main six', config['main_endpoints']), ('Geometry four', config['geometric_endpoints'])]:
        frame = rows[rows.task.isin(tasks)]
        tasks = sorted(tasks)
        policies = sorted(frame.policy.unique())
        rng = np.random.default_rng(20261108)
        sampled = rng.integers(0, 10, size=(iterations, len(tasks), 10))
        boot = {}
        points = {}
        for policy in policies:
            values, sampled_values = [], []
            for t, task in enumerate(tasks):
                current = frame[(frame.task == task) & (frame.policy == policy)].sort_values(['split_seed', 'seed'])
                assert len(current) == 50
                g = current.full_alternative_test_gain.to_numpy().reshape(10, 5)
                z = current.permission.to_numpy(dtype=bool).reshape(10, 5)
                point = risk_benefit(g.ravel(), z.ravel())
                endpoint_rows.append(dict(group=group_name, task=task, policy=policy, **point))
                values.append(point)
                draws, permissions = g[sampled[:, t, :]], z[sampled[:, t, :]]
                positive, negative = np.maximum(draws, 0), np.maximum(-draws, 0)
                positive_total, negative_total = positive.sum(axis=(1, 2)), negative.sum(axis=(1, 2))
                retained = np.divide((positive * permissions).sum(axis=(1, 2)), positive_total,
                                     out=np.full(iterations, np.nan), where=positive_total > 0)
                incurred = np.divide((negative * permissions).sum(axis=(1, 2)), negative_total,
                                     out=np.full(iterations, np.nan), where=negative_total > 0)
                sampled_values.append(dict(activation=permissions.mean(axis=(1, 2)),
                                           false_activation=(permissions & (draws <= 0)).mean(axis=(1, 2)),
                                           false_rejection=(~permissions & (draws > 0)).mean(axis=(1, 2)),
                                           gain_retention=retained, harm_avoidance=1 - incurred))
            points[policy] = pd.DataFrame(values).mean().to_dict()
            summaries.append(dict(group=group_name, policy=policy, **points[policy]))
            boot[policy] = {key: np.nanmean(np.array([v[key] for v in sampled_values]), axis=0)
                            for key in points[policy]}
            for key, observed in points[policy].items():
                low, high = np.quantile(boot[policy][key], [.025, .975])
                intervals.append(dict(group=group_name, policy=policy, metric=key,
                                      estimate=observed, ci_low=low, ci_high=high))
        for comparator in ['TaskSelect', 'PairedTest-0.025']:
            for key in points['TaskMetric-LCB']:
                low, high = np.quantile(boot['TaskMetric-LCB'][key] - boot[comparator][key], [.025, .975])
                contrasts.append(dict(group=group_name, method='TaskMetric-LCB', comparator=comparator,
                                      metric=key, difference=points['TaskMetric-LCB'][key] - points[comparator][key],
                                      ci_low=low, ci_high=high))
    pd.DataFrame(summaries).to_csv(out / 'macro_risk_benefit.csv', index=False)
    pd.DataFrame(endpoint_rows).to_csv(out / 'endpoint_risk_benefit.csv', index=False)
    pd.DataFrame(intervals).to_csv(out / 'cluster_intervals.csv', index=False)
    pd.DataFrame(contrasts).to_csv(out / 'paired_cluster_contrasts.csv', index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    return pd.DataFrame(summaries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-dir', type=Path, default=Path('results/reference/taskmetric_20261108'))
    parser.add_argument('--out-dir', type=Path, default=Path('results/taskmetric_reference_analysis'))
    parser.add_argument('--iterations', type=int, default=10000)
    args = parser.parse_args()
    analyze(args.reference_dir, args.out_dir, args.iterations)
