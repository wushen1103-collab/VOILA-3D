"""Recompute current permission decisions from the compact paired prediction archive."""
import argparse
from io import BytesIO
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voila3d.permission import bootstrap_lcb, endpoint_gain, paired_test, risk_benefit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference-dir', type=Path, default=Path('results/reference/taskmetric_20261108'))
    p.add_argument('--out-dir', type=Path, default=Path('results/taskmetric_reproduced'))
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(args.reference_dir / 'expert_states.csv.gz')
    molecules = pd.read_csv(args.reference_dir / 'molecule_manifest.csv.gz').set_index(['task', 'row'])
    membership = pd.read_csv(args.reference_dir / 'partition_membership.csv.gz')
    paired_reference = pd.read_csv(args.reference_dir / 'paired_test_rows.csv').set_index(['task', 'split_seed', 'seed'])
    records = []
    with zipfile.ZipFile(args.reference_dir / 'paired_predictions.zip') as archive:
        selected = reference if args.limit is None else reference.head(args.limit)
        for row in selected.itertuples(index=False):
            filename = f'{row.task}_split{row.split_seed}_seed{row.seed}.npz'
            with np.load(BytesIO(archive.read(filename))) as predictions:
                yv, rv, av, yt, rt, at = [predictions[k] for k in
                                         ['y_val', 'ref_val', 'alt_val', 'y_test', 'ref_test', 'alt_test']]
                vi, ti = predictions['val_indices'], predictions['test_indices']
            truth = molecules.loc[row.task].target
            assert np.allclose(yv, truth.loc[vi], rtol=1e-10, atol=1e-12)
            assert np.allclose(yt, truth.loc[ti], rtol=1e-10, atol=1e-12)
            roles = membership[(membership.task == row.task) & (membership.split_seed == row.split_seed)]
            assert np.array_equal(np.sort(vi), np.sort(roles[roles.role == 'val'].row))
            assert np.array_equal(np.sort(ti), np.sort(roles[roles.role == 'test'].row))
            gain = endpoint_gain(row.task_type, yt, rt, at)
            lower = bootstrap_lcb(row.task_type, yv, rv, av, row.seed + 44000)
            assert np.isclose(gain, row.full_alternative_test_gain, rtol=1e-8, atol=1e-10)
            assert np.isclose(lower, row.validation_lcb, rtol=1e-8, atol=1e-10)
            statistical = paired_test(row.task_type, yv, rv, av)
            expected_test = paired_reference.loc[(row.task, row.split_seed, row.seed)]
            assert statistical['permission'] == bool(expected_test.permission)
            assert np.isclose(statistical['pvalue'], expected_test.pvalue, rtol=1e-8, atol=1e-12)
            val_gain = endpoint_gain(row.task_type, yv, rv, av)
            for policy, permit in [('Always-2D', False), ('Always-2D+3D', True),
                                   ('TaskSelect', val_gain > 0),
                                   ('ThresholdSelect-0.5pct', val_gain / abs(row.validation_ref_value) > .005),
                                   ('ThresholdSelect-1pct', val_gain / abs(row.validation_ref_value) > .01),
                                   ('TaskMetric-LCB', bool(np.isfinite(lower) and lower > 0)),
                                   ('PairedTest-0.025', statistical['permission'])]:
                records.append(dict(task=row.task, split_seed=row.split_seed, seed=row.seed,
                                    policy=policy, permission=permit, gain=gain,
                                    policy_gain=gain if permit else 0, validation_lcb=lower))
    result = pd.DataFrame(records)
    result.to_csv(args.out_dir / 'policy_states.csv', index=False)
    summaries = []
    for (task, policy), group in result.groupby(['task', 'policy']):
        summaries.append(dict(task=task, policy=policy,
                              **risk_benefit(group.gain.to_numpy(), group.permission.to_numpy())))
    pd.DataFrame(summaries).to_csv(args.out_dir / 'endpoint_risk_benefit.csv', index=False)
    print(f'Reproduced {len(result)} policy states; all paired gains and bootstrap bounds match.')


if __name__ == '__main__':
    main()
