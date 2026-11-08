"""Rebuild the exact molecule rows using public identities and pinned RDKit."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.run_rejection_rebuild_gate1 import TASK_CACHE
from voila3d.data import assign_splits
from voila3d.features import build_feature_tables


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path,
                   default=Path('results/reference/taskmetric_20261108/molecule_manifest.csv.gz'))
    p.add_argument('--tasks', nargs='+', default=sorted(TASK_CACHE))
    p.add_argument('--jobs', type=int, default=4)
    args = p.parse_args()
    manifest = pd.read_csv(args.manifest)
    previous = {}
    for task in args.tasks:
        cfg = TASK_CACHE[task]
        meta = manifest[manifest.task == task].sort_values('row').copy()
        meta = meta.rename(columns={'target': 'y'})
        assert meta.row.tolist() == list(range(len(meta)))
        meta['mol_id'] = [f'{task}_{i:06d}' for i in range(len(meta))]
        meta['split'] = assign_splits(meta, 'scaffold_randomized', seed=0, frac_train=.8,
                                      frac_val=.1, task_type=cfg['task_type'])
        family = 'usr' if task == 'BBBP' else ('rdkit_full' if task.startswith('QM9_') else 'rdkit_scalar')
        key = (family, tuple(meta.canonical_smiles))
        if key not in previous:
            previous[key] = build_feature_tables(meta, n_bits=2048, n_conformers=10,
                                                 conformer_jobs=args.jobs, seed=0,
                                                 feature3d_set=family)
        ecfp, descriptors, geometry, generation = previous[key]
        root = Path(cfg['root'])
        root.mkdir(parents=True, exist_ok=True)
        prefix = root / cfg['prefix']
        meta.to_csv(str(prefix) + '_meta.csv', index=False)
        for suffix, values in [('ecfp', ecfp), ('desc', descriptors), ('x3d', geometry)]:
            np.save(str(prefix) + '_' + suffix + '.npy', values)
        generation.to_csv(str(prefix) + '_manifest.csv', index=False)
        print(f'Prepared {task}: {len(meta)} molecules, {family}.', flush=True)


if __name__ == '__main__':
    main()
