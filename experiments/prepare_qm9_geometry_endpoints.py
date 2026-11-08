from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from rdkit import Chem


BASE_ROOT = Path("results/qm9_conformation_stress_k10_full_5seed/feature_cache")
BASE_PREFIX = "QM9_R2_random_max2000_k10_rdkit_full"
RAW_PATH = Path("data/raw/QM9.csv")
OUT_ROOT = Path("results/rejection_rebuild_qm9_geometry_endpoints/feature_cache")
TARGETS = {
    "QM9_A": "A",
    "QM9_B": "B",
    "QM9_C": "C",
    "QM9_ALPHA": "alpha",
}


def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    base_meta = pd.read_csv(BASE_ROOT / f"{BASE_PREFIX}_meta.csv")
    raw = pd.read_csv(RAW_PATH, usecols=["smiles", *TARGETS.values()])
    raw["canonical_smiles"] = [canonicalize(value) for value in raw["smiles"]]

    # The two duplicated canonical structures in the 2,000-molecule sample have
    # numerically equivalent QM9 labels. Mean aggregation keeps the mapping
    # deterministic and avoids selecting an arbitrary duplicate row.
    target_map = raw.groupby("canonical_smiles", as_index=True)[list(TARGETS.values())].mean()
    missing = sorted(set(base_meta["canonical_smiles"]) - set(target_map.index))
    if missing:
        raise ValueError(f"Missing {len(missing)} cached molecules in raw QM9")

    for task, column in TARGETS.items():
        prefix = f"{task}_random_max2000_k10_rdkit_full"
        meta = base_meta.copy()
        meta["mol_id"] = [f"{task}_{i:06d}" for i in range(len(meta))]
        meta["y"] = meta["canonical_smiles"].map(target_map[column]).astype(float)
        meta.to_csv(OUT_ROOT / f"{prefix}_meta.csv", index=False)

        for suffix in ("ecfp.npy", "desc.npy", "x3d.npy", "manifest.csv"):
            source = (BASE_ROOT / f"{BASE_PREFIX}_{suffix}").resolve()
            target = OUT_ROOT / f"{prefix}_{suffix}"
            if target.exists() or target.is_symlink():
                target.unlink()
            os.symlink(source, target)
        print(f"prepared {task}: n={len(meta)} target={column}")


if __name__ == "__main__":
    main()
