from __future__ import annotations

import gzip
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .utils import ensure_dir, sha256_file


TaskType = Literal["regression", "classification"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task_type: TaskType
    metric: str
    urls: tuple[str, ...]
    smiles_col: str | tuple[str, ...]
    target_col: str
    license_note: str = "Public MoleculeNet/DeepChem distribution; upstream terms apply."
    raw_name: str | None = None


DATASETS: dict[str, DatasetSpec] = {
    "ESOL": DatasetSpec(
        "ESOL",
        "regression",
        "MAE",
        ("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",),
        "smiles",
        "measured log solubility in mols per litre",
    ),
    "FreeSolv": DatasetSpec(
        "FreeSolv",
        "regression",
        "MAE",
        ("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",),
        "smiles",
        "expt",
    ),
    "Lipophilicity": DatasetSpec(
        "Lipophilicity",
        "regression",
        "MAE",
        ("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",),
        "smiles",
        "exp",
    ),
    "BBBP": DatasetSpec(
        "BBBP",
        "classification",
        "AUROC",
        ("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",),
        "smiles",
        "p_np",
    ),
    "BACE": DatasetSpec(
        "BACE",
        "classification",
        "AUROC",
        ("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",),
        ("mol", "smiles"),
        "Class",
    ),
    "HIV": DatasetSpec(
        "HIV",
        "classification",
        "AUROC",
        ("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",),
        "smiles",
        "HIV_active",
    ),
    "QM9_GAP": DatasetSpec(
        "QM9_GAP",
        "regression",
        "MAE",
        ("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv",),
        "smiles",
        "gap",
        license_note="QM9/DeepChem public dataset; used as a conformation-sensitive quantum-property stress test.",
        raw_name="QM9",
    ),
    "QM9_MU": DatasetSpec(
        "QM9_MU",
        "regression",
        "MAE",
        ("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv",),
        "smiles",
        "mu",
        license_note="QM9/DeepChem public dataset; used as a conformation-sensitive quantum-property stress test.",
        raw_name="QM9",
    ),
    "QM9_R2": DatasetSpec(
        "QM9_R2",
        "regression",
        "MAE",
        ("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv",),
        "smiles",
        "r2",
        license_note="QM9/DeepChem public dataset; used as a conformation-sensitive quantum-property stress test.",
        raw_name="QM9",
    ),
}


def _download(url: str, dest: Path, timeout: int = 120) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    req = Request(url, headers={"User-Agent": "VOILA-3D/0.1"})
    with urlopen(req, timeout=timeout) as r:
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)


def fetch_dataset(spec: DatasetSpec, raw_dir: str | Path) -> Path:
    raw_dir = ensure_dir(raw_dir)
    suffix = ".csv.gz" if spec.urls[0].endswith(".gz") else ".csv"
    dest = raw_dir / f"{spec.raw_name or spec.name}{suffix}"
    if dest.exists():
        return dest
    last_error: Exception | None = None
    for url in spec.urls:
        try:
            _download(url, dest)
            return dest
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
    raise RuntimeError(f"Failed to download {spec.name}: {last_error}")


def _read_csv(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as src:
            tmp = path.with_suffix("")
            with open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return pd.read_csv(tmp)
    return pd.read_csv(path)


def _pick_col(df: pd.DataFrame, col: str | tuple[str, ...]) -> str:
    if isinstance(col, str):
        if col in df.columns:
            return col
        raise KeyError(f"Column {col!r} not found in {list(df.columns)}")
    for c in col:
        if c in df.columns:
            return c
    raise KeyError(f"None of {col!r} found in {list(df.columns)}")


def canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_dataset(name: str, raw_dir: str | Path, max_mols: int | None = None, seed: int = 0) -> tuple[pd.DataFrame, DatasetSpec, Path]:
    spec = DATASETS[name]
    path = fetch_dataset(spec, raw_dir)
    df0 = _read_csv(path)
    smiles_col = _pick_col(df0, spec.smiles_col)
    if spec.target_col not in df0.columns:
        raise KeyError(f"Column {spec.target_col!r} not found in {list(df0.columns)}")

    df = df0[[smiles_col, spec.target_col]].rename(columns={smiles_col: "smiles", spec.target_col: "y"}).copy()
    df["source_row"] = np.arange(len(df))
    df["smiles"] = df["smiles"].astype(str)
    df["canonical_smiles"] = df["smiles"].map(canonical_smiles)
    df["valid_smiles"] = df["canonical_smiles"].notna()
    df = df[df["valid_smiles"]].dropna(subset=["y"]).copy()
    df = df.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
    if spec.task_type == "classification":
        df["y"] = df["y"].astype(int)
        df = df[df["y"].isin([0, 1])].copy()
    else:
        df["y"] = df["y"].astype(float)

    if max_mols is not None and len(df) > max_mols:
        df = df.sample(max_mols, random_state=seed).reset_index(drop=True)
    df["mol_id"] = [f"{name}_{i:06d}" for i in range(len(df))]
    return df, spec, path


def scaffold_for_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold or smiles


def _assign_balanced_scaffold_splits(
    df: pd.DataFrame,
    seed: int = 0,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
) -> pd.Series:
    n = len(df)
    labels = np.array(["test"] * n, dtype=object)
    frac_test = max(0.0, 1.0 - frac_train - frac_val)
    target_sizes = {
        "train": max(1, int(round(frac_train * n))),
        "val": max(1, int(round(frac_val * n))),
        "test": max(1, n - int(round(frac_train * n)) - int(round(frac_val * n))),
    }
    y = df["y"].astype(int).to_numpy()
    total_pos = int(y.sum())
    total_neg = int(n - total_pos)
    target_pos = {
        "train": max(1, int(round(frac_train * total_pos))),
        "val": max(1, int(round(frac_val * total_pos))),
        "test": max(1, total_pos - int(round(frac_train * total_pos)) - int(round(frac_val * total_pos))),
    }
    target_neg = {
        "train": max(1, int(round(frac_train * total_neg))),
        "val": max(1, int(round(frac_val * total_neg))),
        "test": max(1, total_neg - int(round(frac_train * total_neg)) - int(round(frac_val * total_neg))),
    }
    scaffolds: dict[str, list[int]] = {}
    for i, smi in enumerate(df["canonical_smiles"]):
        scaffolds.setdefault(scaffold_for_smiles(smi), []).append(i)
    rng = np.random.default_rng(seed)
    groups = list(scaffolds.values())
    rng.shuffle(groups)
    groups = sorted(groups, key=lambda idx: (-len(idx), -int(y[idx].sum())))
    assigned: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    pos_counts = {"train": 0, "val": 0, "test": 0}
    neg_counts = {"train": 0, "val": 0, "test": 0}

    for group in groups:
        g_pos = int(y[group].sum())
        g_neg = int(len(group) - g_pos)
        best_split = None
        best_score = float("inf")
        for name in ("train", "val", "test"):
            size_after = len(assigned[name]) + len(group)
            pos_after = pos_counts[name] + g_pos
            neg_after = neg_counts[name] + g_neg
            over = max(0, size_after - target_sizes[name]) / max(1, target_sizes[name])
            size_gap = abs(size_after - target_sizes[name]) / max(1, target_sizes[name])
            pos_gap = abs(pos_after - target_pos[name]) / max(1, target_pos[name])
            neg_gap = abs(neg_after - target_neg[name]) / max(1, target_neg[name])
            score = 1.5 * over + size_gap + 0.75 * pos_gap + 0.75 * neg_gap
            if score < best_score:
                best_split = name
                best_score = score
        assert best_split is not None
        assigned[best_split].extend(group)
        pos_counts[best_split] += g_pos
        neg_counts[best_split] += g_neg

    for name, idx in assigned.items():
        labels[idx] = name
    return pd.Series(labels, index=df.index)


def assign_splits(
    df: pd.DataFrame,
    split: str,
    seed: int = 0,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    task_type: TaskType | None = None,
) -> pd.Series:
    n = len(df)
    rng = np.random.default_rng(seed)
    labels = np.array(["test"] * n, dtype=object)
    if split == "random":
        idx = rng.permutation(n)
        n_train = int(frac_train * n)
        n_val = int(frac_val * n)
        labels[idx[:n_train]] = "train"
        labels[idx[n_train:n_train + n_val]] = "val"
        return pd.Series(labels, index=df.index)
    if split == "scaffold_balanced" or (split == "scaffold" and task_type == "classification"):
        return _assign_balanced_scaffold_splits(df, seed=seed, frac_train=frac_train, frac_val=frac_val)
    if split != "scaffold":
        raise ValueError(f"Unknown split: {split}")

    scaffolds: dict[str, list[int]] = {}
    for i, smi in enumerate(df["canonical_smiles"]):
        scaffolds.setdefault(scaffold_for_smiles(smi), []).append(i)
    groups = sorted(scaffolds.values(), key=lambda x: (-len(x), x[0]))
    n_train_target = int(frac_train * n)
    n_val_target = int(frac_val * n)
    train, val, test = [], [], []
    for group in groups:
        if len(train) + len(group) <= n_train_target:
            train.extend(group)
        elif len(val) + len(group) <= n_val_target:
            val.extend(group)
        else:
            test.extend(group)
    labels[train] = "train"
    labels[val] = "val"
    labels[test] = "test"
    return pd.Series(labels, index=df.index)


def dataset_card(
    df: pd.DataFrame,
    spec: DatasetSpec,
    raw_path: Path,
    split: str,
    conformer_success_rate: float | None,
    mean_conformer_time: float | None,
    out_path: str | Path,
) -> None:
    y_counts = df["y"].value_counts(dropna=False).to_dict() if spec.task_type == "classification" else None
    card = {
        "dataset": spec.name,
        "task_type": spec.task_type,
        "primary_metric": spec.metric,
        "split": split,
        "n_samples_after_cleaning": int(len(df)),
        "invalid_smiles_excluded": "computed before de-duplication in loader logs",
        "target_summary": {
            "mean": float(df["y"].mean()) if spec.task_type == "regression" else None,
            "std": float(df["y"].std()) if spec.task_type == "regression" else None,
            "class_counts": y_counts,
        },
        "split_counts": df["split"].value_counts().to_dict() if "split" in df else None,
        "conformer_success_rate": conformer_success_rate,
        "mean_conformer_time_sec": mean_conformer_time,
        "mean_heavy_atoms": float(df.get("heavy_atoms", pd.Series(dtype=float)).mean()) if "heavy_atoms" in df else None,
        "mean_rotatable_bonds": float(df.get("rotatable_bonds", pd.Series(dtype=float)).mean()) if "rotatable_bonds" in df else None,
        "license": spec.license_note,
        "raw_file": str(raw_path),
        "raw_file_sha256": sha256_file(raw_path),
    }
    Path(out_path).write_text(yaml.safe_dump(card, sort_keys=False, allow_unicode=True), encoding="utf-8")
