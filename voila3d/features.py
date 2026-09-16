from __future__ import annotations

import time
import signal
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, MACCSkeys, rdMolAlign, rdMolDescriptors, rdPartialCharges


DESCRIPTOR_NAMES = [
    "mol_wt",
    "heavy_atoms",
    "rotatable_bonds",
    "tpsa",
    "logp",
    "rings",
    "hbd",
    "hba",
    "fraction_csp3",
    "bertz_ct",
]
RDKIT2D_DESCRIPTOR_FNS = list(Descriptors.descList)

RDLogger.DisableLog("rdApp.warning")

ORGANIC_ENSEMBLE_ATOMS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}
FULL_3D_DESCRIPTOR_SPECS = [
    ("autocorr3d", rdMolDescriptors.CalcAUTOCORR3D, 80),
    ("rdf", rdMolDescriptors.CalcRDF, 210),
    ("morse", rdMolDescriptors.CalcMORSE, 224),
    ("whim", rdMolDescriptors.CalcWHIM, 114),
    ("getaway", rdMolDescriptors.CalcGETAWAY, 273),
]
RICH_3D_DESCRIPTOR_SPECS = FULL_3D_DESCRIPTOR_SPECS[:-1]
SCALAR_3D_DESCRIPTOR_SPECS = [
    ("pmi1", rdMolDescriptors.CalcPMI1),
    ("pmi2", rdMolDescriptors.CalcPMI2),
    ("pmi3", rdMolDescriptors.CalcPMI3),
    ("npr1", rdMolDescriptors.CalcNPR1),
    ("npr2", rdMolDescriptors.CalcNPR2),
    ("radius_gyration", rdMolDescriptors.CalcRadiusOfGyration),
    ("inertial_shape_factor", rdMolDescriptors.CalcInertialShapeFactor),
    ("eccentricity", rdMolDescriptors.CalcEccentricity),
    ("asphericity", rdMolDescriptors.CalcAsphericity),
    ("spherocity", rdMolDescriptors.CalcSpherocityIndex),
    ("pbf", rdMolDescriptors.CalcPBF),
]
PHYSICOCHEM_DESCRIPTOR_NAMES = [
    name
    for name, _ in Descriptors.descList
    if (
        "VSA" in name
        or "EState" in name
        or name
        in {
            "MolMR",
            "LabuteASA",
            "BalabanJ",
            "HallKierAlpha",
            "Kappa1",
            "Kappa2",
            "Kappa3",
        }
    )
]
PHYSICOCHEM_DESCRIPTOR_FNS = [
    (name, fn) for name, fn in Descriptors.descList if name in set(PHYSICOCHEM_DESCRIPTOR_NAMES)
]
CHARGE_FEATURE_DIM = 8
SCALAR_3D_DIM = 72 + len(SCALAR_3D_DESCRIPTOR_SPECS) + CHARGE_FEATURE_DIM + len(PHYSICOCHEM_DESCRIPTOR_FNS)
CONF_FEATURE_DIMS = {
    "usr": 72,
    "rdkit_scalar": SCALAR_3D_DIM,
    "rdkit_rich": 72 + sum(width for _, _, width in RICH_3D_DESCRIPTOR_SPECS),
    "rdkit_full": 72 + sum(width for _, _, width in FULL_3D_DESCRIPTOR_SPECS),
}


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(str(smiles))


def ecfp_bits(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    mol = mol_from_smiles(smiles)
    arr = np.zeros((n_bits,), dtype=np.float32)
    if mol is None:
        return arr
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _bitvect_to_array(fp: Any, n_bits: int) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def maccs_bits(smiles: str) -> np.ndarray:
    mol = mol_from_smiles(smiles)
    n_bits = 167
    if mol is None:
        return np.zeros((n_bits,), dtype=np.float32)
    return _bitvect_to_array(MACCSkeys.GenMACCSKeys(mol), n_bits)


def rdkit2d_descriptors(smiles: str) -> np.ndarray:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return np.full(len(RDKIT2D_DESCRIPTOR_FNS), np.nan, dtype=np.float32)
    vals = []
    for _, fn in RDKIT2D_DESCRIPTOR_FNS:
        try:
            vals.append(float(fn(mol)))
        except Exception:
            vals.append(float("nan"))
    arr = np.asarray(vals, dtype=np.float32)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def combo_2d_features(smiles: str, n_bits: int = 2048) -> np.ndarray:
    mol = mol_from_smiles(smiles)
    if mol is None:
        n = n_bits * 4 + 167 + len(RDKIT2D_DESCRIPTOR_FNS)
        return np.full(n, np.nan, dtype=np.float32)
    ecfp3 = _bitvect_to_array(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=n_bits), n_bits)
    fcfp2 = _bitvect_to_array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits, useFeatures=True), n_bits)
    atom_pair = _bitvect_to_array(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits), n_bits)
    torsion = _bitvect_to_array(rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits), n_bits)
    return np.concatenate([ecfp3, fcfp2, atom_pair, torsion, maccs_bits(smiles), rdkit2d_descriptors(smiles)]).astype(np.float32)


def build_2d_feature_matrix(
    smiles: list[str],
    x_ecfp: np.ndarray,
    x_desc: np.ndarray,
    feature2d_set: str = "ecfp_desc",
    n_bits: int = 2048,
    jobs: int = 16,
) -> np.ndarray:
    base = np.hstack([x_ecfp, x_desc]).astype(np.float32)
    if feature2d_set == "ecfp_desc":
        return base
    if feature2d_set == "rdkit2d_combo":
        extras = Parallel(n_jobs=jobs, backend="loky", verbose=0)(
            delayed(combo_2d_features)(s, n_bits=n_bits) for s in smiles
        )
        return np.hstack([base, np.vstack(extras)]).astype(np.float32)
    raise ValueError(f"Unknown 2D feature set: {feature2d_set}")


def cheap_descriptors(smiles: str) -> np.ndarray:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return np.full(len(DESCRIPTOR_NAMES), np.nan, dtype=np.float32)
    vals = [
        Descriptors.MolWt(mol),
        mol.GetNumHeavyAtoms(),
        Lipinski.NumRotatableBonds(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcNumRings(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        Descriptors.BertzCT(mol),
    ]
    return np.asarray(vals, dtype=np.float32)


@dataclass
class ConformerResult:
    mol_id: str
    smiles: str
    success: bool
    elapsed_sec: float
    n_conformers: int
    energy_min: float | None
    energy_mean: float | None
    energy_std: float | None
    rmsd_mean: float | None
    failure_reason: str | None
    features: np.ndarray


class ConformerTimeoutError(TimeoutError):
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    raise ConformerTimeoutError("conformer_generation_timeout")


def _optimize_conformer(mol: Chem.Mol, conf_id: int) -> float | None:
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
            AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", confId=conf_id, maxIters=200)
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
            return float(ff.CalcEnergy()) if ff is not None else None
    except Exception:
        pass
    try:
        if not AllChem.UFFHasAllMoleculeParams(mol):
            return None
        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=200)
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
        return float(ff.CalcEnergy()) if ff is not None else None
    except Exception:
        return None


def _has_force_field_params(mol: Chem.Mol) -> bool:
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            return True
    except Exception:
        pass
    try:
        return bool(AllChem.UFFHasAllMoleculeParams(mol))
    except Exception:
        return False


def _conformer_rmsd_mean(mol: Chem.Mol, conf_ids: list[int]) -> float:
    if len(conf_ids) <= 1:
        return 0.0
    rms_fn = getattr(AllChem, "GetConformerRMSMatrix", None)
    if rms_fn is None:
        rms_fn = getattr(rdMolAlign, "GetConformerRMSMatrix", None)
    if rms_fn is not None:
        try:
            rms = list(rms_fn(mol, prealigned=False))
            return float(np.mean(rms)) if rms else 0.0
        except TypeError:
            rms = list(rms_fn(mol))
            return float(np.mean(rms)) if rms else 0.0

    rms_values: list[float] = []
    for i, ci in enumerate(conf_ids):
        for cj in conf_ids[i + 1 :]:
            try:
                rms_values.append(float(AllChem.GetBestRMS(mol, mol, int(ci), int(cj))))
            except Exception:
                continue
    return float(np.mean(rms_values)) if rms_values else 0.0


def _usr_features_for_conf(mol: Chem.Mol, conf_id: int) -> np.ndarray:
    vals: list[float] = []
    for fn in (rdMolDescriptors.GetUSR, rdMolDescriptors.GetUSRCAT):
        try:
            vals.extend(float(x) for x in fn(mol, confId=conf_id))
        except Exception:
            vals.extend([np.nan] * (12 if fn is rdMolDescriptors.GetUSR else 60))
    return np.asarray(vals, dtype=np.float32)


def _fixed_width(values: list[float], width: int) -> list[float]:
    if len(values) == width:
        return values
    if len(values) > width:
        return values[:width]
    return values + [np.nan] * (width - len(values))


def _rdkit_descriptor_features_for_conf(
    mol: Chem.Mol,
    conf_id: int,
    specs: list[tuple[str, Any, int]],
) -> np.ndarray:
    vals = list(_usr_features_for_conf(mol, conf_id).astype(float))
    for _, fn, width in specs:
        try:
            block = [float(x) for x in fn(mol, confId=conf_id)]
        except Exception:
            block = [np.nan] * width
        vals.extend(_fixed_width(block, width))
    return np.asarray(vals, dtype=np.float32)


def _call_scalar_3d(fn: Any, mol: Chem.Mol, conf_id: int) -> float:
    try:
        return float(fn(mol, confId=conf_id))
    except TypeError:
        return float(fn(mol))
    except Exception:
        return float("nan")


def _charge_summary_features(mol: Chem.Mol) -> list[float]:
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol, throwOnParamFailure=False)
        vals = []
        for atom in mol.GetAtoms():
            raw = atom.GetProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else "nan"
            try:
                value = float(raw)
            except ValueError:
                value = float("nan")
            if np.isfinite(value):
                vals.append(value)
        if not vals:
            return [float("nan")] * CHARGE_FEATURE_DIM
        arr = np.asarray(vals, dtype=float)
        pos = arr[arr > 0.0]
        neg = arr[arr < 0.0]
        return [
            float(np.min(arr)),
            float(np.max(arr)),
            float(np.mean(arr)),
            float(np.std(arr)),
            float(np.mean(np.abs(arr))),
            float(np.max(np.abs(arr))),
            float(np.sum(pos)) if len(pos) else 0.0,
            float(np.sum(neg)) if len(neg) else 0.0,
        ]
    except Exception:
        return [float("nan")] * CHARGE_FEATURE_DIM


def _physicochem_descriptor_features(mol: Chem.Mol) -> list[float]:
    vals = []
    for _, fn in PHYSICOCHEM_DESCRIPTOR_FNS:
        try:
            vals.append(float(fn(mol)))
        except Exception:
            vals.append(float("nan"))
    return vals


def _scalar_3d_features_for_conf(mol: Chem.Mol, conf_id: int) -> np.ndarray:
    vals = list(_usr_features_for_conf(mol, conf_id).astype(float))
    vals.extend(_call_scalar_3d(fn, mol, conf_id) for _, fn in SCALAR_3D_DESCRIPTOR_SPECS)
    vals.extend(_charge_summary_features(mol))
    vals.extend(_physicochem_descriptor_features(mol))
    return np.asarray(vals, dtype=np.float32)


def _features_for_conf(mol: Chem.Mol, conf_id: int, feature3d_set: str) -> np.ndarray:
    if feature3d_set == "usr":
        return _usr_features_for_conf(mol, conf_id)
    if feature3d_set == "rdkit_scalar":
        return _scalar_3d_features_for_conf(mol, conf_id)
    if feature3d_set == "rdkit_rich":
        return _rdkit_descriptor_features_for_conf(mol, conf_id, RICH_3D_DESCRIPTOR_SPECS)
    if feature3d_set == "rdkit_full":
        return _rdkit_descriptor_features_for_conf(mol, conf_id, FULL_3D_DESCRIPTOR_SPECS)
    raise ValueError(f"Unknown 3D feature set: {feature3d_set}")


def conformer_features(
    mol_id: str,
    smiles: str,
    n_conformers: int = 1,
    seed: int = 0,
    max_iterations: int = 1000,
    timeout_sec: int = 45,
    feature3d_set: str = "usr",
) -> ConformerResult:
    start = time.time()
    if feature3d_set not in CONF_FEATURE_DIMS:
        raise ValueError(f"Unknown 3D feature set: {feature3d_set}")
    base_len = CONF_FEATURE_DIMS[feature3d_set] * 2 + 4
    empty = np.full(base_len, np.nan, dtype=np.float32)
    alarm_enabled = hasattr(signal, "SIGALRM") and timeout_sec > 0
    old_handler = None
    if alarm_enabled:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_sec)
    mol = mol_from_smiles(smiles)
    if mol is None:
        if alarm_enabled:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
        return ConformerResult(mol_id, smiles, False, time.time() - start, 0, None, None, None, None, "invalid_smiles", empty)
    if n_conformers > 1 and any(atom.GetAtomicNum() not in ORGANIC_ENSEMBLE_ATOMS for atom in mol.GetAtoms()):
        if alarm_enabled:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
        return ConformerResult(
            mol_id,
            smiles,
            False,
            time.time() - start,
            0,
            None,
            None,
            None,
            None,
            "unsupported_element_for_ensemble",
            empty,
        )
    try:
        mol_h = Chem.AddHs(mol)
        if n_conformers > 1 and not _has_force_field_params(mol_h):
            return ConformerResult(
                mol_id,
                smiles,
                False,
                time.time() - start,
                0,
                None,
                None,
                None,
                None,
                "forcefield_unavailable_for_ensemble",
                empty,
            )
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        params.maxIterations = int(max_iterations)
        params.useRandomCoords = True
        params.useMacrocycleTorsions = True
        params.pruneRmsThresh = 0.1
        conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=int(n_conformers), params=params))
        if not conf_ids:
            return ConformerResult(mol_id, smiles, False, time.time() - start, 0, None, None, None, None, "embed_failed", empty)
        energies: list[float] = []
        usr_rows: list[np.ndarray] = []
        for cid in conf_ids:
            energy = _optimize_conformer(mol_h, cid)
            if energy is not None and np.isfinite(energy):
                energies.append(float(energy))
            usr_rows.append(_features_for_conf(mol_h, cid, feature3d_set))
        usr = np.vstack(usr_rows)
        usr_mean = np.nanmean(usr, axis=0)
        usr_std = np.nanstd(usr, axis=0)
        rmsd_mean = _conformer_rmsd_mean(mol_h, conf_ids)
        e = np.asarray(energies, dtype=float) if energies else np.asarray([np.nan])
        tail = np.asarray([
            np.nanmin(e),
            np.nanmean(e),
            np.nanstd(e),
            rmsd_mean,
        ], dtype=np.float32)
        feats = np.concatenate([usr_mean, usr_std, tail]).astype(np.float32)
        return ConformerResult(
            mol_id,
            smiles,
            True,
            time.time() - start,
            len(conf_ids),
            float(np.nanmin(e)),
            float(np.nanmean(e)),
            float(np.nanstd(e)),
            rmsd_mean,
            None,
            feats,
        )
    except ConformerTimeoutError:
        return ConformerResult(mol_id, smiles, False, time.time() - start, 0, None, None, None, None, "timeout", empty)
    except Exception as exc:
        return ConformerResult(mol_id, smiles, False, time.time() - start, 0, None, None, None, None, type(exc).__name__, empty)
    finally:
        if alarm_enabled:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


def build_feature_tables(
    df: pd.DataFrame,
    n_bits: int = 2048,
    n_conformers: int = 1,
    conformer_jobs: int = 16,
    seed: int = 0,
    feature3d_set: str = "usr",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    smiles = df["canonical_smiles"].tolist()
    x_ecfp = np.vstack([ecfp_bits(s, n_bits=n_bits) for s in smiles])
    x_desc = np.vstack([cheap_descriptors(s) for s in smiles])
    confs = Parallel(n_jobs=conformer_jobs, backend="loky", verbose=5)(
        delayed(conformer_features)(row.mol_id, row.canonical_smiles, n_conformers, seed + i, feature3d_set=feature3d_set)
        for i, row in enumerate(df.itertuples(index=False))
    )
    x_3d = np.vstack([c.features for c in confs])
    manifest = pd.DataFrame([
        {
            "mol_id": c.mol_id,
            "smiles": c.smiles,
            "success": c.success,
            "elapsed_sec": c.elapsed_sec,
            "n_conformers": c.n_conformers,
            "energy_min": c.energy_min,
            "energy_mean": c.energy_mean,
            "energy_std": c.energy_std,
            "rmsd_mean": c.rmsd_mean,
            "failure_reason": c.failure_reason,
        }
        for c in confs
    ])
    return x_ecfp.astype(np.float32), x_desc.astype(np.float32), x_3d.astype(np.float32), manifest


def descriptor_frame(df: pd.DataFrame, x_desc: np.ndarray) -> pd.DataFrame:
    desc = pd.DataFrame(x_desc, columns=DESCRIPTOR_NAMES)
    desc.insert(0, "mol_id", df["mol_id"].to_numpy())
    return desc
