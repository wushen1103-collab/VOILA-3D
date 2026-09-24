from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


RAW_DATA_SHA256 = {
    "BACE.csv": "f3fb9ce90bada3e2bd6148b0df13f8f8145a357bf87df0dd5b391ede974fc737",
    "BBBP.csv": "d07a38487aeac5cee5508413e468043ef3097451d2a112701c2d60be9ec6b662",
    "ESOL.csv": "8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189",
    "FreeSolv.csv": "ab5895d914ee87cb563bd7b9611e869527bba45bec6b014d34dc495a0f9dcb72",
    "HIV.csv": "9ffa7fe57dc86c342627ee1d5255e937e2ab812393c73c4d16c697022f6e1d22",
    "Lipophilicity.csv": "aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd",
    "QM9.csv": "3e668f8c34e4bc392a90d417a50a5eed3b64b842a817a633024bdc054c68ccb4",
}

PRIMARY_TASKS = {"BACE", "BBBP", "ESOL", "FreeSolv", "HIV", "Lipophilicity"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify bundled VOILA-3D reference artifacts.")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("results/reference"),
        help="Directory containing SHA256SUMS.txt and compact reference CSV files.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional raw-data directory to verify against the recorded input hashes.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, relpath = line.split(maxsplit=1)
        entries[relpath.replace("\\", "/")] = digest
    return entries


def verify_reference_files(reference_dir: Path) -> None:
    manifest_path = reference_dir / "SHA256SUMS.txt"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing checksum manifest: {manifest_path}")
    entries = read_manifest(manifest_path)
    if not entries:
        raise ValueError("The reference checksum manifest is empty.")

    failures: list[str] = []
    for relpath, expected in entries.items():
        path = reference_dir / relpath
        if not path.is_file():
            failures.append(f"missing: {relpath}")
            continue
        observed = sha256(path)
        if observed != expected:
            failures.append(f"hash mismatch: {relpath}")
    if failures:
        raise RuntimeError("Reference verification failed:\n" + "\n".join(failures))

    extra_csv = {
        path.relative_to(reference_dir).as_posix()
        for path in reference_dir.rglob("*.csv")
        if path.is_file()
    } - set(entries)
    if extra_csv:
        raise RuntimeError("CSV files missing from checksum manifest: " + ", ".join(sorted(extra_csv)))

    operational = pd.read_csv(
        reference_dir
        / "if_router_gated_selection_lcb_v1_5seed"
        / "budget_integrated_gain_summary.csv"
    )
    tasks = set(operational.loc[operational["router"] == "R9_oof_safe_guardrail", "task"])
    if tasks != PRIMARY_TASKS:
        raise RuntimeError(f"Operational R9 task set mismatch: {sorted(tasks)}")
    if set(operational["n_seeds"].astype(int)) != {5}:
        raise RuntimeError("Operational reference results are not uniformly five-seed summaries.")

    nested = pd.read_csv(
        reference_dir
        / "if_router_nested_calibration_lcb_5seed"
        / "nested_selection_long.csv"
    )
    if set(nested["seed"].astype(int)) != {0, 1, 2, 3, 4}:
        raise RuntimeError("Fold-separated calibration does not contain the expected seeds 0--4.")

    partition_dir = reference_dir / "partition_robustness"
    partition_macro = pd.read_csv(partition_dir / "macro_by_partition.csv")
    if set(partition_macro["split_seed"].astype(int)) != {1, 2, 3, 4, 5}:
        raise RuntimeError("Partition robustness does not contain split seeds 1--5.")
    if set(partition_macro["policy"]) != {"Random", "R7", "R8", "R9"}:
        raise RuntimeError("Partition robustness policy set is incomplete.")
    partition_experts = pd.read_csv(partition_dir / "expert_summary.csv")
    if set(partition_experts["task"]) != PRIMARY_TASKS:
        raise RuntimeError("Partition robustness expert task set is incomplete.")
    if set(partition_experts["n_partitions"].astype(int)) != {5}:
        raise RuntimeError("Partition expert summaries do not uniformly use five assignments.")

    print(f"Verified {len(entries)} reference CSV files and five-seed protocol invariants.")


def verify_raw_data(data_dir: Path) -> None:
    failures: list[str] = []
    for filename, expected in RAW_DATA_SHA256.items():
        path = data_dir / filename
        if not path.is_file():
            failures.append(f"missing: {filename}")
        elif sha256(path) != expected:
            failures.append(f"hash mismatch: {filename}")
    if failures:
        raise RuntimeError("Raw-data verification failed:\n" + "\n".join(failures))
    print(f"Verified {len(RAW_DATA_SHA256)} raw dataset files.")


def main() -> None:
    args = parse_args()
    verify_reference_files(args.reference_dir)
    if args.data_dir is not None:
        verify_raw_data(args.data_dir)


if __name__ == "__main__":
    main()
