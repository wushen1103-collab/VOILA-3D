from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voila3d.data import DATASETS, assign_splits, load_dataset
from voila3d.metrics import classification_metrics, primary_metric, regression_metrics
from voila3d.utils import ensure_dir, now_iso


VARIANTS = {
    "dmpnn": [],
    "dmpnn_rdkit2d": ["--features_generator", "rdkit_2d_normalized", "--no_features_scaling"],
    "dmpnn_rdkit2d_balanced": [
        "--features_generator",
        "rdkit_2d_normalized",
        "--no_features_scaling",
        "--class_balance",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["ESOL", "FreeSolv", "Lipophilicity", "BBBP", "BACE", "HIV"])
    p.add_argument("--variants", nargs="+", default=["dmpnn", "dmpnn_rdkit2d", "dmpnn_rdkit2d_balanced"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--split", choices=["scaffold", "scaffold_balanced", "random"], default="scaffold_balanced")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--max-mols", type=int, default=12000)
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--out-dir", default="results/chemprop_baselines_5seed")
    p.add_argument("--chemprop-bin", default=".chemprop_venv/bin/chemprop_train")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cuda-visible-devices", default=None)
    p.add_argument("--hidden-size", type=int, default=300)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--ffn-num-layers", type=int, default=2)
    p.add_argument("--quiet", action="store_true", default=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _task_metric_args(task_type: str) -> tuple[str, list[str]]:
    if task_type == "classification":
        return "auc", ["prc-auc", "accuracy", "binary_cross_entropy"]
    return "mae", ["rmse", "r2"]


def _write_split_csvs(task: str, args: argparse.Namespace, out_dir: Path) -> tuple[dict[str, Path], str, dict[str, int]]:
    split_dir = ensure_dir(out_dir / "data" / task)
    paths = {name: split_dir / f"{task}_{name}.csv" for name in ["train", "val", "test"]}
    if all(path.exists() for path in paths.values()):
        meta = pd.read_csv(split_dir / f"{task}_meta.csv")
        task_type = str(meta["task_type"].iloc[0])
        counts = {name: int(pd.read_csv(path).shape[0]) for name, path in paths.items()}
        return paths, task_type, counts

    df, spec, _ = load_dataset(task, args.data_dir, args.max_mols, seed=args.split_seed)
    df["split"] = assign_splits(df, args.split, seed=args.split_seed, task_type=spec.task_type)
    for name, path in paths.items():
        sub = df[df["split"] == name][["canonical_smiles", "y"]].rename(columns={"canonical_smiles": "smiles"})
        sub.to_csv(path, index=False)
    meta = pd.DataFrame(
        [
            {
                "task": task,
                "task_type": spec.task_type,
                "split": args.split,
                "split_seed": args.split_seed,
                "max_mols": args.max_mols,
                "n_train": int((df["split"] == "train").sum()),
                "n_val": int((df["split"] == "val").sum()),
                "n_test": int((df["split"] == "test").sum()),
            }
        ]
    )
    meta.to_csv(split_dir / f"{task}_meta.csv", index=False)
    counts = {"train": int(meta["n_train"].iloc[0]), "val": int(meta["n_val"].iloc[0]), "test": int(meta["n_test"].iloc[0])}
    return paths, spec.task_type, counts


def _variant_allowed(variant: str, task_type: str) -> bool:
    if variant == "dmpnn_rdkit2d_balanced" and task_type != "classification":
        return False
    return True


def _run_one(
    task: str,
    task_type: str,
    variant: str,
    seed: int,
    paths: dict[str, Path],
    counts: dict[str, int],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    run_dir = out_dir / "runs" / variant / task / f"seed{seed}"
    ensure_dir(run_dir)
    scores_path = run_dir / "test_scores.csv"
    preds_path = run_dir / "test_preds.csv"
    if scores_path.exists() and preds_path.exists() and not args.overwrite:
        status = "cached"
    else:
        metric, extra_metrics = _task_metric_args(task_type)
        cmd = [
            str(Path(args.chemprop_bin)),
            "--data_path",
            str(paths["train"]),
            "--separate_val_path",
            str(paths["val"]),
            "--separate_test_path",
            str(paths["test"]),
            "--dataset_type",
            task_type,
            "--target_columns",
            "y",
            "--metric",
            metric,
            "--extra_metrics",
            *extra_metrics,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
            "--hidden_size",
            str(args.hidden_size),
            "--depth",
            str(args.depth),
            "--dropout",
            str(args.dropout),
            "--ffn_num_layers",
            str(args.ffn_num_layers),
            "--seed",
            str(seed),
            "--pytorch_seed",
            str(seed),
            "--gpu",
            str(args.gpu),
            "--save_preds",
            "--save_dir",
            str(run_dir),
            "--quiet",
            *VARIANTS[variant],
        ]
        env = os.environ.copy()
        if args.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
            env.setdefault(var, "8")
        log_path = run_dir / "driver.log"
        with log_path.open("w", encoding="utf-8") as log:
            log.write(" ".join(cmd) + "\n\n")
            proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        status = "done" if proc.returncode == 0 else f"failed:{proc.returncode}"
        if proc.returncode != 0:
            return {
                "task": task,
                "task_type": task_type,
                "variant": variant,
                "seed": seed,
                "status": status,
                "run_dir": str(run_dir),
                "primary_metric": primary_metric(task_type),
                "n_train": counts["train"],
                "n_val": counts["val"],
                "n_test": counts["test"],
            }

    y_true = pd.read_csv(paths["test"])["y"].to_numpy()
    pred = pd.read_csv(preds_path)["y"].to_numpy(dtype=float)
    metrics = classification_metrics(y_true.astype(int), pred) if task_type == "classification" else regression_metrics(y_true, pred)
    row = {
        "task": task,
        "task_type": task_type,
        "variant": variant,
        "seed": seed,
        "status": status,
        "run_dir": str(run_dir),
        "primary_metric": primary_metric(task_type),
        "primary_value": float(metrics[primary_metric(task_type)]),
        "n_train": counts["train"],
        "n_val": counts["val"],
        "n_test": counts["test"],
    }
    row.update(metrics)
    return row


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = metrics[metrics["status"].isin(["done", "cached"])].copy()
    for (task, task_type, variant), g in ok.groupby(["task", "task_type", "variant"]):
        pmet = primary_metric(task_type)
        row = {
            "task": task,
            "task_type": task_type,
            "variant": variant,
            "method_group": "Chemprop D-MPNN",
            "primary_metric": pmet,
            "primary_mean": float(g[pmet].mean()),
            "primary_std": float(g[pmet].std(ddof=0)),
            "n_seeds": int(g["seed"].nunique()),
            "n_train": int(g["n_train"].iloc[0]),
            "n_val": int(g["n_val"].iloc[0]),
            "n_test": int(g["n_test"].iloc[0]),
            "source": "ours_rerun_same_split",
        }
        for metric in ["AUROC", "AUPRC", "Accuracy", "LogLoss", "MAE", "RMSE", "R2"]:
            if metric in g.columns:
                vals = pd.to_numeric(g[metric], errors="coerce")
                if vals.notna().any():
                    row[f"{metric}_mean"] = float(vals.mean())
                    row[f"{metric}_std"] = float(vals.std(ddof=0))
                    row[f"{metric}_mean_std"] = f"{vals.mean():.6f} +/- {vals.std(ddof=0):.6f}"
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["task", "variant"]) if rows else pd.DataFrame()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    invalid_tasks = [task for task in args.tasks if task not in DATASETS]
    if invalid_tasks:
        raise ValueError(f"Unknown tasks: {invalid_tasks}")
    invalid_variants = [variant for variant in args.variants if variant not in VARIANTS]
    if invalid_variants:
        raise ValueError(f"Unknown variants: {invalid_variants}")
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "started_at": now_iso(),
                "argv": sys.argv,
                "python": sys.version,
                "platform": platform.platform(),
                "tasks": args.tasks,
                "variants": args.variants,
                "seeds": args.seeds,
                "split": args.split,
                "split_seed": args.split_seed,
                "max_mols": args.max_mols,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "hidden_size": args.hidden_size,
                "depth": args.depth,
                "dropout": args.dropout,
                "ffn_num_layers": args.ffn_num_layers,
                "cuda_visible_devices": args.cuda_visible_devices,
                "note": "Chemprop v1.6.1 D-MPNN baselines rerun on exported same split train/val/test CSV files.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = []
    for task in args.tasks:
        paths, task_type, counts = _write_split_csvs(task, args, out_dir)
        for variant in args.variants:
            if not _variant_allowed(variant, task_type):
                continue
            for seed in args.seeds:
                print(f"[chemprop] task={task} variant={variant} seed={seed}", flush=True)
                row = _run_one(task, task_type, variant, seed, paths, counts, args, out_dir)
                rows.append(row)
                pd.DataFrame(rows).to_csv(out_dir / "metrics_long.csv", index=False)
                summarize(pd.DataFrame(rows)).to_csv(out_dir / "method_summary.csv", index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics_long.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(out_dir / "method_summary.csv", index=False)
    failures = metrics[~metrics["status"].isin(["done", "cached"])]
    if not failures.empty:
        failures.to_csv(out_dir / "failures.csv", index=False)
        raise SystemExit(f"Chemprop failures encountered: {len(failures)}")
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
