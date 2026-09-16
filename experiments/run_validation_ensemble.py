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

from experiments.run_baseline_matrix import _features, _method_specs, _predict_classification, _select_x
from experiments.run_chemprop_baseline import VARIANTS as CHEMPROP_VARIANTS
from voila3d.data import DATASETS, assign_splits, load_dataset
from voila3d.metrics import classification_metrics, higher_is_better, primary_metric, regression_metrics
from voila3d.utils import ensure_dir, now_iso, set_reproducible


CHEMPROP_PREDICT_VARIANT_ARGS = {
    name: [arg for arg in variant_args if arg != "--class_balance"]
    for name, variant_args in CHEMPROP_VARIANTS.items()
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["ESOL", "FreeSolv", "Lipophilicity", "BBBP", "BACE", "HIV"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--split", choices=["scaffold", "scaffold_balanced", "random"], default="scaffold_balanced")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--max-mols", type=int, default=12000)
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--out-dir", default="results/validation_ensemble_5seed")
    p.add_argument("--feature2d-set", choices=["ecfp_desc", "rdkit2d_combo"], default="rdkit2d_combo")
    p.add_argument("--combo-jobs", type=int, default=8)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--xgb-device", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--skip-svm-above", type=int, default=15000)
    p.add_argument("--skip-krr-above", type=int, default=2500)
    p.add_argument("--chemprop-root", action="append", default=[])
    p.add_argument("--chemprop-bin", default=".chemprop_venv/bin/chemprop_predict")
    p.add_argument("--chemprop-gpu", type=int, default=0)
    p.add_argument("--cuda-visible-devices", default=None)
    p.add_argument("--chemprop-no-cuda", action="store_true")
    p.add_argument("--chemprop-batch-size", type=int, default=256)
    p.add_argument("--chemprop-workers", type=int, default=4)
    p.add_argument("--ensemble-rounds", type=int, default=50)
    p.add_argument("--overwrite-preds", action="store_true")
    return p.parse_args()


def _metric_value(task_type: str, y: np.ndarray, pred: np.ndarray) -> float:
    if task_type == "classification":
        return float(classification_metrics(y.astype(int), np.clip(pred, 1e-6, 1 - 1e-6))[primary_metric(task_type)])
    return float(regression_metrics(y, pred)[primary_metric(task_type)])


def _is_better(task_type: str, a: float, b: float) -> bool:
    metric = primary_metric(task_type)
    return a > b if higher_is_better(metric) else a < b


def _eval_row(task: str, task_type: str, seed: int, method: str, y: np.ndarray, pred: np.ndarray) -> dict:
    mets = classification_metrics(y.astype(int), np.clip(pred, 1e-6, 1 - 1e-6)) if task_type == "classification" else regression_metrics(y, pred)
    row = {
        "task": task,
        "task_type": task_type,
        "seed": seed,
        "method": method,
        "primary_metric": primary_metric(task_type),
        "primary_value": float(mets[primary_metric(task_type)]),
    }
    row.update(mets)
    return row


def _summarize(rows: pd.DataFrame, method_group: str = "validation-gated consensus") -> pd.DataFrame:
    out = []
    metrics = [c for c in ["AUROC", "AUPRC", "Accuracy", "LogLoss", "MAE", "RMSE", "R2"] if c in rows.columns]
    for (task, task_type, method), g in rows.groupby(["task", "task_type", "method"]):
        pmet = primary_metric(task_type)
        rec = {
            "task": task,
            "task_type": task_type,
            "method": method,
            "method_group": method_group if method.startswith("val_") else "candidate member",
            "primary_metric": pmet,
            "primary_mean": float(g[pmet].mean()),
            "primary_std": float(g[pmet].std(ddof=0)),
            "primary_mean_std": f"{g[pmet].mean():.6f} +/- {g[pmet].std(ddof=0):.6f}",
            "n_seeds": int(g["seed"].nunique()),
            "source": "ours_rerun_same_split",
        }
        for metric in metrics:
            vals = pd.to_numeric(g[metric], errors="coerce")
            if vals.notna().any():
                rec[f"{metric}_mean"] = float(vals.mean())
                rec[f"{metric}_std"] = float(vals.std(ddof=0))
                rec[f"{metric}_mean_std"] = f"{vals.mean():.6f} +/- {vals.std(ddof=0):.6f}"
        out.append(rec)
    return pd.DataFrame(out).sort_values(["task", "method"]) if out else pd.DataFrame()


def _write_split_csvs(task: str, df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    split_dir = ensure_dir(out_dir / "data" / task)
    paths = {name: split_dir / f"{task}_{name}.csv" for name in ["train", "val", "test"]}
    for name, path in paths.items():
        sub = df[df["split"] == name][["canonical_smiles", "y"]].rename(columns={"canonical_smiles": "smiles"})
        if not path.exists():
            sub.to_csv(path, index=False)
    return paths


def _chemprop_predict(
    task: str,
    variant: str,
    seed: int,
    split_name: str,
    split_path: Path,
    checkpoint_dir: Path,
    args: argparse.Namespace,
    out_dir: Path,
) -> np.ndarray | None:
    if not checkpoint_dir.exists():
        return None
    if not any(checkpoint_dir.rglob("*.pt")):
        return None
    pred_dir = ensure_dir(out_dir / "chemprop_predictions" / variant / task / f"seed{seed}")
    pred_path = pred_dir / f"{split_name}_preds.csv"
    if not pred_path.exists() or args.overwrite_preds:
        cmd = [
            args.chemprop_bin,
            "--test_path",
            str(split_path),
            "--preds_path",
            str(pred_path),
            "--checkpoint_dir",
            str(checkpoint_dir),
            "--batch_size",
            str(args.chemprop_batch_size),
            "--num_workers",
            str(args.chemprop_workers),
            *CHEMPROP_PREDICT_VARIANT_ARGS[variant],
        ]
        if args.chemprop_no_cuda:
            cmd.append("--no_cuda")
        else:
            cmd.extend(["--gpu", str(args.chemprop_gpu)])
        env = os.environ.copy()
        if args.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
            env.setdefault(var, "8")
        log_path = pred_dir / f"{split_name}_predict.log"
        with log_path.open("w", encoding="utf-8") as log:
            log.write(" ".join(cmd) + "\n\n")
            proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            return None
    pred = pd.read_csv(pred_path)["y"].to_numpy(dtype=float)
    return pred


def _collect_chemprop_predictions(
    task: str,
    seed: int,
    paths: dict[str, Path],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for root_s in args.chemprop_root:
        root = Path(root_s)
        for variant in CHEMPROP_VARIANTS:
            run_dir = root / "runs" / variant / task / f"seed{seed}"
            val_pred = _chemprop_predict(task, variant, seed, "val", paths["val"], run_dir, args, out_dir)
            test_pred = _chemprop_predict(task, variant, seed, "test", paths["test"], run_dir, args, out_dir)
            if val_pred is not None and test_pred is not None:
                preds[f"chemprop_{variant}"] = (val_pred, test_pred)
    return preds


def _fit_member_predictions(task: str, task_type: str, df: pd.DataFrame, args: argparse.Namespace) -> dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]:
    _, x_desc, x2d = _features(df, args.feature2d_set, args.combo_jobs)
    y = df["y"].to_numpy()
    split = df["split"].to_numpy()
    train = split == "train"
    val = split == "val"
    test = split == "test"
    by_seed: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for seed in args.seeds:
        set_reproducible(seed)
        members: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        specs = _method_specs(task_type, args.threads, seed, int(train.sum()), args.xgb_device, y[train])
        for method, _, model in specs:
            if method == "SVM_RBF_Desc" and int(train.sum()) > args.skip_svm_above:
                continue
            if method == "KRR_RBF_Desc" and int(train.sum()) > args.skip_krr_above:
                continue
            if task_type == "classification" and method == "SVM_RBF_Desc":
                continue
            x = _select_x(method, x_desc, x2d)
            if task_type == "classification" and len(np.unique(y[train])) < 2:
                continue
            model.fit(x[train], y[train])
            if task_type == "classification":
                val_pred, probability_like_val = _predict_classification(model, x[val])
                test_pred, probability_like_test = _predict_classification(model, x[test])
                if not (probability_like_val and probability_like_test):
                    continue
                val_pred = np.clip(val_pred, 1e-6, 1 - 1e-6)
                test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)
            else:
                val_pred = model.predict(x[val]).astype(float)
                test_pred = model.predict(x[test]).astype(float)
            members[method] = (val_pred, test_pred)
        by_seed[seed] = members
    return by_seed


def _ensemble_selection(
    task_type: str,
    y_val: np.ndarray,
    val_preds: dict[str, np.ndarray],
    rounds: int,
) -> tuple[dict[str, float], float]:
    names = sorted(val_preds)
    best_name = names[0]
    best_score = _metric_value(task_type, y_val, val_preds[best_name])
    for name in names[1:]:
        score = _metric_value(task_type, y_val, val_preds[name])
        if _is_better(task_type, score, best_score):
            best_name = name
            best_score = score

    counts = {name: 0 for name in names}
    counts[best_name] = 1
    current = val_preds[best_name].astype(float).copy()
    total = 1
    for _ in range(max(0, rounds - 1)):
        candidate_best = best_name
        candidate_score = best_score
        for name in names:
            cand = (current * total + val_preds[name]) / float(total + 1)
            score = _metric_value(task_type, y_val, cand)
            if _is_better(task_type, score, candidate_score):
                candidate_best = name
                candidate_score = score
        counts[candidate_best] += 1
        current = (current * total + val_preds[candidate_best]) / float(total + 1)
        total += 1
        best_score = candidate_score
    weights = {name: count / float(total) for name, count in counts.items() if count > 0}
    return weights, best_score


def _weighted_average(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    out = None
    for name, weight in weights.items():
        arr = preds[name].astype(float)
        out = arr * weight if out is None else out + arr * weight
    if out is None:
        raise ValueError("Empty ensemble weights")
    return out


def run_task(task: str, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    df, spec, _ = load_dataset(task, args.data_dir, args.max_mols, seed=args.split_seed)
    df["split"] = assign_splits(df, args.split, seed=args.split_seed, task_type=spec.task_type)
    split_paths = _write_split_csvs(task, df, Path(args.out_dir))
    y = df["y"].to_numpy()
    split = df["split"].to_numpy()
    y_val = y[split == "val"]
    y_test = y[split == "test"]

    classical = _fit_member_predictions(task, spec.task_type, df, args)
    metric_rows: list[dict] = []
    selection_rows: list[dict] = []

    for seed in args.seeds:
        members = dict(classical.get(seed, {}))
        members.update(_collect_chemprop_predictions(task, seed, split_paths, args, Path(args.out_dir)))
        if not members:
            continue
        val_preds = {name: vt[0] for name, vt in members.items()}
        test_preds = {name: vt[1] for name, vt in members.items()}
        for name, pred in test_preds.items():
            metric_rows.append(_eval_row(task, spec.task_type, seed, name, y_test, pred))

        val_scores = {name: _metric_value(spec.task_type, y_val, pred) for name, pred in val_preds.items()}
        best_single = sorted(val_scores, key=lambda n: val_scores[n], reverse=higher_is_better(primary_metric(spec.task_type)))[0]
        best_single_pred = test_preds[best_single]
        metric_rows.append(_eval_row(task, spec.task_type, seed, "val_best_single", y_test, best_single_pred))
        selection_rows.append(
            {
                "task": task,
                "seed": seed,
                "selector": "val_best_single",
                "members": best_single,
                "weights_json": json.dumps({best_single: 1.0}, sort_keys=True),
                "val_primary": float(val_scores[best_single]),
                "n_members_considered": len(members),
            }
        )

        weights, val_primary = _ensemble_selection(spec.task_type, y_val, val_preds, args.ensemble_rounds)
        ensemble_pred = _weighted_average(test_preds, weights)
        if spec.task_type == "classification":
            ensemble_pred = np.clip(ensemble_pred, 1e-6, 1 - 1e-6)
        metric_rows.append(_eval_row(task, spec.task_type, seed, "val_greedy_ensemble", y_test, ensemble_pred))
        selection_rows.append(
            {
                "task": task,
                "seed": seed,
                "selector": "val_greedy_ensemble",
                "members": ";".join(sorted(weights)),
                "weights_json": json.dumps(weights, sort_keys=True),
                "val_primary": float(val_primary),
                "n_members_considered": len(members),
            }
        )

    return metric_rows, selection_rows


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "started_at": now_iso(),
                "argv": sys.argv,
                "python": sys.version,
                "platform": platform.platform(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "tasks": args.tasks,
                "seeds": args.seeds,
                "split": args.split,
                "split_seed": args.split_seed,
                "max_mols": args.max_mols,
                "feature2d_set": args.feature2d_set,
                "chemprop_root": args.chemprop_root,
                "protocol_note": "Validation-only single-model selection and greedy convex ensemble; fixed scaffold-balanced split; no test labels used for selection.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    invalid = [task for task in args.tasks if task not in DATASETS]
    if invalid:
        raise ValueError(f"Unknown tasks: {invalid}")

    metric_rows: list[dict] = []
    selection_rows: list[dict] = []
    for task in args.tasks:
        print(f"[validation-ensemble] {task}", flush=True)
        m_rows, s_rows = run_task(task, args)
        metric_rows.extend(m_rows)
        selection_rows.extend(s_rows)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "metrics_long.csv", index=False)
    _summarize(metrics).to_csv(out_dir / "method_summary.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(out_dir / "selection_by_seed.csv", index=False)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
