from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from xgboost.core import XGBoostError

from experiments.dynamic_features import FEATURE_COLUMNS
from experiments.paper_protocol_features import paper_protocol_feature_frames
from experiments.strict_metrics import choose_threshold, classification_metrics
from experiments.strict_temporal_protocol import FEATURE_SETS, split_summary
from utils import get_data


PAPER_TABLE6_REFERENCE = [
    {
        "dataset": "BitcoinOTC-1",
        "model": "TGN",
        "source": "paper Table 6 BTC-Otc",
        "F1_bin_mean": 0.74,
        "F1_bin_std": 0.02,
        "AUROC_mean": 0.82,
        "AUROC_std": 0.03,
    },
    {
        "dataset": "BitcoinOTC-1",
        "model": "SEMBA",
        "source": "paper Table 6 BTC-Otc",
        "F1_bin_mean": 0.81,
        "F1_bin_std": 0.04,
        "AUROC_mean": 0.79,
        "AUROC_std": 0.02,
    },
]


def _parse_ints(values: Iterable[int]) -> List[int]:
    return [int(value) for value in values]


def _feature_columns(feature_set: str) -> List[str]:
    if feature_set == "all":
        return FEATURE_COLUMNS
    return list(FEATURE_SETS[feature_set])


def _param_grid() -> List[Dict[str, float]]:
    return [
        {
            "n_estimators": n,
            "max_depth": d,
            "learning_rate": lr,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 2.0,
        }
        for n in [300, 600]
        for d in [3, 4, 5]
        for lr in [0.03, 0.05]
    ]


def _base_params(params: Dict[str, float], seed: int, n_jobs: int, device: str) -> Dict[str, object]:
    return {
        **params,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": seed,
        "n_jobs": n_jobs,
        "tree_method": "hist",
        "device": device,
    }


def _fit_xgb(
    x_train,
    y_train,
    x_val,
    y_val,
    seed: int,
    params: Dict[str, float],
    args,
) -> Tuple[XGBClassifier, float, str]:
    devices = ["cuda", "cpu"] if args.xgb_device == "auto" else [args.xgb_device]
    sample_weight = None
    if args.sample_weight == "balanced":
        sample_weight = compute_sample_weight("balanced", y_train)

    last_error = None
    for device in devices:
        model = XGBClassifier(**_base_params(params, seed, args.n_jobs, device))
        started = time.time()
        try:
            model.fit(
                x_train,
                y_train,
                sample_weight=sample_weight,
                eval_set=[(x_val, y_val)],
                verbose=False,
            )
            return model, time.time() - started, device
        except XGBoostError as exc:
            last_error = exc
            if device == "cpu":
                raise
    raise RuntimeError(f"XGBoost fit failed: {last_error}")


def _metrics_with_aliases(y_true, y_prob, threshold: float) -> Dict[str, float]:
    metrics = classification_metrics(y_true, y_prob, threshold)
    metrics["F1_bin"] = metrics["F1_positive"]
    metrics["TN"] = metrics["true_negative"]
    metrics["FP"] = metrics["false_positive"]
    metrics["FN"] = metrics["false_negative"]
    metrics["TP"] = metrics["true_positive"]
    return metrics


def _write_manifest(out_dir: Path, args, split_rows) -> None:
    manifest = {
        "runner": "run_xgb_paper_protocol",
        "task": "sign_class",
        "protocol": "paper_batch_inclusive_to_update_true_approximation",
        "leakage_note": (
            "Current batch is written into history before features are generated, "
            "matching the old SEMBA train.py to_update=True style."
        ),
        "dataset": args.dataset,
        "processed_dir": args.processed_dir,
        "max_events": args.max_events,
        "batch_size": args.batch_size,
        "seeds": args.seeds,
        "feature_sets": args.feature_sets,
        "xgb_device": args.xgb_device,
        "sample_weight": args.sample_weight,
        "split_summary": split_rows,
        "paper_reference": PAPER_TABLE6_REFERENCE,
    }
    (out_dir / "manifest_xgb_paper_protocol.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _grid_search(feature_set, train_df, val_df, args, out_dir: Path):
    cols = _feature_columns(feature_set)
    x_train = train_df[cols]
    y_train = train_df["label"].astype(int).to_numpy()
    x_val = val_df[cols]
    y_val = val_df["label"].astype(int).to_numpy()

    rows = []
    best_paper = None
    best_negative = None
    for index, params in enumerate(_param_grid()):
        model, train_sec, device = _fit_xgb(
            x_train,
            y_train,
            x_val,
            y_val,
            seed=args.grid_seed,
            params=params,
            args=args,
        )
        val_prob = model.predict_proba(x_val)[:, 1]
        fixed_metrics = _metrics_with_aliases(y_val, val_prob, 0.5)
        f1_bin_threshold, f1_bin_score = choose_threshold(y_val, val_prob, "F1_positive")
        f1_macro_threshold, f1_macro_score = choose_threshold(y_val, val_prob, "F1_macro")
        negative_threshold_metrics = _metrics_with_aliases(y_val, val_prob, f1_macro_threshold)

        row = {
            **params,
            "grid_index": index,
            "feature_set": feature_set,
            "seed": args.grid_seed,
            "device": device,
            "train_wall_time_sec": round(train_sec, 6),
            "fixed_0.5_F1_bin": fixed_metrics["F1_bin"],
            "fixed_0.5_AUROC": fixed_metrics["AUROC"],
            "fixed_0.5_F1_macro": fixed_metrics["F1_macro"],
            "fixed_0.5_PR_AUC_negative": fixed_metrics["PR_AUC_negative"],
            "val_F1_bin_threshold": f1_bin_threshold,
            "val_F1_bin_score": f1_bin_score,
            "val_F1_macro_threshold": f1_macro_threshold,
            "val_F1_macro_score": f1_macro_score,
            "val_F1_macro_PR_AUC_negative": negative_threshold_metrics["PR_AUC_negative"],
        }
        rows.append(row)

        paper_key = (row["fixed_0.5_F1_bin"], row["fixed_0.5_AUROC"])
        negative_key = (row["fixed_0.5_F1_macro"], row["fixed_0.5_PR_AUC_negative"])
        if best_paper is None or paper_key > best_paper[0]:
            best_paper = (paper_key, params)
        if best_negative is None or negative_key > best_negative[0]:
            best_negative = (negative_key, params)

    pd.DataFrame(rows).to_csv(out_dir / "xgb_grid_search.csv", index=False)
    return {
        "XGB-paper-selected": best_paper[1],
        "XGB-negative-aware": best_negative[1],
    }, rows


def _run_always_positive(y_test, args) -> List[Dict[str, object]]:
    rows = []
    for seed in args.seeds:
        metrics = _metrics_with_aliases(y_test, np.ones_like(y_test, dtype=float), 0.5)
        rows.append({
            **metrics,
            "dataset": args.dataset,
            "model": "always_positive",
            "feature_set": "none",
            "seed": seed,
            "threshold_mode": "fixed_0.5",
            "val_threshold_score": np.nan,
            "train_wall_time_sec": 0.0,
            "eval_wall_time_sec": 0.0,
            "total_wall_time_sec": 0.0,
            "device": "cpu",
            "num_parameters": 0,
        })
    return rows


def _run_config(model_name, params, feature_set, train_df, val_df, test_df, args, out_dir: Path):
    cols = _feature_columns(feature_set)
    x_train = train_df[cols]
    y_train = train_df["label"].astype(int).to_numpy()
    x_val = val_df[cols]
    y_val = val_df["label"].astype(int).to_numpy()
    x_test = test_df[cols]
    y_test = test_df["label"].astype(int).to_numpy()

    metric_rows = []
    prediction_rows = []
    for seed in args.seeds:
        model, train_sec, device = _fit_xgb(x_train, y_train, x_val, y_val, seed, params, args)
        eval_started = time.time()
        val_prob = model.predict_proba(x_val)[:, 1]
        test_prob = model.predict_proba(x_test)[:, 1]
        eval_sec = time.time() - eval_started

        val_f1_bin_threshold, val_f1_bin_score = choose_threshold(y_val, val_prob, "F1_positive")
        val_f1_macro_threshold, val_f1_macro_score = choose_threshold(y_val, val_prob, "F1_macro")
        threshold_modes = [
            ("fixed_0.5", 0.5, np.nan),
            ("val_F1_bin", val_f1_bin_threshold, val_f1_bin_score),
            ("val_F1_macro", val_f1_macro_threshold, val_f1_macro_score),
        ]

        for threshold_mode, threshold, val_score in threshold_modes:
            metrics = _metrics_with_aliases(y_test, test_prob, threshold)
            metric_rows.append({
                **metrics,
                "dataset": args.dataset,
                "model": model_name,
                "feature_set": feature_set,
                "seed": seed,
                "threshold_mode": threshold_mode,
                "val_threshold_score": val_score,
                "train_wall_time_sec": round(train_sec, 6),
                "eval_wall_time_sec": round(eval_sec, 6),
                "total_wall_time_sec": round(train_sec + eval_sec, 6),
                "device": device,
                "num_parameters": int(params["n_estimators"]),
                **{f"param_{key}": value for key, value in params.items()},
            })

        prediction_rows.extend({
            "dataset": args.dataset,
            "model": model_name,
            "feature_set": feature_set,
            "seed": seed,
            "row_id": idx,
            "y_true": int(y),
            "y_prob": float(prob),
            "threshold_fixed_0.5": 0.5,
            "threshold_val_F1_bin": float(val_f1_bin_threshold),
            "threshold_val_F1_macro": float(val_f1_macro_threshold),
            "y_pred_fixed_0.5": int(prob >= 0.5),
            "y_pred_val_F1_bin": int(prob >= val_f1_bin_threshold),
            "y_pred_val_F1_macro": int(prob >= val_f1_macro_threshold),
        } for idx, (y, prob) in enumerate(zip(y_test, test_prob)))

        pd.DataFrame({
            "feature": cols,
            "importance": model.feature_importances_,
            "seed": seed,
            "model": model_name,
            "feature_set": feature_set,
        }).sort_values("importance", ascending=False).to_csv(
            out_dir / "feature_importance" / f"xgb_{model_name}_{feature_set}_seed_{seed}.csv",
            index=False,
        )

    return metric_rows, prediction_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BitcoinOTC-1")
    parser.add_argument("--processed_dir", default="processed2")
    parser.add_argument("--max_events", type=int, default=None)
    parser.add_argument("--results_dir", default="paper_protocol_sign_class_experiment/results")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--grid_seed", type=int, default=42)
    parser.add_argument("--feature_sets", nargs="+", choices=FEATURE_SETS.keys(), default=["all"])
    parser.add_argument("--sample_weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--xgb_device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--n_jobs", type=int, default=4)
    args = parser.parse_args()
    args.seeds = _parse_ints(args.seeds)

    data, train_data, val_data, test_data = get_data(
        args.dataset,
        f"./data/{args.dataset}",
        "cpu",
        processed_dir=args.processed_dir,
        max_events=args.max_events,
    )

    out_dir = Path(args.results_dir) / args.dataset
    (out_dir / "feature_importance").mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = paper_protocol_feature_frames(
        data,
        train_data,
        val_data,
        test_data,
        batch_size=args.batch_size,
    )
    split_rows = split_summary(data, train_data, val_data, test_data)
    _write_manifest(out_dir, args, split_rows)

    y_test = test_df["label"].astype(int).to_numpy()
    metric_rows = _run_always_positive(y_test, args)
    prediction_rows = []
    grid_rows_all = []

    for feature_set in args.feature_sets:
        selected_params, grid_rows = _grid_search(feature_set, train_df, val_df, args, out_dir)
        grid_rows_all.extend(grid_rows)
        for model_name, params in selected_params.items():
            rows, preds = _run_config(
                model_name,
                params,
                feature_set,
                train_df,
                val_df,
                test_df,
                args,
                out_dir,
            )
            metric_rows.extend(rows)
            prediction_rows.extend(preds)

    pd.DataFrame(grid_rows_all).to_csv(out_dir / "xgb_grid_search.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out_dir / "per_run_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(out_dir / "per_run_predictions.csv", index=False)

    print(json.dumps({
        "runner": "run_xgb_paper_protocol",
        "out_dir": str(out_dir),
        "rows_written": len(metric_rows),
        "models": sorted(set(row["model"] for row in metric_rows)),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
