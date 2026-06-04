from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from experiments.dynamic_features import FEATURE_COLUMNS
from experiments.strict_metrics import choose_threshold, classification_metrics
from experiments.strict_temporal_protocol import (
    FEATURE_SETS,
    global_online_feature_frames,
    split_summary,
)
from utils import get_data


def _parse_ints(values):
    return [int(value) for value in values]


def _feature_columns(feature_set: str):
    if feature_set == "all":
        return FEATURE_COLUMNS
    return FEATURE_SETS[feature_set]


def _write_manifest(out_dir: Path, args, split_rows):
    manifest = {
        "runner": "run_xgb_strict",
        "dataset": args.dataset,
        "processed_dir": args.processed_dir,
        "max_events": args.max_events,
        "history_scope": "global_online",
        "strict_timestamp": True,
        "seeds": args.seeds,
        "feature_sets": args.feature_sets,
        "threshold_metric": args.threshold_metric,
        "split_summary": split_rows,
        "xgboost_note": (
            "xgboost 3.0.2 sklearn wrapper in this environment does not expose "
            "early_stopping_rounds; validation is used for threshold selection."
        ),
    }
    (out_dir / "manifest_xgb.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _append_csv(path: Path, rows):
    df = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path)
        pd.concat([existing, df], ignore_index=True, sort=False).to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def _run_always_positive(y_test, args, out_dir: Path):
    rows = []
    for seed in args.seeds:
        metrics = classification_metrics(y_test, np.ones_like(y_test, dtype=float), threshold=0.5)
        rows.append({
            **metrics,
            "dataset": args.dataset,
            "model": "always_positive",
            "feature_set": "none",
            "seed": seed,
            "threshold_mode": "fixed_0.5",
            "train_wall_time_sec": 0.0,
            "eval_wall_time_sec": 0.0,
            "total_wall_time_sec": 0.0,
            "device": "cpu",
            "num_parameters": 0,
        })
    _append_csv(out_dir / "per_run_metrics.csv", rows)
    return rows


def _fit_xgb(x_train, y_train, x_val, y_val, seed: int, args):
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=args.n_jobs,
    )
    sample_weight = None
    if args.sample_weight == "balanced":
        sample_weight = compute_sample_weight("balanced", y_train)

    started = time.time()
    model.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )
    return model, time.time() - started


def _run_feature_set(feature_set, train_df, val_df, test_df, args, out_dir: Path):
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
        model, train_sec = _fit_xgb(x_train, y_train, x_val, y_val, seed, args)
        eval_started = time.time()
        val_prob = model.predict_proba(x_val)[:, 1]
        test_prob = model.predict_proba(x_test)[:, 1]
        eval_sec = time.time() - eval_started
        val_threshold, val_score = choose_threshold(y_val, val_prob, args.threshold_metric)

        for threshold_mode, threshold in [
            ("val_F1_macro", val_threshold),
            ("fixed_0.5", 0.5),
        ]:
            metrics = classification_metrics(y_test, test_prob, threshold=threshold)
            metric_rows.append({
                **metrics,
                "dataset": args.dataset,
                "model": f"XGB-{feature_set}",
                "feature_set": feature_set,
                "seed": seed,
                "threshold_mode": threshold_mode,
                "val_threshold_score": val_score,
                "train_wall_time_sec": round(train_sec, 6),
                "eval_wall_time_sec": round(eval_sec, 6),
                "total_wall_time_sec": round(train_sec + eval_sec, 6),
                "device": "cpu",
                "num_parameters": int(args.n_estimators),
            })

        prediction_rows.extend({
            "dataset": args.dataset,
            "model": f"XGB-{feature_set}",
            "feature_set": feature_set,
            "seed": seed,
            "row_id": idx,
            "y_true": int(y),
            "y_prob": float(prob),
            "threshold": float(val_threshold),
            "y_pred": int(prob >= val_threshold),
        } for idx, (y, prob) in enumerate(zip(y_test, test_prob)))

        pd.DataFrame({
            "feature": cols,
            "importance": model.feature_importances_,
            "seed": seed,
            "model": f"XGB-{feature_set}",
        }).sort_values("importance", ascending=False).to_csv(
            out_dir / "feature_importance" / f"xgb_{feature_set}_seed_{seed}.csv",
            index=False,
        )

    _append_csv(out_dir / "per_run_metrics.csv", metric_rows)
    _append_csv(out_dir / "per_run_predictions.csv", prediction_rows)
    return metric_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BitcoinOTC-1")
    parser.add_argument("--processed_dir", default="processed2")
    parser.add_argument("--max_events", type=int, default=None)
    parser.add_argument("--results_dir", default="results/strict_compare_pilot")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--feature_sets", nargs="+", choices=FEATURE_SETS.keys(), default=["all"])
    parser.add_argument("--threshold_metric", default="F1_macro")
    parser.add_argument("--sample_weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample_bytree", type=float, default=0.9)
    parser.add_argument("--reg_lambda", type=float, default=2.0)
    parser.add_argument("--n_jobs", type=int, default=4)
    args = parser.parse_args()

    data, train_data, val_data, test_data = get_data(
        args.dataset,
        f"./data/{args.dataset}",
        "cpu",
        processed_dir=args.processed_dir,
        max_events=args.max_events,
    )

    out_dir = Path(args.results_dir) / args.dataset
    (out_dir / "feature_importance").mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = global_online_feature_frames(data, train_data, val_data, strict_timestamp=True)
    split_rows = split_summary(data, train_data, val_data, test_data)
    _write_manifest(out_dir, args, split_rows)

    y_test = test_df["label"].astype(int).to_numpy()
    rows = _run_always_positive(y_test, args, out_dir)
    for feature_set in args.feature_sets:
        rows.extend(_run_feature_set(feature_set, train_df, val_df, test_df, args, out_dir))

    print(json.dumps({
        "runner": "run_xgb_strict",
        "out_dir": str(out_dir),
        "rows_written": len(rows),
        "models": sorted(set(row["model"] for row in rows)),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
