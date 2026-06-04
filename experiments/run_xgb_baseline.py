from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from experiments.dynamic_features import (
    FEATURE_COLUMNS,
    build_sign_class_rows,
    events_from_temporal_data,
)
from experiments.metrics import binary_classification_metrics
from utils import get_data


def _rows_from_split(data):
    return build_sign_class_rows(events_from_temporal_data(data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BitcoinOTC-1")
    parser.add_argument("--processed_dir", default="processed2")
    parser.add_argument("--max_events", type=int, default=5000)
    parser.add_argument("--results_dir", default="results/xgb")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.time()
    data, train_data, val_data, test_data = get_data(
        args.dataset,
        f"./data/{args.dataset}",
        "cpu",
        processed_dir=args.processed_dir,
        max_events=args.max_events,
    )

    train_df = pd.DataFrame(_rows_from_split(train_data))
    val_df = pd.DataFrame(_rows_from_split(val_data))
    test_df = pd.DataFrame(_rows_from_split(test_data))
    fit_df = pd.concat([train_df, val_df], ignore_index=True)

    x_train = fit_df[FEATURE_COLUMNS]
    y_train = fit_df["label"].astype(int)
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"].astype(int)

    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    scale_pos_weight = max(1.0, negative / max(1, positive))

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=args.seed,
        n_jobs=4,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(x_train, y_train)

    y_prob = model.predict_proba(x_test)[:, 1]
    metrics = binary_classification_metrics(y_test, y_prob)
    metrics.update({
        "dataset": args.dataset,
        "model": "xgb_history_features",
        "task": "sign_class",
        "processed_dir": args.processed_dir,
        "max_events": args.max_events,
        "train_events": int(train_data.num_events),
        "val_events": int(val_data.num_events),
        "test_events": int(test_data.num_events),
        "num_nodes": int(data.num_nodes),
        "positive_train": positive,
        "negative_train": negative,
        "wall_time_sec": round(time.time() - started, 4),
    })

    out_dir = Path(args.results_dir) / args.dataset / f"max_events_{args.max_events}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).to_csv(out_dir / "feature_importance.csv", index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

