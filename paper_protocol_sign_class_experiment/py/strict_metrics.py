from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=int)
    y_prob = np.asarray(list(y_prob), dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result = {
        "threshold": float(threshold),
        "num_examples": int(y_true.size),
        "positive_count": int((y_true == 1).sum()),
        "negative_count": int((y_true == 0).sum()),
        "positive_rate": float((y_true == 1).mean()) if y_true.size else 0.0,
        "predicted_positive_count": int((y_pred == 1).sum()),
        "predicted_negative_count": int((y_pred == 0).sum()),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "F1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1_positive": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "F1_negative": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    if len(np.unique(y_true)) == 2:
        result["AUROC"] = float(roc_auc_score(y_true, y_prob))
        result["PR_AUC_positive"] = float(average_precision_score(y_true, y_prob))
        result["PR_AUC_negative"] = float(average_precision_score(1 - y_true, 1 - y_prob))
    else:
        result["AUROC"] = float("nan")
        result["PR_AUC_positive"] = float("nan")
        result["PR_AUC_negative"] = float("nan")

    return result


def choose_threshold(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    metric: str = "F1_macro",
) -> Tuple[float, float]:
    y_true = np.asarray(list(y_true), dtype=int)
    y_prob = np.asarray(list(y_prob), dtype=float)
    thresholds = np.unique(np.r_[0.0, np.quantile(y_prob, np.linspace(0, 1, 201)), 1.0])

    best_threshold = 0.5
    best_value = -1.0
    for threshold in thresholds:
        metrics = classification_metrics(y_true, y_prob, float(threshold))
        value = metrics[metric]
        if value > best_value:
            best_threshold = float(threshold)
            best_value = float(value)
    return best_threshold, best_value


def mean_std_summary(df, group_cols):
    metric_cols = [
        "PR_AUC_negative",
        "F1_negative",
        "F1_macro",
        "balanced_accuracy",
        "AUROC",
        "F1_weighted",
        "train_wall_time_sec",
        "eval_wall_time_sec",
        "total_wall_time_sec",
    ]
    existing = [col for col in metric_cols if col in df.columns]
    grouped = df.groupby(group_cols, dropna=False)[existing].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()
