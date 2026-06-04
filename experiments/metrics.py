from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def binary_classification_metrics(
    y_true: Iterable[int],
    y_prob: Iterable[float],
) -> Dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=int)
    y_prob = np.asarray(list(y_prob), dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    result = {
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
        "F1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "negative_class_F1": float(f1_score(1 - y_true, 1 - y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    if len(np.unique(y_true)) == 2:
        result["AUROC"] = float(roc_auc_score(y_true, y_prob))
        result["PR_AUC_negative"] = float(average_precision_score(1 - y_true, 1 - y_prob))
    else:
        result["AUROC"] = float("nan")
        result["PR_AUC_negative"] = float("nan")

    return result
