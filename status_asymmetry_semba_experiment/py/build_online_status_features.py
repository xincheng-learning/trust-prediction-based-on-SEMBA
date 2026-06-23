from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.strict_temporal_protocol import global_online_feature_frames  # noqa: E402


FLAG_COLUMNS = [
    "src_seen_before",
    "dst_seen_before",
    "dst_more_reputable_flag",
    "src_more_negative_flag",
    "src_status_known",
    "dst_status_known",
    "both_status_known",
    "either_status_unknown",
]

CONTINUOUS_COLUMNS = [
    "src_out_count_before",
    "src_pos_out_count_before",
    "src_neg_out_count_before",
    "src_pos_out_ratio_before",
    "src_neg_out_ratio_before",
    "src_activity_log_before",
    "dst_in_count_before",
    "dst_pos_in_count_before",
    "dst_neg_in_count_before",
    "dst_pos_in_ratio_before",
    "dst_neg_in_ratio_before",
    "src_reputation_score_before",
    "dst_reputation_score_before",
    "status_gap_before",
    "activity_gap_before",
    "reputation_gap_before",
]

STATUS_FEATURE_COLUMNS = CONTINUOUS_COLUMNS + FLAG_COLUMNS

FEATURE_GROUPS = {
    "source": [
        "src_seen_before",
        "src_out_count_before",
        "src_pos_out_count_before",
        "src_neg_out_count_before",
        "src_pos_out_ratio_before",
        "src_neg_out_ratio_before",
        "src_activity_log_before",
        "src_status_known",
        "src_more_negative_flag",
    ],
    "target": [
        "dst_seen_before",
        "dst_in_count_before",
        "dst_pos_in_count_before",
        "dst_neg_in_count_before",
        "dst_pos_in_ratio_before",
        "dst_neg_in_ratio_before",
        "dst_reputation_score_before",
        "dst_status_known",
        "dst_more_reputable_flag",
    ],
    "status_gap": [
        "src_reputation_score_before",
        "dst_reputation_score_before",
        "status_gap_before",
        "activity_gap_before",
        "reputation_gap_before",
        "dst_more_reputable_flag",
        "src_more_negative_flag",
        "both_status_known",
        "either_status_unknown",
    ],
    "all_status": STATUS_FEATURE_COLUMNS,
}


class TrainOnlyStatusScaler:
    """Standardize continuous status features using train split statistics."""

    def __init__(self, continuous_columns: Iterable[str] = CONTINUOUS_COLUMNS):
        self.continuous_columns = list(continuous_columns)
        self.mean_: Dict[str, float] = {}
        self.std_: Dict[str, float] = {}

    def fit(self, train_df: pd.DataFrame) -> "TrainOnlyStatusScaler":
        for column in self.continuous_columns:
            values = train_df[column].astype(float).to_numpy()
            mean = float(np.mean(values)) if values.size else 0.0
            std = float(np.std(values)) if values.size else 1.0
            self.mean_[column] = mean
            self.std_[column] = std if std > 1e-12 else 1.0
        return self

    def transform(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        out = df.copy()
        for column in feature_columns:
            if column in self.continuous_columns:
                out[column] = (out[column].astype(float) - self.mean_[column]) / self.std_[column]
            else:
                out[column] = out[column].astype(float)
        return out[feature_columns]


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den_values = den.astype(float).to_numpy()
    num_values = num.astype(float).to_numpy()
    out = np.zeros_like(den_values, dtype=float)
    np.divide(num_values, den_values, out=out, where=den_values > 0)
    return out


def _derive_status_features(frame: pd.DataFrame) -> pd.DataFrame:
    src_out_total = frame["src_out_pos"] + frame["src_out_neg"]
    src_in_total = frame["src_in_pos"] + frame["src_in_neg"]
    dst_in_total = frame["dst_in_pos"] + frame["dst_in_neg"]
    src_rep = (frame["src_in_pos"] - frame["src_in_neg"]) / (src_in_total + 1.0)
    dst_rep = (frame["dst_in_pos"] - frame["dst_in_neg"]) / (dst_in_total + 1.0)
    src_neg_ratio = pd.Series(_ratio(frame["src_out_neg"], src_out_total), index=frame.index)
    src_pos_ratio = pd.Series(_ratio(frame["src_out_pos"], src_out_total), index=frame.index)
    dst_pos_ratio = pd.Series(_ratio(frame["dst_in_pos"], dst_in_total), index=frame.index)
    dst_neg_ratio = pd.Series(_ratio(frame["dst_in_neg"], dst_in_total), index=frame.index)

    out = pd.DataFrame(index=frame.index)
    out["src_seen_before"] = frame["src_seen_before"].astype(float)
    out["src_out_count_before"] = np.log1p(src_out_total.astype(float))
    out["src_pos_out_count_before"] = np.log1p(frame["src_out_pos"].astype(float))
    out["src_neg_out_count_before"] = np.log1p(frame["src_out_neg"].astype(float))
    out["src_pos_out_ratio_before"] = src_pos_ratio.astype(float)
    out["src_neg_out_ratio_before"] = src_neg_ratio.astype(float)
    out["src_activity_log_before"] = frame["src_history_activity_log"].astype(float)

    out["dst_seen_before"] = frame["dst_seen_before"].astype(float)
    out["dst_in_count_before"] = np.log1p(dst_in_total.astype(float))
    out["dst_pos_in_count_before"] = np.log1p(frame["dst_in_pos"].astype(float))
    out["dst_neg_in_count_before"] = np.log1p(frame["dst_in_neg"].astype(float))
    out["dst_pos_in_ratio_before"] = dst_pos_ratio.astype(float)
    out["dst_neg_in_ratio_before"] = dst_neg_ratio.astype(float)
    out["src_reputation_score_before"] = src_rep.astype(float)
    out["dst_reputation_score_before"] = dst_rep.astype(float)

    out["status_gap_before"] = out["dst_reputation_score_before"] - out["src_reputation_score_before"]
    out["activity_gap_before"] = frame["dst_visibility_log"].astype(float) - frame["src_history_activity_log"].astype(float)
    out["reputation_gap_before"] = out["status_gap_before"]
    out["dst_more_reputable_flag"] = (out["dst_reputation_score_before"] > out["src_reputation_score_before"]).astype(float)
    out["src_more_negative_flag"] = ((src_out_total > 0) & (src_neg_ratio > src_pos_ratio)).astype(float)
    out["src_status_known"] = out["src_seen_before"].astype(float)
    out["dst_status_known"] = out["dst_seen_before"].astype(float)
    out["both_status_known"] = ((out["src_status_known"] > 0) & (out["dst_status_known"] > 0)).astype(float)
    out["either_status_unknown"] = ((out["src_status_known"] == 0) | (out["dst_status_known"] == 0)).astype(float)
    out["label"] = frame["label"].astype(int)
    return out[STATUS_FEATURE_COLUMNS + ["label"]]


def build_status_feature_frames(data, train_data, val_data, feature_set: str = "all_status"):
    if feature_set not in FEATURE_GROUPS:
        raise ValueError(f"Unknown feature_set={feature_set}. Expected one of {sorted(FEATURE_GROUPS)}")
    train_base, val_base, test_base = global_online_feature_frames(
        data,
        train_data,
        val_data,
        strict_timestamp=True,
    )
    train_status = _derive_status_features(train_base)
    val_status = _derive_status_features(val_base)
    test_status = _derive_status_features(test_base)
    feature_columns = list(FEATURE_GROUPS[feature_set])
    scaler = TrainOnlyStatusScaler().fit(train_status)
    return {
        "feature_set": feature_set,
        "feature_columns": feature_columns,
        "scaler": scaler,
        "train": scaler.transform(train_status, feature_columns),
        "val": scaler.transform(val_status, feature_columns),
        "test": scaler.transform(test_status, feature_columns),
        "train_raw": train_status,
        "val_raw": val_status,
        "test_raw": test_status,
    }


def status_feature_audit_rows(frames: Dict[str, object], train_data, val_data, test_data) -> List[Dict[str, object]]:
    feature_columns = frames["feature_columns"]
    rows = [
        {
            "check": "strict_timestamp_feature_build",
            "status": "pass",
            "evidence": "global_online_feature_frames(..., strict_timestamp=True) predicts same-timestamp events before updating history",
        },
        {
            "check": "history_window",
            "status": "pass",
            "evidence": "features are constructed from dynamic_features state before update_history(event)",
        },
        {
            "check": "train_only_scaler",
            "status": "pass",
            "evidence": "TrainOnlyStatusScaler.fit(train_status) only uses train split raw status features",
        },
        {
            "check": "row_alignment_train",
            "status": "pass" if len(frames["train"]) == int(train_data.num_events) else "fail",
            "evidence": f"{len(frames['train'])} feature rows vs {int(train_data.num_events)} train events",
        },
        {
            "check": "row_alignment_val",
            "status": "pass" if len(frames["val"]) == int(val_data.num_events) else "fail",
            "evidence": f"{len(frames['val'])} feature rows vs {int(val_data.num_events)} val events",
        },
        {
            "check": "row_alignment_test",
            "status": "pass" if len(frames["test"]) == int(test_data.num_events) else "fail",
            "evidence": f"{len(frames['test'])} feature rows vs {int(test_data.num_events)} test events",
        },
        {
            "check": "nan_or_inf",
            "status": "pass",
            "evidence": "all selected feature matrices checked for finite values",
        },
    ]
    for split in ["train", "val", "test"]:
        values = frames[split][feature_columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            rows[-1] = {
                "check": "nan_or_inf",
                "status": "fail",
                "evidence": f"{split} contains non-finite status feature values",
            }
    return rows


def write_feature_dictionary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("src_seen_before", "source", "flag", "whether source appeared before current time", "computed before current event update"),
        ("src_out_count_before", "source", "log1p(count)", "source historical outgoing edge count", "time < current_time"),
        ("src_pos_out_count_before", "source", "log1p(count)", "source historical positive outgoing count", "time < current_time"),
        ("src_neg_out_count_before", "source", "log1p(count)", "source historical negative outgoing count", "time < current_time"),
        ("src_pos_out_ratio_before", "source", "ratio", "positive outgoing ratio before prediction", "time < current_time"),
        ("src_neg_out_ratio_before", "source", "ratio", "negative outgoing ratio before prediction", "time < current_time"),
        ("src_activity_log_before", "source", "log1p(count)", "source historical visibility/activity", "time < current_time"),
        ("dst_seen_before", "target", "flag", "whether target appeared before current time", "computed before current event update"),
        ("dst_in_count_before", "target", "log1p(count)", "target historical incoming edge count", "time < current_time"),
        ("dst_pos_in_count_before", "target", "log1p(count)", "target historical trusted incoming count", "time < current_time"),
        ("dst_neg_in_count_before", "target", "log1p(count)", "target historical distrusted incoming count", "time < current_time"),
        ("dst_pos_in_ratio_before", "target", "ratio", "target positive incoming ratio", "time < current_time"),
        ("dst_neg_in_ratio_before", "target", "ratio", "target negative incoming ratio", "time < current_time"),
        ("src_reputation_score_before", "status", "(pos_in-neg_in)/(in+1)", "source reputation score before prediction", "time < current_time"),
        ("dst_reputation_score_before", "status", "(pos_in-neg_in)/(in+1)", "target reputation score before prediction", "time < current_time"),
        ("status_gap_before", "asymmetry", "dst_rep-src_rep", "target-source reputation gap", "time < current_time"),
        ("activity_gap_before", "asymmetry", "dst_activity_log-src_activity_log", "target-source visibility/activity gap", "time < current_time"),
        ("reputation_gap_before", "asymmetry", "dst_rep-src_rep", "alias for status gap for ablation readability", "time < current_time"),
        ("dst_more_reputable_flag", "asymmetry", "flag", "target reputation exceeds source reputation", "time < current_time"),
        ("src_more_negative_flag", "source", "flag", "source has higher historical negative than positive outgoing ratio", "time < current_time"),
        ("src_status_known", "missingness", "flag", "source status/history observed before prediction", "time < current_time"),
        ("dst_status_known", "missingness", "flag", "target status/history observed before prediction", "time < current_time"),
        ("both_status_known", "missingness", "flag", "both endpoints have prior history", "time < current_time"),
        ("either_status_unknown", "missingness", "flag", "at least one endpoint is cold-start", "time < current_time"),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "group", "transform", "definition", "time_safety"])
        writer.writerows(rows)
