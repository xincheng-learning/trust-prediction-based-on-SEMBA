from __future__ import annotations

from typing import Iterable, Iterator, List, Optional, Tuple

import pandas as pd

from experiments.dynamic_features import build_sign_class_rows, events_from_temporal_data


FEATURE_SETS = {
    "all": None,
    "dst_reputation": [
        "dst_in_pos",
        "dst_in_neg",
        "dst_in_pos_ratio",
        "q_dst",
        "dst_trusted_in_count",
        "dst_trusted_ratio",
        "dst_visibility_log",
        "dst_high_visibility",
        "dst_visibility_share_of_max",
    ],
    "no_dst_reputation": [
        "src_out_pos",
        "src_out_neg",
        "src_in_pos",
        "src_in_neg",
        "dst_out_pos",
        "dst_out_neg",
        "src_pos_ratio",
        "coverage_uv",
        "coverage_vu",
        "pair_pos_count",
        "pair_neg_count",
        "rev_pair_pos_count",
        "rev_pair_neg_count",
        "src_recency_log",
        "dst_recency_log",
        "src_seen_before",
        "dst_seen_before",
        "src_history_activity_log",
    ],
}


def global_online_feature_frames(
    data,
    train_data,
    val_data,
    strict_timestamp: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = build_sign_class_rows(
        events_from_temporal_data(data),
        strict_timestamp=strict_timestamp,
    )
    all_df = pd.DataFrame(rows)
    train_end = int(train_data.num_events)
    val_end = train_end + int(val_data.num_events)
    return (
        all_df.iloc[:train_end].reset_index(drop=True),
        all_df.iloc[train_end:val_end].reset_index(drop=True),
        all_df.iloc[val_end:].reset_index(drop=True),
    )


def _timestamp_group_slices(times: List[int]) -> Iterator[Tuple[int, int]]:
    start = 0
    while start < len(times):
        t = times[start]
        end = start + 1
        while end < len(times) and times[end] == t:
            end += 1
        yield start, end
        start = end


def iter_time_blocks(data, max_events_per_block: Optional[int] = None):
    """Yield chronological blocks without splitting identical timestamps.

    If `max_events_per_block` is None or <= 0, this yields exact timestamp
    blocks. If it is positive, multiple timestamp groups can be combined into a
    larger no-leakage mini-batch. The larger batch is conservative because
    later timestamps in the same batch do not see earlier timestamps from that
    batch.
    """
    times = [int(t) for t in data.t.tolist()]
    if max_events_per_block is None or max_events_per_block <= 0:
        for start, end in _timestamp_group_slices(times):
            yield data[start:end]
        return

    block_start = 0
    block_end = 0
    for start, end in _timestamp_group_slices(times):
        if block_end > block_start and end - block_start > max_events_per_block:
            yield data[block_start:block_end]
            block_start = start
        block_end = end
    if block_end > block_start:
        yield data[block_start:block_end]


def split_summary(data, train_data, val_data, test_data):
    rows = []
    for name, split in [
        ("full", data),
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ]:
        labels = split.y.cpu()
        events = int(labels.numel())
        positives = int(labels.sum().item())
        negatives = events - positives
        rows.append({
            "split": name,
            "events": events,
            "positive": positives,
            "negative": negatives,
            "positive_rate": positives / events if events else 0.0,
        })
    return rows
