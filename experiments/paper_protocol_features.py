from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import log1p
from typing import Dict, Iterable, List, MutableMapping, Optional, Set, Tuple

import pandas as pd

from experiments.dynamic_features import FEATURE_COLUMNS, events_from_temporal_data


Event = Dict[str, int]
FeatureRow = Dict[str, float]


@dataclass
class PaperHistoryState:
    out_pos: MutableMapping[int, int] = field(default_factory=lambda: defaultdict(int))
    out_neg: MutableMapping[int, int] = field(default_factory=lambda: defaultdict(int))
    in_pos: MutableMapping[int, int] = field(default_factory=lambda: defaultdict(int))
    in_neg: MutableMapping[int, int] = field(default_factory=lambda: defaultdict(int))
    pair_pos: MutableMapping[Tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))
    pair_neg: MutableMapping[Tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))
    neighbors: MutableMapping[int, Set[int]] = field(default_factory=lambda: defaultdict(set))
    last_time: Dict[int, int] = field(default_factory=dict)
    seen_nodes: Set[int] = field(default_factory=set)
    node_visibility: MutableMapping[int, int] = field(default_factory=lambda: defaultdict(int))
    max_visibility_seen: int = 0


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def update_history(state: PaperHistoryState, event: Event) -> None:
    src = int(event["src"])
    dst = int(event["dst"])
    t = int(event["t"])
    label = int(event["y"])

    if label == 1:
        state.out_pos[src] += 1
        state.in_pos[dst] += 1
        state.pair_pos[(src, dst)] += 1
    else:
        state.out_neg[src] += 1
        state.in_neg[dst] += 1
        state.pair_neg[(src, dst)] += 1

    state.neighbors[src].add(dst)
    state.neighbors[dst].add(src)
    state.last_time[src] = t
    state.last_time[dst] = t
    state.seen_nodes.add(src)
    state.seen_nodes.add(dst)
    state.node_visibility[src] += 1
    state.node_visibility[dst] += 1
    state.max_visibility_seen = max(
        state.max_visibility_seen,
        state.node_visibility[src],
        state.node_visibility[dst],
    )


def make_feature_row(
    state: PaperHistoryState,
    event: Event,
    high_visibility_min_events: int = 10,
) -> FeatureRow:
    src = int(event["src"])
    dst = int(event["dst"])
    t = int(event["t"])
    label = int(event["y"])

    src_total_out = state.out_pos[src] + state.out_neg[src]
    dst_total_in = state.in_pos[dst] + state.in_neg[dst]
    src_history_activity = state.node_visibility[src]
    dst_visibility = state.node_visibility[dst]
    q_src = log1p(state.in_pos[src]) - log1p(state.in_neg[src])
    q_dst = log1p(state.in_pos[dst]) - log1p(state.in_neg[dst])

    n_src = state.neighbors[src]
    n_dst = state.neighbors[dst]
    common = len(n_src.intersection(n_dst))

    src_last = state.last_time.get(src)
    dst_last = state.last_time.get(dst)
    src_gap = max(0, t - src_last) if src_last is not None else None
    dst_gap = max(0, t - dst_last) if dst_last is not None else None

    row = {
        "src_out_pos": float(state.out_pos[src]),
        "src_out_neg": float(state.out_neg[src]),
        "src_in_pos": float(state.in_pos[src]),
        "src_in_neg": float(state.in_neg[src]),
        "dst_out_pos": float(state.out_pos[dst]),
        "dst_out_neg": float(state.out_neg[dst]),
        "dst_in_pos": float(state.in_pos[dst]),
        "dst_in_neg": float(state.in_neg[dst]),
        "src_pos_ratio": _ratio(state.out_pos[src], src_total_out),
        "dst_in_pos_ratio": _ratio(state.in_pos[dst], dst_total_in),
        "q_src": float(q_src),
        "q_dst": float(q_dst),
        "status_gap": float(q_dst - q_src),
        "coverage_uv": _ratio(common, len(n_src)),
        "coverage_vu": _ratio(common, len(n_dst)),
        "pair_pos_count": float(state.pair_pos[(src, dst)]),
        "pair_neg_count": float(state.pair_neg[(src, dst)]),
        "rev_pair_pos_count": float(state.pair_pos[(dst, src)]),
        "rev_pair_neg_count": float(state.pair_neg[(dst, src)]),
        # Batch-inclusive history can contain a later event from the same batch,
        # so recency gaps are clipped at zero.
        "src_recency_log": float(log1p(src_gap)) if src_gap is not None else 0.0,
        "dst_recency_log": float(log1p(dst_gap)) if dst_gap is not None else 0.0,
        "src_seen_before": float(src in state.seen_nodes),
        "dst_seen_before": float(dst in state.seen_nodes),
        "src_history_activity_log": float(log1p(src_history_activity)),
        "dst_visibility_log": float(log1p(dst_visibility)),
        "dst_trusted_in_count": float(state.in_pos[dst]),
        "dst_trusted_ratio": _ratio(state.in_pos[dst], dst_total_in),
        "dst_high_visibility": float(dst_visibility >= high_visibility_min_events),
        "dst_visibility_share_of_max": _ratio(dst_visibility, state.max_visibility_seen),
        "label": float(label),
    }
    return row


def build_rows_for_split_paper_batch(
    events: Iterable[Event],
    state: Optional[PaperHistoryState],
    batch_size: int,
    high_visibility_min_events: int = 10,
) -> Tuple[List[FeatureRow], PaperHistoryState]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if state is None:
        state = PaperHistoryState()

    rows: List[FeatureRow] = []
    sorted_events = sorted(events, key=lambda row: int(row["t"]))

    for start in range(0, len(sorted_events), batch_size):
        batch = sorted_events[start:start + batch_size]
        for event in batch:
            update_history(state, event)
        for event in batch:
            rows.append(make_feature_row(state, event, high_visibility_min_events))

    return rows, state


def _frame(rows: List[FeatureRow]) -> pd.DataFrame:
    columns = FEATURE_COLUMNS + ["label"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]


def paper_protocol_feature_frames(data, train_data, val_data, test_data, batch_size: int):
    _ = data
    state = PaperHistoryState()
    train_rows, state = build_rows_for_split_paper_batch(
        events_from_temporal_data(train_data),
        state,
        batch_size=batch_size,
    )
    val_rows, state = build_rows_for_split_paper_batch(
        events_from_temporal_data(val_data),
        state,
        batch_size=batch_size,
    )
    test_rows, state = build_rows_for_split_paper_batch(
        events_from_temporal_data(test_data),
        state,
        batch_size=batch_size,
    )
    return _frame(train_rows), _frame(val_rows), _frame(test_rows)
