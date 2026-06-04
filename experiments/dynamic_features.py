from __future__ import annotations

from collections import defaultdict
from math import log1p
from typing import Dict, Iterable, List, MutableMapping, Set


Event = Dict[str, int]
FeatureRow = Dict[str, float]


FEATURE_COLUMNS = [
    "src_out_pos",
    "src_out_neg",
    "src_in_pos",
    "src_in_neg",
    "dst_out_pos",
    "dst_out_neg",
    "dst_in_pos",
    "dst_in_neg",
    "src_pos_ratio",
    "dst_in_pos_ratio",
    "q_src",
    "q_dst",
    "status_gap",
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
    "dst_visibility_log",
    "dst_trusted_in_count",
    "dst_trusted_ratio",
    "dst_high_visibility",
    "dst_visibility_share_of_max",
]


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def build_sign_class_rows(
    events: Iterable[Event],
    high_visibility_min_events: int = 10,
) -> List[FeatureRow]:
    out_pos = defaultdict(int)
    out_neg = defaultdict(int)
    in_pos = defaultdict(int)
    in_neg = defaultdict(int)
    pair_pos = defaultdict(int)
    pair_neg = defaultdict(int)
    neighbors: MutableMapping[int, Set[int]] = defaultdict(set)
    last_time = {}
    seen_nodes = set()
    node_visibility = defaultdict(int)
    max_visibility_seen = 0
    rows: List[FeatureRow] = []

    sorted_events = sorted(events, key=lambda row: row["t"])

    for event in sorted_events:
        src = int(event["src"])
        dst = int(event["dst"])
        t = int(event["t"])
        label = int(event["y"])

        src_total_out = out_pos[src] + out_neg[src]
        dst_total_in = in_pos[dst] + in_neg[dst]
        src_history_activity = node_visibility[src]
        dst_visibility = node_visibility[dst]
        q_src = log1p(in_pos[src]) - log1p(in_neg[src])
        q_dst = log1p(in_pos[dst]) - log1p(in_neg[dst])

        n_src = neighbors[src]
        n_dst = neighbors[dst]
        common = len(n_src.intersection(n_dst))

        src_last = last_time.get(src)
        dst_last = last_time.get(dst)

        rows.append({
            "src_out_pos": float(out_pos[src]),
            "src_out_neg": float(out_neg[src]),
            "src_in_pos": float(in_pos[src]),
            "src_in_neg": float(in_neg[src]),
            "dst_out_pos": float(out_pos[dst]),
            "dst_out_neg": float(out_neg[dst]),
            "dst_in_pos": float(in_pos[dst]),
            "dst_in_neg": float(in_neg[dst]),
            "src_pos_ratio": _ratio(out_pos[src], src_total_out),
            "dst_in_pos_ratio": _ratio(in_pos[dst], dst_total_in),
            "q_src": float(q_src),
            "q_dst": float(q_dst),
            "status_gap": float(q_dst - q_src),
            "coverage_uv": _ratio(common, len(n_src)),
            "coverage_vu": _ratio(common, len(n_dst)),
            "pair_pos_count": float(pair_pos[(src, dst)]),
            "pair_neg_count": float(pair_neg[(src, dst)]),
            "rev_pair_pos_count": float(pair_pos[(dst, src)]),
            "rev_pair_neg_count": float(pair_neg[(dst, src)]),
            "src_recency_log": float(log1p(t - src_last)) if src_last is not None else 0.0,
            "dst_recency_log": float(log1p(t - dst_last)) if dst_last is not None else 0.0,
            "src_seen_before": float(src in seen_nodes),
            "dst_seen_before": float(dst in seen_nodes),
            "src_history_activity_log": float(log1p(src_history_activity)),
            "dst_visibility_log": float(log1p(dst_visibility)),
            "dst_trusted_in_count": float(in_pos[dst]),
            "dst_trusted_ratio": _ratio(in_pos[dst], dst_total_in),
            "dst_high_visibility": float(dst_visibility >= high_visibility_min_events),
            "dst_visibility_share_of_max": _ratio(dst_visibility, max_visibility_seen),
            "label": float(label),
        })

        if label == 1:
            out_pos[src] += 1
            in_pos[dst] += 1
            pair_pos[(src, dst)] += 1
        else:
            out_neg[src] += 1
            in_neg[dst] += 1
            pair_neg[(src, dst)] += 1

        neighbors[src].add(dst)
        neighbors[dst].add(src)
        last_time[src] = t
        last_time[dst] = t
        seen_nodes.add(src)
        seen_nodes.add(dst)
        node_visibility[src] += 1
        node_visibility[dst] += 1
        max_visibility_seen = max(max_visibility_seen, node_visibility[src], node_visibility[dst])

    return rows


def events_from_temporal_data(data) -> List[Event]:
    return [
        {"src": int(src), "dst": int(dst), "t": int(t), "y": int(y)}
        for src, dst, t, y in zip(
            data.src.tolist(),
            data.dst.tolist(),
            data.t.tolist(),
            data.y.tolist(),
        )
    ]
