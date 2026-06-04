# Phase 1 Smoke Results

Date: 2026-06-04

Branch/worktree: `exp/xgb-history`

Data cache: shared local `data/<dataset>/processed2/`

Dataset: `BitcoinOTC-1`

Task: `sign_class`

Subset: first `max_events=5000` temporal events after `processed2` sorting/splitting

## Data Split

```text
events=5000
nodes=7165
train=3500
val=751
test=749
positive fraction=0.9840
```

This subset is extremely imbalanced. XGB trained on train+val sees:

```text
positive_train=4209
negative_train=42
```

Interpret these runs as engineering smoke tests, not final research results.

## Quick Results

| Model | Epochs | Wall Time | Test AUC | Test F1_weighted | Test F1_binary / negative-F1 | Notes |
|---|---:|---:|---:|---:|---:|---|
| XGB history features | n/a | 0.402s | 0.8338 | 0.9721 | negative-F1 0.6984 | Uses online historical features only. |
| SEMBA | 1 | 1.305s | 0.7879 | 0.6024 | F1_bin 0.6262 | Original-code-compatible smoke. |
| TGN | 1 | 0.666s | 0.8071 | 0.5569 | F1_bin 0.5791 | Original-code-compatible smoke. |

## XGB Top Features

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `dst_trusted_ratio` | 0.2885 |
| 2 | `dst_in_neg` | 0.2563 |
| 3 | `q_dst` | 0.1545 |
| 4 | `dst_in_pos_ratio` | 0.1390 |
| 5 | `src_out_neg` | 0.0281 |
| 6 | `dst_visibility_share_of_max` | 0.0204 |
| 7 | `dst_visibility_log` | 0.0199 |
| 8 | `status_gap` | 0.0157 |
| 9 | `dst_recency_log` | 0.0152 |
| 10 | `src_out_pos` | 0.0147 |

## Early Interpretation

The first smoke run supports the user's hypothesis: online historical node and pair features can be very strong for dynamic trust sign prediction.

The strongest XGB signals are target-side reputation and status:

```text
target trusted ratio
target negative in-degree
target status q_dst
target visibility
```

Next experiments should use larger event windows and stricter protocols before making research claims.

## Next Runs

Recommended next matrix:

```text
BitcoinOTC-1 max_events=10000, 20000
BitcoinAlpha-1 max_events=5000, 10000
Models: XGB, SEMBA, TGN, SEMBA-noprop
Metrics: F1_macro, negative-F1, PR-AUC-negative, wall time
```

