# BitcoinOTC Strict XGB Comparison Summary

Results directory: `results\strict_compare_pilot_20000\BitcoinOTC-1`
Threshold mode: `val_F1_macro`

## Summary

| model | threshold | seeds | PR_AUC_negative | F1_negative | F1_macro | balanced_accuracy | AUROC | F1_weighted | train_wall_time_sec | eval_wall_time_sec | total_wall_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGB-all | val_F1_macro | 3 | 0.5639 +/- 0.0055 | 0.5754 +/- 0.0075 | 0.7729 +/- 0.0036 | 0.7459 +/- 0.0118 | 0.8576 +/- 0.0015 | 0.9416 +/- 0.0009 | 0.43 +/- 0.06 | 0.01 +/- 0.00 | 0.45 +/- 0.06 |
| XGB-dst_reputation | val_F1_macro | 3 | 0.4181 +/- 0.0032 | 0.4780 +/- 0.0095 | 0.7216 +/- 0.0060 | 0.6922 +/- 0.0058 | 0.7542 +/- 0.0025 | 0.9297 +/- 0.0031 | 0.27 +/- 0.01 | 0.01 +/- 0.00 | 0.27 +/- 0.01 |
| XGB-no_dst_reputation | val_F1_macro | 3 | 0.3893 +/- 0.0046 | 0.4122 +/- 0.0126 | 0.6800 +/- 0.0138 | 0.6975 +/- 0.0306 | 0.8187 +/- 0.0026 | 0.9087 +/- 0.0150 | 0.31 +/- 0.00 | 0.01 +/- 0.00 | 0.32 +/- 0.00 |
| semba | val_F1_macro | 3 | 0.1709 +/- 0.0537 | 0.2292 +/- 0.0411 | 0.5645 +/- 0.0211 | 0.6112 +/- 0.0377 | 0.6644 +/- 0.0325 | 0.8507 +/- 0.0079 | 68.80 +/- 0.24 | 0.21 +/- 0.00 | 69.01 +/- 0.24 |
| semba-noprop | val_F1_macro | 3 | 0.1132 +/- 0.0090 | 0.1624 +/- 0.0300 | 0.5287 +/- 0.0164 | 0.5530 +/- 0.0253 | 0.6409 +/- 0.0064 | 0.8415 +/- 0.0059 | 63.12 +/- 1.80 | 0.22 +/- 0.09 | 63.34 +/- 1.85 |
| tgn | val_F1_macro | 3 | 0.1087 +/- 0.0365 | 0.1735 +/- 0.0131 | 0.5078 +/- 0.0063 | 0.5734 +/- 0.0200 | 0.5785 +/- 0.0643 | 0.7933 +/- 0.0226 | 41.24 +/- 7.74 | 0.20 +/- 0.08 | 41.44 +/- 7.78 |

## Notes

- Main ranking uses `PR_AUC_negative` first, then negative-class F1.
- `F1_weighted` is reported only as a secondary class-imbalance-sensitive metric.
- Graph rows use predict-before-update no-leakage mini-blocks.