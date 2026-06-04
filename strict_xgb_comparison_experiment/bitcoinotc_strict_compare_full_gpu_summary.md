# BitcoinOTC Strict XGB Comparison Summary

Results directory: `results\strict_compare_full_gpu\BitcoinOTC-1`
Threshold mode: `val_F1_macro`

## Summary

| model | threshold | seeds | PR_AUC_negative | F1_negative | F1_macro | balanced_accuracy | AUROC | F1_weighted | train_wall_time_sec | eval_wall_time_sec | total_wall_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGB-all | val_F1_macro | 5 | 0.5472 +/- 0.0055 | 0.4900 +/- 0.0077 | 0.6970 +/- 0.0096 | 0.7168 +/- 0.0091 | 0.8235 +/- 0.0055 | 0.8455 +/- 0.0116 | 0.81 +/- 0.03 | 0.02 +/- 0.00 | 0.83 +/- 0.03 |
| XGB-dst_reputation | val_F1_macro | 5 | 0.4160 +/- 0.0069 | 0.4645 +/- 0.0041 | 0.6845 +/- 0.0030 | 0.6954 +/- 0.0025 | 0.7443 +/- 0.0043 | 0.8424 +/- 0.0026 | 0.47 +/- 0.02 | 0.02 +/- 0.00 | 0.49 +/- 0.02 |
| semba | val_F1_macro | 5 | 0.4096 +/- 0.0194 | 0.4101 +/- 0.0563 | 0.6524 +/- 0.0238 | 0.6665 +/- 0.0408 | 0.7505 +/- 0.0162 | 0.8262 +/- 0.0119 | 188.04 +/- 56.06 | 0.20 +/- 0.00 | 188.24 +/- 56.06 |
| tgn | val_F1_macro | 5 | 0.4087 +/- 0.0295 | 0.4036 +/- 0.0396 | 0.6606 +/- 0.0209 | 0.6433 +/- 0.0219 | 0.7257 +/- 0.0107 | 0.8449 +/- 0.0083 | 97.99 +/- 14.09 | 0.15 +/- 0.01 | 98.14 +/- 14.09 |
| XGB-no_dst_reputation | val_F1_macro | 5 | 0.3814 +/- 0.0057 | 0.3914 +/- 0.0255 | 0.6464 +/- 0.0089 | 0.6460 +/- 0.0231 | 0.7571 +/- 0.0051 | 0.8293 +/- 0.0046 | 0.62 +/- 0.03 | 0.02 +/- 0.00 | 0.64 +/- 0.02 |
| semba-noprop | val_F1_macro | 5 | 0.2649 +/- 0.0494 | 0.3495 +/- 0.0496 | 0.6009 +/- 0.0366 | 0.6308 +/- 0.0327 | 0.6879 +/- 0.0360 | 0.7812 +/- 0.0274 | 163.55 +/- 27.87 | 0.24 +/- 0.08 | 163.79 +/- 27.91 |

## Notes

- Main ranking uses `PR_AUC_negative` first, then negative-class F1.
- `F1_weighted` is reported only as a secondary class-imbalance-sensitive metric.
- Graph rows use predict-before-update no-leakage mini-blocks.