# BitcoinOTC-1 SEMBA-status-decoder Smoke Report

## 运行范围

- 数据集：BitcoinOTC-1
- seed：42
- 协议：strict predict-before-update
- 阶段：smoke / minimum viable experiment
- 事件数：12,000
- 切分：train 8,400 / val 1,800 / test 1,800
- 结果目录：`status_asymmetry_semba_experiment/results/BitcoinOTC-1/smoke_seed42`

## 时间安全审计

审计结论：`pass`

审计证据：

- `global_online_feature_frames(..., strict_timestamp=True)` 用于构造 status/asymmetry 特征。
- 同一时间戳事件先统一构造特征，再统一更新历史。
- 当前事件不参与自身特征。
- scaler 只在 train split 上 fit，valid/test 只 transform。
- smoke 范围内 feature rows 与 train/val/test event rows 完全对齐。

## 主要指标（val_F1_macro threshold）

| model | feature_set | F1_negative | PR_AUC_negative | F1_macro | AUROC | F1_weighted |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SEMBA | graph_memory | 0.0986 | 0.0696 | 0.5308 | 0.6373 | 0.9318 |
| SEMBA-status-decoder | all_status | 0.4370 | 0.4199 | 0.7089 | 0.7349 | 0.9611 |
| XGB-all-reference-smoke | all | 0.4167 | 0.3622 | 0.6983 | 0.8206 | 0.9595 |

## SEMBA-status-decoder 相对 SEMBA 的提升

| metric | delta |
| --- | ---: |
| F1_negative | +0.3384 |
| PR_AUC_negative | +0.3503 |
| F1_macro | +0.1781 |
| AUROC | +0.0976 |

## 初步判断

smoke 阶段未发现时间泄露或当前事件特征污染。`SEMBA-status-decoder` 在 4 个主指标上均明显高于 SEMBA，因此进入 BitcoinOTC-1 全量 5 seeds 实验是合理的。
