# BitcoinOTC-1 SEMBA-status-decoder Final Report

## 1. 实验范围

- 数据集：BitcoinOTC-1
- 缓存：`processed2`
- 任务：`sign_class`
- 协议：strict predict-before-update
- seeds：42, 43, 44, 45, 46
- 图模型 epoch：最多 6，early stop patience 3
- 主报告阈值：validation split 选择 `F1_macro` 最优阈值
- 结果目录：`status_asymmetry_semba_experiment/results/full_5seeds_20260623/BitcoinOTC-1/full`

本实验的目标是验证 Version A：

```text
[z_src, z_dst, abs(z_src - z_dst), z_src * z_dst, online_status_features] -> MLP -> sign prediction
```

其中 `online_status_features` 严格来自当前事件时间之前的历史事件。

## 2. 时间安全审计

审计结论：`pass`

| check | status | evidence |
| --- | --- | --- |
| strict_timestamp_feature_build | pass | 同一时间戳事件先取特征，再统一更新历史 |
| history_window | pass | 特征在 `update_history(event)` 之前构造 |
| train_only_scaler | pass | scaler 只 fit train split |
| row_alignment_train | pass | 24,915 feature rows vs 24,915 train events |
| row_alignment_val | pass | 5,338 feature rows vs 5,338 val events |
| row_alignment_test | pass | 5,339 feature rows vs 5,339 test events |
| nan_or_inf | pass | 所有选中特征矩阵均为有限值 |

因此，本轮结果没有发现当前事件特征污染或同 batch 标签泄露。

## 3. 5 seeds 汇总结果

以下为 `val_F1_macro` 阈值策略下的 5 seeds 均值：

| model | feature_set | F1_negative | PR_AUC_negative | F1_macro | AUROC | F1_weighted |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SEMBA | graph_memory | 0.3374 ± 0.0424 | 0.3882 ± 0.0217 | 0.6232 ± 0.0141 | 0.7360 ± 0.0101 | 0.8282 ± 0.0113 |
| SEMBA-status-decoder | all_status | 0.5406 ± 0.0117 | 0.6001 ± 0.0147 | 0.7204 ± 0.0105 | 0.8537 ± 0.0067 | 0.8493 ± 0.0096 |
| XGB-all-reference | all | 0.4900 ± 0.0077 | 0.5472 ± 0.0055 | 0.6970 ± 0.0096 | 0.8235 ± 0.0055 | 0.8455 ± 0.0116 |

固定阈值 0.5 下的参考结果：

| model | feature_set | F1_negative | PR_AUC_negative | F1_macro | AUROC |
| --- | --- | ---: | ---: | ---: | ---: |
| SEMBA | graph_memory | 0.2289 ± 0.0588 | 0.3882 ± 0.0217 | 0.5787 ± 0.0297 | 0.7360 ± 0.0101 |
| SEMBA-status-decoder | all_status | 0.5534 ± 0.0081 | 0.6001 ± 0.0147 | 0.7383 ± 0.0054 | 0.8537 ± 0.0067 |
| XGB-all-reference | all | 0.4406 ± 0.0043 | 0.5472 ± 0.0055 | 0.6341 ± 0.0026 | 0.8235 ± 0.0055 |

## 4. SEMBA-status-decoder 相对 SEMBA 的提升

5 个 seeds 上，`SEMBA-status-decoder` 相对 SEMBA 的提升方向全部为正：

| metric | mean delta | all 5 seeds positive |
| --- | ---: | --- |
| F1_negative | +0.2032 | yes |
| PR_AUC_negative | +0.2118 | yes |
| F1_macro | +0.0971 | yes |
| AUROC | +0.1177 | yes |

逐 seed 差值：

| seed | ΔF1_negative | ΔPR_AUC_negative | ΔF1_macro | ΔAUROC |
| ---: | ---: | ---: | ---: | ---: |
| 42 | +0.2243 | +0.2079 | +0.1120 | +0.1240 |
| 43 | +0.1999 | +0.1841 | +0.0903 | +0.1116 |
| 44 | +0.1955 | +0.1941 | +0.0847 | +0.0965 |
| 45 | +0.1343 | +0.2309 | +0.0763 | +0.1226 |
| 46 | +0.2620 | +0.2422 | +0.1224 | +0.1338 |

## 5. 与 XGB-all 参照的关系

XGB-all 在本报告中只作为参照，不作为本轮唯一成功标准。

在 `val_F1_macro` 阈值下，`SEMBA-status-decoder` 也超过了 XGB-all-reference：

| metric | SEMBA-status-decoder | XGB-all-reference | delta |
| --- | ---: | ---: | ---: |
| F1_negative | 0.5406 | 0.4900 | +0.0507 |
| PR_AUC_negative | 0.6001 | 0.5472 | +0.0528 |
| F1_macro | 0.7204 | 0.6970 | +0.0234 |
| AUROC | 0.8537 | 0.8235 | +0.0302 |

这个结果说明，在当前实现和评测协议下，status/asymmetry 特征并不是只对表格模型有效；它作为 decoder 条件输入也能显著补充 SEMBA 的 signed memory 表示。

## 6. 成功标准判断

按计划中的“强成功”标准：

- F1_negative 提升 ≥ 0.02：满足，平均 +0.2032。
- PR_AUC_negative 提升 ≥ 0.02：满足，平均 +0.2118。
- F1_macro 提升 ≥ 0.01：满足，平均 +0.0971。
- AUROC 提升 ≥ 0.01：满足，平均 +0.1177。
- 5 seeds 下提升方向稳定：满足，4 个主指标全部 5/5 seeds 为正。

结论：

```text
Version A: SEMBA + Online Status Feature Decoder 在 BitcoinOTC-1 sign_class 严格在线协议下有效。
```

## 7. 当前限制

1. 本轮只做了 all-status 特征，没有完成 source-only、target-only、status-gap-only 消融。
2. 本轮只跑 BitcoinOTC-1，还没有扩展到 BitcoinAlpha-1。
3. 当前实现只改 decoder，没有进入 message gate 或 memory update 层面。
4. 与 TGN、SEMBA-noprop 的 full 5 seeds 对比可以复用既有严格实验结果，但本轮 runner 未重新跑这两个模型。

## 8. 下一步建议

优先顺序：

1. 在同一 runner 中补 source-only、target-only、status-gap-only 消融。
2. 复用本实验结构跑 BitcoinAlpha-1。
3. 如果两个数据集都稳定，再做 Version B：status/asymmetry gate。
4. 写论文时主张应聚焦为：在 SEMBA signed memory 与 balanced aggregation 基础上，引入严格在线 status/asymmetry-aware decoder，增强少数类 distrust 识别。
