# BitcoinOTC-1 论文二分类协议 XGB 对比报告

## 实验口径

- 任务：`sign_class` 二分类，即已知未来有边，预测这条边是正边还是负边。
- 标签：`1` 是正边/信任，`0` 是负边/不信任。
- 协议：paper protocol，刻意模拟旧 `train.py` 的 `to_update=True`，当前 batch 先写入历史/记忆，再预测当前 batch。
- 风险：这个协议可能有 batch 内信息泄漏，因此不能替代上一轮严格无泄漏结论。
- 数据目录：`paper_protocol_sign_class_experiment/results_smoke/BitcoinOTC-1/`

## 指标中文解释

- `F1_bin`：二分类正类 F1，默认正类是 `1`。公式是 `2 * precision * recall / (precision + recall)`。
- `AUROC`：ROC 曲线下面积，可理解为随机正边得分高于随机负边得分的概率。
- `F1_negative`：负边/不信任边这一类的 F1。
- `F1_macro`：正类 F1 和负类 F1 的平均，类别不平衡时比加权 F1 更公平。
- `PR_AUC_negative`：把负边当成目标类时的 precision-recall 曲线面积。
- `F1_weighted`：按各类别样本数加权的 F1，不是 1-10 分评分；正边多时它会更受正边表现影响。
- `total sec`：训练和评估耗时之和，不含原始数据下载。

## 论文 Table 6 外部参考

| model | source | paper F1_bin | paper AUROC |
| --- | --- | --- | --- |
| TGN | paper Table 6 BTC-Otc | 0.74 +/- 0.02 | 0.82 +/- 0.03 |
| SEMBA | paper Table 6 BTC-Otc | 0.81 +/- 0.04 | 0.79 +/- 0.02 |

## 本地 paper protocol 主结果：固定阈值 0.5

固定阈值 0.5 最接近旧 SEMBA/TGN 的 `classify=True` 口径，也是和论文 Table 6 对比时的主表。

| model | F1_bin | AUROC | F1_negative | F1_macro | PR_AUC_negative | total sec |
| --- | --- | --- | --- | --- | --- | --- |
| XGB-negative-aware-all | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.72 |
| XGB-paper-selected-all | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.73 |
| always_positive | 0.9740 | 0.5000 | 0.0000 | 0.4870 | 0.0507 | 0.00 |
| semba-emb64 | 0.9589 | 0.6951 | 0.3256 | 0.6423 | 0.1552 | 1.72 |

## 辅助结果：验证集选择正类 F1 阈值

| model | F1_bin | AUROC | F1_negative | F1_macro | PR_AUC_negative | total sec |
| --- | --- | --- | --- | --- | --- | --- |
| XGB-negative-aware-all | 0.9740 | 1.0000 | 0.6786 | 0.8263 | 1.0000 | 0.72 |
| XGB-paper-selected-all | 0.9740 | 1.0000 | 0.6786 | 0.8263 | 1.0000 | 0.73 |
| semba-emb64 | 0.9740 | 0.6951 | 0.0000 | 0.4870 | 0.1552 | 1.72 |

## 辅助结果：验证集选择 macro-F1 阈值

| model | F1_bin | AUROC | F1_negative | F1_macro | PR_AUC_negative | total sec |
| --- | --- | --- | --- | --- | --- | --- |
| XGB-negative-aware-all | 0.9740 | 1.0000 | 0.6786 | 0.8263 | 1.0000 | 0.72 |
| XGB-paper-selected-all | 0.9740 | 1.0000 | 0.6786 | 0.8263 | 1.0000 | 0.73 |
| semba-emb64 | 0.9461 | 0.6951 | 0.2991 | 0.6226 | 0.1552 | 1.72 |

## 混淆矩阵均值：固定阈值 0.5

| model | TN | FP | FN | TP | pred neg | pred pos |
| --- | --- | --- | --- | --- | --- | --- |
| XGB-negative-aware-all | 38.0 | 0.0 | 0.0 | 711.0 | 38.0 | 711.0 |
| XGB-paper-selected-all | 38.0 | 0.0 | 0.0 | 711.0 | 38.0 | 711.0 |
| always_positive | 0.0 | 38.0 | 0.0 | 711.0 | 0.0 | 749.0 |
| semba-emb64 | 14.0 | 24.0 | 34.0 | 677.0 | 48.0 | 701.0 |

## 上一轮严格无泄漏结果对照

| model | strict F1_macro | strict F1_negative | strict PR_AUC_negative | strict AUROC |
| --- | --- | --- | --- | --- |
| XGB-all | 0.6970 +/- 0.0096 | 0.4900 +/- 0.0077 | 0.5472 +/- 0.0055 | 0.8235 +/- 0.0055 |
| XGB-dst_reputation | 0.6845 +/- 0.0030 | 0.4645 +/- 0.0041 | 0.4160 +/- 0.0069 | 0.7443 +/- 0.0043 |
| tgn | 0.6606 +/- 0.0209 | 0.4036 +/- 0.0396 | 0.4087 +/- 0.0295 | 0.7257 +/- 0.0107 |
| semba | 0.6524 +/- 0.0238 | 0.4101 +/- 0.0563 | 0.4096 +/- 0.0194 | 0.7505 +/- 0.0162 |
| XGB-no_dst_reputation | 0.6464 +/- 0.0089 | 0.3914 +/- 0.0255 | 0.3814 +/- 0.0057 | 0.7571 +/- 0.0051 |
| semba-noprop | 0.6009 +/- 0.0366 | 0.3495 +/- 0.0496 | 0.2649 +/- 0.0494 | 0.6879 +/- 0.0360 |

## 512 维单 seed 诊断

512 维诊断结果尚未运行。

## 必答问题

1. 是否使用 `sign_class` 二分类任务：是。本实验只处理正边/负边二分类。
2. 是否使用 70/15/15 时间切分：是。由项目 `TemporalData.train_val_test_split` 产生，manifest 中记录 split summary。
3. 是否使用旧代码一致的 `to_update=True` paper protocol：是。图模型 runner 在预测前调用 `to_update=True`；XGB 特征先写入当前 batch 再生成当前 batch 特征。
4. XGB 是否使用 batch-inclusive 特征：是。这是本实验刻意模拟论文/旧代码口径的核心。
5. 本地 SEMBA/TGN 是否接近论文 Table 6：见固定阈值 0.5 主表，应以 `F1_bin` 和 `AUROC` 对照论文参考行。
6. 若本地 SEMBA/TGN 未接近论文：不能直接归因于环境，需要同时检查参数规模、processed 缓存、PyTorch/PyG 版本和旧代码协议。
7. 论文协议下 XGB 与 SEMBA/TGN 如何：以固定阈值 0.5 主表为准，辅助阈值表只能作为补充。
8. 负边识别上 XGB 是否仍强：看 `F1_negative`、`F1_macro`、`PR_AUC_negative`。
9. 与上一轮严格无泄漏结果是否方向一致：看“上一轮严格无泄漏结果对照”。
10. 是否值得融合 XGB 特征进 SEMBA：若 XGB 在主表或负边指标上接近/超过图模型，则值得继续做 `XGB features + SEMBA decoder`。
