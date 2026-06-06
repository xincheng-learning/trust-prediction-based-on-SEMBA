# BitcoinOTC-1 论文二分类协议 XGB 对比报告

## 实验口径

- 任务：`sign_class` 二分类，即已知未来有边，预测这条边是正边还是负边。
- 标签：`1` 是正边/信任，`0` 是负边/不信任。
- 协议：paper protocol，刻意模拟旧 `train.py` 的 `to_update=True`，当前 batch 先写入历史/记忆，再预测当前 batch。
- 风险：这个协议可能有 batch 内信息泄漏，因此不能替代上一轮严格无泄漏结论。
- 数据目录：`paper_protocol_sign_class_experiment/results/BitcoinOTC-1/`

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
| XGB-negative-aware-all | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.77 +/- 0.01 |
| XGB-negative-aware-no_dst_reputation | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.77 +/- 0.04 |
| XGB-paper-selected-all | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.76 +/- 0.01 |
| XGB-paper-selected-no_dst_reputation | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.78 +/- 0.05 |
| always_positive | 0.9239 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.4620 +/- 0.0000 | 0.1414 +/- 0.0000 | 0.00 +/- 0.00 |
| XGB-negative-aware-dst_reputation | 0.8969 +/- 0.0045 | 0.9449 +/- 0.0007 | 0.6181 +/- 0.0086 | 0.7575 +/- 0.0065 | 0.7455 +/- 0.0038 | 2.29 +/- 0.05 |
| XGB-paper-selected-dst_reputation | 0.8969 +/- 0.0045 | 0.9449 +/- 0.0007 | 0.6181 +/- 0.0086 | 0.7575 +/- 0.0065 | 0.7455 +/- 0.0038 | 2.35 +/- 0.19 |
| semba-emb64 | 0.8547 +/- 0.0251 | 0.7899 +/- 0.0219 | 0.4519 +/- 0.0252 | 0.6533 +/- 0.0251 | 0.4863 +/- 0.0292 | 102.70 +/- 0.28 |
| tgn-emb64 | 0.8400 +/- 0.0219 | 0.8093 +/- 0.0173 | 0.4478 +/- 0.0243 | 0.6439 +/- 0.0230 | 0.5626 +/- 0.0143 | 39.93 +/- 0.14 |

## 关键结论

- `XGB-all` 在 paper protocol 固定阈值 0.5 下达到 `F1_bin=1.0000 +/- 0.0000`、`AUROC=1.0000 +/- 0.0000`。这个结果几乎一定受到 batch-inclusive 泄漏影响，因为 pair/history 特征会在预测当前 batch 前看到当前 batch。
- `XGB-dst_reputation` 不使用 pair history，但仍达到 `F1_bin=0.8969 +/- 0.0045`、`AUROC=0.9449 +/- 0.0007`。这说明目标节点声誉/可见性在论文协议下依然是强信号，但它也会受到当前 batch 先写入历史的影响。
- 本地 `SEMBA-emb64` 固定阈值 0.5 为 `F1_bin=0.8547 +/- 0.0251`、`AUROC=0.7899 +/- 0.0219`；与论文 Table 6 的 SEMBA `F1=0.81 +/- 0.04, AUROC=0.79 +/- 0.02` 相比，F1 更高，AUROC 基本贴近。
- 本地 `TGN-emb64` 固定阈值 0.5 为 `F1_bin=0.8400 +/- 0.0219`、`AUROC=0.8093 +/- 0.0173`；与论文 TGN `F1=0.74 +/- 0.02, AUROC=0.82 +/- 0.03` 相比，F1 更高，AUROC 略低但接近。
- 因此，本轮 paper protocol 实验主要说明：按论文/旧代码协议，表格历史特征会变得极强，但其中一部分强度来自协议泄漏；研究上应保留 XGB 作为强 baseline，同时不能把该协议下的 1.0 当成严格泛化能力。

## XGB 特征重要性证据

| model | top features |
| --- | --- |
| XGB-paper-selected-all | pair_pos_count=0.540, pair_neg_count=0.458, dst_in_pos_ratio=0.003, src_pos_ratio=0.000, src_out_pos=0.000 |
| XGB-paper-selected-no_dst_reputation | pair_pos_count=0.434, src_pos_ratio=0.339, pair_neg_count=0.222, src_out_pos=0.004, dst_out_pos=0.000 |
| XGB-paper-selected-dst_reputation | dst_trusted_ratio=0.581, dst_in_pos_ratio=0.339, dst_in_neg=0.040, q_dst=0.012, dst_visibility_share_of_max=0.007 |

## 辅助结果：验证集选择正类 F1 阈值

| model | F1_bin | AUROC | F1_negative | F1_macro | PR_AUC_negative | total sec |
| --- | --- | --- | --- | --- | --- | --- |
| XGB-negative-aware-all | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.77 +/- 0.01 |
| XGB-negative-aware-no_dst_reputation | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.77 +/- 0.04 |
| XGB-paper-selected-all | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.76 +/- 0.01 |
| XGB-paper-selected-no_dst_reputation | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.78 +/- 0.05 |
| XGB-negative-aware-dst_reputation | 0.9399 +/- 0.0010 | 0.9449 +/- 0.0007 | 0.6164 +/- 0.0076 | 0.7782 +/- 0.0037 | 0.7455 +/- 0.0038 | 2.29 +/- 0.05 |
| XGB-paper-selected-dst_reputation | 0.9399 +/- 0.0010 | 0.9449 +/- 0.0007 | 0.6164 +/- 0.0076 | 0.7782 +/- 0.0037 | 0.7455 +/- 0.0038 | 2.35 +/- 0.19 |
| tgn-emb64 | 0.9392 +/- 0.0006 | 0.8093 +/- 0.0173 | 0.4779 +/- 0.0431 | 0.7086 +/- 0.0215 | 0.5626 +/- 0.0143 | 39.93 +/- 0.14 |
| semba-emb64 | 0.9273 +/- 0.0059 | 0.7899 +/- 0.0219 | 0.4311 +/- 0.0584 | 0.6792 +/- 0.0293 | 0.4863 +/- 0.0292 | 102.70 +/- 0.28 |

## 辅助结果：验证集选择 macro-F1 阈值

| model | F1_bin | AUROC | F1_negative | F1_macro | PR_AUC_negative | total sec |
| --- | --- | --- | --- | --- | --- | --- |
| XGB-negative-aware-all | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.77 +/- 0.01 |
| XGB-negative-aware-no_dst_reputation | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.77 +/- 0.04 |
| XGB-paper-selected-all | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.76 +/- 0.01 |
| XGB-paper-selected-no_dst_reputation | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.78 +/- 0.05 |
| XGB-negative-aware-dst_reputation | 0.9402 +/- 0.0008 | 0.9449 +/- 0.0007 | 0.6607 +/- 0.0175 | 0.8005 +/- 0.0085 | 0.7455 +/- 0.0038 | 2.29 +/- 0.05 |
| XGB-paper-selected-dst_reputation | 0.9402 +/- 0.0008 | 0.9449 +/- 0.0007 | 0.6607 +/- 0.0175 | 0.8005 +/- 0.0085 | 0.7455 +/- 0.0038 | 2.35 +/- 0.19 |
| tgn-emb64 | 0.9328 +/- 0.0062 | 0.8093 +/- 0.0173 | 0.5291 +/- 0.0146 | 0.7310 +/- 0.0099 | 0.5626 +/- 0.0143 | 39.93 +/- 0.14 |
| semba-emb64 | 0.9150 +/- 0.0075 | 0.7899 +/- 0.0219 | 0.4875 +/- 0.0297 | 0.7012 +/- 0.0179 | 0.4863 +/- 0.0292 | 102.70 +/- 0.28 |

## 混淆矩阵均值：固定阈值 0.5

| model | TN | FP | FN | TP | pred neg | pred pos |
| --- | --- | --- | --- | --- | --- | --- |
| XGB-negative-aware-all | 755.0 | 0.0 | 0.0 | 4584.0 | 755.0 | 4584.0 |
| XGB-negative-aware-no_dst_reputation | 755.0 | 0.0 | 0.0 | 4584.0 | 755.0 | 4584.0 |
| XGB-paper-selected-all | 755.0 | 0.0 | 0.0 | 4584.0 | 755.0 | 4584.0 |
| XGB-paper-selected-no_dst_reputation | 755.0 | 0.0 | 0.0 | 4584.0 | 755.0 | 4584.0 |
| always_positive | 0.0 | 755.0 | 0.0 | 4584.0 | 0.0 | 5339.0 |
| XGB-negative-aware-dst_reputation | 701.2 | 53.8 | 813.0 | 3771.0 | 1514.2 | 3824.8 |
| XGB-paper-selected-dst_reputation | 701.2 | 53.8 | 813.0 | 3771.0 | 1514.2 | 3824.8 |
| semba-emb64 | 500.2 | 254.8 | 968.8 | 3615.2 | 1469.0 | 3870.0 |
| tgn-emb64 | 533.6 | 221.4 | 1101.4 | 3482.6 | 1635.0 | 3704.0 |

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

| model | F1_bin | AUROC | F1_negative | F1_macro | PR_AUC_negative | total sec |
| --- | --- | --- | --- | --- | --- | --- |
| semba-emb512 | 0.8363 | 0.7774 | 0.4393 | 0.6378 | 0.5005 | 226.63 |

## processed 缓存检查

| processed_dir | status | events/error | nodes | train | val | test | pos_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| processed | error | RuntimeError: The 'data' object was created by an older version of PyG. If this error occurred while loading an already existing dataset, remove the 'processed/' directory in the dataset's root folder and try again. |  |  |  |  |  |
| processed2 | ok | 35592 | 12005 | 24915 | 5338 | 5339 | 0.899893 |

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
