# BitcoinOTC strict comparison final report

日期：2026-06-05

## 实验设置

本实验在 `exp/xgb-history` 工作区执行，结果目录为：

```text
results/strict_compare_full_gpu/BitcoinOTC-1/
```

数据与协议：

- 数据集：`BitcoinOTC-1`
- 缓存：`processed2`
- 任务：`sign_class`
- 时间切分：70% train / 15% val / 15% test
- full events：35592
- test events：5339，其中正边 4584、负边 755，test 正边比例 0.8586
- seeds：`42, 43, 44, 45, 46`
- 图模型：CUDA 环境 `C:\ProgramData\anaconda3\envs\pytorch\python.exe`
- GPU：NVIDIA GeForce RTX 3060 Ti

严格协议：

- XGB 使用 `global_online + strict_timestamp` 在线历史特征。
- 图模型使用 predict-before-update 的 no-leakage mini-block 协议。
- `max_events_per_block=1000`，不会拆开相同 timestamp。
- 当前 block 预测和反传完成后，才把当前 block 写入 memory/history。

主表使用 val 上最大化 `F1_macro` 的阈值。固定 `0.5` 阈值另作辅助检查。

## 主结果

| model | PR_AUC_negative | F1_negative | F1_macro | balanced_accuracy | AUROC | F1_weighted | train sec | total sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGB-all | 0.5472 +/- 0.0055 | 0.4900 +/- 0.0077 | 0.6970 +/- 0.0096 | 0.7168 +/- 0.0091 | 0.8235 +/- 0.0055 | 0.8455 +/- 0.0116 | 0.81 +/- 0.03 | 0.83 +/- 0.03 |
| XGB-dst_reputation | 0.4160 +/- 0.0069 | 0.4645 +/- 0.0041 | 0.6845 +/- 0.0030 | 0.6954 +/- 0.0025 | 0.7443 +/- 0.0043 | 0.8424 +/- 0.0026 | 0.47 +/- 0.02 | 0.49 +/- 0.02 |
| SEMBA | 0.4096 +/- 0.0194 | 0.4101 +/- 0.0563 | 0.6524 +/- 0.0238 | 0.6665 +/- 0.0408 | 0.7505 +/- 0.0162 | 0.8262 +/- 0.0119 | 188.04 +/- 56.06 | 188.24 +/- 56.06 |
| TGN | 0.4087 +/- 0.0295 | 0.4036 +/- 0.0396 | 0.6606 +/- 0.0209 | 0.6433 +/- 0.0219 | 0.7257 +/- 0.0107 | 0.8449 +/- 0.0083 | 97.99 +/- 14.09 | 98.14 +/- 14.09 |
| XGB-no_dst_reputation | 0.3814 +/- 0.0057 | 0.3914 +/- 0.0255 | 0.6464 +/- 0.0089 | 0.6460 +/- 0.0231 | 0.7571 +/- 0.0051 | 0.8293 +/- 0.0046 | 0.62 +/- 0.03 | 0.64 +/- 0.02 |
| SEMBA-noprop | 0.2649 +/- 0.0494 | 0.3495 +/- 0.0496 | 0.6009 +/- 0.0366 | 0.6308 +/- 0.0327 | 0.6879 +/- 0.0360 | 0.7812 +/- 0.0274 | 163.55 +/- 27.87 | 163.79 +/- 27.91 |

## 固定 0.5 阈值辅助检查

| model | PR_AUC_negative | F1_negative | F1_macro | balanced_accuracy | F1_weighted |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGB-all | 0.5472 | 0.4406 | 0.6341 | 0.7353 | 0.7730 |
| XGB-dst_reputation | 0.4160 | 0.3988 | 0.6259 | 0.6759 | 0.7888 |
| SEMBA | 0.4096 | 0.2851 | 0.6048 | 0.5842 | 0.8341 |
| TGN | 0.4087 | 0.2297 | 0.5785 | 0.5672 | 0.8287 |
| SEMBA-noprop | 0.2649 | 0.3482 | 0.6042 | 0.6270 | 0.7877 |
| always_positive | 0.1414 | 0.0000 | 0.4620 | 0.5000 | 0.7933 |

固定阈值下，XGB-all 的 `F1_weighted` 不高，这是阈值偏向负类带来的代价；但它的 `PR_AUC_negative` 和 AUROC 不依赖阈值，仍然明显强于图模型。

## 混淆矩阵均值

主阈值 `val_F1_macro` 下，test 集平均混淆矩阵：

| model | TN | FP | FN | TP | predicted negative | predicted positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XGB-all | 412.8 | 342.2 | 518.8 | 4065.2 | 931.6 | 4407.4 |
| XGB-dst_reputation | 374.8 | 380.2 | 484.2 | 4099.8 | 859.0 | 4480.0 |
| SEMBA | 339.2 | 415.8 | 532.8 | 4051.2 | 872.0 | 4467.0 |
| TGN | 262.6 | 492.4 | 280.2 | 4303.8 | 542.8 | 4796.2 |
| SEMBA-noprop | 340.6 | 414.4 | 869.2 | 3714.8 | 1209.8 | 4129.2 |

XGB-all 对负边的平均命中最多，约 413/755；SEMBA 约 339/755；TGN 约 263/755。

## 关键判断

### 表格模型是否接近图模型

在当前严格协议和当前实现下，XGB-all 不只是接近图模型，而是在主要少数类指标上超过了图模型：

- 比 SEMBA 高：
  - `PR_AUC_negative`：+0.1376
  - `F1_negative`：+0.0799
  - `F1_macro`：+0.0446
  - `AUROC`：+0.0730
- 比 TGN 高：
  - `PR_AUC_negative`：+0.1386
  - `F1_negative`：+0.0864
  - `F1_macro`：+0.0364
  - `AUROC`：+0.0978

`F1_weighted` 上 XGB-all 与 TGN 几乎持平，XGB-all 为 0.8455，TGN 为 0.8449；但这个指标受类别不平衡影响大，不作为主结论。

### 资源优势是否明显

非常明显。

- XGB-all full 5 seed 平均每次约 0.83 秒。
- TGN CUDA 平均约 98.14 秒。
- SEMBA CUDA 平均约 188.24 秒。
- SEMBA-noprop CUDA 平均约 163.79 秒。

按 `PR_AUC_negative / total_wall_time_sec` 粗略看效率：

| model | efficiency |
| --- | ---: |
| XGB-dst_reputation | 0.8536 |
| XGB-all | 0.6566 |
| XGB-no_dst_reputation | 0.5933 |
| TGN | 0.0042 |
| SEMBA | 0.0022 |
| SEMBA-noprop | 0.0016 |

这说明 XGB 的低资源优势不是一点点，而是数量级差异。

### 用户提出的目标节点历史特征是否重要

重要。

`XGB-dst_reputation` 只使用目标节点声誉/可见性相关特征，就已经达到：

- `PR_AUC_negative=0.4160`
- `F1_negative=0.4645`
- `F1_macro=0.6845`

它在负类 F1 和 F1_macro 上接近 XGB-all，并超过 SEMBA/TGN 的均值。这说明“目标用户历史被信任程度、历史可见度、高可见节点”等特征确实是强信号。

但 XGB-all 仍然明显更强，尤其 `PR_AUC_negative=0.5472`，说明除了目标声誉，还需要源节点历史、pair history、互惠/覆盖等组合特征。

## 研究结论

在 BitcoinOTC-1 的严格时间设定下，XGB 历史统计模型具有明确研究价值：

1. 它是强 baseline，不应该只作为“简单传统机器学习”附录。
2. 它在少数类不信任边识别上优于当前严格 runner 下的 TGN/SEMBA。
3. 它消耗的训练时间远低于动态图模型，适合作为低资源预测器。
4. 它能解释哪些历史结构信号关键，尤其是目标节点历史声誉和可见性。
5. 它适合继续做融合：XGB early-exit、XGB features + SEMBA classifier。

谨慎点：

- 这说明“在当前代码、当前严格协议、当前超参下 XGB 很强”，不等于理论上 XGB 永远优于图模型。
- 图模型可能通过更细的 batch 策略、更强负类损失、更长训练、更好的超参继续提升。
- 但即便图模型后续提升，XGB 的资源优势和解释性仍然足够支撑研究价值。

## 推荐下一步

优先做两个方向：

1. **XGB early-exit + graph hard cases**
   - XGB 高置信样本直接输出。
   - 低置信样本交给 SEMBA/TGN。
   - 目标是在接近 XGB/SEMBA 最优指标的同时减少图模型调用比例。

2. **XGB features + SEMBA classifier**
   - 把 XGB 在线历史特征拼接到 SEMBA 的 pair classifier 输入。
   - 检验显式历史声誉特征是否能补 SEMBA 的短板。

当前最值得写进研究叙述的一句话是：

> 在 BitcoinOTC-1 的严格在线时间设定下，显式历史统计特征构成了一个极强且极低成本的 baseline；特别是目标节点历史声誉/可见性特征，对负边识别有稳定贡献。
