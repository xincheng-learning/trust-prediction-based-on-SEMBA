# Status/Asymmetry-aware SEMBA 最小可行实验计划

## 1. 实验目标

本实验验证一个最小改动版本：

```text
SEMBA signed memory + online status/asymmetry decoder
```

核心问题：

> 在不改 SEMBA signed memory、balanced aggregation 和 long-term propagation 主体的前提下，把严格在线的源用户行为倾向、目标用户历史信誉、源-目标地位差异加入 decoder，是否能改善 BitcoinOTC-1 `sign_class` 任务，尤其是负类 distrust 预测。

本轮不以证明公开代码协议泄露为目标，也不以证明 XGB 强 baseline 为目标。XGB 只作为参照模型。

## 2. Version A 设计

原始 SEMBA decoder：

```text
[z_src, z_dst] -> MLP -> sign probability
```

本实验 decoder：

```text
[z_src, z_dst, abs(z_src - z_dst), z_src * z_dst, online_status_features] -> MLP -> sign probability
```

MLP：

```text
Linear(input_dim, hidden_dim)
ReLU
Dropout(0.1)
Linear(hidden_dim, 1)
```

实现策略：

- 不修改 `src/semba.py`。
- 不修改 SEMBA memory update。
- 不修改 `model_wrapper.py` 的原始 SEMBA 路径。
- 新增 runner 内部组合 `STGNN('semba')` 和 `StatusPairDecoder`。

## 3. 数据和协议

- 数据集：`BitcoinOTC-1`
- 缓存：`data/BitcoinOTC-1/processed2`
- 任务：`sign_class`
- 时间切分：70% train / 15% val / 15% test
- smoke seed：`42`
- 第一阶段：`--smoke`，限制事件数并减少 epoch，只验证代码路径和方向。

严格在线协议：

```text
predict current event before updating memory/history with current event
```

具体约束：

- 当前事件不能参与自身 status feature。
- 同一时间戳事件先统一构造特征，再统一更新历史。
- validation/test 顺序滚动更新。
- scaler 只在 train split 上 fit。
- 阈值只在 validation 上选择，不使用 test 调阈值。

## 4. 在线 status/asymmetry 特征

特征来源复用：

```text
experiments/dynamic_features.py
experiments/strict_temporal_protocol.py
```

核心特征组：

- Source 行为倾向：历史出边数、正/负出边数、正/负出边比例、历史活跃度。
- Target 信誉状态：历史入边数、正/负入边数、被信任比例、信誉分数。
- Status/asymmetry 差异：目标-源信誉差、目标-源活跃度差、目标是否更有信誉。
- 缺失/冷启动标记：源/目标是否历史可见、双方是否均已知。

连续特征使用 train split 拟合的 scaler 标准化；known flag 不标准化。

## 5. 对比模型

smoke 最低对比：

| 模型 | 作用 |
| --- | --- |
| SEMBA | 原始图模型 baseline |
| SEMBA-status-decoder | 本次核心改进 |
| XGB-all-reference | 已有严格协议结果参照，不作为本轮核心目标 |

后续 full run 可扩展：

- SEMBA-noprop
- TGN
- source-only / target-only / status-gap 消融
- 5 seeds：42, 43, 44, 45, 46

## 6. 主指标

必须报告：

- `F1_negative`
- `PR_AUC_negative`
- `F1_macro`
- `AUROC`
- `F1_bin`
- `F1_weighted`

优先看：

```text
threshold_mode = val_F1_macro
```

同时保留：

```text
fixed_0.5
```

## 7. 成功/失败判断

smoke 阶段只判断是否值得进入 5 seeds，不得写最终结论。

进入 5 seeds 的建议条件：

- `SEMBA-status-decoder` 相比 `SEMBA` 至少两个主指标提升；或
- 负类指标明显提升，且没有时间安全审计失败；或
- 指标相近但训练稳定，说明实现可继续做消融。

暂不进入 5 seeds 的条件：

- 时间安全审计失败；
- 所有主指标均低于 SEMBA；
- 模型明显塌缩到只预测单一类别；
- 结果只能由 current event 或 current batch 污染解释。

## 8. 输出物

代码：

```text
status_asymmetry_semba_experiment/py/build_online_status_features.py
status_asymmetry_semba_experiment/py/run_semba_status_decoder.py
status_asymmetry_semba_experiment/py/aggregate_status_decoder_results.py
status_asymmetry_semba_experiment/py/audit_strict_online_status_features.py
```

表格：

```text
status_asymmetry_semba_experiment/tables/status_feature_dictionary.csv
status_asymmetry_semba_experiment/tables/status_feature_audit.csv
status_asymmetry_semba_experiment/tables/status_decoder_metrics_seed42.csv
status_asymmetry_semba_experiment/tables/status_decoder_metrics_summary.csv
status_asymmetry_semba_experiment/tables/status_decoder_ablation.csv
```

报告：

```text
status_asymmetry_semba_experiment/md/bitcoinotc_status_decoder_smoke_report.md
status_asymmetry_semba_experiment/md/bitcoinotc_status_decoder_final_report.md
```
