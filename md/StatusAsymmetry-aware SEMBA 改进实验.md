# Codex 执行指令：Status/Asymmetry-aware SEMBA 最小可行改进实验

## 0. 任务目标

当前任务不是继续证明 SEMBA 公开协议是否存在 batch-inclusive 风险，也不是继续证明 XGB 历史统计特征是强 baseline。

本次实验的唯一核心目标是：

> 在 SEMBA 的 signed memory 机制基础上，加入 status / reputation / asymmetry-aware 的地位不对称建模，快速验证这种机制是否能在严格在线 sign_class 任务中改善负类 distrust 预测、macro-F1 或 AUROC。

请优先做一个最小可行版本，不要一开始做复杂大改。

---

## 1. 当前项目背景

项目目录：

```text
E:\导师之任\导师之任\社交信任\semba-main
```

主要实验 worktree：

```text
E:\导师之任\导师之任\社交信任\semba-main\.worktrees\exp-xgb-history
```

当前主要分支：

```text
exp/xgb-history
```

远端仓库：

```text
https://github.com/xincheng-learning/trust-prediction-based-on-SEMBA
```

已有结论：

1. SEMBA 原模型包含 signed memory、balanced aggregation 和 long-term propagation。
2. 严格在线协议下，已有 SEMBA / TGN / XGB 对比实验已经完成。
3. XGB 历史统计特征显示出很强预测能力，但本次任务不再围绕证明 XGB 强基线展开。
4. 当前需要把 XGB 中有效的在线历史统计信号转化为 SEMBA 的 status / asymmetry-aware 机制。
5. 本次实验目标是改进 SEMBA，而不是批判原论文。

---

## 2. 本次实验的研究问题

请围绕以下问题执行：

> SEMBA 已经区分正负 memory 和结构平衡传播，但它没有显式建模“谁更有信誉、谁更活跃、目标用户是否更容易被信任、源用户是否更容易给负边”这些地位不对称信息。若把这些在线历史统计信息加入 SEMBA 的 decoder 或 gate，是否能提高动态符号信任预测，尤其是负类 distrust 预测？

---

## 3. 最小可行改进方向

本次只做两个版本，优先完成 Version A。

---

# Version A：SEMBA + Online Status Feature Decoder

## 3.1 思路

不改 SEMBA 的 memory update 主体，只改最终 sign prediction decoder。

原始 SEMBA 大致是：

```text
z_src, z_dst -> MLP decoder -> sign prediction
```

改为：

```text
z_src, z_dst, online_status_features(src, dst, t) -> MLP decoder -> sign prediction
```

这里的 `online_status_features` 必须是严格在线特征，只能由时间 `< t` 的历史事件构造。

## 3.2 为什么先做 decoder 版本

原因：

1. 改动最小；
2. 不破坏 SEMBA 主体结构；
3. 容易和 SEMBA 原模型做消融；
4. 能快速验证 status / asymmetry 信号是否能补充 signed memory；
5. 如果有效，再进入 gate 版本。

---

# Version B：SEMBA + Status/Asymmetry Gate

## 3.3 思路

在 Version A 跑通后，再做 gate 版本。

用在线 status 特征生成 gate：

```text
gate = sigmoid(MLP(status_features))
```

然后控制：

```text
message = gate * message
```

或分别控制：

```text
positive_message = gate_pos * positive_message
negative_message = gate_neg * negative_message
```

Version B 是后续增强，不作为本轮必须完成项。

---

## 4. 必须使用的在线 status / asymmetry 特征

请优先复用已有动态特征代码：

```text
strict_xgb_comparison_experiment/py/dynamic_features.py
strict_xgb_comparison_experiment/py/strict_temporal_protocol.py
```

如果已有函数可复用，优先复用，不要重复造轮子。

每条事件 `(src, dst, t)` 至少构造以下特征：

### 4.1 Source 行为倾向

```text
src_seen_before
src_out_count_before
src_pos_out_count_before
src_neg_out_count_before
src_pos_out_ratio_before
src_neg_out_ratio_before
src_activity_log_before
```

解释：

- `src_seen_before`：源用户过去是否出现过；
- `src_out_count_before`：源用户过去发出过多少边；
- `src_pos_out_ratio_before`：源用户过去更倾向给正边还是负边；
- `src_neg_out_ratio_before`：源用户是否更“严格”或更容易给负边。

### 4.2 Target 信誉状态

```text
dst_seen_before
dst_in_count_before
dst_pos_in_count_before
dst_neg_in_count_before
dst_pos_in_ratio_before
dst_neg_in_ratio_before
dst_reputation_score_before
```

解释：

- `dst_seen_before`：目标用户过去是否出现过；
- `dst_pos_in_count_before`：目标用户历史上被信任的次数；
- `dst_neg_in_count_before`：目标用户历史上被不信任的次数；
- `dst_reputation_score_before`：目标信誉，可定义为：
  - `(dst_pos_in_count - dst_neg_in_count) / (dst_in_count + 1)`
  - 或 `log1p(dst_pos_in_count) - log1p(dst_neg_in_count)`。

### 4.3 Status / Asymmetry 差异

```text
status_gap_before
activity_gap_before
reputation_gap_before
dst_more_reputable_flag
src_more_negative_flag
```

解释：

- `status_gap_before = dst_reputation_score_before - src_reputation_score_before`
- `activity_gap_before = log1p(dst_activity) - log1p(src_activity)`
- `dst_more_reputable_flag`：目标用户历史信誉是否高于源用户；
- `src_more_negative_flag`：源用户是否更倾向发负边。

### 4.4 缺失与冷启动标记

```text
src_status_known
dst_status_known
both_status_known
either_status_unknown
```

解释：

status 特征不能把缺失直接解释成真实 0。必须增加 known flag。

---

## 5. 严格在线要求

本次模型必须遵守严格在线：

```text
predict current event before updating memory/history with current event
```

具体要求：

1. 当前事件不能参与自身特征；
2. 当前 batch 不能先写入 history 再预测当前 batch；
3. 如果多个事件具有相同时间戳，优先采用“先预测同时间戳所有事件，再统一更新”的安全策略；
4. status feature 的历史窗口必须是 `< current_time`，不是 `<= current_time`；
5. 任何用于当前事件的特征都不能包含当前事件 label；
6. valid / test 只能按历史顺序滚动更新，不能一次性提前写入。

注意：

本任务不需要再写大量文字证明公开代码泄露问题，但代码实现必须采用 strict online。

---

## 6. 实验数据

先只做：

```text
BitcoinOTC-1
```

使用缓存：

```text
data/BitcoinOTC-1/processed2
```

任务：

```text
sign_class
```

含义：

- 未来边已知会出现；
- 预测该边是 positive 还是 negative；
- positive label = 1；
- negative label = 0。

时间切分：

```text
70% train / 15% val / 15% test
```

seeds：

```text
42
```

先只跑 seed=42，作为 smoke / minimum viable experiment。
如果有效，再扩展：

```text
42, 43, 44, 45, 46
```

---

## 7. 必须对比的模型

本轮最小实验必须包含：

| 模型 | 作用 |
|---|---|
| SEMBA | 原始 SEMBA baseline |
| SEMBA-noprop | 检验 long-term propagation |
| TGN | 普通动态图 memory baseline |
| SEMBA-status-decoder | 本次核心改进模型 |
| XGB-all | 只作为强 baseline 参照，不作为本轮研究重心 |

如果时间不足，最低要求：

```text
SEMBA vs SEMBA-status-decoder vs XGB-all
```

---

## 8. SEMBA-status-decoder 的实现要求

### 8.1 新增模型命名

建议命名：

```text
SEMBAStatusDecoder
```

或配置名：

```text
semba-status-decoder
```

### 8.2 输入

每个 batch 中，对每条边准备：

```text
src
dst
t
label
z_src
z_dst
status_features
```

其中：

- `z_src`, `z_dst` 来自 SEMBA 原 embedding；
- `status_features` 来自严格在线历史统计。

### 8.3 Decoder 结构

建议最小结构：

```text
pair_emb = concat([
    z_src,
    z_dst,
    abs(z_src - z_dst),
    z_src * z_dst,
    status_features
])

logit = MLP(pair_emb)
```

MLP 最小配置：

```text
Linear(input_dim, hidden_dim)
ReLU
Dropout(0.1)
Linear(hidden_dim, 1)
```

先不要上复杂 attention。

### 8.4 特征标准化

status features 需要标准化：

- scaler 只能在 train 上 fit；
- valid / test 用 train scaler transform；
- known flag 不需要标准化；
- count 特征建议使用 `log1p`。

---

## 9. 指标

必须报告以下指标：

```text
F1_negative
PR_AUC_negative
F1_macro
AUROC
F1_bin
F1_weighted
```

其中主指标是：

```text
F1_negative
PR_AUC_negative
F1_macro
AUROC
```

不要只看正类 F1，因为 positive 是多数类。

---

## 10. 阈值策略

至少保留两种：

1. fixed threshold = 0.5；
2. threshold selected by validation F1_macro；
3. 可选：threshold selected by validation F1_negative。

优先报告：

```text
val_F1_macro threshold
```

同时输出 fixed 0.5 结果作为参考。

---

## 11. 输出目录

请新建实验目录：

```text
status_asymmetry_semba_experiment/
```

建议结构：

```text
status_asymmetry_semba_experiment/
├── py/
│   ├── build_online_status_features.py
│   ├── run_semba_status_decoder.py
│   ├── aggregate_status_decoder_results.py
│   └── audit_strict_online_status_features.py
├── md/
│   ├── status_asymmetry_semba_plan.md
│   ├── bitcoinotc_status_decoder_smoke_report.md
│   └── bitcoinotc_status_decoder_final_report.md
├── tables/
│   ├── status_feature_dictionary.csv
│   ├── status_feature_audit.csv
│   ├── status_decoder_metrics_seed42.csv
│   ├── status_decoder_metrics_summary.csv
│   └── status_decoder_ablation.csv
└── figures/
    ├── metric_comparison.png
    └── negative_class_comparison.png
```

如果项目已有类似目录，可以复用，但必须避免覆盖旧结果。

---

## 12. 必须生成的报告

### 12.1 计划报告

```text
status_asymmetry_semba_experiment/md/status_asymmetry_semba_plan.md
```

内容：

1. 实验目标；
2. 为什么选择 status/asymmetry-aware decoder；
3. 使用数据；
4. 模型对比；
5. 严格在线约束；
6. 主指标；
7. 成功/失败判断标准。

### 12.2 Smoke 报告

```text
status_asymmetry_semba_experiment/md/bitcoinotc_status_decoder_smoke_report.md
```

内容：

1. seed；
2. 数据集；
3. 模型；
4. 是否跑通；
5. 是否存在时间违规；
6. 主要指标；
7. 初步判断。

### 12.3 Final 报告

```text
status_asymmetry_semba_experiment/md/bitcoinotc_status_decoder_final_report.md
```

内容：

1. 5 seeds 汇总；
2. 与 SEMBA 对比；
3. 与 SEMBA-noprop 对比；
4. 与 TGN 对比；
5. 与 XGB-all 参照对比；
6. negative-class 指标；
7. 消融实验；
8. 是否支持 status/asymmetry-aware SEMBA；
9. 下一步是否扩展到 BitcoinAlpha-1。

---

## 13. 消融实验

最小消融：

| 模型 | 含义 |
|---|---|
| SEMBA | 原始模型 |
| SEMBA + source features | 只加源用户行为倾向 |
| SEMBA + target features | 只加目标用户信誉 |
| SEMBA + status gap | 只加 src/dst 差异 |
| SEMBA + all status features | 完整改进 |

优先观察：

1. 目标信誉是否最重要；
2. 源用户负边倾向是否提升 F1_negative；
3. status gap 是否提升 AUROC；
4. all status features 是否优于 SEMBA 原模型。

---

## 14. 成功判断标准

不要要求一定超过 XGB-all。
本轮核心是证明 SEMBA 内部改进是否有效。

### 强成功

满足以下任意两项：

1. SEMBA-status-decoder 的 F1_negative 高于 SEMBA，提升 ≥ 0.02；
2. SEMBA-status-decoder 的 PR_AUC_negative 高于 SEMBA，提升 ≥ 0.02；
3. SEMBA-status-decoder 的 F1_macro 高于 SEMBA，提升 ≥ 0.01；
4. SEMBA-status-decoder 的 AUROC 高于 SEMBA，提升 ≥ 0.01；
5. 5 seeds 下提升方向稳定。

### 中等成功

满足：

1. 负类指标提升，但总体 AUROC 不明显；
2. target feature 或 source feature 消融显示某一类 status 特征有效；
3. 结果能说明 status/asymmetry 信息对 SEMBA 有补充价值。

### 不成功

出现以下情况：

1. 所有指标均不如 SEMBA；
2. 只在单 seed 上提升；
3. 提升来自明显时间泄露；
4. 加入 status features 后模型只偏向预测负类，导致 F1_macro 下降。

若不成功，请输出原因分析，不要强行写成有效。

---

## 15. 最快执行步骤

请按以下顺序执行：

### Step 1：确认分支和环境

```powershell
git status
git branch --show-current
```

如果当前不在实验分支，请新建：

```powershell
git switch -c exp/status-asymmetry-semba
```

或在已有 worktree 下创建新实验目录。

### Step 2：查找现有代码

先搜索：

```powershell
dir strict_xgb_comparison_experiment\py
dir src
dir experiments
```

重点找：

```text
dynamic_features.py
strict_temporal_protocol.py
model_wrapper.py
src/semba.py
train.py
```

### Step 3：复用在线特征构造

优先从：

```text
strict_xgb_comparison_experiment/py/dynamic_features.py
```

提取或封装 online status features。

不要重复写一套互相不一致的特征逻辑。

### Step 4：实现 SEMBA-status-decoder

尽量新增文件，不要大幅修改原文件。

建议新增：

```text
status_asymmetry_semba_experiment/py/run_semba_status_decoder.py
```

必要时在 `model_wrapper.py` 中增加可选 decoder，不要破坏原 SEMBA 路径。

### Step 5：先跑 smoke

```powershell
python status_asymmetry_semba_experiment/py/run_semba_status_decoder.py --data BitcoinOTC-1 --seed 42 --smoke
```

如果脚本参数和项目实际不一致，请根据项目代码调整，但必须记录实际运行命令。

### Step 6：生成 smoke 报告

生成：

```text
status_asymmetry_semba_experiment/md/bitcoinotc_status_decoder_smoke_report.md
```

### Step 7：如果 smoke 有效，再跑 5 seeds

```powershell
python status_asymmetry_semba_experiment/py/run_semba_status_decoder.py --data BitcoinOTC-1 --seeds 42 43 44 45 46
```

### Step 8：汇总结果

```powershell
python status_asymmetry_semba_experiment/py/aggregate_status_decoder_results.py
```

### Step 9：输出最终报告

生成：

```text
status_asymmetry_semba_experiment/md/bitcoinotc_status_decoder_final_report.md
```

---

## 16. 严禁事项

1. 不要把当前事件写入 history 后再预测当前事件；
2. 不要用 paper protocol 作为主实验；
3. 不要只报告 F1_bin；
4. 不要用 test 调阈值；
5. 不要删除原始数据和 processed 缓存；
6. 不要覆盖已有 results；
7. 不要 force push；
8. 不要用 `git add .`；
9. 不要把大结果、模型权重、processed2 提交到 GitHub；
10. 不要把“是否超过 XGB”作为唯一成功标准；
11. 不要写“证明原论文泄露”作为本实验目标；
12. 不要强行把单 seed 小提升写成最终结论。

---

## 17. 最终交付物

本轮最小可行实验完成后，请交付：

1. 代码：
   - `build_online_status_features.py`
   - `run_semba_status_decoder.py`
   - `aggregate_status_decoder_results.py`
   - `audit_strict_online_status_features.py`

2. 表格：
   - `status_feature_dictionary.csv`
   - `status_feature_audit.csv`
   - `status_decoder_metrics_seed42.csv`
   - `status_decoder_metrics_summary.csv`
   - `status_decoder_ablation.csv`

3. 报告：
   - `status_asymmetry_semba_plan.md`
   - `bitcoinotc_status_decoder_smoke_report.md`
   - `bitcoinotc_status_decoder_final_report.md`

4. Git：
   - 一个独立实验分支；
   - 清晰 commit；
   - 不提交大文件。

---

## 18. 最终研究表述

如果实验有效，后续论文可以表述为：

> 在 SEMBA 的 signed memory 与 balanced aggregation 基础上，本文进一步引入严格在线的 status/asymmetry-aware decoder，将源用户行为倾向、目标用户历史信誉以及源-目标地位差异作为动态历史条件输入，用于增强动态有符号信任预测。实验重点评估该机制对少数类 distrust 识别、macro-F1 和 AUROC 的影响。

如果实验无效，论文也可以保留以下结论：

> 在当前 BitcoinOTC-1 严格在线任务中，status/asymmetry 特征虽在表格模型中有效，但直接并入 SEMBA decoder 未能稳定提升图模型表现，说明后续需要在 memory update 或 propagation gate 层面引入更深的结构不对称机制。
