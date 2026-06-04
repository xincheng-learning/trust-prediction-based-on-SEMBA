# XGB 模型严格对比实验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `BitcoinOTC-1` 上用统一时间协议、统一指标、多 seed、多 epoch 严格比较 XGB 历史统计模型与动态图模型，判断 XGB 是否有研究价值、资源优势和可结合方向。

**Architecture:** 先把评估协议独立出来，保证 XGB、TGN、SEMBA 都遵守“预测当前时间块之前不能看到当前时间块标签/边”的 online setting；再用统一指标、统一结果格式和统一聚合脚本输出最终表格。XGB 作为强历史统计 baseline，图模型作为动态图表示学习方法，二者比较时同时报告性能和资源成本。

**Tech Stack:** Python, PyTorch, PyTorch Geometric, XGBoost, scikit-learn, pandas, NumPy, Git worktree `exp/xgb-history`。

---

## 1. 实验核心问题

这次实验不是继续证明 `F1_weighted` 高，而是回答三个更严格的问题：

1. **性能问题：** 在严格 online 时间设定下，XGB 的负类识别、macro 指标、AUROC、PR-AUC 是否仍然有竞争力。
2. **资源问题：** 如果 XGB 性能接近图模型，但训练/推理时间显著更低，是否可以作为低资源强 baseline 或预筛模型。
3. **结合问题：** XGB 的历史统计特征是否能辅助 SEMBA/TGN，例如作为 pair classifier 的额外输入、hard-case reranker 或 early-exit 模块。

本实验不能只看 `F1_weighted`。BitcoinOTC 的正边比例高，`F1_weighted` 很容易被多数类抬高。

---

## 2. 固定数据协议

数据集固定为：

```text
dataset = BitcoinOTC-1
processed_dir = processed2
task = sign_class
val_ratio = 0.15
test_ratio = 0.15
full events = 35592
```

当前 `processed2` 的完整切分统计：

| split | events | positive | negative | positive rate |
| --- | ---: | ---: | ---: | ---: |
| full | 35592 | 32029 | 3563 | 0.899893 |
| train | 24915 | 23324 | 1591 | 0.936143 |
| val | 5338 | 4121 | 1217 | 0.772012 |
| test | 5339 | 4584 | 755 | 0.858588 |

实验只允许按时间顺序切分，不能随机打乱边。

---

## 3. 严格时间协议

统一采用：

```text
history_scope = global_online
strict_timestamp = true
```

含义：

1. 按完整时间流构建历史状态，再切分 train/val/test。
2. 对同一个 timestamp 的所有边，先用 timestamp 之前的历史生成特征/embedding 并预测。
3. 当前 timestamp 的所有边预测完成后，才能把这些边及其标签写入历史。
4. test 阶段可以使用 train 和 val 中已经发生过的历史，因为真实在线预测进入 test 时间段时，val 时间段已经是过去。
5. test 阶段不能使用 test 内部未来边；同一 timestamp 内也不能用已经预测过的同 timestamp 边更新历史。

这个协议对 BitcoinOTC-1 很重要，但影响不会像 BitcoinAlpha 那么极端。此前审计显示 BitcoinOTC-1 的 repeated timestamp fraction 约为 `0.0078`；即便如此，正式实验仍统一使用 strict timestamp，避免口径争议。

---

## 4. 当前代码风险与必须修正点

当前 `experiments/run_xgb_baseline.py` 已经支持：

```bash
python -m experiments.run_xgb_baseline ^
  --dataset BitcoinOTC-1 ^
  --processed_dir processed2 ^
  --max_events 35592 ^
  --history_scope global_online ^
  --strict_timestamp
```

但它还不适合作为最终严格实验脚本，因为当前脚本会把 train 和 val 拼起来训练 XGB。正式实验中，val 应该用于 early stopping、阈值选择和模型选择，不能直接并入训练集。

当前 `train.py` / `model_wrapper.py` 也不适合直接作为最终严格对比脚本。原因是图模型 forward 中存在同批次边先进入传播上下文再预测的风险：模型可能在预测当前 batch 时已经把当前 batch 的边写入了 `seen_ei`。正式对比必须改成：

```text
predict current timestamp block -> compute loss/metrics -> update memory/history
```

不能是：

```text
update current batch -> propagate with current batch -> predict current batch
```

因此，本实验第一步不是跑大实验，而是先实现严格 runner。

---

## 5. 模型组

最小完整模型组：

| group | model | purpose |
| --- | --- | --- |
| dummy | always_positive | 检测类别不平衡导致的虚高 |
| tabular | XGB-all | 使用全部在线历史统计特征 |
| tabular ablation | XGB-dst-reputation | 只使用目标节点历史声誉/被信任/入边特征 |
| tabular ablation | XGB-no-dst-reputation | 移除目标节点声誉和可见性特征 |
| graph | TGN | 动态图记忆模型基线 |
| graph | SEMBA | 论文主模型 |
| graph ablation | SEMBA-noprop | 检查无传播记忆模型是否已经足够 |

如果资源紧张，第一轮只跑：

```text
always_positive
XGB-all
XGB-dst-reputation
TGN
SEMBA
```

如果第一轮显示 XGB 和 SEMBA 接近，再补 `XGB-no-dst-reputation` 和 `SEMBA-noprop`。

---

## 6. 特征组定义

XGB-all 使用当前 `experiments/dynamic_features.py` 中的全部 `FEATURE_COLUMNS`。

XGB-dst-reputation 使用：

```text
dst_in_pos
dst_in_neg
dst_in_pos_ratio
q_dst
dst_trusted_in_count
dst_trusted_ratio
dst_visibility_log
dst_high_visibility
dst_visibility_share_of_max
```

XGB-no-dst-reputation 从 `FEATURE_COLUMNS` 中移除：

```text
dst_in_pos
dst_in_neg
dst_in_pos_ratio
q_dst
status_gap
dst_visibility_log
dst_trusted_in_count
dst_trusted_ratio
dst_high_visibility
dst_visibility_share_of_max
```

这样可以直接检验用户提出的“目标用户历史可见性、被信任程度、高可见节点”是否是核心贡献。

---

## 7. 多 seed 与多 epoch

正式 seed 固定为：

```text
42, 43, 44, 45, 46
```

低资源 pilot 使用：

```text
42, 43, 44
```

图模型 epoch 设定：

```text
pilot: 10 epochs
formal: max 50 epochs, early stopping patience 8
```

early stopping 监控指标：

```text
primary monitor = val PR_AUC_negative
secondary tie-breaker = val F1_macro
```

理由：负边是少数类，`PR_AUC_negative` 比 `F1_weighted` 更能检验模型是否真的识别不信任边。

XGB 设定：

```text
n_estimators = 1000
learning_rate = 0.03
max_depth = 4
subsample = 0.9
colsample_bytree = 0.9
reg_lambda = 2.0
early_stopping_rounds = 50
eval_set = train + val, with validation metric recorded separately
```

XGB 训练只 fit train；val 用于 early stopping 和阈值选择。

阈值选择：

```text
primary threshold = 在 val 上最大化 F1_macro 的阈值
secondary threshold = 0.5
```

最终 test 表格同时报告 primary threshold 和 fixed 0.5，避免阈值调优掩盖模型本身的排序能力。

---

## 8. 统一指标

每个模型、每个 seed、每个 split 都记录：

```text
positive_count
negative_count
positive_rate
predicted_positive_count
predicted_negative_count
true_negative
false_positive
false_negative
true_positive
accuracy
F1_weighted
F1_macro
F1_positive
F1_negative
balanced_accuracy
AUROC
PR_AUC_positive
PR_AUC_negative
threshold
```

论文表格主指标推荐顺序：

1. `PR_AUC_negative`
2. `F1_negative`
3. `F1_macro`
4. `balanced_accuracy`
5. `AUROC`
6. `F1_weighted`

`F1_weighted` 只作为辅助指标，不作为主要结论依据。

---

## 9. 资源指标

每次 run 记录：

```text
train_wall_time_sec
eval_wall_time_sec
total_wall_time_sec
peak_cpu_memory_mb
peak_gpu_memory_mb
device
num_parameters
model_artifact_size_mb
test_edges_per_second
```

资源比较表必须和性能表分开。XGB 的研究价值可能不体现在所有指标绝对第一，而体现在：

```text
接近图模型性能 + 显著更低训练时间/内存/推理成本
```

建议给出一个效率指标：

```text
efficiency_score = PR_AUC_negative / total_wall_time_sec
```

这个指标不作为论文唯一结论，但适合展示低资源优势。

---

## 10. 结果目录结构

所有大结果仍放在 ignored 的 `results/` 下：

```text
results/
  strict_compare/
    BitcoinOTC-1/
      manifest.json
      per_run_metrics.csv
      per_run_predictions.parquet
      summary_by_model.csv
      feature_importance/
        xgb_all_seed_42.csv
        xgb_all_seed_43.csv
      models/
        xgb_all_seed_42.json
        semba_seed_42.pt
```

Git 只追踪：

```text
xgb模型严格对比实验.md
experiments/*.py
docs/experiments/*summary*.md
```

Git 不追踪：

```text
results/
data/*/processed2/
*.pt
*.pkl
*.json model files under results/models
```

---

## 11. 需要新增或修改的文件

### A. `experiments/strict_metrics.py`

职责：

1. 统一计算分类指标。
2. 输出混淆矩阵。
3. 支持 `threshold` 参数。
4. 支持 positive 和 negative 两个 PR-AUC。

### B. `experiments/strict_temporal_protocol.py`

职责：

1. 生成 strict timestamp 时间块。
2. 提供 `iter_time_blocks(data)`。
3. 提供 `build_global_online_feature_frames(...)`。
4. 保证 XGB 和图模型共享同一时间块逻辑。

### C. `experiments/run_xgb_strict.py`

职责：

1. 只使用 train fit。
2. 使用 val 做 early stopping 和 threshold selection。
3. 在 test 上输出最终指标、预测概率和特征重要性。
4. 支持 `--feature_set all|dst_reputation|no_dst_reputation`。
5. 支持 `--seeds 42 43 44 45 46`。

### D. `experiments/run_graph_strict.py`

职责：

1. 支持 `--model tgn|semba|semba-noprop`。
2. 按 timestamp block 做 predict-before-update。
3. 每个 epoch 后在 val 上评估，保存最佳 checkpoint。
4. 在 test 上用最佳 checkpoint 输出同一套指标。
5. 保存每条 test 边的 `src,dst,t,y_true,y_prob,y_pred,seed,model`。

### E. `experiments/aggregate_strict_compare.py`

职责：

1. 读取所有 per-run metrics。
2. 输出 mean/std。
3. 计算每个模型相对 always-positive、XGB-all、SEMBA 的 delta。
4. 输出 `summary_by_model.csv` 和 Markdown 表格。

---

## 12. 实施步骤

### Phase 0: 协议确认

- [ ] 确认当前分支是 `exp/xgb-history`。

```powershell
git status -sb
```

期望看到：

```text
## exp/xgb-history...
```

- [ ] 确认 processed2 可读。

```powershell
& 'C:\ProgramData\anaconda3\python.exe' - <<'PY'
from utils import get_data
data, train_data, val_data, test_data = get_data(
    'BitcoinOTC-1',
    './data/BitcoinOTC-1',
    'cpu',
    processed_dir='processed2',
)
print(data.num_events, train_data.num_events, val_data.num_events, test_data.num_events)
PY
```

期望输出：

```text
35592 24915 5338 5339
```

### Phase 1: 实现严格指标与时间块协议

- [ ] 新增 `experiments/strict_metrics.py`。
- [ ] 新增 `experiments/strict_temporal_protocol.py`。
- [ ] 写最小测试：同一 timestamp 的边生成特征时不能互相看见。
- [ ] 运行测试或小脚本验证。
- [ ] 提交：

```powershell
git add experiments/strict_metrics.py experiments/strict_temporal_protocol.py
git commit -m "test: add strict temporal metrics protocol"
```

### Phase 2: 实现 XGB 严格 runner

- [ ] 新增 `experiments/run_xgb_strict.py`。
- [ ] 支持 feature set 消融。
- [ ] 支持多 seed。
- [ ] 支持 val threshold selection。
- [ ] 运行 pilot：

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m experiments.run_xgb_strict ^
  --dataset BitcoinOTC-1 ^
  --processed_dir processed2 ^
  --max_events 20000 ^
  --seeds 42 43 44 ^
  --feature_sets all dst_reputation no_dst_reputation ^
  --results_dir results/strict_compare_pilot
```

- [ ] 提交：

```powershell
git add experiments/run_xgb_strict.py
git commit -m "feat: add strict xgb comparison runner"
```

### Phase 3: 实现图模型严格 runner

- [ ] 新增 `experiments/run_graph_strict.py`。
- [ ] 先支持 `tgn`。
- [ ] 再支持 `semba`。
- [ ] 最后支持 `semba-noprop`。
- [ ] 验证 predict-before-update：当前 timestamp block 预测前，当前 block 不能进入 `seen_ei`、memory 或 neighbor loader。
- [ ] 运行 pilot：

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m experiments.run_graph_strict ^
  --dataset BitcoinOTC-1 ^
  --processed_dir processed2 ^
  --max_events 20000 ^
  --models tgn semba ^
  --seeds 42 43 44 ^
  --num_epochs 10 ^
  --batch_size 1000 ^
  --embedding_dim 64 ^
  --device cpu ^
  --results_dir results/strict_compare_pilot
```

- [ ] 提交：

```powershell
git add experiments/run_graph_strict.py
git commit -m "feat: add strict graph comparison runner"
```

### Phase 4: 聚合 pilot 结果

- [ ] 新增 `experiments/aggregate_strict_compare.py`。
- [ ] 聚合 XGB 与图模型 pilot。
- [ ] 生成：

```text
results/strict_compare_pilot/BitcoinOTC-1/per_run_metrics.csv
results/strict_compare_pilot/BitcoinOTC-1/summary_by_model.csv
docs/experiments/bitcoinotc_strict_compare_pilot_summary.md
```

- [ ] 检查 XGB 是否明显强于 always-positive。
- [ ] 检查图模型是否因为 epoch 太少表现不稳定。
- [ ] 提交：

```powershell
git add experiments/aggregate_strict_compare.py docs/experiments/bitcoinotc_strict_compare_pilot_summary.md
git commit -m "docs: summarize bitcoinotc strict comparison pilot"
```

### Phase 5: 正式 full run

只有 Phase 4 没有协议错误时才进入正式实验。

正式命令：

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m experiments.run_xgb_strict ^
  --dataset BitcoinOTC-1 ^
  --processed_dir processed2 ^
  --max_events 35592 ^
  --seeds 42 43 44 45 46 ^
  --feature_sets all dst_reputation no_dst_reputation ^
  --results_dir results/strict_compare
```

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m experiments.run_graph_strict ^
  --dataset BitcoinOTC-1 ^
  --processed_dir processed2 ^
  --max_events 35592 ^
  --models tgn semba semba-noprop ^
  --seeds 42 43 44 45 46 ^
  --num_epochs 50 ^
  --early_stop_patience 8 ^
  --batch_size 1000 ^
  --embedding_dim 64 ^
  --device cpu ^
  --results_dir results/strict_compare
```

如果机器有可用 CUDA，可以另开一组 GPU 实验，但 CPU 与 GPU 结果必须分表展示，不能混在同一张效率表里。

### Phase 6: 最终报告

- [ ] 聚合正式结果：

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m experiments.aggregate_strict_compare ^
  --results_dir results/strict_compare ^
  --dataset BitcoinOTC-1 ^
  --out_md docs/experiments/bitcoinotc_strict_compare_final_summary.md
```

- [ ] 最终报告必须包含：

```text
数据切分统计
协议说明
每模型 mean/std
每 seed 原始表
混淆矩阵均值
负类 PR-AUC/F1
训练时间/推理时间/内存
XGB 特征重要性均值
XGB 消融结果
相对 SEMBA/TGN 的差距
是否值得继续做融合模型
```

- [ ] 提交最终摘要：

```powershell
git add docs/experiments/bitcoinotc_strict_compare_final_summary.md
git commit -m "docs: summarize strict bitcoinotc model comparison"
```

---

## 13. 统计汇报方式

最终表格格式：

| model | seeds | PR_AUC_negative | F1_negative | F1_macro | balanced_accuracy | AUROC | F1_weighted | train sec | eval sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| always_positive | 5 | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std |
| XGB-all | 5 | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std |
| XGB-dst-reputation | 5 | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std |
| TGN | 5 | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std |
| SEMBA | 5 | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std |

同时输出 paired delta：

```text
XGB-all - always_positive
XGB-all - TGN
XGB-all - SEMBA
SEMBA - TGN
```

如果保存了 per-edge predictions，再做 1000 次 paired bootstrap，随机种子固定为 `20260604`，给出主要指标 delta 的 95% CI。

---

## 14. 判断标准

XGB 有研究价值的最低证据：

1. `XGB-all` 明显超过 `always_positive`：
   - `F1_negative > 0`
   - `balanced_accuracy > 0.5`
   - `PR_AUC_negative` 明显高于 test 负类比例
2. `XGB-all` 在 full BitcoinOTC 上不是只靠 `F1_weighted` 赢。
3. `XGB-dst-reputation` 接近 `XGB-all` 或显著强于 `XGB-no-dst-reputation`，说明用户提出的目标节点历史声誉/可见性特征确实关键。
4. 如果 `XGB-all` 性能低于 SEMBA，但训练/推理成本低一个数量级，仍可作为低资源 baseline 和 early-exit 模型。
5. 如果 `XGB-all` 在 `PR_AUC_negative`、`F1_negative` 或 `balanced_accuracy` 上接近或超过 SEMBA/TGN，则值得写成强 baseline 发现。

不允许的结论：

```text
只因为 F1_weighted 高，就说 XGB 优于图模型。
只因为 1 epoch smoke 低，就说 SEMBA/TGN 不如 XGB。
只因为单 seed 结果好，就说模型稳定。
```

---

## 15. 融合模型后续方向

如果正式比较后 XGB 仍有优势或接近图模型，优先做以下两个结合方向。

### Direction A: XGB early-exit + graph hard cases

流程：

```text
XGB 先预测所有边
高置信样本直接输出
低置信样本交给 SEMBA/TGN
```

高置信定义：

```text
p <= 0.1 or p >= 0.9
```

需要报告：

```text
调用图模型比例
总推理时间
整体 PR_AUC_negative / F1_negative / F1_macro
```

这条路线最符合“节约资源”目标。

### Direction B: XGB features + SEMBA classifier

流程：

```text
SEMBA 生成 src/dst embedding
拼接 XGB 在线历史特征
pair classifier 输入 = [z_src, z_dst, xgb_features]
```

这条路线检验 XGB 特征是否补充图模型没有显式编码的声誉/可见性信号。

暂不优先做 `SEMBA embeddings + XGB`，因为它需要先稳定训练 SEMBA，再导出 embedding，工程成本更高。

---

## 16. Git 与实验管理

本实验继续在：

```text
E:\导师之任\导师之任\社交信任\semba-main\.worktrees\exp-xgb-history
branch = exp/xgb-history
```

推荐 commit 节奏：

```text
test: add strict temporal metrics protocol
feat: add strict xgb comparison runner
feat: add strict graph comparison runner
docs: summarize bitcoinotc strict comparison pilot
docs: summarize strict bitcoinotc model comparison
```

不推送 GitHub，除非用户单独确认。

---

## 17. 本方案的执行顺序

推荐下一步从 Phase 1 开始，不直接跑正式实验。

最小安全路径：

```text
Phase 1 strict metrics/protocol
Phase 2 XGB strict runner
Phase 3 graph strict runner
Phase 4 pilot on max_events=20000, 3 seeds, 10 epochs
Phase 5 full BitcoinOTC, 5 seeds, 50 epochs
Phase 6 final report and decide fusion direction
```

这样可以先用较低成本发现协议问题，再把算力花在真正可信的正式实验上。
