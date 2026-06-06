# 论文二分类协议下 XGB 对比 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 本计划只定义实验任务明细；执行模型训练前必须等待用户确认。

**Goal:** 在 SEMBA 论文的二分类任务协议下，对 `BitcoinOTC-1` 上的 XGB 历史统计模型、原论文 SEMBA/TGN 报告值、以及本地旧代码口径 SEMBA/TGN 进行可追溯对比。

**Architecture:** 新增一套独立的 `paper_protocol` 实验脚本，不复用上一轮严格无泄漏 runner 的评估口径。XGB 特征构造将刻意模拟旧 `train.py` 的 `to_update=True` batch-inclusive 行为；图模型 runner 将保留原 SEMBA/TGN 模型结构和旧代码的先更新再预测协议。所有输出写入新的 `results/paper_protocol_sign_class/BitcoinOTC-1/`，不覆盖上一轮 `strict_compare` 结果。

**Tech Stack:** Python, PyTorch, PyTorch Geometric, XGBoost, scikit-learn, pandas, Windows PowerShell, Anaconda env `C:\ProgramData\anaconda3\envs\pytorch\python.exe` when CUDA is available.

---

## 0. 这次实验和上一轮严格实验的区别

上一轮实验是严格在线预测：先预测当前事件，再把当前事件写入历史。它主要回答“无泄漏条件下，XGB 历史特征是否强”。

这次实验刻意改成论文/旧代码口径：复刻原 `train.py` 中 `to_update=True` 的行为。也就是说，当前 batch 会先进入模型 memory/history，再预测当前 batch。这个协议可能存在信息泄漏，但它更接近论文代码的实际实验方式，因此适合回答：

> 如果我们按论文二分类协议跑，XGB 表格模型和论文 SEMBA/TGN 的表现相比如何？

必须在最终报告中同时写明两点：

1. 这次结果可以和论文 Table 6 的二分类 sign prediction 口径做近似对比。
2. 这次结果不能替代上一轮严格无泄漏结论，因为两者协议不同。

---

## 1. 指标缩写中文解释

本实验默认标签为：

- `1`：正边，表示信任、正向评分。
- `0`：负边，表示不信任、负向评分。

| 缩写 | 中文名 | 简单公式或解释 | 本实验用途 |
| --- | --- | --- | --- |
| `AUROC` | ROC 曲线下面积 | 可以近似理解为 `P(随机正边得分 > 随机负边得分)` | 论文 Table 6 主指标之一，衡量概率排序能力 |
| `F1_bin` | 二分类正类 F1 | `F1 = 2 * Precision * Recall / (Precision + Recall)`，这里默认正类是 `1` | 对齐旧 `train.py` 里 `f1_score(true, pred)` 的输出，近似对齐论文 Table 6 的 `F1` |
| `F1_positive` | 正边 F1 | 和 `F1_bin` 基本一致，正类为信任边 | 看多数类正边预测是否好 |
| `F1_negative` | 负边 F1 | 把负边 `0` 当成关注类后计算 F1 | 看模型是否真的能识别不信任边 |
| `F1_macro` | 宏平均 F1 | `(F1_positive + F1_negative) / 2` | 类别不平衡时更公平 |
| `F1_weighted` / `F1_wt` | 加权 F1 | 每一类 F1 按该类样本数加权平均 | 旧代码输出之一，但容易被多数正边影响 |
| `PR_AUC_negative` | 负类 PR-AUC | 把负边当成目标类，计算 precision-recall 曲线面积 | 负边是少数类，因此这是关键辅助指标 |
| `Acc` | 准确率 | `(TP + TN) / 全部样本数` | 只作辅助，类别不平衡时容易误导 |
| `TP` | 真正例 | 真实正边且预测为正边 | 混淆矩阵解释 |
| `TN` | 真负例 | 真实负边且预测为负边 | 负边命中数量 |
| `FP` | 假正例 | 真实负边但预测为正边 | 把不信任误判成信任 |
| `FN` | 假负例 | 真实正边但预测为负边 | 把信任误判成不信任 |
| `total sec` | 总耗时 | 训练耗时 + 评估耗时，不含原始数据下载 | 看资源消耗 |

最终结论中，论文对比主看 `F1_bin` 和 `AUROC`；研究判断还必须看 `F1_negative`、`F1_macro` 和 `PR_AUC_negative`。

---

## 2. 文件结构

### 计划执行后应新增的文件

- Create: `tests/test_paper_protocol_features.py`
  - 用小型 toy events 验证 XGB 的 paper protocol 特征确实是 batch-inclusive。

- Create: `experiments/paper_protocol_features.py`
  - 负责构造论文/旧代码口径下的 XGB 历史特征。
  - 和上一轮 `experiments/dynamic_features.py` 分开，避免混淆严格无泄漏特征和论文协议特征。

- Create: `experiments/run_xgb_paper_protocol.py`
  - 负责 XGB 小范围调参、5 seeds 训练、验证集选参、测试集输出。
  - 默认输出到 `results/paper_protocol_sign_class/BitcoinOTC-1/`。

- Create: `experiments/run_graph_paper_protocol.py`
  - 负责本地 SEMBA/TGN 二分类旧代码口径复现实验。
  - 保留 `to_update=True`，保留 `sign_class`，输出结构化 CSV。

- Create: `experiments/aggregate_paper_protocol_compare.py`
  - 聚合 XGB、本地 SEMBA/TGN、论文 Table 6 报告值。

- Create: `strict_xgb_comparison_experiment/bitcoinotc_paper_protocol_sign_class_summary.md`
  - 最终中文报告。

### 原则上不修改的文件

- Do not modify: `train.py`
  - 原始旧代码保留作为事实依据。

- Do not modify: `data/*/raw`
  - 用户明确禁止删除或改动。

- Do not modify/delete: `data/*/processed`
  - 用户明确禁止删除。

### 可以读取但不覆盖的目录

- Read: `data/BitcoinOTC-1/processed`
- Read: `data/BitcoinOTC-1/processed2`

如果 `processed` 在当前 PyG 版本无法加载，则使用 `processed2`，并在 manifest 中写明原因。

---

## 3. 实验协议定义

### 3.1 论文二分类任务

任务名：Dynamic Link Sign Prediction。

项目参数：

```text
--task sign_class
```

含义：已知未来某个 `(src, dst, time)` 有边，只预测这条边是正边还是负边。

论文 Table 6 中 BTC-Otc 的参考值：

| model | paper F1 | paper AUROC |
| --- | ---: | ---: |
| TGN | 0.74 +/- 0.02 | 0.82 +/- 0.03 |
| SEMBA | 0.81 +/- 0.04 | 0.79 +/- 0.02 |

这些值只作为外部参考行写入最终报告，不重新声称自己复现了论文环境。

### 3.2 本地旧代码口径

从当前代码确认：

- `train.py` 的 `train()` 和 `test()` 中都调用模型 forward，并传入 `to_update=True`。
- `utils.seq_batches()` 使用连续 batch，Bitcoin 数据默认 batch size 为 `1000`。
- `best_run_details.csv` 中 `BitcoinOTC-1 + sign_class + semba/tgn` 的推荐设置为 `num_epochs=15, lr_init=0.01`。

本实验主图模型设置：

```text
dataset      = BitcoinOTC-1
task         = sign_class
batch_size   = 1000
num_epochs   = 15
lr_init      = 0.01
feat_type    = zeros
num_feats    = 8
embedding_dim= 64
seeds        = 42 43 44 45 46
device       = cuda if available, else cpu
processed_dir= processed2 unless processed loads cleanly and matches counts
```

说明：`src/config/BitcoinOTC-1.yaml` 写有 `embedding_dim=512, num_epochs=30`，但 `run_baselines.sh` 和 `best_run_details.csv` 的实际命令没有显式传入 `embedding_dim`，因此会使用 `parser.py` 默认 `64`。主实验先按旧脚本实际命令口径跑；如果用户同意，再把 `embedding_dim=512, num_epochs=30` 作为二级诊断实验。

### 3.3 XGB 的论文协议特征

上一轮严格特征是：

```text
先读历史 -> 生成当前边特征 -> 再把当前边写入历史
```

这次 paper protocol XGB 特征应模拟旧图模型：

```text
先把当前 batch 的所有边写入历史 -> 再为当前 batch 的每条边生成特征
```

这意味着当前 batch 里的边会影响当前 batch 的特征。它可能泄漏，但正是本次要对齐论文/旧代码口径的地方。

split 边界必须和旧 `train.py` 一致：

1. 先处理 train split，batch 从 train 内部重新开始切。
2. 再处理 val split，继承 train 历史，batch 从 val 内部重新开始切。
3. 最后处理 test split，继承 train + val 历史，batch 从 test 内部重新开始切。

不要用一个全数据流直接按 1000 切 batch，因为那可能让 batch 跨过 train/val/test 边界，和旧代码不一致。

---

## 4. Task 1: 写 paper protocol 特征单元测试

**Files:**

- Create: `tests/test_paper_protocol_features.py`
- Create if missing: `tests/__init__.py`

- [ ] **Step 1: 创建测试文件**

测试目标：证明 batch-inclusive 特征会先吸收当前 batch，再给当前 batch 生成特征。

```python
from experiments.paper_protocol_features import build_rows_for_split_paper_batch


def test_paper_batch_features_include_current_batch_before_prediction():
    events = [
        {"src": 1, "dst": 10, "t": 1, "y": 1},
        {"src": 2, "dst": 10, "t": 2, "y": 0},
        {"src": 3, "dst": 10, "t": 3, "y": 1},
    ]

    state = None
    rows, state = build_rows_for_split_paper_batch(events, state=state, batch_size=2)

    assert len(rows) == 3

    # 前两个事件在同一个 batch。
    # 如果是严格无泄漏，第一行 dst_in_pos/dst_in_neg 都应为 0。
    # 但 paper protocol 是先更新当前 batch，再预测当前 batch，
    # 所以第一行和第二行都能看到当前 batch 内的一正一负。
    assert rows[0]["dst_in_pos"] == 1.0
    assert rows[0]["dst_in_neg"] == 1.0
    assert rows[1]["dst_in_pos"] == 1.0
    assert rows[1]["dst_in_neg"] == 1.0

    # 第三个事件单独在第二个 batch。
    # 生成第三行特征前，它自己这条正边已经写入历史。
    assert rows[2]["dst_in_pos"] == 2.0
    assert rows[2]["dst_in_neg"] == 1.0


def test_paper_batch_state_continues_across_splits_but_batches_reset():
    train_events = [
        {"src": 1, "dst": 10, "t": 1, "y": 1},
        {"src": 2, "dst": 10, "t": 2, "y": 0},
    ]
    val_events = [
        {"src": 3, "dst": 10, "t": 3, "y": 1},
    ]

    train_rows, state = build_rows_for_split_paper_batch(train_events, state=None, batch_size=1000)
    val_rows, state = build_rows_for_split_paper_batch(val_events, state=state, batch_size=1000)

    assert train_rows[0]["dst_in_pos"] == 1.0
    assert train_rows[0]["dst_in_neg"] == 1.0

    # val 继承 train 历史，并且 val 当前 batch 自己也先写入。
    assert val_rows[0]["dst_in_pos"] == 2.0
    assert val_rows[0]["dst_in_neg"] == 1.0
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run:

```powershell
& 'C:\ProgramData\anaconda3\envs\pytorch\python.exe' -m unittest tests.test_paper_protocol_features -v
```

Expected:

```text
ModuleNotFoundError: No module named 'experiments.paper_protocol_features'
```

---

## 5. Task 2: 实现 paper protocol XGB 特征构造

**Files:**

- Create: `experiments/paper_protocol_features.py`

- [ ] **Step 1: 实现可复用的状态对象和 split 级特征函数**

实现要求：

- 复用上一轮 XGB 的特征列名，导入 `FEATURE_COLUMNS`。
- 每个 split 内按 `batch_size` 切 batch。
- 对每个 batch：先 `update_history(event)`，再 `append_feature_row(event)`。
- 返回 `(rows, state)`，让 train -> val -> test 能连续继承历史。

核心实现轮廓：

```python
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import log1p
from typing import Dict, Iterable, List, MutableMapping, Set, Tuple

import pandas as pd

from experiments.dynamic_features import FEATURE_COLUMNS, events_from_temporal_data


Event = Dict[str, int]
FeatureRow = Dict[str, float]


@dataclass
class PaperHistoryState:
    out_pos: MutableMapping[int, int] = field(default_factory=lambda: defaultdict(int))
    out_neg: MutableMapping[int, int] = field(default_factory=lambda: defaultdict(int))
    in_pos: MutableMapping[int, int] = field(default_factory=lambda: defaultdict(int))
    in_neg: MutableMapping[int, int] = field(default_factory=lambda: defaultdict(int))
    pair_pos: MutableMapping[Tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))
    pair_neg: MutableMapping[Tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))
    neighbors: MutableMapping[int, Set[int]] = field(default_factory=lambda: defaultdict(set))
    last_time: Dict[int, int] = field(default_factory=dict)
    seen_nodes: Set[int] = field(default_factory=set)
    node_visibility: MutableMapping[int, int] = field(default_factory=lambda: defaultdict(int))
    max_visibility_seen: int = 0


def build_rows_for_split_paper_batch(
    events: Iterable[Event],
    state: PaperHistoryState | None,
    batch_size: int,
    high_visibility_min_events: int = 10,
) -> tuple[list[FeatureRow], PaperHistoryState]:
    if state is None:
        state = PaperHistoryState()
    rows: list[FeatureRow] = []
    sorted_events = sorted(events, key=lambda row: int(row["t"]))

    for start in range(0, len(sorted_events), batch_size):
        batch = sorted_events[start:start + batch_size]
        for event in batch:
            update_history(state, event)
        for event in batch:
            rows.append(make_feature_row(state, event, high_visibility_min_events))

    return rows, state
```

还需要在同一文件中完整实现 `ratio()`、`update_history()`、`make_feature_row()`。三者字段计算必须和 `experiments/dynamic_features.py` 的现有 XGB 特征保持同名同义，只改变“先更新 batch、再生成特征”的协议顺序。

- [ ] **Step 2: 实现 train/val/test frame 构造函数**

接口：

```python
def paper_protocol_feature_frames(data, train_data, val_data, test_data, batch_size: int):
    state = PaperHistoryState()
    train_rows, state = build_rows_for_split_paper_batch(
        events_from_temporal_data(train_data), state, batch_size=batch_size
    )
    val_rows, state = build_rows_for_split_paper_batch(
        events_from_temporal_data(val_data), state, batch_size=batch_size
    )
    test_rows, state = build_rows_for_split_paper_batch(
        events_from_temporal_data(test_data), state, batch_size=batch_size
    )
    return (
        pd.DataFrame(train_rows)[FEATURE_COLUMNS + ["label"]],
        pd.DataFrame(val_rows)[FEATURE_COLUMNS + ["label"]],
        pd.DataFrame(test_rows)[FEATURE_COLUMNS + ["label"]],
    )
```

- [ ] **Step 3: 运行测试，确认通过**

Run:

```powershell
& 'C:\ProgramData\anaconda3\envs\pytorch\python.exe' -m unittest tests.test_paper_protocol_features -v
```

Expected:

```text
OK
```

---

## 6. Task 3: 实现 XGB paper protocol runner

**Files:**

- Create: `experiments/run_xgb_paper_protocol.py`

- [ ] **Step 1: 新建 runner 参数**

必须支持：

```text
--dataset BitcoinOTC-1
--processed_dir processed2
--results_dir results/paper_protocol_sign_class
--batch_size 1000
--seeds 42 43 44 45 46
--feature_sets all dst_reputation no_dst_reputation
--xgb_device auto
--n_jobs 4
```

- [ ] **Step 2: 加载数据并写 manifest**

manifest 必须包含：

```json
{
  "runner": "run_xgb_paper_protocol",
  "task": "sign_class",
  "protocol": "paper_batch_inclusive_to_update_true_approximation",
  "leakage_note": "Current batch is written into history before features are generated, matching the old SEMBA train.py to_update=True style.",
  "dataset": "BitcoinOTC-1",
  "processed_dir": "processed2",
  "batch_size": 1000,
  "seeds": [42, 43, 44, 45, 46]
}
```

- [ ] **Step 3: 实现简单 XGB 超参数搜索**

只在 seed `42` 上搜索，不做大范围搜索。

搜索空间：

```python
param_grid = [
    {"n_estimators": n, "max_depth": d, "learning_rate": lr, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 2.0}
    for n in [300, 600]
    for d in [3, 4, 5]
    for lr in [0.03, 0.05]
]
```

选择两个配置：

1. `XGB-paper-selected`：验证集 `F1_bin` 最大，若并列则 `AUROC` 最大。
2. `XGB-negative-aware`：验证集 `F1_macro` 最大，若并列则 `PR_AUC_negative` 最大。

这样可以同时回答两个问题：

- 按论文 F1/AUROC 口径，XGB 能否接近或超过 SEMBA？
- 按负边识别口径，XGB 是否仍然有研究价值？

- [ ] **Step 4: GPU 优先策略**

XGBoost 2.x 优先尝试：

```python
XGBClassifier(
    tree_method="hist",
    device="cuda",
    objective="binary:logistic",
    eval_metric="logloss",
)
```

如果当前 XGBoost 不支持 GPU 或 CUDA 报错，自动 fallback：

```python
XGBClassifier(
    tree_method="hist",
    device="cpu",
    objective="binary:logistic",
    eval_metric="logloss",
)
```

必须把实际使用设备写入 `per_run_metrics.csv` 的 `device` 列。

- [ ] **Step 5: 输出 CSV**

输出目录：

```text
results/paper_protocol_sign_class/BitcoinOTC-1/
```

必须生成：

```text
manifest_xgb_paper_protocol.json
xgb_grid_search.csv
per_run_metrics.csv
per_run_predictions.csv
feature_importance/xgb_<model>_<feature_set>_seed_<seed>.csv
```

`per_run_metrics.csv` 至少包含：

```text
dataset, model, feature_set, seed, threshold_mode,
AUROC, F1_bin, F1_positive, F1_negative, F1_macro, F1_weighted,
PR_AUC_negative, balanced_accuracy,
TN, FP, FN, TP,
train_wall_time_sec, eval_wall_time_sec, total_wall_time_sec, device
```

- [ ] **Step 6: smoke run**

Run:

```powershell
& 'C:\ProgramData\anaconda3\envs\pytorch\python.exe' -m experiments.run_xgb_paper_protocol --dataset BitcoinOTC-1 --processed_dir processed2 --max_events 5000 --results_dir results/paper_protocol_sign_class_smoke --seeds 42 --feature_sets all --batch_size 1000
```

Expected:

```text
"runner": "run_xgb_paper_protocol"
"rows_written": should be an integer greater than 0
```

---

## 7. Task 4: 实现本地图模型 paper protocol runner

**Files:**

- Create: `experiments/run_graph_paper_protocol.py`

- [ ] **Step 1: 复制旧代码核心协议，不改模型结构**

必须保留：

```python
z = model(
    x,
    pos_ei_batch,
    neg_ei_batch,
    t_pos,
    t_neg,
    weight_pos,
    weight_neg,
    to_update=True,
)
```

这个 runner 的目的不是修复泄漏，而是复刻旧论文代码口径。

- [ ] **Step 2: 支持 SEMBA 和 TGN**

参数：

```text
--models semba tgn
--task sign_class
--dataset BitcoinOTC-1
--processed_dir processed2
--batch_size 1000
--num_epochs 15
--lr_init 0.01
--embedding_dim 64
--seeds 42 43 44 45 46
--device cuda
```

- [ ] **Step 3: 复刻旧代码 stateful 流程**

每个 seed：

1. `model.reset_memory()`
2. train split 内按 batch 训练，`to_update=True`
3. val split 内按 batch 评估，`to_update=True`
4. 每个 epoch 重复 1-3
5. 最后一个 epoch 的 val 结束后，不重置 memory
6. test split 内按 batch 评估，`to_update=True`

这样与旧 `train.py` 的状态流尽量一致。

- [ ] **Step 4: 输出和 XGB 一致的指标**

对 `sign_class` 输出：

```text
AUROC
F1_bin
F1_positive
F1_negative
F1_macro
F1_weighted
PR_AUC_negative
balanced_accuracy
TN, FP, FN, TP
```

其中 `F1_bin` 必须等价于：

```python
f1_score(true, pred)
```

也就是旧 `train.py` 的 `F1_bin`。

- [ ] **Step 5: smoke run**

Run:

```powershell
& 'C:\ProgramData\anaconda3\envs\pytorch\python.exe' -m experiments.run_graph_paper_protocol --dataset BitcoinOTC-1 --processed_dir processed2 --models semba --max_events 5000 --num_epochs 2 --seeds 42 --device cuda --results_dir results/paper_protocol_sign_class_smoke
```

Expected:

```text
"runner": "run_graph_paper_protocol"
"models": ["semba"]
```

如果 CUDA 不可用，命令自动改为：

```powershell
& 'C:\ProgramData\anaconda3\envs\pytorch\python.exe' -m experiments.run_graph_paper_protocol --dataset BitcoinOTC-1 --processed_dir processed2 --models semba --max_events 5000 --num_epochs 2 --seeds 42 --device cpu --results_dir results/paper_protocol_sign_class_smoke
```

---

## 8. Task 5: 运行正式实验

**Files:**

- Output only: `results/paper_protocol_sign_class/BitcoinOTC-1/*`

执行前必须等待用户确认。

- [ ] **Step 1: XGB 正式实验**

Run:

```powershell
& 'C:\ProgramData\anaconda3\envs\pytorch\python.exe' -m experiments.run_xgb_paper_protocol --dataset BitcoinOTC-1 --processed_dir processed2 --results_dir results/paper_protocol_sign_class --seeds 42 43 44 45 46 --feature_sets all dst_reputation no_dst_reputation --batch_size 1000 --xgb_device auto --n_jobs 4
```

Expected files:

```text
results/paper_protocol_sign_class/BitcoinOTC-1/manifest_xgb_paper_protocol.json
results/paper_protocol_sign_class/BitcoinOTC-1/xgb_grid_search.csv
results/paper_protocol_sign_class/BitcoinOTC-1/per_run_metrics.csv
```

- [ ] **Step 2: SEMBA/TGN 本地 paper protocol 实验**

Run:

```powershell
& 'C:\ProgramData\anaconda3\envs\pytorch\python.exe' -m experiments.run_graph_paper_protocol --dataset BitcoinOTC-1 --processed_dir processed2 --results_dir results/paper_protocol_sign_class --models semba tgn --task sign_class --batch_size 1000 --num_epochs 15 --lr_init 0.01 --embedding_dim 64 --seeds 42 43 44 45 46 --device cuda
```

Expected files:

```text
results/paper_protocol_sign_class/BitcoinOTC-1/manifest_graph_paper_protocol.json
results/paper_protocol_sign_class/BitcoinOTC-1/per_run_metrics.csv
results/paper_protocol_sign_class/BitcoinOTC-1/per_run_predictions.csv
```

- [ ] **Step 3: 如果 CUDA 不可用，记录 CPU fallback**

Run:

```powershell
& 'C:\ProgramData\anaconda3\envs\pytorch\python.exe' -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Expected on this machine:

```text
True
NVIDIA GeForce RTX 3060 Ti
```

---

## 9. Task 6: 聚合报告

**Files:**

- Create: `experiments/aggregate_paper_protocol_compare.py`
- Create: `strict_xgb_comparison_experiment/bitcoinotc_paper_protocol_sign_class_summary.md`

- [ ] **Step 1: 聚合 per-run metrics**

Run:

```powershell
& 'C:\ProgramData\anaconda3\envs\pytorch\python.exe' -m experiments.aggregate_paper_protocol_compare --results_dir results/paper_protocol_sign_class --dataset BitcoinOTC-1 --out_md strict_xgb_comparison_experiment/bitcoinotc_paper_protocol_sign_class_summary.md
```

Expected files:

```text
results/paper_protocol_sign_class/BitcoinOTC-1/summary_by_model.csv
strict_xgb_comparison_experiment/bitcoinotc_paper_protocol_sign_class_summary.md
```

- [ ] **Step 2: 最终报告必须包含这些表**

表 1：论文 Table 6 外部参考值。

| model | source | F1_bin | AUROC |
| --- | --- | ---: | ---: |
| TGN | paper Table 6 BTC-Otc | 0.74 +/- 0.02 | 0.82 +/- 0.03 |
| SEMBA | paper Table 6 BTC-Otc | 0.81 +/- 0.04 | 0.79 +/- 0.02 |

表 2：本地 paper protocol 结果，5 seeds 均值。

必须包含：

```text
model, F1_bin, AUROC, F1_negative, F1_macro, PR_AUC_negative, total sec
```

表 3：XGB 特征组消融。

必须包含：

```text
XGB-all
XGB-dst_reputation
XGB-no_dst_reputation
always_positive
```

表 4：混淆矩阵均值。

必须包含：

```text
TN, FP, FN, TP
```

- [ ] **Step 3: 最终报告必须回答这些问题**

1. 是否真的使用了 `sign_class` 二分类任务？
2. 是否使用了 70/15/15 时间切分？
3. 是否使用了和旧代码一致的 `to_update=True` paper protocol？
4. XGB 是否使用了 batch-inclusive 特征？
5. 本地 SEMBA/TGN 是否接近论文 Table 6？
6. 如果本地 SEMBA/TGN 没接近论文，是环境、超参、processed 数据、还是实现协议导致？只能做证据化判断，不能武断归因。
7. 在论文协议下，XGB 的 `F1_bin/AUROC` 和 SEMBA/TGN 相比如何？
8. 在负边识别上，XGB 的 `F1_negative/F1_macro/PR_AUC_negative` 是否仍然强？
9. 这次 paper protocol 结果和上一轮严格无泄漏结果是否方向一致？
10. 后续是否值得把 XGB 特征融合进 SEMBA？

---

## 10. Task 7: 可选诊断，不作为主实验

这些只在主实验完成且用户同意后执行。

- [ ] **Step 1: SEMBA 512 维诊断**

目的：检查 `src/config/BitcoinOTC-1.yaml` 中 `embedding_dim=512, num_epochs=30` 是否显著影响本地 SEMBA。

Run:

```powershell
& 'C:\ProgramData\anaconda3\envs\pytorch\python.exe' -m experiments.run_graph_paper_protocol --dataset BitcoinOTC-1 --processed_dir processed2 --results_dir results/paper_protocol_sign_class_config512_check --models semba --task sign_class --batch_size 1000 --num_epochs 30 --lr_init 0.01 --embedding_dim 512 --seeds 42 --device cuda
```

Expected:

```text
Only seed 42 is run. This is a configuration diagnostic, not the main multi-seed result.
```

- [ ] **Step 2: processed vs processed2 诊断**

目的：如果 `processed` 能在当前环境加载，比较它和 `processed2` 的事件数、正负边比例、split 大小。

Run:

```powershell
& 'C:\ProgramData\anaconda3\envs\pytorch\python.exe' -c "from utils import get_data; import json; rows=[];
for p in ['processed','processed2']:
    data,tr,val,te=get_data('BitcoinOTC-1','./data/BitcoinOTC-1','cpu',processed_dir=p)
    rows.append({'processed_dir':p,'events':int(data.num_events),'nodes':int(data.num_nodes),'train':int(tr.num_events),'val':int(val.num_events),'test':int(te.num_events),'pos_rate':float(data.y.float().mean())})
print(json.dumps(rows, indent=2))"
```

Expected:

```text
如果 processed 加载失败，只记录失败原因，不删除、不重建 processed。
```

---

## 11. 版本管理要求

每个阶段都应小步提交，但只有用户同意后才推送 GitHub。

建议提交顺序：

```powershell
git add tests/test_paper_protocol_features.py tests/__init__.py experiments/paper_protocol_features.py
git commit -m "test: define paper protocol xgb feature behavior"

git add experiments/run_xgb_paper_protocol.py
git commit -m "feat: add paper protocol xgb runner"

git add experiments/run_graph_paper_protocol.py
git commit -m "feat: add paper protocol graph runner"

git add experiments/aggregate_paper_protocol_compare.py strict_xgb_comparison_experiment/bitcoinotc_paper_protocol_sign_class_summary.md
git commit -m "docs: report paper protocol sign-class comparison"
```

禁止：

```text
git reset --hard
git push --force
删除 data/*/raw
删除 data/*/processed
覆盖上一轮 strict_compare 结果
```

---

## 12. 完成标准

只有同时满足以下条件，才能说本实验完成：

- [ ] 计划文件已由用户确认。
- [ ] `tests/test_paper_protocol_features.py` 通过。
- [ ] `run_xgb_paper_protocol.py` smoke run 成功。
- [ ] `run_graph_paper_protocol.py` smoke run 成功。
- [ ] XGB full run 有 5 seeds 结果。
- [ ] SEMBA/TGN local paper protocol full run 有 5 seeds 结果。
- [ ] 聚合报告存在且包含中文指标解释。
- [ ] 报告明确区分论文协议和上一轮严格无泄漏协议。
- [ ] 报告明确说明 batch-inclusive 可能泄漏。
- [ ] 报告回答“论文协议下 XGB 是否仍无法忽略”。

---

## 13. 预期解释口径

如果实验结果出现以下情况，应这样解释：

### 情况 A：XGB 的 `F1_bin/AUROC` 接近或超过论文 SEMBA

说明表格历史统计在论文协议下也非常强，不能再把 XGB 当作普通 baseline。下一步应优先做 `XGB features + SEMBA decoder` 或 `XGB early-exit + graph hard cases`。

### 情况 B：XGB 的 `F1_bin` 高，但 `F1_negative` 很低

说明 XGB 主要吃到了正边多数类优势。它仍可作为 paper metric baseline，但对“不信任识别”的研究价值要谨慎判断。

### 情况 C：本地 SEMBA 接近论文 Table 6

说明环境影响不大，之前严格实验中 SEMBA 变低主要来自协议变化。此时 XGB 和 SEMBA 的论文协议对比更有说服力。

### 情况 D：本地 SEMBA 明显低于论文 Table 6

不能直接说是环境问题。必须检查：

1. 是否用了 `sign_class`。
2. 是否用了 `batch_size=1000`。
3. 是否用了 `num_epochs=15, lr_init=0.01`。
4. 是否用了 `embedding_dim=64` 或配置中的 `512`。
5. `processed` 和 `processed2` 统计是否一致。
6. 当前 PyTorch/PyG 版本是否可能影响 memory/GAT 行为。

### 情况 E：XGB 在 paper protocol 下比上一轮 strict protocol 更强

这很可能来自 batch-inclusive 泄漏。报告必须把它作为协议效应，而不能把它解释成模型真实泛化能力提升。

---

## 14. 本计划执行前的用户确认点

请用户确认后再执行。以后所有类似确认点都必须写清楚“不同选择会造成什么影响”，不能只问是否同意。

### 14.1 是否同意新增 paper protocol runner 和测试文件

选择 A：同意新增。

- 影响：实验逻辑会独立放在 `experiments/run_xgb_paper_protocol.py`、`experiments/run_graph_paper_protocol.py` 等新文件中，不会改旧 `train.py`。
- 好处：论文协议实验和上一轮严格无泄漏实验能清楚分开，后续可复现。
- 代价：项目会多几个实验脚本，需要用 Git 管理好。

选择 B：不同意新增，只用旧 `train.py` 和临时命令跑。

- 影响：能少写代码，但 XGB 无法自然对齐同一套结构化输出。
- 风险：结果难追溯，后面很难比较 XGB、SEMBA/TGN、论文 Table 6。

推荐：选择 A。

### 14.2 主数据缓存使用 `processed2` 还是 `processed`

选择 A：主实验使用 `processed2`，只读检查 `processed`。

- 影响：最稳妥，当前 PyTorch/PyG 环境已验证 `processed2` 可加载。
- 好处：不会动用户禁止删除的 `data/*/processed`，也不会因为旧缓存兼容问题中断。
- 代价：和论文作者当年环境生成的 `processed` 可能不是完全同一个二进制缓存，但事件数、时间排序、正负边比例会被记录。

选择 B：强行使用 `processed`。

- 影响：更接近原项目已有缓存。
- 风险：旧 PyG 保存格式可能在当前环境加载失败；如果失败，会阻塞实验，但不能删除或重建 `processed`。

推荐：选择 A。

### 14.3 正式实验使用多少个 seeds

选择 A：使用 5 seeds：`42, 43, 44, 45, 46`。

- 影响：结果能报告均值和标准差，和上一轮严格实验保持一致。
- 好处：更稳，不容易被单个随机种子误导。
- 代价：SEMBA/TGN 训练时间更长。

选择 B：先用 1 seed：`42`。

- 影响：速度快，适合检查方向。
- 风险：不能作为正式结论，只能叫 pilot。

选择 C：使用 3 seeds：`42, 43, 44`。

- 影响：时间和稳定性折中。
- 风险：标准差估计比 5 seeds 弱。

推荐：正式实验选择 A；如果担心时间，先 B 做 smoke/pilot，再 A 做正式结果。

### 14.4 主图模型参数用旧脚本实际口径还是配置文件口径

选择 A：旧脚本实际口径：`embedding_dim=64, num_epochs=15, lr_init=0.01`。

- 依据：`parser.py` 默认 `embedding_dim=64`；`best_run_details.csv` 中 `BitcoinOTC-1 + sign_class + semba/tgn` 是 `num_epochs=15, lr_init=0.01`。
- 影响：最接近当前 GitHub 旧训练脚本实际会跑出的参数。
- 好处：能回答“旧项目代码口径下，本地 SEMBA/TGN 和 XGB 怎么比”。
- 风险：如果论文实际内部实验用了更大的隐藏维度，它可能低估 SEMBA。

选择 B：配置文件口径：`embedding_dim=512, num_epochs=30, batch_size=1000`。

- 依据：`src/config/BitcoinOTC-1.yaml` 写有 `embedding_dim=512, num_epochs=30`。
- 影响：容量更大、训练更久，可能更接近论文调参后的模型规模。
- 好处：如果 SEMBA 上升明显，说明参数规模对复现影响较大。
- 代价：训练更慢，显存更吃紧；也不一定等同于论文最终命令，因为旧 `run_baselines.sh` 没有读取这个 yaml。

推荐：主实验先选 A，另加 B 的单 seed 诊断。如果 B 明显提升，再决定是否把 B 扩展成 5 seeds。

### 14.5 是否把 `embedding_dim=512, num_epochs=30` 放进主实验

选择 A：先作为可选诊断，只跑 SEMBA seed 42。

- 影响：快速判断大参数是否显著影响结果。
- 好处：省时间，不会让主实验变得太大。
- 风险：如果 seed 42 偶然性强，需要后续补充。

选择 B：直接放进主实验，5 seeds 全跑。

- 影响：最完整。
- 好处：能更有力地回答“参数规模是不是导致 SEMBA 低于论文”的问题。
- 代价：耗时显著增加，而且会让第一轮 paper protocol 对比变复杂。

推荐：选择 A。先用单 seed 判断是否值得扩展。
