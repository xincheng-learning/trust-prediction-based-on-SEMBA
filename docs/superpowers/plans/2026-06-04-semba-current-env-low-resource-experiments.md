# SEMBA Current-Env Low-Resource Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不删除原有 `data/<dataset>/processed/` 的前提下，用当前 conda 环境创建 `processed2` 数据缓存，先完成小规模可复现实验，再系统评估低资源高精度改进路线，包括 SEMBA 修复版、无注意力版本、coverage/status 不对称特征、XGBoost 非图模型基线、轻量传播和液态时间记忆。

**Architecture:** 第一阶段只做工程安全修复和 `processed2` 兼容，让当前环境可以跑通；第二阶段建立可信评估协议，区分“原代码兼容复现”和“严格 prequential 无泄漏评估”；第三阶段先用低成本非图/浅层方法建立强基线，再逐步把有效特征接回 SEMBA。核心原则是先验证流程，再验证低资源基线，最后动模型主体。

**Tech Stack:** Python 3.11/Anaconda 当前环境，PyTorch 2.7.0+cpu，PyTorch Geometric 2.6.1，scikit-learn 1.5.1，xgboost 3.0.2，pandas 2.2.2，numpy 1.26.4。

---

## 0. 当前证据与约束

### 已确认环境

本机当前可用环境：

```text
torch 2.7.0+cpu
torch_geometric 2.6.1
sklearn 1.5.1
pandas 2.2.2
numpy 1.26.4
xgboost 3.0.2
```

已确认问题：

```text
data/*/processed/data.pt 是旧版 PyG 缓存，当前 PyG 2.6.1 直接读取会报 older version of PyG。
```

用户明确约束：

```text
不能删除 data/<dataset>/processed/
必须在 data/<dataset>/ 下创建 processed2/
优先用当前 conda 环境跑通
先小批量复现，再做改进
注意 attention 机制耗资源
需要评估 XGBoost 这种非图模型可行性
用户已有经验观察：XGB 加入出入度、正负度、历史比例等按时间在线构造的节点信息后，有时能超过图模型
改进方向必须结合论文中提到的 status theory、k-hop signed propagation、signed weight prediction 等方向
```

### 关键研究判断

当前主任务优先级：

```text
1. sign_class: 给定未来边存在，预测正边/负边
2. signlink_class: 正边/负边/无边三分类
3. signwt_pred: 只在 BitcoinOTC-1 / BitcoinAlpha-1 上做带符号权重回归
```

数据集优先级：

```text
1. BitcoinOTC-1: 主 smoke 和主实验
2. BitcoinAlpha-1: 第二主实验
3. wikirfa: 泛化验证，但先修时间排序
4. epinions: 只做 blocked/post-burst 协议，不作为严格 continuous-time 首选
```

模型优先级：

```text
1. SEMBA 原代码兼容版
2. SEMBA strict prequential 修正版
3. semba-noprop 现有无传播消融版
4. XGBoost history-feature baseline
5. SEMBA + coverage/status decoder
6. SEMBA-light no-attention propagation
7. SEMBA + asymmetry message weighting
8. SEMBA + liquid-lite memory update
```

### 为什么 XGBoost 必须进入实验

XGBoost 放进实验不是为了凑一个传统机器学习 baseline，而是因为用户已经观察到：

```text
在按时间先后顺序预测动态信任关系时，
XGB + 出入度/正负度/历史比例/recency 等在线历史特征，
有时能超过图神经网络模型。
```

这说明动态 signed trust prediction 里可能存在一个很强的低资源信号源：

```text
节点和节点对的历史统计状态，本身已经编码了大量信任/不信任演化规律。
```

因此 XGB 在本计划中的角色是三重的：

| 角色 | 含义 |
|---|---|
| 低资源强基线 | 检验不用 GNN、不用 attention 时能达到什么上限。 |
| 图模型必要性检验 | 如果 XGB 超过 SEMBA，说明当前图表示并没有有效利用额外结构信息。 |
| 特征发现器 | 通过特征重要性找出哪些历史机制最有用，再反向指导 SEMBA 改进。 |

所以后续报告不能只写：

```text
SEMBA vs TGN vs SGCN
```

而必须写：

```text
SEMBA vs XGB-History
SEMBA-light vs XGB-History
SEMBA + coverage/status vs XGB-History
```

只有当改进后的动态 signed 模型在少数类负边、跨时间泛化或 signed weight 上超过 XGB，图模型的复杂度才真正有说服力。

---

## 1. 实验总路线

### 路线 A：当前环境复现

目标不是立刻追论文最佳分数，而是确认：

```text
processed2 能生成
当前 conda 环境能读
小批量训练能跑完
指标能输出
结果文件能保存
```

通过标准：

```text
BitcoinOTC-1 + processed2 + max_events=5000 + sign_class + semba 能完成 1 epoch
BitcoinOTC-1 + processed2 + max_events=5000 + sign_class + tgn 能完成 1 epoch
XGBoost sign_class baseline 能完成训练和测试
```

### 路线 B：可信评估协议

原代码中 `model.forward(..., to_update=True)` 会在预测当前 batch 前后混用历史更新和 embedding propagation。需要单独审计是否把当前 batch 的边存在信息泄漏进当前预测。

实验必须分两条报告：

| 协议 | 目的 | 解释 |
|---|---|---|
| `paper_compat` | 对齐原项目代码路径 | 尽量少改原逻辑，方便和作者代码比较。 |
| `strict_prequential` | 做后续研究主协议 | 对每个 batch 先用历史预测，再把当前 batch 更新进 memory。 |

论文创新结果只以 `strict_prequential` 为主，`paper_compat` 只作为复现参照。

### 路线 C：低资源高精度改进

低资源方向按风险从低到高：

```text
1. XGBoost + 动态历史特征
2. SEMBA + cheap asymmetry decoder features
3. semba-noprop / SEMBA-light 去掉 Transformer attention
4. asymmetry-aware message weighting
5. liquid-lite time-aware memory
```

理由：

```text
attention propagation 是 SEMBA 主要资源消耗之一。
coverage/status 特征是 O(1) 或近似 O(degree) 的历史统计，可显著低于 TransformerConv。
XGBoost 能验证“动态信任预测是否必须依赖图神经网络”，尤其检验出入度、正负度、历史正负比例、pair history、recency、coverage/status 这些在线统计特征是否已经足够强。
```

---

## 2. 文件结构规划

### 需要修改的现有文件

| 文件 | 责任 |
|---|---|
| `parser.py` | 增加当前环境实验参数：`--processed_dir`、`--max_events`、`--protocol`、`--memory_dim`、`--time_dim`、`--nbr_size`、`--dropout`、`--results_dir`。 |
| `dataset_loaders/bitcoin_dataset.py` | 支持 `processed2`，不覆盖旧 `processed`。 |
| `dataset_loaders/wikirfa_dataset.py` | 支持 `processed2`，并在 `processed2` 下使用时间升序。 |
| `dataset_loaders/epinions_dataset.py` | 支持 `processed2`，修正 sign mask 写法。 |
| `utils.py` | `get_data()` 支持 `processed_dir`、`max_events`，增加数据统计函数。 |
| `train.py` | 修复当前已发现 bug，支持新参数和协议选择，记录资源指标。 |
| `eval.py` | 同步 `processed_dir` 参数和 signed weight 修正。 |
| `model_wrapper.py` | 修复二分类阈值，后续支持 extra pair features 和 no-attention/light propagation。 |
| `run_baselines.sh` | 修复 epinions 字符串比较。 |

### 需要新增的文件

| 文件 | 责任 |
|---|---|
| `tests/test_processed2_loaders.py` | 验证 `processed2` 不覆盖旧缓存且可读。 |
| `tests/test_training_sanity.py` | 验证小样本训练入口不会因参数、预测阈值、test 调用崩溃。 |
| `experiments/dynamic_features.py` | 从时间边流构造 XGB 和 decoder 可复用的历史特征。 |
| `experiments/run_xgb_baseline.py` | 训练 XGBoost 分类/回归基线。 |
| `experiments/run_matrix.py` | 小规模实验矩阵调度。 |
| `experiments/metrics.py` | 统一 F1、negative-F1、AUROC、PR-AUC、balanced accuracy、RMSE、PCC、R2。 |
| `experiments/resource_monitor.py` | 记录 wall-clock time 和进程 peak RSS。 |
| `scripts/build_processed2.ps1` | 当前 Windows 环境下生成 `processed2`。 |
| `scripts/run_smoke_repro.ps1` | 一键跑小规模 SEMBA/TGN/XGB smoke。 |
| `docs/experiments/README.md` | 实验记录、协议解释和结果表字段说明。 |
| `results/` | 保存所有实验 CSV/JSON，不提交大模型权重。 |

---

## 3. 指标与结果表

### 分类任务指标

所有分类任务必须记录：

```text
loss
F1_weighted
F1_macro
F1_micro
negative_class_F1
balanced_accuracy
AUROC_weighted
AUROC_macro
PR_AUC_negative
wall_time_sec
peak_rss_mb
num_events
num_nodes
batch_size
embedding_dim
memory_dim
protocol
```

为什么必须有 `negative_class_F1`：

```text
Bitcoin 和 Epinions 正边占多数，只看 weighted-F1 容易掩盖负边预测失败。
动态信任/不信任预测的研究价值主要在负边和少数类。
```

### 回归任务指标

`signwt_pred` 只在 Bitcoin 数据上做，必须记录：

```text
RMSE
MAE
PCC
R2
sign_accuracy_after_threshold
wall_time_sec
peak_rss_mb
```

### 结果文件格式

每次实验写入：

```text
results/runs.csv
results/<run_id>/args.json
results/<run_id>/metrics.json
results/<run_id>/stdout.txt
```

`runs.csv` 字段：

```text
run_id,started_at,dataset,task,model,protocol,processed_dir,max_events,
num_epochs,batch_size,embedding_dim,memory_dim,time_dim,nbr_size,seed,
F1_weighted,F1_macro,negative_class_F1,balanced_accuracy,AUROC_macro,
PR_AUC_negative,RMSE,MAE,PCC,R2,wall_time_sec,peak_rss_mb,status
```

---

## 4. Task 1: 支持 processed2 且保留原 processed

**Files:**
- Modify: `dataset_loaders/bitcoin_dataset.py`
- Modify: `dataset_loaders/wikirfa_dataset.py`
- Modify: `dataset_loaders/epinions_dataset.py`
- Modify: `utils.py`
- Modify: `parser.py`
- Test: `tests/test_processed2_loaders.py`

- [ ] **Step 1: 添加 loader 单元测试**

Create `tests/test_processed2_loaders.py`:

```python
import os
import unittest

from dataset_loaders.bitcoin_dataset import tgn_bitcoin
from dataset_loaders.wikirfa_dataset import tgn_wikirfa


class TestProcessed2Loaders(unittest.TestCase):
    def test_bitcoin_processed2_directory_is_used(self):
        dataset = tgn_bitcoin(
            root="./data/BitcoinOTC-1",
            name="BitcoinOTC-1",
            processed_dir_name="processed2",
        )
        expected_suffix = os.path.join("BitcoinOTC-1", "processed2")
        self.assertTrue(dataset.processed_dir.endswith(expected_suffix))
        self.assertTrue(os.path.exists("./data/BitcoinOTC-1/processed2/data.pt"))
        self.assertTrue(os.path.exists("./data/BitcoinOTC-1/processed/data.pt"))
        self.assertGreater(dataset[0].num_events, 0)

    def test_wikirfa_processed2_directory_is_used(self):
        dataset = tgn_wikirfa(
            root="./data/wikirfa",
            name="wikirfa",
            processed_dir_name="processed2",
        )
        expected_suffix = os.path.join("wikirfa", "processed2")
        self.assertTrue(dataset.processed_dir.endswith(expected_suffix))
        self.assertTrue(os.path.exists("./data/wikirfa/processed2/data.pt"))
        self.assertTrue(os.path.exists("./data/wikirfa/processed/data.pt"))
        self.assertGreater(dataset[0].num_events, 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认当前失败**

Run:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m unittest tests.test_processed2_loaders -v
```

Expected:

```text
TypeError: __init__() got an unexpected keyword argument 'processed_dir_name'
```

- [ ] **Step 3: 修改 Bitcoin loader**

In `dataset_loaders/bitcoin_dataset.py`, change `__init__` signature and add `processed_dir`:

```python
class tgn_bitcoin(InMemoryDataset):

    def __init__(self, root: str, edge_window_size: int = 10,
                 name='BitcoinOTC-1',
                 processed_dir_name: str = 'processed',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.edge_window_size = edge_window_size
        self.name = name
        self.processed_dir_name = processed_dir_name
        if self.name == 'BitcoinOTC-1':
            self.url = 'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz'
        elif self.name == 'BitcoinAlpha-1':
            self.url = 'https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz'

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.processed_dir_name)
```

Keep existing `processed_file_names`, `download()`, and `process()` logic unchanged.

- [ ] **Step 4: 修改 WikiRfA loader**

In `dataset_loaders/wikirfa_dataset.py`, change `__init__` and add `processed_dir`:

```python
class tgn_wikirfa(InMemoryDataset):

    def __init__(self, root: str, edge_window_size: int = 10,
                 name='wikirfa',
                 processed_dir_name: str = 'processed',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.edge_window_size = edge_window_size
        self.name = name
        self.processed_dir_name = processed_dir_name
        if self.name == 'wikirfa':
            self.url = 'https://snap.stanford.edu/data/wiki-RfA.txt.gz'

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.processed_dir_name)
```

In `process()`, replace WikiRfA time sorting with:

```python
            if self.processed_dir_name == 'processed2':
                t_sorted, ix = t.sort()
            else:
                t_sorted, ix = t.sort(descending=True)
```

Then replace the assertion with:

```python
            if self.processed_dir_name == 'processed2':
                assert sorted(t.cpu().tolist()) == t.cpu().tolist()
            else:
                assert sorted(t.cpu().tolist(), reverse=True) == t.cpu().tolist()
```

Rationale:

```text
原 processed 保持作者缓存语义；processed2 使用更合理的时间升序，避免 PyG 时间切分时训练集拿到更晚事件。
```

- [ ] **Step 5: 修改 Epinions loader**

In `dataset_loaders/epinions_dataset.py`, change `__init__` and add `processed_dir`:

```python
class tgn_epinions(InMemoryDataset):

    def __init__(self, root: str, edge_window_size: int = 10,
                 name='epinions',
                 processed_dir_name: str = 'processed',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.edge_window_size = edge_window_size
        self.name = name
        self.processed_dir_name = processed_dir_name
        if self.name == 'epinions':
            self.url = 'http://konect.cc/files/download.tsv.epinions.tar.bz2'

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.processed_dir_name)
```

Replace sign mask block with:

```python
            signs_raw = torch.tensor(signs_raw, dtype=torch.float)
            mask_zero = signs_raw != 0
            signs = (signs_raw[mask_zero] > 0).long()

            edge_index = torch.tensor(edge_index, dtype=torch.long).t()[:, mask_zero]
```

- [ ] **Step 6: 修改 parser**

In `parser.py`, add:

```python
    parser.add_argument('--processed_dir', default='processed', type=str)
    parser.add_argument('--max_events', default=None, type=int)
    parser.add_argument('--protocol', default='paper_compat', type=str,
                        choices=['paper_compat', 'strict_prequential'])
    parser.add_argument('--memory_dim', default=64, type=int)
    parser.add_argument('--time_dim', default=32, type=int)
    parser.add_argument('--nbr_size', default=10, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--results_dir', default='results', type=str)
```

- [ ] **Step 7: 修改 utils.get_data**

Change signature:

```python
def get_data(NAME, path, device, val_ratio=0.15, test_ratio=0.15,
             processed_dir='processed', max_events=None):
```

Replace dataset construction with:

```python
    if NAME == 'BitcoinOTC-1' or NAME == 'BitcoinAlpha-1':
        dataset = tgn_bitcoin(path, edge_window_size=1, name=NAME,
                              processed_dir_name=processed_dir)
    elif NAME == 'epinions':
        dataset = tgn_epinions(path, edge_window_size=1, name=NAME,
                               processed_dir_name=processed_dir)
    elif NAME == 'wikirfa':
        dataset = tgn_wikirfa(path, edge_window_size=1, name=NAME,
                              processed_dir_name=processed_dir)
```

After `data = dataset[0].to(device)`, add:

```python
    if max_events is not None:
        data = data[:max_events]
```

- [ ] **Step 8: 修改 train/eval 调用 get_data**

In `train.py` and `eval.py`, replace:

```python
    data, train_data, val_data, test_data = get_data(args.dataset, dataset_path, args.device, val_ratio=args.val_ratio, 
                                                    test_ratio=args.test_ratio)
```

with:

```python
    data, train_data, val_data, test_data = get_data(
        args.dataset,
        dataset_path,
        args.device,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        processed_dir=args.processed_dir,
        max_events=args.max_events,
    )
```

- [ ] **Step 9: 运行 processed2 测试**

Run:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m unittest tests.test_processed2_loaders -v
```

Expected:

```text
test_bitcoin_processed2_directory_is_used ... ok
test_wikirfa_processed2_directory_is_used ... ok
```

- [ ] **Step 10: Commit**

```powershell
git add parser.py utils.py dataset_loaders tests
git commit -m "feat: support processed2 data caches"
```

If this folder remains a non-git snapshot, record the same scope in `docs/experiments/CHANGELOG.md`.

---

## 5. Task 2: 修复小规模训练阻塞 bug

**Files:**
- Modify: `train.py`
- Modify: `eval.py`
- Modify: `model_wrapper.py`
- Modify: `run_baselines.sh`
- Test: `tests/test_training_sanity.py`

- [ ] **Step 1: 添加训练 sanity 测试**

Create `tests/test_training_sanity.py`:

```python
import unittest
import torch

from model_wrapper import STGNN


class TestTrainingSanity(unittest.TestCase):
    def test_binary_classification_threshold_uses_logit_zero(self):
        model = STGNN(
            model_name="gcn",
            task="sign_class",
            num_feats=2,
            num_nodes=3,
            embedding_dim=4,
            num_layers=1,
            device="cpu",
        )
        z = torch.zeros(3, 4)
        with torch.no_grad():
            model.lin.weight.zero_()
            model.lin.bias.fill_(0.1)
        pred = model.predict(z, torch.tensor([0]), torch.tensor([1]), classify=True)
        self.assertEqual(float(pred.item()), 1.0)

    def test_sign_weight_target_maps_negative_edges_to_negative_values(self):
        signs = torch.tensor([1, 0, 1, 0])
        weights = torch.tensor([[3.0], [2.0], [1.0], [4.0]])
        signed_edge_weights = (2 * signs.float() - 1) * weights.float().mean(dim=1)
        self.assertEqual(signed_edge_weights.tolist(), [3.0, -2.0, 1.0, -4.0])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认阈值测试失败**

Run:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m unittest tests.test_training_sanity -v
```

Expected before fix:

```text
test_binary_classification_threshold_uses_logit_zero ... FAIL
```

- [ ] **Step 3: 修复二分类阈值**

In `model_wrapper.py`, replace:

```python
            return (torch.sigmoid(value) if not classify else torch.where(value>0.5, 1., 0.)).squeeze()
```

with:

```python
            return (torch.sigmoid(value) if not classify else torch.where(value > 0, 1., 0.)).squeeze()
```

- [ ] **Step 4: 修复 signed weight 目标**

In `train.py` and `eval.py`, replace every:

```python
            signed_edge_weights = signs * weight.float().mean(dim=1)
```

with:

```python
            signed_edge_weights = (2 * signs.float() - 1) * weight.float().mean(dim=1)
```

- [ ] **Step 5: 修复 train.py 验证集打印**

In `train.py`, replace:

```python
        print(f'Val [Loss: {val_loss:.4f} {metric_string(train_params)}]')
```

with:

```python
        print(f'Val [Loss: {val_loss:.4f} {metric_string(val_params)}]')
```

- [ ] **Step 6: 修复 train.py 测试调用参数数量**

In `train.py`, replace:

```python
    test_loss, test_params = test(args, test_data, epoch, 'test')
    test_trans_loss, test_trans_params = test(args, test_trans_data, epoch, 'test')
    test_ind_loss, test_ind_params = test(args, test_ind_data, epoch, 'test')
```

with:

```python
    test_loss, test_params = test(args, test_data, 'test')
    test_trans_loss, test_trans_params = test(args, test_trans_data, 'test')
    test_ind_loss, test_ind_params = test(args, test_ind_data, 'test')
```

- [ ] **Step 7: 暴露 SEMBA 资源参数**

In `train.py`, replace model construction:

```python
    model = STGNN (args.model, args.task, num_feats, data.num_nodes, args.embedding_dim, args.num_layers, device=args.device, debug=args.debug)
```

with:

```python
    model = STGNN(
        args.model,
        args.task,
        num_feats,
        data.num_nodes,
        args.embedding_dim,
        args.num_layers,
        memory_dim=args.memory_dim,
        time_dim=args.time_dim,
        dropout=args.dropout,
        nbr_size=args.nbr_size,
        device=args.device,
        debug=args.debug,
    )
```

- [ ] **Step 8: 修复 run_baselines.sh 字符串比较**

In `run_baselines.sh`, replace:

```bash
                    if [ "${dataset}" -eq "epinions" ]; then 
```

with:

```bash
                    if [ "${dataset}" = "epinions" ]; then
```

- [ ] **Step 9: 运行 sanity 测试**

Run:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m unittest tests.test_training_sanity -v
```

Expected:

```text
test_binary_classification_threshold_uses_logit_zero ... ok
test_sign_weight_target_maps_negative_edges_to_negative_values ... ok
```

- [ ] **Step 10: 静态编译检查**

Run:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m py_compile train.py eval.py model_wrapper.py parser.py utils.py dataset_loaders\bitcoin_dataset.py dataset_loaders\wikirfa_dataset.py dataset_loaders\epinions_dataset.py
```

Expected:

```text
No output; exit code 0
```

- [ ] **Step 11: Commit**

```powershell
git add train.py eval.py model_wrapper.py run_baselines.sh tests
git commit -m "fix: unblock current environment smoke training"
```

---

## 6. Task 3: 建立 processed2 + 小规模复现脚本

**Files:**
- Create: `scripts/build_processed2.ps1`
- Create: `scripts/run_smoke_repro.ps1`
- Create: `docs/experiments/README.md`

- [ ] **Step 1: 创建 processed2 生成脚本**

Create `scripts/build_processed2.ps1`:

```powershell
$ErrorActionPreference = "Stop"
$Python = "C:\ProgramData\anaconda3\python.exe"
$Datasets = @("BitcoinOTC-1", "BitcoinAlpha-1", "wikirfa", "epinions")

foreach ($Dataset in $Datasets) {
    Write-Host "Building processed2 for $Dataset"
    & $Python -c "from utils import get_data; data,tr,val,te=get_data('$Dataset','./data/$Dataset','cpu',processed_dir='processed2',max_events=None); print('$Dataset', data.num_events, data.num_nodes, tr.num_events, val.num_events, te.num_events)"
}
```

- [ ] **Step 2: 创建 smoke 复现脚本**

Create `scripts/run_smoke_repro.ps1`:

```powershell
$ErrorActionPreference = "Stop"
$Python = "C:\ProgramData\anaconda3\python.exe"
$Common = @(
    "--dataset", "BitcoinOTC-1",
    "--task", "sign_class",
    "--processed_dir", "processed2",
    "--max_events", "5000",
    "--num_epochs", "1",
    "--batch_size", "1000",
    "--num_feats", "8",
    "--feat_type", "zeros",
    "--embedding_dim", "64",
    "--memory_dim", "32",
    "--time_dim", "32",
    "--nbr_size", "5",
    "--device", "cpu"
)

& $Python train.py @Common --model semba
& $Python train.py @Common --model semba-noprop
& $Python train.py @Common --model tgn
```

- [ ] **Step 3: 创建实验 README**

Create `docs/experiments/README.md`:

```markdown
# SEMBA 当前环境实验记录

## 协议

- `paper_compat`: 尽量保留原项目训练顺序，用于复现作者代码路径。
- `strict_prequential`: 每个 batch 先预测，再更新 memory，用于后续研究主结果。

## 数据缓存

- 原缓存：`data/<dataset>/processed/`
- 当前环境缓存：`data/<dataset>/processed2/`

实验禁止删除原 `processed/`。

## 首批 smoke 命令

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_processed2.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_smoke_repro.ps1
```

## 首批判定

- 三个模型都能完成 1 epoch，说明流程可跑。
- `semba-noprop` 如果接近或优于 `semba`，说明 attention propagation 的性价比不足，后续优先低资源替代。
- `semba` 如果显著优于 `semba-noprop`，说明 propagation 有用，后续尝试 no-attention propagation 而不是完全删除传播。
```

- [ ] **Step 4: 运行 processed2 生成**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_processed2.ps1
```

Expected:

```text
Building processed2 for BitcoinOTC-1
BitcoinOTC-1 <events> <nodes> <train> <val> <test>
...
```

- [ ] **Step 5: 运行 smoke 复现**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_smoke_repro.ps1
```

Expected:

```text
每个模型输出 E001 Tr、Val、Overall performance。
```

- [ ] **Step 6: Commit**

```powershell
git add scripts docs/experiments
git commit -m "chore: add current environment smoke scripts"
```

---

## 7. Task 4: 做泄漏审计并实现 strict_prequential 协议

**Files:**
- Modify: `model_wrapper.py`
- Modify: `train.py`
- Test: `tests/test_prequential_protocol.py`

### 背景

当前 `STGNN.forward()` 对 SEMBA/TGN 的顺序是：

```text
读当前 memory
if to_update: update_memory(current batch)
GraphAttentionEmbedding 使用 self.seen_ei
输出 z
```

这可能导致当前 batch 的边存在信息进入当前 batch 的 embedding propagation。对 `link_pred` 和 `signlink_class` 特别敏感。

### 目标

实现两种协议：

| 协议 | 顺序 |
|---|---|
| `paper_compat` | 保持原逻辑。 |
| `strict_prequential` | 先基于历史 `seen_ei` 生成 z 和预测，再更新当前 batch。 |

- [ ] **Step 1: 添加 prequential 测试**

Create `tests/test_prequential_protocol.py`:

```python
import unittest
import torch

from model_wrapper import STGNN


class TestPrequentialProtocol(unittest.TestCase):
    def test_strict_prequential_does_not_add_current_edges_before_embedding(self):
        model = STGNN(
            model_name="semba-noprop",
            task="sign_class",
            num_feats=2,
            num_nodes=4,
            embedding_dim=8,
            num_layers=1,
            memory_dim=4,
            time_dim=4,
            device="cpu",
        )
        x = torch.zeros(4, 2)
        pos_edge_index = torch.tensor([[0], [1]])
        neg_edge_index = torch.empty(2, 0, dtype=torch.long)
        pos_times = torch.tensor([1])
        neg_times = torch.empty(0, dtype=torch.long)
        pos_weights = torch.ones(1, 1)
        neg_weights = torch.empty(0, 1)

        model.forward(
            x,
            pos_edge_index,
            neg_edge_index,
            pos_times,
            neg_times,
            pos_weights,
            neg_weights,
            to_update=False,
        )
        self.assertEqual(model.seen_pos_ei.shape[1], 0)

        model.update_memory(
            pos_edge_index,
            neg_edge_index,
            pos_times,
            neg_times,
            pos_weights,
            neg_weights,
        )
        self.assertEqual(model.seen_pos_ei.shape[1], 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 在 train/test 中分支协议**

In `train.py`, inside both `train(args)` and `test(args, ...)`, replace the single forward call:

```python
        z = model (x, pos_ei_batch, neg_ei_batch, t_pos, t_neg, weight_pos, weight_neg, 
                   to_update=True).to(device=args.device)
```

with:

```python
        if args.protocol == 'strict_prequential':
            z = model(
                x, pos_ei_batch, neg_ei_batch, t_pos, t_neg, weight_pos, weight_neg,
                to_update=False,
            ).to(device=args.device)
        else:
            z = model(
                x, pos_ei_batch, neg_ei_batch, t_pos, t_neg, weight_pos, weight_neg,
                to_update=True,
            ).to(device=args.device)
```

After `optimizer.step()` in `train(args)`, add:

```python
        if args.protocol == 'strict_prequential':
            model.update_memory(pos_ei_batch, neg_ei_batch, t_pos, t_neg, weight_pos, weight_neg)
```

In `test(args, ...)`, after predictions for the batch, add:

```python
        if args.protocol == 'strict_prequential':
            model.update_memory(pos_ei_batch, neg_ei_batch, t_pos, t_neg, weight_pos, weight_neg)
```

- [ ] **Step 3: 运行协议测试**

Run:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m unittest tests.test_prequential_protocol -v
```

Expected:

```text
test_strict_prequential_does_not_add_current_edges_before_embedding ... ok
```

- [ ] **Step 4: 对比两个协议 smoke**

Run:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' train.py --model semba --dataset BitcoinOTC-1 --task sign_class --processed_dir processed2 --max_events 5000 --num_epochs 1 --protocol paper_compat --device cpu
& 'C:\ProgramData\anaconda3\python.exe' train.py --model semba --dataset BitcoinOTC-1 --task sign_class --processed_dir processed2 --max_events 5000 --num_epochs 1 --protocol strict_prequential --device cpu
```

Expected:

```text
两条命令都完成；strict_prequential 指标可能低于 paper_compat。
```

- [ ] **Step 5: Commit**

```powershell
git add train.py model_wrapper.py tests
git commit -m "feat: add strict prequential evaluation protocol"
```

---

## 8. Task 5: 构建 XGBoost 动态历史特征基线

**Files:**
- Create: `experiments/metrics.py`
- Create: `experiments/dynamic_features.py`
- Create: `experiments/run_xgb_baseline.py`
- Test: `tests/test_dynamic_features.py`

### XGB 基线定位

XGBoost 不是图模型，但在这里不是普通 baseline，而是用户已有经验支持的低资源强对照。它回答的问题是：

```text
动态信任预测是否一定需要 GNN？
仅用按时间在线构造的出入度、正负度、历史比例、recency、pair history、coverage/status，
能否接近或超过 SEMBA？
```

如果 XGB 超过 SEMBA，不应该把它视为“意外”，而应该视为重要研究发现：

```text
动态图神经模型的复杂度并不天然带来收益；
动态 signed trust prediction 的关键可能是历史统计状态和不对称机制；
后续 SEMBA 改进应优先吸收这些低成本强信号，而不是简单堆叠 attention 或更深 GNN。
```

XGB 还必须输出 feature importance，用来回答：

```text
到底是出入度、正负比例、pair history、recency、coverage asymmetry，
还是 status gap 在驱动预测？
```

### 首批任务

只做 `sign_class`：

```text
给定未来边存在，只预测正负号。
```

原因：

```text
不需要无边负采样，避免把 link prediction 采样协议复杂性混进第一轮。
```

### 特征定义

对每条事件 `(u, v, t)`，在更新该事件前抽取历史特征：

| 特征 | 定义 |
|---|---|
| `src_out_pos` | t 前 u 发出的正边数 |
| `src_out_neg` | t 前 u 发出的负边数 |
| `src_in_pos` | t 前指向 u 的正边数 |
| `src_in_neg` | t 前指向 u 的负边数 |
| `dst_out_pos` | t 前 v 发出的正边数 |
| `dst_out_neg` | t 前 v 发出的负边数 |
| `dst_in_pos` | t 前指向 v 的正边数 |
| `dst_in_neg` | t 前指向 v 的负边数 |
| `src_pos_ratio` | `src_out_pos / max(1, src_out_pos + src_out_neg)` |
| `dst_in_pos_ratio` | `dst_in_pos / max(1, dst_in_pos + dst_in_neg)` |
| `q_src` | `log1p(src_in_pos) - log1p(src_in_neg)` |
| `q_dst` | `log1p(dst_in_pos) - log1p(dst_in_neg)` |
| `status_gap` | `q_dst - q_src` |
| `coverage_uv` | `|N_u(t) ∩ N_v(t)| / max(1, |N_u(t)|)` |
| `coverage_vu` | `|N_u(t) ∩ N_v(t)| / max(1, |N_v(t)|)` |
| `pair_pos_count` | t 前 u->v 正边次数 |
| `pair_neg_count` | t 前 u->v 负边次数 |
| `rev_pair_pos_count` | t 前 v->u 正边次数 |
| `rev_pair_neg_count` | t 前 v->u 负边次数 |
| `src_recency_log` | `log1p(t - last_event_time[src])`，无历史则 0 |
| `dst_recency_log` | `log1p(t - last_event_time[dst])`，无历史则 0 |

Do not use current edge weight as a feature for `sign_class`, because Bitcoin weight sign and magnitude may leak label information.

- [ ] **Step 1: 添加 feature 测试**

Create `tests/test_dynamic_features.py`:

```python
import unittest

from experiments.dynamic_features import build_sign_class_rows


class TestDynamicFeatures(unittest.TestCase):
    def test_features_use_history_before_current_event(self):
        events = [
            {"src": 1, "dst": 2, "t": 10, "y": 1},
            {"src": 1, "dst": 3, "t": 20, "y": 0},
            {"src": 1, "dst": 2, "t": 30, "y": 0},
        ]
        rows = build_sign_class_rows(events)
        first = rows[0]
        third = rows[2]

        self.assertEqual(first["pair_pos_count"], 0)
        self.assertEqual(first["src_out_pos"], 0)
        self.assertEqual(third["pair_pos_count"], 1)
        self.assertEqual(third["pair_neg_count"], 0)
        self.assertEqual(third["src_out_pos"], 1)
        self.assertEqual(third["src_out_neg"], 1)
        self.assertEqual(third["label"], 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 实现 metrics.py**

Create `experiments/metrics.py`:

```python
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def binary_classification_metrics(y_true: Iterable[int], y_prob: Iterable[float]) -> Dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=int)
    y_prob = np.asarray(list(y_prob), dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)

    result = {
        "F1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "F1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "F1_micro": float(f1_score(y_true, y_pred, average="micro")),
        "negative_class_F1": float(f1_score(1 - y_true, 1 - y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    if len(np.unique(y_true)) == 2:
        result["AUROC"] = float(roc_auc_score(y_true, y_prob))
        result["PR_AUC_negative"] = float(average_precision_score(1 - y_true, 1 - y_prob))
    else:
        result["AUROC"] = float("nan")
        result["PR_AUC_negative"] = float("nan")

    return result
```

- [ ] **Step 3: 实现 dynamic_features.py**

Create `experiments/dynamic_features.py` with this public API:

```python
from __future__ import annotations

from collections import defaultdict
from math import log1p
from typing import Dict, Iterable, List, MutableMapping, Set, Tuple


Event = Dict[str, int]
FeatureRow = Dict[str, float]


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def build_sign_class_rows(events: Iterable[Event]) -> List[FeatureRow]:
    out_pos = defaultdict(int)
    out_neg = defaultdict(int)
    in_pos = defaultdict(int)
    in_neg = defaultdict(int)
    pair_pos = defaultdict(int)
    pair_neg = defaultdict(int)
    neighbors: MutableMapping[int, Set[int]] = defaultdict(set)
    last_time = {}
    rows: List[FeatureRow] = []

    sorted_events = sorted(events, key=lambda row: row["t"])

    for event in sorted_events:
        src = int(event["src"])
        dst = int(event["dst"])
        t = int(event["t"])
        label = int(event["y"])

        src_total_out = out_pos[src] + out_neg[src]
        dst_total_in = in_pos[dst] + in_neg[dst]
        q_src = log1p(in_pos[src]) - log1p(in_neg[src])
        q_dst = log1p(in_pos[dst]) - log1p(in_neg[dst])

        n_src = neighbors[src]
        n_dst = neighbors[dst]
        common = len(n_src.intersection(n_dst))

        src_last = last_time.get(src)
        dst_last = last_time.get(dst)

        row: FeatureRow = {
            "src": float(src),
            "dst": float(dst),
            "t": float(t),
            "src_out_pos": float(out_pos[src]),
            "src_out_neg": float(out_neg[src]),
            "src_in_pos": float(in_pos[src]),
            "src_in_neg": float(in_neg[src]),
            "dst_out_pos": float(out_pos[dst]),
            "dst_out_neg": float(out_neg[dst]),
            "dst_in_pos": float(in_pos[dst]),
            "dst_in_neg": float(in_neg[dst]),
            "src_pos_ratio": _ratio(out_pos[src], src_total_out),
            "dst_in_pos_ratio": _ratio(in_pos[dst], dst_total_in),
            "q_src": float(q_src),
            "q_dst": float(q_dst),
            "status_gap": float(q_dst - q_src),
            "coverage_uv": _ratio(common, len(n_src)),
            "coverage_vu": _ratio(common, len(n_dst)),
            "pair_pos_count": float(pair_pos[(src, dst)]),
            "pair_neg_count": float(pair_neg[(src, dst)]),
            "rev_pair_pos_count": float(pair_pos[(dst, src)]),
            "rev_pair_neg_count": float(pair_neg[(dst, src)]),
            "src_recency_log": float(log1p(t - src_last)) if src_last is not None else 0.0,
            "dst_recency_log": float(log1p(t - dst_last)) if dst_last is not None else 0.0,
            "label": float(label),
        }
        rows.append(row)

        if label == 1:
            out_pos[src] += 1
            in_pos[dst] += 1
            pair_pos[(src, dst)] += 1
        else:
            out_neg[src] += 1
            in_neg[dst] += 1
            pair_neg[(src, dst)] += 1

        neighbors[src].add(dst)
        neighbors[dst].add(src)
        last_time[src] = t
        last_time[dst] = t

    return rows
```

- [ ] **Step 4: 运行 feature 测试**

Run:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m unittest tests.test_dynamic_features -v
```

Expected:

```text
test_features_use_history_before_current_event ... ok
```

- [ ] **Step 5: 实现 run_xgb_baseline.py**

Create `experiments/run_xgb_baseline.py`:

```python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from experiments.dynamic_features import build_sign_class_rows
from experiments.metrics import binary_classification_metrics
from utils import get_data


FEATURE_COLUMNS = [
    "src_out_pos", "src_out_neg", "src_in_pos", "src_in_neg",
    "dst_out_pos", "dst_out_neg", "dst_in_pos", "dst_in_neg",
    "src_pos_ratio", "dst_in_pos_ratio",
    "q_src", "q_dst", "status_gap",
    "coverage_uv", "coverage_vu",
    "pair_pos_count", "pair_neg_count",
    "rev_pair_pos_count", "rev_pair_neg_count",
    "src_recency_log", "dst_recency_log",
]


def temporal_rows_from_data(data):
    events = []
    for src, dst, t, y in zip(data.src.tolist(), data.dst.tolist(), data.t.tolist(), data.y.tolist()):
        events.append({"src": int(src), "dst": int(dst), "t": int(t), "y": int(y)})
    return build_sign_class_rows(events)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BitcoinOTC-1")
    parser.add_argument("--processed_dir", default="processed2")
    parser.add_argument("--max_events", type=int, default=5000)
    parser.add_argument("--results_dir", default="results/xgb")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data, train_data, val_data, test_data = get_data(
        args.dataset,
        f"./data/{args.dataset}",
        "cpu",
        processed_dir=args.processed_dir,
        max_events=args.max_events,
    )

    train_df = pd.DataFrame(temporal_rows_from_data(train_data))
    val_df = pd.DataFrame(temporal_rows_from_data(val_data))
    test_df = pd.DataFrame(temporal_rows_from_data(test_data))

    fit_df = pd.concat([train_df, val_df], ignore_index=True)
    x_train = fit_df[FEATURE_COLUMNS]
    y_train = fit_df["label"].astype(int)
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"].astype(int)

    scale_pos_weight = max(1.0, float((y_train == 0).sum()) / max(1, int((y_train == 1).sum())))
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=args.seed,
        n_jobs=4,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(x_train, y_train)

    y_prob = model.predict_proba(x_test)[:, 1]
    metrics = binary_classification_metrics(y_test, y_prob)
    metrics.update({
        "dataset": args.dataset,
        "model": "xgb_history_features",
        "task": "sign_class",
        "processed_dir": args.processed_dir,
        "max_events": args.max_events,
    })

    out_dir = Path(args.results_dir) / args.dataset / f"max_events_{args.max_events}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).to_csv(out_dir / "feature_importance.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 运行 XGB smoke**

Run:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m experiments.run_xgb_baseline --dataset BitcoinOTC-1 --processed_dir processed2 --max_events 5000
```

Expected:

```text
JSON metrics printed with F1_macro, negative_class_F1, AUROC, PR_AUC_negative.
results/xgb/BitcoinOTC-1/max_events_5000/metrics.csv exists.
```

- [ ] **Step 7: Commit**

```powershell
git add experiments tests
git commit -m "feat: add xgboost dynamic history baseline"
```

---

## 9. Task 6: 第一轮实验矩阵

**Files:**
- Create: `experiments/run_matrix.py`
- Create: `docs/experiments/phase1_matrix.md`

### 实验矩阵

Phase 1 只跑 CPU 小规模：

| dataset | max_events | task | models |
|---|---:|---|---|
| BitcoinOTC-1 | 5000 | sign_class | `semba`, `semba-noprop`, `tgn`, `xgb_history_features` |
| BitcoinAlpha-1 | 5000 | sign_class | `semba`, `semba-noprop`, `tgn`, `xgb_history_features` |
| BitcoinOTC-1 | 10000 | sign_class | `semba`, `semba-noprop`, `tgn`, `xgb_history_features` |

参数：

```text
num_epochs=3
batch_size=1000
embedding_dim=64
memory_dim=32
time_dim=32
nbr_size=5
protocol=strict_prequential
seed=42
```

判定：

| 观察 | 后续动作 |
|---|---|
| `xgb_history_features` 接近或超过 `semba` 的 F1_macro/negative_class_F1 | 优先推进非图模型和 hybrid decoder。 |
| `semba-noprop` 接近 `semba` | attention propagation 性价比低，优先无注意力模型。 |
| `semba` 明显优于 `semba-noprop` | propagation 有价值，尝试 light propagation 替代 attention。 |
| `tgn` 与 `semba` 差距小 | signed memory 的优势不稳定，优先增强 sign/status 特征。 |

- [ ] **Step 1: 创建 phase1 文档**

Create `docs/experiments/phase1_matrix.md`:

```markdown
# Phase 1 小规模复现实验矩阵

## 目的

确认当前 conda 环境、processed2、修复后的训练循环和 XGB baseline 都能跑通。

## 主指标

1. F1_macro
2. negative_class_F1
3. balanced_accuracy
4. AUROC
5. wall_time_sec

## 实验

| run | dataset | max_events | task | model | protocol |
|---|---|---:|---|---|---|
| P1-001 | BitcoinOTC-1 | 5000 | sign_class | semba | strict_prequential |
| P1-002 | BitcoinOTC-1 | 5000 | sign_class | semba-noprop | strict_prequential |
| P1-003 | BitcoinOTC-1 | 5000 | sign_class | tgn | strict_prequential |
| P1-004 | BitcoinOTC-1 | 5000 | sign_class | xgb_history_features | strict_prequential |
| P1-005 | BitcoinAlpha-1 | 5000 | sign_class | semba | strict_prequential |
| P1-006 | BitcoinAlpha-1 | 5000 | sign_class | semba-noprop | strict_prequential |
| P1-007 | BitcoinAlpha-1 | 5000 | sign_class | tgn | strict_prequential |
| P1-008 | BitcoinAlpha-1 | 5000 | sign_class | xgb_history_features | strict_prequential |
```

- [ ] **Step 2: 执行矩阵手动命令**

Run neural models:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' train.py --model semba --dataset BitcoinOTC-1 --task sign_class --processed_dir processed2 --max_events 5000 --num_epochs 3 --batch_size 1000 --embedding_dim 64 --memory_dim 32 --time_dim 32 --nbr_size 5 --protocol strict_prequential --device cpu
& 'C:\ProgramData\anaconda3\python.exe' train.py --model semba-noprop --dataset BitcoinOTC-1 --task sign_class --processed_dir processed2 --max_events 5000 --num_epochs 3 --batch_size 1000 --embedding_dim 64 --memory_dim 32 --time_dim 32 --nbr_size 5 --protocol strict_prequential --device cpu
& 'C:\ProgramData\anaconda3\python.exe' train.py --model tgn --dataset BitcoinOTC-1 --task sign_class --processed_dir processed2 --max_events 5000 --num_epochs 3 --batch_size 1000 --embedding_dim 64 --memory_dim 32 --time_dim 32 --nbr_size 5 --protocol strict_prequential --device cpu
```

Run XGB:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m experiments.run_xgb_baseline --dataset BitcoinOTC-1 --processed_dir processed2 --max_events 5000
```

- [ ] **Step 3: 记录结果**

Append observed metrics to `docs/experiments/phase1_matrix.md` under:

```markdown
## Observed Results

| run | status | F1_macro | negative_class_F1 | AUROC | wall_time_sec | note |
|---|---|---:|---:|---:|---:|---|
```

No row may be left with empty status. Use one of:

```text
pass
failed_import
failed_runtime
failed_metric
timeout
```

---

## 10. Task 7: SEMBA + coverage/status decoder

**Files:**
- Modify: `model_wrapper.py`
- Modify: `train.py`
- Reuse: `experiments/dynamic_features.py`
- Test: `tests/test_asym_decoder.py`

### 目的

只改 decoder，不改 SEMBA memory。这是最稳创新路线：

```text
[z_u, z_v] -> 原 SEMBA decoder
[z_u, z_v, CA_uv, CA_vu, q_u, q_v, status_gap] -> asymmetry decoder
```

对应用户总结中的：

```text
coverage asymmetry
status asymmetry
不对称信任
```

对应论文未来方向：

```text
status theory
```

- [ ] **Step 1: 添加 decoder 测试**

Create `tests/test_asym_decoder.py`:

```python
import unittest
import torch

from model_wrapper import STGNN


class TestAsymDecoder(unittest.TestCase):
    def test_predict_accepts_extra_pair_features(self):
        model = STGNN(
            model_name="gcn",
            task="sign_class",
            num_feats=2,
            num_nodes=3,
            embedding_dim=4,
            num_layers=1,
            extra_pair_feat_dim=3,
            device="cpu",
        )
        z = torch.zeros(3, 4)
        extra = torch.tensor([[0.1, 0.2, -0.1]])
        prob = model.predict(z, torch.tensor([0]), torch.tensor([1]), extra_pair_feats=extra)
        self.assertEqual(prob.ndim, 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 修改 STGNN 初始化参数**

In `model_wrapper.py`, add to `STGNN.__init__`:

```python
        extra_pair_feat_dim=0,
```

Set:

```python
        self.extra_pair_feat_dim = extra_pair_feat_dim
        decoder_in_dim = 2 * embedding_dim + extra_pair_feat_dim
```

Replace task heads:

```python
        if task == 'signlink_class':
            self.lin = torch.nn.Linear(decoder_in_dim, 3).to(device=device)
        elif task == 'sign_class':
            self.lin = torch.nn.Linear(decoder_in_dim, 1).to(device=device)
        elif task == 'link_pred':
            self.lin = torch.nn.Linear(decoder_in_dim, 1).to(device=device)
        elif task == 'signwt_pred':
            self.lin = torch.nn.Linear(decoder_in_dim, 1).to(device=device)
```

- [ ] **Step 3: 修改 predict 签名**

In `model_wrapper.py`, replace:

```python
    def predict(self, z, src, dst, classify=False):
        emb_nodepair = torch.cat((z[src], z[dst]), dim=1)
```

with:

```python
    def predict(self, z, src, dst, classify=False, extra_pair_feats=None):
        emb_nodepair = torch.cat((z[src], z[dst]), dim=1)
        if self.extra_pair_feat_dim:
            if extra_pair_feats is None:
                extra_pair_feats = torch.zeros(
                    emb_nodepair.size(0),
                    self.extra_pair_feat_dim,
                    dtype=emb_nodepair.dtype,
                    device=emb_nodepair.device,
                )
            emb_nodepair = torch.cat((emb_nodepair, extra_pair_feats), dim=1)
```

- [ ] **Step 4: 运行 decoder 测试**

Run:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m unittest tests.test_asym_decoder -v
```

Expected:

```text
test_predict_accepts_extra_pair_features ... ok
```

### 实验

Run after implementing batch feature extraction:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' train.py --model semba --dataset BitcoinOTC-1 --task sign_class --processed_dir processed2 --max_events 10000 --num_epochs 3 --protocol strict_prequential --extra_pair_features coverage_status --device cpu
```

Success criterion:

```text
F1_macro 或 negative_class_F1 比 SEMBA strict_prequential baseline 提升 >= 0.02，且 wall_time 增幅 <= 10%。
```

---

## 11. Task 8: 低资源传播替代 attention

**Files:**
- Modify: `model_wrapper.py`
- Test: `tests/test_light_propagation.py`

### 背景

SEMBA 的 `GraphAttentionEmbedding` 使用 `TransformerConv`，多头 attention 是资源消耗重点。低资源路线按成本排序：

```text
1. semba-noprop: 已存在，完全去掉 propagation
2. semba-lightprop: 使用 mean/SAGE-style propagation，不用 attention
3. semba-small-attn: 降 heads、降 embedding、降 nbr_size
```

### 首先利用现有模型

先跑：

```powershell
& 'C:\ProgramData\anaconda3\python.exe' train.py --model semba-noprop --dataset BitcoinOTC-1 --task sign_class --processed_dir processed2 --max_events 10000 --num_epochs 3 --protocol strict_prequential --device cpu
```

判定：

```text
如果 semba-noprop 与 semba 差距小于 0.02 F1_macro，则 attention propagation 不值得保留。
如果差距大于 0.05，则实现 semba-lightprop。
```

### semba-lightprop 设计

使用 cheap edge aggregation：

```text
输入: [memory_pos, memory_neg, x]
聚合: 对历史邻居做 mean aggregation，不做 attention，不做 edge-wise softmax
输出: embedding_dim
```

成功标准：

```text
相对 SEMBA attention 版本，wall_time 降低 >= 25%，peak_rss 降低 >= 15%；
相对 semba-noprop，F1_macro 提升 >= 0.02。
```

---

## 12. Task 9: 不对称 message weighting

**Files:**
- Modify: `src/semba.py`
- Modify: `model_wrapper.py`
- Reuse: `experiments/dynamic_features.py`

### 目的

在 SEMBA 原 balanced aggregation 基础上加入轻量不对称强度：

```text
alpha(u, v, t) = sigmoid(MLP[CA_uv, CA_vu, status_gap, delta_t])
message = alpha * original_message
```

对应用户总结中的：

```text
不对称调节的正负边传播
```

风险：

```text
需要把 pair-level 动态特征传进 memory/message 模块，接口改动比 decoder 大。
```

进入条件：

```text
SEMBA + coverage/status decoder 已经稳定提升。
```

成功标准：

```text
比 coverage/status decoder 再提升 >= 0.01 negative_class_F1；
额外运行时间不超过 decoder 版本的 20%。
```

---

## 13. Task 10: liquid-lite 时间记忆

**Files:**
- Modify: `src/semba.py`
- Test: `tests/test_liquid_memory.py`

### 目的

借鉴液态神经网络，但不一开始引入复杂依赖。先实现轻量时间衰减记忆：

```text
candidate = GRU(message, old_memory)
decay = exp(-softplus(beta) * log1p(delta_t))
new_memory = decay * old_memory + (1 - decay) * candidate
```

对应用户总结中的：

```text
液态时间记忆更新
```

对应论文挑战：

```text
C1 Temporal-Awareness
C2 Staleness
```

为什么这比完整 LNN 更稳：

```text
参数少
计算量低
不需要新依赖
仍显式建模不规则时间间隔
```

进入条件：

```text
strict_prequential SEMBA baseline、XGB baseline、coverage/status decoder 已完成。
```

成功标准：

```text
在 BitcoinOTC-1 和 BitcoinAlpha-1 上，negative_class_F1 平均提升 >= 0.015；
wall_time 相对 SEMBA baseline 不增加超过 15%。
```

---

## 14. Task 11: signed weight prediction 扩展

**Files:**
- Modify: `train.py`
- Modify: `eval.py`
- Create: `experiments/run_xgb_weight_baseline.py`

### 范围

只在：

```text
BitcoinOTC-1
BitcoinAlpha-1
```

上做，因为它们有 `[-10, +10]` 权重。

### 注意

`signwt_pred` 不能再使用错误目标：

```text
旧: signs * abs(weight)，负边变成 0
新: (2 * signs - 1) * abs(weight)，负边是负数
```

### 模型

| 模型 | 目的 |
|---|---|
| XGBRegressor history features | 非图低资源回归基线 |
| SEMBA signwt_pred | 原神经基线 |
| SEMBA + status/coverage decoder | 测试不对称特征对强度预测是否有效 |

指标：

```text
RMSE
MAE
PCC
R2
sign_accuracy_after_threshold
```

成功标准：

```text
XGBRegressor 如果接近 SEMBA，说明 signed weight 很大程度可由历史统计解释。
SEMBA + 不对称特征如果提升 PCC/R2，说明图表示对强度预测仍有额外价值。
```

---

## 15. 与论文改进方向的对应关系

| 论文/讨论方向 | 本计划对应实验 |
|---|---|
| status theory | `q_src`、`q_dst`、`status_gap`，先进入 XGB 和 decoder。 |
| k-hop signed propagation | 先不直接做昂贵 k-hop GNN，用 coverage/common-neighbor 作为 cheap k-hop proxy；若有效再做 light signed k-hop。 |
| signed weight prediction | Task 11，只在 Bitcoin 数据上做。 |
| staleness | strict_prequential + liquid-lite time decay。 |
| balanced aggregation | 保留 SEMBA 正负双记忆，先不破坏核心机制。 |
| attention resource cost | `semba-noprop`、`semba-lightprop`、XGB baseline。 |
| 更少资源更高精度 | 以 F1_macro/negative_class_F1 与 wall_time/peak_rss 的 Pareto frontier 判断。 |

---

## 16. 阶段性决策门

### Gate 1: 当前环境可跑

通过条件：

```text
processed2 生成成功
SEMBA/TGN/semba-noprop smoke 完成
XGB smoke 完成
```

不通过时：

```text
停止模型创新，先修工程兼容。
```

### Gate 2: 选择主评估协议

通过条件：

```text
paper_compat 与 strict_prequential 都能跑。
```

研究主协议：

```text
strict_prequential
```

### Gate 3: 判断是否值得保留 attention

如果：

```text
SEMBA - semba-noprop < 0.02 F1_macro
```

则：

```text
主线转向 no-prop / XGB / asymmetry decoder。
```

如果：

```text
SEMBA - semba-noprop >= 0.05 F1_macro
```

则：

```text
实现 semba-lightprop 替代 attention。
```

### Gate 4: 判断 XGB 是否成为主 baseline

如果：

```text
XGB negative_class_F1 >= SEMBA strict_prequential negative_class_F1 - 0.02
```

则：

```text
论文实验必须把 XGB 作为强低资源 baseline。
```

如果：

```text
XGB 明显优于 SEMBA
```

则：

```text
优先研究为什么历史不对称统计足够强，并将 SEMBA 改进聚焦到这些特征的神经融合。
```

### Gate 5: 是否进入 liquid memory

只有当：

```text
coverage/status decoder 已经提升
message weighting 至少不退化
```

才进入 liquid-lite memory。否则不要动 memory 主体。

---

## 17. 最终论文级实验表

如果 Phase 1-3 通过，最终表格设计：

| Model | Resource Level | Uses Graph NN | Uses Attention | Uses Asymmetry | Uses Signed Memory |
|---|---|---:|---:|---:|---:|
| XGB-History | Low | No | No | Yes | No |
| TGN | Medium | Yes | Yes | No | No |
| SEMBA-noprop | Medium | Yes | No | No | Yes |
| SEMBA | High | Yes | Yes | No | Yes |
| SEMBA + CS Decoder | Medium/High | Yes | Optional | Yes | Yes |
| SEMBA-LightProp + CS | Medium | Yes | No | Yes | Yes |
| SEMBA-LightProp + CS + Liquid | Medium | Yes | No | Yes | Yes |

主报告：

```text
F1_macro
negative_class_F1
AUROC_macro
PR_AUC_negative
wall_time_sec
peak_rss_mb
```

核心结论目标：

```text
用 coverage/status 不对称特征和轻量传播，在更低资源下达到或超过原 SEMBA 的少数类预测表现。
```

---

## 18. 自检清单

### 需求覆盖

| 用户需求 | 覆盖位置 |
|---|---|
| 当前 conda 环境运行 | Task 1-3 |
| 不删除 processed，创建 processed2 | Task 1 |
| 小批量复现 | Task 3、Task 6 |
| 后续改进最佳方向 | Task 7-11、Gate 3-5 |
| 结合论文改进方向 | Section 15 |
| 注意 attention 耗资源 | Task 8、Gate 3 |
| 尝试 XGB 非图模型 | Task 5、Gate 4 |
| 详细周密实验计划 | 全文 |
| 包含第一个改造实验 | Task 1-4 是工程修复与小规模复现，Task 7 是第一个低风险模型改造 |

### 执行顺序

严格执行顺序：

```text
Task 1 -> Task 2 -> Task 3 -> Task 4 -> Task 5 -> Task 6 -> Task 7 -> Task 8 -> Task 9 -> Task 10 -> Task 11
```

不要跳过 Task 4 的泄漏审计。后续论文级结论必须建立在 strict_prequential 上。
