# SEMBA 项目进展文件明细

更新时间：2026-06-23
主项目目录：`E:\导师之任\导师之任\社交信任\semba-main`
主要实验工作区：`E:\导师之任\导师之任\社交信任\semba-main\.worktrees\exp-xgb-history`
远端仓库：`https://github.com/xincheng-learning/trust-prediction-based-on-SEMBA`
原始项目：`https://github.com/claws-lab/semba`
相关论文：`Representation Learning in Continuous-Time Dynamic Signed Networks`
当前最重要分支：`exp/xgb-history`

这份文档用于记录我们到目前为止围绕 SEMBA 项目做过的理解、工程整理、实验验证、结论判断和后续改进方向。它不是论文正文，而是项目交接和继续实验的主记忆文件。

---

## 1. 当前整体目标

我们最初的目标是：充分理解 SEMBA 项目和论文，并基于它做动态符号信任预测方向的改进。

经过多轮实验后，目标已经从“单纯复现 SEMBA”变成更清晰的研究问题：

```text
在动态符号信任网络中，应该如何在严格在线预测协议下，利用 SEMBA 的 signed memory 机制，同时加入显式历史地位/声誉/可见性特征，提升对负边/不信任边的预测能力？
```

这里的“严格在线预测”指：

```text
对时间 t 的当前事件：
1. 只能使用时间 < t 的历史；
2. 先预测当前事件；
3. 再把当前事件写入 memory/history；
4. 如果多个事件时间戳相同，要先预测同时间戳内全部事件，再统一写入历史。
```

这个标准是后续研究的主标准。论文/旧代码中的 `to_update=True` 协议只作为复现、审计和对照口径，不应作为现实在线预测的主标准。

---

## 2. 已完成工作的总览

目前已经完成以下工作：

| 类别 | 已完成内容 | 当前状态 |
| --- | --- | --- |
| 项目理解 | 阅读 SEMBA 项目结构、核心文件、数据处理逻辑、训练入口和 baselines | 已完成第一轮上手 |
| 论文理解 | 梳理 SEMBA 的 signed memory、balanced aggregation、long-term propagation、time encoding | 已完成第一轮理解 |
| 本地环境 | 使用当前 Anaconda/PyTorch/PyG 环境，建立 `processed2` 数据缓存 | 可用 |
| Git/GitHub | 初始化本地 Git，建立 `main`、`dev/current-env`、`exp/xgb-history`、worktree 实验隔离流程 | 已建立 |
| 远端仓库 | 推送到用户仓库 `xincheng-learning/trust-prediction-based-on-SEMBA` | 已完成一轮推送 |
| XGB smoke | 用在线历史特征做 XGB 小规模实验，发现 XGB 很强 | 已完成 |
| 严格在线对比 | 在 BitcoinOTC-1 上比较 XGB、SEMBA、TGN、SEMBA-noprop | 已完成 5 seeds |
| paper/public-code 协议审计 | 模拟旧代码 `to_update=True`，验证 batch-inclusive 信息泄露风险 | 已完成 5 seeds |
| XGB 特征消融 | 对用户提出的历史可见性、活跃度、目标声誉等特征做消融 | 已完成一轮 |
| SEMBA-status-decoder | 在 SEMBA embedding decoder 输入中加入严格在线 status/asymmetry 特征 | 已完成 smoke + 5 seeds |
| TGB 学习起点 | 克隆 `shenyangHuang/TGB` 到本地，作为后续严格 temporal graph benchmark 学习材料 | 已完成克隆 |

当前最重要的实验结论有三条：

1. 在严格在线 BitcoinOTC-1 协议下，原始 SEMBA runner 没有稳定超过 XGB 历史统计强 baseline。
2. XGB 强的原因不是“随便一个表格模型都强”，而是目标节点历史声誉、可见性、源节点活跃度、pair history 等时间安全的历史统计特征很有信息量。
3. 把这些严格在线状态/地位特征接入 SEMBA decoder 后，`SEMBA-status-decoder` 在 5 seeds 下显著超过原始 SEMBA，也超过本轮 XGB-all reference。

---

## 3. 当前 Git/GitHub 工作流

### 3.1 当前分支和工作区

主工作区：

```text
E:\导师之任\导师之任\社交信任\semba-main
```

主要实验 worktree：

```text
E:\导师之任\导师之任\社交信任\semba-main\.worktrees\exp-xgb-history
```

已建立的分支逻辑：

| 分支 | 用途 |
| --- | --- |
| `main` | 稳定基线，尽量只放确认保留的项目状态 |
| `dev/current-env` | 当前本地环境兼容开发线，包括 `processed2` 兼容、基础文档、低资源 smoke |
| `exp/processed2-smoke` | 第一轮 processed2 和 smoke 实验分支 |
| `exp/xgb-history` | XGB 历史特征、严格在线实验、paper protocol 审计、后续 status decoder 实验主分支 |

目前 `exp/xgb-history` 已推送到 GitHub：

```text
https://github.com/xincheng-learning/trust-prediction-based-on-SEMBA/tree/exp/xgb-history
```

已推送的关键提交：

```text
0cd85b4 docs: summarize xgb protocol audit
0b44b4a docs: report strict bitcoinotc full comparison
7f377f2 feat: add strict bitcoinotc xgb comparison runners
29244f0 docs: add strict xgb comparison plan
c090821 test: audit xgb smoke against class imbalance
8aca6ea feat: add xgb history smoke baseline
```

### 3.2 为什么用 worktree

worktree 的作用是把不同实验放到不同文件夹，避免来回切分支导致文件混乱。

例如：

```text
semba-main/                         -> dev/current-env
semba-main/.worktrees/exp-xgb-history -> exp/xgb-history
```

以后继续实验时，推荐流程是：

```text
1. 从 dev/current-env 或当前稳定实验分支创建 exp/<name>
2. 在 .worktrees/<name> 里做实验
3. 先跑 smoke
4. smoke 有价值再跑 full 多 seed
5. 保存代码、汇总表、报告
6. 经用户确认后 commit
7. 经用户确认后 push
```

### 3.3 什么应该进入 GitHub，什么不应该进入

应该提交：

```text
代码
实验 runner
测试
配置模板
小型汇总 CSV
实验报告 md
关键图表
README/说明文档
```

不应该提交：

```text
data/
processed2/
results/
大型 per-run predictions
模型权重 .pt/.pth/.ckpt
pickle/joblib 模型文件
__pycache__/
.worktrees/
临时日志
```

原因是：GitHub 应保存“可复现的方法和摘要”，不保存本地大型缓存和临时产物。

---

## 4. SEMBA 项目结构说明

当前项目的核心结构可以这样理解：

```text
semba-main/
  README.md
  requirements.txt
  parser.py
  run.sh
  run_baselines.sh
  train.py
  eval.py
  utils.py
  model_wrapper.py
  pipeline.png
  best_run_details.csv

  dataset_loaders/
    bitcoin_dataset.py
    wikirfa_dataset.py
    epinions_dataset.py

  src/
    semba.py
    config/

  baselines/
    tgn/
    sdgnn/
    sgcn/
    sigat/
    sgclstm/
    caw/

  scripts/
    build_processed2.ps1

  md/
    中文理解文档和计划

  docs/
    experiments/
    superpowers/plans/
```

核心文件作用如下：

| 文件/目录 | 作用 |
| --- | --- |
| `train.py` | 原项目训练入口，加载数据、切分 train/val/test、训练模型、保存结果 |
| `eval.py` | 加载模型并评估 |
| `parser.py` | 命令行参数定义，例如 `--model`、`--dataset`、`--task` |
| `utils.py` | 数据加载、时间切分、batch 构造、early stop、指标辅助 |
| `model_wrapper.py` | 把 SEMBA、TGN、GCN、SGCN 等包装成统一接口 |
| `src/semba.py` | SEMBA 核心实现：正负 memory、消息构造、GRU 更新、时间编码 |
| `dataset_loaders/bitcoin_dataset.py` | BitcoinOTC/BitcoinAlpha 原始数据转 PyG `TemporalData` |
| `dataset_loaders/wikirfa_dataset.py` | WikiRfA 数据处理 |
| `dataset_loaders/epinions_dataset.py` | Epinions 数据处理 |
| `baselines/tgn/` | TGN baseline |
| `baselines/*/config/` | 各 baseline 在不同数据集上的配置 |
| `scripts/build_processed2.ps1` | 在当前环境下重建兼容的 `processed2` 缓存 |

推荐小白阅读顺序：

```text
1. README.md
2. parser.py
3. run.sh
4. dataset_loaders/bitcoin_dataset.py
5. utils.py 里的 get_data 和 split
6. train.py 的 train/test 循环
7. model_wrapper.py 里的 STGNN
8. src/semba.py 里的 SEMBA memory
9. baselines/tgn/tgn.py
```

---

## 5. SEMBA 论文核心理解

论文题目：

```text
Representation Learning in Continuous-Time Dynamic Signed Networks
```

研究对象是连续时间动态符号网络：

```text
(source node, target node, timestamp, sign/weight)
```

在信任网络中：

| 边类型 | 含义 |
| --- | --- |
| 正边 | 信任、支持、认可、友好关系 |
| 负边 | 不信任、反对、敌对、否定关系 |

SEMBA 关注三个挑战：

| 挑战 | 含义 |
| --- | --- |
| Temporal-awareness | 边按时间到来，模型必须理解事件顺序 |
| Sign-awareness | 正边和负边含义不同，不能混成普通边 |
| Staleness | 节点长时间不活跃时，旧 embedding 会过期，需要长期邻居传播 |

### 5.1 Signed Memories：正负记忆分开

SEMBA 给每个节点维护两套记忆：

```text
memory_pos：正关系记忆
memory_neg：负关系记忆
```

代码位置：

```text
src/semba.py
```

相关结构：

```text
memory_pos
memory_neg
last_update_pos
last_update_neg
gru_pos
gru_neg
```

直观理解：

```text
一个人在“被信任/信任别人”的历史中形成一套状态；
也会在“被不信任/不信任别人”的历史中形成另一套状态。
这两套状态不能混在一起。
```

### 5.2 Balanced Aggregation：结构平衡理论

结构平衡理论的四条直觉：

```text
朋友的朋友是朋友：+ * + = +
朋友的敌人是敌人：+ * - = -
敌人的朋友是敌人：- * + = -
敌人的敌人是朋友：- * - = +
```

映射到 SEMBA：

| 当前边 | 传播方式 |
| --- | --- |
| 正边 `u -> v` | 同极性传播：正记忆影响正记忆，负记忆影响负记忆 |
| 负边 `u -> v` | 反极性传播：正记忆和负记忆交叉影响 |

代码位置：

```text
src/semba.py::__compute_msgPos__
src/semba.py::__compute_msgNeg__
```

这就是 SEMBA 相对普通 TGN 的核心改造：TGN 通常只有一套 memory，而 SEMBA 把正负关系作为传播路径差异来建模。

### 5.3 Long-term Propagation：长期传播

SEMBA 不只使用当前互动边，还会从历史邻居传播信息，缓解节点长期不活跃导致的 embedding 陈旧问题。

在 `model_wrapper.py` 中，SEMBA 大致由两部分组成：

```text
mem_model -> mem2emb
```

含义：

| 模块 | 作用 |
| --- | --- |
| `mem_model` | 维护每个节点的正负 memory |
| `mem2emb` | 用历史边、时间编码、权重信息把 memory 转成当前 embedding |

### 5.4 时间信息如何建模

SEMBA message 中会计算相对时间：

```text
t_rel = t - last_update[src]
t_enc = TimeEncoder(t_rel)
```

`TimeEncoder` 采用类似：

```text
cos(linear(t))
```

它让模型区分“刚刚发生的互动”和“很久以前发生的互动”。

---

## 6. 数据处理流程

所有数据最终都会转成 PyTorch Geometric 的 `TemporalData`：

```python
TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `src` | 源节点 |
| `dst` | 目标节点 |
| `t` | 时间戳 |
| `msg` | 边消息，一般是权重绝对值或 1 |
| `y` | 边符号，正边为 1，负边为 0 |

### 6.1 BitcoinOTC / BitcoinAlpha

代码位置：

```text
dataset_loaders/bitcoin_dataset.py
```

原始格式类似：

```text
src,dst,weight,timestamp
6,2,4,1289241911.72836
6,5,2,1289241941.53378
```

处理步骤：

```text
1. 读取 CSV
2. weight > 0 标为正边，weight < 0 标为负边
3. 按时间排序
4. dst += src.max() + 1，把 source 和 target 角色拆成两套节点 ID
5. msg = abs(weight)
6. 保存为 TemporalData
```

第 4 步很重要。原项目把同一个用户作为 source 和 destination 时拆成两个角色 ID，这更接近二部图角色建模。做原论文复现时应尊重这一点；如果后续做通用 directed signed graph，可以考虑统一节点空间作为新实验。

### 6.2 processed 与 processed2

原始 `processed` 缓存来自旧 PyG 版本。当前环境加载 `processed` 会报错：

```text
RuntimeError: The 'data' object was created by an older version of PyG.
```

因此我们建立了：

```text
data/BitcoinOTC-1/processed2
```

BitcoinOTC-1 当前 `processed2` 规模：

| 项 | 数值 |
| --- | ---: |
| events | 35,592 |
| nodes | 12,005 |
| train | 24,915 |
| val | 5,338 |
| test | 5,339 |
| 正边比例 | 0.899893 |

---

## 7. 两套实验协议的区别

### 7.1 严格在线协议

严格在线协议是现实在线预测应采用的标准：

```text
先预测当前事件 -> 再把当前事件写入 memory/history
```

我们在严格在线实验中做了这些保护：

| 保护 | 含义 |
| --- | --- |
| `global_online` | 按完整时间流生成历史，不在 val/test 人为重置历史 |
| `strict_timestamp` | 同时间戳内先全部生成特征，再统一更新历史 |
| predict-before-update | 图模型先预测和反传，再写入当前 block |
| train-only scaler | 标准化器只在 train split 上 fit |

### 7.2 paper/public-code 协议

原 SEMBA 公开代码的训练/测试路径中会使用：

```text
to_update=True
```

在 `train.py`、`model_wrapper.py` 相关路径中，大致逻辑是：

```text
当前 batch 先进入 memory/history
然后再用包含当前 batch 的历史做 embedding
最后预测当前 batch
```

这个协议的风险：

```text
当前 batch 的标签/结构信息可能在预测当前 batch 前已经进入模型状态。
```

严谨表述应该是：

```text
公开代码路径存在 batch-level leakage risk。
```

不能武断说作者主观造假，但如果论文表格确实来自这个 runner 或同类协议，那么这些结果不能被解释为严格在线因果预测结果。

---

## 8. 指标解释

后续看实验结果时，优先关注少数类和整体公平指标，不要只看 `F1_weighted`。

| 指标 | 含义 | 注意事项 |
| --- | --- | --- |
| `AUROC` | ROC 曲线下面积，衡量正负样本排序能力 | 不依赖固定阈值 |
| `PR_AUC_negative` | 把负边作为目标类时的 PR 曲线面积 | 对“不信任边识别”很关键 |
| `F1_negative` | 负边这一类的 F1 | 衡量负边 precision 和 recall 平衡 |
| `F1_macro` | 正类 F1 和负类 F1 的简单平均 | 类别不平衡时比 weighted 更公平 |
| `F1_weighted` | 按各类别样本数加权的 F1 | 正边多时会被正类表现主导 |
| `balanced_accuracy` | 正类召回和负类召回的平均 | 比普通 accuracy 更抗不平衡 |
| `total sec` | 训练和评估总耗时 | 用于比较效率 |
| `TN/FP/FN/TP` | 混淆矩阵 | 看模型到底命中了多少负边 |

`F1_weighted` 的“weighted”不是输出 1-10 的信任程度，而是按类别样本数量加权。BitcoinOTC 中正边很多，所以 `F1_weighted` 很容易被正边表现抬高。

---

## 9. 已完成实验一：XGB smoke

位置：

```text
E:\导师之任\导师之任\社交信任\semba-main\.worktrees\exp-xgb-history\docs\experiments\phase1_smoke_results.md
```

设置：

```text
dataset: BitcoinOTC-1
task: sign_class
max_events: 5000
```

小规模 smoke 结果：

| 模型 | Test AUC | Test F1_weighted | 负类/二分类指标 | 备注 |
| --- | ---: | ---: | ---: | --- |
| XGB history features | 0.8338 | 0.9721 | negative-F1 0.6984 | 使用在线历史特征 |
| SEMBA | 0.7879 | 0.6024 | F1_bin 0.6262 | 原代码兼容 smoke |
| TGN | 0.8071 | 0.5569 | F1_bin 0.5791 | 原代码兼容 smoke |

这个阶段只说明：XGB 历史统计特征值得继续研究。由于 5000 条子集正边比例极高，不能直接作为最终论文结论。

---

## 10. 已完成实验二：严格在线 full 5 seeds 对比

正式报告：

```text
E:\导师之任\导师之任\社交信任\semba-main\.worktrees\exp-xgb-history\strict_xgb_comparison_experiment\bitcoinotc_strict_compare_final_report.md
```

结果目录：

```text
E:\导师之任\导师之任\社交信任\semba-main\.worktrees\exp-xgb-history\results\strict_compare_full_gpu\BitcoinOTC-1
```

设置：

| 项 | 设置 |
| --- | --- |
| 数据集 | BitcoinOTC-1 |
| 缓存 | `processed2` |
| 任务 | `sign_class` |
| 切分 | 70% train / 15% val / 15% test |
| seeds | 42, 43, 44, 45, 46 |
| 协议 | strict predict-before-update |
| 图模型设备 | CUDA, NVIDIA GeForce RTX 3060 Ti |
| 主阈值 | validation split 最大化 `F1_macro` |

主结果：

| model | PR_AUC_negative | F1_negative | F1_macro | AUROC | total sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGB-all | 0.5472 ± 0.0055 | 0.4900 ± 0.0077 | 0.6970 ± 0.0096 | 0.8235 ± 0.0055 | 0.83 ± 0.03 |
| XGB-dst_reputation | 0.4160 ± 0.0069 | 0.4645 ± 0.0041 | 0.6845 ± 0.0030 | 0.7443 ± 0.0043 | 0.49 ± 0.02 |
| SEMBA | 0.4096 ± 0.0194 | 0.4101 ± 0.0563 | 0.6524 ± 0.0238 | 0.7505 ± 0.0162 | 188.24 ± 56.06 |
| TGN | 0.4087 ± 0.0295 | 0.4036 ± 0.0396 | 0.6606 ± 0.0209 | 0.7257 ± 0.0107 | 98.14 ± 14.09 |
| XGB-no_dst_reputation | 0.3814 ± 0.0057 | 0.3914 ± 0.0255 | 0.6464 ± 0.0089 | 0.7571 ± 0.0051 | 0.64 ± 0.02 |
| SEMBA-noprop | 0.2649 ± 0.0494 | 0.3495 ± 0.0496 | 0.6009 ± 0.0366 | 0.6879 ± 0.0360 | 163.79 ± 27.91 |

关键结论：

1. 在当前严格在线 runner 和当前超参下，XGB-all 在负类识别指标上超过 SEMBA/TGN。
2. SEMBA 比 SEMBA-noprop 更好，说明 long-term propagation 有价值。
3. SEMBA 与 TGN 的对比不是决定性碾压：TGN 的 `F1_macro` 略高，SEMBA 的 `AUROC` 略高。
4. XGB 的效率优势非常大，秒级完成；图模型是几十秒到上百秒。
5. 这不能说明“图模型永远没用”，只能说明“当前数据、当前严格协议、当前实现下，显式历史统计强 baseline 必须认真对待”。

---

## 11. 已完成实验三：paper/public-code 协议审计

报告位置：

```text
E:\导师之任\导师之任\社交信任\semba-main\.worktrees\exp-xgb-history\paper_protocol_sign_class_experiment\md\bitcoinotc_paper_protocol_sign_class_summary.md
```

实验目的：

```text
模拟旧 train.py 中 to_update=True 的 batch-inclusive 行为，
判断论文/公开代码协议下的结果与严格在线协议有何差异。
```

固定阈值 0.5 的主结果：

| model | F1_bin | AUROC | F1_negative | F1_macro | PR_AUC_negative | total sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XGB-paper/negative-aware all | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 约 0.77 |
| XGB dst_reputation | 0.8969 ± 0.0045 | 0.9449 ± 0.0007 | 0.6181 ± 0.0086 | 0.7575 ± 0.0065 | 0.7455 ± 0.0038 | 约 2.3 |
| SEMBA-emb64 | 0.8547 ± 0.0251 | 0.7899 ± 0.0219 | 0.4519 ± 0.0252 | 0.6533 ± 0.0251 | 0.4863 ± 0.0292 | 约 102.7 |
| TGN-emb64 | 0.8400 ± 0.0219 | 0.8093 ± 0.0173 | 0.4478 ± 0.0243 | 0.6439 ± 0.0230 | 0.5626 ± 0.0143 | 约 39.9 |

论文 Table 6 外部参考：

| model | paper F1_bin | paper AUROC |
| --- | ---: | ---: |
| TGN BTC-Otc | 0.74 ± 0.02 | 0.82 ± 0.03 |
| SEMBA BTC-Otc | 0.81 ± 0.04 | 0.79 ± 0.02 |

严谨结论：

1. 本地 SEMBA-emb64 的 `F1_bin=0.8547`、`AUROC=0.7899`，与论文 SEMBA 的 AUROC 基本贴近，F1 甚至更高。
2. 本地 TGN-emb64 的 `F1_bin=0.8400`、`AUROC=0.8093`，与论文 TGN 也接近。
3. 因此不能说“本地 SEMBA 没复现出来”；在 paper/public-code 协议下，它是接近论文表格口径的。
4. 但该协议存在 batch-inclusive 信息泄露风险，所以这些结果不能被解释为严格在线预测能力。
5. XGB-all 在该协议下达到 1.0，几乎一定不是正常泛化，而是 pair/history 特征提前看到了当前 batch。

---

## 12. 已完成实验四：XGB 特征消融

消融表位置：

```text
E:\导师之任\导师之任\社交信任\semba-main\.worktrees\exp-xgb-history\paper_protocol_sign_class_experiment\md\quick_xgb_ablation\quick_xgb_ablation_summary.csv
```

严格协议下的主要消融结果：

| feature_group | PR_AUC_negative | F1_negative | F1_macro | AUROC |
| --- | ---: | ---: | ---: | ---: |
| old_basic_no_pair | 0.5490 | 0.5031 | 0.7000 | 0.8412 |
| user_added_only | 0.4382 | 0.4855 | 0.6997 | 0.6933 |
| old_basic_pre_user | 0.5461 | 0.5010 | 0.6990 | 0.8403 |
| all_current | 0.5472 | 0.4900 | 0.6970 | 0.8235 |
| target_reputation_user | 0.4160 | 0.4645 | 0.6845 | 0.7443 |
| degree_ratio_status | 0.5228 | 0.4691 | 0.6793 | 0.8255 |
| pair_history_only | 0.1414 | 0.0000 | 0.4620 | 0.5000 |

用户提出过的特征包括：

```text
源用户是否历史可见
目标用户是否历史可见
源用户历史活跃度
目标用户历史被信任程度
目标用户是否是高可见节点
```

消融结论：

1. 这些特征是有价值的，尤其是目标节点历史声誉、目标节点可见性、源节点活跃度。
2. 在严格协议下，单独的用户新增特征不是唯一提升来源，但它们能提供稳定补充。
3. 最强的不是“某一个单点特征”，而是多个历史统计组合作用。
4. pair_history 在 paper 协议下会极强，但在严格协议中单独使用很弱，说明它最容易受 batch-inclusive 协议污染。

---

## 13. 已完成实验五：SEMBA + Online Status Feature Decoder

这是目前最重要的新实验，因为它把 XGB 发现的强历史特征接入了 SEMBA，而不是只停留在“XGB 打败图模型”的结论上。

实验目录：

```text
E:\导师之任\导师之任\社交信任\semba-main\.worktrees\exp-xgb-history\status_asymmetry_semba_experiment
```

最终报告：

```text
E:\导师之任\导师之任\社交信任\semba-main\.worktrees\exp-xgb-history\status_asymmetry_semba_experiment\md\bitcoinotc_status_decoder_final_report.md
```

核心模型：

```text
[z_src, z_dst, abs(z_src - z_dst), z_src * z_dst, online_status_features] -> MLP -> sign prediction
```

其中 `online_status_features` 全部严格来自当前事件时间之前的历史。

### 13.1 新增文件

| 文件 | 作用 |
| --- | --- |
| `py/build_online_status_features.py` | 构建严格在线 status/asymmetry 特征 |
| `py/run_semba_status_decoder.py` | 跑 SEMBA、SEMBA-status-decoder、XGB reference |
| `py/audit_strict_online_status_features.py` | 时间安全审计 |
| `py/aggregate_status_decoder_results.py` | 汇总结果、生成表格和图 |
| `md/status_asymmetry_semba_plan.md` | 实验计划 |
| `md/bitcoinotc_status_decoder_smoke_report.md` | smoke 报告 |
| `md/bitcoinotc_status_decoder_final_report.md` | 5 seeds final 报告 |
| `tables/status_feature_dictionary.csv` | 特征字典 |
| `tables/status_feature_audit.csv` | 时间安全审计表 |
| `tables/status_decoder_metrics_summary.csv` | 5 seeds 指标汇总 |
| `figures/metric_comparison.png` | 主指标图 |
| `figures/negative_class_comparison.png` | 负类指标图 |

### 13.2 时间安全审计

审计结果全部通过：

| check | status | 证据 |
| --- | --- | --- |
| strict_timestamp_feature_build | pass | 同时间戳事件先取特征，再统一更新历史 |
| history_window | pass | 特征在 `update_history(event)` 前构造 |
| train_only_scaler | pass | scaler 只 fit train split |
| row_alignment_train | pass | 24,915 feature rows vs 24,915 train events |
| row_alignment_val | pass | 5,338 feature rows vs 5,338 val events |
| row_alignment_test | pass | 5,339 feature rows vs 5,339 test events |
| nan_or_inf | pass | 所有特征矩阵均为有限值 |

### 13.3 smoke 结果

smoke 设置：

```text
dataset: BitcoinOTC-1
max_events: 12000
seed: 42
models: SEMBA, SEMBA-status-decoder, XGB reference
device: cuda
```

smoke 主结果：

| model | F1_negative | PR_AUC_negative | F1_macro | AUROC |
| --- | ---: | ---: | ---: | ---: |
| SEMBA | 0.0986 | 0.0696 | 0.5308 | 0.6373 |
| SEMBA-status-decoder | 0.4370 | 0.4199 | 0.7089 | 0.7349 |
| XGB-all-reference-smoke | 0.4167 | 0.3622 | 0.6983 | 0.8206 |

smoke 结论：加入严格在线 status/asymmetry 特征后，SEMBA decoder 对负类识别明显提升，值得跑 full 5 seeds。

### 13.4 full 5 seeds 结果

设置：

| 项 | 设置 |
| --- | --- |
| 数据集 | BitcoinOTC-1 |
| seeds | 42, 43, 44, 45, 46 |
| 图模型 epoch | 最多 6 |
| early stop patience | 3 |
| 主阈值 | validation split 最大化 `F1_macro` |
| 协议 | strict predict-before-update |

主结果：

| model | feature_set | F1_negative | PR_AUC_negative | F1_macro | AUROC | F1_weighted |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SEMBA | graph_memory | 0.3374 ± 0.0424 | 0.3882 ± 0.0217 | 0.6232 ± 0.0141 | 0.7360 ± 0.0101 | 0.8282 ± 0.0113 |
| SEMBA-status-decoder | all_status | 0.5406 ± 0.0117 | 0.6001 ± 0.0147 | 0.7204 ± 0.0105 | 0.8537 ± 0.0067 | 0.8493 ± 0.0096 |
| XGB-all-reference | all | 0.4900 ± 0.0077 | 0.5472 ± 0.0055 | 0.6970 ± 0.0096 | 0.8235 ± 0.0055 | 0.8455 ± 0.0116 |

SEMBA-status-decoder 相对 SEMBA 的提升：

| metric | mean delta | 5 seeds 是否全部为正 |
| --- | ---: | --- |
| F1_negative | +0.2032 | yes |
| PR_AUC_negative | +0.2118 | yes |
| F1_macro | +0.0971 | yes |
| AUROC | +0.1177 | yes |

SEMBA-status-decoder 相对 XGB-all-reference 的提升：

| metric | delta |
| --- | ---: |
| F1_negative | +0.0507 |
| PR_AUC_negative | +0.0528 |
| F1_macro | +0.0234 |
| AUROC | +0.0302 |

严谨结论：

```text
Version A: SEMBA + Online Status Feature Decoder 在 BitcoinOTC-1 sign_class 严格在线协议下有效。
```

这说明 status/asymmetry 特征不只是 XGB 能用，作为 decoder 条件输入也能补充 SEMBA signed memory 表示，特别是能提升负边/不信任边识别。

当前限制：

1. 只跑了 `all_status`，还没完成 source-only、target-only、status-gap-only 消融。
2. 只跑了 BitcoinOTC-1，还没扩展到 BitcoinAlpha-1。
3. 当前只改 decoder，没有改 message gate 或 memory update。
4. 本轮 runner 未重新跑 TGN 和 SEMBA-noprop full 5 seeds，只引用已有严格实验结果作参照。

---

## 14. 目前对“数据泄露”的严谨判断

我们不能简单说“论文是空中楼阁”，也不能忽视协议风险。更严谨的判断是：

```text
SEMBA 公开代码路径存在 batch-level leakage risk；
如果论文表格结果来自该路径或同类协议，则它不是严格在线预测结果；
但 SEMBA 的 signed memory、balanced aggregation、long-term propagation 机制本身仍有研究价值。
```

原因：

1. 旧代码使用 `to_update=True`。
2. 当前 batch 可能先进入 memory/history。
3. 然后模型用已经包含当前 batch 的状态预测当前 batch。
4. XGB 在模拟该协议时能达到 1.0，说明这种协议确实会让历史特征异常强。
5. 在严格协议下，模型表现明显回落，这进一步支持协议差异的影响。

这意味着后续论文研究应采用：

```text
主实验：严格在线协议
补充实验：paper/public-code 协议复现或审计
```

不要把 paper 协议结果作为现实在线预测能力的主证据。

---

## 15. 当前研究方向判断

### 15.1 为什么 XGB 会强

XGB 强的根本原因不是“树模型比图模型高级”，而是它直接吃到了非常清楚的历史统计信号：

```text
目标节点过去被信任程度
目标节点过去被不信任程度
目标节点可见性
源节点历史活跃度
源节点历史正负倾向
源目标 pair 历史
源目标地位差
```

这些信号在 BitcoinOTC 这类信任网络中本来就非常强。例如：

```text
一个历史上经常被信任、可见度高的人，更可能继续被信任。
一个历史上收到很多负边的人，更可能继续收到负边。
源用户过去是否活跃、是否负向倾向，也会影响当前边符号。
```

图模型如果没有显式利用这些统计特征，或者训练协议/损失函数没有重点优化负类，就可能输给 XGB。

### 15.2 为什么仍然要研究图模型

图模型仍有价值，因为它适合以下情况：

| 场景 | 图模型价值 |
| --- | --- |
| 需要多跳传播 | XGB 手工特征难覆盖复杂路径 |
| 新节点/稀疏节点 | 图模型可通过邻居传播补充表示 |
| 表格特征不完整 | 图 embedding 可以自动学习隐含结构 |
| 需要端到端表示学习 | 图模型可与下游任务联合训练 |
| 关系类型复杂 | signed memory/gate 可建模不同传播机制 |

当前更合理的路线不是“XGB vs SEMBA 二选一”，而是：

```text
用 XGB 找强历史信号；
用 SEMBA 建模 signed temporal memory；
把强历史信号作为 status/asymmetry 条件接入 SEMBA；
在严格在线协议下验证融合模型是否优于两者单独使用。
```

SEMBA-status-decoder 的 5 seeds 结果已经支持这个路线。

---

## 16. 可复用内容清单

### 16.1 可复用代码

| 位置 | 可复用内容 |
| --- | --- |
| `experiments/strict_temporal_protocol.py` | 严格在线协议、time block 处理 |
| `experiments/dynamic_features.py` | XGB 历史特征构造 |
| `experiments/run_xgb_strict.py` | 严格在线 XGB baseline |
| `experiments/run_graph_strict.py` | 严格在线图模型 baseline |
| `experiments/strict_metrics.py` | 统一指标，包括负类 F1、macro-F1、PR-AUC negative |
| `paper_protocol_sign_class_experiment/py/*` | paper/public-code 协议审计 runner |
| `status_asymmetry_semba_experiment/py/build_online_status_features.py` | 严格在线 status/asymmetry 特征构造 |
| `status_asymmetry_semba_experiment/py/run_semba_status_decoder.py` | SEMBA + status decoder runner |
| `status_asymmetry_semba_experiment/py/audit_strict_online_status_features.py` | 时间安全审计 |

### 16.2 可复用文档

| 位置 | 内容 |
| --- | --- |
| `md/SEMBA_项目论文上手指南.md` | SEMBA 项目和论文初学者指南 |
| `md/SEMBA_动态符号信任预测改进方向总结.md` | 改进方向总结 |
| `docs/experiments/git_github_workflow.md` | Git/GitHub 实验工作流 |
| `strict_xgb_comparison_experiment/bitcoinotc_strict_compare_final_report.md` | 严格在线 5 seeds 对比报告 |
| `paper_protocol_sign_class_experiment/md/bitcoinotc_paper_protocol_sign_class_summary.md` | paper protocol 审计报告 |
| `status_asymmetry_semba_experiment/md/bitcoinotc_status_decoder_final_report.md` | status decoder final 报告 |

### 16.3 可复用实验结论

可以在后续论文或开题中复用的结论：

1. 严格在线预测必须采用 predict-before-update。
2. 原公开代码路径存在 batch-inclusive 风险，应作为审计口径而非主评测口径。
3. XGB 历史统计特征是强 baseline，尤其是目标节点声誉/可见性。
4. SEMBA 的 long-term propagation 有价值，因为 SEMBA 优于 SEMBA-noprop。
5. 单纯原始 SEMBA 在当前严格 runner 下没有稳定超过 XGB。
6. SEMBA + online status/asymmetry decoder 在 BitcoinOTC-1 严格协议下显著提升，并超过 XGB reference。

---

## 17. 后续最推荐的实验路线

优先级从高到低：

### 17.1 补齐 status decoder 消融

目的：证明到底是哪类特征起作用。

建议特征组：

```text
source-only
target-only
status-gap-only
visibility-only
all_status
graph_memory-only
```

判断标准：

```text
F1_negative
PR_AUC_negative
F1_macro
AUROC
```

### 17.2 跑 BitcoinAlpha-1

目的：验证结论是否只在 BitcoinOTC-1 上成立。

注意：

```text
BitcoinAlpha-1 时间戳重复比例很高，必须使用 strict_timestamp。
```

### 17.3 做 Version B：status/asymmetry gate

Version A 只把 status 特征加到 decoder。Version B 可以进一步把 status/asymmetry 用到 message 或 memory update 中：

```text
message_gate = f(status_gap, visibility_gap, source_activity, target_reputation)
```

直观含义：

```text
不同地位差、可见性差、信任/不信任关系，应该影响 memory 更新强度和传播路径。
```

### 17.4 加入信任不对称和覆盖不对称

可定义：

```text
Coverage Asymmetry CA(u -> v, t) = |N_u(t) ∩ N_v(t)| / (|N_u(t)| + eps)
Status Gap SG(u -> v, t) = q_v(t) - q_u(t)
```

其中：

```text
q_u(t) = log(1 + in_pos_u(t)) - log(1 + in_neg_u(t))
```

这些特征应严格从 `< t` 历史构造。

### 17.5 学习和对齐 TGB

TGB 项目已克隆到：

```text
E:\导师之任\导师之任\社交信任\TGB
```

后续要重点学习：

```text
Temporal Graph Benchmark 如何定义 train/val/test
它是否采用 predict-then-update
它如何做 negative sampling
它如何评估 temporal link prediction
```

TGB 更适合作为严格时间图学习协议的参考起点。

---

## 18. 当前未解决问题和风险

| 问题 | 当前状态 | 影响 |
| --- | --- | --- |
| BitcoinAlpha-1 尚未跑 status decoder | 未完成 | 结论暂时只覆盖 BitcoinOTC-1 |
| status 特征组消融不完整 | 未完成 | 还不能精确证明哪一类特征贡献最大 |
| Version B 尚未实现 | 未完成 | 创新还主要在 decoder，尚未深入 memory/gate |
| 图模型超参未做大规模搜索 | 未完成 | 不能断言图模型最佳能力 |
| paper 协议泄露只能称为 risk | 已审计但不能断言作者意图 | 写论文时必须措辞严谨 |
| 当前结果依赖 `processed2` | 已记录 | 与原旧 PyG `processed` 可能存在环境差异 |

---

## 19. 后续和 Codex 协作的推荐说法

以后要继续实验，可以这样说：

```text
请在 exp/xgb-history worktree 里继续 status_asymmetry_semba_experiment，
不要改原始 train.py 和 src/semba.py。
先补 source-only、target-only、status-gap-only 消融，
跑 BitcoinOTC-1 seed=42 smoke，
有效后再跑 5 seeds，并生成中文报告。
不要提交 git，等我确认。
```

如果要开新实验，可以这样说：

```text
请从 dev/current-env 新建 exp/status-gate 分支和 .worktrees/exp-status-gate，
实现 Version B: status/asymmetry gate。
先写实验计划，再做 smoke，不要删除 data/raw 或 data/processed。
```

如果要推送 GitHub，可以这样说：

```text
先解释这次要 push 哪些文件、不会 push 哪些文件、为什么。
等我确认后再 commit 和 push 到 GitHub。
```

---

## 20. 当前阶段最终判断

当前项目已经从“读懂 SEMBA”推进到“有可执行改进方向并跑出正结果”的阶段。

最严谨的研究判断是：

```text
原 SEMBA 机制有价值，但原公开代码协议存在 batch-level leakage risk。
在严格在线协议下，显式历史统计特征是强 baseline。
把这些严格在线 status/asymmetry 特征融合进 SEMBA decoder 后，模型在 BitcoinOTC-1 5 seeds 上稳定提升，
说明后续围绕“status/asymmetry-aware SEMBA”继续做算法改进是有依据的。
```

最推荐的下一步不是马上写大论文，而是补齐两个验证：

```text
1. status feature ablation：证明具体哪类特征贡献最大；
2. BitcoinAlpha-1 replication：证明不是只在 BitcoinOTC-1 成立。
```

完成这两步后，再推进 Version B memory/gate 级别改造，会更稳。
