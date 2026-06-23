# SEMBA 项目与论文上手指南

本指南对应本地项目目录：

`E:\导师之任\导师之任\社交信任\semba-main`

对应 GitHub 仓库：

https://github.com/claws-lab/semba

对应论文：

`E:\导师之任\导师之任\论文\Representation Learning in Continuous-Time Dynamic Signed Networks.pdf`

arXiv 页面：

https://arxiv.org/abs/2207.03408

arXiv PDF：

https://arxiv.org/pdf/2207.03408

## 1. 这个项目一句话在做什么

这个项目实现论文 **Representation Learning in Continuous-Time Dynamic Signed Networks** 中的模型 **SEMBA**。

它要解决的是：

给你一串按时间发生的有向正负关系，例如：

```text
用户 A 在 2012-01-01 信任用户 B，评分 +8
用户 C 在 2012-01-02 不信任用户 A，评分 -3
用户 A 在 2012-01-03 会不会和 D 产生关系？
如果产生，是正关系还是负关系？
如果有评分，评分大概是多少？
```

SEMBA 的目标就是学习每个节点随时间变化的表示向量，然后用这些向量预测未来的边、边符号、边是否存在以及边权重。

关键词拆开看：

| 词 | 小白解释 |
|---|---|
| signed network | 带正负号的网络。正边可以表示信任、支持、赞成；负边可以表示不信任、反对、敌对。 |
| dynamic network | 网络随时间变化。边不是一次性给定，而是按时间不断出现。 |
| continuous-time | 不把时间粗暴切成固定快照，而是保留真实时间戳。 |
| representation learning | 给每个节点学一个向量，向量里编码它过去的关系历史和结构位置。 |
| SEMBA | Signed link's Evolution using Memory modules and Balanced Aggregation，即用记忆模块和平衡聚合学习正负边演化。 |

## 2. 先理解论文：SEMBA 为什么存在

论文认为，以前的方法各有缺口：

| 方法类型 | 能处理时间吗 | 能处理正负号吗 | 主要问题 |
|---|---:|---:|---|
| 静态 signed GNN，如 SGCN、SiGAT | 否 | 是 | 忽略边出现的先后顺序。 |
| 动态 unsigned GNN，如 TGN | 是 | 否 | 把正边和负边当成普通边，忽略信任/不信任这种异质关系。 |
| SEMBA | 是 | 是 | 同时建模时间演化、正负符号和平衡理论。 |

论文提出三个挑战：

| 挑战 | 含义 |
|---|---|
| C1 Temporal-Awareness | 节点表示要知道事件发生顺序，不能只看最终图。 |
| C2 Staleness | 很多节点长时间没有直接交互，表示容易过期。 |
| C3 Sign-Awareness | 正边和负边会导致完全不同的传播逻辑，不能混在一起。 |

SEMBA 的核心做法有三块：

| 模块 | 作用 | 代码位置 |
|---|---|---|
| Signed Memories | 每个节点有两套记忆：正记忆和负记忆。 | `src/semba.py` 中 `memory_pos`、`memory_neg` |
| Balanced Aggregation | 用结构平衡理论决定正负记忆如何相互影响。 | `src/semba.py` 中 `__compute_msgPos__`、`__compute_msgNeg__` |
| Long-Term Propagation | 用 Graph Transformer 从历史邻居聚合信息，缓解节点表示过期。 | `model_wrapper.py` 中 `GraphAttentionEmbedding` |

平衡理论可以用四句话记住：

```text
朋友的朋友是朋友。
朋友的敌人是敌人。
敌人的朋友是敌人。
敌人的敌人是朋友。
```

映射到 SEMBA：

| 新出现的边 | 记忆怎么传 |
|---|---|
| u 和 v 是正边 | u 的正记忆受 v 的正记忆影响，u 的负记忆受 v 的负记忆影响。 |
| u 和 v 是负边 | u 的正记忆受 v 的负记忆影响，u 的负记忆受 v 的正记忆影响。 |

所以，SEMBA 不是简单地“把边的符号作为一个特征塞进去”，而是让正负关系决定消息传递路径。

## 3. 论文里的 4 个任务

项目中的 `--task` 对应论文的四类任务：

| 任务 | 命令参数 | 输入条件 | 要预测什么 | 代码标签 |
|---|---|---|---|---|
| Dynamic Link Existence Prediction | `link_pred` | 只知道历史图 | 未来 u 和 v 有没有边 | 有边 1，无边 0 |
| Dynamic Link Sign Prediction | `sign_class` | 已知未来 u 和 v 有边 | 这条边是正还是负 | 正边 1，负边 0 |
| Dynamic Link Signed Existence Prediction | `signlink_class` | 只知道历史图 | 正边、负边、无边三分类 | 正边 0，负边 1，无边 2 |
| Dynamic Link Signed Weight Prediction | `signwt_pred` | 已知未来 u 和 v 有带权边 | 边的带符号权重 | 回归值 |

注意：README 的任务表主要列了 `signlink_class`、`sign_class`、`link_pred`，但代码和 `run_baselines.sh` 还支持 `signwt_pred`。

## 4. 项目文件夹地图

项目结构可以这样理解：

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
    saved_models/

  baselines/
    gcn/
    sgcn/
    sigat/
    sdgnn/
    sgclstm/
    tgn/
    caw/
    prop/
    semba-noprop/

  data/
    BitcoinOTC-1/
    BitcoinAlpha-1/
    wikirfa/
    epinions/
```

每个核心文件的作用：

| 文件/目录 | 作用 |
|---|---|
| `README.md` | 项目官方说明，列出依赖、数据集、训练命令。 |
| `requirements.txt` | 原作者给的 Linux conda 环境文件，主要是 Python 3.8、PyTorch 1.7、PyG 2.0.2。 |
| `parser.py` | 定义所有命令行参数，例如 `--model`、`--dataset`、`--task`。 |
| `train.py` | 训练主入口：加载数据、切分训练/验证/测试集、跑 epoch、保存模型。 |
| `eval.py` | 加载已保存模型并测试。 |
| `utils.py` | 数据加载、时间切分、早停、学习率调度、指标辅助函数。 |
| `model_wrapper.py` | 最重要的胶水层，把 SEMBA、TGN、SGCN、GCN 等包装成统一接口。 |
| `src/semba.py` | SEMBA 模型主体，实现正负双记忆、消息生成、聚合、GRU 更新。 |
| `dataset_loaders/` | 把原始数据转成 PyG 的 `TemporalData`。 |
| `data/` | 原始数据和处理后缓存。 |
| `baselines/` | 对比模型实现。 |
| `best_run_details.csv` | 作者记录的一些最佳训练轮数和学习率。 |
| `pipeline.png` | 论文模型流程图。 |

推荐阅读顺序：

```text
README.md
parser.py
run.sh
dataset_loaders/bitcoin_dataset.py
utils.py 的 get_data
train.py 的 train/test 循环
model_wrapper.py 的 STGNN
src/semba.py 的 SEMBA
baselines/tgn/tgn.py
```

## 5. 数据是如何处理的

所有数据最终都会变成 PyTorch Geometric 的 `TemporalData`：

```python
TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)
```

字段含义：

| 字段 | 含义 |
|---|---|
| `src` | 源节点编号。 |
| `dst` | 目标节点编号。 |
| `t` | 时间戳。 |
| `msg` | 边消息。Bitcoin 数据用边权绝对值，WikiRfA/Epinions 用全 1。 |
| `y` | 边符号。正边为 1，负边为 0。 |

### 5.1 BitcoinOTC / BitcoinAlpha

代码位置：

`dataset_loaders/bitcoin_dataset.py`

原始格式：

```text
src,dst,weight,timestamp
6,2,4,1289241911.72836
6,5,2,1289241941.53378
```

处理步骤：

1. 读 CSV。
2. `edge_attr > 0` 得到符号，正边为 1，负边为 0。
3. 按时间戳升序排序。
4. `dst += int(src.max()) + 1`，把目标节点空间整体后移。
5. `msg = abs(weight)`，把评分绝对值作为边消息。
6. 保存为 `TemporalData`。

第 4 步很重要：代码把源节点和目标节点拆成两个 ID 空间，类似二部图做法。这样同一个原始用户如果作为 source 和 destination，会在模型里变成两个不同角色 ID。做论文复现时按原代码即可；如果做严谨的通用 signed directed graph 建模，可以考虑改成统一节点空间。

### 5.2 WikiRfA

代码位置：

`dataset_loaders/wikirfa_dataset.py`

原始格式是 7 行一条记录，中间空一行：

```text
SRC:Steel1943
TGT:BDD
VOT:1
RES:1
YEA:2013
DAT:23:13, 19 April 2013
TXT:...
```

处理步骤：

1. 按块读取。
2. 去掉没有日期的记录。
3. 去掉 `VOT:0` 的弃权票。
4. 把用户名映射为整数 ID。
5. `VOT > 0` 为正边，否则负边。
6. 时间字符串转 Unix 秒。
7. `msg = 1`。
8. 目标节点整体后移。
9. 保存为 `TemporalData`。

注意：当前代码使用 `t.sort(descending=True)`，也就是时间降序。论文实验要求按时间排序后 70/15/15 切分，PyG 的 `train_val_test_split` 默认更适合升序时间。这个点如果你要严格复现或做新研究，需要认真检查。

### 5.3 Epinions

代码位置：

`dataset_loaders/epinions_dataset.py`

原始格式：

```text
% asym signed
1 2 -1 979081200
3 4 1 979081200
```

处理步骤：

1. 跳过注释行。
2. 读取 `src dst sign timestamp`。
3. 去掉 sign 为 0 的记录。
4. 按时间升序排序。
5. `msg = 1`。
6. 目标节点整体后移。
7. 保存为 `TemporalData`。

小心：如果以后换成真的含 0 符号的数据，`epinions_dataset.py` 中标签 mask 逻辑最好重新检查，因为代码先 mask 了一次，又重新用未 mask 的 `signs_raw` 生成 `signs`。

### 5.4 我在本地 raw 数据上读到的规模

这些是直接从本地 raw 文件统计的事件行数：

| 数据集 | 事件数 | 原始节点数 | 正边数 | 负边数 | 正边比例 | 时间范围 |
|---|---:|---:|---:|---:|---:|---|
| BitcoinOTC-1 | 35,592 | 5,881 | 32,029 | 3,563 | 0.8999 | 2010-11-08 到 2016-01-25 |
| BitcoinAlpha-1 | 24,186 | 3,783 | 22,650 | 1,536 | 0.9365 | 2010-11-08 到 2016-01-22 |
| wikirfa | 170,500 | 10,595 | 132,145 | 38,355 | 0.7750 | 2004-05-01 到 2013-06-05 |
| epinions | 841,372 | 131,828 | 717,667 | 123,705 | 0.8530 | 2001-01-09 到 2003-08-11 |

论文表 3 报告的是论文实验口径下的数据统计，其中所有数据集都明显正边占多数。这个类别不平衡是为什么论文特别强调 minority negative class。

## 6. 训练流程从代码角度怎么走

以命令为例：

```bash
python train.py --model semba --dataset BitcoinOTC-1 --task signlink_class
```

实际执行流程：

1. `parser.py` 解析参数。
2. `utils.get_data()` 根据数据集名选择 loader。
3. loader 把 raw 数据转成 `TemporalData`。
4. `TemporalData.train_val_test_split()` 按时间切分训练、验证、测试。
5. `train.py` 生成节点特征：
   - `zeros`：全 0，默认。
   - `random`：随机。
   - `one-hot`：单位矩阵。
6. `model_wrapper.STGNN` 根据 `--model` 创建模型。
7. 每个 epoch 开始调用 `model.reset_memory()`，清空历史记忆。
8. `train_data.seq_batches(batch_size=...)` 按时间批次遍历事件。
9. 每个 batch 拆成正边和负边。
10. 如果任务需要无边样本，调用 `negative_sampling()` 采样不存在的边。
11. 调用 `model(...)` 得到节点 embedding。
12. 调用 `model.loss(...)` 计算任务损失。
13. 调用 `model.predict(...)` 得到预测。
14. 反向传播并更新参数。
15. 验证和测试阶段参数固定，但记忆仍按时间在线更新。

训练中的关键变量：

| 变量 | 含义 |
|---|---|
| `pos_ei_batch` | 当前 batch 的正边。 |
| `neg_ei_batch` | 当前 batch 的负边。 |
| `null_ei_batch` | 负采样得到的无边。 |
| `x` | 节点初始特征。默认全 0。 |
| `z` | 模型学到的节点 embedding。 |
| `src/dst/y` | 用于当前任务的样本和标签。 |

## 7. SEMBA 在代码里如何对应论文

论文流程图大致是：

```text
时间批次中的事件
  -> Message Generator
  -> Aggregated Messages
  -> Memory Update
  -> Embedding Generator
  -> Pair Prediction
```

代码对应关系：

| 论文步骤 | 代码 |
|---|---|
| Step 1 Message Generator | `src/semba.py` 中 `__compute_msgPos__`、`__compute_msgNeg__`、`IdentityMessage` |
| Step 2 Message Aggregation | `src/semba.py` 中 `MeanAggregator` 或 `LastAggregator` |
| Step 3 Memory Update | `src/semba.py` 中 `gru_pos`、`gru_neg` |
| Step 4 Embedding Generation | `model_wrapper.py` 中 `GraphAttentionEmbedding`，内部用 `TransformerConv` |
| Pair Prediction | `model_wrapper.py` 中 `predict()`，拼接 `z[src]` 和 `z[dst]` 后接线性层 |

更具体一点：

| 代码元素 | 论文含义 |
|---|---|
| `memory_pos` | 节点正记忆，记录朋友/正关系方向的信息。 |
| `memory_neg` | 节点负记忆，记录敌人/负关系方向的信息。 |
| `last_update_pos` / `last_update_neg` | 每套记忆最近更新时间。 |
| `TimeEncoder` | 把相对时间差编码成向量。 |
| `msg_s_store_pos` / `msg_s_store_neg` | source 方向上的正/负消息缓存。 |
| `msg_d_store_pos` / `msg_d_store_neg` | destination 方向上的正/负消息缓存。 |
| `GRUCell` | 用聚合消息更新节点记忆。 |
| `TransformerConv` | 从历史邻居聚合长期信息，生成最终 embedding。 |

SEMBA 和 TGN 的最关键区别：

| TGN | SEMBA |
|---|---|
| 每个节点一套 memory。 | 每个节点正负两套 memory。 |
| 所有边都作为普通交互。 | 正边和负边走不同消息路径。 |
| 适合 unsigned dynamic graph。 | 适合 signed dynamic graph。 |

## 8. 如何运行项目

### 8.1 原作者推荐环境

README 说原实现使用：

```text
Python 3.8
PyTorch 1.7
PyTorch Geometric
```

并给出：

```bash
conda create --name semba --file requirements.txt
```

注意：`requirements.txt` 是 Linux 64 位环境文件，里面有 CUDA 10.1、PyTorch 1.7.1、PyG 2.0.2。Windows 直接照装可能会不顺。最稳的复现方式是 Linux 或 WSL。

### 8.2 最小训练命令

建议先跑最小模型，不要一上来跑全部 baseline：

```bash
python train.py \
  --model semba \
  --dataset BitcoinOTC-1 \
  --task signlink_class \
  --num_epochs 1 \
  --batch_size 1000 \
  --num_feats 8 \
  --feat_type zeros \
  --null_nsamples 1 \
  --device cpu
```

如果想保存模型，加：

```bash
--to_save
```

保存位置：

```text
src/saved_models/<dataset>/<task>/
```

如果是 baseline 模型，保存位置：

```text
baselines/<model>/saved_models/<dataset>/<task>/
```

### 8.3 用已保存模型评估

```bash
python eval.py --model semba --dataset wikirfa --task sign_class --device cpu
```

如果保存模型不存在，`eval.py` 会打印：

```text
Saved model not available
```

### 8.4 运行全部 baseline

```bash
./run_baselines.sh
```

但这个脚本会非常慢，而且里面有一个 bash 字符串比较问题：

```bash
if [ "${dataset}" -eq "epinions" ]; then
```

这里 `-eq` 是数字比较，字符串应该用：

```bash
if [ "${dataset}" = "epinions" ]; then
```

小白阶段不建议直接跑全部 baseline。先跑一个数据集、一个模型、一个任务，确认环境和流程都通。

## 9. 当前本机环境的注意事项

我在本机做了轻量验证，发现以下情况：

1. 直接运行 `python` 入口会失败，Anaconda Python 路径可用：

```powershell
& 'C:\ProgramData\anaconda3\python.exe'
```

2. 当前 Anaconda 环境是较新的：

```text
torch 2.7.0+cpu
torch_geometric 2.6.1
```

这和项目原始依赖差距很大。

3. 本地 `data/*/processed/data.pt` 是旧版 PyG 生成的缓存。当前 PyG 2.6 读取时会报：

```text
The 'data' object was created by an older version of PyG.
remove the 'processed/' directory in the dataset's root folder and try again.
```

解决思路有两个：

| 方案 | 做法 | 适合谁 |
|---|---|---|
| 复现原论文环境 | 建 Python 3.8 + PyTorch 1.7 + PyG 2.0.2 环境 | 想尽量复现作者结果 |
| 使用当前新环境 | 重命名或删除 `data/<dataset>/processed/`，让 loader 用 raw 重新生成 | 想在当前机器快速跑起来 |

建议用重命名方式，不直接删除：

```powershell
Rename-Item -LiteralPath 'data\BitcoinOTC-1\processed' -NewName 'processed_old'
```

然后重新跑训练命令，PyG 会根据 raw 文件重新生成 `processed/data.pt`。

4. `train.py` 末尾测试调用有一个明显参数数量问题：

当前代码：

```python
test_loss, test_params = test(args, test_data, epoch, 'test')
```

但 `test()` 定义是：

```python
def test(args, inference_data, inference_type='val'):
```

所以应该改成：

```python
test_loss, test_params = test(args, test_data, 'test')
test_trans_loss, test_trans_params = test(args, test_trans_data, 'test')
test_ind_loss, test_ind_params = test(args, test_ind_data, 'test')
```

5. `train.py` 打印验证集指标时用了 `train_params`，应该是 `val_params`：

```python
print(f'Val [Loss: {val_loss:.4f} {metric_string(train_params)}]')
```

应改为：

```python
print(f'Val [Loss: {val_loss:.4f} {metric_string(val_params)}]')
```

6. `model_wrapper.py` 中二分类 `classify=True` 使用 raw logit 和 0.5 比：

```python
torch.where(value > 0.5, 1., 0.)
```

如果按 sigmoid 0.5 阈值，raw logit 应该和 0 比，或者先 sigmoid 再和 0.5 比。

这些问题不影响你理解论文和项目，但如果要真正跑实验，建议先修。

## 10. 论文实验结论怎么读

论文实验关注四个问题：

| 问题 | 结论 |
|---|---|
| SEMBA 能不能预测未来边存在？ | 和 TGN 接近，说明它没有牺牲动态图建模能力。 |
| SEMBA 能不能预测边符号？ | 明显强于 baseline，尤其对负边这种少数类更好。 |
| SEMBA 能不能三分类预测正边/负边/无边？ | 多数数据集优于 TGN、SGCN、SiGAT。 |
| SEMBA 能不能预测带符号权重？ | 有提升但不显著，论文承认这仍有研究空间。 |

几个重要数字：

| 任务 | 论文结果解读 |
|---|---|
| link existence | SEMBA 和 TGN 几乎打平，很多数据集 AUROC 接近 0.99。 |
| sign prediction | SEMBA 明显强于 TGN。例如 WikiRfA 上 SEMBA F1 0.81，TGN F1 0.45。 |
| signlink_class | WikiRfA 上 SEMBA F1wt 0.91、F1mac 0.80，高于 TGN 的 0.85、0.71。 |
| signwt_pred | SEMBA 的 KL-div 比 TGN 低，但 RMSE/R2 提升很有限。 |

论文消融实验说明：

| 去掉什么 | 影响 |
|---|---|
| 去掉 memory | 伤害最大。说明时间记忆是核心。 |
| 去掉 embedding propagation | 也明显下降。说明长期邻域传播有用。 |
| 去掉 balanced aggregation | 也会下降。说明正负记忆路径不是装饰。 |

论文也承认 SEMBA 更耗资源：在 Epinions 上，TGN 每 epoch 约 151.4 秒、11.9GB GPU；SEMBA 每 epoch 约 335.2 秒、12.9GB GPU。

## 11. 你该怎么快速掌握这个项目

### 第一天：会跑、知道每个文件干什么

1. 读 `README.md`。
2. 看 `run.sh`，理解一个最小训练命令。
3. 看 `parser.py`，知道所有参数。
4. 看 `dataset_loaders/bitcoin_dataset.py`，理解 raw 到 `TemporalData`。
5. 跑一个 epoch 的 BitcoinOTC-1 + SEMBA + `signlink_class`。

### 第二天：理解训练和任务

1. 看 `train.py` 中 `train(args)`。
2. 搞懂 `link_pred`、`sign_class`、`signlink_class` 三个任务怎么构造标签。
3. 看 `negative_sampling()` 如何造无边样本。
4. 看 `model_wrapper.py` 中 `predict()` 和 `loss()`。
5. 跑 `sign_class` 和 `link_pred` 对比输出指标。

### 第三天：理解模型

1. 看 `model_wrapper.py` 中 `model_name == 'semba'` 的初始化。
2. 看 `src/semba.py` 的 `memory_pos`、`memory_neg`。
3. 看 `update_state()`，理解新事件如何进入消息缓存。
4. 看 `__compute_msgPos__` 和 `__compute_msgNeg__`，对应平衡理论。
5. 看 `GraphAttentionEmbedding`，理解长期传播。

### 第四天：做自己的改进

从一个小改动开始：

1. 修复训练脚本里的明显 bug。
2. 只跑 BitcoinOTC-1。
3. 只改一个模块。
4. 保存指标。
5. 和原 SEMBA 对比。

## 12. 可以复用的内容

这个项目里最值得复用的部分：

| 可复用模块 | 复用价值 |
|---|---|
| `dataset_loaders` 的结构 | 适合把新时序 signed graph 数据转成 `TemporalData`。 |
| `STGNN` 包装方式 | 适合统一管理多个模型和任务。 |
| `SEMBA` 双记忆模块 | 可以迁移到其他动态正负关系任务，如信任预测、谣言互动、评论支持/反对。 |
| `signlink_class` 任务构造 | 很实用，把“正边、负边、无边”统一成三分类。 |
| `negative_sampling` 思路 | 所有 link prediction 都需要构造无边负样本。 |
| baseline 组织方式 | 方便你新加一个模型并和原模型比较。 |
| `best_run_details.csv` | 可作为跑实验的超参数起点。 |

如果你要接入自己的数据，最小需要实现：

```python
TemporalData(
    src=torch.tensor(...),
    dst=torch.tensor(...),
    t=torch.tensor(...),
    msg=torch.tensor(...),
    y=torch.tensor(...),
)
```

其中：

```text
src/dst 必须是 long
t 必须按时间排序
msg 是 float 或 long 的二维张量，形状通常是 [num_events, msg_dim]
y 是 0/1，正边为 1，负边为 0
```

## 13. 可以怎么改进

### 13.1 工程层面优先修

这些是最建议先做的：

1. 修 `train.py` 的 test 参数数量。
2. 修验证集打印指标。
3. 修 `run_baselines.sh` 的字符串比较。
4. 修二分类预测阈值。
5. 加一个 `requirements-modern.txt` 或 `environment.yml`。
6. 加 smoke test，至少验证一个小数据能完成 1 epoch。
7. 把 YAML config 真正接入训练参数，目前 `src/config/*.yaml` 没有被主训练脚本读取。

### 13.2 数据层面改进

1. 统一源/目标节点 ID 空间，测试是否比当前二部图偏移更合理。
2. 为节点加入真实特征，而不是全 0，例如用户文本、度数、历史正负比例、时间活跃度。
3. 对 Bitcoin 的重复评分、评分尺度和时间间隔做更细处理。
4. 重新检查 WikiRfA 时间排序，确保训练集早于验证集，验证集早于测试集。
5. 对负边少数类做更好的采样或重加权。

### 13.3 模型层面改进

论文自己也提到几个方向：

| 方向 | 思路 |
|---|---|
| k-hop signed propagation | 当前主要是一阶 signed 信息，可以扩展到多跳邻域。 |
| status theory | 除平衡理论外，社会网络里还有地位理论，可融入消息路径。 |
| 更强的 inductive 节点表示 | 对训练时没出现过的节点，当前模型依赖交互历史，冷启动弱。 |
| signed weight distribution | 论文承认权重回归效果有限，可改成分布建模、ordinal regression 或 mixture density。 |
| attention 改进 | 让正邻居、负邻居、时间差、边权分别有独立注意力。 |
| 类别不平衡 | 尝试 focal loss、class-balanced loss、negative-class calibration。 |

### 13.4 实验层面改进

1. 多随机种子，不只跑一次。
2. 报告负类 F1、PR-AUC、macro-F1，不只看 weighted-F1。
3. 单独报告 transductive 和 inductive。
4. 做时间泄漏检查，确保预测某条边时只看到它之前的事件。
5. 做消融实验：无 BA、无 propagation、无 memory、无 edge weight、不同 batch size。

## 14. 最小术语表

| 术语 | 解释 |
|---|---|
| edge/event | 这里一条边就是一个随时间发生的事件。 |
| positive edge | 正关系，如信任、支持。 |
| negative edge | 负关系，如不信任、反对。 |
| no edge/null edge | 负采样造出的不存在边。 |
| memory | 节点短期历史状态。 |
| embedding | 用 memory 和邻域信息生成的最终节点向量。 |
| batch | 按时间顺序切出的一段事件。 |
| inductive | 测试时出现训练没见过的新节点。 |
| transductive | 测试节点训练时已经出现过。 |
| AUROC | 衡量二分类排序能力。 |
| F1_macro | 每类平等平均，适合看少数类表现。 |
| F1_weighted | 按类别样本数加权，容易被多数类影响。 |

## 15. 一张总流程图

```text
raw dataset
  -> dataset_loaders/*.py
  -> TemporalData(src, dst, t, msg, y)
  -> train/val/test temporal split
  -> sequential batches
  -> positive edges + negative edges + sampled null edges
  -> STGNN wrapper
  -> SEMBA memory update
  -> GraphAttentionEmbedding
  -> task head
  -> metrics and saved model
```

最重要的心智模型：

```text
数据是一串带时间的正负边。
SEMBA 给每个节点维护正负两本账。
新边来了，根据平衡理论决定该更新哪本账。
再用历史邻居传播，让没直接交互的节点也能更新表示。
最后拼接两个节点的表示，预测未来关系。
```

## 16. 推荐你的第一个改造实验

为了稳，我建议你的第一个改造不要碰模型主体，而是做一个干净的工程修复实验：

1. 修复 `train.py` 的 test 调用、验证打印、二分类阈值。
2. 重生成 `BitcoinOTC-1` 的 processed 数据。
3. 跑：

```bash
python train.py --model semba --dataset BitcoinOTC-1 --task signlink_class --num_epochs 1 --device cpu
```

4. 再跑：

```bash
python train.py --model tgn --dataset BitcoinOTC-1 --task signlink_class --num_epochs 1 --device cpu
```

5. 比较 SEMBA 和 TGN。

这个实验能让你同时理解：

```text
动态记忆是否有效
正负双记忆是否比单记忆更适合 signed graph
训练脚本如何组织任务
指标如何输出
```

等你能解释这一次实验的每一步，你就真正入门这个项目了。

