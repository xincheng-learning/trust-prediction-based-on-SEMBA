# XGB smoke audit: class imbalance, time protocol, and research value

日期：2026-06-04

本记录回答一个核心问题：`BitcoinOTC-1 max_events=5000` 的 XGB smoke 结果
`F1_weighted = 0.9721`，是否只是正边比例很高导致的“偷懒”结果。

## 结论先行

`F1_weighted = 0.9721` 不是简单等于正边比例，也不是纯粹的永远预测正边。
但是，`F1_weighted` 确实被类别不平衡明显抬高，不能单独作为研究结论。

在 5000 条 smoke 的测试段中：

| 模型/协议 | test 正边比例 | F1_weighted | F1_macro | 负类 F1 | balanced accuracy | 负类命中 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 永远预测正边 | 0.9493 | 0.9246 | 0.4870 | 0.0000 | 0.5000 | 0/38 |
| XGB, split-reset | 0.9493 | 0.9721 | 0.8426 | 0.6984 | 0.7874 | 22/38 |
| XGB, global-online + strict timestamp | 0.9493 | 0.9637 | 0.7829 | 0.5818 | 0.7098 | 16/38 |

所以：高 `F1_weighted` 里有多数类红利，但 XGB 确实识别出了一部分负边。

## 时间协议风险

XGB 历史特征必须只使用当前边之前的历史。当前特征生成已经支持：

- 默认逐边更新：适合时间戳足够细的数据。
- `--strict_timestamp`：同一时间戳的所有边先生成特征，再统一更新历史，避免同时间戳内标签污染。
- `--history_scope global_online`：按完整时间流生成历史特征，再切分 train/val/test，更接近真实在线预测。

时间戳重复情况：

| 数据集 | full events | unique timestamps | repeated fraction | max same timestamp |
| --- | ---: | ---: | ---: | ---: |
| BitcoinOTC-1 | 35592 | 35427 | 0.0078 | 10 |
| BitcoinAlpha-1 | 24186 | 1647 | 0.9949 | 216 |
| wikirfa | 170499 | 162274 | 0.0925 | 11 |
| epinions | 841370 | 939 | 1.0000 | 578994 |

这说明 `BitcoinOTC-1` 的逐边协议影响很小；`BitcoinAlpha-1`、`epinions`
必须使用 strict timestamp，否则同时间戳信息污染会很明显。

严格同时间戳协议下的 XGB 结果：

| 数据集 | max_events | F1_weighted | F1_macro | 负类 F1 | balanced accuracy | AUROC | PR-AUC negative |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BitcoinOTC-1 | 5000 | 0.9637 | 0.7829 | 0.5818 | 0.7098 | 0.8973 | 0.7108 |
| BitcoinOTC-1 | full | 0.8912 | 0.7684 | 0.5973 | 0.7513 | 0.8790 | 0.6727 |
| BitcoinAlpha-1 | 5000 | 0.9214 | 0.5480 | 0.1250 | 0.5334 | 0.6362 | 0.2259 |
| BitcoinAlpha-1 | full | 0.8493 | 0.6803 | 0.4364 | 0.6496 | 0.8286 | 0.5386 |

## 特征价值

用户提出的历史可见性、历史活跃度、历史被信任程度等特征是有价值的。

在 `BitcoinOTC-1 max_events=5000 split-reset` 的消融中：

| 特征组 | F1_weighted | F1_macro | 负类 F1 | balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| all | 0.9721 | 0.8426 | 0.6984 | 0.7874 |
| user_history_only | 0.9699 | 0.8264 | 0.6667 | 0.7618 |
| dst_reputation_only | 0.9755 | 0.8633 | 0.7385 | 0.8137 |
| no_dst_reputation_visibility | 0.9246 | 0.4870 | 0.0000 | 0.5000 |
| dst_trusted_ratio_only | 0.9605 | 0.7641 | 0.5455 | 0.6960 |

这说明目标用户历史声誉/可见性不是装饰特征，而是当前 XGB 的核心信号之一。

## 和图模型的轻量参照

本轮只做 CPU smoke，不把它当最终胜负。

`BitcoinOTC-1 max_events=20000, 1 epoch, CPU, embedding_dim=64, batch_size=1000`：

| 模型 | 训练时间 | Test AUC | Test F1_weighted |
| --- | ---: | ---: | ---: |
| XGB history features | 约 0.21 秒 | 0.9033 | 0.9453 |
| TGN smoke | 约 3.58 秒 | 0.8003 | 0.9230 |
| SEMBA smoke | 约 6.21 秒 | 0.8144 | 0.7471 |

注意：图模型这里只是 1 epoch，不代表充分训练。并且当前原项目的图模型评估只报
`F1_weighted/F1_bin/AUC`，还没有负类 F1、balanced accuracy、PR-AUC negative；
正式比较前应补齐这些指标。

## 研究价值判断

XGB 在这个时间设定下有研究价值，但研究表述应谨慎：

1. 它不应该只被当作“简单 baseline”，而应该作为强结构化历史特征 baseline。
2. 它能验证哪些历史统计信号真正有用，例如目标节点历史被信任比例、可见度、源节点活跃度。
3. 它的资源成本远低于动态图模型，适合作为低资源/快速预筛选方法。
4. 它目前不能证明“XGB 一定超过 SEMBA/TGN”，因为图模型尚未在严格同指标、多 seed、多 epoch 下公平比较。

## 后续正式协议

推荐后续固定为：

1. 使用 `--history_scope global_online --strict_timestamp`。
2. 每个实验同时报告：多数类 baseline、F1_weighted、F1_macro、负类 F1、balanced accuracy、AUROC、PR-AUC negative、训练时间、推理时间。
3. 对 XGB 做特征组消融：user history、dst reputation、pair history、common-neighbor/status。
4. 对图模型补齐同一套指标，并检查 batch 内当前边是否进入同批次消息传播。
5. 至少跑 3 个 seed；低资源阶段可先跑 BitcoinOTC-1 和 BitcoinAlpha-1。

## 可结合方向

可复用和可改进的方向：

- XGB as strong baseline：作为论文中必须打败的历史统计强基线。
- XGB as feature auditor：用特征重要性/消融找出对 SEMBA 有价值的结构信号。
- SEMBA embeddings + XGB：用图模型产生节点/边 embedding，再交给 XGB 做最终分类。
- XGB features + SEMBA：把历史统计特征拼接到 SEMBA 的 pair classifier 输入。
- XGB early-exit：XGB 高置信样本直接预测，低置信样本再调用图模型，节约计算。
- XGB teacher：用 XGB 伪标签或软概率辅助训练小型图模型。

当前最值得优先做的是：`XGB features + SEMBA classifier` 和
`XGB early-exit / hard-case reranking`，因为它们最符合低资源优势。
