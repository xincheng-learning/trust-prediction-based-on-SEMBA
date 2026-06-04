# SEMBA 方向讨论总结：动态符号信任预测的可行改进路线

## 0. 当前研究定位

当前方向不再优先依赖 Epinions raw 的额外交互图，因为不同数据源不一定都包含“用户—内容—作者”交互信息。更稳妥的主线应基于 SEMBA GitHub 项目中的动态符号网络数据：

- Bitcoin-Alpha
- Bitcoin-OTC
- Epinions
- Wiki-RfA

这些数据的共同结构大致是：

```text
src, dst, sign, timestamp
```

即：

```text
源节点, 目标节点, 正负关系, 时间
```

因此最合理的研究任务是：

```text
Dynamic Signed Social Relation Prediction
```

更具体地说，优先做：

```text
Dynamic Link Sign Prediction
```

即：

```text
给定未来某条边会出现，预测它是正边还是负边。
```

在信任网络语境中就是：

```text
给定未来 u -> v 存在关系，预测它是 trust 还是 distrust。
```

---

## 1. SEMBA 原论文与项目的核心思想

SEMBA 研究的是：

```text
continuous-time dynamic signed networks
```

也就是同时具备：

1. 动态性：边按时间发生；
2. 符号性：边有正负关系；
3. 有向性：u -> v 和 v -> u 不等价。

SEMBA 的核心机制是：

```text
signed memory + balanced aggregation
```

每个节点维护两类记忆：

```text
positive memory
negative memory
```

正边和负边的传播方式不同：

```text
正边 u -> v:
    同极性传播
    positive <- positive
    negative <- negative

负边 u -> v:
    反极性传播
    positive <- negative
    negative <- positive
```

这来自结构平衡理论：

```text
朋友的朋友更可能是朋友
敌人的敌人更可能是朋友
朋友的敌人更可能是敌人
敌人的朋友更可能是敌人
```

SEMBA 已经解决了“正负边传播方式不同”这个问题，但它对“信任不对称”和“地位差异”的建模还不充分。

---

## 2. 需要避免的误区

### 2.1 不要把所有数据都当成交互图

Bitcoin、SEMBA Epinions、Wiki-RfA 主要是 signed temporal edge stream。

它们不一定有额外的：

```text
用户评论
用户评分
用户对内容作者的交互
```

所以当前主线不能依赖“用户—内容—作者交互图”。

更稳妥的是用图本身构造结构型不对称。

---

### 2.2 不要直接相信 Epinions 是高质量连续时间流

当前下载的 SEMBA Epinions 原始边文件 `out.epinions` 是：

```text
src dst sign timestamp
```

但它存在明显时间聚集问题：

```text
第一天聚集了大量边
```

因此 Epinions 不适合直接做严格 continuous-time event stream。

Epinions 更适合：

```text
blocked temporal protocol
post-burst protocol
initial snapshot + later dynamic prediction
```

Bitcoin-Alpha / Bitcoin-OTC 的时间更干净，更适合优先验证动态模型。

---

## 3. 最终收敛出的创新点

建议创新点名称：

```text
Asymmetry-aware Liquid Signed Memory Network
```

中文可以写成：

```text
不对称感知的液态符号记忆网络
```

核心思想：

```text
动态符号信任预测中，未来边的正负不仅由历史 signed graph 决定，
还受到两类方向性因素影响：

1. 覆盖不对称
2. 地位不对称

同时，信任边和不信任边应通过不同传播通道演化，
并用液态时间记忆机制处理不规则时间间隔下的动态变化。
```

---

## 4. 创新点一：覆盖不对称

LGDM consensus 论文中有一个非对称隐式信任思想：

```text
Par(u, v) = |I_u ∩ I_v| / |I_u|
```

它的方向性来自分母：

```text
Par(u, v) != Par(v, u)
```

含义是：

```text
v 的经验世界在多大程度上覆盖了 u 的经验世界。
```

但 SEMBA 数据不一定有交互集合，因此可以把它泛化为历史邻居集合：

```text
N_u(t): 节点 u 在 t 之前的历史 signed 邻居集合
```

定义：

```text
CA(u -> v, t) = |N_u(t) ∩ N_v(t)| / (|N_u(t)| + eps)
```

其中：

```text
CA = Coverage Asymmetry
```

这表示：

```text
v 的历史关系环境对 u 的历史关系环境的覆盖程度。
```

需要同时保留反方向：

```text
CA(u -> v, t)
CA(v -> u, t)
```

因为二者一般不相等。

---

## 5. 创新点二：地位不对称

地位不对称比交互图更通用，因为任何有向 signed social network 都有地位结构。

定义一个动态地位分数：

```text
q_u(t) = log(1 + in_pos_u(t)) - log(1 + in_neg_u(t))
```

其中：

```text
in_pos_u(t): t 之前指向 u 的正边数量
in_neg_u(t): t 之前指向 u 的负边数量
```

然后定义地位差：

```text
SG(u -> v, t) = q_v(t) - q_u(t)
```

其中：

```text
SG = Status Gap
```

解释：

```text
如果 SG(u -> v, t) > 0，
说明目标节点 v 的历史地位高于源节点 u。

如果 SG(u -> v, t) < 0，
说明目标节点 v 的历史地位低于源节点 u。
```

这对应 status theory 的思想：

```text
正边常常指向更高地位者；
负边可能反映地位否定、冲突或低评价。
```

---

## 6. 创新点三：不对称调节的正负边传播

SEMBA 已经区分正负边传播通道，但传播强度没有显式受不对称因素调节。

可以加入一个不对称传播权重：

```text
alpha(u, v, t) = MLP[
    CA(u -> v, t),
    CA(v -> u, t),
    SG(u -> v, t),
    delta_t
]
```

其中：

```text
delta_t = 当前事件与节点上次更新之间的时间间隔
```

然后用它调节 message：

```text
message(u <- v, t) = alpha(u, v, t) * original_message(v, t)
```

解释：

```text
不只是正边和负边传播方式不同，
而且传播强度取决于覆盖不对称、地位差异和时间间隔。
```

---

## 7. 创新点四：液态时间记忆更新

DUTrust 液态神经网络论文的启发是：

```text
信任是动态演化的，且受时间间隔、行为变化和上下文变化影响。
```

但 DUTrust 不是 signed graph 模型，也没有 SEMBA 的正负记忆传播机制。

因此不要直接照搬 DUTrust。

更合理的借鉴方式是：

```text
用 Liquid Time-Constant / Liquid Neural Network 替代 SEMBA 中的普通 memory update。
```

原 SEMBA 可能类似：

```text
s_u(t) = RNN_or_GRU_or_LSTM(s_u(t^-), message_u(t))
```

可以改为：

```text
s_u_pos(t) = LTC(s_u_pos(t^-), message_u_pos(t), delta_t)
s_u_neg(t) = LTC(s_u_neg(t^-), message_u_neg(t), delta_t)
```

含义：

```text
正记忆和负记忆分别用液态时间单元更新；
更新时间显式使用 delta_t；
更适合不规则时间间隔的动态信任演化。
```

优先在 Bitcoin-Alpha / Bitcoin-OTC 上验证，因为它们的时间戳质量更适合连续时间建模。

---

## 8. signed weight 的含义与可选扩展

signed edge 是：

```text
sign in {+1, -1}
```

表示：

```text
正关系 / 负关系
trust / distrust
support / oppose
```

signed weight 是更细粒度的边强度，例如 Bitcoin 数据：

```text
weight in [-10, +10]
```

含义：

```text
+10: 强信任
+1 : 弱信任
-1 : 弱不信任
-10: 强不信任
```

SEMBA 原论文结论提到未来可以做：

```text
predicting the signed weights of dynamic links
```

也就是不仅预测正负号，还预测边权强度。

这适合：

```text
Bitcoin-Alpha
Bitcoin-OTC
```

不适合：

```text
Epinions
Wiki-RfA
```

因为后两者通常只有正负标签，没有连续权重。

---

## 9. 数据集使用建议

### 9.1 Bitcoin-Alpha / Bitcoin-OTC

最适合做主验证。

原因：

```text
有时间
有方向
有正负
有权重
时间质量相对更好
```

可做任务：

```text
1. sign prediction
2. signed weight regression
```

优先用于验证：

```text
liquid memory update
status asymmetry
coverage asymmetry
```

---

### 9.2 SEMBA Epinions

适合做经典 signed trust/distrust 网络实验，但需要谨慎处理时间。

建议协议：

```text
1. initial snapshot:
   把首日大桶作为初始历史图

2. post-burst prediction:
   只预测首日之后的边

3. blocked temporal protocol:
   按天、周或月做窗口预测
```

不建议：

```text
直接把所有 timestamp 当成严格连续事件流
```

---

### 9.3 Wiki-RfA

它不是 trust/distrust，而是 support/oppose。

适合作为泛化验证：

```text
signed social relation prediction
```

不能在论文里直接写成 trust prediction。

---

## 10. 推荐实验路线

### 阶段 1：最低风险版本

只改 decoder，不动 SEMBA memory。

```text
Baseline:
    SEMBA

改进 1:
    SEMBA + coverage asymmetry decoder

改进 2:
    SEMBA + status asymmetry decoder

改进 3:
    SEMBA + coverage + status decoder
```

decoder 输入可以是：

```text
[z_u(t), z_v(t),
 CA(u -> v, t),
 CA(v -> u, t),
 q_u(t),
 q_v(t),
 q_v(t) - q_u(t)]
```

目标：

```text
判断双重不对称是否能提升 sign prediction。
```

---

### 阶段 2：中等风险版本

把不对称性放入传播过程。

```text
alpha(u, v, t) = MLP(CA_uv, CA_vu, status_gap, delta_t)

message = alpha * original_message
```

目标：

```text
验证不对称性不只是 decoder 特征，
而是能调节 signed message passing。
```

---

### 阶段 3：较高风险版本

替换 memory update 为液态时间单元。

```text
positive memory update -> LTC
negative memory update -> LTC
```

目标：

```text
验证液态神经网络是否能更好处理动态信任演化。
```

优先在 Bitcoin 数据上做。

---

### 阶段 4：可选扩展

只在 Bitcoin 数据上做：

```text
signed weight prediction
```

任务从分类变成回归：

```text
predict weight in [-10, +10]
```

或者先简化成：

```text
predict abs(weight) as trust strength
predict sign separately
```

---

## 11. 建议的消融实验

必须做以下消融：

```text
SEMBA
SEMBA + CA
SEMBA + Status
SEMBA + CA + Status
SEMBA + CA + Status + Asymmetry-weighted message
SEMBA + CA + Status + Liquid memory
```

其中：

```text
CA = coverage asymmetry
Status = dynamic status asymmetry
Liquid memory = LTC/LNN memory update
```

核心观察指标：

```text
AUC
F1-macro
negative-class F1
balanced accuracy
AP / PR-AUC
calibration if possible
```

不能只看总体 AUC，因为 signed network 里正边通常远多于负边。

---

## 12. 对 Codex 的具体任务提示

Codex 进入 SEMBA 项目文件夹后，优先做这些事：

### 12.1 理解项目结构

先阅读：

```text
README.md
train.py
parser.py
model_wrapper.py
dataset_loaders/
src/
```

目标：

```text
找出：
1. 数据在哪里加载
2. sign_class 任务如何构造
3. SEMBA 模型在哪里定义
4. memory update 在哪里实现
5. decoder 在哪里实现
```

---

### 12.2 先不要直接改全部模型

第一步只新增特征并改 decoder。

需要实现：

```text
compute_dynamic_status(edge_history, t)
compute_coverage_asymmetry(edge_history, u, v, t)
```

输出：

```text
CA_uv
CA_vu
q_u
q_v
status_gap
```

然后拼到 SEMBA 原 decoder 输入中。

---

### 12.3 再考虑 message weight

如果 decoder 版本有效，再实现：

```text
asymmetry-aware message weighting
```

即：

```text
message = alpha(CA_uv, CA_vu, status_gap, delta_t) * message
```

---

### 12.4 最后再考虑 Liquid memory

如果前两步有效，再尝试：

```text
replace RNN/GRU/LSTM memory updater with LTC/LNN-style update
```

不要一开始就改 memory，否则排错成本太高。

---

## 13. 当前最清晰的论文创新表达

可以这样概括：

```text
本文面向动态符号信任预测，提出一种不对称感知的符号记忆网络。
不同于仅依赖结构平衡理论的动态 signed GNN，本文进一步建模两类方向性机制：
一是历史邻居覆盖产生的覆盖不对称，
二是历史正负入边产生的动态地位不对称。
在此基础上，模型使用不对称强度调节正负边消息传播，并进一步探索液态时间记忆单元对不规则时间信任演化的建模能力。
```

---

## 14. 当前最稳结论

最稳的主线不是：

```text
交互图 + 地位 + LNN + signed weight 全部一起做
```

而是：

```text
基于 SEMBA 动态 signed network 框架，
加入覆盖不对称和地位不对称，
再逐步尝试不对称传播权重和液态记忆更新。
```

优先级：

```text
1. SEMBA + coverage/status decoder
2. SEMBA + asymmetry-aware message weighting
3. SEMBA + liquid memory update
4. Bitcoin 上做 signed weight prediction
```

这样既不依赖额外交互数据，又覆盖了当前讨论中的四个核心思想：

```text
信任不对称
时间动态
不信任边
信任边和不信任边传播方式不同
```
