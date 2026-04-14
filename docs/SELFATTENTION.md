# SelfAttention Encoder-Decoder 详细说明

更新时间：2026-04-13

本文档采用你要求的 `encoder-decoder` 口径描述 SelfAttention 路线，并把它明确写成相对 `dyppo` 的进一步升级。

## 1. SelfAttention 的定位

如果把三种方法放在一条升级线上，可以写成：

```text
efficient design
    -> dyppo
    -> selfattention encoder-decoder
```

它们的差异不是简单的“谁更深的网络”，而是设计空间控制能力不同：

### 1.1 Efficient Design
- 固定变量
- 固定每个变量的 attribute levels
- 在离散空间里做最优组合搜索

### 1.2 dynamicPPO
- 仍然固定变量
- 仍然固定每个变量的 attribute levels
- 但可以根据 respondent 和历史反馈自动调整 combo 的选择与分发

### 1.3 SelfAttention Encoder-Decoder
- 不再只是在固定 combo 池里选题
- 可以决定哪些变量激活，哪些变量不激活
- 变量值可以直接生成连续值
- 因而变量值不再局限于预先枚举好的离散 levels

所以这里的升级核心是：

```text
从“固定离散组合选择”升级到“条件约束下的结构化题目生成”。
```

## 2. SelfAttention 相对 dyppo 的升级边界

这部分单独写清楚。

### 2.1 dyppo 的边界

`dyppo` 的设计边界是：
- variables 不变
- variables 的 attribute levels 不变
- action 只是从离散 candidate pool 里选一个 task

### 2.2 encoder-decoder 的升级点

SelfAttention encoder-decoder 可以进一步扩展为：
- 变量可有可无：通过 `mask` 决定变量是否激活
- 变量值可连续：通过 `value head` 生成连续值，而不是只在预设 levels 中挑一个
- block 可生成：不只从现成题库里挑，而是一步一步生成一个 block

因此它的动作不再只是：

```text
选一个离散 combo
```

而是扩展为：

```text
决定当前题要激活哪些变量，并为这些变量生成具体取值
```

## 3. 输入张量的总体定义

本文按你前面讨论过的架构口径，把输入分成 4 组：

```text
X_rp      : respondent / RP 信息
X_env     : 环境与全局收集状态
X_sp_ctx  : 当前 SP 上下文与历史已生成题组
X_cand    : 候选变量或候选属性模板信息
```

### 3.1 `X_rp`

表示当前 respondent 的静态或半静态特征，例如：
- gender
- age_group
- education
- zone_id
- household / profile 派生指标

若 batch size 为 `B`，RP 特征维度为 `d_rp`，则：

```text
X_rp ∈ R^[B, d_rp]
```

### 3.2 `X_env`

表示全局环境与在线采集状态，例如：
- 当前累计样本量占目标样本量的比例
- 当前 zone 的样本偏差
- 当前 design 的分发次数
- 当前参数估计稳定度
- 当前信息增益 / D-error 水平

若环境特征维度为 `d_env`，则：

```text
X_env ∈ R^[B, d_env]
```

### 3.3 `X_sp_ctx`

表示当前 respondent 在本次问卷生成时的上下文，例如：
- 已经生成了多少题
- 已生成题的变量激活情况
- 已生成题的变量值统计
- 已生成题的 expected utility spread
- block 当前位置

若当前 block 长度上限为 `L_ctx`，上下文单步维度为 `d_ctx`，则：

```text
X_sp_ctx ∈ R^[B, L_ctx, d_ctx]
```

### 3.3.1 `X_sp_ctx` 的两层构造思路

更具体地说，`X_sp_ctx` 通常不是一条“单独的标量特征”，而是由两部分组成：

1. 历史题目级 token
   记录当前 respondent 已经生成出的前 `t-1` 道题各自的结构与统计量。
2. 历史汇总 token
   记录截至当前时刻，对前 `t-1` 道题做汇总后得到的整体统计。

若记当前正在生成第 `t` 道题，则历史长度为：

```math
n_t = t - 1
```

推荐的 `X_sp_ctx` 组织方式是：

```math
X_{sp\_ctx}^{(t)}
=
\operatorname{concat}
\bigl(
g_1,\ldots,g_{n_t},
s_{ctx}^{(t)},
\text{padding}
\bigr)
```

其中：
- `g_j` 是第 `j` 道已生成题对应的 question-level context token
- `s_ctx^(t)` 是截至第 `t-1` 道题的 aggregate summary token
- 若 `n_t + 1 < L_ctx`，则剩余位置用 padding 补齐

### 3.3.2 变量激活情况如何统计

设系统把全部可能出现的变量槽位按统一顺序展开成长度为 `V` 的索引表。这里的一个“变量槽位”通常指：

```text
(alternative, variable_name)
```

例如：
- `car.time`
- `pt.wait_time`
- `taxi.cost`

对第 `j` 道已生成题，记其变量激活向量为：

```math
m_j = (m_{j,1}, m_{j,2}, \ldots, m_{j,V}), \quad m_{j,v} \in \{0,1\}
```

其中：
- `m_{j,v} = 1` 表示第 `j` 题中第 `v` 个变量槽位被激活
- `m_{j,v} = 0` 表示该变量槽位未被激活

到第 `t` 步之前，变量 `v` 的累计激活次数为：

```math
c_v^{(t)} = \sum_{j=1}^{n_t} m_{j,v}
```

对应的累计激活频率为：

```math
\rho_v^{(t)} = \frac{c_v^{(t)}}{\max(n_t, 1)}
```

如果希望模型更关注“最近几题”的结构趋势，也可以使用长度为 `W` 的滑动窗口版本：

```math
\rho_{v,W}^{(t)}
=
\frac{
\sum_{j=\max(1, n_t-W+1)}^{n_t} m_{j,v}
}{
\max(\min(W, n_t), 1)
}
```

因此，“已生成题的变量激活情况”在实现上通常不是一句口头描述，而是一个向量：

```math
\rho^{(t)} = (\rho_1^{(t)}, \rho_2^{(t)}, \ldots, \rho_V^{(t)})
```

它描述的是：截至当前，哪些变量被频繁激活，哪些变量很少或从未出现。

### 3.3.3 变量值统计如何计算

对第 `j` 道已生成题，记归一化后的变量值向量为：

```math
\tilde{x}_j = (\tilde{x}_{j,1}, \tilde{x}_{j,2}, \ldots, \tilde{x}_{j,V})
```

这里推荐先做边界归一化，再送入上下文统计：

```math
\tilde{x}_{j,v}
=
\frac{x_{j,v} - l_v}{u_v - l_v + \varepsilon}
```

其中：
- `x_{j,v}` 是原始生成值
- `l_v, u_v` 是该变量允许的下界和上界
- `\varepsilon` 是防止分母为 0 的很小常数

注意：如果变量 `v` 在第 `j` 题没有激活，则不能直接把其值当成真实观测值参与统计，而应通过 `m_{j,v}` 进行掩码控制。

因此，到第 `t` 步之前，变量 `v` 的有效观测次数为：

```math
n_v^{act,(t)} = \sum_{j=1}^{n_t} m_{j,v}
```

变量 `v` 的激活条件下均值为：

```math
\mu_v^{(t)}
=
\frac{
\sum_{j=1}^{n_t} m_{j,v}\tilde{x}_{j,v}
}{
\max(n_v^{act,(t)}, 1)
}
```

变量 `v` 的激活条件下方差与标准差为：

```math
(\sigma_v^{(t)})^2
=
\frac{
\sum_{j=1}^{n_t} m_{j,v}\bigl(\tilde{x}_{j,v}-\mu_v^{(t)}\bigr)^2
}{
\max(n_v^{act,(t)}, 1)
}
```

```math
\sigma_v^{(t)} = \sqrt{(\sigma_v^{(t)})^2}
```

如果需要更丰富的上下文，还可以继续保留：

```math
\tilde{x}_{v,\min}^{(t)} = \min_{j:m_{j,v}=1}\tilde{x}_{j,v}
```

```math
\tilde{x}_{v,\max}^{(t)} = \max_{j:m_{j,v}=1}\tilde{x}_{j,v}
```

所以“已生成题的变量值统计”最常见的实现不是单个数，而是一组按变量展开的统计向量，例如：

```math
\mu^{(t)}, \sigma^{(t)}, \tilde{x}_{\min}^{(t)}, \tilde{x}_{\max}^{(t)}
```

### 3.3.4 expected utility spread 如何计算

这是 `X_sp_ctx` 中非常关键但容易写虚的一项。它本质上是在问：

```text
当前已经生成出的题，在现有参数估计下，备选项之间的效用差距有多大？
```

对第 `j` 道已生成题，设其备选项集合为 `\mathcal{A}`。对其中某个备选项 `a`，若当前可用参数估计为 `\hat{\beta}`，则其系统效用可写为：

```math
u_{j,a}
=
\operatorname{ASC}_a
+
\sum_{v=1}^{V} m_{j,a,v}\hat{\beta}_{a,v}x_{j,a,v}
```

其中：
- `m_{j,a,v}` 表示第 `j` 题里备选项 `a` 的第 `v` 个变量是否激活
- `x_{j,a,v}` 表示对应变量值
- `\hat{\beta}_{a,v}` 可以来自：
  - 初始先验参数
  - 最近一轮参数重估结果
  - 或在线更新后的当前参数估计

第 `j` 道题的 expected utility spread 定义为：

```math
\operatorname{spread}_j
=
\max_{a \in \mathcal{A}} u_{j,a}
-
\min_{a \in \mathcal{A}} u_{j,a}
```

它的含义是：
- spread 很大：题目可能接近主导/被主导，区分度虽然强，但可能太“容易选”
- spread 很小：题目可能过于接近，respondent 选择会更随机

因此常用的不是只保留单题 spread，而是保留历史 spread 的统计量，例如：

```math
\bar{s}^{(t)}
=
\frac{1}{\max(n_t,1)}
\sum_{j=1}^{n_t}\operatorname{spread}_j
```

```math
s_{\max}^{(t)} = \max_{1 \le j \le n_t}\operatorname{spread}_j
```

```math
s_{last}^{(t)} =
\begin{cases}
\operatorname{spread}_{n_t}, & n_t \ge 1 \\
0, & n_t = 0
\end{cases}
```

在真正送入模型之前，还可以把 spread 做一个有界化处理，例如：

```math
\tilde{s}_j = \frac{\operatorname{spread}_j}{1 + |\operatorname{spread}_j|}
```

这样可以避免极端大 spread 在数值上压制其它上下文特征。

### 3.3.5 推荐的上下文 token 组成

如果把每道已生成题编码为一个 question token，则第 `j` 个 token 可以写成：

```math
g_j
=
\Bigl[
\frac{j}{T_{block}},
\frac{\sum_{v=1}^{V} m_{j,v}}{V},
\tilde{s}_j,
m_j,
\tilde{x}_j
\Bigr]
```

其原始维度约为：

```math
d_{ctx,step} = 3 + V + V = 3 + 2V
```

对应的汇总 token 可以写成：

```math
s_{ctx}^{(t)}
=
\Bigl[
\frac{n_t}{T_{block}},
\rho^{(t)},
\mu^{(t)},
\sigma^{(t)},
\bar{s}^{(t)},
s_{\max}^{(t)},
s_{last}^{(t)}
\Bigr]
```

其原始维度约为：

```math
d_{ctx,sum} = 1 + V + V + V + 3 = 3V + 4
```

由于 `d_{ctx,step}` 与 `d_{ctx,sum}` 不一定相同，工程上通常会先各自过一层投影网络，再映射到统一的 `d_ctx`：

```math
\operatorname{Proj}_{step}: d_{ctx,step} \to d_{ctx}
```

```math
\operatorname{Proj}_{sum}: d_{ctx,sum} \to d_{ctx}
```

最终再拼成：

```math
X_{sp\_ctx}^{(t)} \in \mathbb{R}^{[B, L_{ctx}, d_{ctx}]}
```

### 3.4 `X_cand`

这里不再把它理解为固定的离散 combo 池，而是理解为：
- 候选变量集合
- 候选变量的边界信息
- 候选变量的逻辑约束信息
- 变量所属 alternative / group 的结构信息

若候选变量 token 数量为 `L_cand`，每个 token 维度为 `d_cand`，则：

```text
X_cand ∈ R^[B, L_cand, d_cand]
```

## 4. Encoder 的输入与输出

### 4.1 输入整合

先把四类输入投影到统一模型维度 `d_model`：

```text
Embed_rp   : d_rp   -> d_model
Embed_env  : d_env  -> d_model
Embed_ctx  : d_ctx  -> d_model
Embed_cand : d_cand -> d_model
```

得到：

```text
E_rp   ∈ R^[B, 1, d_model]
E_env  ∈ R^[B, 1, d_model]
E_ctx  ∈ R^[B, L_ctx, d_model]
E_cand ∈ R^[B, L_cand, d_model]
```

然后拼接为 encoder 输入：

```text
X_enc = concat(E_rp, E_env, E_ctx, E_cand)
X_enc ∈ R^[B, L_enc, d_model]
```

其中：

```text
L_enc = 1 + 1 + L_ctx + L_cand
```

### 4.2 Encoder 做什么

Encoder 的职责是：
- 把 respondent、环境、当前上下文、候选变量模板编码成统一的上下文表示
- 让模型知道“当前是谁”“当前采集到了什么程度”“还有哪些变量可以用”“当前 block 生成到哪一步”

经过多层 self-attention 后，输出：

```text
H_enc ∈ R^[B, L_enc, d_model]
```

这就是 decoder 后续条件生成时要读取的“上下文记忆”。

## 5. Decoder 的输入与输出

### 5.1 Decoder 输入是什么

Decoder 输入不是 respondent 原始特征，而是“已经生成出的题目 token 序列”。

记：
- 当前 block 允许生成最多 `T_block` 道题
- 每道题都会被编码成一个 decoder token

则 decoder 输入为：

```text
Y_in ∈ R^[B, T_dec, d_model]
```

其中：
- 第一个 token 可以是 `BOS` / block-begin token
- 后续 token 表示已经生成出的前 `t-1` 道题

### 5.2 Decoder 过程

每一步 decoder 做三件事：
1. masked self-attention
   只能看当前步之前已经生成出的题
2. cross-attention
   读取 encoder 的上下文 `H_enc`
3. FFN
   输出当前步隐藏表示 `h_t`

所以 decoder 单步输出隐藏状态：

```text
h_t ∈ R^[B, d_model]
```

## 6. Decoder 输出 head 的设计

为了实现“变量可有可无、变量值可连续”，decoder 不应该只接一个离散 softmax，而应拆成多个 head。

### 6.1 `mask_head`

作用：
- 决定每个变量是否激活

若变量总数为 `V`，则：

```text
mask_logits ∈ R^[B, V]
mask_prob   ∈ [0,1]^[B, V]
mask        ∈ {0,1}^[B, V]
```

解释：
- `mask[v] = 1` 表示该变量在当前题中启用
- `mask[v] = 0` 表示该变量在当前题中关闭

这就是“变量可有可无”的实现入口。

### 6.2 `value_head`

作用：
- 为每个变量生成数值

若变量总数为 `V`，则：

```text
raw_value ∈ R^[B, V]
```

再通过边界映射转成实际值，例如：

```text
value_v = lower_v + sigmoid(raw_value_v) * (upper_v - lower_v)
```

或：

```text
value_v = center_v + scale_v * tanh(raw_value_v)
```

这样得到：

```text
value ∈ R^[B, V]
```

它可以是连续值，因此不需要预先枚举离散 levels。

### 6.3 `score_head`

作用：
- 评价当前生成题的质量或优先级
- 也可作为训练时的辅助目标，用于评估当前题是否值得发给 respondent

输出：

```text
score ∈ R^[B, 1]
```

### 6.4 `stop_head`

作用：
- 决定 block 是否生成结束

输出：

```text
stop_logit ∈ R^[B, 1]
stop_prob  ∈ [0,1]^[B, 1]
```

若当前已经达到最小题数，并且 `stop_prob` 超过阈值，则可以停止生成 block。

## 7. Q / K / V 的维度流转

设：
- `B`：batch size
- `L`：序列长度
- `d_model`：模型维度
- `h`：head 数
- `d_h = d_model / h`

### 7.1 Encoder 自注意力

输入：

```text
X_enc ∈ R^[B, L_enc, d_model]
```

投影：

```text
Q_e, K_e, V_e ∈ R^[B, L_enc, d_model]
```

reshape：

```text
Q_e, K_e, V_e ∈ R^[B, h, L_enc, d_h]
```

注意力分数：

```text
S_e = Q_e K_e^T / sqrt(d_h)
S_e ∈ R^[B, h, L_enc, L_enc]
```

softmax 后：

```text
A_e ∈ R^[B, h, L_enc, L_enc]
```

输出：

```text
O_e = A_e V_e ∈ R^[B, h, L_enc, d_h]
concat(O_e) ∈ R^[B, L_enc, d_model]
```

### 7.2 Decoder masked self-attention

输入：

```text
Y_in ∈ R^[B, T_dec, d_model]
```

得到：

```text
Q_d, K_d, V_d ∈ R^[B, h, T_dec, d_h]
```

masked attention 分数：

```text
S_d ∈ R^[B, h, T_dec, T_dec]
```

上三角未来位会被 mask 掉，所以 decoder 只能看已生成历史。

### 7.3 Decoder cross-attention

Query 来自 decoder：

```text
Q_cross ∈ R^[B, h, T_dec, d_h]
```

Key / Value 来自 encoder：

```text
K_cross, V_cross ∈ R^[B, h, L_enc, d_h]
```

分数矩阵：

```text
S_cross ∈ R^[B, h, T_dec, L_enc]
```

输出再拼回：

```text
O_cross ∈ R^[B, T_dec, d_model]
```

## 8. encoder-decoder 与 dyppo 的建模差异

本节只讨论两种路线的建模对象、适用场景与计算代价差异，而不做简单的“强弱”判断。

### 8.1 dyppo 的建模范围

dyppo 适合处理的问题是：
- 在一个固定离散题库里选题
- 根据 respondent 与历史反馈动态重排题目组合

在当前设计边界下，它通常不负责：
- 自动决定某个变量不出现
- 自动给变量生成一个新的连续值
- 自动改变题目的结构模板

### 8.2 encoder-decoder 的建模扩展

encoder-decoder 的不同点在于，它把“题目结构本身”也纳入可学习对象。具体包括：

- 把变量激活当成一个结构决策问题
- 把变量取值当成一个连续生成问题
- 把 block 的长度、结构和参数共同作为可学习对象

因此，从问题定义上看，它对应的是从：

```text
fixed discrete combo selection
```

扩展为：

```text
conditional structured question generation
```

### 8.3 计算代价与工程取舍

这种扩展并不意味着 encoder-decoder 在所有场景下都优于 dyppo。二者更准确地说是关注重点不同：

- `dyppo`
  - 优点是设计空间清晰，动作定义明确，在线更新相对直接
  - 在变量与 levels 固定的前提下，更容易控制可行性与计算成本
  - 更适合“固定离散题库上的动态分发”问题

- `encoder-decoder`
  - 优点是结构表达更灵活，可以把变量激活、变量值生成和 block 结构一起建模
  - 代价是模型参数更多、训练更复杂、推理与约束修正成本也更高
  - 更适合“题目结构本身需要动态生成”的问题

因此，若研究目标仍然是“在固定离散设计空间中动态选题”，`dyppo` 往往已经足够；若研究目标进一步扩展到“变量是否出现、变量值如何生成、block 结构如何形成”，则更适合考虑 encoder-decoder。

## 9. Encoder-Decoder 的训练策略与 Loss Function

encoder-decoder 路线通常不适合“一上来就完全在线强化学习”。更稳妥的策略是分阶段训练：

1. 先用可解释、可控的数据做结构 warmup
2. 再逐步引入真实 respondent 反馈
3. 最后在在线阶段使用混合目标持续微调

### 9.1 训练样本的基本单位

训练的最自然单位不是“单个 respondent 的单次回答”，而是：

```text
一个 respondent 对应的一整个 block
```

设第 `b` 个训练样本对应的 teacher block 含 `T_b` 道题，则它的监督目标可写成：

```math
\mathcal{Y}^{(b)}
=
\Bigl\{
(m_t^*, x_t^*, q_t^*, z_t^*)
\Bigr\}_{t=1}^{T_b}
```

其中：
- `m_t^*`：第 `t` 题的目标激活 mask
- `x_t^*`：第 `t` 题的目标变量值
- `q_t^*`：第 `t` 题的目标质量分数
- `z_t^*`：第 `t` 步是否应结束 block 的 stop 标签

因此，decoder 在 teacher forcing 下的每一步并不是只预测一个类别，而是在预测：

```text
结构(mask) + 数值(value) + 质量(score) + 是否结束(stop)
```

### 9.2 阶段一：结构 warmup / teacher forcing

这一阶段的目标不是立即追求最优在线策略，而是先让模型学会：
- 生成语义上合理的题
- 生成满足约束的题
- 学会 block 内部的顺序结构

训练数据可以来自：
- efficient design 生成的参考 block
- dyppo 在历史数据中表现较好的 block
- 人工或规则系统校验通过的 block

训练时，第 `t` 步 decoder 的输入使用 teacher block 的前 `t-1` 题：

```math
Y_{in}^{(t)} = [\text{BOS}, y_1^*, y_2^*, \ldots, y_{t-1}^*]
```

模型输出：

```math
\hat{m}_t,\ \hat{x}_t,\ \hat{q}_t,\ \hat{z}_t
```

然后与目标：

```math
m_t^*,\ x_t^*,\ q_t^*,\ z_t^*
```

做监督学习。

### 9.3 `mask_head` 的损失函数

`mask_head` 本质上是对每个变量槽位做一个 Bernoulli 决策。因此最常见的损失是二元交叉熵：

```math
L_{mask}^{(t)}
=
- \frac{1}{V}
\sum_{v=1}^{V}
\Bigl[
m_{t,v}^* \log \hat{p}_{t,v}
+
(1-m_{t,v}^*) \log (1-\hat{p}_{t,v})
\Bigr]
```

其中：
- `\hat{p}_{t,v}` 是第 `t` 步第 `v` 个变量被激活的预测概率
- `m_{t,v}^*` 是目标标签

如果变量激活非常稀疏，也可以使用加权 BCE：

```math
L_{mask,w}^{(t)}
=
- \frac{1}{V}
\sum_{v=1}^{V}
\Bigl[
w_1 m_{t,v}^* \log \hat{p}_{t,v}
+
w_0 (1-m_{t,v}^*) \log (1-\hat{p}_{t,v})
\Bigr]
```

这样可以避免模型一味偏向“全部不激活”。

### 9.4 `value_head` 的损失函数

`value_head` 只应在“该变量被激活”时计算回归误差。否则，没有激活的变量不应被当成有效目标值参与训练。

因此，先把目标值和预测值都缩放到统一范围，例如 `[0,1]`：

```math
\tilde{x}_{t,v}^*
=
\frac{x_{t,v}^* - l_v}{u_v - l_v + \varepsilon}
```

```math
\hat{\tilde{x}}_{t,v}
=
\frac{\hat{x}_{t,v} - l_v}{u_v - l_v + \varepsilon}
```

再定义 masked regression loss。常见选择是 L1、SmoothL1 或 MSE。若用 L1，可写成：

```math
L_{value}^{(t)}
=
\frac{
\sum_{v=1}^{V} m_{t,v}^* \left| \hat{\tilde{x}}_{t,v} - \tilde{x}_{t,v}^* \right|
}{
\max\left(\sum_{v=1}^{V} m_{t,v}^*, 1\right)
}
```

这样保证：
- 未激活变量不会产生伪误差
- 每一步损失会按真实激活变量数做归一化

### 9.5 `score_head` 的监督目标与损失函数

`score_head` 不应被理解为最终的 respondent 选择概率，而更适合被理解为：

```text
当前生成题在训练目标下的“质量估计”或“优先级估计”
```

因此需要先定义一个 teacher score `q_t^*`。一个常见做法是把多个指标归一化后线性组合：

```math
q_t^*
=
\alpha_1 \widetilde{\Delta D}_t
+
\alpha_2 \widetilde{IG}_t
+
\alpha_3 \widetilde{CoverageGain}_t
-
\alpha_4 \widetilde{DominancePenalty}_t
-
\alpha_5 \widetilde{ConstraintPenalty}_t
```

其中：
- `\widetilde{\Delta D}_t`：D-error 改进量的归一化值
- `\widetilde{IG}_t`：信息增益的归一化值
- `\widetilde{CoverageGain}_t`：覆盖改善的归一化值
- `\widetilde{DominancePenalty}_t`：主导/被主导惩罚
- `\widetilde{ConstraintPenalty}_t`：约束违反惩罚

在这种定义下，`score_head` 最常用 MSE 回归：

```math
L_{score}^{(t)} = \left(\hat{q}_t - q_t^*\right)^2
```

如果更关注排序，也可以改成 pairwise ranking loss，但在问卷系统里，MSE 往往已经足够稳定。

### 9.6 `stop_head` 的损失函数

`stop_head` 的目标是学习：

```text
当前 block 是否应在第 t 步结束
```

设目标 stop 标签为：

```math
z_t^* \in \{0,1\}
```

预测概率为：

```math
\hat{\pi}_t^{stop} \in [0,1]
```

则其 BCE 损失为：

```math
L_{stop}^{(t)}
=
-
\Bigl[
z_t^* \log \hat{\pi}_t^{stop}
+
(1-z_t^*)\log(1-\hat{\pi}_t^{stop})
\Bigr]
```

在实际使用中，通常还会配合最小题量约束：

```math
t < T_{min} \Rightarrow z_t^* = 0
```

也就是说，在未达到最小题数前，stop 标签被强制设为 0。

### 9.7 约束修正损失

如果系统允许连续值生成，仅靠 `value_head` 回归还不够，因为还需要保证：
- 值在允许上下界内
- 变量之间满足逻辑约束
- 不出现显然不可行的题目结构

因此建议额外加入一个约束惩罚项：

```math
L_{cons}^{(t)}
=
\lambda_{bound} L_{bound}^{(t)}
+
\lambda_{logic} L_{logic}^{(t)}
+
\lambda_{dom} L_{dom}^{(t)}
```

其中：
- `L_bound` 处罚越界值
- `L_logic` 处罚违反逻辑条件的输出
- `L_dom` 处罚明显主导/被主导结构

若前向阶段已经做了“硬裁剪 + 硬约束修正”，这里的 `L_cons` 仍有意义，因为它能推动模型在训练时学会主动少犯这类错误，而不是每次都依赖后处理修正。

### 9.8 监督阶段总损失

把各个 head 的损失加权汇总，可得到 teacher forcing 阶段的总损失：

```math
L_{sup}
=
\sum_{t=1}^{T_b}
\Bigl(
\lambda_m L_{mask}^{(t)}
+
\lambda_v L_{value}^{(t)}
+
\lambda_q L_{score}^{(t)}
+
\lambda_s L_{stop}^{(t)}
+
\lambda_c L_{cons}^{(t)}
\Bigr)
+
\lambda_{reg}\|\theta\|_2^2
```

这里：
- `\lambda_m, \lambda_v, \lambda_q, \lambda_s, \lambda_c` 是各项损失权重
- `\lambda_reg` 是参数正则项权重

### 9.9 阶段二：scheduled sampling 与混合解码

如果模型只在纯 teacher forcing 下训练，推理时可能出现 exposure bias：

```text
训练时总是看见正确历史；推理时只能看见自己刚生成的历史。
```

因此在监督 warmup 稳定后，可以进入过渡阶段：
- 以概率 `p_teacher` 使用 teacher 历史 token
- 以概率 `1-p_teacher` 使用模型自己生成的历史 token

这个过程通常称为 scheduled sampling。其作用是逐步缩小训练分布和推理分布之间的差距。

### 9.10 阶段三：真实 respondent 反馈驱动微调

当模型已经具备基本的结构生成能力后，才适合引入在线反馈。此时训练目标不再只是模仿 teacher，而是要提升真实问卷发放效果。

设第 `b` 个 respondent 完成一个 block 后得到 block-level reward：

```math
R_b
=
w_1 \widetilde{\Delta D}_b
+
w_2 \widetilde{InfoGain}_b
+
w_3 \widetilde{CoverageImprove}_b
+
w_4 \widetilde{SensitivityGain}_b
-
w_5 Penalty_b
```

其中 `Penalty_b` 可以包含：
- 约束违反
- 题量过长
- 题目过于主导
- respondent 选择熵异常

### 9.10.1 本项目里一个 episode 的基本单位

这里需要先把系统单位说清楚，否则后面的 PPO 定义会混乱。

在本项目里，在线发放的最自然单位不是：
- 单一题目
- 也不是一个 respondent 的单个选项点击

而是：

```text
一个 respondent 对应一个 block
```

也就是说，模型的工作流程是：

1. 先读取当前 respondent 的 RP 信息与全局环境状态
2. 再一次性自回归生成一个 block 内的多道 SP 题
3. 把这一整个 block 一次性发给 respondent
4. respondent 一次性完成该 block 的回答并提交
5. 系统再根据这一整个 block 的回答结果回写状态并训练

因此：
- block 内部的“逐题生成”属于策略展开过程
- respondent 的真实选择反馈通常在 block 结束后才整体返回
- PPO 的一个 rollout episode 可以自然定义为“生成并完成一个 block”

### 9.10.2 本项目里的状态 `s_t` 定义

在第 `t` 步生成当前 block 的第 `t` 道题时，策略所看到的状态不是单一向量，而是一个结构化状态：

```math
s_t
=
\Bigl(
X_{rp}^{(b)},
X_{env}^{(g)},
X_{sp\_ctx}^{(t)},
X_{cand},
\tau_t
\Bigr)
```

其中：
- `X_{rp}^{(b)}`：当前第 `b` 个 respondent 的 RP 特征
- `X_{env}^{(g)}`：当前全局收集状态与环境统计
- `X_{sp_ctx}^{(t)}`：当前 block 已生成前 `t-1` 道题后的上下文
- `X_{cand}`：候选变量模板、变量边界、逻辑约束等静态候选信息
- `\tau_t = t / T_{block,max}`：当前 block 内的相对生成位置

如果写成张量形式，可以理解为：

```math
X_{rp}^{(b)} \in \mathbb{R}^{[1, d_{rp}]}
```

```math
X_{env}^{(g)} \in \mathbb{R}^{[1, d_{env}]}
```

```math
X_{sp\_ctx}^{(t)} \in \mathbb{R}^{[1, L_{ctx}, d_{ctx}]}
```

```math
X_{cand} \in \mathbb{R}^{[1, L_{cand}, d_{cand}]}
```

这里还有一个很关键的工程点：

```text
在同一个 block 的逐题生成过程中，respondent 的真实回答还没有返回，
因此 X_rp 基本不变，X_env 也通常只做轻微的“生成进度型”更新；
真正显著变化的是 X_sp_ctx。
```

也就是说，本项目里 block 内部的状态演化主要来自：
- 已生成题数量变化
- 已生成题的变量激活统计变化
- 已生成题的变量值统计变化
- 已生成题的 expected utility spread 与结构多样性变化

而不是来自 respondent 在 block 内逐题实时作答。

### 9.10.3 本项目里的动作 `a_t` 定义

在 encoder-decoder + PPO 口径下，第 `t` 步的动作不是“从题库里选一个离散题号”，而是一个复合结构动作：

```math
a_t = (m_t, x_t, z_t)
```

其中：

1. `mask` 动作

```math
m_t \in \{0,1\}^{V}
```

表示当前题中哪些变量被激活。

2. `value` 动作

```math
x_t \in \mathbb{R}^{V}
```

表示各变量的具体数值；只有 `m_{t,v}=1` 的槽位才被视为有效输出。

3. `stop` 动作

```math
z_t \in \{0,1\}
```

表示在生成完当前题后是否结束当前 block。

因此，策略分布可以写成：

```math
\pi_\theta(a_t \mid s_t)
=
\pi_\theta(m_t, x_t, z_t \mid s_t)
```

在实现上一般再分解为：

```math
\pi_\theta(a_t \mid s_t)
=
\pi_{mask}(m_t \mid s_t)
\cdot
\pi_{value}(x_t \mid s_t, m_t)
\cdot
\pi_{stop}(z_t \mid s_t)
```

这里还要特别强调：

```text
score_head 的输出不是环境动作本身，而是一个辅助估计量。
```

也就是说：
- `score_head` 可以用来估计当前题的优先级、预期价值或信息质量
- 但真正送进 PPO 概率比 `r_t(\theta)` 的动作，仍然是 `mask/value/stop`

### 9.10.4 本项目里的奖励 `r_t` 与 `R_b` 定义

由于 respondent 的真实回答是在整个 block 结束后才整体返回，所以奖励最自然地分成两层：

1. 题目生成阶段的逐步 shaping reward
2. respondent 完成 block 后的终端 block reward

#### A. 逐步 shaping reward

在第 `t` 步题目刚生成出来时，即使 respondent 还没回答，我们也可以对“题目结构质量”做即时评价：

```math
r_t^{shape}
=
\eta_1 \widetilde{Feasible}_t
+
\eta_2 \widetilde{Diversity}_t
+
\eta_3 \widetilde{SpreadTarget}_t
+
\eta_4 \widetilde{CoverageGain}_t
-
\eta_5 \widetilde{DominancePenalty}_t
-
\eta_6 \widetilde{RepeatPenalty}_t
-
\eta_7 \widetilde{ConstraintPenalty}_t
```

这里：
- `Feasible`：是否满足边界、逻辑、变量依赖等硬约束
- `Diversity`：相对当前 block 已有题目是否保持结构多样性
- `SpreadTarget`：expected utility spread 是否落在合理区间
- `CoverageGain`：是否让变量覆盖更均衡
- `DominancePenalty`：是否出现主导/被主导倾向
- `RepeatPenalty`：是否与前题过度重复
- `ConstraintPenalty`：是否需要大量后处理修正

#### B. block 结束后的终端奖励

当 respondent 完成一整个 block 后，系统拿到了：
- 该 respondent 的整组选择结果
- 更新后的参数估计结果
- 更新后的 D-error / 信息量 / 覆盖情况

这时可定义 block 级回报：

```math
R_b
=
w_1 \widetilde{reward_{D,b}}
+
w_2 \widetilde{reward_{H,b}}
+
w_3 \widetilde{\Delta \bar{\rho}_b^2}
+
w_4 \widetilde{ResponseValidity_b}
+
w_5 \widetilde{SensitivityGain_b}
-
w_6 \widetilde{BoundPenalty_b}
-
w_7 \widetilde{BurdenPenalty_b}
```

其中：

```math
reward_{D,b} = \frac{1}{1 + D_b}
```

也就是：
- `D-error` 越小，则 `reward_{D,b}` 越大

adjusted pseudo R^2 的增量采用：

```math
\Delta \bar{\rho}_b^2
=
\bar{\rho}_{after}^2 - \bar{\rho}_{before}^2
```

这里的 `\bar{\rho}^2` 明确指 adjusted McFadden pseudo `R^2`，也可以写成：

```math
\bar{\rho}^2
=
1 - \frac{LL(\hat{\beta}) - K}{LL(0)}
```

其中：
- `LL(\hat{\beta})`：估计后模型的对数似然
- `LL(0)`：空模型或仅常数项模型的对数似然
- `K`：参数个数

在本项目里的使用方式是：
- 先用当前 design 历史已收集样本重估一次 MNL，得到 `\bar{\rho}_{before}^2`
- 再把当前 respondent 这一个 block 的回答追加进去，重估一次 MNL，得到 `\bar{\rho}_{after}^2`
- 若该 block 让模型解释力提升，则 `\Delta \bar{\rho}_b^2 > 0`

参数越界惩罚采用：

```math
BoundPenalty_b
=
\operatorname{penalty}\bigl(\hat{\beta}_{raw,b}, [\beta_{min}, \beta_{max}]\bigr)
```

也就是：
- 先做一版未裁剪参数估计 `\hat{\beta}_{raw,b}`
- 再检查它是否落在前端 `sp_design.html` 中定义的 `beta_min / beta_max` 区间内
- 若越界，则按越界幅度产生惩罚

其余几项可以理解为：
- `reward_H_b`：block 内题目选择熵是否保持适中
- `SensitivityGain_b`：对敏感变量或关键替代项的辨识能力是否提升
- `ResponseValidity_b`：回答是否完整、是否满足基本一致性
- `BoundPenalty_b`：参数估计是否跑出预定义区间
- `BurdenPenalty_b`：题量过大、过难、无效题过多所带来的惩罚

#### B.1 当前项目实现里的“越界后大概率 mask 掉”是什么意思

理论上的 encoder-decoder 可以通过 `mask_head` 直接降低某些变量再次被激活的概率。

但在当前项目代码的实际实现里，SelfAttention 路线仍然是：
- 在统一的 feasible combo pool 上打分
- 再从候选题中选题

因此当前版本的“mask 掉”实现为：

```text
如果某个参数估计越界，就对包含该变量的 candidate task 整体降权。
```

也就是说，它在当前工程里更准确地是：

```text
candidate-level suppression
```

而不是 decoder 内部显式逐变量采样的硬 mask。

#### B.2 推荐的 block reward 组合指标

如果把当前项目的 reward 章节写成推荐口径，而不是只列可能项，更建议使用下面这组组合指标：

```math
R_b
=
\lambda_D \, reward_{D,b}
+
\lambda_H \, reward_{H,b}
+
\lambda_{\rho} \, \Delta \bar{\rho}_b^2
+
\lambda_S \, SensitivityGain_b
+
\lambda_V \, ResponseValidity_b
-
\lambda_B \, BoundPenalty_b
-
\lambda_U \, BurdenPenalty_b
```

其中更推荐的理解方式是：
- `reward_{D,b}`：回答加入后，这个 block 对设计效率是否有贡献
- `reward_{H,b}`：题目是否既不是完全主导，也不是完全随机
- `\Delta \bar{\rho}_b^2`：这一整个 block 是否提升了行为模型的整体解释力
- `SensitivityGain_b`：是否增强了关键变量的辨识能力
- `ResponseValidity_b`：回答是否完整、有效、一致
- `BoundPenalty_b`：参数估计是否跑出预定义区间
- `BurdenPenalty_b`：问卷负担是否过高

如果从工程优先级排序，更推荐：

1. `reward_{D,b}`
2. `\Delta \bar{\rho}_b^2`
3. `BoundPenalty_b`
4. `reward_{H,b}`
5. 其余辅助项

原因是：
- `D-error` 更偏设计效率
- adjusted McFadden pseudo `R^2` 更偏行为模型解释力
- 参数越界惩罚更偏可解释性与先验约束一致性

三者合起来，比只盯住某一个指标更稳。

#### B.3 为什么 adjusted McFadden pseudo R^2 在这里适合放在 block reward，而不是 step reward

这是 reward 设计里最容易混淆的一点。

adjusted McFadden pseudo `R^2` 更适合放在 block reward，而不适合放在单步 reward，原因是：

1. 它必须在 respondent 完成一个 block 后，结合整组选择结果一起重估 MNL，才有意义。
2. 它天然是 “before / after” 的整体模型指标，不是单题即时反馈。
3. 它更适合回答：

```text
当前这一整份 block 加进来之后，
整体行为模型是不是更能解释选择行为了？
```

所以：
- `step reward` 更适合用结构质量、可行性、spread、重复惩罚等局部指标
- `block reward` 更适合用 `D-error`、adjusted McFadden pseudo `R^2`、参数越界惩罚等整体指标

#### B.4 SP 在线生成系统的 reward 指标分层表

为了避免把“训练时 reward”“阶段性评估”“仪表盘展示”混在一起，更建议按四层来组织指标：

| 层级 | 主要作用 | 推荐指标 | 是否直接进入 PPO / RL reward | 说明 |
| --- | --- | --- | --- | --- |
| `step reward` | 约束每一步生成质量 | `Feasible`、`Diversity`、`SpreadTarget`、`DominancePenalty`、`RepeatPenalty`、`ConstraintPenalty` | `是` | 适合做稠密反馈，帮助模型在 block 内逐步避免生成无效题、重复题、主导题 |
| `block reward` | 评价一整个 respondent block 的贡献 | `reward_D`、`reward_H`、`delta adjusted McFadden pseudo R^2`、`BoundPenalty`、`SensitivityGain`、`ResponseValidity`、`BurdenPenalty` | `是` | 是在线 PPO 最核心的终端奖励层；当前项目建议把 adjusted McFadden pseudo `R^2` 放在这一层 |
| `batch evaluation` | 每累计一批样本后重新看模型质量 | `adjusted McFadden pseudo R^2`、`log-likelihood`、`AIC/BIC`、参数标准误、参数符号稳定性、zone 覆盖偏差 | `否，通常不直接逐步进入` | 更适合每 `N` 个 respondent 或每个 batch 重估一次，用于判断策略是否需要调权重或切换探索强度 |
| `global dashboard metric` | 面向全局监控与可视化 | 样本量进度、各区覆盖情况、设计分发次数、choice share、参数历史轨迹、batch 间 `R^2` 变化、越界参数告警 | `否` | 主要用于“查看收集情况”页面和人工监控，不应直接作为单步 reward 使用 |

如果进一步压缩成一句原则，可以写成：

```text
局部结构问题放 step reward，
整份问卷有效性放 block reward，
阶段性模型质量放 batch evaluation，
系统运行状态放 global dashboard metric。
```

#### C. block 级奖励如何分配回每一步

由于 `R_b` 是 block 完成后才知道的，所以需要把它回传给 block 内每个生成步。工程上常见的几种做法是：

1. 均匀分摊

```math
r_t = r_t^{shape} + \frac{R_b}{T_b}
```

2. 按 score 分摊

```math
\alpha_t
=
\frac{\exp(\hat{q}_t)}{\sum_{j=1}^{T_b}\exp(\hat{q}_j)}
```

```math
r_t = r_t^{shape} + \alpha_t R_b
```

3. 末步承载大部分终端奖励

```math
r_t =
\begin{cases}
r_t^{shape}, & t < T_b \\
r_t^{shape} + R_b, & t = T_b
\end{cases}
```

对本项目而言，更稳妥的做法通常是：
- 先保留 `r_t^{shape}`
- 再把 `R_b` 按均匀或按 score 的方式分配回 block 内各步
- 最后通过 GAE 自动平滑 credit assignment

### 9.10.5 一个 block 完成后如何形成 PPO rollout

当第 `b` 个 respondent 的一个 block 生成并完成回答后，训练缓存里至少要保存：

```math
\bigl(
s_t,\ a_t,\ \log \pi_{\theta_{old}}(a_t \mid s_t),\ V_\psi(s_t),\ r_t,\ done_t
\bigr)_{t=1}^{T_b}
```

其中：
- `s_t`：第 `t` 步生成时的状态
- `a_t`：第 `t` 步复合动作 `(mask_t, value_t, stop_t)`
- `\log \pi_{\theta_{old}}(a_t \mid s_t)`：旧策略下的动作对数概率
- `V_\psi(s_t)`：critic 对该状态的价值估计
- `r_t`：该步奖励
- `done_t`：该步是否为 block 终止步

通常定义为：

```math
done_t =
\begin{cases}
0, & t < T_b \\
1, & t = T_b
\end{cases}
```

这就意味着：
- 一个 respondent 的一个 block 就是一条 episode
- episode 内部长度是 `T_b`
- 若 block 长度可变，则不同 episode 的 `T_b` 可以不同

### 9.10.6 本项目里的 GAE 与 PPO 更新闭环

有了 rollout 之后，可以按标准 PPO 思路计算 advantage，但要注意它是在“block episode”上做的。

先计算一步 TD 残差：

```math
\delta_t
=
r_t
+
\gamma (1-done_t)V_\psi(s_{t+1})
-
V_\psi(s_t)
```

再反向递推得到 GAE：

```math
\hat{A}_t
=
\delta_t
+
\gamma \lambda (1-done_t)\hat{A}_{t+1}
```

对应的 return 为：

```math
\hat{R}_t = \hat{A}_t + V_\psi(s_t)
```

然后对缓存中的所有 block episode 打平或按 mini-batch 采样，执行 PPO-clip 更新：

```math
r_t(\theta)
=
\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}
```

```math
L_{clip}
=
-
\mathbb{E}_t
\Bigl[
\min
\bigl(
r_t(\theta)\hat{A}_t,
\operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t
\bigr)
\Bigr]
```

```math
L_{vf}
=
\mathbb{E}_t
\Bigl[
\bigl(V_\psi(s_t) - \hat{R}_t\bigr)^2
\Bigr]
```

```math
L_{ent}
=
-
\mathbb{E}_t
\bigl[
\mathcal{H}(\pi_{mask})
+
\mathcal{H}(\pi_{value})
+
\mathcal{H}(\pi_{stop})
\bigr]
```

最终在线阶段目标可写成：

```math
L
=
\lambda_{clip}L_{clip}
+
\lambda_{vf}L_{vf}
+
\lambda_{ent}L_{ent}
+
\lambda_{aux}L_{aux}
+
\lambda_{cons}L_{cons}
```

其中：
- `L_aux` 可以继续保留 `score_head` 的辅助监督
- `L_cons` 用来压制越界、逻辑冲突、主导结构等不合法生成

### 9.10.7 一个贴近项目流程的例子

假设当前系统准备给第 `b=37` 个 respondent 发一份 block，目标长度上限是 8 题。

#### 第 1 步：读取初始状态

系统拿到：
- 该 respondent 的 RP 信息 `X_rp`
- 当前全局收集状态 `X_env`
- 空的 block 上下文 `X_sp_ctx^{(1)}`
- 当前允许使用的候选变量模板 `X_cand`

这时状态是：

```math
s_1 = (X_{rp}, X_{env}, X_{sp\_ctx}^{(1)}, X_{cand}, \tau_1)
```

#### 第 2 步：生成第 1 题

decoder 输出：
- `m_1`：激活哪些变量
- `x_1`：这些变量的具体取值
- `z_1 = 0`：先不结束

于是得到第 1 题，并根据结构质量计算一个 `r_1^{shape}`。

#### 第 3 步：继续生成第 2 到第 8 题

每生成一题，就把该题编码成新的 question token 写回 `X_sp_ctx`，得到：

```math
s_2,\ s_3,\ \ldots,\ s_8
```

假设第 8 题生成后 `z_8 = 1`，表示 block 结束。

#### 第 4 步：一次性发放并接收回答

系统把 8 道题作为一个 block 一次性发给 respondent。respondent 完成后，一次性返回这 8 道题的选择结果。

这时系统再根据这组答案：
- 更新参数估计
- 重新计算 `D-error`
- 更新覆盖统计和灵敏度统计
- 得到该 block 的终端奖励 `R_{37}`

#### 第 5 步：把 block 奖励回传给各步

例如使用均匀分摊：

```math
r_t = r_t^{shape} + \frac{R_{37}}{8}, \quad t=1,\ldots,8
```

此时整条 episode 就完整了。

#### 第 6 步：做 GAE 与 PPO 更新

对这一条 episode 以及其它 respondent 新收集到的 block episode 一起：
- 计算 `\delta_t`
- 计算 `\hat{A}_t`
- 计算 `\hat{R}_t`
- 做若干个 epoch 的 PPO mini-batch 更新

于是模型在下一次给新 respondent 生成 block 时，就会使用更新后的策略参数。

### 9.11 在线阶段为什么通常要有 critic

这里需要特别说明一个容易混淆的问题：

```text
score_head 不等于 critic
```

原因是：
- `score_head` 评估的是“当前题的质量分数”
- `critic` 评估的是“当前状态往后整条生成过程的期望回报”

如果要在在线阶段使用 PPO / actor-critic，通常需要额外增加：

```math
V_\psi(s_t)
```

作为状态价值函数近似器。

### 9.12 在线阶段的策略分布

若采用策略梯度或 PPO 风格更新，则三类动作的策略可以写成：

1. `mask_head`
   对每个变量激活与否建模为 Bernoulli：

```math
\pi_{mask}(m_t \mid s_t)
=
\prod_{v=1}^{V}
\operatorname{Bernoulli}(m_{t,v}; \hat{p}_{t,v})
```

2. `value_head`
   对每个激活变量的值建模为截断高斯或 Beta 分布：

```math
\pi_{value}(x_t \mid s_t, m_t)
=
\prod_{v:m_{t,v}=1}
\operatorname{TruncNormal}(x_{t,v}; \mu_{t,v}, \sigma_{t,v}, l_v, u_v)
```

3. `stop_head`
   对停止与否建模为 Bernoulli：

```math
\pi_{stop}(z_t \mid s_t)
=
\operatorname{Bernoulli}(z_t; \hat{\pi}_t^{stop})
```

整体策略可以写成：

```math
\pi_\theta(a_t \mid s_t)
=
\pi_{mask}\pi_{value}\pi_{stop}
```

其中 `a_t` 在这里不再是单个离散索引，而是一个复合动作：

```text
a_t = (mask_t, value_t, stop_t)
```

### 9.13 在线阶段的 PPO 型目标

若保存旧策略 `\pi_{\theta_{old}}`，则第 `t` 步的概率比为：

```math
r_t(\theta)
=
\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}
```

优势函数可以通过 GAE 或 n-step return 估计得到，记为 `\hat{A}_t`。则 PPO-clip 部分可写成：

```math
L_{ppo}
=
-
\mathbb{E}_t
\Bigl[
\min
\bigl(
r_t(\theta)\hat{A}_t,
\operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t
\bigr)
\Bigr]
```

critic 损失为：

```math
L_{vf}
=
\mathbb{E}_t
\Bigl[
\bigl(V_\psi(s_t) - \hat{R}_t\bigr)^2
\Bigr]
```

熵正则项为：

```math
L_{ent}
=
-
\mathbb{E}_t\bigl[\mathcal{H}(\pi_\theta(\cdot \mid s_t))\bigr]
```

则在线阶段总损失可写成：

```math
L_{online}
=
\lambda_{ppo}L_{ppo}
+
\lambda_{vf}L_{vf}
+
\lambda_{ent}L_{ent}
+
\lambda_{aux}L_{sup\_aux}
+
\lambda_c L_{cons}
```

其中 `L_sup_aux` 可以保留一小部分监督项，例如：
- `score_head` 的辅助回归
- `stop_head` 的规则标签监督

这样做的目的是防止模型在纯在线更新中快速漂移，导致结构合法性变差。

### 9.14 一个更稳妥的工程训练顺序

如果从项目落地角度来排训练顺序，比较稳妥的做法是：

1. 先用 efficient / dyppo / 人工规则 block 做纯监督 warmup
2. 再用 scheduled sampling 缩小训练-推理分布差
3. 再在真实 respondent 数据上做小学习率在线微调
4. 在线阶段保留约束修正与辅助监督，避免结构崩坏

因此，encoder-decoder 的训练不是一条单一 loss 的“直接端到端黑盒训练”，而更接近：

```text
监督结构学习 + 约束学习 + 在线反馈优化
```

## 10. 一个 block 的生成流程

按 encoder-decoder 口径，一个 respondent 的 block 生成过程可以写成：

1. 输入 `X_rp, X_env, X_sp_ctx, X_cand`
2. encoder 得到上下文记忆 `H_enc`
3. decoder 从 `BOS` 开始
4. 第 `t` 步输出：
   - `mask_t`
   - `value_t`
   - `score_t`
   - `stop_t`
5. 若未停止，则把本步生成题编码成 token，送回 decoder 作为下一步输入
6. 直到达到停止条件，输出整份 block
7. respondent 完成该 block 的作答
8. 把本次反馈回写到 `X_env` 与后续训练集中

这构成了 encoder-decoder 路线下完整的“生成 -> 反馈 -> 再生成”闭环。

## 11. 一句话总结

```text
SelfAttention encoder-decoder 与 dyppo 的主要差异不在“是否动态”，而在“动态作用于哪个层面”：dyppo 主要在固定 variables 与固定 attribute levels 所定义的离散可行空间中动态选题；encoder-decoder 则进一步把变量激活、变量值生成和 block 结构生成本身纳入可学习过程，因此在建模上更灵活，但训练与推理成本也更高。
```
