# ATTENTION 模块维度说明

更新时间：2026-04-30

本文档说明当前项目 `selfattention` 路线中 attention 相关模块的作用、数据维度变化、功能边界和前后衔接。

---

## 1. 模块总览

当前 attention 链路由四部分组成：

```text
Encoder Self-Attention
    X_enc -> H_enc

Question-Set Self-Attention
    Q_slot^(0) -> H_slot^self

Cross-Attention
    H_slot^self + H_enc -> H_slot^cross

Output Heads
    H_enc / H_slot^cross -> sampling / count / slot / mask / value / score
```

各模块的职责如下：

- `Encoder Self-Attention`：融合 RP、环境、历史问卷统计和候选变量模板。
- `Question-Set Self-Attention`：建模同一份 respondent block 内候选题位之间的互补、重复和覆盖关系。
- `Cross-Attention`：让每个候选题位读取 encoder 中的 respondent、环境、历史和候选变量信息。
- `Output Heads`：输出采样建议、题数、题位选择、变量激活、变量取值和候选题质量分数。

---

## 2. Encoder Self-Attention

### 2.1 输入

encoder 输入来自 embedding 与 concat 后的统一表示：

```text
X_enc ∈ R^[B, L_enc, d_model]
L_enc = 1 + 1 + L_hist + L_cand
```

其中：

- `B`：batch size。
- `1`：RP token，对应当前 respondent 的个人/家庭/区域特征。
- `1`：environment token，对应采样进度、覆盖偏差、模型稳定性等环境状态。
- `L_hist`：历史问卷上下文 token 数。
- `L_cand`：候选变量模板 token 数。
- `d_model`：统一隐藏维度。

### 2.2 Q / K / V 投影

设 attention head 数为 `h`，单头维度为：

```text
d_h = d_model / h
```

线性投影为：

```math
Q_e = X_{enc} W_Q,
K_e = X_{enc} W_K,
V_e = X_{enc} W_V
```

权重维度：

```text
W_Q, W_K, W_V ∈ R^[d_model, h·d_h]
```

投影并 reshape 后：

```text
Q_e, K_e, V_e ∈ R^[B, h, L_enc, d_h]
```

### 2.3 Attention 分数与输出

```math
S_e = \frac{Q_e K_e^T}{\sqrt{d_h}}
```

```text
S_e ∈ R^[B, h, L_enc, L_enc]
A_e = softmax(S_e) ∈ R^[B, h, L_enc, L_enc]
O_e = A_e V_e ∈ R^[B, h, L_enc, d_h]
Concat(O_e) ∈ R^[B, L_enc, d_model]
```

之后进入标准残差、归一化和前馈层：

```text
Z_1 = LayerNorm(X_enc + Concat(O_e))
Z_2 = LayerNorm(Z_1 + FFN(Z_1))
H_enc = Z_2 ∈ R^[B, L_enc, d_model]
```

### 2.4 输出衔接

`H_enc` 同时服务两类后续模块：

- 全局池化后输入 `respondent_target_head`，输出下一阶段建议补充的 RP cell。
- 作为 `Cross-Attention` 的 `K/V`，供 decoder 的候选题位读取条件信息。

---

## 3. Question-Set Self-Attention

### 3.1 输入

decoder 使用一组并行候选题位 query：

```text
Q_slot^(0) ∈ R^[B, T_max, d_model]
```

其中：

- `T_max`：一份问卷允许生成的最大候选题位数。
- 每个 slot 表示一个候选 SP 题位置。
- slot 编号是模型内部候选位置，不等同于页面最终展示顺序。

### 3.2 Q / K / V 投影

```math
Q_s = Q_{slot}^{(0)} W_Q,
K_s = Q_{slot}^{(0)} W_K,
V_s = Q_{slot}^{(0)} W_V
```

reshape 后：

```text
Q_s, K_s, V_s ∈ R^[B, h, T_max, d_h]
```

### 3.3 题位关系矩阵

```math
S_s = \frac{Q_s K_s^T}{\sqrt{d_h}}
```

```text
S_s ∈ R^[B, h, T_max, T_max]
A_s = softmax(S_s) ∈ R^[B, h, T_max, T_max]
O_s = A_s V_s ∈ R^[B, h, T_max, d_h]
H_slot^self ∈ R^[B, T_max, d_model]
```

`A_s[i,j]` 表示同一份 respondent block 中第 `i` 个候选 slot 对第 `j` 个候选 slot 的参考权重。该模块用于让候选题位之间形成互补、去重和覆盖协调。

### 3.4 输出衔接

`H_slot^self` 进入 cross-attention，作为读取 encoder 条件信息的 query。

---

## 4. Cross-Attention

### 4.1 输入

```text
H_slot^self ∈ R^[B, T_max, d_model]
H_enc ∈ R^[B, L_enc, d_model]
```

### 4.2 投影与分数矩阵

```text
Q_c ∈ R^[B, h, T_max, d_h]
K_c,V_c ∈ R^[B, h, L_enc, d_h]
S_c ∈ R^[B, h, T_max, L_enc]
A_c ∈ R^[B, h, T_max, L_enc]
```

### 4.3 输出

```text
O_c = A_c V_c ∈ R^[B, h, T_max, d_h]
H_slot^cross ∈ R^[B, T_max, d_model]
```

cross-attention 的作用是让每个候选 slot 根据当前 respondent、环境状态、历史收集情况和候选变量模板生成题目结构表示。

---

## 5. Output Heads

### 5.1 respondent_target_head

该分支从 encoder 全局状态输出下一阶段建议补充的 RP cell。

```text
g_enc = mean(H_enc, axis=1) ∈ R^[B, d_model]
z_sample ∈ R^[B, C_sample]
p_sample = softmax(z_sample)
```

其中 `C_sample` 是由 PopSim/ActivitySim 兼容统计文件解析出的采样 cell 数，例如区、性别、年龄、教育等组合或边际项。

若 dashboard 已有当前样本覆盖和目标分布，可融合覆盖缺口：

```text
priority = 0.45 * model_priority + 0.55 * target_gap_priority
```

### 5.2 count_head

输出当前 respondent block 的题数类别：

```text
z_count ∈ R^[B, K_count]
p_count = softmax(z_count) ∈ R^[B, K_count]
K_count = T_max - T_min + 1
```

### 5.3 slot_select_head

从 `T_max` 个候选 slot 中选择进入最终问卷的题位集合：

```text
slot_logits ∈ R^[B, T_max]
M_slot ∈ {0,1}^[B, T_max]
```

给定题数 `T_q` 后，从 `slot_logits` 中选出 top-k 个候选 slot，得到 `M_slot`。

### 5.4 mask_head

输出每道候选题中各变量是否激活：

```text
mask_logits ∈ R^[B, T_max, V]
M_var ∈ {0,1}^[B, T_max, V]
```

其中 `V` 是展平后的变量槽位数。

### 5.5 value_head

输出变量取值，并在 `M_slot` 与 `M_var` 条件下形成有效题目值：

```text
X_raw ∈ R^[B, T_max, V]
X_eff = M_slot[:, :, None] ⊙ M_var ⊙ X_raw
```

对应的条件分解可写为：

```math
p(T_q, S, M, X | H)
=
p(T_q|H)\cdot p(S|T_q,H)\cdot p(M|S,T_q,H)\cdot p(X|M,S,T_q,H)
```

### 5.6 score_head

输出候选题质量分数：

```text
score ∈ R^[B, T_max, 1]
```

该分数可用于后处理排序、约束修正、题组质量评估和训练 reward 构造。

---

## 6. 最终 block 维度

候选题集合的展平表示为：

```text
Q_candidate ∈ R^[B, T_max, V]
```

结合 `M_slot` 后得到当前 respondent 的有效题组：

```text
Q_block = {q_i | M_slot[i] = 1}
```

按问卷展示结构还原为：

```text
Q_block ∈ R^[B, T_q, A, K]
```

其中：

- `T_q`：本份问卷最终题数。
- `A`：每题中的方案数。
- `K`：每个方案中的变量数。

---

## 7. 模块衔接摘要

```text
X_rp, X_env, X_hist, X_cand
    -> embedding
    -> concat as X_enc
    -> Encoder Self-Attention
    -> H_enc

Q_slot^(0)
    -> Question-Set Self-Attention
    -> Cross-Attention(H_enc)
    -> H_slot^cross
    -> count / slot_select / mask / value / score heads
    -> Q_block

H_enc
    -> respondent_target_head
    -> sampling recommendation
```

训练时可按集合对齐预测题和 teacher 题，例如使用 Hungarian matching 或等价 set matching，让模型 slot 与 teacher 题目建立匹配关系。
