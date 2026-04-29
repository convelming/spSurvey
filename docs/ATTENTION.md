# ATTENTION 模块维度说明

更新时间：2026-04-28

本文档单独把当前项目 `selfattention` 路线中的 attention 部分拆开说明，重点回答三个问题：

1. 为什么 decoder 不再使用 `Masked Multi-Head Attention`
2. `Question-Set Self-Attention` 的输入输出维度是什么
3. `respondent-target / count / slot-select / variable-mask / value` 各自处于哪个层级

---

## 1. Decoder attention 的正确口径

如果 decoder 是标准语言模型式的自回归结构，常见做法是：

```text
第 t 个 token 只能看 1..t 的历史 token
```

于是需要上三角 causal mask，也就是常说的 `Masked Multi-Head Attention`。

但当前项目的最终设定不是这样：

- 一份问卷中的 `T_q` 道 SP 题一次性生成；
- respondent 一次性看到并填写整份问卷；
- 模型内部需要建模的是“整份问卷内部各题之间的结构关系”，而不是“下一时刻 token 预测”。

所以更合理的结构是：

```text
parallel question queries
    -> full set self-attention over candidate slots
    -> count/slot-select/mask/value/score heads

encoder pooled state
    -> respondent_target_head
    -> sampling recommendation
```

也就是说：
- decoder 内部不再需要“遮住未来题”的 attention mask；
- attention 矩阵 `S ∈ R^[B,h,T_max,T_max]` 保持全连接；
- 如果还保留 “mask” 这个词，只能指：
  - `M_slot`：哪些候选 slot 被选入最终 block；
  - `M_var`：题内哪些变量有效。

---

## 2. Question queries 的输入定义

记：
- `B`：batch size
- `T_max`：允许生成的最大题数
- `d_model`：统一隐藏维度

则 decoder 并行输入写成：

```text
Q_slot^(0) ∈ R^[B, T_max, d_model]
```

其中第二维 `T_max` 表示最大候选题位数，不是时间步，也不是最终页面题号。

---

## 3. Question-set self-attention 的维度

设：
- `h`：head 数
- `d_h = d_model / h`

### 3.1 Q / K / V

```math
Q = Q_{slot}^{(0)} W_Q,
K = Q_{slot}^{(0)} W_K,
V = Q_{slot}^{(0)} W_V
```

其中：

```math
W_Q, W_K, W_V ∈ R^{d_{model}×(h·d_h)}
```

投影后：

```text
Q,K,V ∈ R^[B, T_max, h·d_h]
```

reshape：

```text
Q,K,V ∈ R^[B, h, T_max, d_h]
```

### 3.2 注意力分数矩阵

```math
S = QK^T / \sqrt{d_h}
```

维度：

```text
S ∈ R^[B, h, T_max, T_max]
```

这里最后两个维度表示：
- 第 3 维：当前候选 slot `i`
- 第 4 维：被看的候选 slot `j`

### 3.3 不加 causal mask

当前项目下：

```text
S 不做上三角 causal mask
```

softmax 后：

```text
A ∈ R^[B, h, T_max, T_max]
```

### 3.4 输出

```math
O = AV
```

维度：

```text
O ∈ R^[B, h, T_max, d_h]
```

拼接回去：

```text
Concat(O) ∈ R^[B, T_max, h·d_h]
H_slot ∈ R^[B, T_max, d_model]
```

这一步的逻辑是：

```text
第 i 个候选 slot 可以参考其余候选 slot，
从而让整份问卷内部形成更好的区分度、互补性和多样性。
```

---

## 4. Cross-attention 的维度

encoder 输出：

```text
H_enc ∈ R^[B, L_enc, d_model]
```

decoder 候选 slot 表示：

```text
H_slot ∈ R^[B, T_max, d_model]
```

cross-attention 后：

```text
Q_c ∈ R^[B, h, T_max, d_h]
K_c,V_c ∈ R^[B, h, L_enc, d_h]
S_c ∈ R^[B, h, T_max, L_enc]
A_c ∈ R^[B, h, T_max, L_enc]
O_c ∈ R^[B, T_max, d_model]
```

它的作用是：
- 每个候选 slot query
- 去读取 respondent、环境、历史上下文和候选变量模板
- 让每道题在这些条件下生成。

---

## 5. `respondent_target_head`、`count_head`、`slot_select_head`、`variable mask`、`value`

### 5.0 `respondent_target_head`

这一支从 encoder 的全局状态输出定向采样建议，不参与问卷内题目排序。

```text
H_enc ∈ R^[B, L_enc, d_model]
g_enc = mean(H_enc, axis=1) ∈ R^[B, d_model]
z_sample ∈ R^[B, C_sample]
p_sample = softmax(z_sample)
```

其中 `C_sample` 是 PopSim/ActivitySim 兼容统计文件解析出的边际采样 cell 数，例如：
- `zone=天河区`
- `gender=female`
- `age_group=18-30`
- `education=college`

如果 dashboard 已统计当前样本的 `sample_cell_counts` 和目标分布 `sample_cell_targets`，最终优先级会融合模型输出和实际覆盖缺口：

```text
priority = 0.45 * model_priority + 0.55 * target_gap_priority
```

### 5.1 `count_head`

输出本次问卷题数：

```text
z_count ∈ R^[B, K_count]
p_count ∈ R^[B, K_count]
```

其中：

```text
K_count = T_max - T_min + 1
```

### 5.2 `slot_select_head`

`count_head` 只给出题数，不应默认“前 k 个 slot 有效”。因此还需要一个 slot selection head：

```text
slot_logits ∈ R^[B, T_max]
M_slot ∈ {0,1}^[B, T_max]
```

给定 `T_q` 后，从 `slot_logits` 中选出 top-k 个候选 slot，得到 `M_slot`。这里的 `M_slot` 是集合选择结果，不是前缀 mask。

### 5.3 `mask_head`

若每题展平后有 `V` 个变量槽位：

```text
mask_logits ∈ R^[B, T_max, V]
M_var ∈ {0,1}^[B, T_max, V]
```

表示题内变量结构。

### 5.4 `value_head`

```text
X_raw ∈ R^[B, T_max, V]
X_eff = M_slot[:, :, None] ⊙ M_var ⊙ X_raw
```

最重要的是：

```text
value 不是和 mask 完全独立并列，
而是条件于 mask 才真正生效。
```

更严格的因子分解是：

```math
p(T_q, S, M, X | H)
=
p(T_q|H)·p(S|T_q,H)·p(M|S,T_q,H)·p(X|M,S,T_q,H)
```

---

## 6. 最终 block 维度

网络内部常用展平写法，先生成候选题集合：

```text
Q_candidate ∈ R^[B, T_max, V]
```

结合 `M_slot` 后只保留被选中的 `T_q` 个候选题：

```text
Q_block = {q_i | M_slot[i] = 1}
```

若按问卷展示结构还原，则可 reshape 为：

```text
Q_block ∈ R^[B, T_q, A, K]
```

其中：
- `A`：每题里的方案数
- `K`：每个方案里的变量数

---

## 7. 本项目里 attention 的一句话定义

```text
当前 selfattention 的 attention，建模的是“整份问卷内部候选 slot 之间的并行关系”，
不是“未来时间步不能被看到”的语言模型式上三角因果序列关系。
```

训练时也应按集合对齐：teacher block 的第 1 题不天然对应模型 slot 1，建议使用 Hungarian matching 或等价 set matching 来对齐预测候选题和 teacher 题目。
