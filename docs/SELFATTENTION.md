# SelfAttention 并行题组生成详细说明

更新时间：2026-04-28

本文档采用当前项目统一的模块说明口径描述 `selfattention` 路线，重点说明各模块的作用、数据维度变化、功能边界和前后衔接。

当前结构由以下模块组成：

- `encoder` 读取 RP / 环境 / 历史问卷统计 / 候选变量模板；
- `respondent_target_head` 从 encoder 的全局状态输出剩余样本的定向采样建议；
- `parallel question queries` 并行表示一组候选题位 proposal slots；
- `question-set self-attention` 建模整份问卷内部各候选题之间的结构关系；
- `cross-attention` 让候选题位读取 encoder 条件信息；
- `count_head` 决定本次问卷题数；
- `slot_select_head` 决定从候选题位中选出哪些题进入最终问卷；
- `mask_head` 决定每题哪些变量激活；
- `value_head` 给激活变量赋值；
- `score_head` 评估每题质量。

整体流程是：先并行构造一个完整 question block，再一次性发放给 respondent 填写，提交后回写统计状态和模型训练数据。

---

## 1. SelfAttention 在三种策略中的定位

```text
efficient design
    -> dynamicPPO
    -> selfattention parallel block generator
```

### 1.1 Efficient Design
- 变量固定
- 每个变量的 levels 固定
- 在离散组合空间中做 D-error 优化

### 1.2 dynamicPPO
- 仍然固定变量
- 仍然固定 levels
- 但会根据 respondent 和历史反馈动态调整题组分发

### 1.3 SelfAttention
- 将动作空间从固定 combo 池扩展为结构化题组生成
- 支持决定每道题中哪些变量激活
- 支持直接生成连续值或映射后的离散值
- 支持一次性生成一整份 respondent block
- 支持额外输出“下一阶段更建议收集哪些 RP cell”的采样建议，例如某些区、性别、年龄或教育分段仍然不足。

这里的升级核心是：

```text
从“离散题库选择”升级为“条件约束下的并行结构化题组生成”。
```

---

## 2. Block 生成模块总览

当前 respondent block 的生成链路如下：

```text
X_rp, X_env, X_hist, X_cand
    -> embedding
    -> concat
    -> encoder
    -> H_enc

Q_slot^(0)
    -> question-set self-attention
    -> cross-attention(H_enc)
    -> count / slot_select / mask / value / score heads
    -> Q_block
```

其中：

- `X_rp` 表示当前 respondent 的个人、家庭和区域特征。
- `X_env` 表示当前采样进度、覆盖偏差、模型稳定性和设计质量状态。
- `X_hist` 表示前 `n-1` 份已完成问卷 / blocks 提炼出的历史 SP 上下文。
- `X_cand` 表示当前 design 下可被激活或赋值的候选变量模板。
- `H_enc` 是 encoder 融合后的条件表示。
- `Q_slot^(0)` 是 decoder 的并行候选题位输入。
- `Q_block` 是最终组装后写回 SP 页面的一整份问卷题组。

该模块按 respondent block 生成整份 SP 问卷，slot 只是模型内部候选位置，最终页面展示顺序由后处理规则确定。

---

## 3. 输入张量定义

### 3.1 RP 输入 `X_rp`

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

### 3.2 环境输入 `X_env`

表示全局环境与在线采集状态，例如：
- 当前累计样本量占目标样本量比例
- 当前 zone 的覆盖偏差
- 当前 design 的分发次数
- 当前参数估计稳定度
- 当前 adjusted McFadden pseudo R^2
- 当前设计约束越界惩罚摘要

若环境特征维度为 `d_env`，则：

```text
X_env ∈ R^[B, d_env]
```

### 3.3 历史问卷上下文 `X_hist`

这里正式把过去文档里的 `X_sp_ctx` 收敛成更清楚的名字：

```text
X_hist : 来自前 n-1 份已完成 SP 问卷 / blocks 的历史上下文
```

它不是：
- 当前 respondent 尚未提交的实时回答；
- 当前 block 内部的题目顺序、展示顺序或作答顺序。

它表示的是：
- 历史题组的变量激活频率；
- 历史题组的变量值统计；
- 历史题组的信息量、覆盖度和区域偏差；
- 供当前 respondent 复用的历史 SP token / summary token。

若历史 token 数为 `L_hist`，单 token 特征维度为 `d_hist`，则：

```text
X_hist ∈ R^[B, L_hist, d_hist]
```

这里的 `L_hist` 不是“问卷内题目序列长度”，也不是翻译模型里一句话的 token 长度。它表示历史样本摘要 token 的最大保留数。工程上建议：
- 小样本 pretest：`L_hist=16~32`
- 中等规模在线调查：`L_hist=32~64`
- 若希望保留更多分区/人群/变量统计摘要，可扩到 `128`，但计算量会随 attention 长度平方增长
- 不足 `L_hist` 时用 0 padding，超过时优先保留最近 batch、覆盖偏差最大 cell、参数变化最大的摘要 token

### 3.4 候选变量模板 `X_cand`

表示当前 design 中允许出现的变量模板、变量类型和范围约束，例如：
- alternative 名称
- variable 名称
- variable_type
- level / bound 信息
- 条件约束摘要

若候选模板 token 数为 `L_cand`，每个 token 维度为 `d_cand`，则：

```text
X_cand ∈ R^[B, L_cand, d_cand]
```

---

## 4. Encoder 结构与维度

四类输入先分别投影到统一 `d_model`，再拼接。

```text
X_rp, X_env, X_hist, X_cand
  -> 各自 Embedding
  -> E_rp, E_env, E_hist, E_cand
  -> concat
  -> X_enc
  -> Encoder Blocks
  -> H_enc
```

### 4.1 各自 embedding 后的形状

```text
E_rp   ∈ R^[B, 1, d_model]
E_env  ∈ R^[B, 1, d_model]
E_hist ∈ R^[B, L_hist, d_model]
E_cand ∈ R^[B, L_cand, d_model]
```

### 4.2 拼接后

```text
X_enc ∈ R^[B, L_enc, d_model]
L_enc = 1 + 1 + L_hist + L_cand
```

### 4.3 Encoder 输出

经过若干层 self-attention + FFN 后得到：

```text
H_enc ∈ R^[B, L_enc, d_model]
```

`H_enc` 表示：
- 当前 respondent 是谁；
- 当前环境与采集状态如何；
- 历史问卷已经覆盖到什么程度；
- 当前 design 允许哪些变量模板和取值范围。

### 4.4 Encoder 的两个输出分支

`H_enc` 不是只给 SP 题组生成器使用。当前项目把 encoder 输出分成两个用途：

```text
H_enc
  -> pooled global state
  -> respondent_target_head
  -> sampling_recommendation

H_enc
  -> question-set/block decoder
  -> count / slot_select / mask / value / score
```

#### respondent_target_head

把 `H_enc` 沿 token 维度做平均池化：

```text
g_enc = mean(H_enc, axis=1)
g_enc ∈ R^[B, d_model]
```

然后输出采样 cell 的优先级：

```text
z_sample ∈ R^[B, C_sample]
p_sample = softmax(z_sample)
```

其中 `C_sample` 来自 PopSim / ActivitySim 兼容统计文件中的边际 cell，例如：
- `zone=天河区`
- `gender=female`
- `age_group=18-30`
- `education=college`

如果 `policy_state` 中已经有 dashboard 统计出的 `sample_cell_counts` 和 `sample_cell_targets`，系统会把模型输出和实际缺口融合：

```text
priority = 0.45 * model_priority + 0.55 * target_gap_priority
```

这一路输出只用于“建议接下来更应定向收集哪类 RP 样本”，不是直接强制改变样本。实际调查仍应记录抽样方式，后续可用加权或分层校正减少人为干预造成的样本偏差。

---

## 5. Decoder 并行 question queries 输入

Decoder 输入采用一组 **并行 question queries**，这些 queries 更准确地说是内部 **proposal slots**：

- slot 编号只用于模型内部区分候选生成位置；
- slot 编号不等于页面上的第 1 题、第 2 题；
- 最终问卷展示顺序可以由前端或后处理规则排序；
- 训练时不应假定 teacher block 的第 1 题必须对应模型 slot 1。

记：

- `T_min`：问卷最小题数；
- `T_max`：问卷最大题数；
- `T_q`：当前这份问卷最终生成的题数，满足 `T_min ≤ T_q ≤ T_max`。

Decoder 初始输入定义为：

```text
Q_slot^(0) ∈ R^[B, T_max, d_model]
```

其中：

- 第二维的 `T_max` 表示“待生成题位”的最大并行数量；
- 每个 slot 都是一个 question query；
- 每个 slot 表示“这一份问卷中潜在的一道候选题”。

---

## 6. Question-Set Self-Attention 的维度与逻辑

这是当前项目里最重要、也最容易和旧口径混淆的部分。

### 6.1 输入

```text
Q_slot^(0) ∈ R^[B, T_max, d_model]
```

### 6.2 线性投影到 Q / K / V

设：
- `h`：attention head 数
- `d_h = d_model / h`

则：

```math
Q_s = Q_{slot}^{(0)} W_Q,
K_s = Q_{slot}^{(0)} W_K,
V_s = Q_{slot}^{(0)} W_V
```

其中：

```math
W_Q, W_K, W_V ∈ R^{d_{model}×(h·d_h)}
```

投影后：

```text
Q_s, K_s, V_s ∈ R^[B, T_max, h·d_h]
```

reshape 后：

```text
Q_s, K_s, V_s ∈ R^[B, h, T_max, d_h]
```

### 6.3 计算题位之间的相关矩阵

```math
S_s = \frac{Q_s K_s^T}{\sqrt{d_h}}
```

维度：

```text
S_s ∈ R^[B, h, T_max, T_max]
```

这里的最后两个维度表示：
- 第 3 维：当前题位 `i`
- 第 4 维：它正在看的另一个题位 `j`

注意：

```text
这里的 i 和 j 都是“同一份问卷中的内部候选 slot 索引”，
不是生成时间步，也不是最终展示题号。
```

### 6.4 Attention 权重的作用

softmax 后得到：

```text
A_s ∈ R^[B, h, T_max, T_max]
```

`A_s[b, head, i, j]` 表示在第 `b` 个样本、第 `head` 个注意力头中，第 `i` 个候选 slot 对第 `j` 个候选 slot 的参考权重。

该矩阵用于表达同一份 respondent block 内候选题位之间的关系，例如：

- 两个候选题是否覆盖了相似变量；
- 两个候选题的变量值是否过于接近；
- 某个候选题是否可以补充当前题组缺少的变量组合；
- 当前题组内部是否形成足够的敏感性分析覆盖。

### 6.5 头内输出

```math
O_s = A_s V_s
```

维度：

```text
O_s ∈ R^[B, h, T_max, d_h]
```

拼接后：

```text
Concat(O_s) ∈ R^[B, T_max, h·d_h]
```

再回投影：

```math
H_s = Concat(O_s) W_O
```

其中：

```math
W_O ∈ R^{(h·d_h)×d_{model}}
```

得到：

```text
H_s ∈ R^[B, T_max, d_model]
```

### 6.6 逻辑解释

这一层解决的问题不是：

```text
“第 1 题能不能看第 2 题的未来信息”
```

而是：

```text
“在并行生成整份问卷时，第 i 个题位应该如何参考其余题位，
从而让整份问卷既有区分度，又不过度重复，还保持结构多样性。”
```

---

## 7. Cross-Attention 的维度与逻辑

question queries 经过 self-attention 后，还需要读取 encoder 的条件记忆。

### 7.1 输入

query 来自题位表示：

```text
H_s ∈ R^[B, T_max, d_model]
```

key / value 来自 encoder 输出：

```text
H_enc ∈ R^[B, L_enc, d_model]
```

### 7.2 投影后

```text
Q_c ∈ R^[B, h, T_max, d_h]
K_c, V_c ∈ R^[B, h, L_enc, d_h]
```

### 7.3 分数矩阵

```text
S_c ∈ R^[B, h, T_max, L_enc]
A_c ∈ R^[B, h, T_max, L_enc]
```

### 7.4 输出

```text
O_c ∈ R^[B, T_max, d_model]
```

逻辑上表示：
- 每个候选题位
- 去读取 respondent 特征、环境统计、历史问卷上下文和候选变量模板
- 最终形成“这个候选题在当前条件下应如何生成”的表示。

---

## 8. `respondent_target_head` 与 SP 生成 heads

当前 selfattention 有两类输出 head：

```text
采样控制 head:
  respondent_target_head

SP block 生成 heads:
  count_head
  slot_select_head
  mask_head
  value_head
  score_head
```

`respondent_target_head` 服务于全样本质量控制，回答“下一阶段更缺哪类 respondent”；SP block heads 服务于当前 respondent 的问卷生成，回答“当前这份问卷出多少题、哪些题、哪些变量、变量值是多少”。

### 8.1 `respondent_target_head`

输入：

```text
g_enc ∈ R^[B, d_model]
```

输出：

```text
z_sample ∈ R^[B, C_sample]
p_sample ∈ R^[B, C_sample]
```

返回到接口时会转成：

```json
[
  {"cell": "zone=天河区", "priority": 0.18, "needed_n": 23},
  {"cell": "age_group=18-30", "priority": 0.13, "needed_n": 17}
]
```

它的作用是根据目标总体分布、当前样本覆盖偏差和模型内部状态给出定向采样建议。它不等于问卷题目，也不等于 `mask_head`。

### 8.2 `count_head`

`count_head` 输出当前 respondent block 的题数类别，用于决定本次问卷生成多少道 SP 题。

若允许的题数范围为 `T_min..T_max`，则可把它建模成分类头：

```text
z_count ∈ R^[B, K_count]
K_count = T_max - T_min + 1
```

softmax 后得到：

```text
p_count ∈ R^[B, K_count]
```

最终题数：

```text
T_q = argmax(p_count) + T_min
```

### 8.3 `slot_select_head`

`count_head` 只回答“这一份问卷需要多少题”。slot 编号是模型内部的候选位置，最终进入问卷的候选题位由 `slot_select_head` 单独决定。

因此需要 `slot_select_head` 对所有候选题位打分：

```text
slot_logits ∈ R^[B, T_max]
```

给定 `count_head` 得到的题数 `T_q` 后，对每个 respondent 从 `slot_logits[b,:]` 中选出 top-k，其中 `k=T_q[b]`：

```text
M_slot ∈ {0,1}^[B, T_max]
```

它表示：
- 被选中的 slot 会进入最终问卷；
- 未选中的 slot 只是候选生成位置，不参与最终渲染和作答；
- `M_slot` 不是前缀 mask，不默认 `1..T_q` 有效。

这一步把候选题位转换为最终 respondent block：问卷内部的题目按集合处理，页面显示顺序由后处理规则确定。

### 8.4 `mask_head`

若把每道题内部所有变量槽位按统一顺序展平，总长度记为 `V`，则：

```text
mask_logits ∈ R^[B, T_max, V]
mask_prob   ∈ [0,1]^[B, T_max, V]
M_var       ∈ {0,1}^[B, T_max, V]
```

它决定：

```text
每一道题里，哪些变量激活，哪些变量不激活。
```

### 8.5 `value_head`

`value_head` 在 `mask_head` 的结构条件下输出变量取值。对应的条件分解为：

```math
p(T_q, S, M, X \mid H)
=
p(T_q \mid H)
\cdot p(S \mid T_q,H)
\cdot p(M \mid S,T_q,H)
\cdot p(X \mid M,S,T_q,H)
```

也就是说：
- `count_head` 先决定题数；
- `slot_select_head` 决定哪些候选 slot 进入题组集合；
- `mask_head` 再决定被选中题目的题内结构；
- `value_head` 最后在结构条件下给激活变量赋值。

若内部仍按展平 `V` 处理，则：

```text
X_raw ∈ R^[B, T_max, V]
```

最终生效值为：

```math
X_eff = M_{slot}[:, :, None] \odot M_{var} \odot X_{raw}
```

因此：

```text
mask 先决定“哪里有值”，
value 再决定“这些位置的值是多少”。
```

### 8.6 `score_head`

用于给每道题一个质量分数：

```text
S_score ∈ R^[B, T_max, 1]
```

它可以辅助评估：
- 信息量
- 覆盖度
- 重复度
- 与约束的一致性

---

## 9. 最终 question block 的维度

若网络内部用展平后的变量维度 `V`，模型会先并行生成 `T_max` 个候选题：

```text
Q_candidate ∈ R^[B, T_max, V]
```

再通过 `M_slot` 选择其中 `T_q` 个候选题，得到最终问卷集合：

```text
Q_block = {q_i | M_slot[i] = 1}
```

若恢复到更结构化的形式，也可以写成：

```text
Q_block ∈ R^[B, T_q, A, K]
```

其中：
- `A`：每题的方案数
- `K`：每个方案的变量数

工程上更推荐：
- 网络内部用展平 `V`
- 页面渲染时 reshape 回 `A × K`
- 每道题生成稳定 `question_id`
- respondent 回答按 `question_id` 回写，而不是按“第几题”的序号作为训练语义

---

## 10. 训练建议

### 10.1 warmup / teacher 阶段

建议使用：
- efficient design 生成的优质 block
- 历史人工或规则设计 block
- 已收集问卷中质量较好的题组

监督信号包括：
- `count label`
- `slot selection label`
- `variable mask label`
- `value label`
- `score label`

由于问卷内部题目是集合，teacher block 的第 1 题不应强行对齐模型 slot 1。更合适的训练方式是 set matching：
- 模型输出 `T_max` 个候选题；
- teacher block 有 `T_q` 道真实题；
- 用 Hungarian matching 或等价的最小代价匹配，把预测候选题和 teacher 题目配对；
- 匹配代价可以由变量激活差异、变量值差异、题目质量差异、约束违背和重复度共同构成；
- 未匹配的候选 slot 作为未选中题位，参与 `slot_select` 的负样本监督。

### 10.2 loss 建议

```math
L
=
\lambda_c L_{count}
+
\lambda_{sel} L_{slot\_select}
+
\lambda_{tar} L_{sample\_target}
+
\lambda_m L_{mask}
+
\lambda_v L_{value}
+
\lambda_s L_{score}
+
\lambda_{cons} L_{cons}
```

其中：
- `L_count`：题数分类 loss
- `L_slot_select`：题位选择 loss，可用 BCE、ranking loss 或 top-k matching loss
- `L_sample_target`：采样 cell 建议 loss，当前实现用多标签 respondent cell 作为 warmup 监督；若有 dashboard 目标缺口，可转为 target-gap 分类/排序监督
- `L_mask`：变量激活 BCE
- `L_value`：激活变量上的 MSE / SmoothL1
- `L_score`：题质量回归损失
- `L_cons`：越界、逻辑冲突、主导题、重复度等约束惩罚

### 10.3 在线更新

respondent 提交整份问卷后：
- 一次性读取整份 block 的答案；
- 更新历史上下文 `X_hist`；
- 更新参数估计 / pseudo R^2 / zone 覆盖统计；
- 再用小学习率做 supervised / actor-critic 微调。

这里要注意：
- 在线更新的单位是“整份 block”；
- respondent 提交整份问卷后，再统一回写训练数据与统计状态。

---

## 11. 一份问卷的完整流程

1. 读取当前 respondent 的 `X_rp`
2. 读取当前环境与采集状态 `X_env`
3. 读取前 `n-1` 份已完成问卷形成的 `X_hist`
4. 读取当前 design 的 `X_cand`
5. encoder 生成 `H_enc`
6. `respondent_target_head` 输出下一阶段采样建议，用于 dashboard 或调查员调度参考
7. decoder 从 `T_max` 个并行 question queries 出发
8. question-set self-attention 建模题组内部结构关系
9. cross-attention 读取 `H_enc`
10. `count_head` 决定题数 `T_q`
11. `slot_select_head` 从 `T_max` 个候选 slot 中选出 `T_q` 个有效题
12. `mask_head` 决定被选中题目中哪些变量激活
13. `value_head` 给激活变量赋值
14. `score_head` 给每个候选题输出质量分数
15. 后处理按题组质量、去重、约束和展示需要排序
16. 组装整份 `Q_block`
17. 一次性写回 SP 页面
18. respondent 一次性填写整份问卷
19. 系统根据提交结果更新历史统计与模型权重

---

## 12. 一句话总结

```text
当前项目里的 selfattention 是一个 respondent block generator：
它在 respondent / 环境 / 历史问卷 / 候选模板条件下，
通过并行 question queries 生成候选题集合，
再用 count + slot selection + mask/value 组装整份 SP block，
同时用 respondent_target_head 为剩余样本覆盖提供定向采样建议。
```
