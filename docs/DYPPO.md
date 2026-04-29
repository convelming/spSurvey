# dynamicPPO 详细说明

更新时间：2026-04-13

本文档按当前项目的 `dyppo` 口径描述其算法角色、`state / action / reward` 定义以及训练逻辑。

对应代码主要在：
- `/Users/convel/PycharmProjects/spSurvey/app.py`
- `/Users/convel/PycharmProjects/spSurvey/engine/dynamicPPO.py`

## 1. dynamicPPO 的定位

`dyppo` 不是 `A3C`，也不是 encoder-decoder。当前口径下它是：
- `efficient design` 的强化学习升级版
- 使用 `PPO-clip + GAE + Actor-Critic`
- 在固定变量、固定 attribute levels 的离散可行空间中，动态选择更合适的 `combo`

因此它的升级点是：
- `efficient`：一次性离线生成一套题
- `dyppo`：在同一套变量和 levels 的前提下，依据 respondent 与历史反馈动态调整题组

但它不做这些事：
- 不新增变量
- 不删除变量
- 不把离散 levels 改成连续值
- 不直接生成超出 `design_spec` 的新变量取值

## 2. dyppo 相对 efficient design 的升级边界

本节单独列出，是为了明确 `dyppo` 与 `efficient design` 在设计空间定义上的继承关系。

### 2.1 相同点

`dyppo` 与 `efficient design` 共用：
- 同一个 `design_spec`
- 同一组 variables
- 同一组 variables 的 attribute levels
- 同一组条件约束 `conditions`
- 同一套 dominance / feasibility 边界

也就是说，`dyppo` 的搜索空间仍然是：

```text
固定 variables × 固定 attribute levels -> 可行离散组合空间
```

### 2.2 升级点

`dyppo` 相对 `efficient` 的提升不在“改变量定义”，而在“改出题策略”：
- 不再一次性固定唯一题组。
- 会根据 respondent 当前状态重新打分 candidate pool。
- 会根据已收集的回答数据在线更新策略。
- 因而不同 respondent 可以拿到不同的 combo。

一句话概括：

```text
dyppo 是在 efficient design 的离散设计空间上，加了一层可在线更新的动态出题策略。
```

## 3. Action 是什么

当前 `dyppo` 中的动作定义为：

```text
a_t = 在时刻 t 从 feasible combo pool 中选择的 task 索引
```

若候选池大小为 `A`，则：

```text
a_t ∈ {0,1,2,...,A-1}
```

这里 action 不是：
- respondent 的回答
- 某个变量值
- 某个 block 编号

而是：
- “当前该发哪一道 SP 题”

## 4. feasible combo pool 是什么

候选动作空间来自 `app.py::_build_candidate_pool_for_ppo()`。

构造逻辑：
1. 从 `design_spec` 展开所有 alternatives 和 variables 的离散 levels。
2. 若全因子离散组合空间规模不超过预设枚举阈值，则对该空间进行显式枚举并生成候选题。
3. 若全因子空间规模超过阈值，则采用构造式抽样方法生成一批候选题，再作为候选池的初始来源。
4. 依据 `conditions` 进行可行性过滤。
5. 依据先验参数方向可识别的 dominance 关系进行支配性过滤。
6. 对剩余题目去重，形成最终 `candidate_pool`。

因此：

```text
action space = fixed variables + fixed levels + feasibility filter + dominance filter
```

## 5. Efficient 在 dyppo 中扮演什么角色

当前 `efficient` 在 `dyppo` 中不是动作空间本身，而是：

```text
expert prior / teacher prior
```

具体作用：
- 先运行一次 efficient design，得到一组参考题组。
- 将这组参考题映射到 `candidate_pool` 上。
- 与 actor 输出的概率做凸组合。

但它不会：
- 把动作空间缩小成 efficient 的结果
- 取代 RL 策略本身

所以 `dyppo` 的本质仍然是：
- 在完整 feasible combo pool 上做动态选择
- efficient 只提供 warm-start 偏置

## 6. State 是什么

当前单步状态向量来自 `_task_state_vector()`：

```text
s_t = [ respondent_one_hot , spread_hint , mean_entropy , progress , selected_ratio ]
```

即：
- respondent 的 RP / zone 离散编码
- 历史已选题的统计摘要
- 当前 block / rollout 进度

### 6.1 respondent_one_hot

由 `_build_feature_spec()` 和 `_respondent_feat()` 生成。

它来自：
- `attr_segments`
- `zone_id`
- PopSim 统计文件里的分类口径

例如如果 `key_format = gender|age_group|education`，则 respondent 先被转为：

```text
attr_segments = [gender_value, age_group_value, education_value]
```

再按列 one-hot。

### 6.2 spread_hint

对历史已选题中的每道题：

```text
spread(task) = max(U_j) - min(U_j)
```

然后：

```text
mean_spread = average(spread(task_h))
spread_hint = 1 / (1 + |mean_spread|)
```

### 6.3 mean_entropy

对历史已选题中的每道题：

```text
p_j = softmax(U)_j
entropy_norm = -Σ p_j log p_j / log(J)
```

然后：

```text
mean_entropy = average(entropy_norm(task_h))
```

### 6.4 progress

```text
progress = step_index / max(episode_len - 1, 1)
```

### 6.5 selected_ratio

```text
selected_ratio = len(prior_tasks) / episode_len
```

## 7. Reward 是什么

当前 reward 分两层：

### 7.1 单步 reward

`_task_step_reward()`：

```text
chosen_prob = p(chosen_alt | task, respondent)
entropy_norm = normalized_entropy(task)
spread_bonus = 1 / (1 + |spread|)

r_t = 0.30 * chosen_prob
    + 0.15 * entropy_norm
    + 0.10 * spread_bonus
    + terminal_bonus
```

它鼓励：
- 有辨识度但不压倒性的题
- respondent 确实作出了有信息的选择
- 不出现过大 utility gap 的主导题

### 7.2 整份问卷级奖励

`_questionnaire_reward()`：

```text
d_err = D_error(tasks)
reward_d = 1 / (1 + d_err)
reward_h = mean(entropy_norm(task_i))
delta_adj_rho2 = adjusted_pseudo_r2(after) - adjusted_pseudo_r2(before)
bound_penalty = 参数估计越界惩罚
episode_bonus
    = w_d * reward_d
    + w_h * reward_h
    + w_rho * delta_adj_rho2
    - w_bound * bound_penalty
```

含义：
- D-error 越小越好
- 题内选择越均衡越好
- 当前 block 若使 adjusted McFadden pseudo R^2 提升，则获得额外正奖励
- 若参数估计超出前端定义的 `beta_min / beta_max` 区间，则产生负奖励

最后一步会把 `episode_bonus` 加到 `terminal_bonus` 上。

### 7.3 adjusted pseudo R^2 的接入方式

这里采用的是“block 级 before / after 增量”口径，而不是把 pseudo R^2 当成唯一 reward。

具体流程：

1. 把当前 design 历史上已经收集到的回答整理成 `obs_before`
2. 用 `obs_before` 重估一次 MNL，得到 `adjusted_pseudo_r2_before`
3. 再把当前 respondent 这一个 block 的回答追加进去，得到 `obs_after`
4. 用 `obs_after` 重估一次 MNL，得到 `adjusted_pseudo_r2_after`
5. 计算：

```text
delta_adj_rho2 = adjusted_pseudo_r2_after - adjusted_pseudo_r2_before
```

它衡量的是：

```text
当前 block 加进来以后，整体行为模型解释力有没有提升。
```

### 7.4 参数越界惩罚与候选题抑制

前端在 `sp_design.html` 中为每个变量提供了：
- `beta_min`
- `beta_max`

因此在线更新时，当前实现会先做一版“未裁剪参数估计”，专门用于检查：

```text
估计出的参数值是否跑出了预定义区间。
```

若出现越界，则会同时触发两类后果：

1. 在 `episode_bonus` 里加入 `bound_penalty`
2. 在下一轮候选题打分时，对包含这些越界变量的 candidate task 做高概率降权

由于 dyppo 的动作仍然是“选哪一道离散候选题”，这里的“mask”不是逐变量显式 mask，而是：

```text
对涉及越界变量的题目整体抑制，
从候选池分发效果上近似“大概率把这些变量 mask 掉”。
```

## 8. 代理效用模型在 dyppo 中的作用

当前 dyppo 的环境反馈并不来自一个外部仿真器，而是依赖代理效用：

```text
U(alt) = Σ_k beta[alt.k] * value_k + Σ_i beta[seg_i.segment_value.alt]
```

它的用途包括：
- 计算题内 choice probability
- 计算 `spread`
- 计算 `entropy_norm`
- 在 warmup 阶段模拟 respondent 的伪作答

所以当前 RL 训练是：
- 真实 rows 存在时，优先使用真实 rows
- 真实 rows 不足时，用 PopSim + 代理效用做 warm-start

## 9. rollout 如何构造

PPO 训练吃的不是原始 payload，而是 rollout：

```text
states  ∈ R^[T,D]
actions ∈ Z^[T]
rewards ∈ R^[T]
dones   ∈ {0,1}^[T]
```

### 9.1 真实 rows -> rollout

`_build_rollouts_from_rows()` 中：
- 一个 respondent 的一条记录视为一个 episode
- 每道题是一个 time step
- task 的 `sig` 会映射回 candidate pool 索引，形成 `action`

### 9.2 synthetic respondents -> rollout

如果真实 rows 不足，则：
1. 根据 PopSim 目标分布采样 synthetic respondents
2. 在 candidate pool 里选题
3. 用代理效用生成伪作答
4. 构造一批 warm-start rollout

## 10. 网络结构

当前 `dyppo` 是共享骨干的 Actor-Critic：

```text
state x: [B,D]
 -> Linear(D->64)
 -> Tanh
 -> Linear(64->64)
 -> Tanh
 -> actor_head(64->A)
 -> critic_head(64->1)
```

输出：
- `logits ∈ R^[B,A]`
- `value ∈ R^[B]`

## 11. 训练逻辑写细一点

### 11.1 old policy 基准

训练开始前，先用当前模型对整批 rollout 前向一次，得到：
- `old_logits`
- `old_logp`
- `old_values`

这一步的作用是固定 PPO 的旧策略参照。

### 11.2 GAE

然后按：

```text
delta_t = r_t + gamma * V(s_{t+1}) * (1-done_t) - V(s_t)
A_t = delta_t + gamma * lambda * (1-done_t) * A_{t+1}
return_t = A_t + V(s_t)
```

计算：
- `advantages`
- `returns`

随后对 advantage 标准化。

### 11.3 PPO-clip

在 mini-batch 上：

```text
ratio = exp(logp_new - logp_old)
surr1 = ratio * A
surr2 = clip(ratio, 1-eps, 1+eps) * A
policy_loss = -mean(min(surr1, surr2))
value_loss = mean((V-return)^2)
loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
```

### 11.4 mini-batch 与多 epoch

当前对同一批 rollout 会：
- 打乱顺序
- 切 mini-batch
- 训练多个 epoch

这正是 PPO 的典型多轮重用样本方式。

### 11.5 early stop by KL

如果某个 epoch 后：

```text
mean_kl > target_kl
```

则提前停止这一轮 PPO 优化，以避免策略更新幅度过大。

## 12. 发题逻辑

训练结束后，`dyppo` 不会直接返回固定 teacher 题组，而是会根据当前 respondent 动态生成 block。

每一题的生成过程：
1. 用当前 respondent 和已挑中的 `picked` 构造 `state_x`
2. actor 前向得到对所有 candidates 的概率
3. 与 efficient prior 混合
4. 只在尚未选过的 `remain_idx` 中归一化
5. 按 epsilon-greedy 或概率抽样选一题
6. 把该题加入 `picked`
7. 继续下一题

因此：

```text
dyppo 调整的是 combo 的选择顺序与具体内容，但变量与 levels 仍然固定。
```

## 13. 在线更新逻辑

respondent 提交完整个 block 后：
1. 把实际下发的 task 映射回 action 索引
2. 根据本次 respondent 的 choices 重构一个 episode
3. 用当前 actor/critic 权重作为初始化
4. 对这一整份问卷做若干轮 PPO 更新
5. 写回新的 `policy_state`

所以当前更新粒度是：

```text
一个 respondent / 一个 block 更新一次
```

不是每题更新一次。

## 14. 一句话总结

```text
dyppo 是 efficient design 的 PPO 升级版：它保留固定 variables 和固定 attribute levels 的离散设计空间，不改变量定义，而是在这个空间上依据 respondent 状态和历史反馈动态调整 combo，并通过在线更新逐步把题组分发策略学出来。
```
