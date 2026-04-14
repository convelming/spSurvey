# Efficient Design 详细说明

更新时间：2026-04-13

本文档只描述当前项目中仍在生效的 `efficient` 设计逻辑，对应代码主要在：
- `/Users/convel/PycharmProjects/spSurvey/app.py`
  - `validate_sp_design_payload()`
  - `_build_param_keys()`
  - `_design_matrix_generic()`
  - `_bayesian_d_error_generic()`
  - `_build_tasks_from_spec()`
  - `_design_axes_from_spec()`
  - `_full_combo_count()`
  - `_infer_dominance_directions()`
  - `_task_satisfies_conditions()`
  - `_compute_efficient()`

## 1. 当前 efficient 的定位

当前项目里的 `efficient` 不是历史版 planning/seed design 的旧路径，而是：
- 直接接收前端传来的 `design_spec`
- 直接在后端生成题组
- 直接用当前代码内的 Bayesian D-error 计算题组信息量
- 在当前可行题组集合上执行基于单行替换的局部改进迭代 `row-exchange`

也就是说，当前 `efficient` 是一个完整可运行的后端实现，不再依赖已删除的 `engine/design.py` 或 `planning.py`。

## 2. 输入对象与基本概念

### 2.1 前端输入的核心字段

`efficient` 当前最关心的 payload 字段有：
- `design_spec`
- `beta_defaults`
- `beta_bounds`
- `sample_size`
- `target_block_sample`
- `design_options.efficient.tasks_per_person`
- `design_options.efficient.row_exchange_iterations`

其中 `design_spec` 结构为：

```text
design_spec
  ├── alternatives: list[alternative]
  │     ├── name
  │     └── variables: list[variable]
  │           ├── name
  │           └── levels: list[level]
  ├── asc_base_alternative
  └── conditions
```

### 2.2 当前算法中的术语

- `row`
  一道 SP 题。
- `block`
  一个 respondent 会拿到的一组题。
- `tasks_per_person`
  每个 respondent 看到多少题。
- `rows`
  当前设计总题数，等于所有 blocks 的题数总和。
- `sample_size`
  目标样本量。
- `target_block_sample`
  每个 block 预计服务多少 respondent，用于估计 block 数量。

## 3. 参数键与设计矩阵

### 3.1 参数键如何生成

`_build_param_keys(spec)` 当前把设计参数分成两类：

1. ASC 参数
2. 变量参数

生成规则：
- 若某 alternative 不是 `asc_base_alternative`，则生成 `"<alt>.asc"`
- 每个变量生成 `"<alt>.<var>"`

例如：

```text
asc_base_alternative = car
alternatives = [car, bus]
variables:
  car: time, cost
  bus: time, cost, wait
```

则参数键是：

```text
bus.asc
car.time
car.cost
bus.time
bus.cost
bus.wait
```

### 3.2 单个 task 的设计矩阵

`_design_matrix_generic(task, param_keys)` 会把一个 task 变成 `X`。

设：
- 一道题有 `J` 个 alternatives
- 参数个数为 `K`

则：

```text
X ∈ R^[J, K]
```

含义：
- 第 `j` 行对应第 `j` 个 alternative
- ASC 只在对应 alternative 上为 1，其它行为 0
- 变量值只在对应 alternative 的行出现

### 3.3 小例子

如果一道题是：

```json
{
  "alternatives": {
    "car": {"time": 30, "cost": 10},
    "bus": {"time": 40, "cost": 3, "wait": 5}
  }
}
```

参数键为：

```text
[bus.asc, car.time, car.cost, bus.time, bus.cost, bus.wait]
```

则设计矩阵 `X` 为：

```text
car 行: [0, 30, 10,  0, 0, 0]
bus 行: [1,  0,  0, 40, 3, 5]
```

## 4. 当前 D-error 的计算逻辑

`_bayesian_d_error_generic()` 是当前 efficient 的核心评价函数。

### 4.1 先验均值与方差

对每个参数键 `k`：
- 均值来自 `beta_defaults[k]`
- 标准差优先由 `beta_bounds[k].min/max` 推导
- 若没有 bounds，则用默认规则：

```text
std_k = max(abs(beta_default_k) * 0.2, 0.1)
```

若有上下界：

```text
std_k = max((max_k - min_k) / 4, 1e-3)
```

这里本质上是假设参数大致服从正态分布，且上下界覆盖约 4 个标准差左右的范围。

### 4.2 Bayesian draws

当前会抽若干组参数：

```text
beta^(m) ~ Normal(mean_vec, std_vec)
m = 1, 2, ..., M
```

当前默认：
- `beta_draws = 24` 用于最终评价
- 迭代中某些位置会用更少的 draws 以降低计算量

### 4.3 Fisher 信息矩阵构造

对于每组 `beta^(m)`，对每道题执行：

```text
u = X beta
p = softmax(u)
W = diag(p) - p p^T
M += X^T W X
```

其中：
- `u ∈ R^J` 是每个 alternative 的系统效用
- `p ∈ R^J` 是该题的 logit 选择概率
- `W ∈ R^[J, J]` 是 multinomial logit 对应的信息权重矩阵
- `M ∈ R^[K, K]` 是所有题累加后的 Fisher 信息矩阵

### 4.4 Ridge 修正与 logdet

为了避免矩阵奇异，当前实现会做：

```text
M <- M + 1e-6 I
```

然后计算：

```text
sign, logdet = slogdet(M)
```

若 `sign <= 0`，直接认为这个设计不可接受，返回极大惩罚值：

```text
D_error = 1e9
```

否则：

```text
D_error = exp((-logdet) / K)
```

### 4.5 多抽样平均

对所有抽样的 `D_error^(m)` 取平均：

```text
D_error_final = mean_m D_error^(m)
```

这就是当前代码里 Bayesian D-error 的最终输出。

## 5. 条件约束与 dominance 过滤

在真正开始 row-exchange 之前，后端先做两类可行性过滤。

### 5.1 条件约束 `conditions`

`_task_satisfies_conditions(task, conditions)` 支持按行输入的比较式，例如：

```text
car.time > bus.time
car.cost <= 20
bus.wait < car.time < 60
```

处理流程是：
1. 逐行读取条件
2. 解析比较符号 `> >= < <=`
3. 把 `alt.var` 替换为 task 里的数值
4. 逐段比较，任何一段不满足则整道题剔除

### 5.2 dominance 过滤

`_infer_dominance_directions(spec, beta_defaults)` 当前不是按变量名预先固定方向，而是根据先验系数方向自动推断。

某变量只有在以下条件同时满足时才参与 dominance 判断：
- 至少出现在两个 alternatives 中
- 这些 alternatives 上的先验系数符号一致

方向定义：
- `-1`：值越小越好
- `+1`：值越大越好

然后 `_alt_dominates_generic(a, b, directions)` 检查：
- 在所有可比较变量上，A 不差于 B
- 且至少一维严格优于 B

若成立，则判定存在 dominance，被支配的题剔除。

### 5.3 当前 dominance 的边界

这意味着当前实现非常保守：
- 如果 ASC 或先验符号不足以判定方向，就不做 dominance 过滤
- 不会因为没有把握就硬判 dominated

这是为了避免误杀可用题目。

## 6. 初始题组的构造方式

### 6.1 不是在 full factorial 组合空间上直接求解全局最优

当前 `_compute_efficient()` 的起点不是“先枚举全部 full factorial 组合，再在整个离散空间上直接求解全局最优”，而是采用“先构造初始设计，再做局部改进”的两阶段流程：

1. 依据 `rows = tasks_per_person × blocks` 确定当前需要生成的题目总数。
2. 调用 `_build_tasks_from_spec(spec, rows, seed)` 构造一组初始题组。
3. 对初始题组执行 `conditions` 过滤；若过滤后为空，则退化为重新生成一个最小可展示题组。
4. 计算初始题组的 Bayesian D-error，作为后续局部搜索的基准值。
5. 在该初始题组上执行单行替换式 `row-exchange` 迭代。

### 6.2 `_build_tasks_from_spec()` 的行为

对每一行 `i`，该函数会按以下方式构造题目：
- 遍历 `design_spec` 中的每个 `alternative`
- 遍历该 `alternative` 下的每个变量
- 对第 `v_idx` 个变量，从其 `levels` 中选取一个取值
- 将各变量取值拼装成该行题目的 `alternatives`

更具体地说，代码使用：

```text
level_index = (i + v_idx + random_offset) mod len(levels)
```

其中：
- `i` 表示当前正在生成的第几行题目
- `v_idx` 表示该变量在当前 `alternative` 内的顺序位置
- `random_offset` 由给定随机种子下的伪随机数生成器产生

因此，这里的初始题组生成属于一种“带随机扰动的构造式初始化”：
- 它会尽量让不同 row 在 level 组合上产生位移差异
- 它可重复，因为随机种子固定时输出可复现
- 它本身不是一个优化过程，而只是为后续 D-error 改进提供可行起点

### 6.3 为什么不直接从 full factorial 空间做全局优化

原因在于，当前 `efficient` 的实现目标是为前端页面提供可在有限时间内生成的可行设计，而不是在 full factorial 离散空间中执行高计算量的全局组合优化。

因此，当前实现采用的是：
- 先以构造式方法生成一个可行初始设计
- 再通过局部替换逐步降低 D-error

这更准确地说应理解为“基于初始设计的局部搜索”，而不是“对完整 full factorial 空间的一步式全局最优求解”。

## 7. rows 和 blocks 是怎么定的

当前 `_compute_efficient()` 中：

```text
blocks = max(1, round(sample_size / target_block_sample))
rows = tasks_per_person * blocks
```

这意味着：
- 若总体样本量较小，可能只有 1 个 block
- 若总体样本量较大，block 数会增加
- 每个 block 默认包含 `tasks_per_person` 道题

### 7.1 例子

若：
- `sample_size = 600`
- `target_block_sample = 100`
- `tasks_per_person = 8`

则：

```text
blocks = round(600 / 100) = 6
rows = 8 * 6 = 48
```

最终会生成 48 道题，并按每 8 道题切成一个 block。

## 8. Row-exchange 单行替换式迭代的详细过程

当前 `efficient` 的核心改进步骤不是完整的 `coordinate-exchange` 或多行联合替换框架，而是一个基于“单行替换 + 严格改进接受”的 `row-exchange` 局部搜索过程。

### 8.1 初始状态

设：
- 当前设计题组为 `D^(t)`
- 其对应目标函数值为 `d^(t)`，其中目标函数为 Bayesian D-error
- 最大迭代次数为 `T`

### 8.2 单步迭代做什么

第 `t` 次迭代包含以下步骤：

1. 复制当前设计，形成候选容器：

```text
D_cand <- copy(D^(t))
```

2. 选一个待替换位置：

```text
pos_t <- t mod |D^(t)|
```

这表示当前实现按题组位置做循环扫描，而不是在每步迭代中随机选位。

3. 使用新的随机种子重新构造 1 道候选题：

```text
q_t <- generate_one_task(spec, seed = 1000 + t)
```

4. 若 `q_t` 满足 `conditions` 可行性约束，则将其放入 `pos_t` 位置，形成候选设计：

```text
D'_t = replace(D^(t), pos_t, q_t)
```

5. 重新计算候选设计 `D'_t` 的 Bayesian D-error：

```text
d'_t = D_error(D'_t)
```

6. 若 `d'_t < d^(t)`，则接受本次替换；否则拒绝本次替换并保留原设计：

```text
if d'_t < d^(t):
    D^(t+1) <- D'_t
    d^(t+1) <- d'_t
else:
    D^(t+1) <- D^(t)
    d^(t+1) <- d^(t)
```

因此，单步迭代的本质是：在当前设计中只替换 1 行，并用目标函数值是否严格下降来决定是否接受该替换。

### 8.3 接受准则

当前实现采用严格改进接受准则：

```text
accept(q_t) <=> D_error(D'_t) < D_error(D^(t))
```

没有：
- 模拟退火
- 随机接受差解
- tabu memory
- 多行联合交换

因此，更正式地说，它属于一种单调下降的局部搜索算法：只有当新设计在当前评价函数下优于旧设计时，才发生状态更新。

### 8.4 伪代码

```text
输入: 初始设计 D^(0), 最大迭代次数 T
计算 d^(0) = D_error(D^(0))

for t in 0..T-1:
    D_cand <- copy(D^(t))
    pos_t <- t mod |D^(t)|
    q_t <- generate_one_task(spec, seed=1000+t)

    if satisfies_conditions(q_t):
        D'_t <- replace(D_cand, pos_t, q_t)
        d'_t <- D_error(D'_t)

        if d'_t < d^(t):
            D^(t+1) <- D'_t
            d^(t+1) <- d'_t
        else:
            D^(t+1) <- D^(t)
            d^(t+1) <- d^(t)
    else:
        D^(t+1) <- D^(t)
        d^(t+1) <- d^(t)
```

### 8.5 当前 row-exchange 的性质

优点：
- 目标函数含义明确，每一步操作都可追踪
- 任何已接受的替换都保证 D-error 不上升
- 实现复杂度低，便于在交互式页面中快速返回结果

局限：
- 由于只接受局部严格改进，算法可能停留在局部最优附近
- 每次仅替换 1 行，搜索半径有限
- 初始设计若较弱，后续改进幅度可能受限
- 当前目标函数中没有显式加入 `level balance`、`overlap penalty` 等附加项，主要依赖 D-error 与可行性过滤

## 9. 最终题组如何编号

在 row-exchange 完成后，当前会把每道题重新补足：
- `block`
- `row_in_block`
- `id`

规则是：

```text
block = idx // tasks_per_person + 1
row_in_block = idx % tasks_per_person + 1
id = preview_b{block}_r{row_in_block}
```

这样前端就能按 block 和 row 顺序展示。

## 10. Efficient 的输出结构

当前 `_compute_efficient()` 返回：

- `mode = efficient_compute`
- `recommendation`
- `comb`
- `d_error`
- `iteration_log`
- `model_state = None`

其中：

### 10.1 `recommendation`

包含：
- `tasks_per_person`
- `blocks`
- `rows`
- `row_exchange_iterations`

### 10.2 `comb`

是最终题组数组，每个元素包含：
- `id`
- `alternatives`
- `block`
- `row_in_block`

### 10.3 `d_error`

包含：
- `value`
- `beta_draws`

### 10.4 `iteration_log`

记录：
- 初始 D-error
- 若干中间迭代点的 D-error
- 最终 D-error

## 11. Efficient 在 dyppo / selfattention 中的角色

虽然 `efficient` 本身可以单独作为设计策略，但在 `dyppo` 和 `selfattention` 中，它还有第二个角色：
- 作为 `expert prior`
- 只提供参考题组
- 不接管动作空间

也就是说：
- 如果页面选择 `efficient`，那它自己就是最终设计器
- 如果页面选择 `dyppo` 或 `selfattention`，它只是 teacher / warm-start / prior

## 12. 当前实现的边界与建议

### 12.1 当前实现已经做了什么

已经做了：
- 通用 alternatives / variables 解析
- 通用 D-error
- 通用条件约束
- 基于先验方向的 dominance 过滤
- block / row 组织
- 前端可直接展示的输出

### 12.2 当前还没有做什么

当前还没有：
- 更完整的 coordinate-exchange
- 多行联动交换
- 显式 level balance 惩罚
- overlap penalty
- 多起点重启搜索
- 自适应停止准则

如果以后要继续增强 `efficient`，最合理的路径是：
1. 先保留当前接口不动
2. 在 row-exchange 内部加入多起点或更充分的替换策略
3. 再考虑 balance/overlap 的附加目标
