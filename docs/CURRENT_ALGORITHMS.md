# 当前算法文档索引

更新时间：2026-04-28

当前项目的 SP 设计文档按三层升级关系组织：

1. [Efficient Design 详细说明](/Users/convel/PycharmProjects/spSurvey/docs/EFFICIENT_DESIGN.md)
2. [dynamicPPO 详细说明](/Users/convel/PycharmProjects/spSurvey/docs/DYPPO.md)
3. [SelfAttention 并行题组生成详细说明](/Users/convel/PycharmProjects/spSurvey/docs/SELFATTENTION.md)
4. [ATTENTION 模块维度说明](/Users/convel/PycharmProjects/spSurvey/docs/ATTENTION.md)
5. [三种方法对照表](/Users/convel/PycharmProjects/spSurvey/docs/ALGORITHM_COMPARISON.md)

## 三层升级关系

### 1. Efficient Design
- 设计空间由前端 `design_spec` 完整给定。
- 变量固定。
- 每个变量的 levels 固定。
- 后端在离散设计空间内，以 Bayesian D-error 为目标执行局部搜索 `row-exchange`。

### 2. dynamicPPO
- 可以理解为 `efficient design` 的强化学习升级版。
- 升级点在于：题组 `combo` 不再一次性固定，而是会根据 respondent 状态、历史题组表现、已收集数据动态调整。
- 但它仍然继承 `efficient design` 的设计边界：
  - 变量集合不变。
  - 每个变量的 levels 不变。
  - 动作空间仍然是 feasible combo pool。

### 3. SelfAttention
- 可以理解为在 `dyppo` 之上的表达能力升级版。
- 升级点不只是“重新选 combo”，而是把题目生成过程本身做成并行 block 生成器：
  - 变量可以激活或关闭；
  - 变量值可以直接生成，而不只在固定 levels 中挑选；
  - 整份问卷 block 一次性生成；
  - respondent 也是一次性看到并填写整份问卷。
- 它的主结构可以概括为：
  - `encoder`
    - `X_rp / X_env / X_hist / X_cand`
    - `->` 各自投影到 `d_model`
    - `-> concat`
    - `-> Multi-Head Self-Attention + Add&Norm + FFN + Add&Norm`
    - `-> H_enc`
  - `respondent sampling branch`
    - `H_enc -> pooled global state`
    - `-> respondent_target_head`
    - `-> sampling_recommendation`
  - `parallel block decoder`
    - `Q_slot^(0) ∈ R^[B,T_max,d_model]`
    - `-> Question-Set Self-Attention`，不使用 causal masked attention
    - `-> Cross-Attention(H_enc)`
    - `-> FFN`
    - `-> count / slot_select / mask / value / score heads`
  - 口径说明
    - `X_hist` 对应前 `n-1` 份已完成问卷 / blocks 提炼出的历史 SP 上下文
    - `question queries` 是并行候选 slot，不是 shifted-right 序列，也不是最终页面题号
    - `count_head` 决定本次问卷题数
    - `slot_select_head` 从候选 slot 中选出最终进入问卷的题集合
    - `mask_head` 决定题内变量结构
    - `value_head` 在 `mask` 条件下给激活变量赋值
    - `respondent_target_head` 输出剩余样本的定向采样建议，不直接强制抽样

## 当前主代码入口

- `app.py`
  负责 design payload 校验、candidate pool 构造、D-error 计算、设计存取、发题与在线更新调用。
- `engine/dynamicPPO.py`
  负责 dynamicPPO 的状态构造、rollout 构造、GAE、PPO-clip、在线更新。
- `engine/selfattention.py`
  当前文档口径已经切到“encoder 采样建议分支 + 并行 question queries + count_head + slot_select_head + mask/value 顺序结构”。代码实现以 set/block generator 为目标，并保持 `app.py` 的接口兼容。
- `data/config.json`
  负责 `dynamic_ppo`、`self_attention`、`data_sources` 的运行时配置。

## 推荐阅读顺序

1. 先看 [Efficient Design 详细说明](/Users/convel/PycharmProjects/spSurvey/docs/EFFICIENT_DESIGN.md)
2. 再看 [dynamicPPO 详细说明](/Users/convel/PycharmProjects/spSurvey/docs/DYPPO.md)
3. 再看 [SelfAttention 并行题组生成详细说明](/Users/convel/PycharmProjects/spSurvey/docs/SELFATTENTION.md)
4. 对 attention 维度和 `count/mask/value` 关系有疑问时，再看 [ATTENTION 模块维度说明](/Users/convel/PycharmProjects/spSurvey/docs/ATTENTION.md)
5. 横向比较时看 [三种方法对照表](/Users/convel/PycharmProjects/spSurvey/docs/ALGORITHM_COMPARISON.md)
