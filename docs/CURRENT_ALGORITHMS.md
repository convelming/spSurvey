# 当前算法文档索引

更新时间：2026-04-13

当前项目的 SP 设计文档按三层升级关系组织：

1. [Efficient Design 详细说明](/Users/convel/PycharmProjects/spSurvey/docs/EFFICIENT_DESIGN.md)
2. [dynamicPPO 详细说明](/Users/convel/PycharmProjects/spSurvey/docs/DYPPO.md)
3. [SelfAttention Encoder-Decoder 详细说明](/Users/convel/PycharmProjects/spSurvey/docs/SELFATTENTION.md)
4. [三种方法对照表](/Users/convel/PycharmProjects/spSurvey/docs/ALGORITHM_COMPARISON.md)

## 三层升级关系

### 1. Efficient Design
- 设计空间由前端 `design_spec` 完整给定。
- 变量固定。
- 每个变量的 attribute levels 固定。
- 后端在该离散设计空间内，以 Bayesian D-error 为目标执行单行替换式局部搜索 `row-exchange`。

### 2. dynamicPPO
- 可以理解为 `efficient design` 的强化学习升级版。
- 升级点在于：题组 `combo` 不再一次性固定，而是会根据 respondent 状态、历史题组表现、已收集数据动态调整。
- 但它仍然继承 `efficient design` 的设计边界：
  - 变量集合不变。
  - 每个变量的 attribute levels 不变。
  - 动作空间仍然是由这些固定变量和固定离散 levels 组合出来的 feasible combo pool。

### 3. SelfAttention Encoder-Decoder
- 可以理解为在 `dyppo` 之上的表达能力升级版。
- 升级点不只是“重新选 combo”，而是把题目生成过程本身做成 encoder-decoder：
  - 变量可以激活或关闭，即变量可有可无。
  - 变量值不再局限于预先枚举的离散 levels。
  - 变量值可以是连续值，只需满足范围约束、逻辑约束和可行性约束。
- 因而它对应的是“从离散题库选择”走向“条件约束下的动态生成”。

## 当前主代码入口

- `app.py`
  负责 design payload 校验、candidate pool 构造、D-error 计算、设计存取、发题与在线更新调用。
- `engine/dynamicPPO.py`
  负责 dynamicPPO 的状态构造、rollout 构造、GAE、PPO-clip、在线更新。
- `engine/selfattention.py`
  当前代码文件仍承载 selfattention 后端；本文档对它采用 encoder-decoder 的架构口径描述。
- `data/config.json`
  负责 `dynamic_ppo`、`self_attention`、`data_sources` 的运行时配置。

## 推荐阅读顺序

1. 先看 [Efficient Design 详细说明](/Users/convel/PycharmProjects/spSurvey/docs/EFFICIENT_DESIGN.md)
   理清固定变量、固定 levels、Bayesian D-error 与单行替换式局部改进过程。
2. 再看 [dynamicPPO 详细说明](/Users/convel/PycharmProjects/spSurvey/docs/DYPPO.md)
   理清它如何在不改变量与 levels 的前提下，自动调整 combo。
3. 最后看 [SelfAttention Encoder-Decoder 详细说明](/Users/convel/PycharmProjects/spSurvey/docs/SELFATTENTION.md)
   理清它如何进一步把变量激活、变量值生成、block 生成做成 encoder-decoder。
4. 如需横向比较，直接看 [三种方法对照表](/Users/convel/PycharmProjects/spSurvey/docs/ALGORITHM_COMPARISON.md)。
