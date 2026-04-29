# Efficient vs dyppo vs SelfAttention 对照表

更新时间：2026-04-28

## 1. 总体对照

| 维度 | Efficient Design | dynamicPPO | SelfAttention Set/Block Generator |
|---|---|---|---|
| 基本定位 | 传统离散设计优化 | Efficient 的 RL 升级版 | dyppo 之上的结构化生成升级版 |
| 变量集合 | 固定 | 固定 | 可选，可激活/关闭 |
| attribute levels | 固定离散 | 固定离散 | 可连续生成，不限于离散 levels |
| 动作本质 | 选出一组离散题 | 从离散 feasible combo pool 选题 | 生成候选题集合，再决定题数、选中题位、变量 mask 与 value |
| respondent 自适应 | 无或很弱 | 有 | 有，且可作用于题目结构层 |
| 样本覆盖控制 | 依赖人工配额 | 可把覆盖偏差写入 state/reward | 额外输出 respondent_target_head 采样建议 |
| 是否在线更新 | 否 | 是 | 是 |
| 生成方式 | 一次性优化题组 | 在固定离散池中动态调 combo | 条件约束下动态生成 block |
| 约束处理 | conditions + dominance | conditions + dominance + RL 分配 | conditions + 结构约束 + 连续值边界约束 |
| 主要优点 | 稳定、易解释 | 能自动调整 combo | 变量可有可无，值可连续，结构表达更灵活 |
| 主要局限 | 不会因 respondent 变化而自适应 | 变量与 levels 仍固定定义在 `design_spec` 中 | 模型、训练与推理成本更高 |

## 2. 三者的升级关系

### 2.1 Efficient -> dyppo

升级内容：
- 不再固定一套唯一题组给所有 respondent
- 可以根据 respondent 与历史反馈自动调整 combo
- 可以随着已收集数据在线更新发题策略

不变内容：
- variables 不变
- variables 的 attribute levels 不变
- 搜索空间仍然是离散的 feasible combo pool

### 2.2 dyppo -> SelfAttention Set/Block Generator

升级内容：
- 不再局限于从离散 combo 池中选题
- 可以决定变量激活与否
- 可以为变量生成连续值
- block 可以按 set/block 机制构造，而不是只从固定离散题库抽取
- 问卷内部题目没有时序语义，训练时应按集合匹配而不是按题号对齐
- encoder 可额外输出采样建议，辅助剩余样本向覆盖不足的 RP cell 倾斜

不变内容：
- 仍需满足总体逻辑约束与可行性约束
- 仍需服务于 D-error、信息增益、样本覆盖等目标

## 3. 一句话结论

```text
Efficient 解决的是固定离散设计的优化问题；dyppo 解决的是在同一离散设计空间上动态调整 combo 的问题；SelfAttention set/block generator 则进一步把题数、候选题选择、变量是否出现、变量取什么值、block 如何生成以及剩余样本采集建议都纳入可学习过程。三者并非简单的强弱关系，而是对应不同的建模层级、灵活性需求与计算成本。
```
