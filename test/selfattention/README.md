# selfattention tests

当前主测试入口：
- `gz_pt_share_selfattention_debug_flow.py`

旧入口兼容文件：
- `xjh_limit1_selfattention_debug_flow.py`

旧入口只负责转发到当前主脚本，便于 PyCharm 旧 Run Configuration 继续使用。

## 测试目标

本目录测试当前唯一保留的 SP Design 配置：
- `data/sp_design/gz_pt_share_variable_universe_v1.json`

脚本会把保存文件中的 `payload.design_spec` 作为基础设计配置，并在测试时临时切换为 `selfattention`。原始配置文件仍保留 `type=efficient`，因为 efficient design 仍用于生成 teacher/warmup 参考方案。

## 数据流

1. 读取 `gz_pt_share_variable_universe_v1.json`。
2. 若 `preview_tasks` 为空，则即时运行一次 efficient design，生成 teacher preview。
3. 按当前 design spec 构造 feasible combo pool，作为 SelfAttention 的动作候选空间。
4. 按 `data/config.json` 中的 PopSim/统计配置模拟 respondent 行数据。
5. 一行 JSONL 表示一个 respondent 对一整份 SP block 的填写结果。
6. 执行 SelfAttention block generator 的离线 warmup。
7. 可选执行逐 respondent 在线更新。

注意：当前口径是一份问卷 block 一次性生成、一次性填写，不再把同一份问卷中的 SP 题目解释成 shifted-right 的题内时序。

## 快速 smoke test

```bash
python3 test/selfattention/gz_pt_share_selfattention_debug_flow.py \
  --train-respondents 4 \
  --train-epochs 1 \
  --max-online-updates 1 \
  --timer-interval 10
```

## 较完整调试

```bash
python3 test/selfattention/gz_pt_share_selfattention_debug_flow.py \
  --train-respondents 120 \
  --train-epochs 50 \
  --max-online-updates 20 \
  --timer-interval 30
```

## 运行产物

运行产物统一写入：
- `test/selfattention/runs/YYYYMMDD_HHMM/`

主要文件：
- `gz_pt_share_selfattention_rows.jsonl`：一行一个 respondent 的模拟填写结果。
- `gz_pt_share_selfattention_candidate_pool.json`：feasible combo pool 与 efficient teacher 的重叠检查。
- `gz_pt_share_selfattention_trace.json`：完整 payload、训练输出、在线更新记录。
- `gz_pt_share_selfattention_summary.json`：摘要指标和输出路径。
