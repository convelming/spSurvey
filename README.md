!!!! 这个项目基本全部由CodeX生成，全部算法经过人工校核，但由于某些点的修改会导致整个项目的架构和逻辑改动，从而导致某些算法在人工校核后有变化，还需要大量工作去校核与修正，望周知！

# rp and sp Questinnaires

当前项目是一个基于 Flask 的出行调查系统，包含：
- RP/Profile 页
- Trip Diary 页
- SP Design 页
- SP Questionnaire 页
- Web Map 选址页
- 数据收集情况页

当前主流程已经统一到 `app.py + templates/ + static/ + engine/` 这套架构上；历史上用于预生成候选池、seed design、行块启发式推荐的旧文件已移除。

详细算法文档见：
- [`docs/CURRENT_ALGORITHMS.md`](/Users/convel/PycharmProjects/mappoQuestinnaire/docs/CURRENT_ALGORITHMS.md)
- [`docs/EFFICIENT_DESIGN.md`](/Users/convel/PycharmProjects/mappoQuestinnaire/docs/EFFICIENT_DESIGN.md)
- [`docs/DYPPO.md`](/Users/convel/PycharmProjects/mappoQuestinnaire/docs/DYPPO.md)
- [`docs/SELFATTENTION.md`](/Users/convel/PycharmProjects/mappoQuestinnaire/docs/SELFATTENTION.md)
- [`docs/ALGORITHM_COMPARISON.md`](/Users/convel/PycharmProjects/mappoQuestinnaire/docs/ALGORITHM_COMPARISON.md)

## 当前项目结构

```text
mappoQuestinnaire/
├── app.py
├── README.md
├── requirements.txt
├── engine/
│   ├── __init__.py
│   ├── config.py
│   ├── dynamicPPO.py
│   ├── selfattention.py
│   └── storage.py
├── static/
│   └── app.js
├── templates/
│   ├── collection_dashboard.html
│   ├── index.html
│   ├── profile.html
│   ├── sp_design.html
│   ├── sp_questionnaire.html
│   ├── trip_diary.html
│   └── web_map.html
├── test/
│   ├── README.md
│   ├── dyppo/
│   │   ├── interactive_dyppo_live_demo.py
│   │   ├── interactive_dyppo_manual_input_demo.py
│   │   ├── sz5v_debug_flow.py
│   │   └── runs/
│   ├── efficient/
│   │   └── README.md
│   └── selfattention/
│       ├── README.md
│       ├── xjh_limit1_selfattention_debug_flow.py
│       └── runs/
└── data/
    ├── config.json
    ├── sp_design/
    │   └── weights/
    ├── respondents/
    ├── profile_submissions.jsonl
    ├── trip_diary_submissions.jsonl
    └── sp_submissions.jsonl
```

## 当前后端模块

### `app.py`
主应用入口，负责：
- 页面路由
- 设计计算接口
- respondent 数据保存
- SP 发题与提交
- design 配置存取
- collection dashboard 汇总接口

### `engine/config.py`
默认配置与数据目录常量。

### `engine/storage.py`
JSON / JSONL 读写、时间戳等基础存储工具。

### `engine/dynamicPPO.py`
`dynamicPPO` 设计策略实现：
- 候选动作来自 feasible combo pool
- efficient design 作为 expert prior
- 训练使用 rollout + GAE + PPO-clip
- 支持在线更新

### `engine/selfattention.py`
`SelfAttention` 设计策略实现：
- 与 `dynamicPPO` 共用设计输入与保存格式
- 候选动作来自 feasible combo pool
- 使用自注意力 Actor-Critic 处理 respondent/block 内状态序列
- 支持在线更新

## 页面路由

- `/` 首页
- `/survey/profile` Profile 页
- `/survey/trip-diary` Trip Diary 页
- `/survey/sp-design` SP Design 页
- `/survey/sp` SP Questionnaire 页
- `/survey/map` Web Map 页
- `/survey/collection-dashboard` 数据收集情况页

## 主要 API

- `GET /api/config`
- `POST /api/design/spec`
- `GET /api/design/spec`
- `POST /api/design/compute`
- `GET /api/design/compute/<job_id>`
- `GET /api/design/compute/<job_id>/result`
- `POST /api/design/issue`
- `POST /api/survey/profile`
- `POST /api/survey/trip-diary`
- `POST /api/survey/sp-submit`
- `GET /api/survey/respondent/<respondent_id>`
- `GET /api/dashboard/collection-summary`

## 设计类型

`sp_design` 页面当前支持三种设计类型：
- `efficient`
- `dyppo`
- `selfattention`

其中：
- `efficient` 直接按 D-error 相关逻辑生成题组
- `dyppo` 和 `selfattention` 都使用统一的 feasible combo pool
- 两类 RL/序列策略都沿用相同的 design 保存格式与 `data/sp_design/weights/*.pt` 权重文件组织方式

## 数据落盘

### Design
- `data/sp_design/*.json`
  每个保存名一个 design 文件，含：
  - `save_name`
  - `type`
  - `payload`
  - `preview_tasks`
  - `recommendation`
  - `runtime`

- `data/sp_design/weights/*.pt`
  `dyppo` / `selfattention` 对应的策略权重。

### Respondent
- `data/respondents/<respondent_id>.json`
  每个 respondent 一份汇总数据，含：
  - `profile`
  - `trip_diary`
  - `sp`

### Submission Logs
- `data/profile_submissions.jsonl`
- `data/trip_diary_submissions.jsonl`
- `data/sp_submissions.jsonl`

## 配置

运行时配置位于：
- `data/config.json`

当前有效顶层配置项：
- `dynamic_ppo`
- `self_attention`
- `data_sources`

## 运行

```bash
cd /Users/convel/PycharmProjects/spSurvey
python3 -m pip install -r requirements.txt
python3 app.py
```

默认访问：
- [http://127.0.0.1:5055](http://127.0.0.1:5055)

## 说明

本 README 仅描述当前仍在使用的项目结构。
历史遗留的以下旧文件/旧流程已移除：
- 预生成候选池与 seed design 的 bootstrap 流程
- 旧版 `engine/design.py`
- 旧版 `engine/planning.py`
- 未接入当前模板体系的 `base.html`
- 独立二维码临时脚本
- 旧版 walkthrough 文档
