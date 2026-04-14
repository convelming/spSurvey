from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app import (
    _build_candidate_pool_for_ppo,
    _ensure_spec_policy,
    _load_design_policy_state,
    _spec_id_from_payload,
)
from engine.dynamicPPO import (
    TORCH_AVAILABLE,
    _collect_attr_categories,
    _infer_attr_width,
    _load_pop_stats,
    _sample_respondents,
    _utility_for_alt,
    online_update_dynamic_ppo,
    train_dynamic_ppo,
)
from engine.storage import load_json, load_jsonl, utc_now_iso

DATA_DIR = PROJECT_DIR / 'data'
TEST_ROOT = PROJECT_DIR / 'test'
ALGO_TEST_DIR = TEST_ROOT / 'dyppo'
RUNS_DIR = ALGO_TEST_DIR / 'runs'
RUN_STAMP = datetime.now().strftime('%Y%m%d_%H%M')
RUN_DIR = RUNS_DIR / RUN_STAMP

DESIGN_SAVE_NAME = 'sp_efficient_20260324_175903_SZ5V'
DESIGN_FILE = DATA_DIR / 'sp_design' / f'{DESIGN_SAVE_NAME}.json'
CONFIG_FILE = DATA_DIR / 'config.json'

ROWS_JSONL = RUN_DIR / 'sz5v_respondent_rows.jsonl'
TRACE_JSON = RUN_DIR / 'sz5v_flow_trace.json'
SUMMARY_JSON = RUN_DIR / 'sz5v_training_summary.json'
CANDIDATE_POOL_JSON = RUN_DIR / 'sz5v_candidate_pool.json'


def _deterministic_rng(*parts: str) -> np.random.Generator:
    h = hashlib.sha1('||'.join(parts).encode('utf-8')).hexdigest()[:16]
    seed = int(h, 16) % (2**32 - 1)
    return np.random.default_rng(seed)


def _attr_dim_names(pop_stats: dict) -> list[str]:
    key_format = str((pop_stats or {}).get('key_format', '') or '').strip()
    if key_format:
        return [str(x).strip() or f'attr_{i}' for i, x in enumerate(key_format.split('|'))]
    width = _infer_attr_width(pop_stats)
    return [f'attr_{i}' for i in range(width)]


def _simulate_choices_for_row(respondent: dict, tasks: list[dict], beta_defaults: dict) -> dict:
    respondent_id = str((respondent or {}).get('respondent_id', ''))
    choices = {}
    for t in tasks:
        tid = str((t or {}).get('id', ''))
        alts = (t or {}).get('alternatives', {}) if isinstance((t or {}).get('alternatives', {}), dict) else {}
        if not tid or not alts:
            continue
        scored = []
        for alt_name, attrs in alts.items():
            scored.append((str(alt_name), _utility_for_alt(str(alt_name), attrs or {}, respondent, beta_defaults)))
        best_u = max(u for _, u in scored)
        best_alts = [a for a, u in scored if abs(u - best_u) <= 1e-12]
        rng = _deterministic_rng(DESIGN_SAVE_NAME, respondent_id, tid)
        chosen = best_alts[int(rng.integers(0, len(best_alts)))]
        choices[tid] = chosen
    return choices


def _collect_rows(design_payload: dict, tasks: list[dict], config: dict) -> tuple[list[dict], dict]:
    beta_defaults = (design_payload or {}).get('beta_defaults', {}) or {}
    pop_stats = _load_pop_stats(design_payload, data_dir=DATA_DIR, config=config if isinstance(config, dict) else {})
    attr_categories = _collect_attr_categories(pop_stats)
    attr_dim_names = _attr_dim_names(pop_stats)
    dyn_cfg = ((config or {}).get('dynamic_ppo', {}) if isinstance((config or {}).get('dynamic_ppo', {}), dict) else {})
    seed = int((dyn_cfg.get('seed', 42)) or 42)
    design_options = (design_payload or {}).get('design_options', {}) if isinstance((design_payload or {}).get('design_options', {}), dict) else {}
    dyppo_opts = (design_options.get('dyppo', {}) if isinstance(design_options.get('dyppo', {}), dict) else {})
    # `test_respondents` 仅用于手动覆盖调试规模；
    # 若未显式指定，则默认与正式训练的 `dynamic_ppo.train_respondents` 保持一致。
    n_rows = int(dyppo_opts.get('test_respondents', dyn_cfg.get('train_respondents', 300)) or dyn_cfg.get('train_respondents', 300) or 300)
    respondents = _sample_respondents(pop_stats, n_rows, seed)
    rows = []
    for respondent in respondents:
        rid = str((respondent or {}).get('respondent_id', '')).strip()
        choices = _simulate_choices_for_row(respondent, tasks, beta_defaults)
        row = {
            'respondent_id': rid,
            'design_save_name': DESIGN_SAVE_NAME,
            'generated_at': utc_now_iso(),
            'respondent': {
                'respondent_id': rid,
                'zone_id': respondent.get('zone_id'),
                'attr_segments': list((respondent or {}).get('attr_segments', []) or []),
                'attr_key': '|'.join([str(x) for x in ((respondent or {}).get('attr_segments', []) or [])]),
                'attr_dim_names': list(attr_dim_names),
            },
            'tasks': deepcopy(tasks),
            'choices': choices,
        }
        rows.append(row)
    meta = {
        'attr_dim_names': list(attr_dim_names),
        'attr_categories': deepcopy(attr_categories),
        'n_rows': len(rows),
        'n_rows_source': 'dyppo.test_respondents' if dyppo_opts.get('test_respondents') is not None else 'config.dynamic_ppo.train_respondents',
    }
    return rows, meta


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def _task_sig(task: dict) -> str:
    alts = (task or {}).get('alternatives', {}) if isinstance(task, dict) else {}
    norm = {}
    for alt, attrs in alts.items():
        norm[str(alt)] = {str(k): attrs.get(k) for k in sorted((attrs or {}).keys())}
    return hashlib.sha1(str(sorted(norm.items())).encode('utf-8')).hexdigest()[:16]


def run() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    if not TORCH_AVAILABLE:
        raise SystemExit(
            "PyTorch未安装，无法执行 sz5v_debug_flow.py 的 dynamicPPO 训练测试。\n"
            "请先安装 torch，例如：\n"
            "python3 -m venv .venv\n"
            "source .venv/bin/activate\n"
            "pip install torch"
        )

    design_rec = load_json(DESIGN_FILE, {})
    config = load_json(CONFIG_FILE, {})
    payload = (design_rec or {}).get('payload', {}) if isinstance((design_rec or {}).get('payload', {}), dict) else {}
    tasks = (design_rec or {}).get('preview_tasks', []) if isinstance((design_rec or {}).get('preview_tasks', []), list) else []
    if not tasks:
        raise RuntimeError('design preview_tasks is empty, cannot build test rows')

    # 1) Generate one-line-per-respondent synthetic answer rows.
    rows, row_meta = _collect_rows(payload, tasks, config)
    _write_jsonl(ROWS_JSONL, rows)
    rows_for_train = load_jsonl(ROWS_JSONL)

    # 2) Train dynamicPPO once with the selected design tasks.
    loaded_policy_state, _rec_for_save = _load_design_policy_state(DESIGN_SAVE_NAME)
    spec_id = _spec_id_from_payload(payload if isinstance(payload, dict) else {})
    policy_state: dict = _ensure_spec_policy(loaded_policy_state, spec_id)
    candidate_pool = _build_candidate_pool_for_ppo(payload, pool_size=120)
    efficient_result = {
        'comb': deepcopy(tasks),
        'd_error': {'value': None},
    }
    expert_sigs = {_task_sig(t) for t in tasks if isinstance(t, dict)}
    candidate_records = []
    overlap_count = 0
    for idx, task in enumerate(candidate_pool, start=1):
        sig = str((task or {}).get('sig') or _task_sig(task))
        is_expert = sig in expert_sigs
        if is_expert:
            overlap_count += 1
        candidate_records.append(
            {
                'idx': idx,
                'sig': sig,
                'is_in_expert_preview': is_expert,
                'alternatives': deepcopy((task or {}).get('alternatives', {}) or {}),
            }
        )
    CANDIDATE_POOL_JSON.write_text(
        json.dumps(
            {
                'design_save_name': DESIGN_SAVE_NAME,
                'candidate_pool_size': len(candidate_pool),
                'expert_preview_size': len(tasks),
                'overlap_with_expert': overlap_count,
                'candidate_pool': candidate_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )

    #
    trained = train_dynamic_ppo(
        payload=payload,
        policy_state=policy_state,
        candidate_pool=deepcopy(candidate_pool),
        expert_result=efficient_result,
        rows=rows_for_train,
        current_respondent=((rows_for_train[0] or {}).get('respondent', {}) if rows_for_train else {}),
        data_dir=DATA_DIR,
        config=config if isinstance(config, dict) else {},
    )

    # 3) Replay each row as online updates (one respondent per step).
    online_updates = []
    for i, row in enumerate(rows, start=1):
        # Breakpoint B: inspect row['tasks']/row['choices'] before each update.
        out = online_update_dynamic_ppo(
            payload=payload,
            policy_state=policy_state,
            tasks=(row.get('tasks') or []),
            choices=(row.get('choices') or {}),
            respondent=(row.get('respondent') or {}),
        )
        online_updates.append(
            {
                'step': i,
                'respondent_id': row.get('respondent_id'),
                'n_choices': len((row.get('choices') or {}).keys()),
                'update': out,
            }
        )

    trace = {
        'design_save_name': DESIGN_SAVE_NAME,
        'design_file': str(DESIGN_FILE),
        'rows_jsonl': str(ROWS_JSONL),
        'n_rows': len(rows),
        'rows_loaded_for_train': len(rows_for_train),
        'policy_state_loaded': bool(loaded_policy_state),
        'candidate_pool_json': str(CANDIDATE_POOL_JSON),
        'candidate_pool_size': len(candidate_pool),
        'expert_preview_size': len(tasks),
        'candidate_expert_overlap': overlap_count,
        'row_meta': row_meta,
        'train_output': trained,
        'online_updates': online_updates,
        'policy_state_after': policy_state,
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            {
                'design_save_name': DESIGN_SAVE_NAME,
                'generated_rows': len(rows),
                'rows_loaded_for_train': len(rows_for_train),
                'policy_state_loaded': bool(loaded_policy_state),
                'candidate_pool_size': len(candidate_pool),
                'expert_preview_size': len(tasks),
                'candidate_expert_overlap': overlap_count,
                'train_model_state': trained.get('model_state', {}),
                'train_iteration_log': trained.get('iteration_log', []),
                'online_update_steps': len(online_updates),
                'final_policy_version': int(policy_state.get('response_count', 0) or 0),
                'row_meta': row_meta,
                'outputs': {
                    'rows_jsonl': str(ROWS_JSONL),
                    'trace_json': str(TRACE_JSON),
                    'candidate_pool_json': str(CANDIDATE_POOL_JSON),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )

    TRACE_JSON.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding='utf-8')

    print('DONE')
    print(f'  rows_jsonl: {ROWS_JSONL}')
    print(f'  candidate_pool_json: {CANDIDATE_POOL_JSON}')
    print(f'  trace_json: {TRACE_JSON}')
    print(f'  summary_json: {SUMMARY_JSON}')
    print(f'  generated_rows: {len(rows)}')


if __name__ == '__main__':
    run()
