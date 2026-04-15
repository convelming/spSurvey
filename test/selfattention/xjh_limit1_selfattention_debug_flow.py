from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app import _build_candidate_pool_for_ppo
from engine.dynamicPPO import (
    TORCH_AVAILABLE,
    _collect_attr_categories,
    _infer_attr_width,
    _load_pop_stats,
    _sample_respondents,
    _utility_for_alt,
)
from engine.selfattention import online_update_self_attention_ppo, train_self_attention_ppo
from engine.storage import load_json, load_jsonl, utc_now_iso

DATA_DIR = PROJECT_DIR / 'data'
TEST_ROOT = PROJECT_DIR / 'test'
ALGO_TEST_DIR = TEST_ROOT / 'selfattention'
RUNS_DIR = ALGO_TEST_DIR / 'runs'
RUN_STAMP = datetime.now().strftime('%Y%m%d_%H%M')
RUN_DIR = RUNS_DIR / RUN_STAMP

DEFAULT_DESIGN_SAVE_NAME = 'gz_pt_share_variable_universe_v1'
CONFIG_FILE = DATA_DIR / 'config.json'


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class RuntimeTicker:
    """后台计时器：长任务运行时周期性输出耗时，便于判断程序仍在正常执行。"""

    def __init__(self, interval_seconds: float = 30.0, enabled: bool = True) -> None:
        self.interval_seconds = max(1.0, float(interval_seconds or 30.0))
        self.enabled = bool(enabled)
        self.stage = 'initializing'
        self.start_time = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self.start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, name='selfattention-runtime-ticker', daemon=True)
        self._thread.start()

    def set_stage(self, stage: str) -> None:
        self.stage = str(stage or 'running')

    def elapsed_seconds(self) -> float:
        if not self.start_time:
            return 0.0
        return time.perf_counter() - self.start_time

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            print(
                f"[timer] elapsed={_format_elapsed(self.elapsed_seconds())} stage={self.stage}",
                flush=True,
            )


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


def _task_sig(task: dict) -> str:
    alts = (task or {}).get('alternatives', {}) if isinstance(task, dict) else {}
    norm = {}
    for alt, attrs in alts.items():
        norm[str(alt)] = {str(k): attrs.get(k) for k in sorted((attrs or {}).keys())}
    return hashlib.sha1(str(sorted(norm.items())).encode('utf-8')).hexdigest()[:16]


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
        rng = _deterministic_rng(DEFAULT_DESIGN_SAVE_NAME, respondent_id, tid)
        chosen = best_alts[int(rng.integers(0, len(best_alts)))]
        choices[tid] = chosen
    return choices


def _normalize_payload_for_selfattention(raw_payload: dict, config: dict, args: argparse.Namespace) -> dict:
    payload = deepcopy(raw_payload if isinstance(raw_payload, dict) else {})
    payload['design_type'] = 'selfattention'

    cfg = (config.get('self_attention', {}) if isinstance(config.get('self_attention', {}), dict) else {})
    dy = ((payload.get('design_options', {}) or {}).get('dyppo', {}) if isinstance((payload.get('design_options', {}) or {}).get('dyppo', {}), dict) else {})
    rec_tpr = int(payload.get('recommendation', {}).get('tasks_per_person', 0) or 0) if isinstance(payload.get('recommendation', {}), dict) else 0
    tasks_per_round = int(args.tasks_per_round or dy.get('tasks_per_round', rec_tpr or 6) or 6)

    self_opts = {
        'tasks_per_round': tasks_per_round,
        'explore_epsilon': float(args.explore_epsilon if args.explore_epsilon is not None else dy.get('explore_epsilon', cfg.get('explore_epsilon', 0.15)) or cfg.get('explore_epsilon', 0.15) or 0.15),
        'window_length': int(args.window_length or cfg.get('window_length', 16) or 16),
        'hidden_dim': int(args.hidden_dim or cfg.get('hidden_dim', 64) or 64),
        'num_heads': int(args.num_heads or cfg.get('num_heads', 4) or 4),
        'train_respondents': int(args.train_respondents or cfg.get('train_respondents', 300) or 300),
        'train_epochs': int(args.train_epochs or cfg.get('train_epochs', 200) or 200),
        'train_lr': float(args.train_lr if args.train_lr is not None else cfg.get('train_lr', 0.03) or 0.03),
        'batch_size': int(args.batch_size or cfg.get('batch_size', 128) or 128),
        'gamma': float(args.gamma if args.gamma is not None else cfg.get('gamma', 0.99) or 0.99),
        'gae_lambda': float(args.gae_lambda if args.gae_lambda is not None else cfg.get('gae_lambda', 0.95) or 0.95),
        'clip_eps': float(args.clip_eps if args.clip_eps is not None else cfg.get('clip_eps', 0.2) or 0.2),
        'value_coef': float(args.value_coef if args.value_coef is not None else cfg.get('value_coef', 0.5) or 0.5),
        'entropy_coef': float(args.entropy_coef if args.entropy_coef is not None else cfg.get('entropy_coef', 0.01) or 0.01),
        'target_kl': float(args.target_kl if args.target_kl is not None else cfg.get('target_kl', 0.03) or 0.03),
        'online_lr': float(args.online_lr if args.online_lr is not None else cfg.get('online_lr', 0.005) or 0.005),
        'online_epochs': int(args.online_epochs or cfg.get('online_epochs', 2) or 2),
        'online_batch_size': int(args.online_batch_size or cfg.get('online_batch_size', cfg.get('batch_size', 128)) or cfg.get('batch_size', 128) or 128),
    }
    payload['design_options'] = {'selfattention': self_opts}
    return payload


def _collect_rows(payload: dict, tasks: list[dict], config: dict, n_rows: int | None = None) -> tuple[list[dict], dict]:
    beta_defaults = (payload or {}).get('beta_defaults', {}) or {}
    pop_stats = _load_pop_stats(payload, data_dir=DATA_DIR, config=config if isinstance(config, dict) else {})
    attr_categories = _collect_attr_categories(pop_stats)
    attr_dim_names = _attr_dim_names(pop_stats)
    self_opts = ((payload or {}).get('design_options', {}) or {}).get('selfattention', {}) if isinstance(((payload or {}).get('design_options', {}) or {}).get('selfattention', {}), dict) else {}
    cfg = ((config or {}).get('self_attention', {}) if isinstance((config or {}).get('self_attention', {}), dict) else {})
    dyn_cfg = ((config or {}).get('dynamic_ppo', {}) if isinstance((config or {}).get('dynamic_ppo', {}), dict) else {})
    seed = int((cfg.get('seed', dyn_cfg.get('seed', 42))) or dyn_cfg.get('seed', 42) or 42)
    total_rows = int(n_rows or self_opts.get('train_respondents', cfg.get('train_respondents', 300)) or cfg.get('train_respondents', 300) or 300)
    respondents = _sample_respondents(pop_stats, total_rows, seed)
    rows = []
    for respondent in respondents:
        rid = str((respondent or {}).get('respondent_id', '')).strip()
        choices = _simulate_choices_for_row(respondent, tasks, beta_defaults)
        row = {
            'respondent_id': rid,
            'design_save_name': DEFAULT_DESIGN_SAVE_NAME,
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
        'n_rows_source': 'selfattention.train_respondents',
    }
    return rows, meta


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Debug flow for selfattention using gz_pt_share_variable_universe_v1 config.')
    parser.add_argument('--design-save-name', default=DEFAULT_DESIGN_SAVE_NAME)
    parser.add_argument('--train-respondents', type=int, default=None)
    parser.add_argument('--train-epochs', type=int, default=None)
    parser.add_argument('--tasks-per-round', type=int, default=None)
    parser.add_argument('--window-length', type=int, default=None)
    parser.add_argument('--hidden-dim', type=int, default=None)
    parser.add_argument('--num-heads', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--online-batch-size', type=int, default=None)
    parser.add_argument('--online-epochs', type=int, default=None)
    parser.add_argument('--train-lr', type=float, default=None)
    parser.add_argument('--online-lr', type=float, default=None)
    parser.add_argument('--explore-epsilon', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--gae-lambda', type=float, default=None)
    parser.add_argument('--clip-eps', type=float, default=None)
    parser.add_argument('--value-coef', type=float, default=None)
    parser.add_argument('--entropy-coef', type=float, default=None)
    parser.add_argument('--target-kl', type=float, default=None)
    parser.add_argument('--timer-interval', type=float, default=30.0, help='周期性输出已耗时，单位秒，默认30秒')
    parser.add_argument('--quiet', action='store_true', help='关闭训练过程打印')
    return parser


def run(args: argparse.Namespace) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    verbose = not bool(args.quiet)
    overall_start = time.perf_counter()
    ticker = RuntimeTicker(interval_seconds=args.timer_interval, enabled=True)
    ticker.start()

    try:
        if verbose:
            print(
                f"[selfattention/debug] run_dir={RUN_DIR} started_at={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                flush=True,
            )

        if not TORCH_AVAILABLE:
            raise SystemExit(
                'PyTorch未安装，无法执行 selfattention 测试。\n'
                '请先安装 torch，例如：\n'
                'python3 -m venv .venv\n'
                'source .venv/bin/activate\n'
                'pip install torch'
            )

        design_save_name = str(args.design_save_name or DEFAULT_DESIGN_SAVE_NAME).strip()
        design_file = DATA_DIR / 'sp_design' / f'{design_save_name}.json'
        rows_jsonl = RUN_DIR / 'xjh_limit1_selfattention_rows.jsonl'
        trace_json = RUN_DIR / 'xjh_limit1_selfattention_trace.json'
        summary_json = RUN_DIR / 'xjh_limit1_selfattention_summary.json'
        candidate_pool_json = RUN_DIR / 'xjh_limit1_selfattention_candidate_pool.json'

        ticker.set_stage('loading design and config')
        design_rec = load_json(design_file, {})
        config = load_json(CONFIG_FILE, {})
        raw_payload = (design_rec or {}).get('payload', {}) if isinstance((design_rec or {}).get('payload', {}), dict) else {}
        preview_tasks = (design_rec or {}).get('preview_tasks', []) if isinstance((design_rec or {}).get('preview_tasks', []), list) else []
        if not preview_tasks:
            raise RuntimeError('design preview_tasks is empty, cannot build selfattention test rows')
        if verbose:
            print(
                f"[selfattention/debug] loaded design: save_name={design_save_name} preview_tasks={len(preview_tasks)} elapsed={_format_elapsed(ticker.elapsed_seconds())}",
                flush=True,
            )

        payload = _normalize_payload_for_selfattention(raw_payload, config if isinstance(config, dict) else {}, args)
        if verbose:
            self_opts = (((payload.get('design_options', {}) or {}).get('selfattention', {}) or {})
                         if isinstance((payload.get('design_options', {}) or {}).get('selfattention', {}), dict) else {})
            print(
                "[selfattention/debug] normalized payload: "
                f"tasks_per_round={int(self_opts.get('tasks_per_round', 0) or 0)} "
                f"train_epochs={int(self_opts.get('train_epochs', 0) or 0)} "
                f"train_respondents={int(self_opts.get('train_respondents', 0) or 0)} "
                f"batch_size={int(self_opts.get('batch_size', 0) or 0)} "
                f"window_length={int(self_opts.get('window_length', 0) or 0)} "
                f"timer_interval={float(args.timer_interval or 30.0):.1f}s",
                flush=True,
            )

        ticker.set_stage('generating synthetic warmup rows')
        stage_start = time.perf_counter()
        if verbose:
            print("[selfattention/debug] generating synthetic rows for warmup...", flush=True)
        rows, row_meta = _collect_rows(payload, preview_tasks, config if isinstance(config, dict) else {}, n_rows=args.train_respondents)
        _write_jsonl(rows_jsonl, rows)
        rows_for_train = load_jsonl(rows_jsonl)
        if verbose:
            print(
                "[selfattention/debug] rows ready: "
                f"generated={len(rows)} loaded={len(rows_for_train)} attr_dims={len(row_meta.get('attr_dim_names', []) or [])} "
                f"stage_elapsed={_format_elapsed(time.perf_counter() - stage_start)} total_elapsed={_format_elapsed(ticker.elapsed_seconds())}",
                flush=True,
            )

        tasks_per_round = int((((payload.get('design_options', {}) or {}).get('selfattention', {}) or {}).get('tasks_per_round', 6)) or 6)
        candidate_pool_target = max(120, tasks_per_round * 20)
        ticker.set_stage('building candidate pool')
        stage_start = time.perf_counter()
        if verbose:
            print(f"[selfattention/debug] building candidate pool: target={candidate_pool_target}", flush=True)
        candidate_pool = _build_candidate_pool_for_ppo(payload, pool_size=candidate_pool_target)

        expert_result = {
            'comb': deepcopy(preview_tasks),
            'd_error': deepcopy((design_rec or {}).get('d_error', {'value': None})) if isinstance((design_rec or {}).get('d_error', {}), dict) else {'value': None},
        }
        expert_sigs = {_task_sig(t) for t in preview_tasks if isinstance(t, dict)}
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
        candidate_pool_json.write_text(
            json.dumps(
                {
                    'design_save_name': design_save_name,
                    'candidate_pool_size': len(candidate_pool),
                    'expert_preview_size': len(preview_tasks),
                    'overlap_with_expert': overlap_count,
                    'candidate_pool': candidate_records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding='utf-8',
        )
        if verbose:
            print(
                "[selfattention/debug] candidate pool ready: "
                f"size={len(candidate_pool)} overlap_with_expert={overlap_count} "
                f"stage_elapsed={_format_elapsed(time.perf_counter() - stage_start)} total_elapsed={_format_elapsed(ticker.elapsed_seconds())}",
                flush=True,
            )

        policy_state: dict = {}
        ticker.set_stage('offline training')
        stage_start = time.perf_counter()
        if verbose:
            print("[selfattention/debug] start offline training...", flush=True)
        trained = train_self_attention_ppo(
            payload=payload,
            policy_state=policy_state,
            candidate_pool=deepcopy(candidate_pool),
            expert_result=expert_result,
            rows=rows_for_train,
            current_respondent=((rows_for_train[0] or {}).get('respondent', {}) if rows_for_train else {}),
            data_dir=DATA_DIR,
            config=config if isinstance(config, dict) else {},
            verbose=verbose,
        )
        trained_policy_state = trained.get('policy_state', policy_state) if isinstance(trained, dict) else policy_state
        if verbose:
            model_state = (trained.get('model_state', {}) if isinstance(trained, dict) else {}) or {}
            print(
                "[selfattention/debug] offline training done: "
                f"train_samples={model_state.get('train_samples')} "
                f"rollout_steps={model_state.get('rollout_steps')} "
                f"policy_version={model_state.get('policy_version')} "
                f"stage_elapsed={_format_elapsed(time.perf_counter() - stage_start)} total_elapsed={_format_elapsed(ticker.elapsed_seconds())}",
                flush=True,
            )

        ticker.set_stage('online updates')
        stage_start = time.perf_counter()
        online_updates = []
        for i, row in enumerate(rows, start=1):
            if verbose:
                print(
                    f"[selfattention/debug] online update {i}/{len(rows)}: respondent_id={row.get('respondent_id')} "
                    f"n_tasks={len(row.get('tasks') or [])} total_elapsed={_format_elapsed(ticker.elapsed_seconds())}",
                    flush=True,
                )
            out = online_update_self_attention_ppo(
                payload=payload,
                policy_state=trained_policy_state,
                tasks=(row.get('tasks') or []),
                choices=(row.get('choices') or {}),
                respondent=(row.get('respondent') or {}),
                config=config if isinstance(config, dict) else {},
                verbose=False,
            )
            if verbose:
                if out.get('updated'):
                    print(
                        f"[selfattention/debug] online update done {i}/{len(rows)}: "
                        f"policy_version={out.get('policy_version')} "
                        f"loss={out.get('loss')} reward={out.get('mean_episode_reward')} "
                        f"total_elapsed={_format_elapsed(ticker.elapsed_seconds())}",
                        flush=True,
                    )
                else:
                    print(
                        f"[selfattention/debug] online update skipped {i}/{len(rows)}: "
                        f"reason={out.get('reason', 'unknown')} "
                        f"total_elapsed={_format_elapsed(ticker.elapsed_seconds())}",
                        flush=True,
                    )
            online_updates.append(
                {
                    'step': i,
                    'respondent_id': row.get('respondent_id'),
                    'n_choices': len((row.get('choices') or {}).keys()),
                    'update': out,
                }
            )
        online_elapsed = time.perf_counter() - stage_start

        total_elapsed = time.perf_counter() - overall_start
        trace = {
            'source_design_save_name': design_save_name,
            'effective_design_type': 'selfattention',
            'design_file': str(design_file),
            'rows_jsonl': str(rows_jsonl),
            'n_rows': len(rows),
            'rows_loaded_for_train': len(rows_for_train),
            'candidate_pool_json': str(candidate_pool_json),
            'candidate_pool_size': len(candidate_pool),
            'candidate_pool_target': candidate_pool_target,
            'expert_preview_size': len(preview_tasks),
            'candidate_expert_overlap': overlap_count,
            'row_meta': row_meta,
            'payload_used': payload,
            'train_output': trained,
            'online_updates': online_updates,
            'policy_state_after': trained_policy_state,
            'timing': {
                'total_elapsed_seconds': round(total_elapsed, 3),
                'total_elapsed_hms': _format_elapsed(total_elapsed),
                'online_updates_elapsed_seconds': round(online_elapsed, 3),
                'online_updates_elapsed_hms': _format_elapsed(online_elapsed),
                'timer_interval_seconds': float(args.timer_interval or 30.0),
            },
        }

        summary_json.write_text(
            json.dumps(
                {
                    'source_design_save_name': design_save_name,
                    'effective_design_type': 'selfattention',
                    'generated_rows': len(rows),
                    'rows_loaded_for_train': len(rows_for_train),
                    'candidate_pool_size': len(candidate_pool),
                    'candidate_pool_target': candidate_pool_target,
                    'expert_preview_size': len(preview_tasks),
                    'candidate_expert_overlap': overlap_count,
                    'train_model_state': trained.get('model_state', {}) if isinstance(trained, dict) else {},
                    'train_iteration_log': trained.get('iteration_log', []) if isinstance(trained, dict) else [],
                    'online_update_steps': len(online_updates),
                    'final_policy_version': int((trained_policy_state or {}).get('response_count', 0) or 0),
                    'row_meta': row_meta,
                    'timing': {
                        'total_elapsed_seconds': round(total_elapsed, 3),
                        'total_elapsed_hms': _format_elapsed(total_elapsed),
                        'online_updates_elapsed_seconds': round(online_elapsed, 3),
                        'online_updates_elapsed_hms': _format_elapsed(online_elapsed),
                        'timer_interval_seconds': float(args.timer_interval or 30.0),
                    },
                    'outputs': {
                        'rows_jsonl': str(rows_jsonl),
                        'trace_json': str(trace_json),
                        'summary_json': str(summary_json),
                        'candidate_pool_json': str(candidate_pool_json),
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding='utf-8',
        )
        trace_json.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding='utf-8')

        print('DONE')
        print(f'  design_save_name: {design_save_name}')
        print(f'  effective_design_type: selfattention')
        print(f'  rows_jsonl: {rows_jsonl}')
        print(f'  candidate_pool_json: {candidate_pool_json}')
        print(f'  trace_json: {trace_json}')
        print(f'  summary_json: {summary_json}')
        print(f'  generated_rows: {len(rows)}')
        print(f'  total_elapsed_seconds: {total_elapsed:.3f}')
        print(f'  total_elapsed_hms: {_format_elapsed(total_elapsed)}')
        print('[finish-reminder] selfattention debug flow finished.')
        print('\a', end='', flush=True)
    finally:
        ticker.stop()


if __name__ == '__main__':
    parser = build_arg_parser()
    run(parser.parse_args())
