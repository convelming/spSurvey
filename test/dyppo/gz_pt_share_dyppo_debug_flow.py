#!/usr/bin/env python3
"""诊断广州公交客流场景 DyPPO 生成方案耗时。

本脚本专门用于排查 `gz_pt_share_variable_universe_v1.json` 在 SP Design
页面选择 DyPPO 后“长时间运行”的问题。它会分别测试两类 payload：

1. saved: 直接使用 JSON 文件里保存的 payload，保留 train_respondents /
   train_epochs / target_kl 等训练参数。
2. frontend_old: 模拟旧版 sp_design.html 的 buildPayload 行为，只提交页面可见的
   DyPPO 参数，用于复现前端丢失训练参数并回退到 data/config.json。
3. frontend / frontend_fixed: 模拟修复后的前端行为，保留已加载 JSON 中的
   隐藏训练参数，同时用页面可见字段覆盖 tasks_per_round 等交互项。

运行示例：

    python3 test/dyppo/gz_pt_share_dyppo_debug_flow.py --mode both
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app import (  # noqa: E402
    _bayesian_d_error_generic,
    _build_candidate_pool_for_ppo,
    _build_tasks_from_spec,
    _compute_efficient,
    _full_combo_count,
    _load_runtime_config,
    _task_satisfies_conditions,
    validate_sp_design_payload,
)
from engine.dynamicPPO import TORCH_AVAILABLE, train_dynamic_ppo  # noqa: E402


CONFIG_PATH = PROJECT_DIR / "data" / "sp_design" / "gz_pt_share_variable_universe_v1.json"
RUNS_DIR = PROJECT_DIR / "test" / "dyppo" / "runs"


class StageTimer:
    """记录每个诊断阶段的耗时。"""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    @contextlib.contextmanager
    def stage(self, name: str):
        started = time.perf_counter()
        print(f"\n[stage] {name} ...", flush=True)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self.items.append({"name": name, "seconds": round(elapsed, 3)})
            print(f"[stage] {name} done in {elapsed:.3f}s", flush=True)


class TimeoutGuard:
    """macOS/Linux 下用 SIGALRM 给单个模式设置硬超时。"""

    def __init__(self, seconds: int) -> None:
        self.seconds = int(seconds or 0)
        self._old_handler = None

    def __enter__(self):
        if self.seconds > 0 and hasattr(signal, "SIGALRM"):
            self._old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, self._handle_timeout)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.seconds > 0 and hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old_handler)
        return False

    def _handle_timeout(self, _signum, _frame):
        raise TimeoutError(f"diagnostic timeout after {self.seconds}s")


def _json_default(obj: Any) -> Any:
    """让 numpy / Path 等对象可以写入 JSON。"""

    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def _load_payload(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    payload = obj.get("payload", obj) if isinstance(obj, dict) else {}
    if not isinstance(payload, dict):
        raise ValueError(f"invalid payload in {path}")
    return payload


def _simulate_frontend_payload(payload: dict[str, Any], *, preserve_hidden: bool) -> dict[str, Any]:
    """模拟 sp_design.html buildPayload 对 DyPPO 参数的提交方式。

    preserve_hidden=False 时复现旧逻辑，只保留页面可见参数；
    preserve_hidden=True 时复现修复后的逻辑，先继承已加载配置中的隐藏参数，
    再用页面可见参数覆盖对应字段。
    """

    out = copy.deepcopy(payload)
    mopt = (((payload.get("design_options") or {}).get("dyppo") or {}) if isinstance(payload.get("design_options"), dict) else {})
    sample_size = int(out.get("sample_size", 600) or 600)
    tpr = int(mopt.get("tasks_per_round", (out.get("tasks_per_person_candidates") or [6])[0]) or 6)
    out["design_type"] = "dyppo"
    out["tasks_per_person_candidates"] = [max(1, tpr)]
    out["min_block_sample"] = 1
    out["target_block_sample"] = sample_size
    dyppo_opts = copy.deepcopy(mopt) if preserve_hidden else {}
    out["design_options"] = {
        "dyppo": {
            **dyppo_opts,
            "tasks_per_round": max(1, tpr),
            "batch_size": int(mopt.get("batch_size", 16) or 16),
            "explore_epsilon": float(mopt.get("explore_epsilon", 0.2) or 0.2),
            "reward_weight_d_error": float(mopt.get("reward_weight_d_error", 1.0) or 1.0),
        }
    }
    return out


def _condition_sample_stats(payload: dict[str, Any], sample_rows: int) -> dict[str, Any]:
    spec = payload.get("design_spec", {}) if isinstance(payload, dict) else {}
    conditions = spec.get("conditions", []) if isinstance(spec, dict) else []
    tasks = _build_tasks_from_spec(spec, sample_rows, seed=20260506)
    passed = sum(1 for task in tasks if _task_satisfies_conditions(task, conditions))
    return {
        "sample_rows": int(sample_rows),
        "passed": int(passed),
        "pass_rate": round(passed / max(1, len(tasks)), 4),
        "conditions_count": len(conditions or []),
    }


def _summarize_task(task: dict[str, Any]) -> dict[str, Any]:
    alts = {}
    for alt_name, vals in ((task or {}).get("alternatives") or {}).items():
        alts[str(alt_name)] = dict(vals or {})
    return {
        "id": task.get("id"),
        "block": task.get("block"),
        "row_in_block": task.get("row_in_block"),
        "alternatives": alts,
    }


def run_one(mode: str, base_payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if mode in {"frontend", "frontend_fixed"}:
        payload = _simulate_frontend_payload(base_payload, preserve_hidden=True)
    elif mode == "frontend_old":
        payload = _simulate_frontend_payload(base_payload, preserve_hidden=False)
    else:
        payload = copy.deepcopy(base_payload)
    timer = StageTimer()
    summary: dict[str, Any] = {
        "mode": mode,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(CONFIG_PATH),
        "torch_available": bool(TORCH_AVAILABLE),
        "status": "running",
    }

    started_all = time.perf_counter()
    try:
        with TimeoutGuard(args.max_seconds):
            with timer.stage("validate payload"):
                ok, msg = validate_sp_design_payload(payload)
                if not ok:
                    raise ValueError(msg)
                spec = payload.get("design_spec", {}) or {}
                summary["design_type"] = payload.get("design_type")
                summary["save_name"] = payload.get("save_name")
                summary["alternatives"] = [a.get("name") for a in spec.get("alternatives", [])]
                summary["variable_counts"] = {
                    a.get("name"): len(a.get("variables", []) or []) for a in spec.get("alternatives", [])
                }
                summary["dyppo_options"] = copy.deepcopy(((payload.get("design_options") or {}).get("dyppo") or {}))
                summary["target_block_sample"] = payload.get("target_block_sample")
                summary["full_combo_count"] = _full_combo_count(spec)

            with timer.stage("sample condition pass-rate"):
                summary["condition_sample"] = _condition_sample_stats(payload, args.condition_sample)

            with timer.stage("build feasible candidate pool"):
                tpr = int((((payload.get("design_options") or {}).get("dyppo") or {}).get("tasks_per_round", 6)) or 6)
                pool_size = int(args.candidate_pool or max(120, tpr * 20))
                candidate_pool = _build_candidate_pool_for_ppo(payload, pool_size=pool_size)
                summary["candidate_pool_target"] = pool_size
                summary["candidate_pool_size"] = len(candidate_pool)
                if not candidate_pool:
                    raise RuntimeError("candidate pool is empty after conditions/dominance filtering")

            with timer.stage("compute efficient teacher prior"):
                payload_eff = copy.deepcopy(payload)
                payload_eff["design_type"] = "efficient"
                payload_eff["design_options"] = {
                    "efficient": {
                        "tasks_per_person": max(1, tpr),
                        "row_exchange_iterations": int(args.teacher_iterations),
                    }
                }
                efficient_result = _compute_efficient(payload_eff)
                summary["teacher_rows"] = len(efficient_result.get("comb", []) or [])
                summary["teacher_d_error"] = ((efficient_result.get("d_error", {}) or {}).get("value"))

            if args.skip_train:
                trained = {
                    "comb": [],
                    "iteration_log": [],
                    "model_state": {"trained": False, "skipped": True},
                    "policy_state": {},
                }
            else:
                if not TORCH_AVAILABLE:
                    raise RuntimeError("torch is not available; install torch before testing dyppo training")
                with timer.stage("train dynamicPPO"):
                    trained = train_dynamic_ppo(
                        payload=payload,
                        policy_state={},
                        candidate_pool=candidate_pool,
                        expert_result=efficient_result,
                        current_respondent=None,
                        data_dir=PROJECT_DIR / "data",
                        config=_load_runtime_config(),
                    )

            tasks = trained.get("comb", []) if isinstance(trained, dict) else []
            with timer.stage("evaluate generated block d-error"):
                final_d = None
                if tasks:
                    final_d = _bayesian_d_error_generic(
                        tasks,
                        payload.get("design_spec", {}) or {},
                        payload.get("beta_defaults", {}) or {},
                        payload.get("beta_bounds", {}) or {},
                        beta_draws=24,
                        seed=57,
                    )
                summary["generated_rows"] = len(tasks)
                summary["generated_d_error"] = round(float(final_d), 6) if final_d is not None else None
                summary["generated_preview"] = [_summarize_task(t) for t in tasks[: min(3, len(tasks))]]

            logs = trained.get("iteration_log", []) if isinstance(trained, dict) else []
            model_state = trained.get("model_state", {}) if isinstance(trained, dict) else {}
            summary["iteration_log_count"] = len(logs)
            summary["iteration_log_head"] = logs[:5]
            summary["iteration_log_tail"] = logs[-5:]
            summary["model_state"] = model_state
            summary["status"] = "ok"
    except Exception as exc:
        summary["status"] = "error"
        summary["error"] = str(exc)
        summary["traceback"] = traceback.format_exc()
        print(f"[error] {mode}: {exc}", file=sys.stderr, flush=True)
    finally:
        summary["stage_timings"] = timer.items
        summary["elapsed_seconds"] = round(time.perf_counter() - started_all, 3)
        summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    return summary


def main() -> int:
    global CONFIG_PATH
    parser = argparse.ArgumentParser(description="Test gz PT-share DyPPO design generation.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="SP design JSON path.")
    parser.add_argument("--mode", choices=["saved", "frontend", "frontend_fixed", "frontend_old", "both", "all"], default="both")
    parser.add_argument("--max-seconds", type=int, default=180, help="Timeout per mode; 0 disables timeout.")
    parser.add_argument("--candidate-pool", type=int, default=120)
    parser.add_argument("--condition-sample", type=int, default=960)
    parser.add_argument("--teacher-iterations", type=int, default=30)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    CONFIG_PATH = args.config.resolve()
    run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M")
    run_dir.mkdir(parents=True, exist_ok=True)

    base_payload = _load_payload(CONFIG_PATH)
    if args.mode == "both":
        modes = ["saved", "frontend"]
    elif args.mode == "all":
        modes = ["saved", "frontend_old", "frontend"]
    else:
        modes = [args.mode]
    all_summaries = []
    overall_started = time.perf_counter()
    for mode in modes:
        print(f"\n========== DyPPO diagnostic mode: {mode} ==========", flush=True)
        summary = run_one(mode, base_payload, args)
        all_summaries.append(summary)
        out_path = run_dir / f"gz_pt_share_dyppo_{mode}_summary.json"
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        print(f"[write] {out_path}", flush=True)

    combined = {
        "config_path": str(CONFIG_PATH),
        "run_dir": str(run_dir),
        "elapsed_seconds": round(time.perf_counter() - overall_started, 3),
        "summaries": all_summaries,
    }
    combined_path = run_dir / "gz_pt_share_dyppo_compare_summary.json"
    combined_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(f"\n[write] {combined_path}", flush=True)
    print("\n========== Summary ==========", flush=True)
    for s in all_summaries:
        ms = s.get("model_state", {}) if isinstance(s.get("model_state"), dict) else {}
        print(
            f"{s.get('mode')}: status={s.get('status')} elapsed={s.get('elapsed_seconds')}s "
            f"pool={s.get('candidate_pool_size')} generated={s.get('generated_rows')} "
            f"episodes={ms.get('rollout_episodes')} epochs={ms.get('train_epochs')} "
            f"target_kl={ms.get('target_kl')} d_error={s.get('generated_d_error')}",
            flush=True,
        )
    return 0 if all(s.get("status") == "ok" for s in all_summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
