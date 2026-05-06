from __future__ import annotations

import argparse
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
    _bayesian_d_error_generic,
    _build_candidate_pool_for_ppo,
    _ensure_spec_policy,
    _estimate_mnl,
    _load_design_policy_state,
    _obs_to_design_rows,
    _param_keys_for_spec,
    _payload_save_name,
    _spec_id_from_payload,
)
from engine.dynamicPPO import (
    TORCH_AVAILABLE,
    _blend_with_expert_prior,
    _build_feature_spec,
    _load_pop_stats,
    _safe_float,
    _sample_respondents,
    _score_candidates,
    _task_signature,
    _task_state_vector,
    _utility_for_alt,
    online_update_dynamic_ppo,
    train_dynamic_ppo,
)
from engine.storage import load_json

DATA_DIR = PROJECT_DIR / "data"
TEST_ROOT = PROJECT_DIR / "test"
ALGO_TEST_DIR = TEST_ROOT / "dyppo"
RUNS_DIR = ALGO_TEST_DIR / "runs"
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M")
RUN_DIR = RUNS_DIR / RUN_STAMP
CONFIG_FILE = DATA_DIR / "config.json"
DEFAULT_DESIGN_SAVE_NAME = "gz_pt_share_variable_universe_v1"
SESSION_ROWS_JSONL = RUN_DIR / "dyppo_live_session_rows.jsonl"


def _deterministic_rng(*parts: str) -> np.random.Generator:
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:16]
    seed = int(digest, 16) % (2**32 - 1)
    return np.random.default_rng(seed)


def _simulate_choices(
    respondent: dict,
    tasks: list[dict],
    beta_defaults: dict,
    *,
    choice_mode: str = "random",
) -> tuple[dict, list[dict]]:
    """为当前 respondent 模拟每道题的选择。

    `choice_mode` 支持：
    - `random`: 完全随机，从所有选项里等概率抽取
    - `argmax`: 选择 utility 最大的选项；并列时随机打破
    - `softmax`: 按 `exp(U)` 概率抽样
    """
    respondent_id = str((respondent or {}).get("respondent_id", ""))
    choices: dict[str, str] = {}
    details: list[dict] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("id", "")).strip()
        alts = (task or {}).get("alternatives", {}) if isinstance((task or {}).get("alternatives", {}), dict) else {}
        if not task_id or not alts:
            continue
        scored = []
        for alt_name, attrs in alts.items():
            util = _utility_for_alt(str(alt_name), attrs or {}, respondent, beta_defaults)
            scored.append((str(alt_name), float(util)))
        rng = _deterministic_rng(respondent_id, task_id)
        alt_names = [a for a, _u in scored]
        util_vec = np.array([u for _a, u in scored], dtype=float)
        if str(choice_mode).strip().lower() == "argmax":
            best_u = max(u for _, u in scored)
            best_alts = [a for a, u in scored if abs(u - best_u) <= 1e-12]
            chosen = best_alts[int(rng.integers(0, len(best_alts)))]
        elif str(choice_mode).strip().lower() == "softmax":
            z = util_vec - np.max(util_vec)
            p = np.exp(z)
            p = p / np.sum(p)
            chosen = alt_names[int(rng.choice(len(alt_names), p=p))]
        else:
            chosen = alt_names[int(rng.integers(0, len(alt_names)))]
        choices[task_id] = chosen
        details.append(
            {
                "task_id": task_id,
                "chosen": chosen,
                "choice_mode": str(choice_mode),
                "utilities": {name: round(val, 6) for name, val in scored},
            }
        )
    return choices, details


def _state_l2_diff(old_state: dict | None, new_state: dict | None) -> float:
    """计算两组 JSON 形式权重的 L2 差值，便于展示“权重是否更新”。"""
    if not isinstance(old_state, dict) or not isinstance(new_state, dict):
        return 0.0
    total = 0.0
    hit = False
    for key, old_val in old_state.items():
        if key not in new_state:
            continue
        try:
            arr_old = np.asarray(old_val, dtype=float)
            arr_new = np.asarray(new_state[key], dtype=float)
            if arr_old.shape != arr_new.shape:
                continue
            total += float(np.sum((arr_new - arr_old) ** 2))
            hit = True
        except Exception:
            continue
    return float(np.sqrt(total)) if hit else 0.0


def _format_respondent(respondent: dict, dim_names: list[str]) -> str:
    parts = list((respondent or {}).get("attr_segments", []) or [])
    labels = []
    for idx, val in enumerate(parts):
        label = dim_names[idx] if idx < len(dim_names) else f"attr_{idx}"
        labels.append(f"{label}={val}")
    zone_id = str((respondent or {}).get("zone_id", "") or "")
    if zone_id:
        labels.append(f"zone_id={zone_id}")
    return ", ".join(labels)


def _collect_existing_obs_rows(design_save_name: str, spec: dict) -> tuple[list[dict], dict]:
    """从 respondents 主记录里收集指定 design 的历史 SP 观测。"""
    keys, alt_names, _base = _param_keys_for_spec(spec)
    obs_rows: list[dict] = []
    summary = {"n_submissions": 0, "n_observations": 0, "choice_counts": {}}
    for rec_file in (DATA_DIR / "respondents").glob("*.json"):
        rec = load_json(rec_file, {})
        sp = rec.get("sp", {}) if isinstance(rec.get("sp", {}), dict) else {}
        submissions = sp.get("submissions", []) if isinstance(sp.get("submissions", []), list) else []
        for sub in submissions:
            if not isinstance(sub, dict):
                continue
            if str(sub.get("design_save_name", "")).strip() != str(design_save_name).strip():
                continue
            summary["n_submissions"] += 1
            tasks_map = {
                str((t or {}).get("id")): t
                for t in (sub.get("tasks", []) or [])
                if isinstance(t, dict) and str((t or {}).get("id", "")).strip()
            }
            for task_id, chosen in (sub.get("choices", {}) or {}).items():
                if chosen not in alt_names:
                    continue
                task = tasks_map.get(str(task_id))
                if not isinstance(task, dict):
                    continue
                x = _obs_to_design_rows(task, alt_names, keys)
                y = alt_names.index(chosen)
                obs_rows.append({"x": x, "y": y})
                summary["n_observations"] += 1
                summary["choice_counts"][str(chosen)] = int(summary["choice_counts"].get(str(chosen), 0) or 0) + 1
    return obs_rows, summary


def _estimate_beta_dict_from_obs(obs_rows: list[dict], spec: dict, beta_defaults: dict, beta_bounds: dict) -> tuple[dict, dict]:
    """根据当前观测重估 MNL 参数，返回 beta 字典与估计摘要。"""
    keys, alt_names, _base = _param_keys_for_spec(spec)
    beta0 = np.array([float(beta_defaults.get(k, 0.0)) for k in keys], dtype=float)
    if obs_rows:
        beta_hat = _estimate_mnl(obs_rows, keys, alt_names, beta0, beta_bounds=beta_bounds, epochs=250, lr=0.08)
    else:
        beta_hat = beta0.copy()
    beta_dict = dict(beta_defaults or {})
    for i, key in enumerate(keys):
        beta_dict[key] = float(beta_hat[i])
    summary = {
        "param_keys": keys,
        "initial_beta": {k: round(float(beta0[i]), 6) for i, k in enumerate(keys)},
        "estimated_beta": {k: round(float(beta_hat[i]), 6) for i, k in enumerate(keys)},
        "delta_beta": {k: round(float(beta_hat[i] - beta0[i]), 6) for i, k in enumerate(keys)},
        "n_observations": int(len(obs_rows)),
    }
    return beta_dict, summary


def _print_tasks(tasks: list[dict]) -> None:
    for idx, task in enumerate(tasks, start=1):
        print(f"  题目 {idx} | id={task.get('id')}")
        alternatives = (task or {}).get("alternatives", {}) if isinstance((task or {}).get("alternatives", {}), dict) else {}
        for alt_name, attrs in alternatives.items():
            attr_txt = ", ".join([f"{k}={attrs[k]}" for k in sorted((attrs or {}).keys())])
            print(f"    - {alt_name}: {attr_txt}")


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _issue_tasks_once(
    *,
    payload: dict,
    policy_state: dict,
    candidate_pool: list[dict],
    expert_result: dict | None,
    current_respondent: dict,
    config: dict,
) -> tuple[list[dict], dict]:
    """使用当前权重直接生成下一位 respondent 的题组，不做离线重训。"""
    dyn_cfg = config.get("dynamic_ppo", {}) if isinstance(config.get("dynamic_ppo", {}), dict) else {}
    mopt = ((payload.get("design_options", {}) or {}).get("dyppo", {}) or {})
    tpr = int(mopt.get("tasks_per_round", 6) or 6)
    eps = float(mopt.get("explore_epsilon", 0.2) or 0.2)
    seed = int(dyn_cfg.get("seed", 42))

    candidates = [deepcopy(c) for c in (candidate_pool or [])]
    for c in candidates:
        c["sig"] = str(c.get("sig") or _task_signature(c))

    input_dim = int(policy_state.get("input_dim", dyn_cfg.get("input_dim", 6)) or dyn_cfg.get("input_dim", 6) or 6)
    output_dim = int(policy_state.get("output_dim", max(1, len(candidates))) or max(1, len(candidates)))
    actor_state = policy_state.get("actor_state", {}) if isinstance(policy_state.get("actor_state", {}), dict) else {}
    feature_spec = {
        "attr_dim_names": deepcopy(policy_state.get("attr_dim_names", [])),
        "attr_categories": deepcopy(policy_state.get("attr_categories", [])),
        "zone_categories": deepcopy(policy_state.get("zone_categories", [])),
    }

    rng = np.random.default_rng(seed + 99 + int(policy_state.get("response_count", 0)))
    remain_idx = list(range(len(candidates)))
    picked: list[dict] = []
    last_expert_stats = {"expert_overlap": 0, "expert_prior_weight": 0.0}

    for _ in range(min(max(1, tpr), len(remain_idx))):
        state_x = _task_state_vector(
            current_respondent,
            picked,
            beta_defaults=(payload.get("beta_defaults", {}) or {}),
            feature_spec=feature_spec,
            step_index=len(picked),
            episode_len=tpr,
            input_dim=input_dim,
        )
        probs = _score_candidates(
            candidates,
            state_x=state_x,
            actor_state=actor_state,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        probs, last_expert_stats = _blend_with_expert_prior(probs, candidates, expert_result)
        sub_probs = np.array([probs[i] for i in remain_idx], dtype=float)
        if np.sum(sub_probs) <= 0:
            sub_probs = np.ones_like(sub_probs) / len(sub_probs)
        else:
            sub_probs = sub_probs / np.sum(sub_probs)
        if rng.random() < max(0.0, min(1.0, eps)):
            pick_pos = int(rng.integers(0, len(remain_idx)))
        else:
            pick_pos = int(rng.choice(len(remain_idx), p=sub_probs))
        idx = remain_idx.pop(pick_pos)
        picked.append(deepcopy(candidates[idx]))

    final_tasks = []
    for idx, task in enumerate(picked, start=1):
        final_tasks.append(
            {
                **task,
                "block": 1,
                "row_in_block": idx,
                "id": f"live_b1_r{idx}",
            }
        )
    return final_tasks, {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "candidate_pool_size": len(candidates),
        "policy_version": int(policy_state.get("response_count", 0)),
        "expert_overlap": int(last_expert_stats.get("expert_overlap", 0) or 0),
        "expert_prior_weight": float(last_expert_stats.get("expert_prior_weight", 0.0) or 0.0),
    }


def _bootstrap_policy_if_needed(
    *,
    payload: dict,
    policy_state: dict,
    candidate_pool: list[dict],
    expert_result: dict | None,
    config: dict,
    expected_input_dim: int,
    expected_output_dim: int,
    expected_candidate_signatures: list[str],
) -> dict:
    """若当前没有可用权重，则用 synthetic rollout 做一次 warm-start。"""
    has_actor = isinstance(policy_state.get("actor_state", {}), dict) and bool(policy_state.get("actor_state", {}))
    has_critic = isinstance(policy_state.get("critic_state", {}), dict) and bool(policy_state.get("critic_state", {}))
    state_input_dim = int(policy_state.get("input_dim", 0) or 0)
    state_output_dim = int(policy_state.get("output_dim", 0) or 0)
    state_sigs = [str(x) for x in (policy_state.get("candidate_signatures", []) or [])]
    state_compatible = (
        state_input_dim == int(expected_input_dim)
        and state_output_dim == int(expected_output_dim)
        and state_sigs == list(expected_candidate_signatures)
    )
    if has_actor and has_critic and state_compatible:
        return {
            "bootstrapped": False,
            "reason": "loaded_existing_weights",
        }
    if not state_compatible:
        policy_state["actor_state"] = {}
        policy_state["critic_state"] = {}
    out = train_dynamic_ppo(
        payload=payload,
        policy_state=policy_state,
        candidate_pool=deepcopy(candidate_pool),
        expert_result=expert_result,
        rows=None,
        current_respondent=None,
        data_dir=DATA_DIR,
        config=config,
    )
    return {
        "bootstrapped": True,
        "reason": "synthetic_warm_start",
        "model_state": deepcopy(out.get("model_state", {})),
        "iteration_log": deepcopy(out.get("iteration_log", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="交互式 dynamicPPO 实时发题/更新 演示")
    parser.add_argument("--design", default=DEFAULT_DESIGN_SAVE_NAME, help="已保存的 SP design 名称")
    parser.add_argument("--max-respondents", type=int, default=200, help="最多模拟多少位 respondent")
    parser.add_argument(
        "--choice-mode",
        choices=["random", "argmax", "softmax"],
        default="random",
        help="模拟 respondent 选择的方式；当前默认 random",
    )
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        raise SystemExit(
            "PyTorch 未安装，无法运行 interactive_dyppo_live_demo.py。\n"
            "请先安装 torch，例如：\n"
            "python3 -m venv .venv\n"
            "source .venv/bin/activate\n"
            "pip install torch"
        )

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if SESSION_ROWS_JSONL.exists():
        SESSION_ROWS_JSONL.unlink()

    config = load_json(CONFIG_FILE, {})
    design_file = DATA_DIR / "sp_design" / f"{args.design}.json"
    design_rec = load_json(design_file, {})
    if not design_rec:
        raise SystemExit(f"找不到 design: {design_file}")

    payload = deepcopy((design_rec or {}).get("payload", {}) if isinstance((design_rec or {}).get("payload", {}), dict) else {})
    if not payload:
        raise SystemExit("design payload 为空，无法继续。")
    payload["save_name"] = args.design
    payload["design_save_name"] = args.design

    candidate_pool = _build_candidate_pool_for_ppo(payload, pool_size=max(120, int(((payload.get("design_options", {}) or {}).get("dyppo", {}) or {}).get("tasks_per_round", 6) * 20)))
    preview_tasks = deepcopy((design_rec or {}).get("preview_tasks", []) if isinstance((design_rec or {}).get("preview_tasks", []), list) else [])
    expert_result = {"comb": preview_tasks, "d_error": deepcopy((design_rec or {}).get("d_error", {"value": None}))}

    loaded_policy_state, _ = _load_design_policy_state(args.design)
    spec_id = _spec_id_from_payload(payload)
    policy_state = _ensure_spec_policy(loaded_policy_state, spec_id)

    pop_stats = _load_pop_stats(payload, data_dir=DATA_DIR, config=config)
    feature_spec = _build_feature_spec(pop_stats)
    input_dim = max(int(((config.get("dynamic_ppo", {}) or {}).get("input_dim", feature_spec.get("feature_dim", 6)) or feature_spec.get("feature_dim", 6))), int(feature_spec.get("feature_dim", 6) or 6))
    policy_state["input_dim"] = int(policy_state.get("input_dim", input_dim) or input_dim)
    policy_state["output_dim"] = max(1, len(candidate_pool))
    policy_state["attr_dim_names"] = deepcopy(feature_spec.get("attr_dim_names", []))
    policy_state["attr_categories"] = deepcopy(feature_spec.get("attr_categories", []))
    policy_state["zone_categories"] = deepcopy(feature_spec.get("zone_categories", []))
    policy_state["candidate_signatures"] = [str((c or {}).get("sig") or _task_signature(c)) for c in candidate_pool]

    bootstrap_info = _bootstrap_policy_if_needed(
        payload=payload,
        policy_state=policy_state,
        candidate_pool=candidate_pool,
        expert_result=expert_result,
        config=config,
        expected_input_dim=input_dim,
        expected_output_dim=max(1, len(candidate_pool)),
        expected_candidate_signatures=policy_state["candidate_signatures"],
    )

    dyn_cfg = (config.get("dynamic_ppo", {}) if isinstance(config.get("dynamic_ppo", {}), dict) else {})
    base_seed = int(dyn_cfg.get("seed", 42) or 42)
    base_beta_defaults = payload.get("beta_defaults", {}) if isinstance(payload.get("beta_defaults", {}), dict) else {}
    beta_bounds = payload.get("beta_bounds", {}) if isinstance(payload.get("beta_bounds", {}), dict) else {}
    spec = payload.get("design_spec", {}) if isinstance(payload.get("design_spec", {}), dict) else {}
    historical_obs_rows, historical_summary = _collect_existing_obs_rows(args.design, spec)
    active_beta_defaults, beta_est_summary = _estimate_beta_dict_from_obs(historical_obs_rows, spec, base_beta_defaults, beta_bounds)

    d_error_history: list[float] = []
    unique_task_sigs: set[str] = set()
    choice_counts: dict[str, int] = {}
    session_obs_rows = list(historical_obs_rows)

    print("=" * 80)
    print("dynamicPPO 实时采集演示")
    print(f"design = {args.design}")
    print(f"candidate_pool_size = {len(candidate_pool)}")
    print(f"loaded_policy_state = {bool(loaded_policy_state)}")
    print(f"bootstrap = {bootstrap_info.get('reason')}")
    print(f"choice_mode = {args.choice_mode}")
    print(f"historical_submissions_for_design = {historical_summary.get('n_submissions', 0)}")
    print(f"historical_observations_for_design = {historical_summary.get('n_observations', 0)}")
    print(f"initial_beta_source = {'historical_mnl_estimate' if historical_obs_rows else 'payload.beta_defaults'}")
    print(f"initial_estimated_beta = {json.dumps(beta_est_summary.get('estimated_beta', {}), ensure_ascii=False)}")
    if bootstrap_info.get("bootstrapped"):
        print(f"bootstrap_logs = {json.dumps(bootstrap_info.get('iteration_log', [])[:3], ensure_ascii=False)}")
    print("说明：按回车继续；输入 q 后回车退出。")
    print("=" * 80)

    for idx in range(1, max(1, int(args.max_respondents)) + 1):
        raw = input(f"\n[{idx}] 按回车显示下一位 respondent 个人信息，输入 q 退出: ").strip().lower()
        if raw == "q":
            break

        respondent = deepcopy(_sample_respondents(pop_stats, 1, base_seed + idx + int(policy_state.get("response_count", 0)))[0])
        # `_sample_respondents(..., n=1)` 每次都会返回 `sim_0001`。
        # 如果后续随机种子又依赖 respondent_id，就会导致“每位 respondent 都重复同一套伪随机选择”。
        # 这里显式重写为逐位唯一的 demo respondent_id。
        respondent["respondent_id"] = f"demo_{idx:04d}"
        print(f"\nRespondent {idx}")
        print(f"  respondent_id = {respondent.get('respondent_id')}")
        print(f"  profile = {_format_respondent(respondent, feature_spec.get('attr_dim_names', []))}")

        raw = input("按回车生成当前 respondent 的 combo... ").strip().lower()
        if raw == "q":
            break

        tasks, issue_meta = _issue_tasks_once(
            payload={**payload, "beta_defaults": deepcopy(active_beta_defaults)},
            policy_state=policy_state,
            candidate_pool=candidate_pool,
            expert_result=expert_result,
            current_respondent=respondent,
            config=config,
        )
        combo_d_error = _bayesian_d_error_generic(
            tasks,
            spec,
            active_beta_defaults,
            beta_bounds,
            beta_draws=24,
            seed=base_seed + idx,
        ) if tasks else None
        if combo_d_error is not None:
            d_error_history.append(float(combo_d_error))
        print("\n生成的 combo")
        _print_tasks(tasks)
        print(f"  d_error = {round(float(combo_d_error), 6) if combo_d_error is not None else None}")
        print(f"  issue_meta = {json.dumps(issue_meta, ensure_ascii=False)}")

        raw = input("按回车模拟 respondent 的选择并更新权重... ").strip().lower()
        if raw == "q":
            break

        old_actor_state = deepcopy(policy_state.get("actor_state", {}))
        choices, choice_details = _simulate_choices(
            respondent,
            tasks,
            active_beta_defaults,
            choice_mode=args.choice_mode,
        )
        update = online_update_dynamic_ppo(
            payload={**payload, "beta_defaults": deepcopy(active_beta_defaults)},
            policy_state=policy_state,
            tasks=tasks,
            choices=choices,
            respondent=respondent,
        )
        weight_delta = _state_l2_diff(old_actor_state, policy_state.get("actor_state", {}))

        keys, alt_names, _base = _param_keys_for_spec(spec)
        for task in tasks:
            task_id = str((task or {}).get("id", "")).strip()
            chosen = str((choices or {}).get(task_id, "") or "")
            if not task_id or chosen not in alt_names:
                continue
            session_obs_rows.append({"x": _obs_to_design_rows(task, alt_names, keys), "y": alt_names.index(chosen)})
        active_beta_defaults, beta_est_summary = _estimate_beta_dict_from_obs(session_obs_rows, spec, base_beta_defaults, beta_bounds)

        for task in tasks:
            unique_task_sigs.add(str((task or {}).get("sig") or _task_signature(task)))
        for chosen in choices.values():
            choice_counts[str(chosen)] = int(choice_counts.get(str(chosen), 0) or 0) + 1

        row = {
            "respondent_id": respondent.get("respondent_id"),
            "respondent": deepcopy(respondent),
            "tasks": deepcopy(tasks),
            "choices": deepcopy(choices),
            "choice_details": deepcopy(choice_details),
            "combo_d_error": float(combo_d_error) if combo_d_error is not None else None,
            "update": deepcopy(update),
        }
        _append_jsonl(SESSION_ROWS_JSONL, row)

        print("\n模拟选择结果")
        for detail in choice_details:
            print(
                f"  - {detail['task_id']}: chosen={detail['chosen']} | utilities={json.dumps(detail['utilities'], ensure_ascii=False)}"
            )
        print("\n更新后统计")
        print(f"  policy_version = {update.get('policy_version')}")
        print(f"  online_episode_reward = {update.get('online_episode_reward')}")
        print(f"  weight_delta_l2 = {round(weight_delta, 6)}")
        print(f"  collected_respondents = {idx}")
        print(f"  current_respondent_choice_counts = {json.dumps({k: list(choices.values()).count(k) for k in sorted(set(choices.values()))}, ensure_ascii=False)}")
        print(f"  current_respondent_n_choices = {len(choices)}")
        print(f"  unique_task_sigs = {len(unique_task_sigs)} / {len(candidate_pool)}")
        print(f"  avg_d_error = {round(float(np.mean(d_error_history)), 6) if d_error_history else None}")
        print(f"  choice_counts = {json.dumps(choice_counts, ensure_ascii=False)}")
        print(f"  estimated_beta = {json.dumps(beta_est_summary.get('estimated_beta', {}), ensure_ascii=False)}")
        print(f"  session_rows_jsonl = {SESSION_ROWS_JSONL}")

    print("\n演示结束。")
    print(f"本次采集行数据已写入: {SESSION_ROWS_JSONL}")


if __name__ == "__main__":
    main()



