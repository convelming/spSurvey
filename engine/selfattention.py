"""SelfAttention 并行题组生成与在线更新模块。

本文件采用当前项目统一后的实现口径：

1. 一份 SP 问卷中的题目是一次性并行生成的；
2. respondent 也是一次性看到并填写整份 block；
3. decoder 不再使用 shifted-right + 上三角因果遮罩式 attention 的自回归解释；
4. 取而代之的是：
   - encoder 读取 `X_rp / X_env / X_hist / X_cand`
   - encoder 汇总状态同时分出 `respondent_target_head`，给出剩余样本的定向采样建议
   - 并行 question queries 表示 `T_max` 个待生成题位
   - question-slot self-attention 建模整份问卷内部题位之间的关系
   - count_head 决定本次问卷题数
   - slot_select_head 决定哪些候选题位进入最终 block
   - mask_head 决定题内哪些变量激活
   - value_head 在 mask 条件下给激活变量赋值
   - score_head 评估每题质量

为了保持与 `app.py` 的现有接口兼容，本文件仍然暴露：
- `train_self_attention_ppo(...)`
- `online_update_self_attention_ppo(...)`

但这两个函数现在内部不再执行“候选题池级别的 PPO 逐题选题”，
而是执行“并行 block 生成器”的监督 warmup / 在线微调流程。

对外返回结构保持不变：
- `comb`: 生成出的题组
- `model_state`: 调试摘要
- `policy_state`: 可继续在线更新的权重与元信息
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    F = None
    optim = None
    TORCH_AVAILABLE = False

try:
    from . import dynamicPPO as dyppo_shared
except ImportError:
    from engine import dynamicPPO as dyppo_shared

_MISSING_ATTR_TOKEN = "__missing__"


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _progress_print(verbose: bool, prefix: str, message: str) -> None:
    if verbose:
        print(f"[{prefix}] {message}", flush=True)


def _task_signature(task: dict) -> str:
    alts = (task or {}).get("alternatives", {}) if isinstance(task, dict) else {}
    norm = {}
    for alt, attrs in alts.items():
        norm[str(alt)] = {str(k): attrs.get(k) for k in sorted((attrs or {}).keys())}
    return hashlib.sha1(str(sorted(norm.items())).encode("utf-8")).hexdigest()[:16]


def _normalize_heads(hidden_dim: int, num_heads: int) -> int:
    heads = max(1, min(int(num_heads), int(hidden_dim)))
    while heads > 1 and hidden_dim % heads != 0:
        heads -= 1
    return max(1, heads)


if TORCH_AVAILABLE:
    class ParallelQuestionBlockGenerator(nn.Module):
        """并行 question-block 生成器。

        输入:
            encoder_tokens ∈ R^[B, L_enc, D_ctx]

        输出:
            count_logits ∈ R^[B, K_count]
            mask_logits  ∈ R^[B, T_max, V]
            value_raw    ∈ R^[B, T_max, V]
            slot_select_logits ∈ R^[B, T_max]
            score_raw          ∈ R^[B, T_max]
            sample_target_logits ∈ R^[B, C_sample]

        其中：
            - `T_max` 是本次问卷允许的最大题数
            - `V` 是展平后的变量槽位数
            - `K_count = T_max - T_min + 1`
        """

        def __init__(
            self,
            *,
            context_dim: int,
            slot_dim: int,
            hidden_dim: int = 64,
            num_heads: int = 4,
            max_questions: int = 8,
            count_classes: int = 1,
            sample_target_dim: int = 1,
        ) -> None:
            super().__init__()
            self.context_dim = int(context_dim)
            self.slot_dim = int(slot_dim)
            self.hidden_dim = max(16, int(hidden_dim))
            self.num_heads = _normalize_heads(self.hidden_dim, int(num_heads))
            self.max_questions = max(1, int(max_questions))
            self.count_classes = max(1, int(count_classes))
            self.sample_target_dim = max(1, int(sample_target_dim))

            self.context_proj = nn.Linear(self.context_dim, self.hidden_dim)
            self.encoder_attn = nn.MultiheadAttention(self.hidden_dim, self.num_heads, batch_first=True)
            self.encoder_norm1 = nn.LayerNorm(self.hidden_dim)
            self.encoder_ffn = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            )
            self.encoder_norm2 = nn.LayerNorm(self.hidden_dim)

            self.slot_queries = nn.Parameter(torch.randn(self.max_questions, self.hidden_dim) * 0.02)
            self.slot_attn = nn.MultiheadAttention(self.hidden_dim, self.num_heads, batch_first=True)
            self.slot_norm1 = nn.LayerNorm(self.hidden_dim)
            self.cross_attn = nn.MultiheadAttention(self.hidden_dim, self.num_heads, batch_first=True)
            self.slot_norm2 = nn.LayerNorm(self.hidden_dim)
            self.slot_ffn = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            )
            self.slot_norm3 = nn.LayerNorm(self.hidden_dim)

            self.count_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.count_classes),
            )
            self.respondent_target_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.sample_target_dim),
            )
            self.slot_select_head = nn.Linear(self.hidden_dim, 1)
            self.mask_head = nn.Linear(self.hidden_dim, self.slot_dim)
            self.mask_context = nn.Linear(self.slot_dim, self.hidden_dim)
            self.value_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.slot_dim),
            )
            self.score_head = nn.Linear(self.hidden_dim, 1)

        def forward(self, encoder_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
            if encoder_tokens.dim() == 2:
                encoder_tokens = encoder_tokens.unsqueeze(0)
            if encoder_tokens.dim() != 3:
                raise ValueError(f"expected [B,L_enc,D_ctx], got {tuple(encoder_tokens.shape)}")

            enc = self.context_proj(encoder_tokens)
            enc_attn, _ = self.encoder_attn(enc, enc, enc, need_weights=False)
            enc = self.encoder_norm1(enc + enc_attn)
            enc_ffn = self.encoder_ffn(enc)
            enc = self.encoder_norm2(enc + enc_ffn)

            batch = enc.shape[0]
            slot = self.slot_queries.unsqueeze(0).expand(batch, -1, -1)
            slot_attn, _ = self.slot_attn(slot, slot, slot, need_weights=False)
            slot = self.slot_norm1(slot + slot_attn)
            cross, _ = self.cross_attn(slot, enc, enc, need_weights=False)
            slot = self.slot_norm2(slot + cross)
            slot_ffn = self.slot_ffn(slot)
            slot = self.slot_norm3(slot + slot_ffn)

            pooled = slot.mean(dim=1)
            pooled_enc = enc.mean(dim=1)
            count_logits = self.count_head(pooled)
            sample_target_logits = self.respondent_target_head(pooled_enc)
            slot_select_logits = self.slot_select_head(slot).squeeze(-1)
            mask_logits = self.mask_head(slot)
            mask_prob = torch.sigmoid(mask_logits)
            value_in = slot + self.mask_context(mask_prob)
            value_raw = self.value_head(value_in)
            score_raw = self.score_head(slot).squeeze(-1)
            return {
                "encoder_hidden": enc,
                "slot_hidden": slot,
                "count_logits": count_logits,
                "sample_target_logits": sample_target_logits,
                "slot_select_logits": slot_select_logits,
                "mask_logits": mask_logits,
                "value_raw": value_raw,
                "score_raw": score_raw,
            }
else:
    class ParallelQuestionBlockGenerator:
        pass


def _state_dict_to_json(sd: dict[str, torch.Tensor]) -> dict[str, list]:
    if not TORCH_AVAILABLE:
        return {}
    out = {}
    for k, v in sd.items():
        out[k] = v.detach().cpu().numpy().tolist()
    return out


def _load_state_dict_from_json(model: nn.Module, raw: dict | None) -> bool:
    if not TORCH_AVAILABLE:
        return False
    if not isinstance(raw, dict) or not raw:
        return False
    cur = model.state_dict()
    loaded = {}
    try:
        for k, v in cur.items():
            if k not in raw:
                return False
            arr = np.array(raw[k], dtype=np.float32)
            if tuple(arr.shape) != tuple(v.shape):
                return False
            loaded[k] = torch.tensor(arr, dtype=v.dtype)
        model.load_state_dict(loaded, strict=True)
        return True
    except Exception:
        return False


def _sa_cfg(config: dict | None) -> dict:
    cfg = (config or {}).get("self_attention", {}) if isinstance((config or {}).get("self_attention", {}), dict) else {}
    dyn = (config or {}).get("dynamic_ppo", {}) if isinstance((config or {}).get("dynamic_ppo", {}), dict) else {}
    return {
        "seed": int(cfg.get("seed", dyn.get("seed", 42)) or 42),
        "hidden_dim": int(cfg.get("hidden_dim", 64) or 64),
        "num_heads": int(cfg.get("num_heads", 4) or 4),
        "train_respondents": int(cfg.get("train_respondents", 300) or 300),
        "train_epochs": int(cfg.get("train_epochs", 120) or 120),
        "train_lr": float(cfg.get("train_lr", 0.01) or 0.01),
        "online_lr": float(cfg.get("online_lr", 0.002) or 0.002),
        "online_epochs": int(cfg.get("online_epochs", 8) or 8),
        "batch_size": int(cfg.get("batch_size", 64) or 64),
        "sample_target_dim": int(cfg.get("sample_target_dim", 16) or 16),
        "target_sample_size": int(cfg.get("target_sample_size", 200) or 200),
        "count_loss_weight": float(cfg.get("count_loss_weight", 1.0) or 1.0),
        "slot_select_loss_weight": float(cfg.get("slot_select_loss_weight", 0.8) or 0.8),
        "mask_loss_weight": float(cfg.get("mask_loss_weight", 0.8) or 0.8),
        "value_loss_weight": float(cfg.get("value_loss_weight", 1.2) or 1.2),
        "score_loss_weight": float(cfg.get("score_loss_weight", 0.4) or 0.4),
        "sample_target_loss_weight": float(cfg.get("sample_target_loss_weight", 0.3) or 0.3),
    }


def _infer_variable_type(var_name: str | None, levels: list | None, given: str | None = None) -> str:
    raw = str(given or "").strip().lower()
    if raw in {"continuous", "ordinal", "categorical"}:
        return raw
    vals = list(levels or [])
    numeric = []
    for lv in vals:
        try:
            numeric.append(float(lv))
        except Exception:
            numeric = []
            break
    if numeric:
        return "continuous" if len(vals) > 4 else "ordinal"
    return "categorical"


def _extract_design_slots(spec: dict) -> tuple[list[dict], list[str]]:
    alternatives = spec.get("alternatives", []) if isinstance(spec, dict) else []
    slot_specs: list[dict] = []
    alt_order: list[str] = []
    for alt_idx, alt in enumerate(alternatives or []):
        if not isinstance(alt, dict):
            continue
        alt_name = str(alt.get("name") or alt.get("label") or f"alt_{alt_idx + 1}")
        alt_order.append(alt_name)
        for var_idx, var in enumerate(alt.get("variables", []) or []):
            if not isinstance(var, dict):
                continue
            var_name = str(var.get("name") or f"var_{var_idx + 1}")
            levels = list(var.get("levels", []) or [])
            variable_type = _infer_variable_type(var_name, levels, var.get("variable_type"))
            numeric_levels = []
            for lv in levels:
                try:
                    numeric_levels.append(float(lv))
                except Exception:
                    numeric_levels = []
                    break
            if numeric_levels:
                lower = float(min(numeric_levels))
                upper = float(max(numeric_levels))
                default = float(np.mean(numeric_levels))
            else:
                lower = 0.0
                upper = float(max(len(levels) - 1, 1))
                default = 0.0
            slot_specs.append(
                {
                    "index": len(slot_specs),
                    "alt_name": alt_name,
                    "alt_index": alt_idx,
                    "var_name": var_name,
                    "slot_key": f"{alt_name}.{var_name}",
                    "levels": levels,
                    "variable_type": variable_type,
                    "lower": lower,
                    "upper": upper,
                    "default": default,
                    "description": str(var.get("description") or var_name),
                }
            )
    return slot_specs, alt_order


def _normalize_slot_value(value, slot: dict) -> float:
    levels = list(slot.get("levels", []) or [])
    variable_type = str(slot.get("variable_type") or "continuous")
    lower = float(slot.get("lower", 0.0) or 0.0)
    upper = float(slot.get("upper", 1.0) or 1.0)
    if value is None:
        return 0.0
    if variable_type == "categorical":
        if not levels:
            return 0.0
        value_str = str(value)
        for idx, lv in enumerate(levels):
            if str(lv) == value_str:
                return float(idx / max(len(levels) - 1, 1))
        return 0.0
    num = _safe_float(value, lower)
    span = max(upper - lower, 1e-8)
    return float(np.clip((num - lower) / span, 0.0, 1.0))


def _denormalize_slot_value(value_norm: float, slot: dict):
    p = float(np.clip(value_norm, 0.0, 1.0))
    levels = list(slot.get("levels", []) or [])
    variable_type = str(slot.get("variable_type") or "continuous")
    lower = float(slot.get("lower", 0.0) or 0.0)
    upper = float(slot.get("upper", 1.0) or 1.0)

    if variable_type == "categorical" and levels:
        idx = int(round(p * max(len(levels) - 1, 0)))
        idx = max(0, min(idx, len(levels) - 1))
        return levels[idx]

    value = lower + p * max(upper - lower, 1e-8)
    if variable_type == "ordinal" and levels:
        numeric_levels = []
        for lv in levels:
            try:
                numeric_levels.append(float(lv))
            except Exception:
                numeric_levels = []
                break
        if numeric_levels:
            nearest = min(numeric_levels, key=lambda x: abs(x - value))
            if all(float(lv).is_integer() for lv in numeric_levels):
                return int(round(nearest))
            return float(nearest)
    if levels:
        numeric_levels = []
        for lv in levels:
            try:
                numeric_levels.append(float(lv))
            except Exception:
                numeric_levels = []
                break
        if numeric_levels and variable_type != "continuous":
            nearest = min(numeric_levels, key=lambda x: abs(x - value))
            return float(nearest)
    if float(lower).is_integer() and float(upper).is_integer() and abs(value - round(value)) < 1e-6:
        return int(round(value))
    return round(float(value), 3)


def _task_to_slot_arrays(task: dict, slot_specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((len(slot_specs),), dtype=np.float32)
    values = np.zeros((len(slot_specs),), dtype=np.float32)
    alts = (task or {}).get("alternatives", {}) if isinstance(task, dict) else {}
    for idx, slot in enumerate(slot_specs):
        attrs = (alts.get(slot["alt_name"], {}) or {}) if isinstance(alts, dict) else {}
        if slot["var_name"] in attrs:
            mask[idx] = 1.0
            values[idx] = _normalize_slot_value(attrs.get(slot["var_name"]), slot)
    return mask, values


def _group_tasks_into_blocks(tasks: list[dict], default_block_size: int) -> list[list[dict]]:
    blocks: dict[int, list[dict]] = {}
    for idx, task in enumerate(tasks or []):
        block_id = int((task or {}).get("block", 1) or 1)
        blocks.setdefault(block_id, []).append(task)
    out: list[list[dict]] = []
    if blocks:
        for block_id in sorted(blocks.keys()):
            block_tasks = sorted(blocks[block_id], key=lambda t: int((t or {}).get("row_in_block", 0) or 0))
            out.append(block_tasks)
        return out
    chunk = max(1, int(default_block_size))
    for start in range(0, len(tasks or []), chunk):
        out.append(list((tasks or [])[start:start + chunk]))
    return out


def _iter_submission_tasks(rows: list[dict] | None) -> list[list[dict]]:
    blocks: list[list[dict]] = []
    for row in rows or []:
        tasks = row.get("tasks", []) if isinstance(row, dict) else []
        if isinstance(tasks, list) and tasks:
            blocks.append([t for t in tasks if isinstance(t, dict)])
    return blocks


def _chunk_feature_vector(vec: np.ndarray, token_count: int, width: int) -> np.ndarray:
    token_count = max(1, int(token_count))
    width = max(4, int(width))
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros((token_count, width), dtype=np.float32)
    if arr.size < token_count * width:
        arr = np.pad(arr, (0, token_count * width - arr.size))
    else:
        arr = arr[: token_count * width]
    return arr.reshape(token_count, width)


def _build_cand_tokens(slot_specs: list[dict], *, token_count: int, width: int) -> np.ndarray:
    if not slot_specs:
        return np.zeros((token_count, width), dtype=np.float32)
    meta_rows: list[np.ndarray] = []
    max_alt = max([int(s.get("alt_index", 0) or 0) for s in slot_specs] + [0]) + 1
    for slot in slot_specs:
        levels = list(slot.get("levels", []) or [])
        vec = np.zeros((width,), dtype=np.float32)
        vec[0] = float((int(slot.get("alt_index", 0) or 0) + 1) / max(max_alt, 1))
        vtype = str(slot.get("variable_type") or "continuous")
        vec[1] = 1.0 if vtype == "continuous" else 0.0
        vec[2] = 1.0 if vtype == "ordinal" else 0.0
        vec[3] = 1.0 if vtype == "categorical" else 0.0
        vec[4] = float(min(len(levels), 12) / 12.0)
        vec[5] = float(slot.get("lower", 0.0) or 0.0)
        vec[6] = float(slot.get("upper", 0.0) or 0.0)
        vec[7] = float((float(slot.get("upper", 0.0) or 0.0) - float(slot.get("lower", 0.0) or 0.0)) / max(abs(float(slot.get("upper", 0.0) or 0.0)), 1.0))
        meta_rows.append(vec)
    meta = np.vstack(meta_rows)
    if meta.shape[0] < token_count:
        out = np.zeros((token_count, width), dtype=np.float32)
        out[: meta.shape[0], :] = meta
        return out
    chunks = np.array_split(meta, token_count, axis=0)
    return np.vstack([np.mean(c, axis=0) if len(c) else np.zeros((width,), dtype=np.float32) for c in chunks])


def _build_hist_tokens(historical_rows: list[dict] | None, slot_specs: list[dict], *, token_count: int, width: int) -> np.ndarray:
    task_blocks = _iter_submission_tasks(historical_rows)
    if not task_blocks:
        return np.zeros((token_count, width), dtype=np.float32)
    masks = []
    values = []
    counts = []
    for tasks in task_blocks:
        for task in tasks:
            m, v = _task_to_slot_arrays(task, slot_specs)
            masks.append(m)
            values.append(v)
            counts.append(float(np.sum(m)))
    if not masks:
        return np.zeros((token_count, width), dtype=np.float32)
    mask_mean = np.mean(np.vstack(masks), axis=0)
    value_mean = np.mean(np.vstack(values), axis=0)
    value_std = np.std(np.vstack(values), axis=0)
    density = np.array([float(np.mean(counts) / max(len(slot_specs), 1))], dtype=np.float32)
    joined = np.concatenate([mask_mean, value_mean, value_std, density], axis=0)
    return _chunk_feature_vector(joined, token_count, width)


def _pad_vector(vec: np.ndarray, width: int) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size < width:
        arr = np.pad(arr, (0, width - arr.size))
    else:
        arr = arr[:width]
    return arr.astype(np.float32)


def _build_env_vector(policy_state: dict, historical_rows: list[dict] | None, width: int) -> np.ndarray:
    signal = policy_state.get("current_mnl_signal", {}) if isinstance(policy_state.get("current_mnl_signal", {}), dict) else {}
    bound_violation = signal.get("bound_violation", {}) if isinstance(signal.get("bound_violation", {}), dict) else {}
    viol_vals = [_safe_float(v, 0.0) for v in bound_violation.values()]
    vec = np.zeros((width,), dtype=np.float32)
    vec[0] = float(min(1.0, int(policy_state.get("response_count", 0) or 0) / 200.0))
    vec[1] = float(np.clip(_safe_float(signal.get("adjusted_pseudo_r2"), 0.0), -1.0, 1.0))
    vec[2] = float(np.mean(viol_vals)) if viol_vals else 0.0
    vec[3] = float(min(1.0, len(historical_rows or []) / 200.0))
    vec[4] = float(min(1.0, int(policy_state.get("online_updates", 0) or 0) / 100.0))
    return vec


def _build_sample_target_cells(feature_spec: dict, max_cells: int) -> list[str]:
    """构造定向采样建议 head 的离散 cell 列表。

    输入:
        feature_spec: dynamicPPO 共用的人口编码规格，通常包含
            `zone_categories`、`attr_dim_names`、`attr_categories`。
        max_cells: 最多保留多少个采样 cell，避免 head 维度随 PopSim 配置无限膨胀。

    返回:
        list[str]: 稳定排序后的 cell 名称，例如 `zone=天河区`、`gender=female`。

    说明:
        这里不做全量笛卡尔积。全量 `zone × gender × age × edu ...` 很容易爆炸，
        也会使早期样本过稀。因此先使用“边际分布 cell”作为采样建议单元，
        后续 dashboard 可以再把多个 cell 组合成更细的人工配额建议。
    """
    spec = feature_spec if isinstance(feature_spec, dict) else {}
    cells: list[str] = []
    for zone in spec.get("zone_categories", []) if isinstance(spec.get("zone_categories", []), list) else []:
        z = str(zone)
        if z and z != _MISSING_ATTR_TOKEN:
            cells.append(f"zone={z}")
    names = spec.get("attr_dim_names", []) if isinstance(spec.get("attr_dim_names", []), list) else []
    cats_by_dim = spec.get("attr_categories", []) if isinstance(spec.get("attr_categories", []), list) else []
    for idx, cats in enumerate(cats_by_dim):
        dim_name = str(names[idx]) if idx < len(names) else f"attr_{idx}"
        for cat in cats if isinstance(cats, list) else []:
            c = str(cat)
            if c and c != _MISSING_ATTR_TOKEN:
                cells.append(f"{dim_name}={c}")
    deduped = list(dict.fromkeys(cells))
    return deduped[: max(1, int(max_cells))] or ["all"]


def _respondent_target_vector(respondent: dict, cells: list[str], feature_spec: dict | None = None) -> np.ndarray:
    """将单个 respondent 映射为采样 cell 的多标签向量。

    输入:
        respondent: 当前或合成受访者，至少包含 `zone_id` 和 `attr_segments`。
        cells: `_build_sample_target_cells` 返回的 cell 列表。
        feature_spec: 人口编码规格，用于把 `attr_segments` 的列位置映射到列名。

    返回:
        np.ndarray: 形状为 `[C_sample]` 的 0/1 多标签向量。
            respondent 可以同时命中 `zone=...`、`gender=...`、`age=...` 等多个 cell。
    """
    cells = list(cells or ["all"])
    vec = np.zeros((len(cells),), dtype=np.float32)
    if not cells:
        return vec
    if cells == ["all"]:
        vec[0] = 1.0
        return vec

    spec = feature_spec if isinstance(feature_spec, dict) else {}
    names = spec.get("attr_dim_names", []) if isinstance(spec.get("attr_dim_names", []), list) else []
    parts = respondent.get("attr_segments", []) if isinstance(respondent, dict) else []
    if not isinstance(parts, list):
        parts = []
    attr_map = {str(names[idx] if idx < len(names) else f"attr_{idx}"): str(val) for idx, val in enumerate(parts)}
    zone_id = str((respondent or {}).get("zone_id", "") or _MISSING_ATTR_TOKEN)

    for idx, cell in enumerate(cells):
        text = str(cell)
        if text == "all":
            vec[idx] = 1.0
        elif text.startswith("zone="):
            vec[idx] = 1.0 if zone_id == text.split("=", 1)[1] else 0.0
        elif "=" in text:
            key, value = text.split("=", 1)
            vec[idx] = 1.0 if attr_map.get(key) == value else 0.0
    if float(np.sum(vec)) <= 0:
        # 兜底：若 respondent 不在当前截断后的 cell 列表里，至少给一个弱监督信号。
        vec[0] = 1.0
    return vec


def _build_sample_cell_targets(pop_stats: dict | None, cells: list[str], feature_spec: dict | None = None) -> dict[str, float]:
    """从 PopSim 统计文件中提取采样 cell 的目标边际占比。

    输入:
        pop_stats: `popSimStats.json` 风格配置。
        cells: `_build_sample_target_cells` 生成的 cell 名称。
        feature_spec: 人口编码规格，用于识别属性列名。

    返回:
        dict[str, float]: cell -> target share。能从 PopSim 推断的使用目标值；
            推断不了的 zone cell 默认按分区均匀分配，保证采样建议不会全为空。
    """
    cells = list(cells or [])
    if not cells:
        return {}
    spec = feature_spec if isinstance(feature_spec, dict) else {}
    names = spec.get("attr_dim_names", []) if isinstance(spec.get("attr_dim_names", []), list) else []
    targets = {str(c): 0.0 for c in cells}

    zone_targets = {}
    default_target = {}
    expected_len = max(1, len(names))
    if isinstance(pop_stats, dict):
        default_target = pop_stats.get("default_target", {}) if isinstance(pop_stats.get("default_target", {}), dict) else {}
        zone_targets = pop_stats.get("zone_targets", {}) if isinstance(pop_stats.get("zone_targets", {}), dict) else {}
        if not zone_targets and isinstance(pop_stats.get("zone2_targets", {}), dict):
            zone_targets = pop_stats.get("zone2_targets", {})
        key_format = str(pop_stats.get("key_format", "") or "").strip()
        if key_format:
            expected_len = max(1, len(key_format.split("|")))

    zone_cells = [c for c in cells if str(c).startswith("zone=")]
    if zone_cells:
        zone_share = 1.0 / max(len(zone_cells), 1)
        for cell in zone_cells:
            targets[str(cell)] = zone_share

    attr_acc = {str(c): 0.0 for c in cells if "=" in str(c) and not str(c).startswith("zone=")}
    distributions: list[tuple[dict, float]] = []
    if isinstance(default_target, dict) and default_target:
        distributions.append((default_target, 1.0))
    if isinstance(zone_targets, dict) and zone_targets:
        z_weight = 1.0 / max(len(zone_targets), 1)
        for zv in zone_targets.values():
            if isinstance(zv, dict) and zv:
                distributions.append((zv, z_weight))
            elif isinstance(default_target, dict) and default_target:
                distributions.append((default_target, z_weight))

    for dist, dist_weight in distributions:
        total = sum(max(0.0, _safe_float(v, 0.0)) for v in (dist or {}).values())
        if total <= 0:
            continue
        for raw_key, raw_val in (dist or {}).items():
            parts = [str(x).strip() or _MISSING_ATTR_TOKEN for x in str(raw_key or "").split("|")]
            if len(parts) < expected_len:
                parts.extend([_MISSING_ATTR_TOKEN] * (expected_len - len(parts)))
            weight = dist_weight * max(0.0, _safe_float(raw_val, 0.0)) / total
            for idx, part in enumerate(parts[:expected_len]):
                dim_name = str(names[idx]) if idx < len(names) else f"attr_{idx}"
                cell = f"{dim_name}={part}"
                if cell in attr_acc:
                    attr_acc[cell] += weight

    attr_sum = sum(attr_acc.values())
    if attr_sum > 0:
        for cell, val in attr_acc.items():
            targets[cell] = float(val / attr_sum)
    return {k: round(float(v), 8) for k, v in targets.items()}


def _sample_target_recommendations(
    logits: np.ndarray,
    cells: list[str],
    policy_state: dict,
    *,
    target_sample_size: int,
    top_k: int = 8,
) -> list[dict]:
    """把 respondent_target_head 的 logits 转成人可读的定向采样建议。

    输入:
        logits: 模型输出，形状 `[C_sample]`。
        cells: cell 名称列表。
        policy_state: 当前策略状态，读取 `response_count` 估计剩余样本量。
        target_sample_size: 计划总样本量。
        top_k: 最多返回多少条建议。

    返回:
        list[dict]: 每条包含 `cell / priority / needed_n / reason`。
    """
    cells = list(cells or ["all"])
    arr = np.asarray(logits, dtype=np.float32).reshape(-1)
    if arr.size < len(cells):
        arr = np.pad(arr, (0, len(cells) - arr.size))
    arr = arr[: len(cells)]
    z = arr - float(np.max(arr)) if arr.size else arr
    prob = np.exp(z)
    prob = prob / max(float(np.sum(prob)), 1e-8)
    response_count = int((policy_state or {}).get("response_count", 0) or 0)
    remaining = max(0, int(target_sample_size) - response_count)

    # 若外部已经把 dashboard 统计出来的 sample_cell_counts / sample_cell_targets
    # 写入 policy_state，则优先把“缺口比例”融合进优先级；否则只使用模型 logits。
    counts = (policy_state or {}).get("sample_cell_counts", {})
    targets = (policy_state or {}).get("sample_cell_targets", {})
    if isinstance(counts, dict) and isinstance(targets, dict) and targets:
        gap = np.zeros_like(prob)
        for i, cell in enumerate(cells):
            target_share = _safe_float(targets.get(cell), 0.0)
            observed_share = _safe_float(counts.get(cell), 0.0) / max(response_count, 1)
            gap[i] = max(0.0, target_share - observed_share)
        if float(np.sum(gap)) > 0:
            gap = gap / max(float(np.sum(gap)), 1e-8)
            prob = 0.45 * prob + 0.55 * gap
            prob = prob / max(float(np.sum(prob)), 1e-8)

    order = list(np.argsort(-prob))[: max(1, int(top_k))]
    out = []
    for idx in order:
        out.append(
            {
                "cell": str(cells[int(idx)]),
                "priority": round(float(prob[int(idx)]), 6),
                "needed_n": int(round(float(prob[int(idx)]) * remaining)),
                "reason": "respondent_target_head_priority",
            }
        )
    return out


def _build_encoder_tokens(
    respondent: dict,
    *,
    policy_state: dict,
    feature_spec: dict,
    slot_specs: list[dict],
    historical_rows: list[dict] | None,
    context_dim: int,
    hist_token_count: int = 4,
    cand_token_count: int = 5,
) -> np.ndarray:
    rp_feat = dyppo_shared._respondent_feat(respondent, feature_spec=feature_spec)
    env_feat = _build_env_vector(policy_state, historical_rows, context_dim)
    rp_token = _pad_vector(rp_feat, context_dim)
    env_token = _pad_vector(env_feat, context_dim)
    hist_tokens = _build_hist_tokens(historical_rows, slot_specs, token_count=hist_token_count, width=context_dim)
    cand_tokens = _build_cand_tokens(slot_specs, token_count=cand_token_count, width=context_dim)
    return np.vstack([rp_token.reshape(1, -1), env_token.reshape(1, -1), hist_tokens, cand_tokens]).astype(np.float32)


def _slot_score_label(task: dict, beta_defaults: dict, respondent: dict) -> float:
    stats = dyppo_shared._task_choice_stats(task, beta_defaults, respondent, chosen_alt=None)
    spread = float(stats.get("spread", 0.0) or 0.0)
    entropy = float(stats.get("entropy_norm", 0.0) or 0.0)
    score = (spread / (1.0 + spread)) * 0.7 + entropy * 0.3
    return float(np.clip(score, 0.0, 1.0))


def _block_to_labels(
    tasks: list[dict],
    *,
    slot_specs: list[dict],
    count_min: int,
    count_max: int,
    beta_defaults: dict,
    respondent: dict,
) -> dict:
    t_max = max(1, int(count_max))
    v_dim = len(slot_specs)
    count = max(int(count_min), min(int(len(tasks)), int(count_max))) if tasks else int(count_min)
    count_label = int(max(0, min(count - int(count_min), int(count_max) - int(count_min))))
    slot_mask = np.zeros((t_max,), dtype=np.float32)
    slot_mask[:count] = 1.0
    mask_labels = np.zeros((t_max, v_dim), dtype=np.float32)
    value_labels = np.zeros((t_max, v_dim), dtype=np.float32)
    score_labels = np.zeros((t_max,), dtype=np.float32)
    for idx, task in enumerate(tasks[:t_max]):
        m, v = _task_to_slot_arrays(task, slot_specs)
        mask_labels[idx] = m
        value_labels[idx] = v
        score_labels[idx] = _slot_score_label(task, beta_defaults, respondent)
    return {
        "count_label": count_label,
        "slot_mask": slot_mask,
        "mask_labels": mask_labels,
        "value_labels": value_labels,
        "score_labels": score_labels,
    }


def _sample_teacher_block(
    *,
    candidate_pool: list[dict],
    expert_blocks: list[list[dict]],
    count: int,
    rng: np.random.Generator,
) -> list[dict]:
    pool = [deepcopy(t) for t in candidate_pool if isinstance(t, dict)]
    if not pool:
        return []
    picked: list[dict] = []
    used: set[str] = set()
    base = expert_blocks[int(rng.integers(0, len(expert_blocks)))] if expert_blocks else []
    for task in list(base)[:count]:
        sig = str(task.get("sig") or _task_signature(task))
        if sig in used:
            continue
        used.add(sig)
        picked.append(deepcopy(task))
    while len(picked) < count and pool:
        task = deepcopy(pool[int(rng.integers(0, len(pool)))])
        sig = str(task.get("sig") or _task_signature(task))
        if sig in used:
            continue
        used.add(sig)
        picked.append(task)
    return picked[:count]


def _build_teacher_samples(
    *,
    payload: dict,
    candidate_pool: list[dict],
    expert_result: dict | None,
    rows: list[dict] | None,
    respondents: list[dict],
    slot_specs: list[dict],
    feature_spec: dict,
    context_dim: int,
    count_min: int,
    count_max: int,
    beta_defaults: dict,
    seed: int,
    sample_target_cells: list[str] | None = None,
) -> dict:
    row_blocks = _iter_submission_tasks(rows)
    expert_blocks = _group_tasks_into_blocks((expert_result or {}).get("comb", []) or [], default_block_size=max(1, count_max))
    rng = np.random.default_rng(seed)
    samples = []

    if row_blocks:
        for idx, tasks in enumerate(row_blocks):
            respondent = respondents[idx % len(respondents)] if respondents else {
                "respondent_id": f"row_{idx+1:03d}",
                "zone_id": _MISSING_ATTR_TOKEN,
                "attr_segments": [_MISSING_ATTR_TOKEN] * max(1, len(feature_spec.get("attr_dim_names", []))),
            }
            labels = _block_to_labels(
                tasks,
                slot_specs=slot_specs,
                count_min=count_min,
                count_max=count_max,
                beta_defaults=beta_defaults,
                respondent=respondent,
            )
            tokens = _build_encoder_tokens(
                respondent,
                policy_state={},
                feature_spec=feature_spec,
                slot_specs=slot_specs,
                historical_rows=rows,
                context_dim=context_dim,
            )
            sample_target_label = _respondent_target_vector(respondent, sample_target_cells or ["all"], feature_spec)
            samples.append({"tokens": tokens, "sample_target_label": sample_target_label, **labels})
    else:
        respondent_list = respondents if respondents else [
            {
                "respondent_id": "sim_default",
                "zone_id": _MISSING_ATTR_TOKEN,
                "attr_segments": [_MISSING_ATTR_TOKEN] * max(1, len(feature_spec.get("attr_dim_names", []))),
            }
        ]
        for respondent in respondent_list:
            count = int(rng.integers(count_min, count_max + 1)) if count_max > count_min else int(count_min)
            tasks = _sample_teacher_block(
                candidate_pool=candidate_pool,
                expert_blocks=expert_blocks,
                count=count,
                rng=rng,
            )
            labels = _block_to_labels(
                tasks,
                slot_specs=slot_specs,
                count_min=count_min,
                count_max=count_max,
                beta_defaults=beta_defaults,
                respondent=respondent,
            )
            tokens = _build_encoder_tokens(
                respondent,
                policy_state={},
                feature_spec=feature_spec,
                slot_specs=slot_specs,
                historical_rows=None,
                context_dim=context_dim,
            )
            sample_target_label = _respondent_target_vector(respondent, sample_target_cells or ["all"], feature_spec)
            samples.append({"tokens": tokens, "sample_target_label": sample_target_label, **labels})

    if not samples:
        return {
            "encoder_tokens": np.zeros((0, 1, context_dim), dtype=np.float32),
            "count_labels": np.zeros((0,), dtype=np.int64),
            "slot_masks": np.zeros((0, count_max), dtype=np.float32),
            "mask_labels": np.zeros((0, count_max, len(slot_specs)), dtype=np.float32),
            "value_labels": np.zeros((0, count_max, len(slot_specs)), dtype=np.float32),
            "score_labels": np.zeros((0, count_max), dtype=np.float32),
            "sample_target_labels": np.zeros((0, len(sample_target_cells or ["all"])), dtype=np.float32),
        }

    return {
        "encoder_tokens": np.stack([s["tokens"] for s in samples]).astype(np.float32),
        "count_labels": np.array([s["count_label"] for s in samples], dtype=np.int64),
        "slot_masks": np.stack([s["slot_mask"] for s in samples]).astype(np.float32),
        "mask_labels": np.stack([s["mask_labels"] for s in samples]).astype(np.float32),
        "value_labels": np.stack([s["value_labels"] for s in samples]).astype(np.float32),
        "score_labels": np.stack([s["score_labels"] for s in samples]).astype(np.float32),
        "sample_target_labels": np.stack([s["sample_target_label"] for s in samples]).astype(np.float32),
    }


def _train_parallel_generator(
    dataset: dict,
    *,
    context_dim: int,
    slot_dim: int,
    hidden_dim: int,
    num_heads: int,
    max_questions: int,
    count_classes: int,
    sample_target_dim: int,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    count_loss_weight: float,
    slot_select_loss_weight: float,
    mask_loss_weight: float,
    value_loss_weight: float,
    score_loss_weight: float,
    sample_target_loss_weight: float,
    init_state: dict | None,
    verbose: bool = False,
    progress_prefix: str = "selfattention/train",
) -> tuple[dict, list[dict]]:
    if not TORCH_AVAILABLE:
        return {}, [{"epoch": 0, "msg": "torch not installed"}]

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = ParallelQuestionBlockGenerator(
        context_dim=context_dim,
        slot_dim=slot_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        max_questions=max_questions,
        count_classes=count_classes,
        sample_target_dim=sample_target_dim,
    )
    _load_state_dict_from_json(model, init_state if isinstance(init_state, dict) else None)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    enc_np = np.asarray(dataset.get("encoder_tokens", np.zeros((0, 1, context_dim))), dtype=np.float32)
    count_np = np.asarray(dataset.get("count_labels", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    slot_np = np.asarray(dataset.get("slot_masks", np.zeros((0, max_questions))), dtype=np.float32)
    mask_np = np.asarray(dataset.get("mask_labels", np.zeros((0, max_questions, slot_dim))), dtype=np.float32)
    value_np = np.asarray(dataset.get("value_labels", np.zeros((0, max_questions, slot_dim))), dtype=np.float32)
    score_np = np.asarray(dataset.get("score_labels", np.zeros((0, max_questions))), dtype=np.float32)
    sample_target_np = np.asarray(
        dataset.get("sample_target_labels", np.zeros((0, sample_target_dim), dtype=np.float32)),
        dtype=np.float32,
    )
    if sample_target_np.ndim != 2 or sample_target_np.shape[1] != int(sample_target_dim):
        fixed = np.zeros((enc_np.shape[0], int(sample_target_dim)), dtype=np.float32)
        rows = min(fixed.shape[0], sample_target_np.shape[0] if sample_target_np.ndim >= 1 else 0)
        cols = min(fixed.shape[1], sample_target_np.shape[1] if sample_target_np.ndim == 2 else 0)
        if rows and cols:
            fixed[:rows, :cols] = sample_target_np[:rows, :cols]
        sample_target_np = fixed

    if enc_np.size == 0:
        return _state_dict_to_json(model.state_dict()), [{"epoch": 0, "loss": None, "samples": 0}]

    _progress_print(
        verbose,
        progress_prefix,
        (
            f"start: samples={enc_np.shape[0]} context_dim={context_dim} slot_dim={slot_dim} "
            f"max_questions={max_questions} count_classes={count_classes} hidden_dim={hidden_dim} heads={num_heads}"
        ),
    )

    x = torch.tensor(enc_np, dtype=torch.float32)
    y_count = torch.tensor(count_np, dtype=torch.long)
    y_slot = torch.tensor(slot_np, dtype=torch.float32)
    y_mask = torch.tensor(mask_np, dtype=torch.float32)
    y_value = torch.tensor(value_np, dtype=torch.float32)
    y_score = torch.tensor(score_np, dtype=torch.float32)
    y_sample_target = torch.tensor(sample_target_np, dtype=torch.float32)

    n = int(enc_np.shape[0])
    batch_size = max(1, min(int(batch_size), n))
    rng = np.random.default_rng(seed + 29)
    logs: list[dict] = []

    for ep in range(max(1, int(epochs))):
        perm = rng.permutation(n)
        ep_total = []
        ep_count = []
        ep_mask = []
        ep_value = []
        ep_score = []
        ep_slot = []
        ep_sample = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            idx_t = torch.tensor(idx, dtype=torch.long)
            b_x = x[idx_t]
            b_count = y_count[idx_t]
            b_slot = y_slot[idx_t]
            b_mask = y_mask[idx_t]
            b_value = y_value[idx_t]
            b_score = y_score[idx_t]
            b_sample_target = y_sample_target[idx_t]

            out = model(b_x)
            count_logits = out["count_logits"]
            slot_select_logits = out["slot_select_logits"]
            sample_target_logits = out["sample_target_logits"]
            mask_logits = out["mask_logits"]
            value_pred = torch.sigmoid(out["value_raw"])
            score_pred = torch.sigmoid(out["score_raw"])

            count_loss = F.cross_entropy(count_logits, b_count)
            slot_select_loss = F.binary_cross_entropy_with_logits(slot_select_logits, b_slot)

            sample_mass = b_sample_target.sum(dim=1, keepdim=True)
            sample_valid = sample_mass.squeeze(-1) > 0
            if bool(torch.any(sample_valid)):
                sample_dist = b_sample_target / torch.clamp(sample_mass, min=1.0)
                sample_target_loss_vec = -(
                    sample_dist * F.log_softmax(sample_target_logits, dim=-1)
                ).sum(dim=-1)
                sample_target_loss = sample_target_loss_vec[sample_valid].mean()
            else:
                sample_target_loss = sample_target_logits.sum() * 0.0

            slot_w = b_slot.unsqueeze(-1)
            mask_loss_raw = F.binary_cross_entropy_with_logits(mask_logits, b_mask, reduction="none")
            mask_denom = torch.clamp(slot_w.sum() * b_mask.shape[-1], min=1.0)
            mask_loss = (mask_loss_raw * slot_w).sum() / mask_denom

            value_w = slot_w * b_mask
            value_denom = torch.clamp(value_w.sum(), min=1.0)
            value_loss = (((value_pred - b_value) ** 2) * value_w).sum() / value_denom

            score_denom = torch.clamp(b_slot.sum(), min=1.0)
            score_loss = (((score_pred - b_score) ** 2) * b_slot).sum() / score_denom

            loss = (
                float(count_loss_weight) * count_loss
                + float(slot_select_loss_weight) * slot_select_loss
                + float(mask_loss_weight) * mask_loss
                + float(value_loss_weight) * value_loss
                + float(score_loss_weight) * score_loss
                + float(sample_target_loss_weight) * sample_target_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_total.append(float(loss.detach().cpu().item()))
            ep_count.append(float(count_loss.detach().cpu().item()))
            ep_mask.append(float(mask_loss.detach().cpu().item()))
            ep_value.append(float(value_loss.detach().cpu().item()))
            ep_score.append(float(score_loss.detach().cpu().item()))
            ep_slot.append(float(slot_select_loss.detach().cpu().item()))
            ep_sample.append(float(sample_target_loss.detach().cpu().item()))

        mean_total = float(np.mean(ep_total)) if ep_total else 0.0
        mean_count = float(np.mean(ep_count)) if ep_count else 0.0
        mean_mask = float(np.mean(ep_mask)) if ep_mask else 0.0
        mean_value = float(np.mean(ep_value)) if ep_value else 0.0
        mean_score = float(np.mean(ep_score)) if ep_score else 0.0
        mean_slot = float(np.mean(ep_slot)) if ep_slot else 0.0
        mean_sample = float(np.mean(ep_sample)) if ep_sample else 0.0
        if ep == 0 or ep == epochs - 1 or ep % max(1, epochs // 5) == 0:
            logs.append(
                {
                    "epoch": int(ep + 1),
                    "loss": round(mean_total, 6),
                    "count_loss": round(mean_count, 6),
                    "slot_select_loss": round(mean_slot, 6),
                    "mask_loss": round(mean_mask, 6),
                    "value_loss": round(mean_value, 6),
                    "score_loss": round(mean_score, 6),
                    "sample_target_loss": round(mean_sample, 6),
                    "samples": n,
                }
            )
        _progress_print(
            verbose,
            progress_prefix,
            (
                f"epoch {ep + 1}/{max(1, int(epochs))}: loss={mean_total:.6f} "
                f"count={mean_count:.6f} slot={mean_slot:.6f} mask={mean_mask:.6f} "
                f"value={mean_value:.6f} score={mean_score:.6f} sample={mean_sample:.6f}"
            ),
        )

    return _state_dict_to_json(model.state_dict()), logs


def _prepare_candidate_library(candidate_pool: list[dict], slot_specs: list[dict]) -> list[dict]:
    library = []
    for task in candidate_pool or []:
        if not isinstance(task, dict):
            continue
        mask, value = _task_to_slot_arrays(task, slot_specs)
        library.append(
            {
                "task": deepcopy(task),
                "sig": str(task.get("sig") or _task_signature(task)),
                "mask": mask,
                "value": value,
            }
        )
    return library


def _pick_prototype(
    target_mask: np.ndarray,
    target_value: np.ndarray,
    library: list[dict],
    used: set[str],
) -> dict | None:
    best = None
    best_score = None
    for item in library:
        sig = str(item.get("sig") or "")
        if sig in used:
            continue
        cand_mask = np.asarray(item.get("mask"), dtype=np.float32)
        cand_value = np.asarray(item.get("value"), dtype=np.float32)
        mask_gap = float(np.mean(np.abs(target_mask - cand_mask)))
        active = np.maximum(target_mask, cand_mask)
        value_gap = float(np.sum(np.abs(target_value - cand_value) * active) / max(np.sum(active), 1.0))
        score = mask_gap + 0.6 * value_gap
        if best is None or score < float(best_score):
            best = item
            best_score = score
    return best


def _slot_arrays_to_task(
    mask_vec: np.ndarray,
    value_vec: np.ndarray,
    *,
    slot_specs: list[dict],
    alt_order: list[str],
    prototype: dict | None,
    row_in_block: int,
    block_id: int,
    slot_score: float,
) -> dict:
    alts = {str(alt): {} for alt in alt_order}
    proto_alts = ((prototype or {}).get("alternatives", {}) if isinstance(prototype, dict) else {}) or {}
    for idx, slot in enumerate(slot_specs):
        alt_name = str(slot["alt_name"])
        var_name = str(slot["var_name"])
        if float(mask_vec[idx]) >= 0.5:
            alts[alt_name][var_name] = _denormalize_slot_value(float(value_vec[idx]), slot)
    # 若某个备选项被全部 mask 掉，则退回 prototype 中对应的属性；若还没有，则至少填一个变量。
    for alt_name in alt_order:
        alt_name = str(alt_name)
        if alts[alt_name]:
            continue
        if alt_name in proto_alts and isinstance(proto_alts.get(alt_name), dict) and proto_alts.get(alt_name):
            alts[alt_name] = deepcopy(proto_alts.get(alt_name))
            continue
        for idx, slot in enumerate(slot_specs):
            if str(slot["alt_name"]) == alt_name:
                alts[alt_name][str(slot["var_name"])] = _denormalize_slot_value(float(value_vec[idx]), slot)
                break
    task = {
        "block": int(block_id),
        "row_in_block": int(row_in_block),
        "alternatives": alts,
        "slot_score": round(float(slot_score), 6),
    }
    task["sig"] = _task_signature(task)
    task["id"] = f"preview_b{int(block_id)}_r{int(row_in_block)}"
    return task


def _predict_count_range(count_logits: np.ndarray, count_min: int) -> tuple[int, np.ndarray]:
    shifted = np.exp(count_logits - np.max(count_logits))
    probs = shifted / max(np.sum(shifted), 1e-8)
    count_idx = int(np.argmax(probs))
    return int(count_min + count_idx), probs


def _generate_parallel_block(
    *,
    payload: dict,
    policy_state: dict,
    current_respondent: dict,
    candidate_pool: list[dict],
    expert_result: dict | None,
    slot_specs: list[dict],
    alt_order: list[str],
    feature_spec: dict,
    context_dim: int,
    hidden_dim: int,
    num_heads: int,
    sample_target_dim: int,
    sample_target_cells: list[str],
    target_sample_size: int,
    count_min: int,
    count_max: int,
    historical_rows: list[dict] | None,
    verbose: bool = False,
) -> tuple[list[dict], dict]:
    if not TORCH_AVAILABLE:
        return [], {}
    count_classes = max(1, int(count_max - count_min + 1))
    model = ParallelQuestionBlockGenerator(
        context_dim=context_dim,
        slot_dim=len(slot_specs),
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        max_questions=count_max,
        count_classes=count_classes,
        sample_target_dim=sample_target_dim,
    )
    _load_state_dict_from_json(model, policy_state.get("selfattention_state", {}))
    model.eval()

    tokens = _build_encoder_tokens(
        current_respondent,
        policy_state=policy_state,
        feature_spec=feature_spec,
        slot_specs=slot_specs,
        historical_rows=historical_rows,
        context_dim=context_dim,
    )
    x = torch.tensor(tokens, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        count_logits = out["count_logits"].squeeze(0).cpu().numpy()
        slot_select_prob = torch.sigmoid(out["slot_select_logits"]).squeeze(0).cpu().numpy()
        sample_target_logits = out["sample_target_logits"].squeeze(0).cpu().numpy()
        mask_prob = torch.sigmoid(out["mask_logits"]).squeeze(0).cpu().numpy()
        value_prob = torch.sigmoid(out["value_raw"]).squeeze(0).cpu().numpy()
        score_prob = torch.sigmoid(out["score_raw"]).squeeze(0).cpu().numpy()

    question_count, count_probs = _predict_count_range(count_logits, count_min)
    question_count = max(count_min, min(count_max, question_count))
    slot_quality = 0.65 * slot_select_prob + 0.35 * score_prob
    slot_indices = list(np.argsort(-slot_quality)[:question_count])
    slot_indices.sort(key=lambda idx: (-float(slot_quality[idx]), idx))

    expert_blocks = _group_tasks_into_blocks((expert_result or {}).get("comb", []) or [], default_block_size=question_count)
    expert_proto = expert_blocks[0] if expert_blocks else []
    library = _prepare_candidate_library(candidate_pool, slot_specs)
    used_sigs: set[str] = set()
    tasks: list[dict] = []

    signal = policy_state.get("current_mnl_signal", {}) if isinstance(policy_state.get("current_mnl_signal", {}), dict) else {}
    bound_violation = signal.get("bound_violation", {}) if isinstance(signal.get("bound_violation", {}), dict) else {}

    for out_idx, slot_idx in enumerate(slot_indices, start=1):
        target_mask = (mask_prob[slot_idx] >= 0.5).astype(np.float32)
        # 若某些变量当前估计出现越界，则降低激活概率；这里只做轻量惩罚，不直接清零。
        for dim, slot in enumerate(slot_specs):
            penalty = _safe_float(bound_violation.get(str(slot.get("slot_key")), 0.0), 0.0)
            if penalty > 0:
                target_mask[dim] = 1.0 if (float(mask_prob[slot_idx, dim]) * max(0.2, 1.0 - penalty)) >= 0.5 else 0.0
        target_value = np.clip(value_prob[slot_idx], 0.0, 1.0)
        proto = expert_proto[(out_idx - 1) % len(expert_proto)] if expert_proto else None
        picked_proto = _pick_prototype(target_mask, target_value, library, used_sigs)
        if picked_proto is not None:
            proto = deepcopy(picked_proto.get("task"))
            used_sigs.add(str(picked_proto.get("sig") or ""))
        proto_mask, proto_value = _task_to_slot_arrays(proto, slot_specs) if isinstance(proto, dict) else (
            np.zeros((len(slot_specs),), dtype=np.float32),
            np.zeros((len(slot_specs),), dtype=np.float32),
        )
        if float(np.sum(target_mask)) <= 0:
            target_mask = proto_mask.copy() if float(np.sum(proto_mask)) > 0 else np.ones_like(target_mask)
        blended_value = 0.7 * target_value + 0.3 * proto_value
        task = _slot_arrays_to_task(
            target_mask,
            blended_value,
            slot_specs=slot_specs,
            alt_order=alt_order,
            prototype=proto,
            row_in_block=out_idx,
            block_id=1,
            slot_score=float(slot_quality[slot_idx]),
        )
        tasks.append(task)

    sampling_recommendation = _sample_target_recommendations(
        sample_target_logits,
        sample_target_cells,
        policy_state,
        target_sample_size=target_sample_size,
        top_k=min(8, len(sample_target_cells or [])),
    )
    model_summary = {
        "count_probs": [round(float(x), 6) for x in count_probs.tolist()],
        "predicted_question_count": int(question_count),
        "slot_scores": [round(float(slot_quality[idx]), 6) for idx in slot_indices],
        "slot_select_scores": [round(float(slot_select_prob[idx]), 6) for idx in slot_indices],
        "quality_scores": [round(float(score_prob[idx]), 6) for idx in slot_indices],
        "slot_indices": [int(idx) for idx in slot_indices],
        "sampling_recommendation": sampling_recommendation,
    }
    _progress_print(
        verbose,
        "selfattention/generate",
        f"generated block: question_count={question_count} slot_indices={slot_indices}",
    )
    return tasks, model_summary


def train_self_attention_ppo(
    *,
    payload: dict,
    policy_state: dict,
    candidate_pool: list[dict],
    expert_result: dict | None,
    rows: list[dict] | None = None,
    current_respondent: dict | None = None,
    data_dir: Path,
    config: dict,
    verbose: bool = False,
) -> dict:
    if not TORCH_AVAILABLE:
        return {
            "comb": [],
            "d_error": {"value": None},
            "iteration_log": [{"epoch": 0, "msg": "PyTorch未安装，SelfAttention无法训练。请先安装torch。"}],
            "model_state": {
                "trained": False,
                "backend": "missing_torch",
                "required_package": "torch",
                "install_hint": "请先在虚拟环境中安装 PyTorch，再运行 SelfAttention。",
            },
            "policy_state": policy_state,
        }

    sopt = ((payload.get("design_options", {}) or {}).get("selfattention", {}) or {}) if isinstance(payload, dict) else {}
    scfg = _sa_cfg(config)
    seed = int(scfg.get("seed", 42))
    tpr = int(sopt.get("tasks_per_round", 6) or 6)
    count_min = int(sopt.get("count_min", tpr) or tpr)
    count_max = int(sopt.get("count_max", count_min) or count_min)
    if count_max < count_min:
        count_max = count_min
    count_max = max(1, min(count_max, max(len(candidate_pool or []), count_max)))
    context_dim = max(24, int(sopt.get("context_dim", 48) or 48))
    hidden_dim = max(16, int(sopt.get("hidden_dim", scfg.get("hidden_dim", 64)) or scfg.get("hidden_dim", 64)))
    num_heads = _normalize_heads(hidden_dim, int(sopt.get("num_heads", scfg.get("num_heads", 4)) or scfg.get("num_heads", 4)))
    train_n = int(sopt.get("train_respondents", scfg.get("train_respondents", 300)) or scfg.get("train_respondents", 300))
    epochs = int(sopt.get("train_epochs", scfg.get("train_epochs", 120)) or scfg.get("train_epochs", 120))
    lr = float(sopt.get("train_lr", scfg.get("train_lr", 0.01)) or scfg.get("train_lr", 0.01))
    batch_size = int(sopt.get("batch_size", scfg.get("batch_size", 64)) or scfg.get("batch_size", 64))
    sample_target_dim_cfg = int(sopt.get("sample_target_dim", scfg.get("sample_target_dim", 16)) or scfg.get("sample_target_dim", 16))
    target_sample_size = int(sopt.get("target_sample_size", scfg.get("target_sample_size", payload.get("sample_size", 200))) or scfg.get("target_sample_size", 200))
    count_loss_weight = float(sopt.get("count_loss_weight", scfg.get("count_loss_weight", 1.0)) or scfg.get("count_loss_weight", 1.0))
    slot_select_loss_weight = float(sopt.get("slot_select_loss_weight", scfg.get("slot_select_loss_weight", 0.8)) or scfg.get("slot_select_loss_weight", 0.8))
    mask_loss_weight = float(sopt.get("mask_loss_weight", scfg.get("mask_loss_weight", 0.8)) or scfg.get("mask_loss_weight", 0.8))
    value_loss_weight = float(sopt.get("value_loss_weight", scfg.get("value_loss_weight", 1.2)) or scfg.get("value_loss_weight", 1.2))
    score_loss_weight = float(sopt.get("score_loss_weight", scfg.get("score_loss_weight", 0.4)) or scfg.get("score_loss_weight", 0.4))
    sample_target_loss_weight = float(sopt.get("sample_target_loss_weight", scfg.get("sample_target_loss_weight", 0.3)) or scfg.get("sample_target_loss_weight", 0.3))

    spec = payload.get("design_spec", {}) if isinstance(payload.get("design_spec", {}), dict) else {}
    beta_defaults = payload.get("beta_defaults", {}) if isinstance(payload.get("beta_defaults", {}), dict) else {}
    slot_specs, alt_order = _extract_design_slots(spec)
    if not slot_specs:
        return {
            "comb": [],
            "d_error": {"value": None},
            "iteration_log": [{"epoch": 0, "msg": "design_spec 中没有可用变量槽位。"}],
            "model_state": {"trained": False, "backend": "invalid_design_spec"},
            "policy_state": policy_state,
        }

    pop_stats = dyppo_shared._load_pop_stats(payload, data_dir=data_dir, config=config)
    feature_spec = dyppo_shared._build_feature_spec(pop_stats)
    sample_target_cells = _build_sample_target_cells(feature_spec, sample_target_dim_cfg)
    sample_cell_targets = _build_sample_cell_targets(pop_stats, sample_target_cells, feature_spec)
    sample_target_dim = len(sample_target_cells)
    respondents = dyppo_shared._sample_respondents(pop_stats, train_n, seed + int(policy_state.get("response_count", 0)))
    dataset = _build_teacher_samples(
        payload=payload,
        candidate_pool=list(candidate_pool or []),
        expert_result=expert_result,
        rows=rows,
        respondents=respondents,
        slot_specs=slot_specs,
        feature_spec=feature_spec,
        context_dim=context_dim,
        count_min=count_min,
        count_max=count_max,
        beta_defaults=beta_defaults,
        seed=seed + 7,
        sample_target_cells=sample_target_cells,
    )

    init_state = policy_state.get("selfattention_state", {}) if isinstance(policy_state.get("selfattention_state", {}), dict) else None
    new_state, logs = _train_parallel_generator(
        dataset,
        context_dim=context_dim,
        slot_dim=len(slot_specs),
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        max_questions=count_max,
        count_classes=max(1, count_max - count_min + 1),
        sample_target_dim=sample_target_dim,
        seed=seed,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        count_loss_weight=count_loss_weight,
        slot_select_loss_weight=slot_select_loss_weight,
        mask_loss_weight=mask_loss_weight,
        value_loss_weight=value_loss_weight,
        score_loss_weight=score_loss_weight,
        sample_target_loss_weight=sample_target_loss_weight,
        init_state=init_state,
        verbose=verbose,
        progress_prefix="selfattention/train",
    )

    score_respondent = current_respondent if isinstance(current_respondent, dict) and current_respondent else None
    if not score_respondent and respondents:
        score_respondent = respondents[0]
    if not score_respondent:
        score_respondent = {
            "respondent_id": "default",
            "zone_id": _MISSING_ATTR_TOKEN,
            "attr_segments": [_MISSING_ATTR_TOKEN] * max(1, len(feature_spec.get("attr_dim_names", []))),
        }

    policy_state["selfattention_state"] = new_state
    policy_state["trained"] = True
    policy_state["context_dim"] = int(context_dim)
    policy_state["slot_dim"] = int(len(slot_specs))
    policy_state["hidden_dim"] = int(hidden_dim)
    policy_state["num_heads"] = int(num_heads)
    policy_state["count_min"] = int(count_min)
    policy_state["count_max"] = int(count_max)
    policy_state["sample_target_dim"] = int(sample_target_dim)
    policy_state["sample_target_cells"] = deepcopy(sample_target_cells)
    policy_state["sample_cell_targets"] = deepcopy(sample_cell_targets)
    policy_state["target_sample_size"] = int(target_sample_size)
    policy_state["architecture_mode"] = "parallel_question_block_generator"
    policy_state["response_count"] = int(policy_state.get("response_count", 0) or 0)
    policy_state["candidate_signatures"] = [str((c or {}).get("sig") or _task_signature(c)) for c in (candidate_pool or [])]
    policy_state["attr_dim_names"] = deepcopy(feature_spec.get("attr_dim_names", []))
    policy_state["attr_categories"] = deepcopy(feature_spec.get("attr_categories", []))
    policy_state["zone_categories"] = deepcopy(feature_spec.get("zone_categories", []))

    current_mnl_signal = policy_state.get("current_mnl_signal", {}) if isinstance(policy_state.get("current_mnl_signal", {}), dict) else {}
    generated_tasks, gen_summary = _generate_parallel_block(
        payload=payload,
        policy_state=policy_state,
        current_respondent=score_respondent,
        candidate_pool=list(candidate_pool or []),
        expert_result=expert_result,
        slot_specs=slot_specs,
        alt_order=alt_order,
        feature_spec=feature_spec,
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        sample_target_dim=sample_target_dim,
        sample_target_cells=sample_target_cells,
        target_sample_size=target_sample_size,
        count_min=count_min,
        count_max=count_max,
        historical_rows=rows,
        verbose=verbose,
    )

    model_state = {
        "trained": True,
        "backend": "pytorch_parallel_selfattention",
        "architecture_mode": "parallel_question_block_generator",
        "question_generation_mode": "parallel_queries",
        "context_dim": int(context_dim),
        "slot_dim": int(len(slot_specs)),
        "max_questions": int(count_max),
        "count_range": [int(count_min), int(count_max)],
        "count_head_classes": int(max(1, count_max - count_min + 1)),
        "variable_semantics": "count -> slot_select -> variable_mask -> value",
        "sampling_semantics": "respondent_target_head -> recommended RP cells for remaining sample collection",
        "sample_target_dim": int(sample_target_dim),
        "sample_target_cells": deepcopy(sample_target_cells),
        "sample_cell_targets": deepcopy(sample_cell_targets),
        "train_samples": int(dataset.get("encoder_tokens", np.zeros((0, 1, context_dim))).shape[0]),
        "train_epochs": int(epochs),
        "hidden_dim": int(hidden_dim),
        "num_heads": int(num_heads),
        "prototype_pool_size": int(len(candidate_pool or [])),
        "expert_demo_size": int(len((expert_result or {}).get("comb", []) or [])),
        "current_mnl_signal": deepcopy(current_mnl_signal),
        "generated_summary": gen_summary,
    }
    sampling_recommendation = gen_summary.get("sampling_recommendation", []) if isinstance(gen_summary, dict) else []
    policy_state["last_sampling_recommendation"] = deepcopy(sampling_recommendation)
    model_state["sampling_recommendation"] = deepcopy(sampling_recommendation)

    return {
        "comb": generated_tasks,
        "d_error": {"value": None},
        "iteration_log": logs,
        "model_state": model_state,
        "policy_state": policy_state,
        "sampling_recommendation": sampling_recommendation,
    }


def online_update_self_attention_ppo(
    *,
    payload: dict,
    policy_state: dict,
    tasks: list[dict],
    choices: dict,
    respondent: dict | None = None,
    config: dict | None = None,
    historical_rows: list[dict] | None = None,
    verbose: bool = False,
) -> dict:
    if not TORCH_AVAILABLE:
        return {"updated": False, "reason": "torch not installed"}
    if not tasks:
        return {"updated": False, "reason": "empty tasks"}
    if not policy_state.get("selfattention_state"):
        return {"updated": False, "reason": "weights not initialized"}

    sopt = ((payload.get("design_options", {}) or {}).get("selfattention", {}) or {}) if isinstance(payload, dict) else {}
    scfg = _sa_cfg(config)
    context_dim = int(policy_state.get("context_dim", sopt.get("context_dim", 48)) or 48)
    hidden_dim = int(policy_state.get("hidden_dim", scfg.get("hidden_dim", 64)) or scfg.get("hidden_dim", 64))
    num_heads = int(policy_state.get("num_heads", scfg.get("num_heads", 4)) or scfg.get("num_heads", 4))
    count_min = int(policy_state.get("count_min", sopt.get("count_min", len(tasks))) or len(tasks))
    count_max = int(policy_state.get("count_max", sopt.get("count_max", max(len(tasks), count_min))) or max(len(tasks), count_min))
    online_lr = float(sopt.get("online_lr", scfg.get("online_lr", 0.002)) or scfg.get("online_lr", 0.002))
    online_epochs = int(sopt.get("online_epochs", scfg.get("online_epochs", 8)) or scfg.get("online_epochs", 8))
    batch_size = int(sopt.get("online_batch_size", scfg.get("batch_size", 64)) or scfg.get("batch_size", 64))
    sample_target_dim_cfg = int(policy_state.get("sample_target_dim", sopt.get("sample_target_dim", scfg.get("sample_target_dim", 16))) or 16)
    count_loss_weight = float(sopt.get("count_loss_weight", scfg.get("count_loss_weight", 1.0)) or scfg.get("count_loss_weight", 1.0))
    slot_select_loss_weight = float(sopt.get("slot_select_loss_weight", scfg.get("slot_select_loss_weight", 0.8)) or scfg.get("slot_select_loss_weight", 0.8))
    mask_loss_weight = float(sopt.get("mask_loss_weight", scfg.get("mask_loss_weight", 0.8)) or scfg.get("mask_loss_weight", 0.8))
    value_loss_weight = float(sopt.get("value_loss_weight", scfg.get("value_loss_weight", 1.2)) or scfg.get("value_loss_weight", 1.2))
    score_loss_weight = float(sopt.get("score_loss_weight", scfg.get("score_loss_weight", 0.4)) or scfg.get("score_loss_weight", 0.4))
    sample_target_loss_weight = float(sopt.get("sample_target_loss_weight", scfg.get("sample_target_loss_weight", 0.3)) or scfg.get("sample_target_loss_weight", 0.3))

    spec = payload.get("design_spec", {}) if isinstance(payload.get("design_spec", {}), dict) else {}
    beta_defaults = payload.get("beta_defaults", {}) if isinstance(payload.get("beta_defaults", {}), dict) else {}
    beta_bounds = payload.get("beta_bounds", {}) if isinstance(payload.get("beta_bounds", {}), dict) else {}
    slot_specs, _alt_order = _extract_design_slots(spec)
    if not slot_specs:
        return {"updated": False, "reason": "no slot specs"}

    pop_stats = dyppo_shared._load_pop_stats(payload, data_dir=Path("."), config=config or {})
    feature_spec = dyppo_shared._build_feature_spec(pop_stats)
    sample_target_cells = policy_state.get("sample_target_cells", [])
    if not isinstance(sample_target_cells, list) or not sample_target_cells:
        sample_target_cells = _build_sample_target_cells(feature_spec, sample_target_dim_cfg)
    sample_cell_targets = policy_state.get("sample_cell_targets", {})
    if not isinstance(sample_cell_targets, dict) or not sample_cell_targets:
        sample_cell_targets = _build_sample_cell_targets(pop_stats, sample_target_cells, feature_spec)
    sample_target_dim = len(sample_target_cells)
    current_respondent = respondent if isinstance(respondent, dict) and respondent else {
        "respondent_id": "online_missing",
        "zone_id": _MISSING_ATTR_TOKEN,
        "attr_segments": [_MISSING_ATTR_TOKEN] * max(1, len(feature_spec.get("attr_dim_names", []))),
    }

    reward_metrics = dyppo_shared._questionnaire_reward_metrics(
        tasks,
        current_respondent,
        choices or {},
        spec=spec,
        beta_defaults=beta_defaults,
        beta_bounds=beta_bounds,
        prior_obs_rows=dyppo_shared._collect_obs_rows_from_submission_rows(list(historical_rows or []), spec),
        config=config,
    )

    labels = _block_to_labels(
        tasks,
        slot_specs=slot_specs,
        count_min=count_min,
        count_max=count_max,
        beta_defaults=beta_defaults,
        respondent=current_respondent,
    )
    tokens = _build_encoder_tokens(
        current_respondent,
        policy_state=policy_state,
        feature_spec=feature_spec,
        slot_specs=slot_specs,
        historical_rows=historical_rows,
        context_dim=context_dim,
    )
    dataset = {
        "encoder_tokens": np.expand_dims(tokens, axis=0),
        "count_labels": np.array([labels["count_label"]], dtype=np.int64),
        "slot_masks": np.expand_dims(labels["slot_mask"], axis=0),
        "mask_labels": np.expand_dims(labels["mask_labels"], axis=0),
        "value_labels": np.expand_dims(labels["value_labels"], axis=0),
        "score_labels": np.expand_dims(labels["score_labels"], axis=0),
        "sample_target_labels": np.expand_dims(
            _respondent_target_vector(current_respondent, sample_target_cells, feature_spec),
            axis=0,
        ),
    }

    seed = int(policy_state.get("response_count", 0) or 0) + 1707
    new_state, logs = _train_parallel_generator(
        dataset,
        context_dim=context_dim,
        slot_dim=len(slot_specs),
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        max_questions=count_max,
        count_classes=max(1, count_max - count_min + 1),
        sample_target_dim=sample_target_dim,
        seed=seed,
        epochs=online_epochs,
        lr=online_lr,
        batch_size=max(1, min(batch_size, 1)),
        count_loss_weight=count_loss_weight,
        slot_select_loss_weight=slot_select_loss_weight,
        mask_loss_weight=mask_loss_weight,
        value_loss_weight=value_loss_weight,
        score_loss_weight=score_loss_weight,
        sample_target_loss_weight=sample_target_loss_weight,
        init_state=policy_state.get("selfattention_state", {}),
        verbose=verbose,
        progress_prefix="selfattention/online",
    )

    policy_state["selfattention_state"] = new_state
    policy_state["online_updates"] = int(policy_state.get("online_updates", 0) or 0) + 1
    policy_state["response_count"] = int(policy_state.get("response_count", 0) or 0) + 1
    policy_state["sample_target_dim"] = int(sample_target_dim)
    policy_state["sample_target_cells"] = deepcopy(sample_target_cells)
    policy_state["sample_cell_targets"] = deepcopy(sample_cell_targets)
    policy_state["current_mnl_signal"] = {
        "n_observations_before": int(reward_metrics.get("n_observations_before", 0) or 0),
        "n_observations_after": int(reward_metrics.get("n_observations_after", 0) or 0),
        "adjusted_pseudo_r2_before": reward_metrics.get("adjusted_pseudo_r2_before"),
        "adjusted_pseudo_r2_after": reward_metrics.get("adjusted_pseudo_r2_after"),
        "delta_adjusted_pseudo_r2": reward_metrics.get("delta_adjusted_pseudo_r2"),
        "estimated_beta_raw": deepcopy(reward_metrics.get("estimated_beta_raw", {})),
        "estimated_beta": deepcopy(reward_metrics.get("estimated_beta", {})),
        "bound_violation": deepcopy(reward_metrics.get("bound_violation", {})),
    }

    last_log = logs[-1] if logs else {}
    return {
        "updated": True,
        "policy_version": int(policy_state.get("response_count", 0) or 0),
        "mean_episode_reward": round(float(reward_metrics.get("reward", 0.0) or 0.0), 6),
        "steps": len(tasks),
        "online_epochs": int(online_epochs),
        "loss": last_log.get("loss"),
        "approx_kl": None,
        "reward_metrics": {
            "components": deepcopy(reward_metrics.get("components", {})),
            "d_error": reward_metrics.get("d_error"),
            "entropy_mean": reward_metrics.get("entropy_mean"),
            "adjusted_pseudo_r2_before": reward_metrics.get("adjusted_pseudo_r2_before"),
            "adjusted_pseudo_r2_after": reward_metrics.get("adjusted_pseudo_r2_after"),
            "delta_adjusted_pseudo_r2": reward_metrics.get("delta_adjusted_pseudo_r2"),
            "bound_penalty": reward_metrics.get("bound_penalty"),
            "bound_violation": deepcopy(reward_metrics.get("bound_violation", {})),
            "estimated_beta_raw": deepcopy(reward_metrics.get("estimated_beta_raw", {})),
            "estimated_beta": deepcopy(reward_metrics.get("estimated_beta", {})),
        },
    }
