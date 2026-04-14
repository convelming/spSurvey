from __future__ import annotations
"""dynamicPPO 训练与推理辅助模块。

本模块实现了用于 SP 设计的 PPO-Clip + GAE 流程：
1) 从历史 rows 或合成 respondent 轨迹构造 rollout；
2) 使用共享 Actor-Critic 网络计算 old_log_prob / value；
3) 基于 GAE 计算 advantage 与 return；
4) 用 PPO-clip 目标训练策略；
5) 依据当前 respondent 与已选题组状态，逐题生成新的 SP task。
"""

import hashlib
from copy import deepcopy
from pathlib import Path

import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    optim = None
    Categorical = None
    TORCH_AVAILABLE = False

try:
    from .storage import load_json
except ImportError:
    from engine.storage import load_json

_MISSING_ATTR_TOKEN = "__missing__"


def _safe_float(v, default: float = 0.0) -> float:
    """尽力将输入转为 float，失败时返回默认值。

    参数:
        v: 任意输入值，通常来自表单、JSON 或配置项。
        default: 当 `v` 无法转换为浮点数时返回的默认值。

    返回:
        float: 转换成功后的浮点数，或 `default`。
    """
    try:
        return float(v)
    except Exception:
        return default


def _safe_opt_float(v) -> float | None:
    """尽力将输入转为有限浮点数，失败时返回 `None`。"""
    try:
        out = float(v)
        return out if np.isfinite(out) else None
    except Exception:
        return None


def _task_signature(task: dict) -> str:
    """为任务生成稳定短签名，用于去重与日志追踪。

    参数:
        task: 单个 SP 任务字典，预期包含 `alternatives` 字段。

    返回:
        str: 固定长度（16位）的 SHA1 前缀签名。
    """
    alts = (task or {}).get("alternatives", {}) if isinstance(task, dict) else {}
    norm = {}
    for alt, attrs in alts.items():
        norm[str(alt)] = {str(k): attrs.get(k) for k in sorted((attrs or {}).keys())}
    return hashlib.sha1(str(sorted(norm.items())).encode("utf-8")).hexdigest()[:16]


def _softmax(logits: np.ndarray) -> np.ndarray:
    """对一维 logits 进行数值稳定的 softmax。

    参数:
        logits: 一维实数向量。

    返回:
        np.ndarray: 与输入同长度的概率向量，和为 1。
    """
    z = logits - np.max(logits)
    e = np.exp(z)
    s = np.sum(e)
    if s <= 0:
        return np.ones_like(logits) / max(len(logits), 1)
    return e / s


if TORCH_AVAILABLE:
    class ActorCriticNet(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
            """初始化共享骨干网络的 Actor-Critic 模型。

            参数:
                input_dim: 输入特征维度。
                output_dim: 动作空间维度（候选任务数）。
                hidden_dim: 隐层维度。
            """
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            self.actor_head = nn.Linear(hidden_dim, output_dim)
            self.critic_head = nn.Linear(hidden_dim, 1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """前向传播。

            参数:
                x: 形状为 `[B, input_dim]` 的输入张量。

            返回:
                tuple[torch.Tensor, torch.Tensor]:
                    - logits: 形状 `[B, output_dim]` 的动作 logits。
                    - value: 形状 `[B]` 的状态价值估计。
            """
            h = self.backbone(x)
            logits = self.actor_head(h)
            value = self.critic_head(h).squeeze(-1)
            return logits, value


def _state_dict_to_json(sd: dict[str, torch.Tensor]) -> dict[str, list]:
    """将 torch 的 state_dict 张量转为可 JSON 序列化的 list。

    参数:
        sd: PyTorch 模型的参数字典。

    返回:
        dict[str, list]: 可直接写入 JSON 的参数字典。
    """
    if not TORCH_AVAILABLE:
        return {}
    out = {}
    for k, v in sd.items():
        out[k] = v.detach().cpu().numpy().tolist()
    return out


def _load_state_dict_from_json(model: nn.Module, raw: dict | None) -> bool:
    """若形状匹配，则将 JSON 序列化参数加载到模型中。

    参数:
        model: 目标模型实例。
        raw: JSON 反序列化后的参数字典。

    返回:
        bool: 成功加载返回 True，否则返回 False。
    """
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


def _load_pop_stats(payload: dict, *, data_dir: Path, config: dict | None = None) -> dict:
    """读取 config.json 中配置的人口分布（pop-sim）文件。

    参数:
        payload: 前端请求负载（当前版本中不直接使用，保留签名稳定性）。
        data_dir: 数据目录路径。
        config: 系统配置字典，包含 `data_sources.popsim_stats_path`。

    返回:
        dict: 解析后的人口统计分布字典。
    """
    _ = payload  # 保持函数签名稳定；文件路径由 config.json 决定
    cfg_ds = (config or {}).get("data_sources", {}) if isinstance((config or {}).get("data_sources", {}), dict) else {}
    p = str(cfg_ds.get("popsim_stats_path", "popSimStats_gz_template.json")).strip() or "popSimStats_gz_template.json"
    f = Path(p)
    if not f.is_absolute():
        f = data_dir / p
    return load_json(f, {})


def _reward_cfg(config: dict | None = None) -> dict:
    """读取在线奖励与参数越界控制相关配置。"""
    rcfg = (config or {}).get("online_reward", {}) if isinstance((config or {}).get("online_reward", {}), dict) else {}
    return {
        "weight_d_error": float(rcfg.get("weight_d_error", 1.0) or 1.0),
        "weight_entropy": float(rcfg.get("weight_entropy", 0.2) or 0.2),
        "weight_adj_pseudo_r2": float(rcfg.get("weight_adj_pseudo_r2", 2.0) or 2.0),
        "weight_param_bound_penalty": float(rcfg.get("weight_param_bound_penalty", 0.8) or 0.8),
        "mnl_min_observations": int(rcfg.get("mnl_min_observations", 20) or 20),
        "mnl_fit_epochs": int(rcfg.get("mnl_fit_epochs", 180) or 180),
        "mnl_fit_lr": float(rcfg.get("mnl_fit_lr", 0.08) or 0.08),
        "candidate_bound_mask_strength": float(rcfg.get("candidate_bound_mask_strength", 2.5) or 2.5),
        "candidate_bound_mask_floor": float(rcfg.get("candidate_bound_mask_floor", 0.05) or 0.05),
    }


def _zone_targets_obj(pop_stats: dict | None) -> dict:
    """兼容读取 `zone_targets` / `zone2_targets` 两种写法。

    历史文件里既出现过 `zone_targets`，也出现过 `zone2_targets`。
    这里统一折叠为一个分区目标映射，供后续采样与 one-hot 建模使用。
    """
    if not isinstance(pop_stats, dict):
        return {}
    z1 = pop_stats.get("zone_targets", {})
    if isinstance(z1, dict) and z1:
        return z1
    z2 = pop_stats.get("zone2_targets", {})
    if isinstance(z2, dict) and z2:
        return z2
    return {}


def _infer_attr_width(pop_stats: dict | None) -> int:
    """推断属性键的标准列数。

    优先使用 `key_format` 指定的列数；若未提供，则扫描
    `default_target/zone_targets` 中出现过的最大分段数。

    参数:
        pop_stats: PopulationSim 风格分布字典。

    返回:
        int: 期望的属性分段列数，最少为 1。
    """
    if not isinstance(pop_stats, dict):
        return 1
    key_format = str(pop_stats.get("key_format", "") or "").strip()
    if key_format:
        return max(1, len(key_format.split("|")))
    width = 1
    default_target = pop_stats.get("default_target", {}) if isinstance(pop_stats.get("default_target", {}), dict) else {}
    zone_targets = _zone_targets_obj(pop_stats)
    for k in list(default_target.keys()):
        width = max(width, len(str(k or "").split("|")))
    for zv in zone_targets.values():
        if isinstance(zv, dict):
            for k in zv.keys():
                width = max(width, len(str(k or "").split("|")))
    return max(1, width)


def _parse_attr_key(k: str, expected_len: int | None = None) -> tuple[str, ...]:
    """将属性键解析为保留空位的可变长度分段。

    示例：
    - "male|18-30|college" -> ("male","18-30","college")
    - "male||college" -> ("male","__missing__","college")
    - "a|b|c|d|e" -> ("a","b","c","d","e")
    - 空值/非法值 -> ("unknown",) 或按 expected_len 补齐缺失列
    参数:
        k: 原始属性键字符串，通常由 `|` 分隔。
        expected_len: 期望列数；若提供，则会保留列位置并补齐到指定长度。

    返回:
        tuple[str, ...]: 分段后的属性元组；中间空字段不会被压缩。
    """
    raw = str(k or "")
    parts = [str(p).strip() for p in raw.split("|")]
    if expected_len is not None:
        if len(parts) < expected_len:
            parts.extend([""] * (expected_len - len(parts)))
        elif len(parts) > expected_len:
            parts = parts[:expected_len]
    norm = [p if p != "" else _MISSING_ATTR_TOKEN for p in parts]
    if any(p != _MISSING_ATTR_TOKEN for p in norm):
        return tuple(norm)
    if expected_len is not None and expected_len > 1:
        return tuple([_MISSING_ATTR_TOKEN] * expected_len)
    return ("unknown",)


def _collect_attr_categories(pop_stats: dict | None) -> list[list[str]]:
    """按列收集 PopulationSim 属性键的所有离散类别。

    该函数会保留空列位置，并把空值统一映射为 `__missing__`，
    便于后续做 one-hot / ordinal 编码时按列建字典。

    参数:
        pop_stats: PopulationSim 风格分布字典。

    返回:
        list[list[str]]: 每一列的去重类别列表，按列序返回。
    """
    width = _infer_attr_width(pop_stats)
    cats = [set() for _ in range(width)]
    if not isinstance(pop_stats, dict):
        return [[] for _ in range(width)]
    default_target = pop_stats.get("default_target", {}) if isinstance(pop_stats.get("default_target", {}), dict) else {}
    zone_targets = _zone_targets_obj(pop_stats)
    all_keys = list(default_target.keys())
    for zv in zone_targets.values():
        if isinstance(zv, dict):
            all_keys.extend(zv.keys())
    for k in all_keys:
        parts = _parse_attr_key(str(k or ""), expected_len=width)
        for idx, part in enumerate(parts):
            cats[idx].add(str(part))
    return [sorted(list(x)) for x in cats]


def _attr_dim_names(pop_stats: dict | None) -> list[str]:
    """获取属性键各列的名称。"""
    width = _infer_attr_width(pop_stats)
    if isinstance(pop_stats, dict):
        key_format = str(pop_stats.get("key_format", "") or "").strip()
        if key_format:
            names = [str(x).strip() or f"attr_{i}" for i, x in enumerate(key_format.split("|"))]
            if len(names) < width:
                names.extend([f"attr_{i}" for i in range(len(names), width)])
            return names[:width]
    return [f"attr_{i}" for i in range(width)]


def _zone_categories(pop_stats: dict | None) -> list[str]:
    """收集 zone_id 的离散类别。"""
    if not isinstance(pop_stats, dict):
        return [_MISSING_ATTR_TOKEN]
    zone_targets = _zone_targets_obj(pop_stats)
    zones = sorted([str(z) for z in zone_targets.keys()])
    if _MISSING_ATTR_TOKEN not in zones:
        zones.append(_MISSING_ATTR_TOKEN)
    return zones


def _build_feature_spec(pop_stats: dict | None) -> dict:
    """根据 PopSim 配置构建 respondent 编码规格。"""
    attr_categories = _collect_attr_categories(pop_stats)
    dim_names = _attr_dim_names(pop_stats)
    zone_categories = _zone_categories(pop_stats)
    respondent_dim = int(sum(len(x) for x in attr_categories) + len(zone_categories))
    return {
        "attr_dim_names": dim_names,
        "attr_categories": attr_categories,
        "zone_categories": zone_categories,
        "respondent_dim": respondent_dim,
        # 额外 4 维是 PPO 的动态状态摘要：
        # 1) 已选题平均 spread_hint
        # 2) 已选题平均 entropy
        # 3) 当前题位进度 progress
        # 4) 已选题占目标题数比例 selected_ratio
        "feature_dim": respondent_dim + 4,
    }


def _zone_dist(zone_value, default_target: dict) -> dict[str, float]:
    """归一化分区目标分布的读取逻辑。

    参数:
        zone_value: 某个 zone 的目标分布，可能为 dict、特殊标记字符串等。
        default_target: 默认目标分布字典。

    返回:
        dict[str, float]: 可用于采样的分布映射（值已转浮点）。
    """
    if isinstance(zone_value, dict):
        return {str(k): _safe_float(v, 0.0) for k, v in zone_value.items()}
    if isinstance(zone_value, str) and zone_value == "__copy_from_default_target__":
        return {str(k): _safe_float(v, 0.0) for k, v in (default_target or {}).items()}
    return {str(k): _safe_float(v, 0.0) for k, v in (default_target or {}).items()}


def _sample_respondents(pop_stats: dict, n: int, seed: int) -> list[dict]:
    """按分区目标分布采样合成受访者。

    参数:
        pop_stats: 人口分布配置字典（zone_targets/default_target）。
        n: 采样数量。
        seed: 随机种子。

    返回:
        list[dict]: 合成受访者列表，每个元素包含 `respondent_id/zone_id/attr_segments`。
    """
    rng = np.random.default_rng(seed)
    zone_targets = _zone_targets_obj(pop_stats)
    default_target = pop_stats.get("default_target", {}) if isinstance(pop_stats, dict) else {}
    zone_ids = sorted(zone_targets.keys()) if isinstance(zone_targets, dict) and zone_targets else ["default"]
    expected_len = _infer_attr_width(pop_stats)
    if not zone_ids:
        zone_ids = ["default"]

    respondents = []
    for i in range(max(1, n)):
        zid = str(zone_ids[int(rng.integers(0, len(zone_ids)))])
        dist = _zone_dist((zone_targets or {}).get(zid), default_target)
        keys = [k for k, v in dist.items() if v > 0]
        vals = [dist[k] for k in keys]
        if not keys:
            keys = [k for k in (default_target or {}).keys()]
            vals = [_safe_float(default_target.get(k), 0.0) for k in keys]
        if not keys:
            keys = ["male|31-45|college_plus"]
            vals = [1.0]
        prob = np.array(vals, dtype=float)
        prob = prob / np.sum(prob)
        picked = keys[int(rng.choice(len(keys), p=prob))]
        parts = _parse_attr_key(picked, expected_len=expected_len)
        respondents.append(
            {
                "respondent_id": f"sim_{i+1:04d}",
                "zone_id": zid,
                "attr_segments": list(parts),
            }
        )
    return respondents


def _respondent_feat(r: dict, *, feature_spec: dict | None = None) -> np.ndarray:
    """将受访者编码为按列 one-hot 的数值特征。

    参数:
        r: 单个受访者字典，包含 `attr_segments` 和 `zone_id`。
        feature_spec: 由 `_build_feature_spec` 生成的编码规格。

    返回:
        np.ndarray: 受访者 one-hot 特征向量。
    """
    parts = r.get("attr_segments", None)
    if not isinstance(parts, list) or not parts:
        parts = []
        for k in sorted([k for k in r.keys() if str(k).startswith("attr_")]):
            parts.append(str(r.get(k)))
    spec = feature_spec if isinstance(feature_spec, dict) else {}
    attr_categories = spec.get("attr_categories", []) if isinstance(spec.get("attr_categories", []), list) else []
    zone_categories = spec.get("zone_categories", []) if isinstance(spec.get("zone_categories", []), list) else []
    width = len(attr_categories) if attr_categories else len(parts)
    parsed = _parse_attr_key("|".join([str(x) for x in parts]), expected_len=max(1, width))
    feats: list[float] = []
    if attr_categories:
        for idx, cats in enumerate(attr_categories):
            cur = str(parsed[idx]) if idx < len(parsed) else _MISSING_ATTR_TOKEN
            if cur not in cats and _MISSING_ATTR_TOKEN in cats:
                cur = _MISSING_ATTR_TOKEN
            feats.extend([1.0 if cur == str(cat) else 0.0 for cat in cats])
    else:
        feats.extend([1.0 if str(x) != _MISSING_ATTR_TOKEN else 0.0 for x in parsed])
    zid = str(r.get("zone_id", "") or _MISSING_ATTR_TOKEN)
    if zone_categories:
        if zid not in zone_categories and _MISSING_ATTR_TOKEN in zone_categories:
            zid = _MISSING_ATTR_TOKEN
        feats.extend([1.0 if zid == str(z) else 0.0 for z in zone_categories])
    return np.array(feats, dtype=float)


def _utility_for_alt(alt_name: str, attrs: dict, respondent: dict, beta_defaults: dict) -> float:
    """用于生成伪标签的合成效用函数。

    完全通用，不写死任何备选项名称（如 car/pt）。
    效用由两部分组成：
    1) 属性线性项：beta_defaults["<alt>.<var>"] * value
    2) 可选的人群分段交互项：
       beta_defaults["seg<idx>.<segment_value>.<alt>"]
       例如：seg0.female.bus = 0.2
    参数:
        alt_name: 备选项名称。
        attrs: 该备选项的属性-水平字典。
        respondent: 单个受访者字典（含 `attr_segments`）。
        beta_defaults: 先验系数字典。

    返回:
        float: 该备选项的合成效用值。
    """
    u = 0.0
    for k, v in (attrs or {}).items():
        key = f"{alt_name}.{k}"
        u += _safe_float(beta_defaults.get(key), 0.0) * _safe_float(v, 0.0)
    parts = respondent.get("attr_segments", []) if isinstance(respondent.get("attr_segments", []), list) else []
    for idx, seg in enumerate(parts):
        ikey = f"seg{idx}.{seg}.{alt_name}"
        u += _safe_float(beta_defaults.get(ikey), 0.0)
    return float(u)


def _task_meta_features(task: dict, beta_defaults: dict, respondent: dict) -> tuple[float, str]:
    """返回单个任务的代理效用分数与最佳选项。

    参数:
        task: 单个任务字典。
        beta_defaults: 先验系数字典。
        respondent: 受访者字典。

    返回:
        tuple[float, str]:
            - float: 任务最大效用值（代理分数）。
            - str: 最大效用对应的备选项名称（并列时随机打破平局）。
    """
    alts = (task or {}).get("alternatives", {}) or {}
    if not alts:
        return 0.0, ""
    util_items = []
    for alt_name, attrs in alts.items():
        util_items.append((str(alt_name), _utility_for_alt(str(alt_name), attrs or {}, respondent, beta_defaults)))
    max_u = max(v for _, v in util_items)
    best_alts = [n for n, v in util_items if abs(v - max_u) <= 1e-9]
    # 当多个备选项并列最高时，随机打破平局。
    pick = best_alts[int(np.random.default_rng().integers(0, len(best_alts)))]
    return float(max_u), str(pick)


def _task_choice_stats(task: dict, beta_defaults: dict, respondent: dict, chosen_alt: str | None = None) -> dict:
    """计算单个 task 在给定 respondent 下的代理选择统计。

    返回内容包括：
    - `spread`: 备选项最大效用差
    - `entropy_norm`: 选择概率分布的归一化熵
    - `chosen_prob`: 若给定了实际选择项，则返回该项的代理概率
    """
    alts = (task or {}).get("alternatives", {}) or {}
    if not alts:
        return {"spread": 0.0, "entropy_norm": 0.0, "chosen_prob": 0.0}
    alt_names = []
    utils = []
    for alt_name, attrs in alts.items():
        alt_names.append(str(alt_name))
        utils.append(_utility_for_alt(str(alt_name), attrs or {}, respondent, beta_defaults))
    u = np.array(utils, dtype=float)
    p = _softmax(u)
    spread = float(np.max(u) - np.min(u)) if u.size else 0.0
    e = float(-np.sum(p * np.log(np.maximum(p, 1e-12)))) if p.size else 0.0
    e_max = float(np.log(max(len(alt_names), 1))) if alt_names else 0.0
    entropy_norm = float(e / e_max) if e_max > 0 else 0.0
    chosen_prob = 0.0
    if chosen_alt is not None and str(chosen_alt) in alt_names:
        chosen_prob = float(p[alt_names.index(str(chosen_alt))])
    return {
        "spread": spread,
        "entropy_norm": entropy_norm,
        "chosen_prob": chosen_prob,
    }


def _task_state_vector(
    respondent: dict,
    prior_tasks: list[dict] | None = None,
    *,
    beta_defaults: dict,
    feature_spec: dict | None = None,
    step_index: int = 0,
    episode_len: int = 1,
    input_dim: int,
) -> np.ndarray:
    """构造策略在第 `t` 步做题目选择前的状态向量 `s_t`。

    这里不把“当前将要选择的 task 属性”直接塞进状态，
    因为那会和“action=选择哪个 task”产生语义循环。
    当前状态只描述：
    1) 当前 respondent 的 RP / 分区 one-hot 特征；
    2) 之前已经选出的题组统计摘要；
    3) 当前题位进度。

    参数:
        respondent: 当前 respondent 的离散属性快照。
        prior_tasks: 在当前 episode 中，已经选出的历史 task 列表。
        beta_defaults: 代理效用先验，用于把历史 task 变成统计摘要。
        feature_spec: respondent 编码规格。
        step_index: 当前是本问卷的第几题，从 0 开始。
        episode_len: 计划总题数。
        input_dim: 最终网络输入维度。

    返回:
        np.ndarray: 长度固定为 `input_dim` 的状态向量。
    """
    rf = _respondent_feat(respondent, feature_spec=feature_spec)
    history = list(prior_tasks or [])
    if history:
        spread_vals = []
        entropy_vals = []
        for task in history:
            stats = _task_choice_stats(task, beta_defaults, respondent, chosen_alt=None)
            spread_vals.append(float(stats.get("spread", 0.0) or 0.0))
            entropy_vals.append(float(stats.get("entropy_norm", 0.0) or 0.0))
        mean_spread = float(np.mean(spread_vals)) if spread_vals else 0.0
        mean_entropy = float(np.mean(entropy_vals)) if entropy_vals else 0.0
    else:
        mean_spread = 0.0
        mean_entropy = 0.0
    spread_hint = 1.0 / (1.0 + abs(mean_spread))
    denom = max(int(episode_len) - 1, 1)
    progress = float(step_index) / float(denom)
    selected_ratio = float(len(history)) / float(max(int(episode_len), 1))
    x = np.concatenate([rf, np.array([spread_hint, mean_entropy, progress, selected_ratio], dtype=float)])
    if len(x) < input_dim:
        x = np.concatenate([x, np.zeros((input_dim - len(x),), dtype=float)])
    elif len(x) > input_dim:
        x = x[:input_dim]
    return x


def _param_keys_for_spec(spec: dict) -> tuple[list[str], list[str], str]:
    """从 design_spec 中抽取参数键顺序与备选项顺序。"""
    alts = spec.get("alternatives", []) if isinstance(spec, dict) else []
    alt_names = [str((a or {}).get("name", "")).strip() for a in alts]
    alt_names = [x for x in alt_names if x]
    base = str(spec.get("asc_base_alternative", "")).strip() or (alt_names[0] if alt_names else "")
    keys: list[str] = []
    for a in alt_names:
        if a != base:
            keys.append(f"{a}.asc")
    for alt in alts:
        a = str((alt or {}).get("name", "")).strip()
        if not a:
            continue
        for var in (alt or {}).get("variables", []) or []:
            v = str((var or {}).get("name", "")).strip()
            if v:
                keys.append(f"{a}.{v}")
    return keys, alt_names, base


def _obs_to_design_rows(task: dict, alt_names: list[str], keys: list[str]) -> np.ndarray:
    """把单题 task 转成 MNL 估计所需的 `[J,K]` 设计矩阵。"""
    alts = (task or {}).get("alternatives", {}) if isinstance(task, dict) else {}
    rows = []
    for a in alt_names:
        attrs = alts.get(a, {}) if isinstance(alts.get(a, {}), dict) else {}
        row = []
        for key in keys:
            if key.endswith(".asc"):
                row.append(1.0 if key == f"{a}.asc" else 0.0)
                continue
            aa, vv = key.split(".", 1)
            row.append(float(attrs.get(vv, 0.0)) if aa == a else 0.0)
        rows.append(row)
    return np.array(rows, dtype=float)


def _build_obs_rows_from_tasks(tasks: list[dict], choices: dict, spec: dict) -> list[dict]:
    """把一份 block 的 tasks + 选择结果转换为 MNL 观测行。"""
    keys, alt_names, _base = _param_keys_for_spec(spec)
    if not keys or not alt_names:
        return []
    out: list[dict] = []
    for task in tasks or []:
        if not isinstance(task, dict):
            continue
        task_id = str((task or {}).get("id", "")).strip()
        chosen = str((choices or {}).get(task_id, "")).strip()
        if chosen not in alt_names:
            continue
        out.append({"x": _obs_to_design_rows(task, alt_names, keys), "y": alt_names.index(chosen)})
    return out


def _collect_obs_rows_from_submission_rows(rows: list[dict], spec: dict) -> list[dict]:
    """把历史 submission/row 列表展开成 MNL 观测行。"""
    out: list[dict] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        tasks = row.get("tasks", []) if isinstance(row.get("tasks", []), list) else []
        choices = row.get("choices", {}) if isinstance(row.get("choices", {}), dict) else {}
        out.extend(_build_obs_rows_from_tasks(tasks, choices, spec))
    return out


def _clip_beta_to_bounds(beta: np.ndarray, keys: list[str], beta_bounds: dict | None) -> np.ndarray:
    """按 beta_bounds 对参数做逐维裁剪。"""
    clipped = np.array(beta, dtype=float).copy()
    if not isinstance(beta_bounds, dict):
        return clipped
    for i, key in enumerate(keys):
        bounds = beta_bounds.get(key, {}) if isinstance(beta_bounds.get(key, {}), dict) else {}
        lo = _safe_opt_float(bounds.get("min"))
        hi = _safe_opt_float(bounds.get("max"))
        if lo is not None:
            clipped[i] = max(clipped[i], lo)
        if hi is not None:
            clipped[i] = min(clipped[i], hi)
    return clipped


def _estimate_mnl(
    obs_rows: list[dict],
    keys: list[str],
    alt_names: list[str],
    beta0: np.ndarray,
    *,
    beta_bounds: dict | None = None,
    epochs: int = 200,
    lr: float = 0.08,
    reg: float = 1e-4,
    clip_to_bounds: bool = True,
) -> np.ndarray:
    """使用简单梯度上升近似重估 MNL 参数。"""
    _ = alt_names
    if not obs_rows or not keys:
        return beta0.copy()
    beta = beta0.astype(float).copy()
    n = len(obs_rows)
    for _epoch in range(max(1, int(epochs))):
        grad = np.zeros_like(beta)
        for obs in obs_rows:
            x = np.asarray(obs["x"], dtype=float)
            y = int(obs["y"])
            u = x @ beta
            u = u - np.max(u)
            p = np.exp(u)
            denom = np.sum(p)
            if denom <= 0:
                continue
            p = p / denom
            grad += x[y] - (p.reshape(-1, 1) * x).sum(axis=0)
        step = float(lr) * (grad / max(n, 1) - float(reg) * beta)
        beta = beta + step
        if clip_to_bounds:
            beta = _clip_beta_to_bounds(beta, keys, beta_bounds)
        if float(np.linalg.norm(step)) < 1e-6:
            break
    return beta


def _mnl_log_likelihood(obs_rows: list[dict], beta: np.ndarray) -> float:
    """计算给定参数下的 MNL 对数似然。"""
    if not obs_rows:
        return 0.0
    ll = 0.0
    for obs in obs_rows:
        x = np.asarray(obs["x"], dtype=float)
        y = int(obs["y"])
        u = x @ beta
        u = u - np.max(u)
        exp_u = np.exp(u)
        denom = float(np.sum(exp_u))
        if denom <= 0 or y < 0 or y >= len(exp_u):
            continue
        prob = max(float(exp_u[y]) / denom, 1e-12)
        ll += np.log(prob)
    return float(ll)


def _adjusted_mcfadden_pseudo_r2(obs_rows: list[dict], beta: np.ndarray, n_params: int) -> float | None:
    """计算 adjusted McFadden pseudo R^2。"""
    if not obs_rows:
        return None
    ll_hat = _mnl_log_likelihood(obs_rows, beta)
    ll_null = 0.0
    for obs in obs_rows:
        x = np.asarray(obs["x"], dtype=float)
        n_alt = int(x.shape[0]) if x.ndim == 2 else 0
        if n_alt <= 0:
            continue
        ll_null += -np.log(float(n_alt))
    if abs(ll_null) <= 1e-12:
        return None
    return float(1.0 - ((ll_hat - float(max(n_params, 0))) / ll_null))


def _beta_bound_violation_summary(
    beta_raw: np.ndarray,
    keys: list[str],
    beta_bounds: dict | None,
    *,
    mask_strength: float,
    mask_floor: float,
) -> dict:
    """汇总参数估计越界幅度，并给出下一轮候选题的降权因子。"""
    if not isinstance(beta_bounds, dict) or not keys:
        return {
            "penalty": 0.0,
            "violating_count": 0,
            "violating_params": [],
            "param_multipliers": {},
        }

    violating = []
    param_multipliers: dict[str, float] = {}
    total_severity = 0.0
    for i, key in enumerate(keys):
        bounds = beta_bounds.get(key, {}) if isinstance(beta_bounds.get(key, {}), dict) else {}
        lo = _safe_opt_float(bounds.get("min"))
        hi = _safe_opt_float(bounds.get("max"))
        if lo is None and hi is None:
            continue
        val = float(beta_raw[i])
        gap = 0.0
        side = None
        if lo is not None and val < lo:
            gap = float(lo - val)
            side = "below_min"
        elif hi is not None and val > hi:
            gap = float(val - hi)
            side = "above_max"
        if gap <= 0:
            continue
        scale = None
        if lo is not None and hi is not None and hi > lo:
            scale = float(hi - lo)
        if scale is None or scale <= 1e-9:
            scale = max(abs(lo or 0.0), abs(hi or 0.0), 1.0)
        severity = float(gap / max(scale, 1e-9))
        total_severity += severity
        multiplier = max(float(mask_floor), float(np.exp(-float(mask_strength) * severity)))
        param_multipliers[key] = float(multiplier)
        violating.append(
            {
                "key": key,
                "value_raw": round(val, 6),
                "min": lo,
                "max": hi,
                "gap": round(gap, 6),
                "severity": round(severity, 6),
                "side": side,
                "candidate_multiplier": round(float(multiplier), 6),
            }
        )
    return {
        "penalty": float(total_severity / max(len(keys), 1)),
        "violating_count": int(len(violating)),
        "violating_params": violating,
        "param_multipliers": param_multipliers,
    }


def _mnl_signal_from_obs_rows(
    obs_rows: list[dict],
    *,
    spec: dict,
    beta_defaults: dict,
    beta_bounds: dict,
    config: dict | None = None,
) -> dict:
    """从一批观测行中估计 MNL 参数、adjusted pseudo R^2 与越界信息。"""
    keys, alt_names, _base = _param_keys_for_spec(spec)
    beta0 = np.array([float(beta_defaults.get(k, 0.0)) for k in keys], dtype=float)
    cfg = _reward_cfg(config)
    min_obs = max(1, int(cfg.get("mnl_min_observations", 20) or 20))
    if not keys or len(obs_rows or []) < min_obs:
        return {
            "ready": False,
            "n_observations": int(len(obs_rows or [])),
            "param_keys": keys,
            "estimated_beta_raw": {},
            "estimated_beta": {},
            "adjusted_pseudo_r2": None,
            "bound_violation": {
                "penalty": 0.0,
                "violating_count": 0,
                "violating_params": [],
                "param_multipliers": {},
            },
        }

    beta_raw = _estimate_mnl(
        obs_rows,
        keys,
        alt_names,
        beta0,
        beta_bounds=beta_bounds,
        epochs=int(cfg.get("mnl_fit_epochs", 180) or 180),
        lr=float(cfg.get("mnl_fit_lr", 0.08) or 0.08),
        clip_to_bounds=False,
    )
    beta_clipped = _clip_beta_to_bounds(beta_raw, keys, beta_bounds)
    adj_pseudo_r2 = _adjusted_mcfadden_pseudo_r2(obs_rows, beta_clipped, len(keys))
    bound_violation = _beta_bound_violation_summary(
        beta_raw,
        keys,
        beta_bounds,
        mask_strength=float(cfg.get("candidate_bound_mask_strength", 2.5) or 2.5),
        mask_floor=float(cfg.get("candidate_bound_mask_floor", 0.05) or 0.05),
    )
    return {
        "ready": True,
        "n_observations": int(len(obs_rows or [])),
        "param_keys": keys,
        "estimated_beta_raw": {k: round(float(beta_raw[i]), 6) for i, k in enumerate(keys)},
        "estimated_beta": {k: round(float(beta_clipped[i]), 6) for i, k in enumerate(keys)},
        "adjusted_pseudo_r2": round(float(adj_pseudo_r2), 6) if adj_pseudo_r2 is not None else None,
        "bound_violation": bound_violation,
    }


def _task_param_keys(task: dict) -> list[str]:
    """抽取单题中出现过的 `alt.var` 参数键。"""
    out: list[str] = []
    alts = (task or {}).get("alternatives", {}) if isinstance(task, dict) else {}
    for alt_name, attrs in alts.items():
        if not isinstance(attrs, dict):
            continue
        for var_name in attrs.keys():
            out.append(f"{alt_name}.{var_name}")
    return out


def _apply_candidate_bound_mask(probs: np.ndarray, candidates: list[dict], bound_mask_state: dict | None) -> tuple[np.ndarray, dict]:
    """按越界参数的抑制因子，对包含相关变量的候选题整体降权。"""
    base = np.array(probs, dtype=float)
    state = bound_mask_state if isinstance(bound_mask_state, dict) else {}
    param_multipliers = state.get("param_multipliers", {}) if isinstance(state.get("param_multipliers", {}), dict) else {}
    if base.size == 0 or not param_multipliers:
        return base, {"active_bound_mask_params": 0, "bound_masked_candidates": 0, "bound_mask_min_factor": 1.0}

    factors = []
    masked_count = 0
    min_factor = 1.0
    for task in candidates:
        keys = _task_param_keys(task)
        task_factor = 1.0
        for key in keys:
            if key in param_multipliers:
                task_factor = min(task_factor, float(param_multipliers[key]))
        if task_factor < 0.999999:
            masked_count += 1
            min_factor = min(min_factor, task_factor)
        factors.append(task_factor)

    weighted = base * np.asarray(factors, dtype=float)
    if float(np.sum(weighted)) > 0:
        weighted = weighted / np.sum(weighted)
    return weighted, {
        "active_bound_mask_params": int(len(param_multipliers)),
        "bound_masked_candidates": int(masked_count),
        "bound_mask_min_factor": round(float(min_factor), 6) if masked_count else 1.0,
    }


def _questionnaire_reward_metrics(
    tasks: list[dict],
    respondent: dict,
    choices: dict,
    *,
    spec: dict,
    beta_defaults: dict,
    beta_bounds: dict,
    prior_obs_rows: list[dict] | None = None,
    config: dict | None = None,
) -> dict:
    """计算整份问卷的终端奖励与组成项。"""
    cfg = _reward_cfg(config)
    d_err = None
    try:
        from app import _bayesian_d_error_generic as _d_error_fn

        d_err = _d_error_fn(tasks, spec, beta_defaults, beta_bounds, beta_draws=8, seed=17)
    except Exception:
        d_err = None
    d_err = float(d_err) if d_err is not None else 1.0
    reward_d_raw = 1.0 / (1.0 + max(float(d_err), 1e-6))

    entropy_vals = []
    for task in tasks:
        task_id = str((task or {}).get("id", "")).strip()
        chosen_alt = choices.get(task_id) if isinstance(choices, dict) else None
        stats = _task_choice_stats(task, beta_defaults, respondent, chosen_alt=chosen_alt)
        entropy_vals.append(float(stats.get("entropy_norm", 0.0) or 0.0))
    reward_h_raw = float(np.mean(entropy_vals)) if entropy_vals else 0.0

    current_obs_rows = _build_obs_rows_from_tasks(tasks, choices if isinstance(choices, dict) else {}, spec)
    prev_obs_rows = list(prior_obs_rows or [])
    before_signal = _mnl_signal_from_obs_rows(
        prev_obs_rows,
        spec=spec,
        beta_defaults=beta_defaults,
        beta_bounds=beta_bounds,
        config=config,
    )
    after_signal = _mnl_signal_from_obs_rows(
        prev_obs_rows + current_obs_rows,
        spec=spec,
        beta_defaults=beta_defaults,
        beta_bounds=beta_bounds,
        config=config,
    )

    adj_before = before_signal.get("adjusted_pseudo_r2")
    adj_after = after_signal.get("adjusted_pseudo_r2")
    delta_adj = 0.0
    if adj_after is not None:
        delta_adj = float(adj_after) - float(adj_before or 0.0)

    bound_penalty_raw = float(((after_signal.get("bound_violation", {}) or {}).get("penalty", 0.0) or 0.0))

    reward_d = float(cfg.get("weight_d_error", 1.0)) * reward_d_raw
    reward_h = float(cfg.get("weight_entropy", 0.2)) * reward_h_raw
    reward_adj = float(cfg.get("weight_adj_pseudo_r2", 2.0)) * float(delta_adj)
    reward_bound = -float(cfg.get("weight_param_bound_penalty", 0.8)) * bound_penalty_raw
    reward_total = float(reward_d + reward_h + reward_adj + reward_bound)

    return {
        "reward": reward_total,
        "components": {
            "d_error_reward": round(float(reward_d), 6),
            "entropy_reward": round(float(reward_h), 6),
            "adj_pseudo_r2_reward": round(float(reward_adj), 6),
            "bound_penalty_reward": round(float(reward_bound), 6),
        },
        "d_error": round(float(d_err), 6),
        "entropy_mean": round(float(reward_h_raw), 6),
        "adjusted_pseudo_r2_before": adj_before,
        "adjusted_pseudo_r2_after": adj_after,
        "delta_adjusted_pseudo_r2": round(float(delta_adj), 6),
        "bound_penalty": round(float(bound_penalty_raw), 6),
        "bound_violation": deepcopy(after_signal.get("bound_violation", {})),
        "estimated_beta_raw": deepcopy(after_signal.get("estimated_beta_raw", {})),
        "estimated_beta": deepcopy(after_signal.get("estimated_beta", {})),
        "current_obs_rows": current_obs_rows,
        "n_observations_before": int(before_signal.get("n_observations", 0) or 0),
        "n_observations_after": int(after_signal.get("n_observations", 0) or 0),
    }


def _questionnaire_reward(
    tasks: list[dict],
    respondent: dict,
    choices: dict,
    *,
    spec: dict,
    beta_defaults: dict,
    beta_bounds: dict,
    prior_obs_rows: list[dict] | None = None,
    config: dict | None = None,
) -> float:
    """为一整份已完成问卷构造 episode 级奖励。

    当前整份问卷级奖励由四部分组成：
    1) `reward_d`: D-error 信息性（越小越好）
    2) `reward_h`: 题内选择熵（越均衡越好）
    3) `reward_adj`: 当前 block 加入后 adjusted pseudo R^2 的增量
    4) `reward_bound`: 参数估计超出预定义区间时的惩罚
    """
    metrics = _questionnaire_reward_metrics(
        tasks,
        respondent,
        choices,
        spec=spec,
        beta_defaults=beta_defaults,
        beta_bounds=beta_bounds,
        prior_obs_rows=prior_obs_rows,
        config=config,
    )
    return float(metrics.get("reward", 0.0) or 0.0)


def _row_respondent(row: dict) -> dict | None:
    """从逐行训练数据中提取 respondent 特征快照。

    当前优先支持测试脚本使用的结构：
    - row["respondent"] = {respondent_id, zone_id, attr_segments, ...}

    若未来实际入库行也补充了这一结构，则可直接复用。
    """
    if not isinstance(row, dict):
        return None
    respondent = row.get("respondent", {})
    if isinstance(respondent, dict) and respondent:
        return {
            "respondent_id": respondent.get("respondent_id"),
            "zone_id": respondent.get("zone_id"),
            "attr_segments": list((respondent or {}).get("attr_segments", []) or []),
        }
    return None


def _simulate_choice_for_task(
    task: dict,
    respondent: dict,
    beta_defaults: dict,
    *,
    seed_key: str = "",
) -> str:
    """按代理效用为一个 task 模拟 respondent 的选择项。"""
    alts = (task or {}).get("alternatives", {}) or {}
    if not alts:
        return ""
    util_items = []
    for alt_name, attrs in alts.items():
        util_items.append((str(alt_name), _utility_for_alt(str(alt_name), attrs or {}, respondent, beta_defaults)))
    max_u = max(v for _, v in util_items)
    best_alts = [name for name, val in util_items if abs(val - max_u) <= 1e-9]
    if len(best_alts) == 1:
        return best_alts[0]
    digest = hashlib.sha1(str(seed_key or _task_signature(task)).encode("utf-8")).hexdigest()[:12]
    idx = int(digest, 16) % len(best_alts)
    return str(best_alts[idx])


def _task_step_reward(task: dict, respondent: dict, chosen_alt: str, *, beta_defaults: dict, terminal_bonus: float = 0.0) -> float:
    """构造单步 reward。

    设计思路：
    1) `chosen_prob` 越高，说明代理模型下 respondent 对当前题更有明确反馈；
    2) `entropy_norm` 越高，说明题目没有过度压倒性主导；
    3) `spread_bonus` 控制过大 utility gap；
    4) `terminal_bonus` 在最后一步追加整份问卷级信息量奖励（D-error 等）。
    """
    stats = _task_choice_stats(task, beta_defaults, respondent, chosen_alt=chosen_alt)
    chosen_prob = float(stats.get("chosen_prob", 0.0) or 0.0)
    entropy_norm = float(stats.get("entropy_norm", 0.0) or 0.0)
    spread = float(stats.get("spread", 0.0) or 0.0)
    spread_bonus = 1.0 / (1.0 + abs(spread))
    return float(0.30 * chosen_prob + 0.15 * entropy_norm + 0.10 * spread_bonus + terminal_bonus)


def _empty_rollouts(input_dim: int) -> dict:
    """返回一个空 rollout 容器，便于统一后续训练入口。"""
    return {
        "states": np.zeros((0, input_dim), dtype=float),
        "actions": np.zeros((0,), dtype=int),
        "rewards": np.zeros((0,), dtype=float),
        "dones": np.zeros((0,), dtype=float),
        "meta": {
            "episodes": 0,
            "steps": 0,
            "rows_used": 0,
            "mean_episode_reward": 0.0,
        },
    }


def _build_rollouts_from_rows(
    rows: list[dict],
    candidates: list[dict],
    beta_defaults: dict,
    *,
    spec: dict,
    beta_bounds: dict,
    input_dim: int,
    feature_spec: dict | None = None,
    config: dict | None = None,
) -> dict:
    """把逐行已入库问卷转成 PPO rollout。

    每条 respondent row 视为一个 episode：
    - 状态 `s_t`: respondent 特征 + 历史已选题摘要 + 题位进度
    - 动作 `a_t`: 当前发出的 task 在 candidate pool 中的索引
    - 奖励 `r_t`: 单题反馈 + 末步问卷级 D-error bonus
    - `done_t`: 当前 respondent 的最后一题
    """
    if not rows or not candidates:
        return _empty_rollouts(input_dim)

    sig_to_idx = {}
    for idx, task in enumerate(candidates):
        sig_to_idx[str((task or {}).get("sig") or _task_signature(task))] = idx

    states: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[float] = []
    used_rows = 0
    episode_rewards: list[float] = []
    prior_obs_rows: list[dict] = []

    for row in rows:
        respondent = _row_respondent(row)
        if not respondent:
            continue
        tasks = row.get("tasks", []) if isinstance(row.get("tasks", []), list) else []
        choices = row.get("choices", {}) if isinstance(row.get("choices", {}), dict) else {}
        valid_steps: list[tuple[dict, int, str]] = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            task_id = str(task.get("id", "")).strip()
            if choices and task_id and task_id not in choices:
                continue
            sig = str(task.get("sig") or _task_signature(task))
            action_idx = sig_to_idx.get(sig)
            if action_idx is None:
                continue
            valid_steps.append((task, int(action_idx), str(choices.get(task_id, ""))))
        if not valid_steps:
            continue

        episode_tasks = [task for task, _action_idx, _chosen in valid_steps]
        episode_choices = {}
        for task, _action_idx, chosen_alt in valid_steps:
            task_id = str((task or {}).get("id", "")).strip()
            if task_id:
                episode_choices[task_id] = chosen_alt
        reward_metrics = _questionnaire_reward_metrics(
            episode_tasks,
            respondent,
            episode_choices,
            spec=spec,
            beta_defaults=beta_defaults,
            beta_bounds=beta_bounds,
            prior_obs_rows=prior_obs_rows,
            config=config,
        )
        episode_bonus = float(reward_metrics.get("reward", 0.0) or 0.0)
        episode_rewards.append(float(episode_bonus))
        prior_tasks: list[dict] = []
        n_steps = len(valid_steps)
        for step_idx, (task, action_idx, chosen_alt) in enumerate(valid_steps):
            state = _task_state_vector(
                respondent,
                prior_tasks,
                beta_defaults=beta_defaults,
                feature_spec=feature_spec,
                step_index=step_idx,
                episode_len=n_steps,
                input_dim=input_dim,
            )
            terminal_bonus = float(episode_bonus) if step_idx == n_steps - 1 else 0.0
            reward = _task_step_reward(
                task,
                respondent,
                chosen_alt,
                beta_defaults=beta_defaults,
                terminal_bonus=terminal_bonus,
            )
            states.append(state)
            actions.append(int(action_idx))
            rewards.append(float(reward))
            dones.append(1.0 if step_idx == n_steps - 1 else 0.0)
            prior_tasks.append(task)
        prior_obs_rows.extend(reward_metrics.get("current_obs_rows", []) or [])
        used_rows += 1
    if not states:
        return _empty_rollouts(input_dim)
    return {
        "states": np.vstack(states),
        "actions": np.array(actions, dtype=int),
        "rewards": np.array(rewards, dtype=float),
        "dones": np.array(dones, dtype=float),
        "meta": {
            "episodes": int(len(episode_rewards)),
            "steps": int(len(actions)),
            "rows_used": int(used_rows),
            "mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        },
    }


def _build_synthetic_rollouts(
    candidates: list[dict],
    respondents: list[dict],
    beta_defaults: dict,
    *,
    spec: dict,
    beta_bounds: dict,
    expert_result: dict | None,
    tasks_per_round: int,
    input_dim: int,
    feature_spec: dict | None = None,
    seed: int = 42,
    config: dict | None = None,
) -> dict:
    """在没有真实 rows 时，生成一批合成 episode 作为 warm-start rollout。"""
    if not candidates or not respondents:
        return _empty_rollouts(input_dim)

    rng = np.random.default_rng(seed)
    candidate_sigs = [str((c or {}).get("sig") or _task_signature(c)) for c in candidates]
    expert_sigs = _expert_signature_set(expert_result)
    expert_prior = np.array([1.0 if sig in expert_sigs else 0.0 for sig in candidate_sigs], dtype=float)
    if np.sum(expert_prior) > 0:
        expert_prior = expert_prior / np.sum(expert_prior)

    states: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[float] = []
    episode_rewards: list[float] = []
    prior_obs_rows: list[dict] = []

    for epi_idx, respondent in enumerate(respondents):
        remain_idx = list(range(len(candidates)))
        picked_tasks: list[dict] = []
        step_records: list[tuple[np.ndarray, int, dict, str]] = []
        n_steps = min(max(1, int(tasks_per_round)), len(remain_idx))
        for step_idx in range(n_steps):
            state = _task_state_vector(
                respondent,
                picked_tasks,
                beta_defaults=beta_defaults,
                feature_spec=feature_spec,
                step_index=step_idx,
                episode_len=n_steps,
                input_dim=input_dim,
            )
            sub_uniform = np.ones((len(remain_idx),), dtype=float) / max(len(remain_idx), 1)
            if np.sum(expert_prior) > 0:
                sub_prior = np.array([expert_prior[i] for i in remain_idx], dtype=float)
                if np.sum(sub_prior) > 0:
                    sub_prior = sub_prior / np.sum(sub_prior)
                    select_prob = 0.35 * sub_uniform + 0.65 * sub_prior
                else:
                    select_prob = sub_uniform
            else:
                select_prob = sub_uniform
            pick_pos = int(rng.choice(len(remain_idx), p=select_prob))
            action_idx = int(remain_idx.pop(pick_pos))
            task = deepcopy(candidates[action_idx])
            if not str(task.get("id", "")).strip():
                task["id"] = f"sim_ep{epi_idx+1}_r{step_idx+1}"
            chosen_alt = _simulate_choice_for_task(
                task,
                respondent,
                beta_defaults,
                seed_key=f"{respondent.get('respondent_id','')}_{task.get('id','')}",
            )
            step_records.append((state, action_idx, task, chosen_alt))
            picked_tasks.append(task)

        if not step_records:
            continue
        episode_choices = {str(task.get("id", "")): chosen_alt for _state, _action_idx, task, chosen_alt in step_records}
        reward_metrics = _questionnaire_reward_metrics(
            picked_tasks,
            respondent,
            episode_choices,
            spec=spec,
            beta_defaults=beta_defaults,
            beta_bounds=beta_bounds,
            prior_obs_rows=prior_obs_rows,
            config=config,
        )
        episode_bonus = float(reward_metrics.get("reward", 0.0) or 0.0)
        episode_rewards.append(float(episode_bonus))
        for step_idx, (state, action_idx, task, chosen_alt) in enumerate(step_records):
            terminal_bonus = float(episode_bonus) if step_idx == len(step_records) - 1 else 0.0
            reward = _task_step_reward(
                task,
                respondent,
                chosen_alt,
                beta_defaults=beta_defaults,
                terminal_bonus=terminal_bonus,
            )
            states.append(state)
            actions.append(int(action_idx))
            rewards.append(float(reward))
            dones.append(1.0 if step_idx == len(step_records) - 1 else 0.0)
        prior_obs_rows.extend(reward_metrics.get("current_obs_rows", []) or [])

    if not states:
        return _empty_rollouts(input_dim)
    return {
        "states": np.vstack(states),
        "actions": np.array(actions, dtype=int),
        "rewards": np.array(rewards, dtype=float),
        "dones": np.array(dones, dtype=float),
        "meta": {
            "episodes": int(len(episode_rewards)),
            "steps": int(len(actions)),
            "rows_used": 0,
            "mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        },
    }


def _compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    """根据 reward/value/done 计算 GAE advantage 与 return。

    参数:
        rewards: `[T]`，每一步环境奖励。
        values: `[T]`，critic 对每个状态的 `V(s_t)` 估计。
        dones: `[T]`，episode 结束标记，结束步为 1。
        gamma: 折扣因子。
        gae_lambda: GAE 衰减系数。

    返回:
        tuple[np.ndarray, np.ndarray]:
            - advantages: `[T]`
            - returns: `[T] = advantages + values`
    """
    n = int(len(rewards))
    advantages = np.zeros((n,), dtype=float)
    last_gae = 0.0
    for t in range(n - 1, -1, -1):
        next_value = float(values[t + 1]) if t + 1 < n else 0.0
        nonterminal = 1.0 - float(dones[t])
        delta = float(rewards[t]) + float(gamma) * next_value * nonterminal - float(values[t])
        last_gae = delta + float(gamma) * float(gae_lambda) * nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def _train_policy(
    rollouts: dict,
    *,
    input_dim: int,
    output_dim: int,
    seed: int,
    epochs: int,
    lr: float,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    gamma: float,
    gae_lambda: float,
    batch_size: int,
    target_kl: float,
    init_actor_state: dict | None = None,
    init_critic_state: dict | None = None,
) -> tuple[dict, dict, list[dict]]:
    """基于 rollout + GAE 的真正 PPO-clip 训练。

    参数:
        rollouts: 轨迹数据字典，至少包含：
            - `states`: `[T, input_dim]`
            - `actions`: `[T]`
            - `rewards`: `[T]`
            - `dones`: `[T]`
            - `meta`: 轨迹汇总信息
        input_dim: 状态维度。
        output_dim: 动作空间维度（candidate pool 大小）。
        seed: 随机种子。
        epochs: 对同一批 rollout 重复优化的轮数。
        lr: 学习率。
        clip_eps: PPO clip 范围。
        value_coef: critic 损失权重。
        entropy_coef: 熵正则权重。
        gamma: 折扣因子。
        gae_lambda: GAE 系数。
        batch_size: mini-batch 大小。
        target_kl: 若平均 KL 超过该阈值，则提前停止本轮 PPO 优化。
        init_actor_state/init_critic_state: 历史模型参数。

    返回:
        tuple[dict, dict, list[dict]]:
            - actor_state: 训练后模型参数（JSON 形式）
            - critic_state: 与 actor 共用同一骨干网络时，仍返回同一份 state_dict
            - logs: 训练日志
    """
    if not TORCH_AVAILABLE:
        return {}, {}, [{"epoch": 0, "msg": "torch not installed"}]

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = ActorCriticNet(input_dim=input_dim, output_dim=output_dim, hidden_dim=64)
    init_state = init_actor_state if isinstance(init_actor_state, dict) and init_actor_state else init_critic_state
    _load_state_dict_from_json(model, init_state if isinstance(init_state, dict) else None)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    logs: list[dict] = []

    states_np = np.asarray(rollouts.get("states", np.zeros((0, input_dim))), dtype=np.float32)
    actions_np = np.asarray(rollouts.get("actions", np.zeros((0,), dtype=int)), dtype=np.int64)
    rewards_np = np.asarray(rollouts.get("rewards", np.zeros((0,), dtype=float)), dtype=np.float32)
    dones_np = np.asarray(rollouts.get("dones", np.zeros((0,), dtype=float)), dtype=np.float32)
    meta = rollouts.get("meta", {}) if isinstance(rollouts.get("meta", {}), dict) else {}
    if states_np.size == 0 or len(actions_np) == 0:
        sd = _state_dict_to_json(model.state_dict())
        return sd, sd, [{"epoch": 0, "loss": None, "steps": 0, "episodes": 0}]

    x_t = torch.tensor(states_np, dtype=torch.float32)
    a_t = torch.tensor(actions_np, dtype=torch.long)

    with torch.no_grad():
        old_logits, old_values_t = model(x_t)
        old_dist = Categorical(logits=old_logits)
        old_logp = old_dist.log_prob(a_t)
        old_values = old_values_t.detach().cpu().numpy()
    adv_np, ret_np = _compute_gae(rewards_np, old_values, dones_np, gamma, gae_lambda)
    if adv_np.size > 1:
        adv_np = (adv_np - np.mean(adv_np)) / (np.std(adv_np) + 1e-8)
    advantages_t = torch.tensor(adv_np, dtype=torch.float32)
    returns_t = torch.tensor(ret_np, dtype=torch.float32)

    n_steps = int(len(actions_np))
    batch_size = max(1, min(int(batch_size), n_steps))
    rng = np.random.default_rng(seed + 11)

    for ep in range(max(1, int(epochs))):
        perm = rng.permutation(n_steps)
        ep_loss = []
        ep_policy = []
        ep_value = []
        ep_entropy = []
        ep_kl = []
        ep_clip = []
        for start in range(0, n_steps, batch_size):
            idx = perm[start:start + batch_size]
            idx_t = torch.tensor(idx, dtype=torch.long)
            b_x = x_t[idx_t]
            b_a = a_t[idx_t]
            b_old_logp = old_logp[idx_t]
            b_adv = advantages_t[idx_t]
            b_ret = returns_t[idx_t]

            logits, values = model(b_x)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(b_a)
            ratio = torch.exp(logp - b_old_logp)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((values - b_ret) ** 2).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            approx_kl = float((b_old_logp - logp).mean().detach().cpu().item())
            clip_frac = float((torch.abs(ratio - 1.0) > clip_eps).float().mean().detach().cpu().item())
            ep_loss.append(float(loss.detach().cpu().item()))
            ep_policy.append(float(policy_loss.detach().cpu().item()))
            ep_value.append(float(value_loss.detach().cpu().item()))
            ep_entropy.append(float(entropy.detach().cpu().item()))
            ep_kl.append(approx_kl)
            ep_clip.append(clip_frac)

        if ep == 0 or ep == epochs - 1 or ep % max(1, epochs // 5) == 0:
            logs.append(
                {
                    "epoch": int(ep + 1),
                    "loss": round(float(np.mean(ep_loss)) if ep_loss else 0.0, 6),
                    "policy_loss": round(float(np.mean(ep_policy)) if ep_policy else 0.0, 6),
                    "value_loss": round(float(np.mean(ep_value)) if ep_value else 0.0, 6),
                    "entropy": round(float(np.mean(ep_entropy)) if ep_entropy else 0.0, 6),
                    "approx_kl": round(float(np.mean(ep_kl)) if ep_kl else 0.0, 6),
                    "clip_frac": round(float(np.mean(ep_clip)) if ep_clip else 0.0, 6),
                    "mean_reward": round(float(np.mean(rewards_np)) if rewards_np.size else 0.0, 6),
                    "episodes": int(meta.get("episodes", 0) or 0),
                    "steps": int(meta.get("steps", n_steps) or n_steps),
                }
            )
        mean_kl = float(np.mean(ep_kl)) if ep_kl else 0.0
        if target_kl > 0 and mean_kl > target_kl:
            logs.append(
                {
                    "epoch": int(ep + 1),
                    "event": "early_stop_target_kl",
                    "target_kl": float(target_kl),
                    "approx_kl": round(mean_kl, 6),
                }
            )
            break

    sd = _state_dict_to_json(model.state_dict())
    return sd, sd, logs


def _score_candidates(candidates: list[dict], *, state_x: np.ndarray, actor_state: dict, input_dim: int, output_dim: int) -> np.ndarray:
    """执行一次前向计算，并将 actor logits 转为候选任务概率。

    参数:
        candidates: 候选任务列表。
        state_x: 当前状态特征向量。
        actor_state: actor 参数（JSON 形式）。
        input_dim: 模型输入维度。
        output_dim: 模型输出维度。

    返回:
        np.ndarray: 与 candidates 对齐的概率分数向量。
    """
    if not TORCH_AVAILABLE:
        return np.ones((len(candidates),), dtype=float) / max(1, len(candidates))
    model = ActorCriticNet(input_dim=input_dim, output_dim=output_dim, hidden_dim=64)
    _load_state_dict_from_json(model, actor_state)
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(state_x, dtype=torch.float32).reshape(1, -1)
        logits, _v = model(x_t)
        probs_t = torch.softmax(logits.squeeze(0), dim=0)
        probs = probs_t.cpu().numpy()
    if len(probs) < len(candidates):
        probs = np.concatenate([probs, np.zeros((len(candidates) - len(probs),), dtype=float)])
    return probs[: len(candidates)]


def _expert_signature_set(expert_result: dict | None) -> set[str]:
    """提取 efficient/expert 题组的任务签名集合。"""
    out: set[str] = set()
    expert_tasks = (expert_result or {}).get("comb", []) if isinstance(expert_result, dict) else []
    for task in expert_tasks or []:
        if not isinstance(task, dict):
            continue
        out.add(str(task.get("sig") or _task_signature(task)))
    return out


def _blend_with_expert_prior(probs: np.ndarray, candidates: list[dict], expert_result: dict | None) -> tuple[np.ndarray, dict]:
    """把 efficient design 作为专家经验，混入 actor 概率而不收窄动作空间。

    这里不会把 expert 题组当成唯一候选池，而是把它转成一个稀疏先验分布，
    与当前 actor 的输出做凸组合。这样：
    1) action 仍来自完整 feasible combo pool；
    2) efficient 仅提供 warm-start / expert prior；
    3) 若 expert 题组与当前候选池无重叠，则完全退化为 actor 原始输出。
    """
    base = np.array(probs, dtype=float)
    if base.size == 0:
        return base, {"expert_overlap": 0, "expert_prior_weight": 0.0}
    if np.sum(base) <= 0:
        base = np.ones_like(base) / len(base)
    else:
        base = base / np.sum(base)

    expert_sigs = _expert_signature_set(expert_result)
    if not expert_sigs:
        return base, {"expert_overlap": 0, "expert_prior_weight": 0.0}

    prior = np.array(
        [1.0 if str((c or {}).get("sig") or _task_signature(c)) in expert_sigs else 0.0 for c in candidates],
        dtype=float,
    )
    overlap = int(np.sum(prior))
    if overlap <= 0:
        return base, {"expert_overlap": 0, "expert_prior_weight": 0.0}
    prior = prior / np.sum(prior)
    alpha = min(0.35, max(0.1, overlap / max(len(candidates), 1)))
    mixed = (1.0 - alpha) * base + alpha * prior
    mixed = mixed / np.sum(mixed)
    return mixed, {"expert_overlap": overlap, "expert_prior_weight": float(alpha)}


def train_dynamic_ppo(
    *,
    payload: dict,
    policy_state: dict,
    candidate_pool: list[dict],
    expert_result: dict | None,
    rows: list[dict] | None = None,
    current_respondent: dict | None = None,
    data_dir: Path,
    config: dict,
) -> dict:
    """dynamicPPO 设计生成的主训练入口（近离线）。

    算法流程：
    1) 从 payload/config 读取 dyppo 超参数；
    2) 使用 feasible combo pool 作为真实动作空间；
    3) 将 efficient design 结果仅作为 expert prior；
    4) 从真实 rows 或合成 respondent 生成 rollout；
    5) 用 GAE + PPO-clip 训练 Actor-Critic；
    6) 对当前 respondent 按题位逐步生成 task 序列；
    7) 写回 policy_state，供后续在线更新。
    参数:
        payload: 前端传入的设计与超参数配置。
        policy_state: 当前策略状态（会被就地更新）。
        candidate_pool: 经过硬约束/去主导/可行性过滤后的候选动作池。
        expert_result: efficient design 计算结果，仅作为专家经验。
        rows: 可选的逐行训练数据；若提供，则优先使用它训练。
        current_respondent: 当前要发题的 respondent 快照；若提供，则用于生成当前题组。
        data_dir: 数据目录。
        config: 全局配置字典。

    返回:
        dict: 训练与推荐结果，主要字段包括：
            - comb: 推荐题组列表；
            - d_error: D-error 摘要；
            - iteration_log: 训练日志；
            - model_state: 模型元信息；
            - policy_state: 更新后的策略状态。
    """
    if not TORCH_AVAILABLE:
        return {
            "comb": [],
            "d_error": {"value": None},
            "iteration_log": [{"epoch": 0, "msg": "PyTorch未安装，dynamicPPO无法训练。请先安装torch。"}],
            "model_state": {
                "trained": False,
                "backend": "missing_torch",
                "required_package": "torch",
                "install_hint": "请先在虚拟环境中安装 PyTorch，再运行 dynamicPPO。",
            },
            "policy_state": policy_state,
        }
    # 第1步：解析配置与训练超参数。
    design_options = payload.get("design_options", {}) or {}
    if not isinstance(design_options, dict):
        design_options = {}
    mopt = design_options.get("dyppo", {}) or {}
    if not isinstance(mopt, dict):
        mopt = {}
    tpr = int(mopt.get("tasks_per_round", 6) or 6)
    eps = float(mopt.get("explore_epsilon", 0.2) or 0.2)
    dyn_cfg = config.get("dynamic_ppo", {}) if isinstance(config.get("dynamic_ppo", {}), dict) else {}
    train_n = int(mopt.get("train_respondents", dyn_cfg.get("train_respondents", 300) or 300))
    epochs = int(mopt.get("train_epochs", dyn_cfg.get("train_epochs", 200) or 200))
    lr = float(mopt.get("train_lr", dyn_cfg.get("train_lr", 0.03) or 0.03))
    seed = int(dyn_cfg.get("seed", 42))
    gamma = float(mopt.get("gamma", dyn_cfg.get("gamma", 0.99) or 0.99))
    gae_lambda = float(mopt.get("gae_lambda", dyn_cfg.get("gae_lambda", 0.95) or 0.95))
    clip_eps = float(mopt.get("clip_eps", dyn_cfg.get("clip_eps", 0.2) or 0.2))
    value_coef = float(mopt.get("value_coef", dyn_cfg.get("value_coef", 0.5) or 0.5))
    entropy_coef = float(mopt.get("entropy_coef", dyn_cfg.get("entropy_coef", 0.01) or 0.01))
    batch_size = int(mopt.get("batch_size", dyn_cfg.get("batch_size", 128) or 128))
    target_kl = float(mopt.get("target_kl", dyn_cfg.get("target_kl", 0.03) or 0.03))

    # 第2步：候选池来自 feasible combo pool，efficient 仅做 expert prior。
    candidates = list(candidate_pool or [])
    if not candidates:
        return {
            "comb": [],
            "d_error": {"value": None},
            "iteration_log": [{"epoch": 0, "msg": "no feasible candidates"}],
            "model_state": {"trained": False},
            "policy_state": policy_state,
        }
    for c in candidates:
        c["sig"] = str(c.get("sig") or _task_signature(c))

    output_dim = max(1, len(candidates))
    pop_stats = _load_pop_stats(payload, data_dir=data_dir, config=config)
    feature_spec = _build_feature_spec(pop_stats)
    input_dim = max(
        int(config.get("dynamic_ppo", {}).get("input_dim", feature_spec.get("feature_dim", 6)) or feature_spec.get("feature_dim", 6)),
        int(feature_spec.get("feature_dim", 6) or 6),
    )
    beta_defaults = payload.get("beta_defaults", {}) or {}
    beta_bounds = payload.get("beta_bounds", {}) if isinstance(payload.get("beta_bounds", {}), dict) else {}
    spec = payload.get("design_spec", {}) if isinstance(payload.get("design_spec", {}), dict) else {}
    rollout_meta = {"episodes": 0, "steps": 0, "rows_used": 0, "mean_episode_reward": 0.0}
    synthetic_respondents: list[dict] = []
    if rows:
        rollouts = _build_rollouts_from_rows(
            rows,
            candidates,
            beta_defaults,
            spec=spec,
            beta_bounds=beta_bounds,
            input_dim=input_dim,
            feature_spec=feature_spec,
            config=config,
        )
        rollout_meta = rollouts.get("meta", rollout_meta) if isinstance(rollouts.get("meta", {}), dict) else rollout_meta
    else:
        rollouts = _empty_rollouts(input_dim)
    data_source = "rows_jsonl_rollout" if int(rollout_meta.get("steps", 0) or 0) > 0 else "synthetic_popsim_rollout"
    # 第3步：若没有真实逐行样本，则按目标分布生成合成 respondent 轨迹。
    if int(rollout_meta.get("steps", 0) or 0) <= 0:
        synthetic_respondents = _sample_respondents(pop_stats, train_n, seed + int(policy_state.get("response_count", 0)))
        rollouts = _build_synthetic_rollouts(
            candidates,
            synthetic_respondents,
            beta_defaults,
            spec=spec,
            beta_bounds=beta_bounds,
            expert_result=expert_result,
            tasks_per_round=tpr,
            input_dim=input_dim,
            feature_spec=feature_spec,
            seed=seed + int(policy_state.get("response_count", 0)),
            config=config,
        )
        rollout_meta = rollouts.get("meta", rollout_meta) if isinstance(rollouts.get("meta", {}), dict) else rollout_meta

    active_bound_mask = policy_state.get("candidate_bound_mask", {}) if isinstance(policy_state.get("candidate_bound_mask", {}), dict) else {}
    current_mnl_signal = policy_state.get("current_mnl_signal", {}) if isinstance(policy_state.get("current_mnl_signal", {}), dict) else {}
    observed_rows = list(rows or [])
    if observed_rows:
        obs_rows = _collect_obs_rows_from_submission_rows(observed_rows, spec)
        signal = _mnl_signal_from_obs_rows(
            obs_rows,
            spec=spec,
            beta_defaults=beta_defaults,
            beta_bounds=beta_bounds,
            config=config,
        )
        current_mnl_signal = {
            "n_observations": int(signal.get("n_observations", 0) or 0),
            "adjusted_pseudo_r2": signal.get("adjusted_pseudo_r2"),
            "estimated_beta_raw": deepcopy(signal.get("estimated_beta_raw", {})),
            "estimated_beta": deepcopy(signal.get("estimated_beta", {})),
            "bound_violation": deepcopy(signal.get("bound_violation", {})),
        }
        active_bound_mask = deepcopy(signal.get("bound_violation", {}))

    init_actor = policy_state.get("actor_state", {}) if isinstance(policy_state.get("actor_state", {}), dict) else None
    init_critic = policy_state.get("critic_state", {}) if isinstance(policy_state.get("critic_state", {}), dict) else None

    actor_state, critic_state, logs = _train_policy(
        rollouts,
        input_dim=input_dim,
        output_dim=output_dim,
        seed=seed,
        epochs=epochs,
        lr=lr,
        clip_eps=clip_eps,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        gamma=gamma,
        gae_lambda=gae_lambda,
        batch_size=batch_size,
        target_kl=target_kl,
        init_actor_state=init_actor,
        init_critic_state=init_critic,
    )

    # 第5步：使用训练后的 actor，围绕“当前 respondent + 已选题组”逐题生成。
    score_respondent = current_respondent if isinstance(current_respondent, dict) and current_respondent else None
    scoring_state_source = "current_respondent"
    if not score_respondent and rows:
        score_respondent = _row_respondent(rows[0]) if rows else None
        scoring_state_source = "first_training_row"
    if not score_respondent and synthetic_respondents:
        score_respondent = synthetic_respondents[0]
        scoring_state_source = "synthetic_respondent"
    if not score_respondent:
        score_respondent = {
            "respondent_id": "default",
            "zone_id": _MISSING_ATTR_TOKEN,
            "attr_segments": [_MISSING_ATTR_TOKEN] * max(1, len(feature_spec.get("attr_dim_names", []))),
        }
        scoring_state_source = "missing_default"

    # 第6步：对每一个题位做一次策略前向，并在未选动作中采样。
    rng = np.random.default_rng(seed + 99 + int(policy_state.get("response_count", 0)))
    picked = []
    remain_idx = list(range(len(candidates)))
    expert_stats = {"expert_overlap": 0, "expert_prior_weight": 0.0}
    bound_mask_stats = {"active_bound_mask_params": 0, "bound_masked_candidates": 0, "bound_mask_min_factor": 1.0}
    for _ in range(min(max(1, tpr), len(remain_idx))):
        state_x = _task_state_vector(
            score_respondent,
            picked,
            beta_defaults=beta_defaults,
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
        probs, expert_stats = _blend_with_expert_prior(probs, candidates, expert_result)
        probs, bound_mask_stats = _apply_candidate_bound_mask(probs, candidates, active_bound_mask)
        sub_probs = np.array([probs[i] for i in remain_idx], dtype=float)
        if np.sum(sub_probs) <= 0:
            sub_probs = np.ones_like(sub_probs) / len(sub_probs)
        else:
            sub_probs = sub_probs / np.sum(sub_probs)
        if rng.random() < max(0.0, min(1.0, eps)):
            k = int(rng.integers(0, len(remain_idx)))
        else:
            k = int(rng.choice(len(remain_idx), p=sub_probs))
        idx = remain_idx.pop(k)
        picked.append(candidates[idx])

    final_tasks = []
    for i, t in enumerate(picked):
        final_tasks.append({**t, "block": 1, "row_in_block": i + 1, "id": f"preview_b1_r{i+1}"})

    # 第7步：持久化策略状态并组织返回结果。
    policy_state["actor_state"] = actor_state
    policy_state["critic_state"] = critic_state
    policy_state["input_dim"] = input_dim
    policy_state["output_dim"] = output_dim
    policy_state["attr_dim_names"] = deepcopy(feature_spec.get("attr_dim_names", []))
    policy_state["attr_categories"] = deepcopy(feature_spec.get("attr_categories", []))
    policy_state["zone_categories"] = deepcopy(feature_spec.get("zone_categories", []))
    policy_state["candidate_signatures"] = [str((c or {}).get("sig") or _task_signature(c)) for c in candidates]
    policy_state["trained"] = True
    policy_state["train_epochs"] = int(epochs)
    policy_state["gamma"] = float(gamma)
    policy_state["gae_lambda"] = float(gae_lambda)
    policy_state["clip_eps"] = float(clip_eps)
    policy_state["value_coef"] = float(value_coef)
    policy_state["entropy_coef"] = float(entropy_coef)
    policy_state["target_kl"] = float(target_kl)
    policy_state["response_count"] = int(policy_state.get("response_count", 0))
    policy_state["candidate_bound_mask"] = deepcopy(active_bound_mask) if isinstance(active_bound_mask, dict) else {}
    policy_state["current_mnl_signal"] = deepcopy(current_mnl_signal) if isinstance(current_mnl_signal, dict) else {}

    return {
        "comb": final_tasks,
        "d_error": {"value": None},
        "iteration_log": logs,
        "model_state": {
            "trained": True,
            "backend": "pytorch",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "candidate_pool_size": len(candidates),
            "training_data_source": data_source,
            "rows_used": int(rollout_meta.get("rows_used", 0) or 0),
            "rollout_episodes": int(rollout_meta.get("episodes", 0) or 0),
            "rollout_steps": int(rollout_meta.get("steps", 0) or 0),
            "mean_episode_reward": float(rollout_meta.get("mean_episode_reward", 0.0) or 0.0),
            "train_samples": int(rollouts.get("states", np.zeros((0, input_dim))).shape[0]) if isinstance(rollouts, dict) else 0,
            "train_epochs": int(epochs),
            "gamma": float(gamma),
            "gae_lambda": float(gae_lambda),
            "clip_eps": float(clip_eps),
            "value_coef": float(value_coef),
            "entropy_coef": float(entropy_coef),
            "batch_size": int(batch_size),
            "target_kl": float(target_kl),
            "epsilon": float(eps),
            "expert_overlap": int(expert_stats.get("expert_overlap", 0) or 0),
            "expert_prior_weight": float(expert_stats.get("expert_prior_weight", 0.0) or 0.0),
            "active_bound_mask_params": int(bound_mask_stats.get("active_bound_mask_params", 0) or 0),
            "bound_masked_candidates": int(bound_mask_stats.get("bound_masked_candidates", 0) or 0),
            "bound_mask_min_factor": float(bound_mask_stats.get("bound_mask_min_factor", 1.0) or 1.0),
            "scoring_state_source": scoring_state_source,
            "attr_dim_names": deepcopy(feature_spec.get("attr_dim_names", [])),
            "attr_categories": deepcopy(feature_spec.get("attr_categories", [])),
            "zone_categories": deepcopy(feature_spec.get("zone_categories", [])),
            "policy_version": int(policy_state.get("response_count", 0)) + 1,
            "current_mnl_signal": deepcopy(current_mnl_signal),
        },
        "policy_state": policy_state,
    }


def online_update_dynamic_ppo(
    *,
    payload: dict,
    policy_state: dict,
    tasks: list[dict],
    choices: dict,
    respondent: dict | None = None,
    historical_rows: list[dict] | None = None,
    config: dict | None = None,
) -> dict:
    """每次问卷提交后的单 episode GAE-PPO 在线更新。"""
    if not TORCH_AVAILABLE:
        return {"updated": False, "reason": "torch not installed"}
    if not tasks:
        return {"updated": False}
    input_dim = int(policy_state.get("input_dim", 6) or 6)
    output_dim = int(policy_state.get("output_dim", max(1, len(tasks))) or max(1, len(tasks)))
    actor_state = policy_state.get("actor_state", {}) if isinstance(policy_state.get("actor_state", {}), dict) else {}
    critic_state = policy_state.get("critic_state", {}) if isinstance(policy_state.get("critic_state", {}), dict) else {}
    if not actor_state:
        return {"updated": False, "reason": "weights not initialized"}

    feature_spec = {
        "attr_dim_names": deepcopy(policy_state.get("attr_dim_names", [])),
        "attr_categories": deepcopy(policy_state.get("attr_categories", [])),
        "zone_categories": deepcopy(policy_state.get("zone_categories", [])),
    }
    candidate_sigs = [str(x) for x in (policy_state.get("candidate_signatures", []) or [])]
    sig_to_idx = {sig: idx for idx, sig in enumerate(candidate_sigs)}
    if not sig_to_idx:
        return {"updated": False, "reason": "candidate signatures missing"}

    beta_defaults = payload.get("beta_defaults", {}) if isinstance(payload.get("beta_defaults", {}), dict) else {}
    beta_bounds = payload.get("beta_bounds", {}) if isinstance(payload.get("beta_bounds", {}), dict) else {}
    spec = payload.get("design_spec", {}) if isinstance(payload.get("design_spec", {}), dict) else {}
    dyn_opt = ((payload.get("design_options", {}) or {}).get("dyppo", {}) or {})
    seed = int(policy_state.get("response_count", 0) or 0) + 1007
    gamma = float(policy_state.get("gamma", dyn_opt.get("gamma", 0.99)) or 0.99)
    gae_lambda = float(policy_state.get("gae_lambda", dyn_opt.get("gae_lambda", 0.95)) or 0.95)
    clip_eps = float(policy_state.get("clip_eps", dyn_opt.get("clip_eps", 0.2)) or 0.2)
    value_coef = float(policy_state.get("value_coef", dyn_opt.get("value_coef", 0.5)) or 0.5)
    entropy_coef = float(policy_state.get("entropy_coef", dyn_opt.get("entropy_coef", 0.01)) or 0.01)
    target_kl = float(policy_state.get("target_kl", dyn_opt.get("target_kl", 0.03)) or 0.03)
    online_lr = float(dyn_opt.get("online_lr", 0.005) or 0.005)
    online_epochs = int(dyn_opt.get("online_epochs", 2) or 2)

    current_respondent = respondent if isinstance(respondent, dict) and respondent else {
        "respondent_id": "online_missing",
        "zone_id": _MISSING_ATTR_TOKEN,
        "attr_segments": [_MISSING_ATTR_TOKEN] * max(1, len(feature_spec.get("attr_dim_names", []))),
    }

    valid_steps: list[tuple[dict, int, str]] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        sig = str(task.get("sig") or _task_signature(task))
        action_idx = sig_to_idx.get(sig)
        if action_idx is None:
            continue
        task_id = str(task.get("id", "")).strip()
        chosen_alt = str((choices or {}).get(task_id, "") or "")
        valid_steps.append((task, int(action_idx), chosen_alt))
    if not valid_steps:
        return {"updated": False, "reason": "no matched task actions"}

    episode_tasks = [task for task, _action_idx, _chosen_alt in valid_steps]
    episode_choices = {
        str((task or {}).get("id", "")).strip(): chosen_alt
        for task, _action_idx, chosen_alt in valid_steps
        if str((task or {}).get("id", "")).strip()
    }
    reward_metrics = _questionnaire_reward_metrics(
        episode_tasks,
        current_respondent,
        episode_choices,
        spec=spec,
        beta_defaults=beta_defaults,
        beta_bounds=beta_bounds,
        prior_obs_rows=_collect_obs_rows_from_submission_rows(list(historical_rows or []), spec),
        config=config,
    )
    episode_bonus = float(reward_metrics.get("reward", 0.0) or 0.0)
    states: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[float] = []
    prior_tasks: list[dict] = []
    for step_idx, (task, action_idx, chosen_alt) in enumerate(valid_steps):
        state = _task_state_vector(
            current_respondent,
            prior_tasks,
            beta_defaults=beta_defaults,
            feature_spec=feature_spec,
            step_index=step_idx,
            episode_len=len(valid_steps),
            input_dim=input_dim,
        )
        terminal_bonus = float(episode_bonus) if step_idx == len(valid_steps) - 1 else 0.0
        reward = _task_step_reward(
            task,
            current_respondent,
            chosen_alt,
            beta_defaults=beta_defaults,
            terminal_bonus=terminal_bonus,
        )
        states.append(state)
        actions.append(int(action_idx))
        rewards.append(float(reward))
        dones.append(1.0 if step_idx == len(valid_steps) - 1 else 0.0)
        prior_tasks.append(task)

    rollouts = {
        "states": np.vstack(states),
        "actions": np.array(actions, dtype=int),
        "rewards": np.array(rewards, dtype=float),
        "dones": np.array(dones, dtype=float),
        "meta": {
            "episodes": 1,
            "steps": len(actions),
            "rows_used": 1,
            "mean_episode_reward": float(episode_bonus),
        },
    }
    new_actor_state, new_critic_state, logs = _train_policy(
        rollouts,
        input_dim=input_dim,
        output_dim=output_dim,
        seed=seed,
        epochs=max(1, online_epochs),
        lr=online_lr,
        clip_eps=clip_eps,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        gamma=gamma,
        gae_lambda=gae_lambda,
        batch_size=len(actions),
        target_kl=target_kl,
        init_actor_state=actor_state,
        init_critic_state=critic_state,
    )

    policy_state["actor_state"] = new_actor_state
    policy_state["critic_state"] = new_critic_state
    policy_state["response_count"] = int(policy_state.get("response_count", 0)) + 1
    policy_state["candidate_bound_mask"] = deepcopy(reward_metrics.get("bound_violation", {}))
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
    return {
        "updated": True,
        "policy_version": int(policy_state["response_count"]),
        "online_episode_reward": round(float(episode_bonus), 6),
        "online_steps": int(len(actions)),
        "iteration_log": logs,
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
