"""SelfAttention 题目生成与在线更新模块。

本文件实现的是“带自注意力骨干的 Actor-Critic + PPO-clip”版本，
用于在候选题池中为当前 respondent 生成一整份 block，并在 respondent
提交回答后做一次在线增量更新。

整体数据流可以概括为：
1. 候选题池 `candidate_pool`
   形如 `list[task]`，其中单个 `task` 一般为：
   {
       "id": "preview_b1_r1",
       "sig": "a1b2c3...",
       "alternatives": {
           "car": {"time": 20, "cost": 8},
           "pt": {"time": 35, "cost": 4}
       }
   }
2. respondent / 状态特征
   由 `dynamicPPO.py` 中共享的特征构造逻辑生成单步状态向量
   `state_x ∈ R^[D]`，再堆叠成历史窗口
   `state_seq ∈ R^[L, D]`。
3. 自注意力网络
   输入 `x ∈ R^[B, L, D]`，输出：
   - `logits ∈ R^[B, N]`，N 为候选动作数（候选题数）
   - `value ∈ R^[B]`
4. 训练
   使用 rollout 字典：
   {
       "states": np.ndarray[T, D],
       "actions": np.ndarray[T],
       "rewards": np.ndarray[T],
       "dones": np.ndarray[T],
       "meta": {...}
   }
   其中 `T` 是总 step 数。
5. 生成
   对每个题位逐次构造当前状态窗口，调用网络获得对剩余候选题的概率，
   再结合 expert prior、参数越界惩罚 mask 与 epsilon 探索完成抽题。
"""

from __future__ import annotations

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
    from . import dynamicPPO as dyppo_shared
except ImportError:
    from engine import dynamicPPO as dyppo_shared

_MISSING_ATTR_TOKEN = "__missing__"


def _safe_float(v, default: float = 0.0) -> float:
    """尽力把任意输入转成浮点数。

    参数:
        v: 任意待转换对象，常见为 `str / int / float / None`。
        default: 转换失败时返回的默认值。

    返回:
        float: 转换后的浮点数；若失败则返回 `default`。
    """
    try:
        return float(v)
    except Exception:
        return default


def _progress_print(verbose: bool, prefix: str, message: str) -> None:
    """按需输出训练/更新进度。

    参数:
        verbose: 是否开启打印。
        prefix: 日志前缀，例如 `selfattention/train`。
        message: 具体日志内容。

    返回:
        None
    """
    if not verbose:
        return
    print(f"[{prefix}] {message}", flush=True)


def _task_signature(task: dict) -> str:
    """为单个 task 生成稳定短签名。

    参数:
        task: 单个 SP 题目字典，通常至少包含：
            {
                "alternatives": {
                    "<alt_name>": {"<attr_name>": value, ...},
                    ...
                }
            }

    返回:
        str: 长度约 16 的稳定哈希短签名，用于：
            - 判断题目是否与 candidate pool 中的动作一致；
            - 在线更新时把“已出题 task”映射回动作索引。
    """
    alts = (task or {}).get("alternatives", {}) if isinstance(task, dict) else {}
    norm = {}
    for alt, attrs in alts.items():
        norm[str(alt)] = {str(k): attrs.get(k) for k in sorted((attrs or {}).keys())}
    return hashlib.sha1(str(sorted(norm.items())).encode("utf-8")).hexdigest()[:16]


if TORCH_AVAILABLE:
    class SelfAttentionActorCritic(nn.Module):
        """使用多头自注意力骨干的共享 Actor-Critic 网络。

        输入张量默认是 `[B, L, D]`：
        - `B`: batch size
        - `L`: 时间窗口长度（同一 respondent / 同一问卷内的历史状态序列）
        - `D`: 单步状态维度

        若外部只给 `[B, D]`，会自动升成长度为 1 的序列。

        网络内部维度流转如下：
        1. `x ∈ R^[B,L,D]`
        2. `embed(x) -> h ∈ R^[B,L,H]`，其中 `H = hidden_dim`
        3. Multi-head self-attention:
           - Q/K/V 由 `nn.MultiheadAttention` 内部线性层产生
           - 注意力输出 `attn_out ∈ R^[B,L,H]`
        4. 残差 + LayerNorm 后仍为 `R^[B,L,H]`
        5. 前馈层 FFN 后仍为 `R^[B,L,H]`
        6. 取最后一个 token：
           `pooled = h[:, -1, :] ∈ R^[B,H]`
        7. 两个 head：
           - `actor_head(pooled) -> logits ∈ R^[B,N]`
           - `critic_head(pooled) -> value ∈ R^[B]`
        """

        def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, num_heads: int = 4):
            """初始化共享骨干的 SelfAttention Actor-Critic。

            参数:
                input_dim: 单步状态特征维度 `D`。
                output_dim: 动作空间维度 `N`，即候选题池中的候选 task 数量。
                hidden_dim: 自注意力与前馈层内部维度 `H`。
                num_heads: 多头注意力的 head 数；若不能整除 `hidden_dim`，
                    会在 `_normalize_heads()` 中自动下调。

            返回:
                None
            """
            super().__init__()
            self.input_dim = int(input_dim)
            self.output_dim = int(output_dim)
            self.hidden_dim = max(16, int(hidden_dim))
            self.num_heads = self._normalize_heads(self.hidden_dim, int(num_heads))

            self.embed = nn.Linear(self.input_dim, self.hidden_dim)
            self.attn = nn.MultiheadAttention(self.hidden_dim, self.num_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(self.hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            )
            self.norm2 = nn.LayerNorm(self.hidden_dim)
            self.actor_head = nn.Linear(self.hidden_dim, self.output_dim)
            self.critic_head = nn.Linear(self.hidden_dim, 1)

        @staticmethod
        def _normalize_heads(hidden_dim: int, num_heads: int) -> int:
            """把 head 数调整为 `hidden_dim` 的可整除因子。

            参数:
                hidden_dim: 注意力内部总通道数 `H`。
                num_heads: 期望 head 数。

            返回:
                int: 实际使用的 head 数 `M`，满足：
                    - `1 <= M <= hidden_dim`
                    - `hidden_dim % M == 0`

            说明:
                多头注意力中每个 head 的通道数为 `d_head = hidden_dim / M`。
                若 `hidden_dim` 不能被用户给定的 `num_heads` 整除，这里会把
                `num_heads` 递减到最近的可整除值，避免 `nn.MultiheadAttention`
                构造时报错。
            """
            heads = max(1, min(int(num_heads), int(hidden_dim)))
            while heads > 1 and hidden_dim % heads != 0:
                heads -= 1
            return max(1, heads)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """前向传播。

            参数:
                x: `[B, L, D]` 或 `[B, D]`。

            返回:
                tuple[torch.Tensor, torch.Tensor]:
                    - logits: `[B, output_dim]`
                    - value: `[B]`

            维度说明:
                - 输入 `[B, D]` 时，会自动扩展为 `[B, 1, D]`
                - `key_padding_mask ∈ {0,1}^[B,L]`
                - `embed(x) -> [B,L,H]`
                - `attn_out -> [B,L,H]`
                - `pooled -> [B,H]`
                - `logits -> [B,N]`
                - `value -> [B]`
            """
            if x.dim() == 2:
                x = x.unsqueeze(1)
            if x.dim() != 3:
                raise ValueError(f"expected [B,L,D] or [B,D], got shape={tuple(x.shape)}")

            # 全零 token 视为左侧 padding，不参与注意力。
            key_padding_mask = torch.all(torch.abs(x) <= 1e-12, dim=-1)
            if key_padding_mask.ndim == 2:
                all_masked = torch.all(key_padding_mask, dim=1)
                if torch.any(all_masked):
                    key_padding_mask = key_padding_mask.clone()
                    key_padding_mask[all_masked, -1] = False

            # 线性嵌入: [B,L,D] -> [B,L,H]
            h = self.embed(x)
            # 自注意力输出: [B,L,H]
            attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
            h = self.norm1(h + attn_out)
            # 前馈层输出: [B,L,H]
            f = self.ffn(h)
            h = self.norm2(h + f)

            # 由于序列是右对齐构造的，最后一个 token 对应当前题位状态。
            pooled = h[:, -1, :]
            logits = self.actor_head(pooled)
            value = self.critic_head(pooled).squeeze(-1)
            return logits, value
else:
    class SelfAttentionActorCritic:
        pass


def _state_dict_to_json(sd: dict[str, torch.Tensor]) -> dict[str, list]:
    """把 PyTorch `state_dict` 转成可 JSON 序列化的 list 字典。

    参数:
        sd: 模型参数字典，通常来自 `model.state_dict()`。

    返回:
        dict[str, list]:
            键仍为参数名，值从 `torch.Tensor` 转为嵌套 `list`。

    说明:
        这里主要用于把权重暂存进 `policy_state`，便于：
        - 写入 JSON
        - 后续在线更新重新加载
    """
    if not TORCH_AVAILABLE:
        return {}
    out = {}
    for k, v in sd.items():
        out[k] = v.detach().cpu().numpy().tolist()
    return out


def _load_state_dict_from_json(model: nn.Module, raw: dict | None) -> bool:
    """在张量形状一致时，从 JSON 字典恢复模型参数。

    参数:
        model: 目标 PyTorch 模型。
        raw: JSON 反序列化后的参数字典，通常来自 `_state_dict_to_json()`。

    返回:
        bool:
            - `True`: 所有键存在且形状匹配，已成功加载
            - `False`: 任一键缺失、形状不匹配或加载异常
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


def _sa_cfg(config: dict | None) -> dict:
    """读取并标准化 SelfAttention 相关超参数。

    参数:
        config: 整体配置字典，通常来自 `data/config.json`。

    返回:
        dict:
            标准化后的自注意力/PPO 超参数字典，包含：
            - `seed`
            - `window_length`
            - `hidden_dim`
            - `num_heads`
            - `explore_epsilon`
            - `clip_eps`
            - `value_coef`
            - `entropy_coef`
            - `train_respondents`
            - `train_epochs`
            - `train_lr`
            - `online_lr`
            - `online_epochs`
            - `batch_size`
            - `online_batch_size`
            - `gamma`
            - `gae_lambda`
            - `target_kl`

    说明:
        本函数会优先读取 `config["self_attention"]`，
        部分字段缺失时回退到 `config["dynamic_ppo"]` 的默认值，
        这样三种策略的训练参数可以尽量保持口径一致。
    """
    cfg = (config or {}).get("self_attention", {}) if isinstance((config or {}).get("self_attention", {}), dict) else {}
    dyn = (config or {}).get("dynamic_ppo", {}) if isinstance((config or {}).get("dynamic_ppo", {}), dict) else {}
    return {
        "seed": int(cfg.get("seed", dyn.get("seed", 42)) or dyn.get("seed", 42) or 42),
        "window_length": int(cfg.get("window_length", 16) or 16),
        "hidden_dim": int(cfg.get("hidden_dim", 64) or 64),
        "num_heads": int(cfg.get("num_heads", 4) or 4),
        "explore_epsilon": float(cfg.get("explore_epsilon", 0.15) or 0.15),
        "clip_eps": float(cfg.get("clip_eps", dyn.get("clip_eps", 0.2)) or dyn.get("clip_eps", 0.2) or 0.2),
        "value_coef": float(cfg.get("value_coef", dyn.get("value_coef", 0.5)) or dyn.get("value_coef", 0.5) or 0.5),
        "entropy_coef": float(cfg.get("entropy_coef", dyn.get("entropy_coef", 0.01)) or dyn.get("entropy_coef", 0.01) or 0.01),
        "train_respondents": int(cfg.get("train_respondents", dyn.get("train_respondents", 300)) or dyn.get("train_respondents", 300) or 300),
        "train_epochs": int(cfg.get("train_epochs", dyn.get("train_epochs", 200)) or dyn.get("train_epochs", 200) or 200),
        "train_lr": float(cfg.get("train_lr", dyn.get("train_lr", 0.03)) or dyn.get("train_lr", 0.03) or 0.03),
        "online_lr": float(cfg.get("online_lr", 0.005) or 0.005),
        "online_epochs": int(cfg.get("online_epochs", 2) or 2),
        "batch_size": int(cfg.get("batch_size", dyn.get("batch_size", 128)) or dyn.get("batch_size", 128) or 128),
        "online_batch_size": int(cfg.get("online_batch_size", cfg.get("batch_size", dyn.get("batch_size", 128))) or cfg.get("batch_size", dyn.get("batch_size", 128)) or 128),
        "gamma": float(cfg.get("gamma", dyn.get("gamma", 0.99)) or dyn.get("gamma", 0.99) or 0.99),
        "gae_lambda": float(cfg.get("gae_lambda", dyn.get("gae_lambda", 0.95)) or dyn.get("gae_lambda", 0.95) or 0.95),
        "target_kl": float(cfg.get("target_kl", dyn.get("target_kl", 0.03)) or dyn.get("target_kl", 0.03) or 0.03),
    }


def _build_state_sequence_batch(states: np.ndarray, dones: np.ndarray, seq_len: int) -> np.ndarray:
    """把单步状态 `[T, D]` 重组为窗口序列 `[T, L, D]`。

    规则：
    - 仅在同一 episode 内回看历史；
    - 窗口不够长时左侧补 0；
    - 每个样本的最后一个 token 始终是当前时刻状态。

    参数:
        states: 单步状态矩阵 `states ∈ R^[T,D]`
            - `T`: 总 step 数
            - `D`: 单步状态维度
        dones: 终止标记向量 `dones ∈ R^[T]`
            - 当前 step 为 episode 最后一步时，`dones[t] = 1`
        seq_len: 历史窗口长度 `L`

    返回:
        np.ndarray:
            右对齐后的序列张量 `seq ∈ R^[T,L,D]`。
            对于每个 step `t`，`seq[t]` 都是“到当前 step 为止”的历史窗口。
    """
    states_np = np.asarray(states, dtype=np.float32)
    dones_np = np.asarray(dones, dtype=np.float32)
    if states_np.ndim != 2 or states_np.shape[0] == 0:
        return np.zeros((0, max(1, int(seq_len)), states_np.shape[1] if states_np.ndim == 2 else 0), dtype=np.float32)
    n_steps, input_dim = states_np.shape
    l = max(1, int(seq_len))
    out = np.zeros((n_steps, l, input_dim), dtype=np.float32)
    episode_start = 0
    for idx in range(n_steps):
        if idx > 0 and float(dones_np[idx - 1]) >= 0.5:
            episode_start = idx
        start = max(episode_start, idx - l + 1)
        hist = states_np[start:idx + 1]
        out[idx, -len(hist):, :] = hist
    return out


def _history_to_sequence(history_states: list[np.ndarray], seq_len: int, input_dim: int) -> np.ndarray:
    """把当前 respondent 已走过的状态历史转成一个右对齐窗口。

    参数:
        history_states: 历史单步状态列表，列表中每个元素形如 `R^[D]`。
        seq_len: 目标窗口长度 `L`。
        input_dim: 目标单步状态维度 `D`。

    返回:
        np.ndarray:
            `seq ∈ R^[L,D]`。

    说明:
        - 历史不足 `L` 时左侧补 0
        - 历史超过 `L` 时截取最近 `L` 步
        - 若单步状态长度与 `input_dim` 不一致，会做截断或补 0
    """
    l = max(1, int(seq_len))
    out = np.zeros((l, int(input_dim)), dtype=np.float32)
    if not history_states:
        return out
    arr = np.vstack([np.asarray(x, dtype=np.float32).reshape(1, -1) for x in history_states])
    if arr.shape[1] < input_dim:
        pad = np.zeros((arr.shape[0], input_dim - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    elif arr.shape[1] > input_dim:
        arr = arr[:, :input_dim]
    tail = arr[-l:]
    out[-len(tail):, :] = tail
    return out


def _compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    """根据 `reward / value / done` 计算 GAE advantage 与 return。

    参数:
        rewards: 奖励向量 `r ∈ R^[T]`
        values: 状态价值估计 `V(s_t) ∈ R^[T]`
        dones: 终止标记 `done ∈ R^[T]`
        gamma: 折扣因子
        gae_lambda: GAE 衰减系数

    返回:
        tuple[np.ndarray, np.ndarray]:
            - `advantages ∈ R^[T]`
            - `returns ∈ R^[T]`

    说明:
        `returns = advantages + values`
        后续 PPO 中：
        - `advantages` 用于更新策略 head
        - `returns` 用于回归 critic / value head
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
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
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
    init_state: dict | None,
    verbose: bool = False,
    progress_prefix: str = "selfattention/train",
) -> tuple[dict, list[dict]]:
    """基于 rollout + GAE 的 SelfAttention PPO-clip 训练。

    参数:
        rollouts: 训练样本字典，要求至少包含：
            - `states ∈ R^[T,D]`
            - `actions ∈ Z^[T]`
            - `rewards ∈ R^[T]`
            - `dones ∈ {0,1}^[T]`
            - `meta: dict`
          其中：
            - `T` 为总 step 数
            - `D` 为单步状态维度
        input_dim: 单步状态维度 `D`
        output_dim: 动作空间维度 `N`
        seq_len: 历史窗口长度 `L`
        hidden_dim: 注意力内部通道维度 `H`
        num_heads: 多头注意力 head 数 `M`
        seed: 随机种子
        epochs: 对同一批 rollout 重复优化的轮数
        lr: Adam 学习率
        clip_eps: PPO-clip 截断阈值
        value_coef: 价值损失权重
        entropy_coef: 熵正则权重
        gamma: 折扣因子
        gae_lambda: GAE 衰减系数
        batch_size: mini-batch 大小
        target_kl: 提前停止阈值；若平均 KL 超过该值则停止
        init_state: 初始模型参数；为空时随机初始化
        verbose: 是否打印训练过程
        progress_prefix: 日志前缀

    返回:
        tuple[dict, list[dict]]:
            - `new_state`: JSON 可序列化的模型参数字典
            - `logs`: 训练日志列表，每个元素为 epoch 级摘要

    关键张量维度:
        - `states_np ∈ R^[T,D]`
        - `seq_np ∈ R^[T,L,D]`
        - `x_t ∈ R^[T,L,D]`
        - `a_t ∈ Z^[T]`
        - `old_logits ∈ R^[T,N]`
        - `old_values ∈ R^[T]`
        - `advantages_t ∈ R^[T]`
        - `returns_t ∈ R^[T]`
        - mini-batch 内：
          `b_x ∈ R^[B_m,L,D]`, `b_a ∈ Z^[B_m]`, `logits ∈ R^[B_m,N]`
    """
    if not TORCH_AVAILABLE:
        return {}, [{"epoch": 0, "msg": "torch not installed"}]

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = SelfAttentionActorCritic(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
    )
    _load_state_dict_from_json(model, init_state if isinstance(init_state, dict) else None)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    logs: list[dict] = []

    states_np = np.asarray(rollouts.get("states", np.zeros((0, input_dim))), dtype=np.float32)
    actions_np = np.asarray(rollouts.get("actions", np.zeros((0,), dtype=int)), dtype=np.int64)
    rewards_np = np.asarray(rollouts.get("rewards", np.zeros((0,), dtype=float)), dtype=np.float32)
    dones_np = np.asarray(rollouts.get("dones", np.zeros((0,), dtype=float)), dtype=np.float32)
    meta = rollouts.get("meta", {}) if isinstance(rollouts.get("meta", {}), dict) else {}
    if states_np.size == 0 or len(actions_np) == 0:
        return _state_dict_to_json(model.state_dict()), [{"epoch": 0, "loss": None, "steps": 0, "episodes": 0}]

    _progress_print(
        verbose,
        progress_prefix,
        (
            f"start: samples={len(actions_np)} episodes={int(meta.get('episodes', 0) or 0)} "
            f"seq_len={int(seq_len)} input_dim={int(input_dim)} output_dim={int(output_dim)} "
            f"hidden_dim={int(hidden_dim)} heads={int(num_heads)} batch_size={int(max(1, min(int(batch_size), len(actions_np))))} "
            f"epochs={int(max(1, int(epochs)))} lr={float(lr):.6f}"
        ),
    )

    # 把平铺状态 `[T,D]` 重构为序列状态 `[T,L,D]`，
    # 这样每个 step 都能看到其所在 respondent / episode 的历史上下文。
    seq_np = _build_state_sequence_batch(states_np, dones_np, seq_len)
    x_t = torch.tensor(seq_np, dtype=torch.float32)
    a_t = torch.tensor(actions_np, dtype=torch.long)

    with torch.no_grad():
        # 旧策略（theta_old）固定下来，为 PPO ratio 提供分母。
        old_logits, old_values_t = model(x_t)
        old_dist = Categorical(logits=old_logits)
        old_logp = old_dist.log_prob(a_t)
        old_values = old_values_t.detach().cpu().numpy()
    # 基于旧 value 估计计算 advantage / return。
    adv_np, ret_np = _compute_gae(rewards_np, old_values, dones_np, gamma, gae_lambda)
    if adv_np.size > 1:
        adv_np = (adv_np - np.mean(adv_np)) / (np.std(adv_np) + 1e-8)
    advantages_t = torch.tensor(adv_np, dtype=torch.float32)
    returns_t = torch.tensor(ret_np, dtype=torch.float32)

    n_steps = int(len(actions_np))
    batch_size = max(1, min(int(batch_size), n_steps))
    rng = np.random.default_rng(seed + 17)

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
            # PPO ratio: r_t(theta) = pi_theta(a|s) / pi_theta_old(a|s)
            ratio = torch.exp(logp - b_old_logp)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            # critic 回归目标是 `return_t`
            value_loss = ((values - b_ret) ** 2).mean()
            entropy = dist.entropy().mean()
            # 总损失 = 策略损失 + 价值损失 - 熵正则
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

        mean_loss = float(np.mean(ep_loss)) if ep_loss else 0.0
        mean_policy = float(np.mean(ep_policy)) if ep_policy else 0.0
        mean_value = float(np.mean(ep_value)) if ep_value else 0.0
        mean_entropy = float(np.mean(ep_entropy)) if ep_entropy else 0.0
        mean_kl = float(np.mean(ep_kl)) if ep_kl else 0.0
        mean_clip = float(np.mean(ep_clip)) if ep_clip else 0.0

        _progress_print(
            verbose,
            progress_prefix,
            (
                f"epoch {ep + 1}/{max(1, int(epochs))}: "
                f"loss={mean_loss:.6f} policy={mean_policy:.6f} value={mean_value:.6f} "
                f"entropy={mean_entropy:.6f} kl={mean_kl:.6f} clip_frac={mean_clip:.6f} "
                f"mean_reward={float(np.mean(rewards_np)) if rewards_np.size else 0.0:.6f}"
            ),
        )

        if ep == 0 or ep == epochs - 1 or ep % max(1, epochs // 5) == 0:
            logs.append(
                {
                    "epoch": int(ep + 1),
                    "loss": round(mean_loss, 6),
                    "policy_loss": round(mean_policy, 6),
                    "value_loss": round(mean_value, 6),
                    "entropy": round(mean_entropy, 6),
                    "approx_kl": round(mean_kl, 6),
                    "clip_frac": round(mean_clip, 6),
                    "mean_reward": round(float(np.mean(rewards_np)) if rewards_np.size else 0.0, 6),
                    "episodes": int(meta.get("episodes", 0) or 0),
                    "steps": int(meta.get("steps", n_steps) or n_steps),
                }
            )
        if target_kl > 0 and mean_kl > target_kl:
            _progress_print(
                verbose,
                progress_prefix,
                f"early stop at epoch {ep + 1}: approx_kl={mean_kl:.6f} > target_kl={float(target_kl):.6f}",
            )
            logs.append(
                {
                    "epoch": int(ep + 1),
                    "event": "early_stop_target_kl",
                    "target_kl": float(target_kl),
                    "approx_kl": round(mean_kl, 6),
                }
            )
            break

    return _state_dict_to_json(model.state_dict()), logs


def _score_candidates(
    candidates: list[dict],
    *,
    state_seq: np.ndarray,
    model_state: dict,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_heads: int,
) -> np.ndarray:
    """对当前 respondent 的状态序列做一次前向，得到候选题概率。

    参数:
        candidates: 候选题列表，长度记为 `N`。
        state_seq: 当前 respondent 的历史状态窗口，
            形状通常为 `R^[L,D]`。
        model_state: 已训练好的模型参数字典。
        input_dim: 单步状态维度 `D`。
        output_dim: 动作空间维度 `N`。
        hidden_dim: 注意力内部通道维度 `H`。
        num_heads: 多头数 `M`。

    返回:
        np.ndarray:
            候选题概率向量 `probs ∈ R^[N]`。

    说明:
        这是“当前状态 -> 候选题打分”的核心推理函数。
        它不直接输出题目内容，而是输出每个 candidate 的概率，
        后续再结合：
        - expert prior
        - 参数越界 mask
        - epsilon 探索
        来完成最终抽题。
    """
    if not TORCH_AVAILABLE:
        return np.ones((len(candidates),), dtype=float) / max(1, len(candidates))
    model = SelfAttentionActorCritic(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
    )
    _load_state_dict_from_json(model, model_state)
    model.eval()
    seq_arr = np.asarray(state_seq, dtype=np.float32)
    if seq_arr.ndim == 1:
        seq_arr = seq_arr.reshape(1, -1)
    with torch.no_grad():
        x_t = torch.tensor(seq_arr, dtype=torch.float32).reshape(1, seq_arr.shape[0], seq_arr.shape[1])
        logits, _value = model(x_t)
        probs = torch.softmax(logits.squeeze(0), dim=0).cpu().numpy()
    if len(probs) < len(candidates):
        probs = np.concatenate([probs, np.zeros((len(candidates) - len(probs),), dtype=float)])
    return probs[: len(candidates)]


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
    """SelfAttention 设计生成主入口。

    整体逻辑与 dynamicPPO 保持一致：
    1) 动作空间来自统一 feasible combo pool；
    2) efficient 结果只作为专家先验；
    3) 训练时优先使用真实 rows，没有时退回到合成 respondent rollout；
    4) 生成阶段根据当前 respondent 的状态序列逐题打分并抽取。

    参数:
        payload: 当前 design 的标准化配置字典，通常包含：
            - `design_options["selfattention"]`
            - `beta_defaults`
            - `beta_bounds`
            - `design_spec`
        policy_state: 策略缓存字典；会被原地补充/更新，常见字段有：
            - `selfattention_state`
            - `candidate_signatures`
            - `input_dim / output_dim / seq_len`
            - `attr_dim_names / attr_categories / zone_categories`
        candidate_pool: 可行动作池，长度为 `N`；
            单个元素是一个候选题 `task`
        expert_result: efficient design 给出的专家经验结果，主要用于：
            - expert prior 融合
            - 初期保底抽题
        rows: 真实或合成的 respondent 作答记录列表。若可成功映射到
            candidate pool，则优先用作训练 rollout。
        current_respondent: 当前用于生成预览 block 的 respondent 字典
        data_dir: 数据目录，用于读取 pop stats / design 相关文件
        config: 全局配置字典
        verbose: 是否打印训练与生成过程日志

    返回:
        dict:
            {
                "comb": list[task],               # 生成出的预览 block
                "d_error": {"value": ...},
                "iteration_log": list[dict],      # epoch 训练日志
                "model_state": dict,              # 训练摘要
                "policy_state": dict              # 可继续在线更新的策略状态
            }

    重要维度:
        - `N = len(candidate_pool)`：动作空间维度
        - `D = input_dim`：单步状态维度
        - `L = seq_len`：历史窗口长度
        - 训练时：`states ∈ R^[T,D]`
        - 生成时：`state_seq ∈ R^[L,D]`
        - 网络输出：`logits ∈ R^[1,N]`，经 softmax 后得 `probs ∈ R^[N]`
    """
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

    design_options = payload.get("design_options", {}) or {}
    if not isinstance(design_options, dict):
        design_options = {}
    sopt = design_options.get("selfattention", {}) or {}
    if not isinstance(sopt, dict):
        sopt = {}
    scfg = _sa_cfg(config)
    dyn_cfg = config.get("dynamic_ppo", {}) if isinstance(config.get("dynamic_ppo", {}), dict) else {}

    tpr = int(sopt.get("tasks_per_round", 6) or 6)
    eps = float(sopt.get("explore_epsilon", scfg.get("explore_epsilon", 0.15)) or scfg.get("explore_epsilon", 0.15))
    seq_len = max(2, min(int(sopt.get("window_length", scfg.get("window_length", 16)) or scfg.get("window_length", 16)), 128))
    hidden_dim = max(16, int(sopt.get("hidden_dim", scfg.get("hidden_dim", 64)) or scfg.get("hidden_dim", 64)))
    num_heads = SelfAttentionActorCritic._normalize_heads(hidden_dim, int(sopt.get("num_heads", scfg.get("num_heads", 4)) or scfg.get("num_heads", 4)))
    train_n = int(sopt.get("train_respondents", scfg.get("train_respondents", 300)) or scfg.get("train_respondents", 300))
    epochs = int(sopt.get("train_epochs", scfg.get("train_epochs", 200)) or scfg.get("train_epochs", 200))
    lr = float(sopt.get("train_lr", scfg.get("train_lr", 0.03)) or scfg.get("train_lr", 0.03))
    seed = int(scfg.get("seed", 42))
    gamma = float(sopt.get("gamma", scfg.get("gamma", 0.99)) or scfg.get("gamma", 0.99))
    gae_lambda = float(sopt.get("gae_lambda", scfg.get("gae_lambda", 0.95)) or scfg.get("gae_lambda", 0.95))
    clip_eps = float(sopt.get("clip_eps", scfg.get("clip_eps", 0.2)) or scfg.get("clip_eps", 0.2))
    value_coef = float(sopt.get("value_coef", scfg.get("value_coef", 0.5)) or scfg.get("value_coef", 0.5))
    entropy_coef = float(sopt.get("entropy_coef", scfg.get("entropy_coef", 0.01)) or scfg.get("entropy_coef", 0.01))
    batch_size = int(sopt.get("batch_size", scfg.get("batch_size", 128)) or scfg.get("batch_size", 128))
    target_kl = float(sopt.get("target_kl", scfg.get("target_kl", 0.03)) or scfg.get("target_kl", 0.03))

    candidates = list(candidate_pool or [])
    if not candidates:
        return {
            "comb": [],
            "d_error": {"value": None},
            "iteration_log": [{"epoch": 0, "msg": "no feasible candidates"}],
            "model_state": {"trained": False},
            "policy_state": policy_state,
        }
    for cand in candidates:
        cand["sig"] = str(cand.get("sig") or _task_signature(cand))

    _progress_print(
        verbose,
        "selfattention",
        f"candidate_pool ready: size={len(candidates)} tasks_per_round={int(tpr)} epsilon={float(eps):.4f}",
    )

    # respondent 统计配置 -> 特征规范，用于后续统一构造状态向量维度。
    pop_stats = dyppo_shared._load_pop_stats(payload, data_dir=data_dir, config=config)
    feature_spec = dyppo_shared._build_feature_spec(pop_stats)
    input_dim = max(
        int(dyn_cfg.get("input_dim", feature_spec.get("feature_dim", 6)) or feature_spec.get("feature_dim", 6)),
        int(feature_spec.get("feature_dim", 6) or 6),
    )
    output_dim = max(1, len(candidates))
    beta_defaults = payload.get("beta_defaults", {}) if isinstance(payload.get("beta_defaults", {}), dict) else {}
    beta_bounds = payload.get("beta_bounds", {}) if isinstance(payload.get("beta_bounds", {}), dict) else {}
    spec = payload.get("design_spec", {}) if isinstance(payload.get("design_spec", {}), dict) else {}

    rollout_meta = {"episodes": 0, "steps": 0, "rows_used": 0, "mean_episode_reward": 0.0}
    synthetic_respondents: list[dict] = []
    if rows:
        _progress_print(
            verbose,
            "selfattention",
            f"building rollout from rows_jsonl: rows={len(rows)}",
        )
        rollouts = dyppo_shared._build_rollouts_from_rows(
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
        rollouts = dyppo_shared._empty_rollouts(input_dim)
    data_source = "rows_jsonl_rollout" if int(rollout_meta.get("steps", 0) or 0) > 0 else "synthetic_popsim_rollout"

    if int(rollout_meta.get("steps", 0) or 0) <= 0:
        _progress_print(
            verbose,
            "selfattention",
            f"rows rollout unavailable, fallback to synthetic respondents: n={int(train_n)}",
        )
        synthetic_respondents = dyppo_shared._sample_respondents(pop_stats, train_n, seed + int(policy_state.get("response_count", 0)))
        rollouts = dyppo_shared._build_synthetic_rollouts(
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

    _progress_print(
        verbose,
        "selfattention",
        (
            f"rollout prepared: source={data_source} steps={int(rollout_meta.get('steps', 0) or 0)} "
            f"episodes={int(rollout_meta.get('episodes', 0) or 0)} rows_used={int(rollout_meta.get('rows_used', 0) or 0)} "
            f"mean_episode_reward={float(rollout_meta.get('mean_episode_reward', 0.0) or 0.0):.6f}"
        ),
    )

    active_bound_mask = policy_state.get("candidate_bound_mask", {}) if isinstance(policy_state.get("candidate_bound_mask", {}), dict) else {}
    current_mnl_signal = policy_state.get("current_mnl_signal", {}) if isinstance(policy_state.get("current_mnl_signal", {}), dict) else {}
    observed_rows = list(rows or [])
    if observed_rows:
        # 对已观察到的 rows 做一次简化的参数重估，
        # 其结果不会直接替代神经网络，但会作为：
        # 1) 当前设计质量信号
        # 2) 参数越界惩罚 mask
        # 3) dashboard/调试输出
        obs_rows = dyppo_shared._collect_obs_rows_from_submission_rows(observed_rows, spec)
        signal = dyppo_shared._mnl_signal_from_obs_rows(
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

    init_state = policy_state.get("selfattention_state", {}) if isinstance(policy_state.get("selfattention_state", {}), dict) else None
    new_state, logs = _train_policy(
        rollouts,
        input_dim=input_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
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
        init_state=init_state,
        verbose=verbose,
        progress_prefix="selfattention/train",
    )

    score_respondent = current_respondent if isinstance(current_respondent, dict) and current_respondent else None
    scoring_state_source = "current_respondent"
    if not score_respondent and rows:
        score_respondent = dyppo_shared._row_respondent(rows[0]) if rows else None
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

    _progress_print(
        verbose,
        "selfattention",
        f"start scoring preview block: scoring_state_source={scoring_state_source}",
    )

    rng = np.random.default_rng(seed + 99 + int(policy_state.get("response_count", 0)))
    picked: list[dict] = []
    remain_idx = list(range(len(candidates)))
    history_states: list[np.ndarray] = []
    expert_stats = {"expert_overlap": 0, "expert_prior_weight": 0.0}
    bound_mask_stats = {"active_bound_mask_params": 0, "bound_masked_candidates": 0, "bound_mask_min_factor": 1.0}
    # 逐题位生成 block。每次循环只决定一个题位放哪道题。
    for _ in range(min(max(1, tpr), len(remain_idx))):
        state_x = dyppo_shared._task_state_vector(
            score_respondent,
            picked,
            beta_defaults=beta_defaults,
            feature_spec=feature_spec,
            step_index=len(picked),
            episode_len=tpr,
            input_dim=input_dim,
        )
        history_states.append(state_x)
        # 把当前 respondent 的历史状态堆成 `[L,D]` 序列。
        state_seq = _history_to_sequence(history_states, seq_len, input_dim)
        probs = _score_candidates(
            candidates,
            state_seq=state_seq,
            model_state=new_state,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )
        # 先融合 efficient 经验，再应用参数越界惩罚 mask。
        probs, expert_stats = dyppo_shared._blend_with_expert_prior(probs, candidates, expert_result)
        probs, bound_mask_stats = dyppo_shared._apply_candidate_bound_mask(probs, candidates, active_bound_mask)
        sub_probs = np.array([probs[i] for i in remain_idx], dtype=float)
        if np.sum(sub_probs) <= 0:
            sub_probs = np.ones_like(sub_probs) / len(sub_probs)
        else:
            sub_probs = sub_probs / np.sum(sub_probs)
        # epsilon-greedy:
        # - 以 epsilon 概率随机探索
        # - 否则按当前策略分布抽样
        if rng.random() < max(0.0, min(1.0, eps)):
            k = int(rng.integers(0, len(remain_idx)))
        else:
            k = int(rng.choice(len(remain_idx), p=sub_probs))
        idx = remain_idx.pop(k)
        picked.append(candidates[idx])

    final_tasks = []
    for i, task in enumerate(picked):
        final_tasks.append({**task, "block": 1, "row_in_block": i + 1, "id": f"preview_b1_r{i+1}"})

    _progress_print(
        verbose,
        "selfattention",
        f"preview block generated: rows={len(final_tasks)} expert_overlap={int(expert_stats.get('expert_overlap', 0) or 0)}",
    )

    policy_state["selfattention_state"] = new_state
    policy_state["input_dim"] = input_dim
    policy_state["output_dim"] = output_dim
    policy_state["attr_dim_names"] = deepcopy(feature_spec.get("attr_dim_names", []))
    policy_state["attr_categories"] = deepcopy(feature_spec.get("attr_categories", []))
    policy_state["zone_categories"] = deepcopy(feature_spec.get("zone_categories", []))
    policy_state["candidate_signatures"] = [str((c or {}).get("sig") or _task_signature(c)) for c in candidates]
    policy_state["seq_len"] = int(seq_len)
    policy_state["hidden_dim"] = int(hidden_dim)
    policy_state["num_heads"] = int(num_heads)
    policy_state["trained"] = True
    policy_state["train_epochs"] = int(epochs)
    policy_state["gamma"] = float(gamma)
    policy_state["gae_lambda"] = float(gae_lambda)
    policy_state["clip_eps"] = float(clip_eps)
    policy_state["value_coef"] = float(value_coef)
    policy_state["entropy_coef"] = float(entropy_coef)
    policy_state["target_kl"] = float(target_kl)
    policy_state["batch_size"] = int(batch_size)
    policy_state["response_count"] = int(policy_state.get("response_count", 0))
    policy_state["candidate_bound_mask"] = deepcopy(active_bound_mask) if isinstance(active_bound_mask, dict) else {}
    policy_state["current_mnl_signal"] = deepcopy(current_mnl_signal) if isinstance(current_mnl_signal, dict) else {}

    return {
        "comb": final_tasks,
        "d_error": {"value": None},
        "iteration_log": logs,
        "model_state": {
            "trained": True,
            "backend": "pytorch_selfattention",
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
            "seq_len": int(seq_len),
            "hidden_dim": int(hidden_dim),
            "num_heads": int(num_heads),
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
    """在 respondent 提交完一整份 block 后做一次在线 PPO 更新。

    参数:
        payload: 当前 design 配置字典
        policy_state: 已训练好的策略状态，必须至少包含：
            - `selfattention_state`
            - `candidate_signatures`
            - `input_dim / output_dim / seq_len`
        tasks: 本次 respondent 实际看到的题目列表 `list[task]`
        choices: respondent 的答案字典，通常形如：
            `{task_id: chosen_alt_name, ...}`
        respondent: respondent 特征字典
        config: 全局配置字典
        historical_rows: 当前 respondent 提交前，系统里已有的历史 rows，
            用于计算“加上本次回答前后”的 MNL 信号变化
        verbose: 是否打印在线更新日志

    返回:
        dict:
            {
                "updated": bool,
                "policy_version": int,
                "mean_episode_reward": float,
                "steps": int,
                "online_epochs": int,
                "loss": float | None,
                "approx_kl": float | None,
                "reward_metrics": {...}
            }

    关键维度:
        - `len(tasks) = K`：本次 respondent 的 block 题数
        - 构造出的在线 rollout：
          - `states ∈ R^[K,D]`
          - `actions ∈ Z^[K]`
          - `rewards ∈ R^[K]`
          - `dones ∈ {0,1}^[K]`

    说明:
        这里的在线更新不是重新从零训练，而是：
        1. 把本次 block 回答映射成一个 episode rollout
        2. 以当前 `selfattention_state` 为初始权重
        3. 用较小学习率和较少 epoch 做一次增量 PPO 更新
    """
    if not TORCH_AVAILABLE:
        return {"updated": False, "reason": "torch not installed"}
    if not tasks:
        return {"updated": False}

    input_dim = int(policy_state.get("input_dim", 6) or 6)
    output_dim = int(policy_state.get("output_dim", max(1, len(tasks))) or max(1, len(tasks)))
    seq_len = int(policy_state.get("seq_len", 16) or 16)
    hidden_dim = int(policy_state.get("hidden_dim", 64) or 64)
    num_heads = int(policy_state.get("num_heads", 4) or 4)
    scfg = _sa_cfg(config)
    sopt = ((payload.get("design_options", {}) or {}).get("selfattention", {}) or {}) if isinstance(payload, dict) else {}
    gamma = float(policy_state.get("gamma", sopt.get("gamma", scfg.get("gamma", 0.99))) or scfg.get("gamma", 0.99))
    gae_lambda = float(policy_state.get("gae_lambda", sopt.get("gae_lambda", scfg.get("gae_lambda", 0.95))) or scfg.get("gae_lambda", 0.95))
    clip_eps = float(policy_state.get("clip_eps", sopt.get("clip_eps", scfg.get("clip_eps", 0.2))) or scfg.get("clip_eps", 0.2))
    value_coef = float(policy_state.get("value_coef", sopt.get("value_coef", scfg.get("value_coef", 0.5))) or scfg.get("value_coef", 0.5))
    entropy_coef = float(policy_state.get("entropy_coef", sopt.get("entropy_coef", scfg.get("entropy_coef", 0.01))) or scfg.get("entropy_coef", 0.01))
    target_kl = float(policy_state.get("target_kl", sopt.get("target_kl", scfg.get("target_kl", 0.03))) or scfg.get("target_kl", 0.03))
    batch_size = int(
        sopt.get("online_batch_size", policy_state.get("online_batch_size", scfg.get("online_batch_size", scfg.get("batch_size", 128))))
        or policy_state.get("online_batch_size", scfg.get("online_batch_size", scfg.get("batch_size", 128)))
    )
    online_lr = float(sopt.get("online_lr", scfg.get("online_lr", 0.005)) or scfg.get("online_lr", 0.005))
    online_epochs = int(sopt.get("online_epochs", scfg.get("online_epochs", 2)) or scfg.get("online_epochs", 2))
    sa_state = policy_state.get("selfattention_state", {}) if isinstance(policy_state.get("selfattention_state", {}), dict) else {}
    if not sa_state:
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

    current_respondent = respondent if isinstance(respondent, dict) and respondent else {
        "respondent_id": "online_missing",
        "zone_id": _MISSING_ATTR_TOKEN,
        "attr_segments": [_MISSING_ATTR_TOKEN] * max(1, len(feature_spec.get("attr_dim_names", []))),
    }
    beta_defaults = payload.get("beta_defaults", {}) if isinstance(payload.get("beta_defaults", {}), dict) else {}
    beta_bounds = payload.get("beta_bounds", {}) if isinstance(payload.get("beta_bounds", {}), dict) else {}
    spec = payload.get("design_spec", {}) if isinstance(payload.get("design_spec", {}), dict) else {}

    # 先把“实际发出的 task”映射回 candidate pool 中的动作索引。
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

    _progress_print(
        verbose,
        "selfattention/online",
        f"start: respondent_id={str((current_respondent or {}).get('respondent_id', ''))} steps={len(valid_steps)} online_epochs={int(online_epochs)} lr={float(online_lr):.6f}",
    )

    episode_tasks = [task for task, _action_idx, _chosen_alt in valid_steps]
    episode_choices = {
        str((task or {}).get("id", "")).strip(): chosen_alt
        for task, _action_idx, chosen_alt in valid_steps
        if str((task or {}).get("id", "")).strip()
    }
    # 对整份 block 计算全局奖励：
    # 既包含单题的选择信息，也包含 block 级别的 d-error / 熵 / pseudo R^2 / 参数越界等。
    reward_metrics = dyppo_shared._questionnaire_reward_metrics(
        episode_tasks,
        current_respondent,
        episode_choices,
        spec=spec,
        beta_defaults=beta_defaults,
        beta_bounds=beta_bounds,
        prior_obs_rows=dyppo_shared._collect_obs_rows_from_submission_rows(list(historical_rows or []), spec),
        config=config,
    )
    episode_bonus = float(reward_metrics.get("reward", 0.0) or 0.0)

    # 把整份 block 回答重新编码成在线 rollout。
    states: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[float] = []
    prior_tasks: list[dict] = []
    for step_idx, (task, action_idx, chosen_alt) in enumerate(valid_steps):
        state = dyppo_shared._task_state_vector(
            current_respondent,
            prior_tasks,
            beta_defaults=beta_defaults,
            feature_spec=feature_spec,
            step_index=step_idx,
            episode_len=len(valid_steps),
            input_dim=input_dim,
        )
        # block 级奖励只在最后一步注入，前面步骤只保留单题级 reward。
        terminal_bonus = float(episode_bonus) if step_idx == len(valid_steps) - 1 else 0.0
        reward = dyppo_shared._task_step_reward(
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

    # 单个 respondent 的 block -> 一个 episode
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
    seed = int(policy_state.get("response_count", 0) or 0) + 1707
    new_state, logs = _train_policy(
        rollouts,
        input_dim=input_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        seed=seed,
        epochs=online_epochs,
        lr=online_lr,
        clip_eps=clip_eps,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        gamma=gamma,
        gae_lambda=gae_lambda,
        batch_size=max(1, min(batch_size, len(actions))),
        target_kl=target_kl,
        init_state=sa_state,
        verbose=verbose,
        progress_prefix="selfattention/online",
    )

    policy_state["selfattention_state"] = new_state
    policy_state["hidden_dim"] = int(hidden_dim)
    policy_state["num_heads"] = int(num_heads)
    policy_state["seq_len"] = int(seq_len)
    policy_state["online_updates"] = int(policy_state.get("online_updates", 0)) + 1
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

    last_log = logs[-1] if logs else {}
    _progress_print(
        verbose,
        "selfattention/online",
        (
            f"done: policy_version={int(policy_state.get('response_count', 0))} "
            f"loss={last_log.get('loss')} approx_kl={last_log.get('approx_kl')} "
            f"mean_episode_reward={float(episode_bonus):.6f}"
        ),
    )
    return {
        "updated": True,
        "policy_version": int(policy_state.get("response_count", 0)),
        "mean_episode_reward": round(float(episode_bonus), 6),
        "steps": len(actions),
        "online_epochs": int(online_epochs),
        "loss": last_log.get("loss"),
        "approx_kl": last_log.get("approx_kl"),
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
