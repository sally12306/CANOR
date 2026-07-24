import copy
import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import configparser
import gym
import h5py
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR

from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.info import Collision, ReachGoal, Timeout
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState

import matplotlib.pyplot as plt
TensorBatch = List[torch.Tensor]
EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class TrainConfig:
    # ---------------- Experiment ----------------
    device: str = "cuda:0"
    env: str = "CrowdSim-v0"
    seed: int = 0
    eval_freq: int = 1000
    n_episodes: int = 50
    final_test_episodes: int = 500
    max_timesteps: int = 100000

    # ---------------- 7-human environment ----------------
    human_num: int = 7
    human_state_dim: int = 13
    env_config_path: str = (
        r"crowd_nav/configs/env.config"
    )
    offline_dataset_path: str = (
        r""
    )
    checkpoints_path: str = (
        r""
    )
    load_model: str = r""

    # ReplayBuffer放CPU，避免91维、百万容量时占满显存
    replay_buffer_device: str = "cpu"

    # ---------------- IQL ----------------
    buffer_size: int = 1_000_000
    batch_size: int = 256
    discount: float = 0.99

    # 目标Q网络软更新系数，不是IQL expectile参数
    tau: float = 0.005

    # 当前实现使用 exp(beta * advantage)，因此beta是“逆温度”
    beta: float = 3.0

    # expectile regression参数，必须处于(0, 1)
    iql_tau: float = 0.7

    iql_deterministic: bool = False
    vf_lr: float = 3e-4
    qf_lr: float = 3e-4
    actor_lr: float = 3e-4
    actor_dropout: Optional[float] = None

    # ---------------- Online fine-tuning: selective replay V4 ----------------
    online_buffer_max_size: int = 100_000
    success_buffer_max_size: int = 100_000
    online_episodes: int = 50_000

    # Critic在约1000条在线transition后开始学习。
    min_online_steps: int = 1_000

    # Actor只在成功轨迹buffer足够后参与在线更新。
    min_success_actor_steps: int = 2_000
    success_actor_min_return: float = 15.0

    # 每回合更新两次Q/V；Actor每两次更新一次，即每回合约一次。
    updates_per_episode: int = 1
    actor_update_start_steps: int = 5_000
    actor_update_interval: int = 1

    # Q/V使用全部在线轨迹，Actor使用离线+成功在线轨迹。
    critic_online_ratio_start: float = 0.25
    critic_online_ratio_end: float = 0.40
    critic_online_ratio_ramp_episodes: int = 10_000

    actor_online_ratio_start: float = 0.20
    actor_online_ratio_end: float = 0.45
    actor_online_ratio_ramp_episodes: int = 15_000

    # 在线学习率。
    finetune_v_lr: float = 1e-5
    finetune_q_lr: float = 1e-5
    finetune_actor_lr: float = 8e-6

    # V3后期0.05噪声仍持续产生失败轨迹，V4逐步降到0.02。
    initial_noise_scale: float = 0.08
    final_noise_scale: float = 0.02
    noise_decay_episodes: int = 20_000

    # 保持与离线IQL一致的优势定义，但降低极端权重上限。
    online_beta: float = 3.0
    online_normalize_advantage: bool = False
    online_advantage_clip: float = 10.0
    online_exp_adv_max: float = 20.0

    # 梯度裁剪。
    value_grad_clip: float = 5.0
    q_grad_clip: float = 5.0
    actor_grad_clip: float = 5.0

    # 继续恢复离线Adam状态。
    restore_offline_optimizer_state: bool = True

    # EMA策略用于在线采样、评估和最终部署。
    use_ema_actor: bool = True
    ema_decay: float = 0.995
    ema_rollout_start_steps: int = 5_000

    # 达到较高成功率后进入巩固阶段，降低探索和Actor更新强度。
    consolidation_success_threshold: float = 0.78
    consolidation_noise_cap: float = 0.02
    consolidation_critic_ratio_cap: float = 0.30
    consolidation_actor_ratio: float = 0.40
    consolidation_actor_lr: float = 4e-6
    consolidation_actor_update_interval: int = 4

    # 评估保护：连续两次明显下降后，恢复完整trainer与EMA策略。
    eval_guard_enabled: bool = True
    eval_guard_drop_tolerance: float = 0.06
    eval_guard_patience: int = 2
    eval_guard_freeze_episodes: int = 500
    eval_guard_lr_decay: float = 0.80
    eval_guard_min_actor_lr: float = 2e-6
    eval_guard_noise_cap: float = 0.015
    eval_guard_critic_ratio_cap: float = 0.25

    # 日志、曲线和模型选择。
    online_eval_freq: int = 500
    online_eval_episodes: int = 500
    curve_window: int = 200
    curve_save_freq: int = 1000
    save_best_online: bool = True


# ============================================================
# Logging and configuration helpers
# ============================================================
def setup_logger(log_dir: str, name: str = "iql") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(
        os.path.join(log_dir, "train_eval_iql.log"),
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def get_state_dim(human_num: int, human_state_dim: int = 13) -> int:
    if human_num <= 0:
        raise ValueError("human_num必须大于0")
    if human_state_dim <= 0:
        raise ValueError("human_state_dim必须大于0")
    return human_num * human_state_dim


def load_crowdnav_env_config(
    env_config_path: str,
    human_num: int,
) -> configparser.RawConfigParser:
    env_config = configparser.RawConfigParser()
    loaded = env_config.read(env_config_path, encoding="utf-8")
    if not loaded:
        raise FileNotFoundError("未找到环境配置文件: {}".format(env_config_path))
    if not env_config.has_section("sim"):
        raise KeyError("环境配置缺少[sim]段: {}".format(env_config_path))

    # 不修改磁盘文件，只覆盖本次运行读取到的配置
    env_config.set("sim", "human_num", str(human_num))
    return env_config


def validate_dataset(
    dataset: Dict[str, np.ndarray],
    state_dim: int,
    dataset_path: str,
) -> None:
    required_keys = (
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "terminals",
    )
    missing = [key for key in required_keys if key not in dataset]
    if missing:
        raise KeyError(
            "离线数据集缺少字段{}: {}".format(missing, dataset_path)
        )

    observations = dataset["observations"]
    next_observations = dataset["next_observations"]
    if observations.ndim != 2 or next_observations.ndim != 2:
        raise ValueError(
            "observations和next_observations必须为二维数组，实际为{}/{}".format(
                observations.shape, next_observations.shape
            )
        )

    obs_dim = int(observations.shape[1])
    next_obs_dim = int(next_observations.shape[1])
    if obs_dim != state_dim or next_obs_dim != state_dim:
        raise ValueError(
            "7人模型期望状态维数为{}，但数据集为{}/{}。"
            "请使用Crowd_nav_7.hdf5，不能继续使用65维Crowd_nav_5.hdf5。"
            "数据路径: {}".format(
                state_dim, obs_dim, next_obs_dim, dataset_path
            )
        )

    n_transitions = int(observations.shape[0])
    for key in required_keys[1:]:
        if int(dataset[key].shape[0]) != n_transitions:
            raise ValueError(
                "数据长度不一致: observations={}, {}={}".format(
                    n_transitions, key, dataset[key].shape[0]
                )
            )

    if dataset["actions"].ndim != 2 or dataset["actions"].shape[1] != 2:
        raise ValueError(
            "动作数据应为[N, 2]，实际为{}".format(dataset["actions"].shape)
        )


def validate_state_vector(
    state: np.ndarray,
    state_dim: int,
    context: str,
) -> None:
    actual_dim = int(np.asarray(state).reshape(-1).shape[0])
    if actual_dim != state_dim:
        raise ValueError(
            "{}状态维度错误：期望{}维，实际{}维。"
            "请检查env.config中的[sim] human_num是否为7。".format(
                context, state_dim, actual_dim
            )
        )


def set_seed(
    seed: int,
    env: Optional[gym.Env] = None,
    deterministic_torch: bool = False,
) -> None:
    if env is not None:
        env.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            (1.0 - tau) * target_param.data + tau * source_param.data
        )


def to_np(tensor: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if tensor is None:
        return None
    if tensor.nelement() == 0:
        return np.array([])
    return tensor.detach().cpu().numpy()


# ============================================================
# Replay buffers
# ============================================================
class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = int(buffer_size)
        self._pointer = 0
        self._size = 0
        self._device = device

        self._states = torch.zeros(
            (self._buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (self._buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros(
            (self._buffer_size, 1), dtype=torch.float32, device=device
        )
        self._next_states = torch.zeros(
            (self._buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros(
            (self._buffer_size, 1), dtype=torch.float32, device=device
        )

    def _to_tensor(self, data: Any) -> torch.Tensor:
        return torch.as_tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]) -> None:
        if self._size != 0:
            raise ValueError("不能向非空ReplayBuffer重复载入离线数据")

        n_transitions = int(data["observations"].shape[0])
        if n_transitions > self._buffer_size:
            raise ValueError(
                "ReplayBuffer容量{}小于数据量{}".format(
                    self._buffer_size, n_transitions
                )
            )

        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(
            data["rewards"].reshape(-1, 1)
        )
        self._next_states[:n_transitions] = self._to_tensor(
            data["next_observations"]
        )
        self._dones[:n_transitions] = self._to_tensor(
            data["terminals"].reshape(-1, 1)
        )

        self._size = n_transitions
        self._pointer = n_transitions % self._buffer_size
        print("Dataset size: {}".format(n_transitions))

    def sample(self, batch_size: int) -> TensorBatch:
        if self._size <= 0:
            raise RuntimeError("不能从空ReplayBuffer采样")
        indices = np.random.randint(0, self._size, size=int(batch_size))
        return [
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._dones[indices],
        ]

    def add_transition(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: float,
    ) -> None:
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = float(reward)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = float(done)

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)


class DualReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        offline_buffer_size: int,
        online_buffer_size: int,
        device: str = "cpu",
    ):
        self.offline_buffer = ReplayBuffer(
            state_dim, action_dim, offline_buffer_size, device
        )
        self.online_buffer = ReplayBuffer(
            state_dim, action_dim, online_buffer_size, device
        )

    def load_offline_dataset(self, data: Dict[str, np.ndarray]) -> None:
        self.offline_buffer.load_d4rl_dataset(data)

    def add_online_transition(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: float,
    ) -> None:
        self.online_buffer.add_transition(
            state, action, reward, next_state, done
        )

    def sample(self, batch_size: int, online_ratio: float = 0.5) -> TensorBatch:
        if not 0.0 <= online_ratio <= 1.0:
            raise ValueError("online_ratio必须在[0, 1]内")

        n_online = int(round(batch_size * online_ratio))
        n_offline = int(batch_size) - n_online

        if n_online == 0:
            return self.offline_buffer.sample(batch_size)
        if n_offline == 0:
            if self.online_buffer._size < batch_size:
                return self.offline_buffer.sample(batch_size)
            return self.online_buffer.sample(batch_size)
        if self.online_buffer._size < n_online:
            return self.offline_buffer.sample(batch_size)

        offline_batch = self.offline_buffer.sample(n_offline)
        online_batch = self.online_buffer.sample(n_online)
        return [
            torch.cat((offline_item, online_item), dim=0)
            for offline_item, online_item in zip(offline_batch, online_batch)
        ]


# ============================================================
# State preprocessing
# ============================================================
def transform(state: JointState, device: str) -> torch.Tensor:
    if len(state.human_states) == 0:
        raise ValueError("当前环境没有行人状态，无法构造人群导航输入")

    state_tensor = torch.cat(
        [
            torch.tensor(
                [state.self_state + human_state],
                dtype=torch.float32,
                device=device,
            )
            for human_state in state.human_states
        ],
        dim=0,
    )
    return rotate(state_tensor)


def rotate(state: torch.Tensor) -> torch.Tensor:
    batch = state.shape[0]
    dx = (state[:, 5] - state[:, 0]).reshape(batch, -1)
    dy = (state[:, 6] - state[:, 1]).reshape(batch, -1)
    rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

    dg = torch.norm(torch.cat((dx, dy), dim=1), p=2, dim=1, keepdim=True)
    v_pref = state[:, 7].reshape(batch, -1)
    vx = (
        state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)
    ).reshape(batch, -1)
    vy = (
        state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)
    ).reshape(batch, -1)

    radius = state[:, 4].reshape(batch, -1)
    theta = torch.zeros_like(v_pref)
    vx1 = (
        state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)
    ).reshape(batch, -1)
    vy1 = (
        state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)
    ).reshape(batch, -1)
    px1 = (
        (state[:, 9] - state[:, 0]) * torch.cos(rot)
        + (state[:, 10] - state[:, 1]) * torch.sin(rot)
    ).reshape(batch, -1)
    py1 = (
        (state[:, 10] - state[:, 1]) * torch.cos(rot)
        - (state[:, 9] - state[:, 0]) * torch.sin(rot)
    ).reshape(batch, -1)
    radius1 = state[:, 13].reshape(batch, -1)
    radius_sum = radius + radius1
    da = torch.norm(
        torch.cat(
            (
                (state[:, 0] - state[:, 9]).reshape(batch, -1),
                (state[:, 1] - state[:, 10]).reshape(batch, -1),
            ),
            dim=1,
        ),
        p=2,
        dim=1,
        keepdim=True,
    )

    # 每个行人输出13维特征
    return torch.cat(
        (
            dg,
            v_pref,
            theta,
            radius,
            vx,
            vy,
            px1,
            py1,
            vx1,
            vy1,
            radius1,
            da,
            radius_sum,
        ),
        dim=1,
    )


# ============================================================
# Dynamic attention and neural networks
# ============================================================
class DynamicSelfAttention(nn.Module):
    """
    将[B, human_num*13]动态重塑为[B, human_num, 13]。

    原项目的crowd_nav.policy.self_attention.py把状态写死为65维，
    因而7人91维会在reshape(-1, 65)处报错。本类不写死人数，
    对每个行人编码并做注意力池化，最终固定输出56维。
    """

    def __init__(
        self,
        per_human_dim: int = 13,
        hidden_dim: int = 128,
        output_dim: int = 56,
        attention_hidden_dim: int = 64,
    ):
        super().__init__()
        self.per_human_dim = per_human_dim
        self.output_dim = output_dim

        self.human_encoder = nn.Sequential(
            nn.Linear(per_human_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )
        self.attention_network = nn.Sequential(
            nn.Linear(output_dim, attention_hidden_dim),
            nn.ReLU(),
            nn.Linear(attention_hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if state.dim() != 2:
            state = state.reshape(state.shape[0], -1)

        flat_dim = int(state.shape[1])
        if flat_dim % self.per_human_dim != 0:
            raise ValueError(
                "DynamicSelfAttention输入维数{}不是{}的整数倍，输入形状为{}".format(
                    flat_dim, self.per_human_dim, tuple(state.shape)
                )
            )

        human_num = flat_dim // self.per_human_dim
        if human_num <= 0:
            raise ValueError("DynamicSelfAttention检测到的行人数为0")

        human_states = state.reshape(
            state.shape[0], human_num, self.per_human_dim
        )
        human_features = self.human_encoder(human_states)
        attention_logits = self.attention_network(human_features).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=1)
        pooled = torch.sum(
            human_features * attention_weights.unsqueeze(-1), dim=1
        )
        return pooled


class Squeeze(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims: List[int],
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Optional[Callable[[], nn.Module]] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("MLP至少需要输入维和输出维")

        layers = []
        for index in range(len(dims) - 2):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            layers.append(activation_fn())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("squeeze_output=True时最后一维必须为1")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        del state_dim  # 人数由DynamicSelfAttention动态识别
        self.attention = DynamicSelfAttention(output_dim=56)
        self.net = MLP(
            [56] + [hidden_dim] * n_hidden + [act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = float(max_action)
        self.kinematics = "holonomic"
        self.multiagent_training = True

    def forward(self, observations: torch.Tensor) -> Normal:
        features = self.attention(observations)
        mean = self.net(features)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(
        self,
        state: np.ndarray,
        device: str = "cuda",
        deterministic: Optional[bool] = None,
    ) -> np.ndarray:
        state_tensor = torch.as_tensor(
            np.asarray(state).reshape(1, -1),
            dtype=torch.float32,
            device=device,
        )
        distribution = self(state_tensor)
        if deterministic is None:
            deterministic = not self.training
        action = distribution.mean if deterministic else distribution.sample()
        action = torch.clamp(
            self.max_action * action,
            -self.max_action,
            self.max_action,
        )
        return action.detach().cpu().numpy().reshape(-1)


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.attention = DynamicSelfAttention(output_dim=56)
        self.net = MLP(
            [56] + [hidden_dim] * n_hidden + [act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = float(max_action)
        self.kinematics = "holonomic"
        self.multiagent_training = True
        self.state_dim = state_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(self.attention(observations))

    @torch.no_grad()
    def act(
        self,
        state: np.ndarray,
        device: str = "cpu",
        deterministic: Optional[bool] = None,
    ) -> np.ndarray:
        del deterministic
        state_tensor = torch.as_tensor(
            np.asarray(state).reshape(1, -1),
            dtype=torch.float32,
            device=device,
        )
        action = torch.clamp(
            self(state_tensor) * self.max_action,
            -self.max_action,
            self.max_action,
        )
        return action.detach().cpu().numpy().reshape(-1)


class TwinQ(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
    ):
        super().__init__()
        del state_dim
        self.attention = DynamicSelfAttention(output_dim=56)
        dims = [56 + action_dim] + [hidden_dim] * n_hidden + [1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.attention(state)
        state_action = torch.cat((features, action), dim=1)
        return self.q1(state_action), self.q2(state_action)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        q1, q2 = self.both(state, action)
        return torch.minimum(q1, q2)


class ValueFunction(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
    ):
        super().__init__()
        del state_dim
        self.attention = DynamicSelfAttention(output_dim=56)
        self.v = MLP(
            [56] + [hidden_dim] * n_hidden + [1],
            squeeze_output=True,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(self.attention(state))


# ============================================================
# IQL algorithm
# ============================================================
def asymmetric_l2_loss(u: torch.Tensor, expectile: float) -> torch.Tensor:
    if not 0.0 < expectile < 1.0:
        raise ValueError("iql_tau/expectile必须严格位于(0, 1)")
    weight = torch.abs(expectile - (u < 0).float())
    return torch.mean(weight * u.pow(2))


class ImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: Optional[int] = 1_000_000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        if beta < 0.0:
            raise ValueError("beta不能小于0")
        if not 0.0 < iql_tau < 1.0:
            raise ValueError("iql_tau必须严格位于(0, 1)")

        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor

        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = None
        if max_steps is not None and int(max_steps) > 0:
            self.actor_lr_schedule = CosineAnnealingLR(
                self.actor_optimizer,
                T_max=max(1, int(max_steps)),
            )

        self.iql_tau = float(iql_tau)
        self.beta = float(beta)
        self.discount = float(discount)
        self.tau = float(tau)
        self.total_it = 0
        self.device = device

        # 默认保持原始离线IQL行为；在线脚本会显式打开稳定化选项。
        self.normalize_advantage = False
        self.advantage_clip = 20.0
        self.exp_adv_max = float(EXP_ADV_MAX)
        self.value_grad_clip: Optional[float] = None
        self.q_grad_clip: Optional[float] = None
        self.actor_grad_clip: Optional[float] = None

    def reset_actor_scheduler(self, max_steps: Optional[int]) -> None:
        if max_steps is None or int(max_steps) <= 0:
            self.actor_lr_schedule = None
        else:
            self.actor_lr_schedule = CosineAnnealingLR(
                self.actor_optimizer,
                T_max=max(1, int(max_steps)),
            )

    def _update_v(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict[str, float],
    ) -> torch.Tensor:
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        value = self.vf(observations)
        advantage = target_q - value
        value_loss = asymmetric_l2_loss(advantage, self.iql_tau)

        self.v_optimizer.zero_grad()
        value_loss.backward()
        if self.value_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.vf.parameters(), float(self.value_grad_clip)
            )
        self.v_optimizer.step()

        log_dict["value_loss"] = float(value_loss.item())
        log_dict["adv_mean"] = float(advantage.mean().item())
        log_dict["adv_std"] = float(advantage.std(unbiased=False).item())
        return advantage.detach()

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict[str, float],
    ) -> None:
        targets = rewards + (
            1.0 - terminals.float()
        ) * self.discount * next_v.detach()

        q1, q2 = self.qf.both(observations, actions)
        q_loss = 0.5 * (
            F.mse_loss(q1, targets) + F.mse_loss(q2, targets)
        )

        self.q_optimizer.zero_grad()
        q_loss.backward()
        if self.q_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.qf.parameters(), float(self.q_grad_clip)
            )
        self.q_optimizer.step()
        soft_update(self.q_target, self.qf, self.tau)

        log_dict["q_loss"] = float(q_loss.item())
        log_dict["q_mean"] = float(torch.minimum(q1, q2).mean().item())
        log_dict["target_q_mean"] = float(targets.mean().item())

    def _update_policy(
        self,
        advantage: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict[str, float],
    ) -> None:
        weight_advantage = advantage.detach()
        if self.normalize_advantage:
            adv_mean = weight_advantage.mean()
            adv_std = weight_advantage.std(unbiased=False).clamp_min(1e-6)
            weight_advantage = (weight_advantage - adv_mean) / adv_std

        weight_advantage = torch.clamp(
            weight_advantage,
            min=-float(self.advantage_clip),
            max=float(self.advantage_clip),
        )
        exp_adv = torch.exp(self.beta * weight_advantage).clamp(
            max=float(self.exp_adv_max)
        )
        policy_out = self.actor(observations)

        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(dim=-1)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError(
                    "策略输出形状{}与动作形状{}不一致".format(
                        tuple(policy_out.shape), tuple(actions.shape)
                    )
                )
            bc_losses = torch.sum((policy_out - actions).pow(2), dim=1)
        else:
            raise TypeError("不支持的策略输出类型: {}".format(type(policy_out)))

        policy_loss = torch.mean(exp_adv * bc_losses)

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        if self.actor_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), float(self.actor_grad_clip)
            )
        self.actor_optimizer.step()
        if self.actor_lr_schedule is not None:
            self.actor_lr_schedule.step()

        log_dict["actor_loss"] = float(policy_loss.item())
        log_dict["exp_adv_mean"] = float(exp_adv.mean().item())
        log_dict["exp_adv_max"] = float(exp_adv.max().item())
        log_dict["weight_adv_mean"] = float(weight_advantage.mean().item())
        log_dict["weight_adv_std"] = float(
            weight_advantage.std(unbiased=False).item()
        )

    def train(
        self,
        batch: TensorBatch,
        update_actor: bool = True,
        actor_batch: Optional[TensorBatch] = None,
    ) -> Dict[str, float]:
        if len(batch) < 5:
            raise ValueError("IQL训练batch至少应包含5项")

        observations, actions, rewards, next_observations, dones = batch[:5]
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)

        self.total_it += 1
        log_dict: Dict[str, float] = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)

        critic_advantage = self._update_v(observations, actions, log_dict)
        self._update_q(
            next_v,
            observations,
            actions,
            rewards,
            dones,
            log_dict,
        )

        if update_actor:
            if actor_batch is None:
                actor_observations = observations
                actor_actions = actions
                actor_advantage = critic_advantage
            else:
                if len(actor_batch) < 2:
                    raise ValueError("actor_batch至少应包含状态和动作")
                actor_observations = actor_batch[0]
                actor_actions = actor_batch[1]
                with torch.no_grad():
                    actor_advantage = (
                        self.q_target(actor_observations, actor_actions)
                        - self.vf(actor_observations)
                    )

            self._update_policy(
                actor_advantage,
                actor_observations,
                actor_actions,
                log_dict,
            )
            log_dict["actor_updated"] = 1.0
            log_dict["actor_batch_size"] = float(actor_observations.shape[0])
        else:
            log_dict["actor_loss"] = 0.0
            log_dict["exp_adv_mean"] = 0.0
            log_dict["exp_adv_max"] = 0.0
            log_dict["weight_adv_mean"] = 0.0
            log_dict["weight_adv_std"] = 0.0
            log_dict["actor_updated"] = 0.0
            log_dict["actor_batch_size"] = 0.0
        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "qf": self.qf.state_dict(),
            "q_target": self.q_target.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
            "beta": self.beta,
            "iql_tau": self.iql_tau,
            "normalize_advantage": self.normalize_advantage,
            "advantage_clip": self.advantage_clip,
            "exp_adv_max": self.exp_adv_max,
        }
        if self.actor_lr_schedule is not None:
            state["actor_lr_schedule"] = self.actor_lr_schedule.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.qf.load_state_dict(state_dict["qf"])
        if "q_target" in state_dict:
            self.q_target.load_state_dict(state_dict["q_target"])
        else:
            self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(
                self.device
            )

        self.vf.load_state_dict(state_dict["vf"])
        self.actor.load_state_dict(state_dict["actor"])

        if "q_optimizer" in state_dict:
            self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        if "v_optimizer" in state_dict:
            self.v_optimizer.load_state_dict(state_dict["v_optimizer"])
        if "actor_optimizer" in state_dict:
            self.actor_optimizer.load_state_dict(
                state_dict["actor_optimizer"]
            )
        if (
            self.actor_lr_schedule is not None
            and "actor_lr_schedule" in state_dict
        ):
            self.actor_lr_schedule.load_state_dict(
                state_dict["actor_lr_schedule"]
            )

        self.total_it = int(state_dict.get("total_it", 0))


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate_actor(
    actor: nn.Module,
    device: str,
    n_episodes: int,
    seed: int,
    human_num: int,
    human_state_dim: int,
    env_config_path: str,
    phase: str = "test",
    test_case_start: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    global_step: Optional[int] = None,
) -> Dict[str, float]:
    state_dim = get_state_dim(human_num, human_state_dim)
    env_config = load_crowdnav_env_config(env_config_path, human_num)
    eval_env = gym.make("CrowdSim-v0")
    eval_env.configure(env_config)
    eval_env.seed(seed)

    robot = Robot(env_config, "robot")
    eval_env.set_robot(robot)
    robot.set_policy(actor)

    was_training = actor.training
    actor.eval()

    success = 0
    collision = 0
    timeout = 0
    nav_time_total = 0.0
    reward_total = 0.0

    for episode_index in range(int(n_episodes)):
        if test_case_start is None:
            observation = eval_env.reset(phase=phase)
        else:
            observation = eval_env.reset(
                phase=phase,
                test_case=test_case_start + episode_index,
            )

        done = False
        discounted_reward = 0.0
        gamma = 1.0
        info = None

        while not done:
            joint_state = JointState(robot.get_full_state(), observation)
            state = to_np(
                transform(joint_state, device).reshape(1, -1).squeeze(0)
            )
            validate_state_vector(state, state_dim, "评估")

            action = actor.act(
                state,
                device=device,
                deterministic=True,
            )
            observation, reward, done, info = eval_env.step(
                ActionXY(float(action[0]), float(action[1]))
            )
            discounted_reward += float(reward) * gamma
            gamma *= 0.99

        if isinstance(info, ReachGoal):
            success += 1
            nav_time_total += float(eval_env.global_time)
        elif isinstance(info, Collision):
            collision += 1
        elif isinstance(info, Timeout):
            timeout += 1
        reward_total += discounted_reward
        # print(eval_env.global_time)
        # eval_env.render(mode='traj')
        # plt.show()
    denominator = float(max(1, int(n_episodes)))
    metrics = {
        "success_rate": success / denominator,
        "collision_rate": collision / denominator,
        "timeout_rate": timeout / denominator,
        "avg_nav_time": nav_time_total / success if success > 0 else 0.0,
        "avg_reward": reward_total / denominator,
    }

    message = (
        "[EVAL] step={} | succ={:.3f} | coll={:.3f} | timeout={:.3f} "
        "| nav_time={:.3f} | reward={:.4f}"
    ).format(
        global_step,
        metrics["success_rate"],
        metrics["collision_rate"],
        metrics["timeout_rate"],
        metrics["avg_nav_time"],
        metrics["avg_reward"],
    )
    if logger is not None:
        logger.info(message)
    else:
        print(message)

    if was_training:
        actor.train()
    return metrics


# 保留与原项目类似的函数名，方便其他文件导入
def eval_actor(
    env: gym.Env,
    actor: nn.Module,
    list1: List,
    list2: List,
    list3: List,
    list4: List,
    device: str,
    n_episodes: int,
    seed: int,
    logger: Optional[logging.Logger] = None,
    global_step: Optional[int] = None,
    human_num: int = 7,
    human_state_dim: int = 13,
    env_config_path: str = (
        r"crowd_nav\configs\env.config"
    ),
):
    del env
    metrics = evaluate_actor(
        actor=actor,
        device=device,
        n_episodes=n_episodes,
        seed=seed + 100,
        human_num=human_num,
        human_state_dim=human_state_dim,
        env_config_path=env_config_path,
        phase="train",
        test_case_start=20000,
        logger=logger,
        global_step=global_step,
    )
    list1.append(metrics["success_rate"])
    list2.append(metrics["collision_rate"])
    list3.append(metrics["avg_nav_time"])
    list4.append(metrics["avg_reward"])
    return metrics


def test_actor(
    env: gym.Env,
    actor: nn.Module,
    device: str,
    n_episodes: int,
    seed: int,
    logger: Optional[logging.Logger] = None,
    human_num: int = 7,
    human_state_dim: int = 13,
    env_config_path: str = (
        r"crowd_nav\configs\env.config"
    ),
):
    del env
    return evaluate_actor(
        actor=actor,
        device=device,
        n_episodes=n_episodes,
        seed=seed,
        human_num=human_num,
        human_state_dim=human_state_dim,
        env_config_path=env_config_path,
        phase="test",
        logger=logger,
    )


# ============================================================
# Offline training
# ============================================================
@pyrallis.wrap()
def train(config: TrainConfig) -> None:
    if not torch.cuda.is_available() and config.device.startswith("cuda"):
        raise RuntimeError(
            "配置要求使用{}，但当前PyTorch未检测到CUDA".format(config.device)
        )

    state_dim = get_state_dim(config.human_num, config.human_state_dim)
    action_dim = 2
    max_action = 1.0

    env_config = load_crowdnav_env_config(
        config.env_config_path,
        config.human_num,
    )
    env = gym.make(config.env)
    env.configure(env_config)
    set_seed(config.seed, env)

    if not os.path.exists(config.offline_dataset_path):
        raise FileNotFoundError(
            "未找到7人离线数据: {}".format(config.offline_dataset_path)
        )
    with h5py.File(config.offline_dataset_path, "r") as file:
        dataset = {key: np.asarray(value) for key, value in file.items()}
    validate_dataset(dataset, state_dim, config.offline_dataset_path)

    os.makedirs(config.checkpoints_path, exist_ok=True)
    logger = setup_logger(config.checkpoints_path, name="iql")
    logger.info(
        "Start training IQL | Env: %s | Humans: %d | State dim: %d "
        "| beta: %.4f | iql_tau: %.4f",
        config.env,
        config.human_num,
        state_dim,
        config.beta,
        config.iql_tau,
    )
    logger.info("Offline dataset: %s", config.offline_dataset_path)
    logger.info(
        "Dataset shape: states=%s | actions=%s | reward=[%.4f, %.4f]",
        dataset["observations"].shape,
        dataset["actions"].shape,
        float(dataset["rewards"].min()),
        float(dataset["rewards"].max()),
    )

    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
        device=config.replay_buffer_device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    if config.iql_deterministic:
        actor = DeterministicPolicy(
            state_dim,
            action_dim,
            max_action,
            dropout=config.actor_dropout,
        ).to(config.device)
    else:
        actor = GaussianPolicy(
            state_dim,
            action_dim,
            max_action,
            dropout=config.actor_dropout,
        ).to(config.device)

    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    trainer = ImplicitQLearning(
        max_action=max_action,
        actor=actor,
        actor_optimizer=actor_optimizer,
        q_network=q_network,
        q_optimizer=q_optimizer,
        v_network=v_network,
        v_optimizer=v_optimizer,
        iql_tau=config.iql_tau,
        beta=config.beta,
        max_steps=config.max_timesteps,
        discount=config.discount,
        tau=config.tau,
        device=config.device,
    )

    if config.load_model:
        if not os.path.exists(config.load_model):
            raise FileNotFoundError(
                "未找到待加载模型: {}".format(config.load_model)
            )
        checkpoint = torch.load(
            config.load_model,
            map_location=config.device,
        )
        trainer.load_state_dict(checkpoint)
        logger.info("Loaded model: %s", config.load_model)

    success_history = []
    collision_history = []
    nav_time_history = []
    reward_history = []

    logger.info("========== STAGE 1: OFFLINE PRE-TRAINING ==========")
    for timestep in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [item.to(config.device) for item in batch]
        log_dict = trainer.train(batch)

        if (timestep + 1) % int(config.eval_freq) == 0:
            logger.info(
                "Step %d | V_Loss=%.6f | Q_Loss=%.6f | "
                "Actor_Loss=%.6f | exp_adv_mean=%.4f | exp_adv_max=%.4f",
                timestep + 1,
                log_dict["value_loss"],
                log_dict["q_loss"],
                log_dict["actor_loss"],
                log_dict["exp_adv_mean"],
                log_dict["exp_adv_max"],
            )
            eval_actor(
                env,
                actor,
                success_history,
                collision_history,
                nav_time_history,
                reward_history,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
                logger=logger,
                global_step=timestep + 1,
                human_num=config.human_num,
                human_state_dim=config.human_state_dim,
                env_config_path=config.env_config_path,
            )

    logger.info("Offline training finished. Running final test...")
    test_actor(
        env,
        actor,
        device=config.device,
        n_episodes=config.final_test_episodes,
        seed=config.seed,
        logger=logger,
        human_num=config.human_num,
        human_state_dim=config.human_state_dim,
        env_config_path=config.env_config_path,
    )

    iql_path = os.path.join(config.checkpoints_path, "iql_offline.pt")
    actor_path = os.path.join(config.checkpoints_path, "actor_offline.pt")
    metrics_path = os.path.join(config.checkpoints_path, "iql_eval_history.npz")

    torch.save(trainer.state_dict(), iql_path)
    torch.save(actor.state_dict(), actor_path)
    np.savez(
        metrics_path,
        success=np.asarray(success_history),
        collision=np.asarray(collision_history),
        nav_time=np.asarray(nav_time_history),
        reward=np.asarray(reward_history),
    )

    logger.info("Saved IQL checkpoint: %s", iql_path)
    logger.info("Saved actor checkpoint: %s", actor_path)
    logger.info("Saved evaluation history: %s", metrics_path)


if __name__ == "__main__":
    train()
