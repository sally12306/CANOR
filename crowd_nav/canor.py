import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import h5py
import argparse
import configparser
import logging


# ------------- Logging Utilities -------------
import logging as _logging  # alias to avoid shadowing
from datetime import datetime as _dt
import os as _os

import matplotlib.pyplot as plt


def setup_logger(log_dir: str, name: str = "iql"):

    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear() 

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_path = os.path.join(log_dir, "train_eval.log")


    fh = logging.FileHandler(log_path, mode='w', encoding="utf-8")
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ------------- End Logging Utilities -------------
import copy

import torchvision
from crowd_sim.envs.utils.info import ReachGoal, Collision, Timeout
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.robot import Robot

import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
from crowd_sim.envs.utils.info import *
from crowd_nav.policy.self_attention import SelfAttention
from crowd_nav.policy.rnd_state_action_v4_plateau_escape import RND
TensorBatch = List[torch.Tensor]

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class TrainConfig:
    # ---------------- Experiment ----------------
    device: str = "cuda:0"
    env: str = "CrowdSim-v0"
    env_config_path: str = r"crowd_nav/configs/env.config"
    dataset_path: str = r""
    seed: int = 0
    eval_freq: int = 10000
    n_episodes: int = 50
    max_timesteps: int = 100000
    checkpoints_path: Optional[str] = r""
    load_model: str = ""
    online_run_name: str = ""
    allow_online_checkpoint_resume: bool = False

    # ---------------- IQL ----------------
    buffer_size: int = 1000000
    batch_size: int = 256
    offline_batch_size: int = 256
    discount: float = 0.99
    tau: float = 0.005                    
    beta: float = 3.0
    iql_tau: float = 0.80                  
    iql_deterministic: bool = False
    normalize: bool = True
    normalize_reward: bool = False
    vf_lr: float = 3e-4
    qf_lr: float = 3e-4
    actor_lr: float = 3e-4
    actor_dropout: Optional[float] = None

    online_vf_lr: float = 5e-5
    online_qf_lr: float = 5e-5
    online_actor_lr: float = 2e-5
    online_actor_lr_min: float = 8e-6
    actor_lr_decay_start_episode: int = 5000
    actor_lr_decay_end_episode: int = 30000
    use_online_actor_scheduler: bool = True

    # ---------------- Logging ----------------
    project: str = "CORL"
    group: str = "IQL-D4RL"
    name: str = "CANOR"

    # ---------------- Offline stage ----------------
    offline_steps: int = 100000

    # ---------------- Online stage ----------------
    online_episodes: int = 50000
    online_buffer_max_size: int = 100000
    min_online_size_for_updates: int = 512
    online_steps_per_update: int = 1
    online_updates_per_env_step: float = 0.0
    online_min_updates_per_episode: int = 1
    online_max_updates_per_episode: int = 1
    actor_update_interval: int = 1


    online_sampling_ratio_start: float = 0.45
    online_sampling_ratio_final: float = 0.65
    online_sampling_warmup_episodes: int = 20000
    online_sampling_ratio_min: float = 0.35
    replay_safety_collision_target: float = 0.15
    replay_safety_gain: float = 0.75
    bootstrap_p_on: float = 1.0
    bootstrap_p_off: float = 1.0

    collision_backprop_steps: int = 6
    collision_sampling_fraction_start: float = 0.35
    collision_sampling_fraction_final: float = 0.25
    collision_sampling_decay_episodes: int = 10000

    actor_success_only_online: bool = True


    actor_online_loss_fraction: float = 0.40
    actor_online_loss_fraction_start: float = 0.40
    actor_online_loss_fraction_final: float = 0.65
    actor_mix_warmup_episodes: int = 20000
    actor_online_mix_min: float = 0.30
    actor_mix_safety_gain: float = 1.00

    online_positive_advantage_only: bool = True
    online_advantage_margin: float = 0.0

    online_advantage_keep_fraction: float = 0.65
    online_advantage_threshold_floor: float = -0.25
    online_advantage_min_count: int = 8

    actor_failure_recovery_coef: float = 0.15
    actor_failure_recovery_radius: float = 0.08
    actor_failure_recovery_min_q_gain: float = 0.01
    actor_failure_recovery_min_novelty_weight: float = 0.35

    actor_anchor_coef: float = 0.03
    actor_max_relative_drift: float = 0.060

    exploration_std: float = 0.04
    exploration_std_start: float = 0.04
    exploration_std_mid: float = 0.02
    exploration_mid_episode: int = 10000
    exploration_std_final: float = 0.015
    exploration_decay_episodes: int = 50000
    exploration_std_collision_scale: float = 0.75
    action_low: float = -1.0
    action_high: float = 1.0


    rnd_hidden_dim: int = 256
    rnd_output_dim: int = 32
    rnd_lr: float = 3e-4
    online_rnd_lr: float = 2e-7
    novelty_kappa: float = 3.0
    novelty_delta: float = 0.5
    novelty_clip: float = 5.0
    offline_novelty_weight: float = 1.0
    online_novelty_weight_floor: float = 0.05
    actor_novelty_power: float = 1.0

    use_frozen_rnd_anchor: bool = True
    novelty_anchor_relaxation_cap: float = 0.05


    safe_expectile_interpolation: bool = True
    expectile_floor: float = 0.25
    tau_w_min: float = 0.0
    policy_weight_min: float = 0.0


    normalize_policy_advantage: bool = True
    advantage_scale_min: float = 0.05
    online_beta: float = 1.0
    exp_adv_max: float = 5.0
    online_target_tau: float = 0.005

    actor_grad_clip: float = 2.0
    q_grad_clip: float = 5.0
    v_grad_clip: float = 5.0
    normalize_actor_weights: bool = False
    balance_actor_groups: bool = True
    preserve_offline_optimizer_state: bool = False
    freeze_actor_log_std_online: bool = True
    policy_use_post_update_advantage: bool = False
    policy_use_target_advantage: bool = True
    policy_use_mean_mse: bool = True
    cap_negative_terminal_advantage: bool = True
    negative_terminal_actor_weight: float = 0.0
    q_huber_delta: float = 1.0


    freeze_rnd_online: bool = False
    update_rnd_online: bool = True
    online_batch_size: int = 128
    novelty_update_interval: int = 200
    novelty_warmup_episodes: int = 12000
    rnd_online_pair_capacity: int = 20000
    rnd_adapt_success_only: bool = True
    rnd_adapt_min_anchor_weight: float = 0.50
    rnd_use_normalized_error_variance: bool = True
    rnd_grad_clip: float = 0.5

    # Evaluation.
    online_eval_interval: int = 5000
    online_eval_episodes: int = 100
    online_eval_seed: int = 0
    online_log_interval: int = 1
    save_online_best: bool = True
    restore_best_online_at_end: bool = True
    best_checkpoint_name: str = ""
    best_actor_name: str = ""
    minimum_offline_success_rate: float = 0.55
    expected_offline_checkpoint_step: int = 39999


    rnd_w_min: float = 0.0
    min_safe_online: int = 0


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

def clamp_xy(num, min, max):
    if num < min:
        num = min
    elif num > max:
        num = max
    else:
        pass
    return num


class ReplayBuffer:
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            buffer_size: int,
            device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._device = device
        self.rnd_max = 0
        self.rnd_min = 0

        self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)

        self._actor_actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)


        self._weights = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset_old(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError("Replay buffer is smaller than the dataset you are trying to load!")

        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._actor_actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])


        self._weights[:n_transitions] = 0

        self._size += n_transitions
        self._pointer = min(self._size, n_transitions) % self._buffer_size

        print(f"Dataset size: {n_transitions} loaded into unified buffer.")


    def load_d4rl_dataset(self, data: dict, weights=None):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError("Replay buffer is smaller than the dataset you are trying to load!")

        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._actor_actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])

        # 如果传入了权重则加载，否则默认为 1.0
        if weights is not None:
            self._weights[:n_transitions] = self._to_tensor(weights[..., None] if weights.ndim == 1 else weights)
        else:
            self._weights[:n_transitions] = 1.0

        self._size += n_transitions

        self._pointer = min(self._size, n_transitions) % self._buffer_size

        print(f"Dataset size: {n_transitions} loaded into unified buffer.")

    def sample(self, batch_size: int) -> TensorBatch:

        if self._size <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        replace = self._size < batch_size
        indices = np.random.choice(
            self._size,
            size=batch_size,
            replace=replace,
        )

        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]

        # 获取对应的权重
        weights = self._weights[indices]

        actor_actions = self._actor_actions[indices]
        # Seven replay tensors; dual replay appends is_online as the eighth.
        return [states, actions, rewards, next_states, dones, weights, actor_actions]

    def sample_stratified(self, batch_size: int, failure_fraction: float = 0.0):

        if self._size <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        failure_fraction = float(np.clip(failure_fraction, 0.0, 1.0))
        all_idx = np.arange(self._size, dtype=np.int64)
        actor_mask = self._weights[:self._size].view(-1).detach().cpu().numpy()
        failure_idx = all_idx[actor_mask <= 0.5]
        regular_idx = all_idx[actor_mask > 0.5]

        target_failure = int(round(batch_size * failure_fraction))
        if failure_idx.size == 0:
            target_failure = 0
        if regular_idx.size == 0:
            target_failure = batch_size

        n_failure = min(max(target_failure, 0), batch_size)
        n_regular = batch_size - n_failure

        def choose(pool: np.ndarray, n: int):
            if n <= 0:
                return np.empty((0,), dtype=np.int64)
            if pool.size <= 0:
                return np.random.choice(all_idx, size=n, replace=self._size < n)
            return np.random.choice(pool, size=n, replace=pool.size < n)

        indices = np.concatenate([
            choose(failure_idx, n_failure),
            choose(regular_idx, n_regular),
        ])
        np.random.shuffle(indices)

        batch = [
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._dones[indices],
            self._weights[indices],
            self._actor_actions[indices],
        ]
        actual_failure_fraction = float(
            (batch[5].view(-1) <= 0.5).float().mean().item()
        )
        return batch, actual_failure_fraction

    def add_transition(
            self, state, action, reward, next_state, done, weight=1.0,
            actor_action=None,
    ):

        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self._device)
        if not torch.is_tensor(action):
            action = torch.tensor(action, dtype=torch.float32, device=self._device)
        if not torch.is_tensor(next_state):
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self._device)
        if actor_action is None:
            actor_action = action
        if not torch.is_tensor(actor_action):
            actor_action = torch.tensor(actor_action, dtype=torch.float32, device=self._device)


        self._states[self._pointer] = state
        self._actions[self._pointer] = action
        self._actor_actions[self._pointer] = actor_action
        self._rewards[self._pointer] = float(reward)
        self._next_states[self._pointer] = next_state
        self._dones[self._pointer] = float(done)
        self._weights[self._pointer] = float(weight)


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
        self.device = device
        self.offline_buffer = ReplayBuffer(
            state_dim, action_dim, offline_buffer_size, device
        )
        self.online_buffer = ReplayBuffer(
            state_dim, action_dim, online_buffer_size, device
        )

    def load_offline_dataset(self, data: Dict[str, np.ndarray], weights=None):
        self.offline_buffer.load_d4rl_dataset(data, weights)

    def add_online_transition(
            self, state, action, reward, next_state, done, weight=1.0,
            actor_action=None,
    ):
        self.online_buffer.add_transition(
            state, action, reward, next_state, done, weight,
            actor_action=actor_action,
        )

    @staticmethod
    def _select_batch(batch: List[torch.Tensor], mask: torch.Tensor) -> List[torch.Tensor]:
        return [tensor[mask] for tensor in batch]

    def sample(self, batch_size: int, online_ratio: float = 0.5) -> List[torch.Tensor]:

        batch, _ = self.sample_bootstrap_selected(
            batch_size=batch_size,
            online_ratio=online_ratio,
            p_on=1.0,
            p_off=1.0,
            return_info=True,
        )
        return batch

    def sample_bootstrap_selected(
            self,
            batch_size: int,
            online_ratio: float = 0.5,
            p_on: float = 0.95,
            p_off: float = 0.70,
            candidate_multiplier: int = 1,
            online_failure_fraction: float = 0.0,
            return_info: bool = False,
    ):

        del candidate_multiplier  # retained for backwards-compatible calls
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.offline_buffer._size <= 0:
            raise RuntimeError("The offline replay partition is empty")
        if self.online_buffer._size <= 0:
            raise RuntimeError("The online replay partition is empty")

        ratio = float(np.clip(online_ratio, 0.0, 1.0))
        n_online_candidates = int(round(batch_size * ratio))
        n_online_candidates = min(max(n_online_candidates, 1), batch_size - 1)
        n_offline_candidates = batch_size - n_online_candidates

        online_batch, actual_failure_fraction = self.online_buffer.sample_stratified(
            n_online_candidates, failure_fraction=online_failure_fraction
        )
        offline_batch = self.offline_buffer.sample(n_offline_candidates)

        online_mask = torch.rand(n_online_candidates, device=self.device) < float(p_on)
        offline_mask = torch.rand(n_offline_candidates, device=self.device) < float(p_off)

        selected_online = self._select_batch(online_batch, online_mask)
        selected_offline = self._select_batch(offline_batch, offline_mask)
        selected = [
            torch.cat([selected_online[i], selected_offline[i]], dim=0)
            for i in range(len(selected_online))
        ]

        retained_online = int(online_mask.sum().item())
        retained_offline = int(offline_mask.sum().item())
        retained_total = retained_online + retained_offline

        if retained_total == 0:
            selected = self.offline_buffer.sample(1)
            retained_online = 0
            retained_offline = 1
            retained_total = 1
            is_online = torch.zeros((1, 1), device=self.device, dtype=torch.float32)
        else:
            is_online = torch.cat([
                torch.ones((retained_online, 1), device=self.device, dtype=torch.float32),
                torch.zeros((retained_offline, 1), device=self.device, dtype=torch.float32),
            ], dim=0)

        permutation = torch.randperm(retained_total, device=self.device)
        selected = [tensor[permutation] for tensor in selected]
        is_online = is_online[permutation]
        selected.append(is_online)

        info = {
            "candidate_online": n_online_candidates,
            "candidate_offline": n_offline_candidates,
            "retained_online": retained_online,
            "retained_offline": retained_offline,
            "retained_total": retained_total,
            "online_fraction": retained_online / float(retained_total),
            "failure_candidate_fraction": actual_failure_fraction,
        }
        return (selected, info) if return_info else selected


def set_seed(
        seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        # env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()



cnt = 20000


@torch.no_grad()
def eval_actor(
        env: gym.Env, actor: nn.Module, list1: List, list2: List, list3: List, list4: List, device: str,
        n_episodes: int, seed: int
        , logger: Optional[logging.Logger] = None, global_step: Optional[int] = None):
    global cnt
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str,
                        default=r'crowd_nav/configs/env.config')
    args, _ = parser.parse_known_args()
    actor.eval()
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)
    robot.set_policy(actor)
    # robot.print_info()
    phase = 'train'
    success = 0
    collision = 0
    run_time = 0
    avg_nav_time = 0
    avg_reward = 0
    for _ in range(n_episodes):
        observation = env.reset(phase=phase, test_case=cnt)
        done = False
        rewards = 0
        gamma = 1  
        while not done:
            joint_state = JointState(robot.get_full_state(), observation)
            state = to_np(transform(joint_state, device).view(1, -1).squeeze(0))
            # print(state.shape)  
            action = actor.act(state)
            # print(action.shape) 
            action = ActionXY(action[0], action[1])
            observation, reward, done, info = env.step(action)
            # print(reward)
            rewards += reward * gamma
            gamma = gamma * 0.99
        if isinstance(info, ReachGoal):
            success += 1
            avg_nav_time += env.global_time
        elif isinstance(info, Collision):
            collision += 1
        elif isinstance(info, Timeout):
            run_time += 1

        avg_reward += rewards
        cnt += 1
    if success != 0:
        avg_nav_time = avg_nav_time / success
    else:
        avg_nav_time = 0
    avg_reward = avg_reward / n_episodes

    metrics = {
        "success": success,
        "success_rate": success / n_episodes if n_episodes else 0.0,
        "collision": collision,
        "timeout": run_time,
        "avg_nav_time": avg_nav_time,
        "avg_reward": avg_reward,
        "episodes": n_episodes,
        "step": global_step,
    }
    
    try:
        list1.append(metrics["success_rate"])
        list2.append(collision / n_episodes if n_episodes else 0.0)
        list3.append(avg_nav_time)
        list4.append(avg_reward)
    except Exception:
        pass
    if logger is not None:
        try:
            logger.info(
                "[EVAL] step=%s | succ=%d(%.3f) | coll=%d | timeout=%d | nav_time=%.6f | reward=%.6f | episodes=%d",
                str(global_step), success, metrics["success_rate"], collision, run_time, avg_nav_time, avg_reward,
                n_episodes
            )
        except Exception:
            pass
    actor.train()

    # print(
    #     f"成功次数为{success} 成功率为{success / n_episodes} 碰撞次数{collision} 超时次数{run_time} 平均导航时间为{avg_nav_time} 平均奖励为{avg_reward} ")
    list1.append(success / n_episodes)
    list2.append(collision / n_episodes)
    list3.append(avg_nav_time)
    list4.append(avg_reward)
    actor.train()
    return metrics



@torch.no_grad()
def test_actor(
        env: gym.Env,
        actor: nn.Module,
        device: str,
        n_episodes: int,
        seed: int,
        logger: Optional[logging.Logger] = None,
        test_case_start: int = 0,
):

    del env  # evaluation uses an isolated environment instance

    parser = argparse.ArgumentParser("Parse configuration file", add_help=False)
    parser.add_argument(
        "--env_config",
        type=str,
        default=r"crowd_nav/configs/env.config",
    )
    args, _ = parser.parse_known_args()

    was_training = actor.training
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )

    eval_env = None
    try:
        # Seed only inside this sandbox; all global states are restored below.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        actor.eval()
        env_config = configparser.RawConfigParser()
        env_config.read(args.env_config)
        eval_env = gym.make("CrowdSim-v0")
        eval_env.configure(env_config)
        robot = Robot(env_config, "robot")
        eval_env.set_robot(robot)
        robot.set_policy(actor)
        if hasattr(eval_env, "seed"):
            eval_env.seed(seed)

        phase = "test"
        success = 0
        collision = 0
        timeout = 0
        total_nav_time = 0.0
        total_reward = 0.0

        for episode_idx in range(int(n_episodes)):
            case_id = int(test_case_start) + episode_idx
            try:
                observation = eval_env.reset(
                    phase=phase,
                    test_case=case_id,
                )
            except TypeError:
                observation = eval_env.reset(phase=phase)

            done = False
            discounted_reward = 0.0
            gamma = 1.0
            info = None

            while not done:
                joint_state = JointState(
                    robot.get_full_state(),
                    observation,
                )
                state = to_np(
                    transform(joint_state, device)
                    .view(1, -1)
                    .squeeze(0)
                )
                action = actor.act(state, device=device)
                observation, reward, done, info = eval_env.step(
                    ActionXY(float(action[0]), float(action[1]))
                )
                discounted_reward += float(reward) * gamma
                gamma *= 0.99

            if isinstance(info, ReachGoal):
                success += 1
                total_nav_time += float(eval_env.global_time)
            elif isinstance(info, Collision):
                collision += 1
            elif isinstance(info, Timeout):
                timeout += 1
            total_reward += discounted_reward

        avg_nav_time = total_nav_time / success if success > 0 else 0.0
        avg_reward = (
            total_reward / int(n_episodes)
            if int(n_episodes) > 0
            else 0.0
        )
        metrics = {
            "success": success,
            "success_rate": (
                success / int(n_episodes)
                if int(n_episodes) > 0
                else 0.0
            ),
            "collision": collision,
            "collision_rate": (
                collision / int(n_episodes)
                if int(n_episodes) > 0
                else 0.0
            ),
            "timeout": timeout,
            "timeout_rate": (
                timeout / int(n_episodes)
                if int(n_episodes) > 0
                else 0.0
            ),
            "avg_nav_time": avg_nav_time,
            "avg_reward": avg_reward,
            "episodes": int(n_episodes),
        }

        if logger is not None:
            logger.info(
                "[TEST] succ=%d(%.3f) | coll=%d(%.3f) | "
                "timeout=%d(%.3f) | nav_time=%.6f | reward=%.6f "
                "| episodes=%d | test_case_start=%d",
                success,
                metrics["success_rate"],
                collision,
                metrics["collision_rate"],
                timeout,
                metrics["timeout_rate"],
                avg_nav_time,
                avg_reward,
                int(n_episodes),
                int(test_case_start),
            )
        return metrics
    finally:
        if eval_env is not None and hasattr(eval_env, "close"):
            try:
                eval_env.close()
            except Exception:
                pass

        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(torch_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        actor.train(was_training)


def transform(state, device):
    state_tensor = torch.cat([torch.Tensor([state.self_state + human_state]).to(device)
                              for human_state in state.human_states], dim=0)

    state_tensor = rotate(state_tensor)
    return state_tensor


def rotate(state):
    # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
    #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
    batch = state.shape[0]
    dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
    dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
    rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

    dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
    v_pref = state[:, 7].reshape((batch, -1))
    vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
    vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

    radius = state[:, 4].reshape((batch, -1))
    theta = torch.zeros_like(v_pref)
    vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
    vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
    px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
    px1 = px1.reshape((batch, -1))
    py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
    py1 = py1.reshape((batch, -1))
    radius1 = state[:, 13].reshape((batch, -1))
    radius_sum = radius + radius1
    da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                              reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
    # dg = ||p − pg||2 is the robot’s distance to the goal and di = ||p − pi||2 is the robot’s distance to the neighbor i
    new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
    return new_state


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)



def weighted_asymmetric_l2_loss(u: torch.Tensor, tau: float, weights: torch.Tensor) -> torch.Tensor:
    # weights shape: (batch, 1)
    element_wise_loss = torch.abs(tau - (u < 0).float()) * u ** 2
    return torch.mean(weights * element_wise_loss)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
            self,
            dims,
            activation_fn: Callable[[], nn.Module] = nn.ReLU,
            output_activation_fn: Callable[[], nn.Module] = None,
            squeeze_output: bool = False,
            dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
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
        self.net = MLP(
            [56, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action
        self.kinematics = 'holonomic'
        self.multiagent_training = True
        self.attention = SelfAttention()

    def forward(self, obs: torch.Tensor) -> Normal:
        x = self.attention(obs)
        mean = self.net(x)
        # print(mean.shape)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cuda"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


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
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action
        self.kinematics = 'holonomic'
        self.multiagent_training = True

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )


class TwinQ(nn.Module):
    def __init__(
            self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [58, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)
        self.attention = SelfAttention()

    def both(
            self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.attention(state)
        x = torch.cat((x, action), dim=1)
        # sa = torch.cat([state, action], 1)
        return self.q1(x), self.q2(x)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [56, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)
        self.attention = SelfAttention()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.attention(state)
        return self.v(x)


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
            beta: float = 1.0,
            max_steps: Optional[int] = 1000000,
            discount: float = 0.99,
            tau: float = 0.005,
            device: str = "cpu",
            tau_w_min: float = 0.0,
            policy_weight_min: float = 0.0,
            exp_adv_max: float = 20.0,
            actor_grad_clip: float = 2.0,
            q_grad_clip: float = 5.0,
            v_grad_clip: float = 5.0,
            normalize_actor_weights: bool = True,
            policy_use_post_update_advantage: bool = False,
            policy_use_target_advantage: bool = True,
            policy_use_mean_mse: bool = True,
            cap_negative_terminal_advantage: bool = True,
            q_huber_delta: float = 1.0,
            safe_expectile_interpolation: bool = True,
            expectile_floor: float = 0.5,
            normalize_policy_advantage: bool = False,
            advantage_scale_min: float = 0.05,
            actor_novelty_power: float = 0.5,
            actor_online_loss_fraction: float = 0.75,
            balance_actor_groups: bool = False,
            negative_terminal_actor_weight: float = 1.0,
            online_positive_advantage_only: bool = False,
            online_advantage_margin: float = 0.0,
            online_advantage_keep_fraction: float = 1.0,
            online_advantage_threshold_floor: float = -float("inf"),
            online_advantage_min_count: int = 8,
            actor_failure_recovery_coef: float = 0.0,
            actor_failure_recovery_radius: float = 0.05,
            actor_failure_recovery_min_q_gain: float = 0.0,
            actor_failure_recovery_min_novelty_weight: float = 0.0,
            actor_anchor_coef: float = 0.0,
            actor_max_relative_drift: float = 0.0,
    ):
        self.max_action = float(max_action)
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = (
            CosineAnnealingLR(self.actor_optimizer, int(max_steps))
            if max_steps is not None and int(max_steps) > 0
            else None
        )
        self.iql_tau = float(iql_tau)
        self.beta = float(beta)
        self.discount = float(discount)
        self.tau = float(tau)
        self.total_it = 0
        self.device = device
        self.tau_w_min = float(tau_w_min)
        self.policy_weight_min = float(policy_weight_min)
        self.exp_adv_max = float(exp_adv_max)
        self.actor_grad_clip = float(actor_grad_clip)
        self.q_grad_clip = float(q_grad_clip)
        self.v_grad_clip = float(v_grad_clip)
        self.normalize_actor_weights = bool(normalize_actor_weights)
        self.policy_use_post_update_advantage = bool(policy_use_post_update_advantage)
        self.policy_use_target_advantage = bool(policy_use_target_advantage)
        self.policy_use_mean_mse = bool(policy_use_mean_mse)
        self.cap_negative_terminal_advantage = bool(cap_negative_terminal_advantage)
        self.q_huber_delta = float(q_huber_delta)
        self.safe_expectile_interpolation = bool(safe_expectile_interpolation)
        self.expectile_floor = float(expectile_floor)
        self.normalize_policy_advantage = bool(normalize_policy_advantage)
        self.advantage_scale_min = float(advantage_scale_min)
        self.actor_novelty_power = float(actor_novelty_power)
        self.actor_online_loss_fraction = float(actor_online_loss_fraction)
        self.balance_actor_groups = bool(balance_actor_groups)
        self.negative_terminal_actor_weight = float(negative_terminal_actor_weight)
        self.online_positive_advantage_only = bool(online_positive_advantage_only)
        self.online_advantage_margin = float(online_advantage_margin)
        self.online_advantage_keep_fraction = float(online_advantage_keep_fraction)
        self.online_advantage_threshold_floor = float(online_advantage_threshold_floor)
        self.online_advantage_min_count = int(online_advantage_min_count)
        self.actor_failure_recovery_coef = float(actor_failure_recovery_coef)
        self.actor_failure_recovery_radius = float(actor_failure_recovery_radius)
        self.actor_failure_recovery_min_q_gain = float(actor_failure_recovery_min_q_gain)
        self.actor_failure_recovery_min_novelty_weight = float(
            actor_failure_recovery_min_novelty_weight
        )
        self.actor_anchor_coef = float(actor_anchor_coef)
        self.actor_max_relative_drift = float(actor_max_relative_drift)
        self.actor_reference: Optional[nn.Module] = None
        self._actor_reference_parameters: Optional[List[torch.Tensor]] = None

    @staticmethod
    def _policy_mean(policy_output):
        if isinstance(policy_output, torch.distributions.Distribution):
            return policy_output.mean
        if torch.is_tensor(policy_output):
            return policy_output
        raise NotImplementedError("Unsupported actor output type")

    def set_actor_reference(self) -> None:

        self.actor_reference = copy.deepcopy(self.actor).to(self.device)
        self.actor_reference.eval()
        self.actor_reference.requires_grad_(False)
        self._actor_reference_parameters = [
            parameter.detach().clone()
            for parameter in self.actor.parameters()
        ]

    def _project_actor_to_reference(self) -> Dict[str, float]:

        limit = float(self.actor_max_relative_drift)
        reference = self._actor_reference_parameters
        if limit <= 0.0 or reference is None:
            return {"actor_projected": 0.0, "actor_drift_before_projection": 0.0}

        with torch.no_grad():
            ref_sq = torch.zeros((), device=self.device)
            delta_sq = torch.zeros((), device=self.device)
            for parameter, reference_parameter in zip(self.actor.parameters(), reference):
                ref = reference_parameter.to(parameter.device, dtype=parameter.dtype)
                ref_sq = ref_sq + ref.pow(2).sum()
                delta_sq = delta_sq + (parameter - ref).pow(2).sum()
            drift = torch.sqrt(delta_sq / ref_sq.clamp(min=1e-12))
            drift_value = float(drift.item())
            projected = drift_value > limit
            if projected:
                scale = float(limit / max(drift_value, 1e-12))
                for parameter, reference_parameter in zip(self.actor.parameters(), reference):
                    ref = reference_parameter.to(parameter.device, dtype=parameter.dtype)
                    parameter.copy_(ref + scale * (parameter - ref))
            return {
                "actor_projected": float(projected),
                "actor_drift_before_projection": drift_value,
            }

    def reset_online_optimizers(
            self,
            q_lr: float,
            v_lr: float,
            actor_lr: float,
            actor_scheduler_steps: Optional[int] = None,
    ):
        self.q_optimizer = torch.optim.Adam(
            [p for p in self.qf.parameters() if p.requires_grad], lr=float(q_lr)
        )
        self.v_optimizer = torch.optim.Adam(
            [p for p in self.vf.parameters() if p.requires_grad], lr=float(v_lr)
        )
        self.actor_optimizer = torch.optim.Adam(
            [p for p in self.actor.parameters() if p.requires_grad], lr=float(actor_lr)
        )
        self.actor_lr_schedule = (
            CosineAnnealingLR(self.actor_optimizer, int(actor_scheduler_steps))
            if actor_scheduler_steps is not None and int(actor_scheduler_steps) > 0
            else None
        )

    def prepare_online_optimizers(self, q_lr: float, v_lr: float, actor_lr: float):
        for group in self.q_optimizer.param_groups:
            group["lr"] = float(q_lr)
        for group in self.v_optimizer.param_groups:
            group["lr"] = float(v_lr)
        for group in self.actor_optimizer.param_groups:
            group["lr"] = float(actor_lr)
        self.actor_lr_schedule = None

    def set_actor_lr(self, actor_lr: float):
        actor_lr = float(actor_lr)
        for group in self.actor_optimizer.param_groups:
            group["lr"] = actor_lr


    @staticmethod
    def parameter_fingerprint(module: nn.Module) -> Dict[str, float]:
        with torch.no_grad():
            flat = torch.cat([p.detach().float().view(-1).cpu() for p in module.parameters()])
            return {
                "numel": float(flat.numel()),
                "sum": float(flat.sum().item()),
                "abs_sum": float(flat.abs().sum().item()),
                "sq_sum": float(flat.square().sum().item()),
            }

    def _update_v(self, observations, actions, novelty_weights, log_dict):
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
        value = self.vf(observations)
        advantage = target_q - value

        w = novelty_weights.view(-1).detach().clamp(0.0, 1.0)
        if self.safe_expectile_interpolation:
            floor = float(np.clip(self.expectile_floor, 1e-4, self.iql_tau))
            tau_w = floor + (self.iql_tau - floor) * w
        else:
            tau_w = self.iql_tau * w
        tau_w = tau_w.clamp(
            min=max(self.tau_w_min, 1e-6),
            max=1.0 - 1e-6,
        )
        value_loss = (
            torch.abs(tau_w - (advantage < 0).float()) * advantage.pow(2)
        ).mean()

        self.v_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        v_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.vf.parameters(), self.v_grad_clip
        )
        self.v_optimizer.step()

        log_dict.update({
            "value_loss": float(value_loss.item()),
            "v_grad_norm": float(v_grad_norm.item() if torch.is_tensor(v_grad_norm) else v_grad_norm),
            "tau_w_mean": float(tau_w.mean().item()),
            "tau_w_min": float(tau_w.min().item()),
            "tau_w_max": float(tau_w.max().item()),
            "novelty_w_mean": float(w.mean().item()),
        })
        return advantage

    def _update_q(self, next_v, observations, actions, rewards, terminals, log_dict):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        q1, q2 = self.qf.both(observations, actions)
        delta = max(float(self.q_huber_delta), 1e-6)

        def huber(error: torch.Tensor) -> torch.Tensor:
            abs_error = error.abs()
            return torch.where(
                abs_error <= delta,
                0.5 * error.pow(2),
                delta * (abs_error - 0.5 * delta),
            )

        q_loss = 0.5 * (huber(q1 - targets).mean() + huber(q2 - targets).mean())
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        q_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.qf.parameters(), self.q_grad_clip
        )
        self.q_optimizer.step()
        soft_update(self.q_target, self.qf, self.tau)

        log_dict.update({
            "q_loss": float(q_loss.item()),
            "q_grad_norm": float(q_grad_norm.item() if torch.is_tensor(q_grad_norm) else q_grad_norm),
            "q_target_mean": float(targets.mean().item()),
            "q_pred_mean": float(torch.min(q1, q2).mean().item()),
        })

    @staticmethod
    def _robust_advantage_normalize(advantage: torch.Tensor, scale_min: float):


        center = torch.zeros((), device=advantage.device, dtype=advantage.dtype)
        rms = torch.sqrt(torch.mean(advantage.pow(2)) + 1e-8)
        scale = torch.clamp(rms, min=float(scale_min))
        return advantage / scale, center, scale

    @staticmethod
    def _weighted_group_mean(loss, weight, mask, normalize_weights: bool):
        if not bool(mask.any()):
            return None
        group_loss = loss[mask]
        group_weight = weight[mask]
        if normalize_weights:
            positive = group_weight > 0
            if not bool(positive.any()):
                return group_loss.sum() * 0.0
            group_weight = group_weight / group_weight[positive].mean().clamp(min=1e-6)
        return (group_weight * group_loss).mean()

    def _failure_recovery_loss(
            self,
            observations: torch.Tensor,
            policy_mean: torch.Tensor,
            novelty_w: torch.Tensor,
            failure_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float]:

        zero = policy_mean.sum() * 0.0
        if self.actor_failure_recovery_coef <= 0.0 or not bool(failure_mask.any()):
            return zero, 0.0, 0.0

        idx = torch.nonzero(failure_mask, as_tuple=False).view(-1)
        states = observations[idx]
        mean = policy_mean[idx]
        confidence = novelty_w[idx]
        trusted = confidence >= float(self.actor_failure_recovery_min_novelty_weight)
        if not bool(trusted.any()):
            return zero, 0.0, 0.0

        states = states[trusted]
        mean = mean[trusted]
        radius = max(float(self.actor_failure_recovery_radius), 1e-6)
        action_dim = int(mean.shape[-1])

        directions = []
        eye = torch.eye(action_dim, device=mean.device, dtype=mean.dtype)
        for axis in eye:
            directions.extend([axis, -axis])
        if action_dim == 2:
            diag = torch.tensor(
                [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]],
                device=mean.device,
                dtype=mean.dtype,
            ) / np.sqrt(2.0)
            directions.extend([item for item in diag])
        direction_tensor = torch.stack(directions, dim=0)

        candidates = [mean.detach()]
        local = mean.detach().unsqueeze(1) + radius * direction_tensor.unsqueeze(0)
        candidates.extend([local[:, j, :] for j in range(local.shape[1])])
        if self.actor_reference is not None:
            with torch.no_grad():
                reference_mean = self._policy_mean(self.actor_reference(states))
                reference_delta = (reference_mean - mean.detach()).clamp(-radius, radius)
                candidates.append(mean.detach() + reference_delta)

        candidate_actions = torch.stack(candidates, dim=1).clamp(
            -self.max_action, self.max_action
        )
        n_states, n_candidates, _ = candidate_actions.shape
        repeated_states = states.unsqueeze(1).expand(
            n_states, n_candidates, states.shape[-1]
        ).reshape(n_states * n_candidates, states.shape[-1])
        flat_actions = candidate_actions.reshape(n_states * n_candidates, action_dim)
        with torch.no_grad():
            candidate_q = self.q_target(repeated_states, flat_actions).view(
                n_states, n_candidates
            )
            baseline_q = candidate_q[:, 0]
            best_q, best_index = candidate_q.max(dim=1)
            q_gain = best_q - baseline_q
            accepted = q_gain > float(self.actor_failure_recovery_min_q_gain)
            target = candidate_actions[
                torch.arange(n_states, device=mean.device), best_index
            ]

        if not bool(accepted.any()):
            return zero, 0.0, float(q_gain.mean().item())
        per_sample = (mean - target.detach()).pow(2).sum(dim=-1)
        loss = per_sample[accepted].mean()
        return (
            loss,
            float(accepted.float().mean().item()),
            float(q_gain[accepted].mean().item()),
        )

    def _update_policy_without_group(
            self,
            advantage,
            observations,
            actions,
            actor_target_actions,
            novelty_weights,
            rewards,
            terminals,
            replay_actor_weights,
            is_online,
            log_dict,
    ):
        raw_advantage = advantage.detach().view(-1)
        policy_advantage = raw_advantage.clone()

        negative_terminal_mask = (
            (terminals.view(-1) > 0.5) & (rewards.view(-1) < 0.0)
        )
        if self.cap_negative_terminal_advantage and bool(negative_terminal_mask.any()):
            policy_advantage[negative_terminal_mask] = torch.minimum(
                policy_advantage[negative_terminal_mask],
                torch.zeros_like(policy_advantage[negative_terminal_mask]),
            )

        if self.normalize_policy_advantage:
            actor_advantage, adv_center, adv_scale = self._robust_advantage_normalize(
                policy_advantage, self.advantage_scale_min
            )
        else:
            actor_advantage = policy_advantage
            adv_center = torch.zeros((), device=policy_advantage.device)
            adv_scale = torch.ones((), device=policy_advantage.device)

        max_log_weight = float(np.log(max(self.exp_adv_max, 1.0)))
        log_adv_weight = (self.beta * actor_advantage).clamp(
            min=-12.0, max=max_log_weight
        )
        exp_adv = torch.exp(log_adv_weight)

        policy_out = self.actor(observations)
        policy_mean = self._policy_mean(policy_out)
        if isinstance(policy_out, torch.distributions.Distribution):
            if self.policy_use_mean_mse:
                bc_loss = (policy_mean - actor_target_actions).pow(2).sum(dim=-1)
            else:
                bc_loss = -policy_out.log_prob(actor_target_actions).sum(-1)
        elif torch.is_tensor(policy_out):
            bc_loss = (policy_mean - actor_target_actions).pow(2).sum(dim=1)
        else:
            raise NotImplementedError("Unsupported actor output type")

        novelty_w = novelty_weights.view(-1).detach().clamp(
            min=self.policy_weight_min, max=1.0
        )
        actor_novelty_w = novelty_w.pow(max(self.actor_novelty_power, 1e-6))
        replay_actor_w = replay_actor_weights.view(-1).detach().clamp(0.0, 1.0)
        if bool(negative_terminal_mask.any()):
            replay_actor_w = replay_actor_w.clone()
            replay_actor_w[negative_terminal_mask] = self.negative_terminal_actor_weight

        online_mask = is_online.view(-1) > 0.5
        offline_mask = ~online_mask
        valid_online_actor_mask = online_mask & (replay_actor_w > 0.5)
        advantage_threshold = float(self.online_advantage_margin)
        valid_count = int(valid_online_actor_mask.sum().item())
        keep_fraction = float(np.clip(self.online_advantage_keep_fraction, 0.0, 1.0))
        if (
            self.online_positive_advantage_only
            and keep_fraction > 0.0
            and valid_count >= max(self.online_advantage_min_count, 1)
        ):
            quantile = torch.quantile(
                policy_advantage[valid_online_actor_mask],
                q=max(0.0, min(1.0, 1.0 - keep_fraction)),
            )

            advantage_threshold = max(
                float(quantile.item()),
                float(self.online_advantage_threshold_floor),
            )
        online_positive_mask = valid_online_actor_mask & (
            policy_advantage >= advantage_threshold
        )

        awr_weight = (actor_novelty_w * exp_adv * replay_actor_w).detach()
        if self.online_positive_advantage_only and bool(online_mask.any()):
            awr_weight = awr_weight.clone()
            awr_weight[online_mask & ~online_positive_mask] = 0.0

        if self.balance_actor_groups:
            online_loss = self._weighted_group_mean(
                bc_loss, awr_weight, online_mask, self.normalize_actor_weights
            )
            offline_loss = self._weighted_group_mean(
                bc_loss, awr_weight, offline_mask, self.normalize_actor_weights
            )
            if online_loss is not None and offline_loss is not None:
                mix = float(np.clip(self.actor_online_loss_fraction, 0.0, 1.0))
                base_actor_loss = mix * online_loss + (1.0 - mix) * offline_loss
            elif online_loss is not None:
                base_actor_loss = online_loss
            elif offline_loss is not None:
                base_actor_loss = offline_loss
            else:
                base_actor_loss = bc_loss.sum() * 0.0
        else:
            global_weight = awr_weight
            if self.normalize_actor_weights:
                positive = global_weight > 0
                if bool(positive.any()):
                    global_weight = global_weight / global_weight[positive].mean().clamp(min=1e-6)
            base_actor_loss = (global_weight * bc_loss).mean()

        anchor_loss = base_actor_loss.detach() * 0.0
        if (
            self.actor_anchor_coef > 0.0
            and self.actor_reference is not None
            and bool(offline_mask.any())
        ):
            with torch.no_grad():
                reference_mean = self._policy_mean(
                    self.actor_reference(observations)
                )
            anchor_loss = (
                policy_mean[offline_mask] - reference_mean[offline_mask]
            ).pow(2).sum(dim=-1).mean()

        failure_mask = online_mask & (replay_actor_w <= 0.5)
        recovery_loss, recovery_accept_fraction, recovery_q_gain = (
            self._failure_recovery_loss(
                observations,
                policy_mean,
                novelty_w,
                failure_mask,
            )
        )
        actor_loss = (
            base_actor_loss
            + float(self.actor_anchor_coef) * anchor_loss
            + float(self.actor_failure_recovery_coef) * recovery_loss
        )
        if not torch.isfinite(actor_loss):
            raise FloatingPointError(f"Non-finite actor loss: {actor_loss.item()}")

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.actor_grad_clip
        )
        self.actor_optimizer.step()
        projection_stats = self._project_actor_to_reference()
        if self.actor_lr_schedule is not None:
            self.actor_lr_schedule.step()

        collision_mask = replay_actor_w <= 0.5
        noncollision_mask = ~collision_mask
        with torch.no_grad():
            online_count = max(float(online_mask.float().sum().item()), 1.0)
            online_positive_fraction = float(
                online_positive_mask.float().sum().item() / online_count
            )
            log_dict.update({
                "actor_loss": float(actor_loss.item()),
                "actor_base_loss": float(base_actor_loss.item()),
                "actor_anchor_loss": float(anchor_loss.item()),
                "actor_recovery_loss": float(recovery_loss.item()),
                "actor_recovery_accept_fraction": recovery_accept_fraction,
                "actor_recovery_q_gain": recovery_q_gain,
                "online_advantage_threshold": float(advantage_threshold),
                "bc_loss_mean": float(bc_loss.mean().item()),
                "policy_w_mean": float(novelty_w.mean().item()),
                "actor_novelty_w_mean": float(actor_novelty_w.mean().item()),
                "exp_adv_mean": float(exp_adv.mean().item()),
                "exp_adv_max": float(exp_adv.max().item()),
                "awr_weight_mean": float(awr_weight.mean().item()),
                "awr_weight_max": float(awr_weight.max().item()),
                "awr_zero_fraction": float((awr_weight <= 1e-8).float().mean().item()),
                "online_positive_advantage_fraction": online_positive_fraction,
                "advantage_mean": float(raw_advantage.mean().item()),
                "advantage_std": float(raw_advantage.std(unbiased=False).item()),
                "actor_advantage_mean": float(actor_advantage.mean().item()),
                "actor_advantage_std": float(actor_advantage.std(unbiased=False).item()),
                "advantage_center": float(adv_center.item()),
                "advantage_scale": float(adv_scale.item()),
                "advantage_collision_mean": (
                    float(raw_advantage[collision_mask].mean().item())
                    if bool(collision_mask.any()) else float("nan")
                ),
                "capped_advantage_collision_mean": (
                    float(policy_advantage[collision_mask].mean().item())
                    if bool(collision_mask.any()) else float("nan")
                ),
                "advantage_noncollision_mean": (
                    float(raw_advantage[noncollision_mask].mean().item())
                    if bool(noncollision_mask.any()) else float("nan")
                ),
                "exp_adv_collision_mean": (
                    float(exp_adv[collision_mask].mean().item())
                    if bool(collision_mask.any()) else float("nan")
                ),
                "negative_terminal_fraction": float(negative_terminal_mask.float().mean().item()),
                "online_actor_sample_fraction": float(online_mask.float().mean().item()),
                "actor_grad_norm": float(actor_grad_norm.item() if torch.is_tensor(actor_grad_norm) else actor_grad_norm),
                "actor_lr": float(self.actor_optimizer.param_groups[0]["lr"]),
                "q_lr": float(self.q_optimizer.param_groups[0]["lr"]),
                "v_lr": float(self.v_optimizer.param_groups[0]["lr"]),
                **projection_stats,
            })

    def train(
            self,
            batch: TensorBatch,
            weights: Optional[torch.Tensor] = None,
            update_actor: bool = True,
    ) -> Dict[str, float]:
        self.total_it += 1
        if len(batch) < 5:
            raise ValueError(f"Expected at least five tensors, got {len(batch)}")
        observations, actions, rewards, next_observations, dones = batch[:5]
        # v4 layout: [s, executed_a, r, s2, d, actor_mask,
        #             denoised_actor_target_a, is_online].
        # Legacy v3 batches remain supported.
        if len(batch) >= 8:
            replay_actor_weights = batch[5].to(
                observations.device, dtype=torch.float32
            ).view(-1, 1)
            actor_target_actions = batch[6].to(
                observations.device, dtype=torch.float32
            )
            is_online = batch[7].to(
                observations.device, dtype=torch.float32
            ).view(-1, 1)
        elif len(batch) >= 7:
            replay_actor_weights = batch[5].to(
                observations.device, dtype=torch.float32
            ).view(-1, 1)
            actor_target_actions = actions
            is_online = batch[6].to(
                observations.device, dtype=torch.float32
            ).view(-1, 1)
        else:
            replay_actor_weights = torch.ones_like(rewards)
            actor_target_actions = actions
            is_online = torch.zeros_like(rewards)

        if weights is None:
            weights = torch.ones(
                (observations.shape[0], 1),
                device=observations.device,
                dtype=torch.float32,
            )
        else:
            weights = weights.to(observations.device, dtype=torch.float32).view(-1, 1)

        log_dict: Dict[str, float] = {}
        with torch.no_grad():
            next_v = self.vf(next_observations)

        value_advantage = self._update_v(observations, actions, weights, log_dict)
        self._update_q(
            next_v,
            observations,
            actions,
            rewards.squeeze(-1),
            dones.squeeze(-1),
            log_dict,
        )

        if self.policy_use_target_advantage:
            with torch.no_grad():
                policy_advantage = self.q_target(observations, actions) - self.vf(observations)
        elif self.policy_use_post_update_advantage:
            with torch.no_grad():
                policy_advantage = self.qf(observations, actions) - self.vf(observations)
        else:
            policy_advantage = value_advantage.detach()

        if update_actor:
            self._update_policy_without_group(
                policy_advantage,
                observations,
                actions,
                actor_target_actions,
                weights,
                rewards,
                dones,
                replay_actor_weights,
                is_online,
                log_dict,
            )
            log_dict["actor_updated"] = 1.0
        else:
            log_dict.update({
                "actor_updated": 0.0,
                "actor_loss": 0.0,
                "bc_loss_mean": 0.0,
                "actor_grad_norm": 0.0,
                "awr_weight_mean": 0.0,
                "awr_weight_max": 0.0,
                "advantage_mean": float(policy_advantage.mean().item()),
                "advantage_std": float(policy_advantage.std(unbiased=False).item()),
                "actor_lr": float(self.actor_optimizer.param_groups[0]["lr"]),
            })
        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }
        if self.actor_lr_schedule is not None:
            state["actor_lr_schedule"] = self.actor_lr_schedule.state_dict()
        return state

    def load_state_dict(
            self,
            state_dict: Dict[str, Any],
            load_optimizers: bool = True,
            load_scheduler: bool = True,
    ):
        self.qf.load_state_dict(state_dict["qf"], strict=True)
        self.vf.load_state_dict(state_dict["vf"], strict=True)
        self.actor.load_state_dict(state_dict["actor"], strict=True)
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(self.device)

        if load_optimizers:
            if "q_optimizer" in state_dict:
                self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
            if "v_optimizer" in state_dict:
                self.v_optimizer.load_state_dict(state_dict["v_optimizer"])
            if "actor_optimizer" in state_dict:
                self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])

        if (
            load_scheduler
            and self.actor_lr_schedule is not None
            and "actor_lr_schedule" in state_dict
        ):
            self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])
        self.total_it = int(state_dict.get("total_it", 0))


def sample_online_batch(
        online_buffer: List,  # 假设是 list of tuples
        batch_size: int,
        device,
        w_min: float = 0.15,  
        min_safe_num: int = 10,  
):

    n = len(online_buffer)
    if n == 0:
        return None


    weights = np.array([tr[5] for tr in online_buffer], dtype=np.float32)




    safe_mask = weights >= w_min
    safe_indices = np.where(safe_mask)[0]


    if len(safe_indices) < min_safe_num:
        return None

    safe_weights = weights[safe_indices]


    probs = safe_weights / (safe_weights.sum() + 1e-8)


    idx_in_safe = np.random.choice(len(safe_indices), size=batch_size, replace=True, p=probs)
    idx = safe_indices[idx_in_safe]


    batch_data = [online_buffer[i] for i in idx]

    # 拆分数据
    states = torch.tensor(np.array([t[0] for t in batch_data]), device=device, dtype=torch.float32)
    actions = torch.tensor(np.array([t[1] for t in batch_data]), device=device, dtype=torch.float32)
    rewards = torch.tensor(np.array([t[2] for t in batch_data]), device=device, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(np.array([t[3] for t in batch_data]), device=device, dtype=torch.float32)
    dones = torch.tensor(np.array([t[4] for t in batch_data]), device=device, dtype=torch.float32).unsqueeze(1)

    return [states, actions, rewards, next_states, dones]






def sample_online_batch_uniform(online_buffer, batch_size, device):

    n = len(online_buffer)
    assert n >= batch_size, "online_buffer 中样本数量不足"

    idx = np.random.choice(n, size=batch_size, replace=True)

    s = np.array([online_buffer[i][0] for i in idx], dtype=np.float32)
    a = np.array([online_buffer[i][1] for i in idx], dtype=np.float32)
    r = np.array([online_buffer[i][2] for i in idx], dtype=np.float32).reshape(-1, 1)
    ns = np.array([online_buffer[i][3] for i in idx], dtype=np.float32)
    d = np.array([online_buffer[i][4] for i in idx], dtype=np.float32).reshape(-1, 1)

    states = torch.tensor(s, device=device)
    actions = torch.tensor(a, device=device)
    rewards = torch.tensor(r, device=device)
    next_states = torch.tensor(ns, device=device)
    dones = torch.tensor(d, device=device)

    return [states, actions, rewards, next_states, dones]






def _cfg(config: TrainConfig, name: str, default):
    return getattr(config, name, default)


def _linear_schedule(start: float, end: float, step: int, warmup: int) -> float:
    if warmup <= 0:
        return float(end)
    ratio = min(1.0, max(0.0, step / float(warmup)))
    return float(start + ratio * (end - start))


@torch.no_grad()
def select_policy_mean_action(actor: nn.Module, state_vec: np.ndarray, device: str) -> np.ndarray:

    was_training = actor.training
    actor.eval()
    state_tensor = torch.tensor(
        state_vec.reshape(1, -1), device=device, dtype=torch.float32
    )
    output = actor(state_tensor)
    action = output.mean if isinstance(output, torch.distributions.Distribution) else output
    max_action = float(getattr(actor, "max_action", 1.0))
    action = torch.clamp(max_action * action, -max_action, max_action)
    if was_training:
        actor.train()
    return action.detach().cpu().numpy().reshape(-1).astype(np.float32)


@torch.no_grad()
def sample_policy_action(actor: nn.Module, state_vec: np.ndarray, device: str) -> np.ndarray:
    """Sample bar(a)_t from pi_phi(.|s_t) before adding external Gaussian noise."""
    was_training = actor.training
    actor.train()
    state_tensor = torch.tensor(
        state_vec.reshape(1, -1), device=device, dtype=torch.float32
    )
    output = actor(state_tensor)
    if isinstance(output, torch.distributions.Distribution):
        action = output.sample()
    else:
        action = output
    max_action = float(getattr(actor, "max_action", 1.0))
    action = torch.clamp(max_action * action, -max_action, max_action)
    actor.train(was_training)
    return action.detach().cpu().numpy().reshape(-1).astype(np.float32)


def _safe_get_intrinsic_reward_sa(rnd, state_vec, action_vec, update_stats=False) -> float:
    try:
        value = rnd.get_intrinsic_reward(state_vec, action_vec, update_stats=update_stats)
    except TypeError:
        try:
            value = rnd.get_intrinsic_reward(state_vec, action_vec)
        except TypeError:
            value = rnd.get_intrinsic_reward(state_vec)
    return float(np.asarray(value).mean())


def _novelty_to_weight_np(z_value, config: TrainConfig) -> float:
    kappa = float(_cfg(config, "novelty_kappa", 2.0))
    delta = float(_cfg(config, "novelty_delta", 1.5))
    clip_c = float(_cfg(config, "novelty_clip", 5.0))
    z = float(np.clip(float(np.asarray(z_value).mean()), -clip_c, clip_c))
    weight = 1.0 / (1.0 + np.exp(kappa * (z - delta)))
    return float(np.clip(weight, 0.0, 1.0))


def _get_raw_error_batch_safe(rnd, states, actions):
    if hasattr(rnd, "get_raw_error_batch"):
        try:
            error = rnd.get_raw_error_batch(states, actions)
        except TypeError:
            error = rnd.get_raw_error_batch(states)
        if not torch.is_tensor(error):
            error = torch.tensor(error, device=states.device, dtype=torch.float32)
        return error.to(states.device, dtype=torch.float32).view(-1)

    states_np = states.detach().cpu().numpy()
    actions_np = actions.detach().cpu().numpy()
    values = [
        _safe_get_intrinsic_reward_sa(rnd, s, a, update_stats=False)
        for s, a in zip(states_np, actions_np)
    ]
    return torch.tensor(values, device=states.device, dtype=torch.float32)


def _normalized_novelty_batch(rnd, raw_error: torch.Tensor) -> torch.Tensor:
    if hasattr(rnd, "rms") and hasattr(rnd.rms, "mean") and hasattr(rnd.rms, "var"):
        mean = torch.as_tensor(
            rnd.rms.mean, device=raw_error.device, dtype=raw_error.dtype
        )
        var = torch.as_tensor(
            rnd.rms.var, device=raw_error.device, dtype=raw_error.dtype
        )
        return (raw_error - mean) / (torch.sqrt(var.clamp(min=1e-12)) + 1e-8)
    return raw_error


def compute_novelty_weights_batch(
        rnd,
        states: torch.Tensor,
        actions: torch.Tensor,
        config: TrainConfig,
        stored_weights: Optional[torch.Tensor] = None,
        source_is_online: Optional[torch.Tensor] = None,
        return_stats: bool = False,
        rnd_anchor=None,
):

    with torch.no_grad():
        raw_error = _get_raw_error_batch_safe(rnd, states, actions)
        z_adaptive = _normalized_novelty_batch(rnd, raw_error)

        if rnd_anchor is not None:
            raw_error_anchor = _get_raw_error_batch_safe(rnd_anchor, states, actions)
            z_anchor = _normalized_novelty_batch(rnd_anchor, raw_error_anchor)
        else:
            raw_error_anchor = raw_error
            z_anchor = z_adaptive

        clip_c = float(_cfg(config, "novelty_clip", 5.0))
        kappa = float(_cfg(config, "novelty_kappa", 3.0))
        delta = float(_cfg(config, "novelty_delta", 0.5))
        z_adaptive = z_adaptive.clamp(-clip_c, clip_c)
        z_anchor = z_anchor.clamp(-clip_c, clip_c)

        sigmoid_adaptive = 1.0 / (1.0 + torch.exp(kappa * (z_adaptive - delta)))
        sigmoid_anchor = 1.0 / (1.0 + torch.exp(kappa * (z_anchor - delta)))

        if source_is_online is None:
            online_mask = torch.ones_like(sigmoid_adaptive, dtype=torch.bool)
        else:
            online_mask = source_is_online.view(-1).to(states.device) > 0.5
        offline_mask = ~online_mask

        online_floor = float(np.clip(
            _cfg(config, "online_novelty_weight_floor", 0.15), 0.0, 1.0
        ))
        offline_weight = float(np.clip(
            _cfg(config, "offline_novelty_weight", 1.0), 0.0, 1.0
        ))
        adaptive_weight = online_floor + (1.0 - online_floor) * sigmoid_adaptive
        anchor_weight = online_floor + (1.0 - online_floor) * sigmoid_anchor

        use_anchor = bool(_cfg(config, "use_frozen_rnd_anchor", True)) and rnd_anchor is not None
        relaxation_cap = float(np.clip(
            _cfg(config, "novelty_anchor_relaxation_cap", 0.10), 0.0, 1.0
        ))
        if use_anchor:
            online_weight = torch.minimum(
                adaptive_weight,
                (anchor_weight + relaxation_cap).clamp(max=1.0),
            )

            if stored_weights is not None:
                actor_valid = stored_weights.view(-1).to(states.device) > 0.5
                failed_online = online_mask & ~actor_valid
                if bool(failed_online.any()):
                    online_weight = online_weight.clone()
                    online_weight[failed_online] = torch.minimum(
                        adaptive_weight[failed_online],
                        anchor_weight[failed_online],
                    )
        else:
            online_weight = adaptive_weight

        weights = online_weight.clone()
        weights[offline_mask] = offline_weight
        weights = weights.view(-1, 1).clamp(1e-8, 1.0)

        def masked_mean(value, mask):
            return float(value[mask].mean().item()) if bool(mask.any()) else float("nan")

        clipped_relaxation = (
            adaptive_weight > (anchor_weight + relaxation_cap).clamp(max=1.0)
        ) & online_mask
        stats = {
            "rnd_error_mean": float(raw_error.mean().item()),
            "rnd_error_std": float(raw_error.std(unbiased=False).item()),
            "rnd_anchor_error_mean": float(raw_error_anchor.mean().item()),
            "novelty_z_mean": float(z_adaptive.mean().item()),
            "novelty_z_std": float(z_adaptive.std(unbiased=False).item()),
            "novelty_w_mean": float(weights.mean().item()),
            "novelty_w_min": float(weights.min().item()),
            "novelty_w_max": float(weights.max().item()),
            "online_z_mean": masked_mean(z_adaptive, online_mask),
            "online_anchor_z_mean": masked_mean(z_anchor, online_mask),
            "offline_z_mean": masked_mean(z_adaptive, offline_mask),
            "online_w_mean": masked_mean(weights.view(-1), online_mask),
            "online_adaptive_w_mean": masked_mean(adaptive_weight, online_mask),
            "online_anchor_w_mean": masked_mean(anchor_weight, online_mask),
            "offline_w_mean": masked_mean(weights.view(-1), offline_mask),
            "anchor_relax_clip_fraction": (
                float(clipped_relaxation.float().sum().item())
                / max(float(online_mask.float().sum().item()), 1.0)
            ),
        }
        return (weights, stats) if return_stats else weights


def _train_conflict_aware_rnd(
        rnd,
        online_pairs,
        dual_buffer: DualReplayBuffer,
        config: TrainConfig,
):

    batch_size = min(int(config.online_batch_size), len(online_pairs))
    if batch_size <= 0:
        return {"rnd_update": 0.0}

    indices = np.random.choice(len(online_pairs), size=batch_size, replace=False)
    online_states = torch.tensor(
        np.asarray([online_pairs[i][0] for i in indices], dtype=np.float32),
        device=config.device,
    )
    online_actions = torch.tensor(
        np.asarray([online_pairs[i][1] for i in indices], dtype=np.float32),
        device=config.device,
    )
    offline_batch = dual_buffer.offline_buffer.sample(batch_size)
    offline_states = offline_batch[0].to(config.device)
    offline_actions = offline_batch[1].to(config.device)

    if not hasattr(rnd, "train_conflict_aware"):
        raise AttributeError(
            "RND does not implement train_conflict_aware(); the conflict-aware "
            "novelty update required by CANOR cannot run."
        )

    rnd.use_normalized_error_variance = bool(
        _cfg(config, "rnd_use_normalized_error_variance", True)
    )
    rnd.conflict_grad_clip = float(_cfg(config, "rnd_grad_clip", 0.5))

    try:
        result = rnd.train_conflict_aware(
            online_states, online_actions, offline_states, offline_actions
        )
    except TypeError:
        result = rnd.train_conflict_aware(online_states, offline_states)

    metrics = {"rnd_update": 1.0}
    if isinstance(result, dict):
        for key, value in result.items():
            try:
                metrics[f"rnd_{key}"] = float(value)
            except (TypeError, ValueError):
                pass
    elif isinstance(result, (tuple, list)):
        names = ["loss_on", "loss_off", "trust"]
        for name, value in zip(names, result):
            try:
                metrics[f"rnd_{name}"] = float(value)
            except (TypeError, ValueError):
                pass
        metrics.setdefault("rnd_conflict", 0.0)
    elif result is not None:
        try:
            metrics["rnd_loss"] = float(result)
        except (TypeError, ValueError):
            pass
    return metrics


def _mean_logs(logs: List[Dict[str, float]]) -> Dict[str, float]:
    if not logs:
        return {}
    keys = set().union(*(item.keys() for item in logs))
    return {
        key: float(np.mean([item[key] for item in logs if key in item]))
        for key in keys
    }


def _snapshot_parameters(module: nn.Module) -> List[torch.Tensor]:
    return [p.detach().cpu().clone() for p in module.parameters()]


def _relative_parameter_drift(module: nn.Module, reference: List[torch.Tensor]) -> float:
    with torch.no_grad():
        diff_sq = 0.0
        ref_sq = 0.0
        for parameter, ref in zip(module.parameters(), reference):
            current = parameter.detach().cpu()
            diff_sq += float((current - ref).pow(2).sum().item())
            ref_sq += float(ref.pow(2).sum().item())
        return float(np.sqrt(diff_sq / (ref_sq + 1e-12)))


def _scheduled_online_candidate_ratio(config: TrainConfig, episode_index: int) -> float:

    start = float(np.clip(config.online_sampling_ratio_start, 0.0, 1.0))
    final = float(np.clip(config.online_sampling_ratio_final, 0.0, 1.0))
    warmup = int(config.online_sampling_warmup_episodes)
    if warmup <= 0:
        return final
    progress = min(max(float(episode_index) / float(warmup), 0.0), 1.0)
    return start + progress * (final - start)


def _scheduled_exploration_std(config: TrainConfig, episode_index: int) -> float:
    start = float(_cfg(config, "exploration_std_start", config.exploration_std))
    mid = float(_cfg(config, "exploration_std_mid", start))
    mid_ep = max(int(_cfg(config, "exploration_mid_episode", 0)), 0)
    final = float(_cfg(config, "exploration_std_final", mid))
    end_ep = max(int(_cfg(config, "exploration_decay_episodes", mid_ep)), mid_ep + 1)
    ep = max(int(episode_index), 0)
    if mid_ep > 0 and ep <= mid_ep:
        progress = float(ep) / float(mid_ep)
        return float(start + progress * (mid - start))
    progress = float(np.clip((ep - mid_ep) / float(end_ep - mid_ep), 0.0, 1.0))
    return float(mid + progress * (final - mid))



def _linear_schedule(start: float, final: float, episode_index: int, duration: int) -> float:
    if int(duration) <= 0:
        return float(final)
    progress = float(np.clip(float(episode_index) / float(duration), 0.0, 1.0))
    return float(start + progress * (final - start))


def _scheduled_actor_lr(config: TrainConfig, episode_index: int) -> float:
    start_lr = float(config.online_actor_lr)
    min_lr = float(config.online_actor_lr_min)
    start_ep = int(config.actor_lr_decay_start_episode)
    end_ep = max(int(config.actor_lr_decay_end_episode), start_ep + 1)
    if episode_index <= start_ep:
        return start_lr
    if episode_index >= end_ep:
        return min_lr
    progress = (float(episode_index) - start_ep) / float(end_ep - start_ep)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(min_lr + (start_lr - min_lr) * cosine)


def _scheduled_actor_online_mix(config: TrainConfig, episode_index: int) -> float:
    return _linear_schedule(
        float(config.actor_online_loss_fraction_start),
        float(config.actor_online_loss_fraction_final),
        episode_index,
        int(config.actor_mix_warmup_episodes),
    )


def _scheduled_collision_fraction(config: TrainConfig, episode_index: int) -> float:
    return _linear_schedule(
        float(config.collision_sampling_fraction_start),
        float(config.collision_sampling_fraction_final),
        episode_index,
        int(config.collision_sampling_decay_episodes),
    )


def _set_rnd_predictor_trainable(rnd, trainable: bool, lr: Optional[float] = None):
    modules = []
    if hasattr(rnd, "predictor_bilinear"):
        modules.append(rnd.predictor_bilinear)
    if hasattr(rnd, "predictor"):
        modules.append(rnd.predictor)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(bool(trainable))
            if not trainable:
                parameter.grad = None
    if lr is not None and hasattr(rnd, "optimizer"):
        for group in rnd.optimizer.param_groups:
            group["lr"] = float(lr)
    rnd.train(bool(trainable))


def online_finetune_weighted_loss(
        trainer,
        rnd,
        dual_buffer: DualReplayBuffer,
        config: TrainConfig,
        logger: Optional[logging.Logger] = None,
        initial_eval_metrics: Optional[Dict[str, float]] = None,
):

    env_config = configparser.RawConfigParser()
    env_config.read(config.env_config_path)
    env = gym.make(config.env)
    env.configure(env_config)
    robot = Robot(env_config, "robot")
    env.set_robot(robot)
    robot.set_policy(trainer.actor)
    set_seed(config.seed, env)

    device = config.device
    recent_success: List[int] = []
    recent_collision: List[int] = []
    recent_reward: List[float] = []
    recent_online_pairs: List[Tuple[np.ndarray, np.ndarray]] = []

    best_score = -float("inf")
    best_episode = 0
    best_metrics: Dict[str, float] = {}
    best_checkpoint_path = os.path.join(
        config.checkpoints_path, str(config.best_checkpoint_name)
    )
    best_actor_path = os.path.join(
        config.checkpoints_path, str(config.best_actor_name)
    )
    if hasattr(trainer, "set_actor_reference"):
        trainer.set_actor_reference()
    actor_reference = _snapshot_parameters(trainer.actor)
    total_gradient_updates = 0
    total_actor_updates = 0
    total_rnd_updates = 0

    if initial_eval_metrics is not None:
        best_score = float(initial_eval_metrics["success_rate"]) - 2.0 * float(
            initial_eval_metrics["collision_rate"]
        )
        best_metrics = {
            key: float(value)
            for key, value in initial_eval_metrics.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        }
    torch.save(trainer.state_dict(), best_checkpoint_path)
    torch.save(trainer.actor.state_dict(), best_actor_path)


    rnd_anchor = None
    if bool(_cfg(config, "use_frozen_rnd_anchor", True)):
        rnd_anchor = copy.deepcopy(rnd).to(device)
        rnd_anchor.eval()
        rnd_anchor.requires_grad_(False)
        if hasattr(rnd_anchor, "optimizer"):
            rnd_anchor.optimizer = None


    rnd.eval()
    for parameter in rnd.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    if hasattr(rnd, "optimizer"):
        rnd.optimizer.zero_grad(set_to_none=True)
        for group in rnd.optimizer.param_groups:
            group["lr"] = float(config.online_rnd_lr)
    rnd_adaptation_enabled = False

    for ep in range(int(config.online_episodes)):
        observation = env.reset(phase="train")
        done = False
        info = None
        episode_reward = 0.0
        gamma = 1.0
        episode_steps = 0
        episode_transitions = []

        base_sigma = _scheduled_exploration_std(config, ep)
        recent_coll_rate = float(np.mean(recent_collision[-50:])) if recent_collision else 0.0
        safety_scale = 1.0
        if recent_coll_rate > 0.15:
            safety_scale = max(
                0.65,
                1.0 - float(config.exploration_std_collision_scale)
                * (recent_coll_rate - 0.15),
            )
        sigma_exp = float(base_sigma * safety_scale)

        while not done:
            joint_state = JointState(robot.get_full_state(), observation)
            state_vec = to_np(
                transform(joint_state, device).view(1, -1).squeeze(0)
            ).astype(np.float32)

            policy_action = select_policy_mean_action(
                trainer.actor, state_vec, device
            )
            exploration_noise = np.random.normal(
                0.0, sigma_exp, size=policy_action.shape
            ).astype(np.float32)
            executed_action = np.clip(
                policy_action + exploration_noise,
                config.action_low,
                config.action_high,
            ).astype(np.float32)

            next_observation, reward, done, info = env.step(
                ActionXY(float(executed_action[0]), float(executed_action[1]))
            )
            episode_reward += float(reward) * gamma
            gamma *= config.discount
            episode_steps += 1

            next_joint_state = JointState(robot.get_full_state(), next_observation)
            next_state_vec = to_np(
                transform(next_joint_state, device).view(1, -1).squeeze(0)
            ).astype(np.float32)

            episode_transitions.append((
                state_vec,
                executed_action,
                float(reward),
                next_state_vec,
                float(done),
                policy_action,
            ))
            observation = next_observation

        success = int(isinstance(info, ReachGoal))
        collision = int(isinstance(info, Collision))
        timeout = int(isinstance(info, Timeout))
        nav_time = float(env.global_time) if success else 0.0


        backprop_steps = max(int(config.collision_backprop_steps), 1)
        success_only_actor = bool(_cfg(config, "actor_success_only_online", True))
        for index, transition in enumerate(episode_transitions):
            steps_from_end = len(episode_transitions) - index
            failure_boundary = bool(collision) and steps_from_end <= backprop_steps
            if success_only_actor:
                actor_replay_weight = 1.0 if bool(success) else 0.0
            else:
                actor_replay_weight = 0.0 if failure_boundary else 1.0
            dual_buffer.add_online_transition(
                transition[0], transition[1], transition[2], transition[3],
                transition[4], weight=actor_replay_weight,
                actor_action=transition[5],
            )


        allow_rnd_episode = bool(success) or not bool(
            _cfg(config, "rnd_adapt_success_only", True)
        )
        if allow_rnd_episode and episode_transitions:
            pair_states = torch.as_tensor(
                np.asarray([item[0] for item in episode_transitions], dtype=np.float32),
                device=device,
            )
            pair_actions = torch.as_tensor(
                np.asarray([item[1] for item in episode_transitions], dtype=np.float32),
                device=device,
            )
            pair_source = torch.ones(
                (pair_states.shape[0], 1), device=device, dtype=torch.float32
            )
            pair_weight_estimator = rnd_anchor if rnd_anchor is not None else rnd
            trusted_weights = compute_novelty_weights_batch(
                pair_weight_estimator,
                pair_states,
                pair_actions,
                config,
                source_is_online=pair_source,
                return_stats=False,
                rnd_anchor=None,
            ).view(-1)
            min_trusted_weight = float(
                _cfg(config, "rnd_adapt_min_anchor_weight", 0.25)
            )
            trusted_mask = trusted_weights >= min_trusted_weight
            for transition, trusted in zip(episode_transitions, trusted_mask.tolist()):
                if bool(trusted):
                    recent_online_pairs.append((transition[0], transition[1]))

        capacity = max(int(config.rnd_online_pair_capacity), int(config.online_batch_size))
        if len(recent_online_pairs) > capacity:
            recent_online_pairs = recent_online_pairs[-capacity:]

        recent_success.append(success)
        recent_collision.append(collision)
        recent_reward.append(float(episode_reward))
        recent_success = recent_success[-200:]
        recent_collision = recent_collision[-200:]
        recent_reward = recent_reward[-200:]

        base_online_ratio = _scheduled_online_candidate_ratio(config, ep)
        current_actor_lr = _scheduled_actor_lr(config, ep)
        base_actor_mix = _scheduled_actor_online_mix(config, ep)
        current_collision_fraction = _scheduled_collision_fraction(config, ep)

        collision_target = float(_cfg(config, "replay_safety_collision_target", 0.15))
        collision_excess = max(recent_coll_rate - collision_target, 0.0)
        current_online_ratio = max(
            float(_cfg(config, "online_sampling_ratio_min", 0.35)),
            base_online_ratio
            - float(_cfg(config, "replay_safety_gain", 0.75)) * collision_excess,
        )
        current_actor_mix = max(
            float(_cfg(config, "actor_online_mix_min", 0.30)),
            base_actor_mix
            - float(_cfg(config, "actor_mix_safety_gain", 1.0)) * collision_excess,
        )
        trainer.set_actor_lr(current_actor_lr)
        trainer.actor_online_loss_fraction = current_actor_mix
        trainer.normalize_policy_advantage = bool(config.normalize_policy_advantage)

        num_updates = int(
            dual_buffer.online_buffer._size
            >= int(config.min_online_size_for_updates)
        )

        update_logs: List[Dict[str, float]] = []
        batch_logs: List[Dict[str, float]] = []
        for _ in range(num_updates):
            batch, batch_info = dual_buffer.sample_bootstrap_selected(
                batch_size=int(config.batch_size),
                online_ratio=current_online_ratio,
                p_on=float(config.bootstrap_p_on),
                p_off=float(config.bootstrap_p_off),
                candidate_multiplier=1,
                online_failure_fraction=current_collision_fraction,
                return_info=True,
            )
            batch = [tensor.to(device) for tensor in batch]
            novelty_weights, novelty_stats = compute_novelty_weights_batch(
                rnd,
                batch[0],
                batch[1],
                config,
                stored_weights=batch[5],
                source_is_online=batch[7],
                return_stats=True,
                rnd_anchor=rnd_anchor,
            )

            train_stats = trainer.train(
                batch,
                weights=novelty_weights,
                update_actor=True,
            )
            total_gradient_updates += 1
            total_actor_updates += 1
            merged = {**novelty_stats, **train_stats}

            with torch.no_grad():
                failure_mask = batch[5].view(-1) <= 0.5
                online_mask = batch[7].view(-1) > 0.5
                online_failure_mask = failure_mask & online_mask
                merged["failure_sample_fraction"] = float(
                    online_failure_mask.float().sum().item()
                    / max(float(online_mask.float().sum().item()), 1.0)
                )
                merged["failure_w_mean"] = (
                    float(novelty_weights.view(-1)[online_failure_mask].mean().item())
                    if bool(online_failure_mask.any()) else float("nan")
                )
            update_logs.append(merged)
            batch_logs.append({key: float(value) for key, value in batch_info.items()})


        rnd_metrics = {
            "rnd_update": 0.0,
            "rnd_conflict": 0.0,
            "rnd_trust": 1.0,
            "rnd_loss_on": 0.0,
            "rnd_loss_off": 0.0,
        }
        should_update_rnd = (
            bool(config.update_rnd_online)
            and not bool(config.freeze_rnd_online)
            and (ep + 1) >= int(config.novelty_warmup_episodes)
            and int(config.novelty_update_interval) > 0
            and (ep + 1) % int(config.novelty_update_interval) == 0
            and len(recent_online_pairs) >= int(config.online_batch_size)
        )
        if should_update_rnd:
            if not rnd_adaptation_enabled:
                _set_rnd_predictor_trainable(
                    rnd, True, lr=float(config.online_rnd_lr)
                )
                rnd_adaptation_enabled = True
                if logger is not None:
                    logger.info(
                        "Enabled delayed conflict-aware RND adaptation at ep=%d",
                        ep + 1,
                    )
            result = _train_conflict_aware_rnd(
                rnd, recent_online_pairs, dual_buffer, config
            )
            rnd_metrics.update(result)
            total_rnd_updates += int(result.get("rnd_update", 0.0) > 0.5)
            rnd.eval()

        if update_logs:
            update_stats = _mean_logs(update_logs)
            batch_info_mean = _mean_logs(batch_logs)
        else:
            update_stats = {
                "novelty_w_mean": 0.0,
                "online_w_mean": 0.0,
                "offline_w_mean": 0.0,
                "online_z_mean": 0.0,
                "q_loss": 0.0,
                "value_loss": 0.0,
                "actor_loss": 0.0,
                "actor_recovery_loss": 0.0,
                "actor_recovery_accept_fraction": 0.0,
                "actor_recovery_q_gain": 0.0,
                "online_advantage_threshold": 0.0,
                "advantage_mean": 0.0,
                "advantage_std": 0.0,
                "advantage_scale": 0.0,
                "bc_loss_mean": 0.0,
                "q_grad_norm": 0.0,
                "v_grad_norm": 0.0,
                "actor_grad_norm": 0.0,
                "tau_w_mean": 0.0,
                "failure_w_mean": float("nan"),
                "failure_sample_fraction": 0.0,
                "awr_weight_max": 0.0,
                "actor_updated": 0.0,
            }
            batch_info_mean = {
                "candidate_online": 0.0,
                "candidate_offline": 0.0,
                "retained_online": 0.0,
                "retained_offline": 0.0,
                "retained_total": 0.0,
                "online_fraction": 0.0,
                "failure_candidate_fraction": 0.0,
            }

        update_stats.update(rnd_metrics)
        update_stats["actor_relative_drift"] = _relative_parameter_drift(
            trainer.actor, actor_reference
        )
        window_success = float(np.mean(recent_success[-100:]))
        window_collision = float(np.mean(recent_collision[-100:]))
        window_reward = float(np.mean(recent_reward[-100:]))

        if logger is not None:
            logger.info(
                "[ONLINE_EXPLORE] ep=%d/%d | succ=%d | coll=%d | timeout=%d "
                "| nav_time=%.3f | reward=%.4f | steps=%d | buffer=%d "
                "| explore_win_succ=%.3f | explore_win_coll=%.3f | sigma=%.4f "
                "| online_ratio=%.3f | warmup=%d/%d | updates=%d | actor_updates=%d "
                "| total_updates=%d | B_on=%.1f B_off=%.1f | Bsel_on=%.1f "
                "Bsel_off=%.1f Bsel=%.1f | Bsel_online_frac=%.3f "
                "| failure_target=%.3f | failure_actual=%.3f | w_mean=%.4f "
                "| w_on=%.4f | w_anchor=%.4f | w_adapt=%.4f | anchor_clip=%.3f "
                "| w_off=%.4f | z_on=%.4f | z_anchor=%.4f | failure_w=%.4f "
                "| tau_w=%.4f | q_loss=%.5f | v_loss=%.5f | q_grad=%.4f "
                "| v_grad=%.4f | actor_loss=%.5f | anchor_loss=%.5f "
                "| recovery_loss=%.5f | recovery_accept=%.3f "
                "| recovery_q_gain=%.4f | adv_threshold=%.4f "
                "| bc_loss=%.5f | actor_grad=%.4f | online_pos=%.3f "
                "| actor_projected=%d | awr_max=%.3f | adv=%.4f | adv_std=%.4f "
                "| adv_scale=%.4f | actor_drift=%.6f | actor_lr=%.2e "
                "| actor_online_mix=%.3f | adv_norm=%d | rnd_frozen=%d "
                "| rnd_update=%d | rnd_conflict=%d | rnd_trust=%.3f "
                "| rnd_norm_var=%.4f | rnd_grad_cos=%.4f",
                ep + 1,
                int(config.online_episodes),
                success,
                collision,
                timeout,
                nav_time,
                episode_reward,
                episode_steps,
                dual_buffer.online_buffer._size,
                window_success,
                window_collision,
                sigma_exp,
                current_online_ratio,
                min(dual_buffer.online_buffer._size, int(config.min_online_size_for_updates)),
                int(config.min_online_size_for_updates),
                num_updates,
                int(round(update_stats.get("actor_updated", 0.0) * num_updates)),
                total_gradient_updates,
                batch_info_mean.get("candidate_online", 0.0),
                batch_info_mean.get("candidate_offline", 0.0),
                batch_info_mean.get("retained_online", 0.0),
                batch_info_mean.get("retained_offline", 0.0),
                batch_info_mean.get("retained_total", 0.0),
                batch_info_mean.get("online_fraction", 0.0),
                current_collision_fraction,
                update_stats.get("failure_sample_fraction", 0.0),
                update_stats.get("novelty_w_mean", 0.0),
                update_stats.get("online_w_mean", 0.0),
                update_stats.get("online_anchor_w_mean", 0.0),
                update_stats.get("online_adaptive_w_mean", 0.0),
                update_stats.get("anchor_relax_clip_fraction", 0.0),
                update_stats.get("offline_w_mean", 0.0),
                update_stats.get("online_z_mean", 0.0),
                update_stats.get("online_anchor_z_mean", 0.0),
                update_stats.get("failure_w_mean", float("nan")),
                update_stats.get("tau_w_mean", 0.0),
                update_stats.get("q_loss", 0.0),
                update_stats.get("value_loss", 0.0),
                update_stats.get("q_grad_norm", 0.0),
                update_stats.get("v_grad_norm", 0.0),
                update_stats.get("actor_loss", 0.0),
                update_stats.get("actor_anchor_loss", 0.0),
                update_stats.get("actor_recovery_loss", 0.0),
                update_stats.get("actor_recovery_accept_fraction", 0.0),
                update_stats.get("actor_recovery_q_gain", 0.0),
                update_stats.get("online_advantage_threshold", 0.0),
                update_stats.get("bc_loss_mean", 0.0),
                update_stats.get("actor_grad_norm", 0.0),
                update_stats.get("online_positive_advantage_fraction", 0.0),
                int(update_stats.get("actor_projected", 0.0) > 0.5),
                update_stats.get("awr_weight_max", 0.0),
                update_stats.get("advantage_mean", 0.0),
                update_stats.get("advantage_std", 0.0),
                update_stats.get("advantage_scale", 0.0),
                update_stats.get("actor_relative_drift", 0.0),
                current_actor_lr,
                current_actor_mix,
                int(bool(config.normalize_policy_advantage)),
                int(not rnd_adaptation_enabled),
                int(update_stats.get("rnd_update", 0.0) > 0.5),
                int(update_stats.get("rnd_conflict", 0.0) > 0.5),
                update_stats.get("rnd_trust", 1.0),
                update_stats.get("rnd_normalized_error_var", 0.0),
                update_stats.get("rnd_grad_cosine", 0.0),
            )

        if int(config.online_eval_interval) > 0 and (
            ep + 1
        ) % int(config.online_eval_interval) == 0:
            if logger is not None:
                logger.info("========== ONLINE PERIODIC TEST ep=%d ==========", ep + 1)
            metrics = test_actor(
                env,
                trainer.actor,
                device=device,
                n_episodes=int(config.online_eval_episodes),
                seed=int(config.online_eval_seed),
                logger=logger,
                test_case_start=0,
            )
            score = metrics["success_rate"] - 2.0 * metrics["collision_rate"]
            if bool(config.save_online_best) and score > best_score:
                best_score = float(score)
                best_episode = int(ep + 1)
                best_metrics = {
                    key: float(value)
                    for key, value in metrics.items()
                    if isinstance(value, (int, float, np.integer, np.floating))
                }
                torch.save(trainer.state_dict(), best_checkpoint_path)
                torch.save(trainer.actor.state_dict(), best_actor_path)
                if logger is not None:
                    logger.info(
                        "Saved new best online checkpoint: ep=%d | score=%.4f "
                        "| success=%.3f | collision=%.3f",
                        best_episode,
                        best_score,
                        metrics["success_rate"],
                        metrics["collision_rate"],
                    )

    if (
        bool(config.restore_best_online_at_end)
        and bool(config.save_online_best)
        and os.path.isfile(best_checkpoint_path)
    ):
        best_state = torch.load(best_checkpoint_path, map_location=device)
        trainer.load_state_dict(
            best_state, load_optimizers=False, load_scheduler=False
        )
        trainer.reset_online_optimizers(
            q_lr=config.online_qf_lr,
            v_lr=config.online_vf_lr,
            actor_lr=max(config.online_actor_lr_min, 1e-8),
            actor_scheduler_steps=None,
        )
        robot.set_policy(trainer.actor)
        if logger is not None:
            logger.info(
                "Restored best online checkpoint before final evaluation: "
                "ep=%d | score=%.4f | path=%s",
                best_episode,
                best_score,
                best_checkpoint_path,
            )

    return {
        "window_success": float(np.mean(recent_success[-100:])),
        "window_collision": float(np.mean(recent_collision[-100:])),
        "window_reward": float(np.mean(recent_reward[-100:])),
        "best_episode": int(best_episode),
        "best_score": float(best_score),
        "best_success": float(best_metrics.get("success_rate", 0.0)),
        "best_collision": float(best_metrics.get("collision_rate", 0.0)),
        "total_gradient_updates": int(total_gradient_updates),
        "total_actor_updates": int(total_actor_updates),
        "total_rnd_updates": int(total_rnd_updates),
        "best_checkpoint_path": best_checkpoint_path,
        "best_actor_path": best_actor_path,
    }


@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env)

    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]

    state_dim = 91
    action_dim = 2

    # dataset = d4rl.qlearning_dataset(env)

    file_path = r"crowd_nav/Crowd_nav_5/Crowd_nav_5.hdf5"
    with h5py.File(file_path, 'r') as f:
        dataset = {k: np.array(v) for k, v in f.items()}

    print("最小值:", dataset["rewards"].min(), "最大值:", dataset["rewards"].max())

    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset_old(dataset)

    # ====== 新增：初始化 RND，用离线数据训练它 ======
    rnd = RND(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config.rnd_hidden_dim,
        output_dim=config.rnd_output_dim,
        lr=config.rnd_lr,
        device=config.device
    )

    # max_action = float(env.action_space.high[0])
    max_action = float(1)


    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
    ).to(config.device)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
    }

    print("---------------------------------------")
    print(f"Training IQL, Env: {config.env}, Seed: {seed}, Device: {config.device}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ImplicitQLearning(**kwargs)


    os.makedirs(config.checkpoints_path, exist_ok=True)
    logger = setup_logger(config.checkpoints_path, name="iql")
    logger.info("Start training IQL | env=%s | seed=%d | device=%s", config.env, config.seed, config.device)

    # wandb_init(asdict(config))
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    evaluations = []
    for t in range(int(config.max_timesteps)):

        batch = replay_buffer.sample(config.batch_size)
        # print(batch)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)



        observations = batch[0]  # [B, state_dim]
        actions = batch[1]       # [B, action_dim]
        rnd.train_predictor(observations,actions)

        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            metrics = eval_actor(env, actor, list1, list2, list3, list4, device=config.device,
                                 n_episodes=config.n_episodes, seed=config.seed, logger=logger, global_step=t + 1)
            # 保存 checkpoint
            if config.checkpoints_path:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"iql_checkpoint_{t}.pt"),
                )
                rnd_offline_path = os.path.join(config.checkpoints_path, f"rnd_checkpoint_{t}.pt")
                torch.save(
                    {
                        "rnd_state": rnd.state_dict(),
                        "rnd_optimizer": rnd.optimizer.state_dict(),
                    },
                    rnd_offline_path,
                )
    test_actor(env, actor, device=config.device, n_episodes=500, seed=config.seed, logger=logger, )
    list1 = np.array(list1)
    list2 = np.array(list2)
    list3 = np.array(list3)
    list4 = np.array(list4)
    print(list1)
    print(list2)
    print(list3)
    print(list4)
    logger.info("Training finished. Saving model...")
    # torch.save(trainer.state_dict(), os.path.join(config.checkpoints_path, f"iql.pt"), )
    # np.savez('D:\Program Files (x86)\IDE\JetBrains\PycharmProjects\CrowdNav-master_danren\crowd_nav\data\output_iql\success2.pth', success = list1, collision = list2, time = list3, reward = list4)

    # ========= 离线阶段收尾：保存所有在线微调会用到的东西 =========
    logger.info("Offline training finished. Saving offline checkpoints...")

    # 1) 保存完整 IQL trainer（含 actor/q/v + optimizer + lr_schedule + total_it）
    offline_iql_path = os.path.join(config.checkpoints_path, "iql_offline.pt")
    torch.save(trainer.state_dict(), offline_iql_path)

    # 2) 单独保存一份 actor，方便之后直接加载策略部署用
    actor_offline_path = os.path.join(config.checkpoints_path, "actor_offline.pt")
    torch.save(trainer.actor.state_dict(), actor_offline_path)

    # 3) 保存 RND 网络 + 其优化器（在线阶段要继续训 RND）
    rnd_offline_path = os.path.join(config.checkpoints_path, "rnd_offline.pt")
    torch.save(
        {
            "rnd_state": rnd.state_dict(),
            "rnd_optimizer": rnd.optimizer.state_dict(),
        },
        rnd_offline_path,
    )

    logger.info("Saved offline IQL to %s", offline_iql_path)
    logger.info("Saved offline actor to %s", actor_offline_path)
    logger.info("Saved offline RND to %s", rnd_offline_path)




if __name__ == "__main__":
    train()


