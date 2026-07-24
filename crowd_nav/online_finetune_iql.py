import copy
import logging
import math
import os
import sys
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import gym
import h5py
import numpy as np
import pyrallis
import torch

try:
    from crowd_nav.iql.py import (
        ActionXY,
        Collision,
        DeterministicPolicy,
        DualReplayBuffer,
        GaussianPolicy,
        ImplicitQLearning,
        JointState,
        ReachGoal,
        ReplayBuffer,
        Robot,
        Timeout,
        TrainConfig,
        TwinQ,
        ValueFunction,
        get_state_dim,
        load_crowdnav_env_config,
        set_seed,
        test_actor,
        to_np,
        transform,
        validate_dataset,
        validate_state_vector,
    )



def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("iql_online_finetune_selective_v4")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(
        os.path.join(log_dir, "train_eval_iql_online_selective_v4-v1.log"),
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def linear_schedule(
    start: float,
    end: float,
    current: int,
    duration: int,
) -> float:
    if duration <= 0:
        return float(end)
    progress = min(max(float(current) / float(duration), 0.0), 1.0)
    return float(start + (end - start) * progress)


def cosine_schedule(
    start: float,
    end: float,
    current: int,
    duration: int,
) -> float:
    if duration <= 0:
        return float(end)
    progress = min(max(float(current) / float(duration), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(end + (start - end) * cosine)


def set_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)


def get_optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0]["lr"])


def clone_state_dict_cpu(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


@torch.no_grad()
def update_ema_actor(
    ema_actor: torch.nn.Module,
    actor: torch.nn.Module,
    decay: float,
) -> None:
    for ema_param, param in zip(ema_actor.parameters(), actor.parameters()):
        ema_param.data.mul_(float(decay)).add_(param.data, alpha=1.0-float(decay))
    for ema_buffer, buffer in zip(ema_actor.buffers(), actor.buffers()):
        ema_buffer.data.copy_(buffer.data)


def sample_actor_batch(
    offline_buffer: ReplayBuffer,
    success_buffer: ReplayBuffer,
    batch_size: int,
    success_ratio: float,
) -> List[torch.Tensor]:
    success_ratio = float(np.clip(success_ratio, 0.0, 1.0))
    if success_buffer._size <= 0:
        return offline_buffer.sample(batch_size)
    n_success = int(round(int(batch_size) * success_ratio))
    n_success = min(max(n_success, 1), int(batch_size)-1)
    n_offline = int(batch_size) - n_success
    offline_batch = offline_buffer.sample(n_offline)
    success_batch = success_buffer.sample(n_success)
    return [torch.cat((o, s), dim=0) for o, s in zip(offline_batch, success_batch)]


def load_offline_checkpoint(
    trainer: ImplicitQLearning,
    checkpoint: Dict[str, Any],
    restore_optimizer_state: bool,
    v_lr: float,
    q_lr: float,
    actor_lr: float,
) -> None:
    """恢复离线网络；默认同时恢复Adam状态，再覆盖为在线学习率。"""
    if restore_optimizer_state:
        trainer.load_state_dict(checkpoint)
    else:
        trainer.qf.load_state_dict(checkpoint["qf"])
        if "q_target" in checkpoint:
            trainer.q_target.load_state_dict(checkpoint["q_target"])
        else:
            trainer.q_target.load_state_dict(checkpoint["qf"])
        trainer.vf.load_state_dict(checkpoint["vf"])
        trainer.actor.load_state_dict(checkpoint["actor"])

    set_optimizer_lr(trainer.v_optimizer, v_lr)
    set_optimizer_lr(trainer.q_optimizer, q_lr)
    set_optimizer_lr(trainer.actor_optimizer, actor_lr)

    # 在线阶段保持常数学习率，避免离线scheduler状态或过早余弦衰减。
    trainer.reset_actor_scheduler(None)
    trainer.total_it = 0


def moving_average(values: List[float], window: int) -> float:
    if not values:
        return 0.0
    use = values[-max(1, int(window)):]
    return float(np.mean(use))


def save_curves(
    output_path: str,
    episode_index: List[int],
    success_raw: List[float],
    collision_raw: List[float],
    reward_raw: List[float],
    success_roll: List[float],
    collision_roll: List[float],
    reward_roll: List[float],
    noise_curve: List[float],
    online_ratio_curve: List[float],
    eval_episode: List[int],
    eval_success: List[float],
    eval_collision: List[float],
    eval_reward: List[float],
) -> None:
    np.savez(
        output_path,
        episode=np.asarray(episode_index, dtype=np.int64),
        success_raw=np.asarray(success_raw, dtype=np.float32),
        collision_raw=np.asarray(collision_raw, dtype=np.float32),
        reward_raw=np.asarray(reward_raw, dtype=np.float32),
        success_roll=np.asarray(success_roll, dtype=np.float32),
        collision_roll=np.asarray(collision_roll, dtype=np.float32),
        reward_roll=np.asarray(reward_roll, dtype=np.float32),
        noise=np.asarray(noise_curve, dtype=np.float32),
        online_ratio=np.asarray(online_ratio_curve, dtype=np.float32),
        eval_episode=np.asarray(eval_episode, dtype=np.int64),
        eval_success=np.asarray(eval_success, dtype=np.float32),
        eval_collision=np.asarray(eval_collision, dtype=np.float32),
        eval_reward=np.asarray(eval_reward, dtype=np.float32),
    )


@pyrallis.wrap()
def main(config: TrainConfig) -> None:
    if not torch.cuda.is_available() and config.device.startswith("cuda"):
        raise RuntimeError(
            "配置要求使用{}，但当前PyTorch未检测到CUDA".format(config.device)
        )
    if not 0.0 <= config.critic_online_ratio_start <= 1.0:
        raise ValueError("critic_online_ratio_start必须在[0,1]内")
    if not 0.0 <= config.critic_online_ratio_end <= 1.0:
        raise ValueError("critic_online_ratio_end必须在[0,1]内")
    if not 0.0 <= config.actor_online_ratio_start <= 1.0:
        raise ValueError("actor_online_ratio_start必须在[0,1]内")
    if not 0.0 <= config.actor_online_ratio_end <= 1.0:
        raise ValueError("actor_online_ratio_end必须在[0,1]内")
    if config.actor_update_interval <= 0:
        raise ValueError("actor_update_interval必须大于0")
    if config.updates_per_episode <= 0:
        raise ValueError("updates_per_episode必须大于0")

    state_dim = get_state_dim(config.human_num, config.human_state_dim)
    action_dim = 2
    max_action = 1.0

    os.makedirs(config.checkpoints_path, exist_ok=True)
    logger = setup_logger(config.checkpoints_path)
    logger.info("========== IQL ONLINE FINETUNE: SELECTIVE REPLAY V4 ==========")
    logger.info(
        "Env=%s | humans=%d | state_dim=%d | offline_beta=%.3f "
        "| online_beta=%.3f | iql_tau=%.3f",
        config.env,
        config.human_num,
        state_dim,
        config.beta,
        config.online_beta,
        config.iql_tau,
    )
    logger.info(
        "lr(V/Q/A)=%.2e/%.2e/%.2e | critic_ratio=%.3f->%.3f "
        "| actor_ratio=%.3f->%.3f | noise=%.3f->%.3f "
        "| min_online=%d | min_success=%d | updates=%d "
        "| actor_start=%d | actor_interval=%d | ema=%d | guard=%d",
        config.finetune_v_lr,
        config.finetune_q_lr,
        config.finetune_actor_lr,
        config.critic_online_ratio_start,
        config.critic_online_ratio_end,
        config.actor_online_ratio_start,
        config.actor_online_ratio_end,
        config.initial_noise_scale,
        config.final_noise_scale,
        config.min_online_steps,
        config.min_success_actor_steps,
        config.updates_per_episode,
        config.actor_update_start_steps,
        config.actor_update_interval,
        int(config.use_ema_actor),
        int(config.eval_guard_enabled),
    )

    # ---------------- Environment ----------------
    env_config = load_crowdnav_env_config(
        config.env_config_path,
        config.human_num,
    )
    env = gym.make(config.env)
    env.configure(env_config)
    set_seed(config.seed, env)
    robot = Robot(env_config, "robot")
    env.set_robot(robot)

    # ---------------- Dataset and buffers ----------------
    if not os.path.exists(config.offline_dataset_path):
        raise FileNotFoundError(
            "未找到7人离线数据集: {}".format(config.offline_dataset_path)
        )
    with h5py.File(config.offline_dataset_path, "r") as file:
        dataset = {key: np.asarray(value) for key, value in file.items()}
    validate_dataset(dataset, state_dim, config.offline_dataset_path)

    logger.info("Offline dataset: %s", config.offline_dataset_path)
    logger.info(
        "Dataset states=%s | actions=%s | reward=[%.4f, %.4f]",
        dataset["observations"].shape,
        dataset["actions"].shape,
        float(dataset["rewards"].min()),
        float(dataset["rewards"].max()),
    )

    dual_buffer = DualReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        offline_buffer_size=config.buffer_size,
        online_buffer_size=config.online_buffer_max_size,
        device=config.replay_buffer_device,
    )
    dual_buffer.load_offline_dataset(dataset)

    success_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.success_buffer_max_size,
        device=config.replay_buffer_device,
    )

    # ---------------- Networks and fresh online optimizers ----------------
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

    # 在线优化器从零开始，不继承离线Adam一阶/二阶矩。
    v_optimizer = torch.optim.Adam(
        v_network.parameters(), lr=config.finetune_v_lr
    )
    q_optimizer = torch.optim.Adam(
        q_network.parameters(), lr=config.finetune_q_lr
    )
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=config.finetune_actor_lr
    )

    estimated_actor_updates = max(
        1,
        int(config.online_episodes)
        * int(config.updates_per_episode)
        // int(config.actor_update_interval),
    )
    trainer = ImplicitQLearning(
        max_action=max_action,
        actor=actor,
        actor_optimizer=actor_optimizer,
        q_network=q_network,
        q_optimizer=q_optimizer,
        v_network=v_network,
        v_optimizer=v_optimizer,
        iql_tau=config.iql_tau,
        beta=config.online_beta,
        max_steps=None,
        discount=config.discount,
        tau=config.tau,
        device=config.device,
    )

    trainer.normalize_advantage = bool(config.online_normalize_advantage)
    trainer.advantage_clip = float(config.online_advantage_clip)
    trainer.exp_adv_max = float(config.online_exp_adv_max)
    trainer.value_grad_clip = float(config.value_grad_clip)
    trainer.q_grad_clip = float(config.q_grad_clip)
    trainer.actor_grad_clip = float(config.actor_grad_clip)

    # ---------------- Load offline trainer checkpoint ----------------
    offline_checkpoint = config.load_model or os.path.join(
        config.checkpoints_path, "iql_online_final_v4.pt"
    )
    if not os.path.exists(offline_checkpoint):
        raise FileNotFoundError(
            "未找到7人IQL离线模型: {}。请先运行train_7_iql.py。".format(
                offline_checkpoint
            )
        )
    logger.info("Loading offline trainer checkpoint: %s", offline_checkpoint)
    checkpoint = torch.load(offline_checkpoint, map_location=config.device)
    load_offline_checkpoint(
        trainer=trainer,
        checkpoint=checkpoint,
        restore_optimizer_state=bool(config.restore_offline_optimizer_state),
        v_lr=config.finetune_v_lr,
        q_lr=config.finetune_q_lr,
        actor_lr=config.finetune_actor_lr,
    )

    # 恢复检查点后再次明确设置在线IQL目标，防止旧checkpoint覆盖配置。
    trainer.beta = float(config.online_beta)
    trainer.iql_tau = float(config.iql_tau)
    trainer.normalize_advantage = bool(config.online_normalize_advantage)
    trainer.advantage_clip = float(config.online_advantage_clip)
    trainer.exp_adv_max = float(config.online_exp_adv_max)
    trainer.value_grad_clip = float(config.value_grad_clip)
    trainer.q_grad_clip = float(config.q_grad_clip)
    trainer.actor_grad_clip = float(config.actor_grad_clip)

    ema_actor = copy.deepcopy(trainer.actor).to(config.device)
    ema_actor.requires_grad_(False)
    ema_actor.eval()
    robot.set_policy(ema_actor if config.use_ema_actor else trainer.actor)
    logger.info(
        "Offline checkpoint loaded | optimizer_state=%s | beta=%.3f "
        "| normalize_adv=%s | exp_adv_max=%.1f | ema_decay=%.5f",
        str(bool(config.restore_offline_optimizer_state)),
        trainer.beta,
        str(trainer.normalize_advantage),
        trainer.exp_adv_max,
        config.ema_decay,
    )

    # ---------------- Initial evaluation ----------------
    logger.info("========== BEFORE ONLINE FINETUNE ==========")
    initial_metrics = test_actor(
        env,
        ema_actor if config.use_ema_actor else trainer.actor,
        device=config.device,
        n_episodes=config.online_eval_episodes,
        seed=config.seed,
        logger=logger,
        human_num=config.human_num,
        human_state_dim=config.human_state_dim,
        env_config_path=config.env_config_path,
    )

    best_success = float(initial_metrics["success_rate"])
    best_reward = float(initial_metrics["avg_reward"])
    best_episode = 0
    best_iql_path = os.path.join(config.checkpoints_path, "iql_online_best_v4.pt")
    best_actor_path = os.path.join(config.checkpoints_path, "actor_online_best_v4.pt")
    if config.save_best_online:
        torch.save(trainer.state_dict(), best_iql_path)
        torch.save(ema_actor.state_dict(), best_actor_path)

    # Curves
    episode_index: List[int] = []
    success_raw: List[float] = []
    collision_raw: List[float] = []
    reward_raw: List[float] = []
    success_roll: List[float] = []
    collision_roll: List[float] = []
    reward_roll: List[float] = []
    noise_curve: List[float] = []
    online_ratio_curve: List[float] = []
    eval_episode: List[int] = [0]
    eval_success: List[float] = [best_success]
    eval_collision: List[float] = [float(initial_metrics["collision_rate"])]
    eval_reward: List[float] = [best_reward]
    curve_path = os.path.join(
        config.checkpoints_path, "online_curve_iql_selective_v4.npz"
    )

    # ---------------- Online loop ----------------
    logger.info("========== START ONLINE FINETUNE ==========")
    total_online_steps = 0
    total_update_steps = 0
    total_actor_updates = 0
    last_log_dict: Dict[str, float] = {}

    # 完整trainer + EMA策略评估保护。
    best_actor_state = clone_state_dict_cpu(ema_actor)
    eval_drop_count = 0
    actor_freeze_until_episode = 0
    guard_restore_count = 0
    consolidation_active = False

    for episode in range(int(config.online_episodes)):
        noise_scale = cosine_schedule(
            config.initial_noise_scale,
            config.final_noise_scale,
            episode,
            config.noise_decay_episodes,
        )
        critic_online_ratio = linear_schedule(
            config.critic_online_ratio_start,
            config.critic_online_ratio_end,
            episode,
            config.critic_online_ratio_ramp_episodes,
        )
        actor_online_ratio = linear_schedule(
            config.actor_online_ratio_start,
            config.actor_online_ratio_end,
            episode,
            config.actor_online_ratio_ramp_episodes,
        )

        consolidation_active = best_success >= float(config.consolidation_success_threshold)
        if consolidation_active:
            noise_scale = min(noise_scale, float(config.consolidation_noise_cap))
            critic_online_ratio = min(critic_online_ratio, float(config.consolidation_critic_ratio_cap))
            actor_online_ratio = float(config.consolidation_actor_ratio)
            if get_optimizer_lr(trainer.actor_optimizer) > float(config.consolidation_actor_lr):
                set_optimizer_lr(trainer.actor_optimizer, float(config.consolidation_actor_lr))

        guard_active = (episode + 1) <= actor_freeze_until_episode
        if guard_active:
            noise_scale = min(noise_scale, float(config.eval_guard_noise_cap))
            critic_online_ratio = min(critic_online_ratio, float(config.eval_guard_critic_ratio_cap))

        observation = env.reset(phase="train")
        done = False
        episode_reward = 0.0
        episode_steps = 0
        info = None
        episode_transitions: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = []
        gamma = 1.0
        while not done:
            joint_state = JointState(robot.get_full_state(), observation)
            state = to_np(
                transform(joint_state, config.device)
                .reshape(1, -1)
                .squeeze(0)
            )
            validate_state_vector(state, state_dim, "在线采样当前")

            rollout_actor = (
                ema_actor
                if config.use_ema_actor and total_online_steps >= int(config.ema_rollout_start_steps)
                else trainer.actor
            )
            base_action = rollout_actor.act(
                state,
                device=config.device,
                deterministic=True,
            )
            noise = np.random.normal(
                loc=0.0,
                scale=noise_scale,
                size=action_dim,
            )
            action = np.clip(base_action + noise, -max_action, max_action)

            next_observation, reward, done, info = env.step(
                ActionXY(float(action[0]), float(action[1]))
            )
            next_joint_state = JointState(
                robot.get_full_state(), next_observation
            )
            next_state = to_np(
                transform(next_joint_state, config.device)
                .reshape(1, -1)
                .squeeze(0)
            )
            validate_state_vector(next_state, state_dim, "在线采样下一")

            episode_transitions.append((
                np.asarray(state, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                float(done),
            ))
            observation = next_observation
            rewards = reward * gamma
            gamma = gamma * 0.99
            episode_reward += float(rewards)
            episode_steps += 1
            total_online_steps += 1

        success = float(isinstance(info, ReachGoal))
        collision = float(isinstance(info, Collision))
        timeout = float(isinstance(info, Timeout))
        nav_time = float(env.global_time) if success > 0.0 else 0.0

        for transition in episode_transitions:
            dual_buffer.add_online_transition(*transition)
        accepted_success_episode = (
            success > 0.0 and episode_reward >= float(config.success_actor_min_return)
        )
        if accepted_success_episode:
            for transition in episode_transitions:
                success_buffer.add_transition(*transition)

        ready = (
            dual_buffer.online_buffer._size >= int(config.min_online_steps)
            and total_online_steps >= int(config.min_online_steps)
        )
        actor_updated_this_episode = 0

        if ready:
            for _ in range(int(config.updates_per_episode)):
                total_update_steps += 1
                active_actor_interval = int(
                    config.consolidation_actor_update_interval
                    if consolidation_active else config.actor_update_interval
                )
                actor_ready = (
                    total_online_steps >= int(config.actor_update_start_steps)
                    and success_buffer._size >= int(config.min_success_actor_steps)
                    and total_update_steps % max(active_actor_interval, 1) == 0
                    and not guard_active
                )
                critic_batch = dual_buffer.sample(
                    config.batch_size,
                    online_ratio=critic_online_ratio,
                )
                critic_batch = [item.to(config.device) for item in critic_batch]
                actor_batch = None
                if actor_ready:
                    actor_batch = sample_actor_batch(
                        dual_buffer.offline_buffer,
                        success_buffer,
                        config.batch_size,
                        actor_online_ratio,
                    )
                    actor_batch = [item.to(config.device) for item in actor_batch]
                last_log_dict = trainer.train(
                    critic_batch,
                    update_actor=actor_ready,
                    actor_batch=actor_batch,
                )
                if actor_ready:
                    total_actor_updates += 1
                    actor_updated_this_episode = 1
                    if config.use_ema_actor:
                        update_ema_actor(ema_actor, trainer.actor, config.ema_decay)

        # Raw and rolling curves.
        episode_index.append(episode + 1)
        success_raw.append(success)
        collision_raw.append(collision)
        reward_raw.append(episode_reward)
        success_roll.append(
            moving_average(success_raw, config.curve_window)
        )
        collision_roll.append(
            moving_average(collision_raw, config.curve_window)
        )
        reward_roll.append(
            moving_average(reward_raw, config.curve_window)
        )
        noise_curve.append(noise_scale)
        online_ratio_curve.append(critic_online_ratio)

        logger.info(
            "[ONLINE_IQL_V4] ep=%d/%d | env_steps=%d | ep_steps=%d "
            "| succ=%d | coll=%d | timeout=%d | succ_roll=%.3f "
            "| coll_roll=%.3f | nav_time=%.3f | reward=%.4f "
            "| reward_roll=%.4f | buffer=%d | success_buffer=%d | accepted=%d "
            "| ready=%d | noise=%.4f | critic_ratio=%.3f | actor_ratio=%.3f "
            "| actor_upd=%d | total_actor_upd=%d | V=%.5f | Q=%.5f "
            "| Actor=%.5f | w_mean=%.3f | w_max=%.3f | actor_lr=%.2e "
            "| consolidate=%d | guard=%d | restores=%d",
            episode + 1,
            config.online_episodes,
            total_online_steps,
            episode_steps,
            int(success),
            int(collision),
            int(timeout),
            success_roll[-1],
            collision_roll[-1],
            nav_time,
            episode_reward,
            reward_roll[-1],
            dual_buffer.online_buffer._size,
            success_buffer._size,
            int(accepted_success_episode),
            int(ready),
            noise_scale,
            critic_online_ratio,
            actor_online_ratio,
            actor_updated_this_episode,
            total_actor_updates,
            last_log_dict.get("value_loss", 0.0),
            last_log_dict.get("q_loss", 0.0),
            last_log_dict.get("actor_loss", 0.0),
            last_log_dict.get("exp_adv_mean", 0.0),
            last_log_dict.get("exp_adv_max", 0.0),
            get_optimizer_lr(trainer.actor_optimizer),
            int(consolidation_active),
            int(guard_active),
            guard_restore_count,
        )

        if (episode + 1) % int(config.online_eval_freq) == 0:
            logger.info("Evaluating at online episode %d", episode + 1)
            evaluation_actor = ema_actor if config.use_ema_actor else trainer.actor
            metrics = test_actor(
                env,
                evaluation_actor,
                device=config.device,
                n_episodes=config.online_eval_episodes,
                seed=config.seed,
                logger=logger,
                human_num=config.human_num,
                human_state_dim=config.human_state_dim,
                env_config_path=config.env_config_path,
            )
            current_success = float(metrics["success_rate"])
            current_reward = float(metrics["avg_reward"])
            eval_episode.append(episode + 1)
            eval_success.append(current_success)
            eval_collision.append(float(metrics["collision_rate"]))
            eval_reward.append(current_reward)

            is_better = (
                current_success > best_success + 1e-12
                or (
                    abs(current_success - best_success) <= 1e-12
                    and current_reward > best_reward
                )
            )
            if is_better:
                best_success = current_success
                best_reward = current_reward
                best_episode = episode + 1
                best_actor_state = clone_state_dict_cpu(evaluation_actor)
                eval_drop_count = 0
                if config.save_best_online:
                    torch.save(trainer.state_dict(), best_iql_path)
                    torch.save(evaluation_actor.state_dict(), best_actor_path)
                logger.info(
                    "[BEST] ep=%d | succ=%.3f | reward=%.4f",
                    best_episode,
                    best_success,
                    best_reward,
                )


            if (
                bool(config.eval_guard_enabled)
                and not is_better
                and current_success
                < best_success - float(config.eval_guard_drop_tolerance)
            ):
                eval_drop_count += 1
            else:
                eval_drop_count = 0

            if (
                bool(config.eval_guard_enabled)
                and eval_drop_count >= int(config.eval_guard_patience)
            ):
                old_actor_lr = get_optimizer_lr(trainer.actor_optimizer)
                new_actor_lr = max(
                    old_actor_lr * float(config.eval_guard_lr_decay),
                    float(config.eval_guard_min_actor_lr),
                )
                best_trainer_state = torch.load(best_iql_path, map_location=config.device)
                trainer.load_state_dict(best_trainer_state)
                trainer.reset_actor_scheduler(None)
                trainer.beta = float(config.online_beta)
                trainer.iql_tau = float(config.iql_tau)
                trainer.normalize_advantage = bool(config.online_normalize_advantage)
                trainer.advantage_clip = float(config.online_advantage_clip)
                trainer.exp_adv_max = float(config.online_exp_adv_max)
                trainer.value_grad_clip = float(config.value_grad_clip)
                trainer.q_grad_clip = float(config.q_grad_clip)
                trainer.actor_grad_clip = float(config.actor_grad_clip)
                set_optimizer_lr(trainer.actor_optimizer, new_actor_lr)
                set_optimizer_lr(trainer.q_optimizer, float(config.finetune_q_lr))
                set_optimizer_lr(trainer.v_optimizer, float(config.finetune_v_lr))
                ema_actor.load_state_dict(torch.load(best_actor_path, map_location=config.device))
                actor_freeze_until_episode = episode + 1 + int(config.eval_guard_freeze_episodes)
                guard_restore_count += 1
                eval_drop_count = 0
                robot.set_policy(ema_actor if config.use_ema_actor else trainer.actor)
                logger.warning(
                    "[FULL_GUARD] ep=%d | current_succ=%.3f | best_succ=%.3f "
                    "| restore_trainer=1 | actor_lr=%.2e->%.2e | freeze_until=%d",
                    episode + 1,
                    current_success,
                    best_success,
                    old_actor_lr,
                    new_actor_lr,
                    actor_freeze_until_episode,
                )

            checkpoint_path = os.path.join(
                config.checkpoints_path,
                "iql_online_{:06d}.pt".format(episode + 1),
            )
            torch.save(trainer.state_dict(), checkpoint_path)

        if (episode + 1) % int(config.curve_save_freq) == 0:
            save_curves(
                curve_path,
                episode_index,
                success_raw,
                collision_raw,
                reward_raw,
                success_roll,
                collision_roll,
                reward_roll,
                noise_curve,
                online_ratio_curve,
                eval_episode,
                eval_success,
                eval_collision,
                eval_reward,
            )

    # ---------------- Final save ----------------
    save_curves(
        curve_path,
        episode_index,
        success_raw,
        collision_raw,
        reward_raw,
        success_roll,
        collision_roll,
        reward_roll,
        noise_curve,
        online_ratio_curve,
        eval_episode,
        eval_success,
        eval_collision,
        eval_reward,
    )

    last_iql_path = os.path.join(config.checkpoints_path, "iql_online_last_v4.pt")
    last_actor_path = os.path.join(
        config.checkpoints_path, "actor_online_last_v4.pt"
    )
    torch.save(trainer.state_dict(), last_iql_path)
    torch.save(
        ema_actor.state_dict() if config.use_ema_actor else trainer.actor.state_dict(),
        last_actor_path,
    )

    # final默认指向最佳评估模型，而不是最后一步模型。
    final_iql_path = os.path.join(
        config.checkpoints_path, "iql_online_final_v4.pt"
    )
    final_actor_path = os.path.join(
        config.checkpoints_path, "actor_online_final_v4.pt"
    )
    if config.save_best_online and os.path.exists(best_iql_path):
        best_state = torch.load(best_iql_path, map_location="cpu")
        best_actor_state = torch.load(best_actor_path, map_location="cpu")
        torch.save(best_state, final_iql_path)
        torch.save(best_actor_state, final_actor_path)
    else:
        torch.save(trainer.state_dict(), final_iql_path)
        torch.save(trainer.actor.state_dict(), final_actor_path)

    deployment_actor = copy.deepcopy(trainer.actor).to(config.device)
    deployment_actor.load_state_dict(torch.load(final_actor_path, map_location=config.device))
    deployment_actor.eval()
    logger.info("========== FINAL BEST POLICY TEST ==========")
    final_metrics = test_actor(
        env,
        deployment_actor,
        device=config.device,
        n_episodes=config.final_test_episodes,
        seed=config.seed,
        logger=logger,
        human_num=config.human_num,
        human_state_dim=config.human_state_dim,
        env_config_path=config.env_config_path,
    )

    logger.info("========== ONLINE FINETUNE DONE ==========")
    logger.info(
        "Best evaluation: ep=%d | succ=%.3f | reward=%.4f",
        best_episode,
        best_success,
        best_reward,
    )
    logger.info("Last checkpoint: %s", last_iql_path)
    logger.info("Final(best) checkpoint: %s", final_iql_path)
    logger.info("Curve data: %s", curve_path)
    logger.info("Full guard restore count: %d", guard_restore_count)
    logger.info(
        "Final 500-episode test: succ=%.3f | coll=%.3f | timeout=%.3f | reward=%.4f",
        float(final_metrics["success_rate"]),
        float(final_metrics["collision_rate"]),
        float(final_metrics["timeout_rate"]),
        float(final_metrics["avg_reward"]),
    )


if __name__ == "__main__":
    main()
