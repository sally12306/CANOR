import os
import sys
import logging
import h5py
import numpy as np
import torch
import gym
import pyrallis
import configparser


from crowd_nav.train_awac import (
    TrainConfig,
    DualReplayBuffer,
    TwinQ,
    GaussianPolicy,
    DeterministicPolicy,
    AWAC,
    set_seed,
    test_actor,
    transform,
    to_np,
)


from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.info import ReachGoal, Collision, Timeout


def setup_logger(log_path: str) -> logging.Logger:

    logger = logging.getLogger("awac_online_finetune")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(message)s")

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)


    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


@pyrallis.wrap()
def main(config: TrainConfig):

    finetune_lr = 1e-5

    os.makedirs(config.checkpoints_path, exist_ok=True)
    logger = setup_logger(os.path.join(config.checkpoints_path, "train_eval_awac_online.log"))

    logger.info("========== AWAC ONLINE FINETUNE (REUSING train_awac.py) ==========")
    logger.info(f"Device: {config.device} | Beta: {config.beta} | LR: {finetune_lr}")


    env = gym.make(config.env)
    set_seed(config.seed, env)


    env_config_file = "crowd_nav\\configs\\env.config"
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env.configure(env_config)  # 配置环境参数

    state_dim = 91
    action_dim = 2
    max_action = 1.0


    q_network = TwinQ(state_dim, action_dim).to(config.device)


    if config.iql_deterministic:
        actor = DeterministicPolicy(state_dim, action_dim, max_action, dropout=config.actor_dropout)
    else:
        actor = GaussianPolicy(state_dim, action_dim, max_action, dropout=config.actor_dropout)
    actor = actor.to(config.device)

    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=finetune_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=finetune_lr)

    trainer = AWAC(
        max_action=max_action,
        actor=actor,
        actor_optimizer=actor_optimizer,
        q_network=q_network,
        q_optimizer=q_optimizer,
        beta=config.beta,
        discount=config.discount,
        tau=config.tau,
        device=config.device
    )

    offline_ckpt_path = os.path.join(config.checkpoints_path, "")

    logger.info(f"Loading offline checkpoint from: {offline_ckpt_path}")
    if os.path.exists(offline_ckpt_path):
        checkpoint = torch.load(offline_ckpt_path, map_location=config.device)
        trainer.load_state_dict(checkpoint)
        for param_group in trainer.q_optimizer.param_groups:
            param_group['lr'] = finetune_lr
        for param_group in trainer.actor_optimizer.param_groups:
            param_group['lr'] = finetune_lr

        logger.info(">>> Checkpoint loaded. Optimizer LR reset to %.1e", finetune_lr)
    else:
        raise FileNotFoundError(f"未找到离线模型文件: {offline_ckpt_path}")

    dual_buffer = DualReplayBuffer(
        state_dim, action_dim, config.buffer_size, config.online_buffer_max_size, config.device
    )

    hdf5_path = ""
    logger.info(f"Loading offline dataset: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as f:
        dataset = {k: np.array(v) for k, v in f.items()}

    dual_buffer.load_offline_dataset(dataset)
    logger.info(f"Offline Data Size: {dual_buffer.offline_buffer._size}")

    robot = Robot(env_config, 'robot')
    env.set_robot(robot)
    robot.set_policy(trainer.actor)

    logger.info("========== START ONLINE FINETUNE ==========")

    test_actor(env, trainer.actor, config.device, n_episodes=config.n_episodes, seed=config.seed, logger=logger)

    online_episodes = 50000
    # 安全阈值：建议攒够 batch_size * 2 的数据再开始混合训练
    min_safe_online = max(512, 1000)

    for ep in range(online_episodes):
        obs = env.reset(phase='train')
        done = False
        ep_reward = 0
        ep_steps = 0


        success = 0
        collision = 0
        timeout = 0
        episode_reward = 0.0
        nav_time = 0.0
        info = None

        if ep < online_episodes * 0.2:
            noise_scale = 0.2
        else:
            noise_scale = 0.05

        while not done:

            joint_state = JointState(robot.get_full_state(), obs)
            state_vec = to_np(transform(joint_state, config.device).view(1, -1).squeeze(0))


            raw_action = trainer.actor.act(state_vec, device=config.device)


            noise = np.random.normal(0, noise_scale, size=2)
            action_np = np.clip(raw_action + noise, -1, 1)

            vx, vy = action_np[0], action_np[1]

            next_obs, reward, done, info = env.step(ActionXY(vx, vy))
            ep_reward += reward
            ep_steps += 1

            next_joint_state = JointState(robot.get_full_state(), next_obs)
            next_state_vec = to_np(transform(next_joint_state, config.device).view(1, -1).squeeze(0))


            dual_buffer.add_online_transition(
                state_vec, action_np, reward, next_state_vec, float(done)
            )

            obs = next_obs


        # Episode 结束统计
        if isinstance(info, ReachGoal):
            success = 1
            nav_time = env.global_time
        elif isinstance(info, Collision):
            collision = 1
        elif isinstance(info, Timeout):
            timeout = 1
        episode_reward = ep_reward


  
        if dual_buffer.online_buffer._size >= min_safe_online:
            
            batch = dual_buffer.sample(config.batch_size, online_ratio=0.5)
            batch = [b.to(config.device) for b in batch]
            log_dict = trainer.train(batch)



        if logger is not None:
            logger.info(
                "[ONLINE_AWAC] ep=%d/%d | succ=%d | coll=%d | timeout=%d | nav_time=%.4f | reward=%.4f | buffer_size=%d",
                ep + 1,
                online_episodes,
                success,
                collision,
                timeout,
                nav_time,
                episode_reward,
                dual_buffer.online_buffer._size,
            )


        if (ep + 1) % 10 == 0:
            success = isinstance(info, ReachGoal)
            collision = isinstance(info, Collision)

            q_loss = log_dict.get('q_loss', 0.0) if 'log_dict' in locals() else 0.0
            actor_loss = log_dict.get('actor_loss', 0.0) if 'log_dict' in locals() else 0.0


        if (ep + 1) % config.eval_freq == 0:
            logger.info(f"Evaluating at episode {ep + 1}...")
            # 注意：test_actor 内部会创建新的 env，不影响主循环
            test_actor(env, trainer.actor, config.device, n_episodes=config.n_episodes, seed=config.seed, logger=logger)

            # save_path = os.path.join(config.checkpoints_path, f"awac_online_{ep + 1}.pt")
            # torch.save(trainer.state_dict(), save_path)

    # 最终保存
    final_path = os.path.join(config.checkpoints_path, "awac_online_final.pt")
    torch.save(trainer.state_dict(), final_path)
    logger.info(f"Finished. Model saved to {final_path}")


if __name__ == "__main__":
    main()
