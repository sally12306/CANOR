import hashlib
import logging
import os
import sys
from typing import Dict

import gym
import h5py
import numpy as np
import pyrallis
import torch

from crowd_nav.CANOR import (
    TrainConfig,
    DualReplayBuffer,
    TwinQ,
    ValueFunction,
    GaussianPolicy,
    DeterministicPolicy,
    ImplicitQLearning,
    set_seed,
    test_actor,
    online_finetune_weighted_loss,
)
from crowd_nav.policy.rnd_state_action import RND


def setup_logger(log_path: str) -> logging.Logger:
    """Write the same records to stdout and to an append-only log file."""
    logger = logging.getLogger("online_finetune_plateau_escape")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def freeze_rnd_estimator(rnd: torch.nn.Module) -> int:
    """
    Freeze the action-conditioned RND estimator for the entire online stage.

    The randomized target is already frozen by the RND implementation. Freezing
    all RND parameters here additionally freezes the predictor network and avoids
    relying on implementation-specific predictor attribute names.
    """
    rnd.eval()
    frozen_parameters = 0
    for parameter in rnd.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
        frozen_parameters += parameter.numel()

    if hasattr(rnd, "optimizer"):
        rnd.optimizer.zero_grad(set_to_none=True)
    return frozen_parameters


def set_fixed_offline_novelty_statistics(
        rnd,
        dataset: Dict[str, np.ndarray],
        logger: logging.Logger,
) -> None:
    """
    Use the offline dataset to define the fixed normalization reference
    (mu_e, sigma_e) used by Eq. (7) during online fine-tuning.
    """
    logger.info("Computing fixed offline RND prediction-error statistics...")
    try:
        errors = rnd.compute_dataset_novelty(
            dataset["observations"],
            dataset["actions"],
            batch_size=1024,
        )
    except TypeError:
        errors = rnd.compute_dataset_novelty(
            dataset["observations"],
            batch_size=1024,
        )

    if torch.is_tensor(errors):
        errors_np = errors.detach().cpu().numpy().astype(np.float64)
    else:
        errors_np = np.asarray(errors, dtype=np.float64)
    errors_np = errors_np.reshape(-1)

    error_mean = float(np.mean(errors_np))
    error_var = float(np.var(errors_np))
    error_std = float(np.sqrt(error_var))

    if not hasattr(rnd, "rms"):
        raise AttributeError(
            "RND has no rms attribute. Eq. (7) requires fixed mean/variance "
            "statistics for novelty normalization."
        )

    # RunningMeanStd implementations generally expose mutable mean, var, count.
    current_mean = getattr(rnd.rms, "mean", 0.0)
    current_var = getattr(rnd.rms, "var", 1.0)
    if torch.is_tensor(current_mean):
        rnd.rms.mean = torch.as_tensor(
            error_mean,
            device=current_mean.device,
            dtype=current_mean.dtype,
        )
    else:
        rnd.rms.mean = np.asarray(error_mean, dtype=np.float64)

    if torch.is_tensor(current_var):
        rnd.rms.var = torch.as_tensor(
            max(error_var, 1e-12),
            device=current_var.device,
            dtype=current_var.dtype,
        )
    else:
        rnd.rms.var = np.asarray(max(error_var, 1e-12), dtype=np.float64)

    if hasattr(rnd.rms, "count"):
        current_count = getattr(rnd.rms, "count")
        count_value = float(len(errors_np))
        if torch.is_tensor(current_count):
            rnd.rms.count = torch.as_tensor(
                count_value,
                device=current_count.device,
                dtype=current_count.dtype,
            )
        else:
            rnd.rms.count = count_value

    logger.info(
        "Fixed offline RND statistics: n=%d, mean=%.8f, std=%.8f, "
        "min=%.8f, max=%.8f",
        len(errors_np),
        error_mean,
        error_std,
        float(np.min(errors_np)),
        float(np.max(errors_np)),
    )


@pyrallis.wrap()
def main(config: TrainConfig):
    """
    Run CANOR online fine-tuning with exactly one update per episode.

    The online learner uses a frozen offline-RND safety anchor, trusted-only RND
    adaptation, positive-advantage online actor regression, failed-trajectory
    actor masking, and an offline-policy trust region.
    """
    if config.checkpoints_path is None:
        raise ValueError("config.checkpoints_path must not be None")

    # TrainConfig defaults define the stable run. Keeping the parsed values here
    # makes command-line and YAML hyperparameter ablations actually effective.
    if int(config.online_steps_per_update) != 1:
        raise ValueError("This fair-comparison runner requires one update per episode")

    run_name = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(config.online_run_name)
    ).strip("_")
    if not run_name:
        raise ValueError("online_run_name must contain at least one safe character")

    os.makedirs(config.checkpoints_path, exist_ok=True)
    log_path = os.path.join(
        config.checkpoints_path,
        f"train_eval_{run_name}.log",
    )
    logger = setup_logger(log_path)

    logger.info(
        "========== ONLINE FINETUNE: ONE UPDATE PER EPISODE + TAU=%.2f + RND_%s ==========",
        config.iql_tau,
        "FROZEN" if (config.freeze_rnd_online or not config.update_rnd_online) else "DELAYED_CONFLICT_ADAPTIVE",
    )
    script_path = os.path.abspath(__file__)
    with open(script_path, "rb") as script_file:
        script_sha256 = hashlib.sha256(script_file.read()).hexdigest()
    logger.info("Running script: %s", script_path)
    logger.info("Script SHA256: %s", script_sha256)
    logger.info("Fresh log file: %s", os.path.abspath(log_path))
    logger.info("Using device: %s", config.device)
    logger.info(
        "Settings: batch=%d, online_ratio=%.2f->%.2f, p_on=%.2f, p_off=%.2f, "
        "warmup=%d, updates_per_episode=1, sigma=%.3f->%.3f->%.3f, "
        "q/v/actor_lr=%.1e/%.1e/%.1e, actor_lr_min=%.1e, "
        "actor_lr_decay=%d->%d, actor_mix=%.2f->%.2f, "
        "failure_fraction=%.2f->%.2f, collision_backprop_steps=%d, "
        "iql_tau=%.2f, tau_floor=%.2f, beta=%.2f, target_tau=%.3f, "
        "online_w_floor=%.2f, adv_sign_preserving=%d, "
        "RND_update=%d, RND_start_ep=%d, RND_interval=%d, "
        "RND_anchor=%d, anchor_relax_cap=%.2f, RND_success_only=%d, "
        "online_positive_adv=%d, keep_frac=%.2f, failure_recovery=%.3f, "
        "actor_anchor=%.3f, actor_drift_cap=%.3f, checkpoint_step=%d",
        config.batch_size,
        config.online_sampling_ratio_start,
        config.online_sampling_ratio_final,
        config.bootstrap_p_on,
        config.bootstrap_p_off,
        config.min_online_size_for_updates,
        config.exploration_std_start,
        config.exploration_std_mid,
        config.exploration_std_final,
        config.online_qf_lr,
        config.online_vf_lr,
        config.online_actor_lr,
        config.online_actor_lr_min,
        config.actor_lr_decay_start_episode,
        config.actor_lr_decay_end_episode,
        config.actor_online_loss_fraction_start,
        config.actor_online_loss_fraction_final,
        config.collision_sampling_fraction_start,
        config.collision_sampling_fraction_final,
        config.collision_backprop_steps,
        config.iql_tau,
        config.expectile_floor,
        config.online_beta,
        config.online_target_tau,
        config.online_novelty_weight_floor,
        int(bool(config.normalize_policy_advantage)),
        int(bool(config.update_rnd_online) and not bool(config.freeze_rnd_online)),
        config.novelty_warmup_episodes,
        config.novelty_update_interval,
        int(bool(config.use_frozen_rnd_anchor)),
        config.novelty_anchor_relaxation_cap,
        int(bool(config.rnd_adapt_success_only)),
        int(bool(config.online_positive_advantage_only)),
        config.online_advantage_keep_fraction,
        config.actor_failure_recovery_coef,
        config.actor_anchor_coef,
        config.actor_max_relative_drift,
        int(config.expected_offline_checkpoint_step),
    )

    env = gym.make(config.env)
    set_seed(config.seed, env)

    dataset_path = os.path.abspath(config.dataset_path)
    checkpoint_step = int(config.expected_offline_checkpoint_step)
    offline_iql_path = os.path.join(
        config.checkpoints_path,
        f"iql_checkpoint_{checkpoint_step}.pt",
    )
    offline_rnd_path = os.path.join(
        config.checkpoints_path,
        f"rnd_checkpoint_{checkpoint_step}.pt",
    )

    logger.info("Loading offline dataset from: %s", dataset_path)
    with h5py.File(dataset_path, "r") as dataset_file:
        dataset = {
            key: np.asarray(value)
            for key, value in dataset_file.items()
        }

    logger.info(
        "Offline dataset loaded: observations=%s, actions=%s, rewards=[%.3f, %.3f]",
        dataset["observations"].shape,
        dataset["actions"].shape,
        float(dataset["rewards"].min()),
        float(dataset["rewards"].max()),
    )

    state_dim = int(dataset["observations"].shape[1])
    action_dim = int(dataset["actions"].shape[1])
    max_action = 1.0

    # ---------------- RND ----------------
    rnd = RND(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config.rnd_hidden_dim,
        output_dim=config.rnd_output_dim,
        lr=config.rnd_lr,
        device=config.device,
    )

    logger.info("Loading fixed offline RND checkpoint: %s", offline_rnd_path)
    rnd_checkpoint = torch.load(
        offline_rnd_path,
        map_location=config.device,
    )
    rnd.load_state_dict(rnd_checkpoint["rnd_state"])

    # Establish fixed offline normalization statistics before freezing.
    set_fixed_offline_novelty_statistics(rnd, dataset, logger)
    frozen_count = freeze_rnd_estimator(rnd)
    logger.info(
        "RND starts frozen (delayed conflict adaptation enabled=%d): "
        "%d frozen parameters, trainable=%d",
        int(bool(config.update_rnd_online) and not bool(config.freeze_rnd_online)),
        frozen_count,
        sum(parameter.numel() for parameter in rnd.parameters() if parameter.requires_grad),
    )

    # ---------------- Dual replay ----------------
    dual_replay_buffer = DualReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        offline_buffer_size=max(
            config.buffer_size,
            len(dataset["observations"]),
        ),
        online_buffer_size=config.online_buffer_max_size,
        device=config.device,
    )

    offline_placeholder_weights = np.ones(
        len(dataset["observations"]),
        dtype=np.float32,
    )
    dual_replay_buffer.load_offline_dataset(
        dataset,
        weights=offline_placeholder_weights,
    )
    logger.info(
        "Dual replay initialized: offline=%d, online_capacity=%d",
        dual_replay_buffer.offline_buffer._size,
        config.online_buffer_max_size,
    )

    # ---------------- IQL ----------------
    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)

    if config.iql_deterministic:
        actor = DeterministicPolicy(
            state_dim,
            action_dim,
            max_action,
            dropout=config.actor_dropout,
        )
    else:
        actor = GaussianPolicy(
            state_dim,
            action_dim,
            max_action,
            dropout=config.actor_dropout,
        )
    actor = actor.to(config.device)

    trainer = ImplicitQLearning(
        q_network=q_network,
        v_network=v_network,
        actor=actor,
        v_optimizer=torch.optim.Adam(
            v_network.parameters(),
            lr=config.online_vf_lr,
        ),
        q_optimizer=torch.optim.Adam(
            q_network.parameters(),
            lr=config.online_qf_lr,
        ),
        actor_optimizer=torch.optim.Adam(
            actor.parameters(),
            lr=config.online_actor_lr,
        ),
        tau=config.online_target_tau,
        discount=config.discount,
        beta=config.online_beta,
        iql_tau=config.iql_tau,
        max_steps=None,
        device=config.device,
        max_action=max_action,
        tau_w_min=0.0,
        policy_weight_min=0.0,
        exp_adv_max=config.exp_adv_max,
        actor_grad_clip=config.actor_grad_clip,
        q_grad_clip=config.q_grad_clip,
        v_grad_clip=config.v_grad_clip,
        normalize_actor_weights=config.normalize_actor_weights,
        policy_use_post_update_advantage=config.policy_use_post_update_advantage,
        policy_use_target_advantage=config.policy_use_target_advantage,
        policy_use_mean_mse=config.policy_use_mean_mse,
        cap_negative_terminal_advantage=config.cap_negative_terminal_advantage,
        q_huber_delta=config.q_huber_delta,
        safe_expectile_interpolation=config.safe_expectile_interpolation,
        expectile_floor=config.expectile_floor,
        normalize_policy_advantage=config.normalize_policy_advantage,
        advantage_scale_min=config.advantage_scale_min,
        actor_novelty_power=config.actor_novelty_power,
        actor_online_loss_fraction=config.actor_online_loss_fraction,
        balance_actor_groups=config.balance_actor_groups,
        negative_terminal_actor_weight=config.negative_terminal_actor_weight,
        online_positive_advantage_only=config.online_positive_advantage_only,
        online_advantage_margin=config.online_advantage_margin,
        online_advantage_keep_fraction=config.online_advantage_keep_fraction,
        online_advantage_threshold_floor=config.online_advantage_threshold_floor,
        online_advantage_min_count=config.online_advantage_min_count,
        actor_failure_recovery_coef=config.actor_failure_recovery_coef,
        actor_failure_recovery_radius=config.actor_failure_recovery_radius,
        actor_failure_recovery_min_q_gain=config.actor_failure_recovery_min_q_gain,
        actor_failure_recovery_min_novelty_weight=(
            config.actor_failure_recovery_min_novelty_weight
        ),
        actor_anchor_coef=config.actor_anchor_coef,
        actor_max_relative_drift=config.actor_max_relative_drift,
    )

    # Keep a deliberately fixed non-best offline initialization.
    # Preserve offline Adam moments, but disable the exhausted cosine scheduler.
    expected_iql_name = f"iql_checkpoint_{int(config.expected_offline_checkpoint_step)}.pt"
    if os.path.basename(offline_iql_path) != expected_iql_name:
        raise RuntimeError(
            f"Refusing to train from unexpected checkpoint: {offline_iql_path}. "
            f"Expected exactly {expected_iql_name}."
        )
    if not os.path.isfile(offline_iql_path):
        raise FileNotFoundError(offline_iql_path)

    logger.info("Loading fixed offline IQL checkpoint: %s", os.path.abspath(offline_iql_path))
    logger.info("Offline IQL checkpoint size: %d bytes", os.path.getsize(offline_iql_path))
    iql_checkpoint = torch.load(
        offline_iql_path,
        map_location=config.device,
    )
    checkpoint_total_it = int(iql_checkpoint.get("total_it", -1))
    logger.info("Checkpoint metadata total_it=%d", checkpoint_total_it)
    checkpoint_matches_offline = checkpoint_total_it in {
        int(config.expected_offline_checkpoint_step),
        int(config.expected_offline_checkpoint_step) + 1,
    }
    if not checkpoint_matches_offline and not bool(config.allow_online_checkpoint_resume):
        raise RuntimeError(
            "Checkpoint metadata does not match the requested fixed offline model: "
            f"total_it={checkpoint_total_it}. Check the checkpoint directory."
        )
    if not checkpoint_matches_offline:
        logger.warning(
            "Explicit online-checkpoint resume enabled: total_it=%d (fixed offline "
            "reference is %d/%d). Replay contents and optimizer state are not "
            "restored unless separately configured.",
            checkpoint_total_it,
            int(config.expected_offline_checkpoint_step),
            int(config.expected_offline_checkpoint_step) + 1,
        )
    # The checkpoint stores state_dicts, so the exact architecture must be
    # instantiated once. strict=True inside load_state_dict verifies every key
    # and shape. These loaded objects are reused; no network is recreated later.
    trainer.load_state_dict(
        iql_checkpoint,
        load_optimizers=bool(config.preserve_offline_optimizer_state),
        load_scheduler=False,
    )
    # The loaded networks are reused exactly; optimizer-state handling does not
    # recreate or alter any actor/Q/V parameter tensor.
    # Verify every actor tensor is bitwise identical to the checkpoint before
    # any online interaction or gradient update.
    loaded_actor_state = trainer.actor.state_dict()
    for key, checkpoint_tensor in iql_checkpoint["actor"].items():
        if key not in loaded_actor_state:
            raise RuntimeError(f"Loaded actor is missing checkpoint key: {key}")
        if not torch.equal(
            loaded_actor_state[key].detach().cpu(),
            checkpoint_tensor.detach().cpu(),
        ):
            raise RuntimeError(f"Actor parameter mismatch immediately after load: {key}")

    if bool(config.freeze_actor_log_std_online) and hasattr(trainer.actor, "log_std"):
        trainer.actor.log_std.requires_grad_(False)
        trainer.actor.log_std.grad = None
        logger.info(
            "Frozen actor.log_std during online fine-tuning; fixed std=%s",
            np.exp(trainer.actor.log_std.detach().cpu().numpy()).tolist(),
        )

    actor_fp_before_optimizer = trainer.parameter_fingerprint(trainer.actor)
    if bool(config.preserve_offline_optimizer_state):
        trainer.prepare_online_optimizers(
            q_lr=config.online_qf_lr,
            v_lr=config.online_vf_lr,
            actor_lr=config.online_actor_lr,
        )
    else:
        trainer.reset_online_optimizers(
            q_lr=config.online_qf_lr,
            v_lr=config.online_vf_lr,
            actor_lr=config.online_actor_lr,
            actor_scheduler_steps=None,
        )
    actor_fp_after_optimizer = trainer.parameter_fingerprint(trainer.actor)
    if actor_fp_before_optimizer != actor_fp_after_optimizer:
        raise RuntimeError("Preparing online optimizers unexpectedly changed actor parameters")

    actor_fp = actor_fp_after_optimizer
    q_fp = trainer.parameter_fingerprint(trainer.qf)
    v_fp = trainer.parameter_fingerprint(trainer.vf)
    logger.info(
        "Strict checkpoint load passed. Loaded actor/Q/V objects are reused "
        "throughout online fine-tuning; no network is recreated in the loop."
    )
    logger.info(
        "Loaded parameter fingerprints: actor(numel=%.0f,sum=%.8f,abs=%.8f,sq=%.8f) "
        "Q(numel=%.0f,sum=%.8f) V(numel=%.0f,sum=%.8f)",
        actor_fp["numel"], actor_fp["sum"], actor_fp["abs_sum"], actor_fp["sq_sum"],
        q_fp["numel"], q_fp["sum"], v_fp["numel"], v_fp["sum"],
    )
    logger.info(
        "Offline optimizer moments preserved=%d; q/v/actor lr=%.1e/%.1e/%.1e; "
        "offline actor scheduler disabled.",
        int(bool(config.preserve_offline_optimizer_state)),
        config.online_qf_lr,
        config.online_vf_lr,
        config.online_actor_lr,
    )

    logger.info("========== BEFORE ONLINE FINETUNE ==========")
    before_metrics = test_actor(
        env,
        trainer.actor,
        device=config.device,
        n_episodes=config.online_eval_episodes,
        seed=config.online_eval_seed,
        logger=logger,
        test_case_start=0,
    )
    if before_metrics["success_rate"] < float(config.minimum_offline_success_rate):
        raise RuntimeError(
            "Pre-online deterministic success is below the required baseline: "
            f"{before_metrics['success_rate']:.3f} < "
            f"{config.minimum_offline_success_rate:.3f}. "
            "Online fine-tuning has not started. This normally means the wrong "
            "checkpoint, environment configuration, or test-case definition is being used."
        )

    logger.info("========== START ONLINE FINETUNE ==========")
    online_summary = online_finetune_weighted_loss(
        trainer=trainer,
        rnd=rnd,
        dual_buffer=dual_replay_buffer,
        config=config,
        logger=logger,
        initial_eval_metrics=before_metrics,
    )

    logger.info("========== AFTER ONLINE FINETUNE ==========")
    after_metrics = test_actor(
        env,
        trainer.actor,
        device=config.device,
        n_episodes=config.online_eval_episodes,
        seed=config.online_eval_seed,
        logger=logger,
        test_case_start=0,
    )

    logger.info(
        "[IMPROVEMENT] success %.3f -> %.3f (%+.3f) | "
        "collision %.3f -> %.3f (%+.3f) | reward %.4f -> %.4f (%+.4f)",
        before_metrics["success_rate"],
        after_metrics["success_rate"],
        after_metrics["success_rate"] - before_metrics["success_rate"],
        before_metrics["collision_rate"],
        after_metrics["collision_rate"],
        after_metrics["collision_rate"] - before_metrics["collision_rate"],
        before_metrics["avg_reward"],
        after_metrics["avg_reward"],
        after_metrics["avg_reward"] - before_metrics["avg_reward"],
    )
    logger.info("Online rolling summary: %s", online_summary)

    iql_online_path = os.path.join(
        config.checkpoints_path,
        f"iql_{run_name}.pt",
    )
    actor_online_path = os.path.join(
        config.checkpoints_path,
        f"actor_{run_name}.pt",
    )
    rnd_online_path = os.path.join(
        config.checkpoints_path,
        f"rnd_{run_name}.pt",
    )

    torch.save(trainer.state_dict(), iql_online_path)
    torch.save(trainer.actor.state_dict(), actor_online_path)
    torch.save(
        {
            "rnd_state": rnd.state_dict(),
            "frozen_online": bool(config.freeze_rnd_online),
            "update_rnd_online": bool(config.update_rnd_online),
            "novelty_warmup_episodes": int(config.novelty_warmup_episodes),
        },
        rnd_online_path,
    )

    logger.info("Saved online IQL: %s", iql_online_path)
    logger.info("Saved online actor: %s", actor_online_path)
    logger.info("Saved online RND: %s", rnd_online_path)
    logger.info("========== ONLINE FINETUNE DONE ==========")


if __name__ == "__main__":
    main()
