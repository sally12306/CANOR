import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class RunningMeanStd:
    """动态维护流式数据的均值和方差，用于归一化 state-action novelty。"""

    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class FiLM(nn.Module):
    """FiLM 条件化层：用 state 调制 action hidden feature。"""

    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.state_encoder = nn.Linear(state_dim, hidden_dim * 2)

    def forward(self, state, action_hidden):
        gamma_beta = self.state_encoder(state)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        return gamma * action_hidden + beta


class BilinearLayer(nn.Module):
    """双线性条件化层：融合 state 和 action，用于 predictor 的第一层。"""

    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        # Full bilinear tensor. For hidden_dim=256 this is large but follows the original rnd_flim.py design.
        self.bilinear_W = nn.Parameter(torch.empty(hidden_dim, hidden_dim, hidden_dim))
        self.bilinear_b = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.xavier_uniform_(self.bilinear_W)

    def forward(self, state, action):
        s_proj = self.state_proj(state)      # (B, H)
        a_proj = self.action_proj(action)    # (B, H)
        # For each sample b and output k: sum_i sum_j s[b,i] * W[i,j,k] * a[b,j]
        bilinear_term = torch.einsum("bi,ijk,bj->bk", s_proj, self.bilinear_W, a_proj)
        base_feat = s_proj + a_proj
        return base_feat + bilinear_term + self.bilinear_b


class RND(nn.Module):
    """
    Action-conditioned RND: RND(s, a).

    与原 rnd.py 的主要区别：
    1. target/prior network 输入 action，并用 FiLM 由 state 进行条件化；
    2. predictor network 输入 state-action pair，并用 bilinear layer 融合；
    3. 所有 novelty / raw error / pretrain / adaptive / conflict-aware 接口都改成 (state, action)。
    """

    def __init__(self, state_dim, action_dim, hidden_dim, output_dim, lr, device):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # ---------- Target / Prior Network: fixed ----------
        # action -> hidden, then state-conditioned FiLM, then output embedding
        self.target_action_net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.target_film = FiLM(state_dim, hidden_dim)
        self.target_final = nn.Linear(hidden_dim, output_dim)

        # ---------- Predictor Network: trainable ----------
        self.predictor_bilinear = BilinearLayer(state_dim, action_dim, hidden_dim)
        self.predictor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()
        self.to(device)

        # freeze target/prior network
        for module in [self.target_action_net, self.target_film, self.target_final]:
            for param in module.parameters():
                param.requires_grad = False

        # optimize predictor only
        predictor_params = list(self.predictor_bilinear.parameters()) + list(self.predictor.parameters())
        self.optimizer = torch.optim.Adam(predictor_params, lr=lr)

        # Running statistics for normalized state-action novelty z(s,a)
        self.rms = RunningMeanStd()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            return torch.FloatTensor(x).to(self.device)
        if isinstance(x, torch.Tensor):
            return x.to(self.device).float()
        return torch.FloatTensor(x).to(self.device)

    def _prepare_pair(self, states, actions):
        states = self._to_tensor(states)
        actions = self._to_tensor(actions)

        if states.dim() == 1:
            states = states.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        states = states.view(states.shape[0], -1)
        actions = actions.view(actions.shape[0], -1)

        if states.shape[0] != actions.shape[0]:
            raise ValueError(
                f"states and actions must have the same batch size, "
                f"got {states.shape[0]} and {actions.shape[0]}."
            )
        if states.shape[1] != self.state_dim:
            raise ValueError(f"Expected state_dim={self.state_dim}, got {states.shape[1]}.")
        if actions.shape[1] != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {actions.shape[1]}.")
        return states, actions

    def forward_target(self, states, actions):
        """Target/prior network forward for the fixed target embedding."""
        action_hidden = self.target_action_net(actions)
        modulated_hidden = self.target_film(states, action_hidden)
        return self.target_final(modulated_hidden)

    def forward_predictor(self, states, actions):
        """Predictor network forward for the trainable embedding."""
        fused_feat = self.predictor_bilinear(states, actions)
        return self.predictor(fused_feat)

    def _raw_error_tensor(self, states, actions):
        target_feat = self.forward_target(states, actions)
        pred_feat = self.forward_predictor(states, actions)
        return F.mse_loss(pred_feat, target_feat, reduction="none").mean(dim=1)

    def get_intrinsic_reward(self, state, action, update_stats=True):
        """
        计算归一化后的 state-action novelty z(s,a)。

        Args:
            state:  shape=(state_dim,) or (B, state_dim)
            action: shape=(action_dim,) or (B, action_dim)
            update_stats: 训练阶段 True，测试/评估阶段 False
        Returns:
            float or np.ndarray: clipped normalized novelty z(s,a)
        """
        with torch.no_grad():
            states, actions = self._prepare_pair(state, action)
            raw_error = self._raw_error_tensor(states, actions).detach().cpu().numpy()

        if update_stats:
            self.rms.update(raw_error)

        norm_error = (raw_error - self.rms.mean) / (np.sqrt(self.rms.var) + 1e-8)
        norm_error = np.clip(norm_error, -5, 5)
        return norm_error.item() if norm_error.size == 1 else norm_error

    def get_raw_error_batch(self, states, actions):
        """
        只计算 raw prediction error e(s,a)，不更新 RMS 统计量。
        用于训练时构造 novelty weight: e(s,a) -> z(s,a) -> w(s,a)。
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            states, actions = self._prepare_pair(states, actions)
            raw_error = self._raw_error_tensor(states, actions)
        if was_training:
            self.train()
        return raw_error

    def train_predictor(self, state_batch, action_batch):
        """
        离线预训练 predictor，使其拟合 offline social navigation dataset 中的 (s,a)。
        """
        states, actions = self._prepare_pair(state_batch, action_batch)
        target_feat = self.forward_target(states, actions)
        pred_feat = self.forward_predictor(states, actions)
        loss = F.mse_loss(pred_feat, target_feat)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_adaptive(self, online_states, online_actions, offline_states, offline_actions):
        """
        简单混合更新：同时使用 online 与 offline state-action pairs，防止 RND 忘记离线分布。
        """
        online_states, online_actions = self._prepare_pair(online_states, online_actions)
        offline_states, offline_actions = self._prepare_pair(offline_states, offline_actions)

        mixed_states = torch.cat([online_states, offline_states], dim=0)
        mixed_actions = torch.cat([online_actions, offline_actions], dim=0)

        target_feat = self.forward_target(mixed_states, mixed_actions)
        pred_feat = self.forward_predictor(mixed_states, mixed_actions)
        loss = F.mse_loss(pred_feat, target_feat)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def compute_dataset_novelty(self, observations, actions, batch_size=256):
        """
        分批计算整个数据集的 raw prediction error e(s,a)。

        Args:
            observations: np.ndarray or torch.Tensor, shape=(N, state_dim)
            actions:      np.ndarray or torch.Tensor, shape=(N, action_dim)
        Returns:
            torch.Tensor, shape=(N, 1)
        """
        was_training = self.training
        self.eval()

        obs_tensor = self._to_tensor(observations)
        act_tensor = self._to_tensor(actions)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        if act_tensor.dim() == 1:
            act_tensor = act_tensor.unsqueeze(0)
        if obs_tensor.shape[0] != act_tensor.shape[0]:
            raise ValueError("observations and actions must have the same number of samples.")

        novelties = []
        with torch.no_grad():
            for i in range(0, obs_tensor.shape[0], batch_size):
                batch_obs = obs_tensor[i:min(i + batch_size, obs_tensor.shape[0])]
                batch_act = act_tensor[i:min(i + batch_size, act_tensor.shape[0])]
                error = self._raw_error_tensor(batch_obs, batch_act)
                novelties.append(error.cpu())

        if was_training:
            self.train()
        return torch.cat(novelties, dim=0).unsqueeze(1)

    def train_conflict_aware(self, online_states, online_actions, offline_states, offline_actions):
        """Conflict-aware online adaptation of the action-conditioned predictor.

        The target/prior network and the fixed offline normalization statistics
        are never updated. The online gradient is projected only when it conflicts
        with the offline anchor gradient. The returned diagnostics are consumed by
        the CANOR online logger.
        """
        online_states, online_actions = self._prepare_pair(
            online_states, online_actions
        )
        offline_states, offline_actions = self._prepare_pair(
            offline_states, offline_actions
        )

        pred_on = self.forward_predictor(online_states, online_actions)
        with torch.no_grad():
            target_on = self.forward_target(online_states, online_actions)
        loss_on = F.mse_loss(pred_on, target_on)

        pred_off = self.forward_predictor(offline_states, offline_actions)
        with torch.no_grad():
            target_off = self.forward_target(offline_states, offline_actions)
        loss_off = F.mse_loss(pred_off, target_off)

        params = list(self.predictor_bilinear.parameters()) + list(
            self.predictor.parameters()
        )
        trainable_params = [p for p in params if p.requires_grad]
        if not trainable_params:
            raise RuntimeError(
                "RND predictor is frozen. Enable predictor parameters before "
                "calling train_conflict_aware()."
            )

        self.optimizer.zero_grad(set_to_none=True)
        grad_on = torch.autograd.grad(
            loss_on, trainable_params, retain_graph=True, allow_unused=False
        )
        grad_off = torch.autograd.grad(
            loss_off, trainable_params, retain_graph=False, allow_unused=False
        )

        g_on_flat = torch.cat([g.reshape(-1) for g in grad_on])
        g_off_flat = torch.cat([g.reshape(-1) for g in grad_off])
        dot_product = torch.dot(g_on_flat, g_off_flat)
        conflict = bool(dot_product.item() < 0.0)

        if conflict:
            denom = torch.dot(g_off_flat, g_off_flat).clamp(min=1e-8)
            projection_scalar = dot_product / denom
            projected_online = [
                g_on_i - projection_scalar * g_off_i
                for g_on_i, g_off_i in zip(grad_on, grad_off)
            ]
        else:
            projected_online = list(grad_on)

        with torch.no_grad():
            raw_errors = F.mse_loss(
                pred_on, target_on, reduction="none"
            ).mean(dim=1)
            raw_error_var = torch.var(raw_errors, unbiased=False)

            # The supplied run used Var[e] directly.  Its errors are around
            # 1e-5, so Var[e] is around 1e-10 and alpha rounds to 1.000 for every
            # update.  Use the dimensionless offline-normalized error variance
            # so Eq. (20) actually scales heterogeneous online batches.
            if bool(getattr(self, "use_normalized_error_variance", True)):
                mean = torch.as_tensor(
                    getattr(self.rms, "mean", 0.0),
                    device=raw_errors.device,
                    dtype=raw_errors.dtype,
                )
                var = torch.as_tensor(
                    getattr(self.rms, "var", 1.0),
                    device=raw_errors.device,
                    dtype=raw_errors.dtype,
                )
                normalized_errors = (
                    raw_errors - mean
                ) / (torch.sqrt(var.clamp(min=1e-12)) + 1e-8)
                trust_variance = torch.var(normalized_errors, unbiased=False)
            else:
                trust_variance = raw_error_var
            trust = 1.0 / (1.0 + float(trust_variance.item()))

            g_on_norm = torch.linalg.vector_norm(g_on_flat)
            g_off_norm = torch.linalg.vector_norm(g_off_flat)
            grad_cosine = dot_product / (
                g_on_norm * g_off_norm + 1e-12
            )

        final_grads = [
            trust * g_proj_i + g_off_i
            for g_proj_i, g_off_i in zip(projected_online, grad_off)
        ]
        for param, grad in zip(trainable_params, final_grads):
            param.grad = grad.detach().clone()

        grad_clip = float(getattr(self, "conflict_grad_clip", 0.5))
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params, max_norm=max(grad_clip, 1e-8)
        )
        self.optimizer.step()

        return {
            "loss_on": float(loss_on.item()),
            "loss_off": float(loss_off.item()),
            "trust": float(trust),
            "conflict": float(conflict),
            "grad_dot": float(dot_product.item()),
            "grad_cosine": float(grad_cosine.item()),
            "grad_on_norm": float(g_on_norm.item()),
            "grad_off_norm": float(g_off_norm.item()),
            "grad_norm": float(
                grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            ),
            "online_error_mean": float(raw_errors.mean().item()),
            "online_error_var": float(raw_error_var.item()),
            "normalized_error_var": float(trust_variance.item()),
        }

