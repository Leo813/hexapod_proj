"""Independent feedforward PPO mathematics and optimization, without an env loop."""

from typing import NamedTuple

import torch

from .advantages import adjust_timeout_rewards, generalized_advantage_estimate, normalize_advantages
from .config import PPOConfig
from .distributions import diagonal_gaussian, gaussian_log_prob
from .normalization import RunningMeanStd
from .rollout import TrainingBatch, flatten_time_env
from .schedules import adaptive_learning_rate, policy_kl


def clipped_policy_loss(new_log_probs, old_log_probs, advantages, clip):
    if new_log_probs.ndim != 1 or new_log_probs.shape != old_log_probs.shape or advantages.shape != new_log_probs.shape:
        raise ValueError("Log probabilities and advantages must have shape [batch]")
    # RL-Games exp(old_neglogp - new_neglogp), expressed as positive log densities.
    ratio = (new_log_probs - old_log_probs).exp()
    return torch.maximum(-advantages * ratio,
                         -advantages * ratio.clamp(1 - clip, 1 + clip)).mean()


def critic_loss(values, old_values, returns, clip, clip_value=True):
    if values.ndim != 2 or values.shape[-1] != 1 or values.shape != old_values.shape or values.shape != returns.shape:
        raise ValueError("Values and returns must have shape [batch, 1]")
    error = (values - returns).square()
    if clip_value:
        clipped = old_values + (values - old_values).clamp(-clip, clip)
        error = torch.maximum(error, (clipped - returns).square())
    return error.mean()


def action_bounds_loss(means):
    if means.ndim != 2:
        raise ValueError("Policy means must have shape [batch, action_dim]")
    return ((means - 1.1).clamp_min(0).square()
            + (means + 1.1).clamp_max(0).square()).sum(-1).mean()


class LossTerms(NamedTuple):
    total: torch.Tensor
    policy: torch.Tensor
    critic: torch.Tensor
    entropy: torch.Tensor
    bounds: torch.Tensor


def ppo_loss(new_log_probs, old_log_probs, advantages, values, old_values,
             returns, entropy, means, config):
    if (entropy.shape != new_log_probs.shape or means.shape[0] != new_log_probs.shape[0]
            or values.shape[0] != new_log_probs.shape[0]):
        raise ValueError("Entropy, means, and values must match the policy batch size")
    policy = clipped_policy_loss(new_log_probs, old_log_probs, advantages, config.e_clip)
    critic = critic_loss(values, old_values, returns, config.e_clip, config.clip_value)
    entropy = entropy.mean()
    bounds = action_bounds_loss(means)
    total = policy + 0.5 * config.critic_coef * critic - config.entropy_coef * entropy + config.bounds_loss_coef * bounds
    return LossTerms(total, policy, critic, entropy, bounds)


class PolicyStep(NamedTuple):
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    means: torch.Tensor
    stds: torch.Tensor


class PPO:
    """Update an existing Stage 1 ActorCritic on a single device.

    Move the model to its final device BEFORE construction. Sampling accepts
    environment-prepared observations and returns raw actions/reward-unit values.
    No simulation, action transforms, training launcher, or checkpoint format is
    imposed here. Normalizers, optimizer, and scaler expose native state_dicts.
    """

    def __init__(self, model, config=None):
        self.model = model
        self.config = config if config is not None else PPOConfig()
        self.device = next(model.parameters()).device
        c = self.config
        self.observation_normalizer = (RunningMeanStd(model.config.observation_dim,
            epsilon=c.normalization_epsilon, clip=c.normalization_clip).to(self.device)
            if c.normalize_input else None)
        self.value_normalizer = (RunningMeanStd(1, epsilon=c.normalization_epsilon,
            clip=c.normalization_clip).to(self.device) if c.normalize_value else None)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=c.learning_rate,
            eps=c.adam_epsilon, betas=c.adam_betas, weight_decay=c.weight_decay)
        self.amp_enabled = c.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

    @property
    def learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]

    def _rollout_observations(self, observations):
        self.model.eval()
        if self.observation_normalizer is not None:
            self.observation_normalizer.eval()
            observations = self.observation_normalizer(observations)
        if self.value_normalizer is not None:
            self.value_normalizer.eval()
        return observations

    def _denormalize_value(self, values):
        return (self.value_normalizer(values, unnorm=True)
                if self.value_normalizer is not None else values)

    @torch.no_grad()
    def sample(self, observations):
        output = self.model(self._rollout_observations(observations))
        actions = diagonal_gaussian(output.means, output.log_std).sample()
        return PolicyStep(actions, gaussian_log_prob(actions, output.means, output.log_std),
                          self._denormalize_value(output.values), output.means, output.log_std.exp())

    @torch.no_grad()
    def predict_values(self, observations):
        return self._denormalize_value(self.model.critic(self._rollout_observations(observations)))

    @torch.no_grad()
    def prepare_batch(self, rollout):
        c = self.config
        if (rollout.horizon_length, rollout.num_envs, rollout.observation_dim, rollout.action_dim) != (
                c.horizon_length, c.num_actors, self.model.config.observation_dim, self.model.config.action_dim):
            raise ValueError("Rollout dimensions must match PPO and network configurations")
        flat = rollout.flatten()  # also checks completion; owns data separate from reusable buffer
        if rollout.last_values.device != self.device:
            raise ValueError("Rollout and model must be on the same device")
        rewards = adjust_timeout_rewards(rollout.data["rewards"], rollout.data["values"],
            rollout.data["timeouts"], c.gamma, c.value_bootstrap, c.reward_scale)
        _, returns = generalized_advantage_estimate(rewards, rollout.data["values"],
            rollout.data["dones"], rollout.last_values, rollout.last_dones, c.gamma, c.tau)
        returns = flatten_time_env(returns)
        old_values = flat["values"]
        advantages = (returns - old_values).sum(-1)
        if self.value_normalizer is not None:
            # Two separate updates: values and returns use DIFFERENT moments.
            self.value_normalizer.train()
            old_values = self.value_normalizer(old_values)
            returns = self.value_normalizer(returns)
            self.value_normalizer.eval()
        if c.normalize_advantage:
            advantages = normalize_advantages(advantages, c.advantage_epsilon)
        return TrainingBatch(dict(observations=flat["observations"], actions=flat["actions"],
            old_log_probs=flat["log_probs"], old_values=old_values, returns=returns,
            advantages=advantages, means=flat["means"], stds=flat["stds"]))

    def update(self, rollout):
        return self.optimize(self.prepare_batch(rollout))

    def optimize(self, batch):
        """Return per-minibatch metrics; no KL early stopping or batch shuffling."""
        c = self.config
        if batch.size != c.batch_size or batch.data["observations"].device != self.device:
            raise ValueError("Training batch must match the configured size and model device")
        self.model.train()
        if self.observation_normalizer is not None:
            self.observation_normalizer.train()
        metrics = []
        for epoch in range(c.mini_epochs):
            for indices, data in batch.minibatches(c.minibatch_size):
                observations = data["observations"]
                if self.observation_normalizer is not None:
                    observations = self.observation_normalizer(observations)
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    output = self.model(observations)
                    log_probs = gaussian_log_prob(data["actions"], output.means, output.log_std)
                    entropy = diagonal_gaussian(output.means, output.log_std).entropy()
                    terms = ppo_loss(log_probs, data["old_log_probs"], data["advantages"],
                        output.values, data["old_values"], data["returns"], entropy, output.means, c)
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(terms.total).backward()
                grad_norm = None
                if c.truncate_grads:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_norm)
                lr_used = self.learning_rate
                self.scaler.step(self.optimizer)
                self.scaler.update()
                with torch.no_grad():
                    # Forward parameters PRECEDE the optimizer step, as in the reference.
                    means, stds = output.means.detach(), output.log_std.exp().detach()
                    kl = policy_kl(means, stds, data["means"], data["stds"])
                    batch.update_distribution(indices, means, stds)
                    if c.lr_schedule == "adaptive":
                        lr = adaptive_learning_rate(lr_used, kl.item(), c.kl_threshold,
                                                    c.min_learning_rate, c.max_learning_rate)
                        for group in self.optimizer.param_groups:
                            group["lr"] = lr
                metrics.append(dict(epoch=epoch, start=indices.start, learning_rate=lr_used,
                    next_learning_rate=self.learning_rate, kl=kl.item(),
                    grad_norm=None if grad_norm is None else grad_norm.item(),
                    **{name: value.detach().item() for name, value in zip(terms._fields, terms)}))
            # Input moments update in the FIRST mini-epoch only, then freeze.
            if self.observation_normalizer is not None:
                self.observation_normalizer.eval()
        return metrics
