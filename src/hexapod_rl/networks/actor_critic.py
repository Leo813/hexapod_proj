"""Independent feedforward actor-critic for the selected historical baseline.

Behavioral reference: RL-Games v1.6.0 A2CBuilder and
ModelA2CContinuousLogStd, attributed in docs/reference-implementation.md.
No upstream classes are imported or subclassed.
"""

from typing import NamedTuple, Optional

import torch
from torch import Tensor, nn
from torch.distributions import Independent

from ..config import ActorCriticConfig
from ..distributions import diagonal_gaussian, gaussian_log_prob


class ActorCriticOutput(NamedTuple):
    means: Tensor       # [batch, action_dim], unbounded
    log_std: Tensor     # [batch, action_dim], state-independent parameter broadcast
    values: Tensor      # [batch, 1], network output units (normalization is external)


class ActionSample(NamedTuple):
    actions: Tensor     # [batch, action_dim], raw samples, before environment clipping
    log_prob: Tensor    # [batch], joint log density of these exact samples
    values: Tensor     # [batch, 1]


class PolicyEvaluation(NamedTuple):
    log_prob: Tensor    # [batch]
    entropy: Tensor     # [batch], summed over action dimensions
    values: Tensor      # [batch, 1]


def _hidden_layers(config: ActorCriticConfig) -> nn.Sequential:
    activation_type = {"elu": nn.ELU, "relu": nn.ReLU,
                       "tanh": nn.Tanh, "selu": nn.SELU}[config.activation]
    layers = []
    width = config.observation_dim
    for next_width in config.hidden_sizes:
        layers.extend((nn.Linear(width, next_width), activation_type()))
        width = next_width
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Separate actor/critic MLPs and a trainable global log-standard deviation.

    Inputs must be floating [batch, observation_dim] tensors, already prepared by
    the caller. Outputs contain neither normalization nor environment transforms.
    .to(device) / .to(dtype) are native PyTorch operations; no device is hardcoded.
    Values retain [batch, 1]; log densities retain [batch], including batch size 1.
    """

    def __init__(self, config: Optional[ActorCriticConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else ActorCriticConfig()
        self.actor_mlp = _hidden_layers(self.config)
        self.critic_mlp = _hidden_layers(self.config)
        width = self.config.hidden_sizes[-1] if self.config.hidden_sizes else self.config.observation_dim
        # Reference constructs the value head before the mean head.
        self.value_head = nn.Linear(width, 1)
        self.mean_head = nn.Linear(width, self.config.action_dim)
        self.log_std = nn.Parameter(torch.full(
            (self.config.action_dim,), self.config.initial_log_std, dtype=torch.float32
        ))
        # Both reference "default" weight initializers are identity operations:
        # keep nn.Linear's constructor weights; zero all linear biases only.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def _check_observations(self, observations: Tensor) -> None:
        if observations.ndim != 2 or observations.shape[-1] != self.config.observation_dim:
            raise ValueError("Expected observations shaped [batch, observation_dim]")

    def actor(self, observations: Tensor) -> Tensor:
        """Unbounded action means; no output activation."""
        self._check_observations(observations)
        return self.mean_head(self.actor_mlp(observations))

    def critic(self, observations: Tensor) -> Tensor:
        """Scalar value per observation, with identity output activation."""
        self._check_observations(observations)
        return self.value_head(self.critic_mlp(observations))

    def forward(self, observations: Tensor) -> ActorCriticOutput:
        means = self.actor(observations)
        # Match reference broadcast/dtype promotion, including its zero actor path.
        log_std = means * 0.0 + self.log_std
        return ActorCriticOutput(means, log_std, self.critic(observations))

    def distribution(self, observations: Tensor) -> Independent:
        """Batch shape [batch], event shape [action_dim]."""
        means = self.actor(observations)
        return diagonal_gaussian(means, means * 0.0 + self.log_std)

    @torch.no_grad()
    def sample_actions(self, observations: Tensor) -> ActionSample:
        """Rollout samples without a pathwise gradient or environment transforms."""
        output = self(observations)
        actions = diagonal_gaussian(output.means, output.log_std).sample()
        return ActionSample(actions, gaussian_log_prob(actions, output.means, output.log_std), output.values)

    def log_prob(self, observations: Tensor, actions: Tensor) -> Tensor:
        """Evaluate original sampled actions, not their clipped/coupled variants."""
        means = self.actor(observations)
        return gaussian_log_prob(actions, means, means * 0.0 + self.log_std)

    def entropy(self, observations: Tensor) -> Tensor:
        return self.distribution(observations).entropy()

    def evaluate_actions(self, observations: Tensor, actions: Tensor) -> PolicyEvaluation:
        """Differentiable policy and value evaluation for a future PPO update."""
        output = self(observations)
        return PolicyEvaluation(
            gaussian_log_prob(actions, output.means, output.log_std),
            diagonal_gaussian(output.means, output.log_std).entropy(),
            output.values,
        )

    @torch.no_grad()
    def deterministic_actions(self, observations: Tensor) -> Tensor:
        """Return the raw mean without sampling, clipping, or consuming RNG state."""
        return self.actor(observations)
