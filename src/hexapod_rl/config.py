"""Typed network and PPO settings; defaults mirror hexapodPPO.yaml.

This is not a Hydra/YAML translator. Non-default settings describe new experiments, not the
historical baseline.
"""

from dataclasses import dataclass
import math
from typing import Tuple


@dataclass(frozen=True)
class ActorCriticConfig:
    observation_dim: int = 104
    action_dim: int = 18
    hidden_sizes: Tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    initial_log_std: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_sizes", tuple(self.hidden_sizes))
        dimensions = (self.observation_dim, self.action_dim, *self.hidden_sizes)
        if any(type(size) is not int or size <= 0 for size in dimensions):
            raise ValueError("Observation, action, and hidden dimensions must be positive integers")
        if self.activation not in ("elu", "relu", "tanh", "selu"):
            raise ValueError("Supported activations: elu, relu, tanh, selu")
        if not math.isfinite(self.initial_log_std):
            raise ValueError("initial_log_std must be finite")


@dataclass(frozen=True)
class PPOConfig:
    """Single-device feedforward PPO; names match YAML where possible.

    Hydra resolution, environment creation, and experiment/checkpoint settings
    remain outside this configuration. See docs/stage2-implementation.md.
    """

    num_actors: int = 256
    horizon_length: int = 48
    minibatch_size: int = 12288
    mini_epochs: int = 5
    gamma: float = 0.99
    tau: float = 0.95
    e_clip: float = 0.2
    critic_coef: float = 2.0
    entropy_coef: float = 0.0
    bounds_loss_coef: float = 0.001
    clip_value: bool = True
    learning_rate: float = 3e-4
    adam_epsilon: float = 1e-8
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    truncate_grads: bool = True
    grad_norm: float = 1.0
    mixed_precision: bool = True
    normalize_input: bool = True
    normalize_value: bool = True
    normalize_advantage: bool = True
    value_bootstrap: bool = True
    reward_scale: float = 1.0
    normalization_epsilon: float = 1e-5
    normalization_clip: float = 5.0
    advantage_epsilon: float = 1e-8
    lr_schedule: str = "adaptive"
    kl_threshold: float = 0.008
    min_learning_rate: float = 1e-6
    max_learning_rate: float = 1e-2

    @property
    def batch_size(self) -> int:
        return self.num_actors * self.horizon_length

    def __post_init__(self) -> None:
        for name in ("num_actors", "horizon_length", "minibatch_size", "mini_epochs"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_size % self.minibatch_size:
            raise ValueError("minibatch_size must divide the rollout batch size exactly")
        for name in ("clip_value", "truncate_grads", "mixed_precision", "normalize_input",
                     "normalize_value", "normalize_advantage", "value_bootstrap"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool")
        for name in ("gamma", "tau", "e_clip", "critic_coef", "entropy_coef", "bounds_loss_coef",
                     "learning_rate", "adam_epsilon", "weight_decay", "grad_norm", "reward_scale",
                     "normalization_epsilon", "normalization_clip", "advantage_epsilon",
                     "kl_threshold", "min_learning_rate", "max_learning_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        for name in ("gamma", "tau"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0 <= self.e_clip < 1:
            raise ValueError("e_clip must be in [0, 1)")
        for name in ("critic_coef", "entropy_coef", "bounds_loss_coef", "weight_decay"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in ("learning_rate", "adam_epsilon", "grad_norm", "normalization_epsilon",
                     "normalization_clip", "advantage_epsilon", "kl_threshold",
                     "min_learning_rate", "max_learning_rate"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        object.__setattr__(self, "adam_betas", tuple(self.adam_betas))
        if len(self.adam_betas) != 2 or any(
            isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(b) or not 0 <= b < 1
            for b in self.adam_betas
        ):
            raise ValueError("adam_betas must contain two finite numbers in [0, 1)")
        if self.min_learning_rate > self.max_learning_rate:
            raise ValueError("min_learning_rate must not exceed max_learning_rate")
        if self.lr_schedule not in ("adaptive", "constant"):
            raise ValueError("Only adaptive (legacy per-minibatch) or constant schedules are supported")
        # The reference produces undefined sample variance for singleton batches.
        if self.normalize_input and self.minibatch_size < 2:
            raise ValueError("Observation statistics require at least two samples per minibatch")
        if (self.normalize_value or self.normalize_advantage) and self.batch_size < 2:
            raise ValueError("Value/advantage normalization requires at least two rollout samples")
