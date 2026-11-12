"""Stage 1 network settings; defaults mirror hexapodPPO.yaml.

This is not a Hydra/YAML translator. Normalization and trainer configuration are
intentionally deferred. Non-default settings describe new experiments, not the
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
