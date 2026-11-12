"""Unsquashed diagonal Gaussian policy mathematics.

Reference: RL-Games v1.6.0, ModelA2CContinuousLogStd.Network.neglogp.
See docs/reference-implementation.md for pinned source attribution. These are
independent implementations of the equations, with positive log-probabilities
instead of the reference API's negative log-probabilities.
"""

import math

import torch
from torch import Tensor
from torch.distributions import Independent, Normal


def diagonal_gaussian(means: Tensor, log_std: Tensor) -> Independent:
    """Treat the last dimension as one action event; do not squash or clamp.

Normal.sample(), rather than rsample(), matches reference rollout sampling.
The caller owns observation normalization and environment action processing.
"""
    return Independent(Normal(means, log_std.exp(), validate_args=False), 1)


def gaussian_log_prob(actions: Tensor, means: Tensor, log_std: Tensor) -> Tensor:
    """Joint log density, summed over actions and retaining the batch dimension.

log_std may be a per-action vector or already broadcast to means.shape.
No epsilon, standard-deviation clipping, or tanh Jacobian is introduced.
"""
    if actions.shape != means.shape:
        raise ValueError("actions and means must have identical shapes")
    standardized = (actions - means) / log_std.exp()
    quadratic = 0.5 * standardized.square().sum(dim=-1)
    normalizer = 0.5 * math.log(2.0 * math.pi) * actions.shape[-1]
    log_scale = torch.broadcast_to(log_std, means.shape).sum(dim=-1)
    return -(quadratic + normalizer + log_scale)
