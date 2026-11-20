"""Reference-specific KL and legacy per-minibatch adaptive LR, not early stopping.

Source attribution: RL-Games v1.6.0 torch_ext.policy_kl and
common.schedulers.AdaptiveScheduler.
See docs/ppo-implementation.md for the maintained equations and update timing.
"""

import math


def adaptive_learning_rate(current_lr: float, kl: float, threshold: float = 0.008,
                           min_lr: float = 1e-6, max_lr: float = 1e-2) -> float:
    if not all(math.isfinite(x) for x in (current_lr, kl, threshold, min_lr, max_lr)):
        raise ValueError("Scheduler inputs must be finite")
    if current_lr <= 0 or threshold <= 0 or min_lr <= 0 or max_lr < min_lr:
        raise ValueError("Invalid scheduler rates or threshold")
    if kl > 2.0 * threshold:
        return max(current_lr / 1.5, min_lr)
    if kl < 0.5 * threshold:
        return min(current_lr * 1.5, max_lr)
    return current_lr


def policy_kl(current_mean, current_std, reference_mean, reference_std, reduce=True):
    """KL(current || stored) with the reference's two unusual 1e-5 placements.

    Stored means/stds are refreshed after each minibatch. The current parameters
    come from the forward pass BEFORE the optimizer step. Same distributions
    need not give exactly zero; do not clamp or replace this with analytic KL.
    """
    import torch

    if not (current_mean.shape == current_std.shape == reference_mean.shape == reference_std.shape):
        raise ValueError("All distribution parameters must have the same shape")
    log_ratio = torch.log(reference_std / current_std + 1e-5)
    numerator = current_std.square() + (reference_mean - current_mean).square()
    terms = log_ratio + numerator / (2.0 * (reference_std.square() + 1e-5)) - 0.5
    per_sample = terms.sum(dim=-1)
    return per_sample.mean() if reduce else per_sample
