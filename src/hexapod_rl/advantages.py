"""GAE and timeout reward shaping from the selected historical reference.

The done at index t belongs to observation s_t, NOT the result of action a_t.
The outcome's done goes at t+1 (or last_dones at the rollout boundary).
No Gymnasium terminal-observation substitution is performed.
"""

import torch
from torch import Tensor


@torch.no_grad()
def adjust_timeout_rewards(rewards: Tensor, values: Tensor, timeouts: Tensor,
                           gamma: float = 0.99, enabled: bool = True,
                           reward_scale: float = 1.0) -> Tensor:
    """Scale rewards, then add gamma*V(s_t) on timeout outcomes (pre-step V).

    Rewards/values end in a singleton value axis, and timeouts omit that axis.
    Returns new storage; inputs, including the raw environment reward, survive.
    """
    if rewards.shape != values.shape or rewards.shape[-1:] != (1,) or timeouts.shape != rewards.shape[:-1]:
        raise ValueError("Rewards/values need [..., 1] and timeouts need [...]")
    result = rewards * reward_scale
    if enabled:
        result = result + gamma * values * timeouts.to(values.dtype).unsqueeze(-1)
    return result


@torch.no_grad()
def generalized_advantage_estimate(rewards: Tensor, values: Tensor, dones: Tensor,
                                  last_values: Tensor, last_dones: Tensor,
                                  gamma: float = 0.99, tau: float = 0.95):
    """Return (advantages, returns), both [time, env, 1], from shaped rewards."""
    if rewards.ndim != 3 or rewards.shape[-1] != 1 or rewards.shape != values.shape:
        raise ValueError("Rewards and values must have identical [time, env, 1] shapes")
    if rewards.shape[0] == 0 or dones.shape != rewards.shape[:2]:
        raise ValueError("Need a nonempty rollout and [time, env] observation-boundary flags")
    if last_values.shape != values.shape[1:] or last_dones.shape != dones.shape[1:]:
        raise ValueError("Bootstrap values/flags must have shapes [env, 1] / [env]")
    advantages = torch.empty_like(rewards)
    carry = torch.zeros_like(last_values)
    for t in reversed(range(rewards.shape[0])):
        next_value = last_values if t == rewards.shape[0] - 1 else values[t + 1]
        next_done = last_dones if t == rewards.shape[0] - 1 else dones[t + 1]
        continuing = 1.0 - next_done.to(values.dtype).unsqueeze(-1)
        delta = rewards[t] + gamma * next_value * continuing - values[t]
        carry = delta + gamma * tau * continuing * carry
        advantages[t].copy_(carry)
    return advantages, advantages + values


@torch.no_grad()
def normalize_advantages(advantages: Tensor, epsilon: float = 1e-8) -> Tensor:
    """Global sample-standard-deviation normalization, epsilon outside std."""
    if advantages.ndim != 1 or advantages.numel() < 2:
        raise ValueError("Need at least two flattened scalar advantages")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return (advantages - advantages.mean()) / (advantages.std(unbiased=True) + epsilon)
