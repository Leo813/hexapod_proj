"""Owned rollout tensors and deterministic, environment-major minibatches."""

import torch


def flatten_time_env(tensor):
    """[time, env, ...] -> [env * time, ...], in independently owned storage."""
    if tensor.ndim < 2:
        raise ValueError("Expected time and environment dimensions")
    return tensor.transpose(0, 1).reshape(-1, *tensor.shape[2:]).detach().clone()


class RolloutBuffer:
    """Two-phase storage prevents a mutating env.step from corrupting samples.

    Call store_policy_step BEFORE env.step, then store_outcome. Values are in
    reward units, log_probs are positive-sign log densities, dones describe the
    state BEFORE each action. next_dones/timeouts describe the step outcome.
    """

    def __init__(self, horizon_length, num_envs, observation_dim, action_dim,
                 device="cpu", dtype=torch.float32):
        dimensions = (horizon_length, num_envs, observation_dim, action_dim)
        if any(type(x) is not int or x <= 0 for x in dimensions):
            raise ValueError("All buffer dimensions must be positive integers")
        if not dtype.is_floating_point:
            raise ValueError("Rollout storage must use a floating dtype")
        self.horizon_length, self.num_envs = horizon_length, num_envs
        self.observation_dim, self.action_dim = observation_dim, action_dim
        prefix = (horizon_length, num_envs)
        self.data = {}
        for name, tail in (("observations", (observation_dim,)), ("actions", (action_dim,)),
                           ("log_probs", ()), ("values", (1,)), ("rewards", (1,)),
                           ("means", (action_dim,)), ("stds", (action_dim,))):
            self.data[name] = torch.empty(*prefix, *tail, device=device, dtype=dtype)
        for name in ("dones", "next_dones", "timeouts"):
            self.data[name] = torch.empty(prefix, device=device, dtype=torch.bool)
        self.last_values = torch.empty(num_envs, 1, device=device, dtype=dtype)
        self.last_dones = torch.empty(num_envs, device=device, dtype=torch.bool)
        self.reset()

    def reset(self):
        self.position = 0
        self.pending = False
        self.finished = False

    @staticmethod
    def _validate(destination, source, name):
        if destination.shape != source.shape or destination.device != source.device:
            raise ValueError(f"{name}: expected shape {tuple(destination.shape)} on {destination.device}")

    @torch.no_grad()
    def store_policy_step(self, observations, dones, actions, log_probs, values, means, stds):
        if self.pending or self.position >= self.horizon_length or self.finished:
            raise RuntimeError("Finish the pending step or reset a full buffer before storing")
        inputs = dict(observations=observations, dones=dones, actions=actions,
                      log_probs=log_probs, values=values, means=means, stds=stds)
        for name, source in inputs.items():
            self._validate(self.data[name][self.position], source, name)
        for name, source in inputs.items():
            self.data[name][self.position].copy_(source.detach())
        self.pending = True

    @torch.no_grad()
    def store_outcome(self, rewards, next_dones, timeouts=None):
        if not self.pending:
            raise RuntimeError("Store the policy step before its outcome")
        if rewards.shape == (self.num_envs,):
            rewards = rewards.unsqueeze(-1)
        if timeouts is None:
            timeouts = torch.zeros_like(next_dones, dtype=torch.bool)
        inputs = dict(rewards=rewards, next_dones=next_dones, timeouts=timeouts)
        for name, source in inputs.items():
            self._validate(self.data[name][self.position], source, name)
        for name, source in inputs.items():
            self.data[name][self.position].copy_(source.detach())
        self.position += 1
        self.pending = False

    @torch.no_grad()
    def finish(self, last_values):
        if self.pending or self.position != self.horizon_length:
            raise RuntimeError("Only a complete rollout can be bootstrapped")
        self._validate(self.last_values, last_values, "last_values")
        self.last_values.copy_(last_values.detach())
        self.last_dones.copy_(self.data["next_dones"][-1])
        self.finished = True

    def flatten(self):
        if not self.finished:
            raise RuntimeError("Finish the rollout before preparing optimization data")
        return {name: flatten_time_env(value) for name, value in self.data.items()}


class TrainingBatch:
    """Prepared, detached training data with mutable KL distribution references.

    The caller supplies owned tensors. Minibatches are views; only means/stds
    change between passes. Old sampled log densities and value targets stay fixed.
    """

    def __init__(self, data):
        required = {"observations", "actions", "old_log_probs", "old_values",
                    "returns", "advantages", "means", "stds"}
        if set(data) != required:
            raise ValueError(f"Training data must contain exactly {sorted(required)}")
        self.size = data["observations"].shape[0]
        if self.size == 0:
            raise ValueError("Training batches cannot be empty")
        device = data["observations"].device
        for name, tensor in data.items():
            if tensor.shape[0] != self.size or tensor.device != device or tensor.requires_grad:
                raise ValueError(f"{name}: expected detached data on one device with a common batch size")
        for name in ("old_log_probs", "advantages"):
            if data[name].shape != (self.size,):
                raise ValueError(f"{name} must have shape [batch]")
        for name in ("old_values", "returns"):
            if data[name].shape != (self.size, 1):
                raise ValueError(f"{name} must have shape [batch, 1]")
        if data["observations"].ndim != 2 or data["actions"].ndim != 2:
            raise ValueError("Observations and actions must be matrices")
        if data["actions"].shape != data["means"].shape or data["means"].shape != data["stds"].shape:
            raise ValueError("Actions, means, and stds must have the same shape")
        self.data = data

    def minibatches(self, minibatch_size):
        if type(minibatch_size) is not int or minibatch_size <= 0 or self.size % minibatch_size:
            raise ValueError("minibatch_size must divide the batch size")
        for start in range(0, self.size, minibatch_size):
            indices = slice(start, start + minibatch_size)
            yield indices, {name: value[indices] for name, value in self.data.items()}

    @torch.no_grad()
    def update_distribution(self, indices, means, stds):
        for name, value in (("means", means), ("stds", stds)):
            RolloutBuffer._validate(self.data[name][indices], value, name)
        self.data["means"][indices].copy_(means.detach())
        self.data["stds"][indices].copy_(stds.detach())
