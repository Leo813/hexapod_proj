"""Minimal tensor adapter around the unchanged VecTask interface."""

import torch


def close_environment(environment, task_writer=None):
    """Release a constructed task, including cleanup after adapter validation fails."""
    try:
        closer = getattr(environment, "close", None)
        if callable(closer):
            closer()
        else:
            gym = environment.gym
            try:
                if environment.viewer is not None:
                    gym.destroy_viewer(environment.viewer)
            finally:
                gym.destroy_sim(environment.sim)
            display = getattr(environment, "virtual_display", None)
            if display is not None:
                display.stop()
    finally:
        if task_writer is not None:
            task_writer.close()


class IsaacGymAdapter:
    def __init__(self, environment, device, observation_dim=104, action_dim=18, task_writer=None):
        self.environment = environment
        self.device = torch.device(device)
        self.num_envs = environment.num_envs
        self.observation_dim, self.action_dim = observation_dim, action_dim
        self.task_writer, self.closed = task_writer, False
        if environment.num_obs != observation_dim or environment.num_acts != action_dim:
            raise ValueError(f"Environment dimensions must be observations={observation_dim}, actions={action_dim}")
        if type(self.num_envs) is not int or self.num_envs <= 0:
            raise ValueError("Environment num_envs must be a positive integer")
        space = environment.action_space
        low, high = torch.as_tensor(space.low), torch.as_tensor(space.high)
        if tuple(space.shape) != (action_dim,) or not torch.all(low == -1) or not torch.all(high == 1):
            raise ValueError("The hexapod adapter requires continuous action bounds [-1, 1]")

    def _tensor(self, value, shape, name, flag=False):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"{name} must be a tensor shaped {shape}")
        if value.device != self.device:
            raise ValueError(f"{name} is on {value.device}, expected RL device {self.device}; check VecTask rl_device")
        if flag:
            if value.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError(f"{name} must contain integer or boolean flags")
        elif value.dtype != torch.float32:
            raise ValueError(f"{name} must be float32, got {value.dtype}")
        return value

    def observations(self, observation_dict):
        if not isinstance(observation_dict, dict) or "obs" not in observation_dict:
            raise ValueError("VecTask must return an observation dictionary containing 'obs'")
        return self._tensor(observation_dict["obs"], (self.num_envs, self.observation_dim), "observations")

    def reset(self):
        """Obtain existing observations once; never reset_idx or step eagerly."""
        return self.observations(self.environment.reset())

    def step(self, raw_actions):
        self._tensor(raw_actions, (self.num_envs, self.action_dim), "actions")
        # clamp allocates independent storage, even when every action is in bounds.
        actions = raw_actions.detach().clamp(-1, 1)
        obs, rewards, dones, extras = self.environment.step(actions)
        observations = self.observations(obs)  # VecTask has already clipped features
        rewards = self._tensor(rewards, (self.num_envs,), "rewards")
        dones = self._tensor(dones, (self.num_envs,), "dones", flag=True)
        if not isinstance(extras, dict) or "time_outs" not in extras:
            raise ValueError("Historical VecTask must return extras['time_outs']; timeout information cannot be discarded")
        timeouts = self._tensor(extras["time_outs"], (self.num_envs,), "time_outs", flag=True)
        return observations, rewards, dones, timeouts

    def set_train_info(self, frames):
        self.environment.set_train_info(frames)

    def close(self):
        """The pinned VecTask has no close(); release its native handles explicitly."""
        if self.closed:
            return
        self.closed = True
        close_environment(self.environment, self.task_writer)
