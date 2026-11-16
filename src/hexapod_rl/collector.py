"""Persistent production collection and historical episode reward statistics."""

import torch

from .rollout import RolloutBuffer


class EpisodeStatistics:
    """RL-Games AverageMeter's capped batch-weighted mean, not a deque average."""

    def __init__(self, num_envs, device, capacity=100):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("Episode meter capacity must be positive")
        self.capacity = capacity
        self.current_rewards = torch.zeros(num_envs, device=device)
        self.current_lengths = torch.zeros(num_envs, device=device)
        self.mean = torch.zeros(2, device=device)
        self.size = self.completed = 0

    @torch.no_grad()
    def update(self, rewards, dones):
        self.current_rewards.add_(rewards)
        self.current_lengths.add_(1)
        indices = dones.nonzero(as_tuple=False).flatten()
        count = indices.numel()
        if count:
            batch_mean = torch.stack((self.current_rewards[indices].mean(), self.current_lengths[indices].mean()))
            new_size = min(count, self.capacity)
            old_size = min(self.size, self.capacity - new_size)
            self.mean.copy_((self.mean * old_size + batch_mean * new_size) / (old_size + new_size))
            self.size = old_size + new_size
            self.completed += count
        mask = 1 - dones.float()
        self.current_rewards.mul_(mask)
        self.current_lengths.mul_(mask)

    def summary(self):
        return dict(episode_reward=self.mean[0].item() if self.size else None,
                    episode_length=self.mean[1].item() if self.size else None,
                    completed_episodes=self.completed)


class RolloutCollector:
    def __init__(self, algorithm, environment, games_to_track=100):
        self.algorithm, self.environment = algorithm, environment
        c, model = algorithm.config, algorithm.model.config
        if (environment.num_envs, environment.observation_dim, environment.action_dim, environment.device) != (
                c.num_actors, model.observation_dim, model.action_dim, algorithm.device):
            raise ValueError("Collector environment dimensions/device must match the PPO configuration")
        self.buffer = RolloutBuffer(c.horizon_length, c.num_actors, model.observation_dim,
                                    model.action_dim, device=algorithm.device)
        self.statistics = EpisodeStatistics(c.num_actors, algorithm.device, games_to_track)
        self.dones = torch.ones(c.num_actors, dtype=torch.bool, device=algorithm.device)
        self.observations = environment.reset()

    @torch.no_grad()
    def collect(self):
        self.buffer.reset()
        for _ in range(self.algorithm.config.horizon_length):
            policy = self.algorithm.sample(self.observations)
            self.buffer.store_policy_step(self.observations, self.dones, *policy)
            obs, rewards, dones, timeouts = self.environment.step(policy.actions)
            self.buffer.store_outcome(rewards, dones, timeouts)
            self.statistics.update(rewards, dones)  # raw rewards, before timeout shaping
            self.observations = obs
            self.dones = dones.detach().clone()
        self.buffer.finish(self.algorithm.predict_values(self.observations))
        return self.buffer
