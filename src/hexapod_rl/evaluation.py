"""Deterministic evaluation with frozen normalization and no PPO updates."""

import torch

from .collector import EpisodeStatistics


@torch.no_grad()
def evaluate(algorithm, environment, steps=1000, games_to_track=100):
    if type(steps) is not int or steps <= 0:
        raise ValueError("Evaluation steps must be positive")
    try:
        algorithm.model.eval()
        for module in (algorithm.observation_normalizer, algorithm.value_normalizer):
            if module is not None:
                module.eval()
        stats = EpisodeStatistics(environment.num_envs, environment.device, games_to_track)
        observations = environment.reset()
        for _ in range(steps):
            normalized = (algorithm.observation_normalizer(observations)
                          if algorithm.observation_normalizer is not None else observations)
            actions = algorithm.model.deterministic_actions(normalized)
            observations, rewards, dones, _ = environment.step(actions)
            stats.update(rewards, dones)
        return dict(frames=steps * environment.num_envs, **stats.summary())
    finally:
        environment.close()
