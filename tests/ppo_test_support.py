"""Test-only helpers; importing these requires no simulator or installed package."""

import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from hexapod_rl.config import ActorCriticConfig, PPOConfig
    from hexapod_rl.networks import ActorCritic
    from hexapod_rl.ppo import PPO
    from hexapod_rl.rollout import RolloutBuffer


def small_ppo(**overrides):
    torch.manual_seed(73)
    settings = dict(num_actors=2, horizon_length=4, minibatch_size=4, mini_epochs=2)
    settings.update(overrides)
    model = ActorCritic(ActorCriticConfig(observation_dim=3, action_dim=2, hidden_sizes=(8,)))
    return PPO(model, PPOConfig(**settings))


class MockVecTask:
    """Tiny deterministic arithmetic fixture, with reused tensors/deferred resets.

    The pending reset is applied AFTER integrating the next action. This is an
    interface test, not a dynamics or reward model of the hexapod.
    """

    def __init__(self):
        self.observations = torch.zeros(2, 3)
        self.observations[:, 0] = torch.tensor([0.2, -0.3])
        self.age = torch.zeros(2, dtype=torch.long)
        self.dones = torch.zeros(2, dtype=torch.bool)
        self.timeouts = torch.zeros(2, dtype=torch.bool)
        self.rewards = torch.zeros(2)
        self.reset_count = 0

    def step(self, actions):
        pending_reset = self.dones.clone()
        actions[:, 1] = -actions[:, 0]  # stand-in for in-place mimic coupling
        self.observations[:, :2].add_(0.1 * actions)
        self.age.add_(1)
        self.observations[:, 2] = self.age.float()
        self.reset_count += int(pending_reset.sum())
        self.observations[pending_reset] = 0
        self.age[pending_reset] = 0
        self.rewards.copy_(1 - self.observations[:, 0].square())
        self.dones.copy_(self.age >= torch.tensor([2, 3]))
        self.timeouts.copy_(self.dones & torch.tensor([False, True]))
        return {"obs": self.observations}, self.rewards, self.dones, {"time_outs": self.timeouts}


def collect_mock(algorithm, env=None):
    env = MockVecTask() if env is None else env
    c = algorithm.config
    buffer = RolloutBuffer(c.horizon_length, c.num_actors, 3, 2)
    obs, dones = env.observations, env.dones
    for _ in range(c.horizon_length):
        policy = algorithm.sample(obs)
        buffer.store_policy_step(obs, dones, *policy)
        # Future adapter owns this transform. Buffer/sample retain RAW actions.
        observation, reward, dones, info = env.step(policy.actions.clamp(-1, 1).clone())
        buffer.store_outcome(reward, dones, info["time_outs"])
        obs = observation["obs"]
    buffer.finish(algorithm.predict_values(obs))
    return buffer, env
