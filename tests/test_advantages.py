import math
import unittest

from ppo_test_support import HAS_TORCH

if HAS_TORCH:
    import torch
    from hexapod_rl.advantages import adjust_timeout_rewards, generalized_advantage_estimate, normalize_advantages


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class AdvantageTests(unittest.TestCase):
    def test_continuing_trajectory_with_horizon_bootstrap(self):
        rewards = torch.tensor([[[1.]], [[2.]]])
        values = torch.tensor([[[0.5]], [[0.25]]])
        # delta1=2+.9*.4-.25=2.11; delta0=1+.9*.25-.5=.725
        adv, returns = generalized_advantage_estimate(rewards, values, torch.zeros(2, 1),
            torch.tensor([[0.4]]), torch.zeros(1), gamma=.9, tau=.8)
        torch.testing.assert_close(adv, torch.tensor([[[2.2442]], [[2.11]]]))
        torch.testing.assert_close(returns, torch.tensor([[[2.7442]], [[2.36]]]))

    def test_terminal_ignores_final_value(self):
        adv, returns = generalized_advantage_estimate(torch.tensor([[[1.]], [[2.]]]),
            torch.tensor([[[.5]], [[.25]]]), torch.zeros(2, 1), torch.tensor([[999.]]), torch.ones(1))
        torch.testing.assert_close(adv, torch.tensor([[[2.393375]], [[1.75]]]))
        torch.testing.assert_close(returns, torch.tensor([[[2.893375]], [[2.]]]))

    def test_next_step_done_prevents_cross_episode_recursion(self):
        # First observation's done is irrelevant; the boundary at t=1 cuts t=0.
        rewards = torch.tensor([[[1.], [1.]], [[100.], [100.]]])
        values = torch.zeros_like(rewards)
        adv, _ = generalized_advantage_estimate(rewards, values, torch.tensor([[1, 0], [1, 0]]),
            torch.zeros(2, 1), torch.ones(2), gamma=.9, tau=.8)
        torch.testing.assert_close(adv, torch.tensor([[[1.], [73.]], [[100.], [100.]]]))

    def test_timeout_uses_pre_step_value_after_reward_scale(self):
        reward, value = torch.tensor([[[1.]]]), torch.tensor([[[2.]]])
        shaped = adjust_timeout_rewards(reward, value, torch.ones(1, 1), gamma=.9, reward_scale=2)
        self.assertAlmostEqual(shaped.item(), 3.8, places=6)
        adv, ret = generalized_advantage_estimate(shaped, value, torch.zeros(1, 1),
            torch.tensor([[123.]]), torch.ones(1), gamma=.9)
        self.assertAlmostEqual(adv.item(), 1.8, places=6)
        self.assertAlmostEqual(ret.item(), 3.8, places=6)
        self.assertEqual(reward.item(), 1)

    def test_disabled_bootstrap_and_true_terminal(self):
        r, v = torch.ones(1, 2, 1), torch.full((1, 2, 1), 2.)
        shaped = adjust_timeout_rewards(r, v, torch.tensor([[True, False]]), enabled=False)
        torch.testing.assert_close(shaped, r)
        enabled = adjust_timeout_rewards(r, v, torch.tensor([[True, False]]), gamma=.9)
        torch.testing.assert_close(enabled, torch.tensor([[[2.8], [1.]]]))

    def test_zero_discount_and_zero_lambda(self):
        r, v = torch.tensor([[[1.]], [[3.]]]), torch.tensor([[[.5]], [[2.]]])
        args = (r, v, torch.zeros(2, 1), torch.tensor([[4.]]), torch.zeros(1))
        adv, ret = generalized_advantage_estimate(*args, gamma=0)
        torch.testing.assert_close(adv, r - v)
        torch.testing.assert_close(ret, r)
        adv, _ = generalized_advantage_estimate(*args, gamma=.5, tau=0)
        torch.testing.assert_close(adv, torch.tensor([[[1.5]], [[3.]]]))

    def test_advantage_sample_std_and_epsilon_outside(self):
        x = torch.tensor([1., 2., 4.], dtype=torch.float64)
        expected = (x - 7 / 3) / (math.sqrt(7 / 3) + .1)
        torch.testing.assert_close(normalize_advantages(x, .1), expected)
        torch.testing.assert_close(normalize_advantages(torch.ones(4)), torch.zeros(4))
        with self.assertRaises(ValueError):
            normalize_advantages(torch.ones(1))

    def test_invalid_shapes_rejected(self):
        with self.assertRaises(ValueError):
            adjust_timeout_rewards(torch.ones(2), torch.ones(2), torch.ones(2))
        with self.assertRaises(ValueError):
            generalized_advantage_estimate(torch.zeros(2, 1), torch.zeros(2, 1),
                                          torch.zeros(2, 1), torch.zeros(1, 1), torch.zeros(1))
