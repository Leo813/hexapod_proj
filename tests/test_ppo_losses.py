import math
import unittest

from ppo_test_support import HAS_TORCH

if HAS_TORCH:
    import torch
    from hexapod_rl.config import PPOConfig
    from hexapod_rl.distributions import diagonal_gaussian
    from hexapod_rl.ppo import clipped_policy_loss, critic_loss, action_bounds_loss, ppo_loss


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class PPOLossTests(unittest.TestCase):
    def test_clipped_ratios_both_advantage_signs(self):
        ratios = torch.tensor([.5, 1.5, .5, 1.5], dtype=torch.float64)
        advantages = torch.tensor([2., 2., -2., -2.], dtype=torch.float64)
        # Individual losses: -1, -2.4, +1.6, +3; mean=.3.
        loss = clipped_policy_loss(ratios.log(), torch.zeros(4, dtype=torch.float64), advantages, .2)
        self.assertAlmostEqual(loss.item(), .3)

    def test_policy_log_probability_sign_and_unclipped_gradient(self):
        new = torch.tensor([math.log(1.1)], dtype=torch.float64, requires_grad=True)
        loss = clipped_policy_loss(new, torch.zeros(1, dtype=torch.float64), torch.tensor([2.]), .2)
        loss.backward()
        self.assertAlmostEqual(loss.item(), -2.2)
        self.assertAlmostEqual(new.grad.item(), -2.2)

    def test_value_clipping_takes_larger_error_without_half_factor(self):
        old = torch.zeros(2, 1)
        values = torch.tensor([[1.], [-1.]])
        self.assertAlmostEqual(critic_loss(values, old, values, .2).item(), .64, places=6)
        self.assertEqual(critic_loss(values, old, values, .2, False).item(), 0)
        self.assertEqual(critic_loss(values, old, -values, .2).item(), 4)

    def test_mean_bounds_threshold_and_gradient(self):
        means = torch.tensor([[0., 1.1, -1.1, 2.1, -2.1]], dtype=torch.float64, requires_grad=True)
        loss = action_bounds_loss(means)
        self.assertAlmostEqual(loss.item(), 2)
        loss.backward()
        torch.testing.assert_close(means.grad, torch.tensor([[0., 0., 0., 2., -2.]], dtype=torch.float64))

    def test_gaussian_entropy_sums_action_dimensions(self):
        means = torch.zeros(2, 3)
        entropy = diagonal_gaussian(means, torch.full((2, 3), math.log(2))).entropy()
        expected = 3 * (.5 * math.log(2 * math.pi * math.e) + math.log(2))
        torch.testing.assert_close(entropy, torch.full((2,), expected))

    def test_total_coefficients_and_reductions(self):
        c = PPOConfig(critic_coef=3, entropy_coef=.4, bounds_loss_coef=.25)
        terms = ppo_loss(torch.zeros(2), torch.zeros(2), torch.tensor([1., 3.]),
            torch.zeros(2, 1), torch.zeros(2, 1), torch.ones(2, 1),
            torch.tensor([2., 4.]), torch.tensor([[2.1], [-2.1]]), c)
        self.assertAlmostEqual(terms.policy.item(), -2)
        self.assertAlmostEqual(terms.critic.item(), 1)
        self.assertAlmostEqual(terms.entropy.item(), 3)
        self.assertAlmostEqual(terms.bounds.item(), 1, places=6)
        self.assertAlmostEqual(terms.total.item(), -2 + 1.5 - 1.2 + .25, places=6)

    def test_loss_gradcheck_away_from_clip_boundaries(self):
        x = torch.tensor([.01, -.02], dtype=torch.float64, requires_grad=True)
        old = torch.zeros_like(x)
        advantage = torch.tensor([1., -2.], dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(lambda v: clipped_policy_loss(v, old, advantage, .2), (x,)))
        means = torch.tensor([[1.5, -.5]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(action_bounds_loss, (means,)))

    def test_broadcasting_mistakes_rejected(self):
        with self.assertRaises(ValueError):
            clipped_policy_loss(torch.zeros(2), torch.zeros(2), torch.zeros(2, 1), .2)
        with self.assertRaises(ValueError):
            critic_loss(torch.zeros(2, 1), torch.zeros(2), torch.zeros(2, 1), .2)
