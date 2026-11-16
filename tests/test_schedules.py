import math
import unittest

from ppo_test_support import HAS_TORCH
from hexapod_rl.schedules import adaptive_learning_rate

if HAS_TORCH:
    import torch
    from hexapod_rl.schedules import policy_kl


class AdaptiveScheduleTests(unittest.TestCase):
    def test_high_low_and_middle_kl(self):
        self.assertAlmostEqual(adaptive_learning_rate(3e-4, .017), 2e-4)
        self.assertAlmostEqual(adaptive_learning_rate(3e-4, .003), 4.5e-4)
        self.assertEqual(adaptive_learning_rate(3e-4, .008), 3e-4)

    def test_thresholds_are_strict(self):
        self.assertEqual(adaptive_learning_rate(3e-4, .016), 3e-4)
        self.assertEqual(adaptive_learning_rate(3e-4, .004), 3e-4)

    def test_limits_and_reference_negative_kl(self):
        self.assertEqual(adaptive_learning_rate(1e-6, .1), 1e-6)
        self.assertEqual(adaptive_learning_rate(1e-2, 0), 1e-2)
        self.assertAlmostEqual(adaptive_learning_rate(3e-4, -.0001), 4.5e-4)

    def test_invalid_inputs(self):
        for args in ((0, 0), (1e-3, float("nan")), (1e-3, 0, 0), (1e-3, 0, .008, .1, .01)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                adaptive_learning_rate(*args)


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class PolicyKLTests(unittest.TestCase):
    def test_direction_and_epsilon_placements(self):
        m0, s0 = torch.tensor([[1., -1.]], dtype=torch.float64), torch.tensor([[2., .5]], dtype=torch.float64)
        m1, s1 = torch.tensor([[0., 2.]], dtype=torch.float64), torch.tensor([[.3, 3.]], dtype=torch.float64)
        expected = sum(math.log(b / a + 1e-5) + (a*a + (y-x)**2) / (2*(b*b+1e-5)) - .5
                       for x, a, y, b in ((1, 2, 0, .3), (-1, .5, 2, 3)))
        self.assertAlmostEqual(policy_kl(m0, s0, m1, s1).item(), expected)
        self.assertNotAlmostEqual(policy_kl(m1, s1, m0, s0).item(), expected)

    def test_same_distribution_is_not_forced_to_zero(self):
        mean = torch.zeros(1, 1, dtype=torch.float64)
        std = torch.full_like(mean, .1)
        expected = math.log(1 + 1e-5) + .01 / (2 * (.01 + 1e-5)) - .5
        self.assertLess(expected, 0)
        self.assertAlmostEqual(policy_kl(mean, std, mean, std).item(), expected)

    def test_sum_over_actions_mean_over_samples(self):
        mean, std = torch.zeros(2, 3), torch.ones(2, 3)
        reference = torch.tensor([[1., 1., 1.], [2., 2., 2.]])
        each = policy_kl(mean, std, reference, std, reduce=False)
        expected = torch.tensor([3 * (math.log(1.00001) + 2 / 2.00002 - .5),
                                 3 * (math.log(1.00001) + 5 / 2.00002 - .5)])
        torch.testing.assert_close(each, expected)
        torch.testing.assert_close(policy_kl(mean, std, reference, std), expected.mean())
