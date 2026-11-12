"""Numerical tests: require torch, never require Isaac Gym or RL-Games."""

import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hexapod_rl import ActorCriticConfig

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch import nn
    from hexapod_rl.distributions import diagonal_gaussian, gaussian_log_prob
    from hexapod_rl.networks import ActorCritic


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed; numerical validation is blocked")
class ActorCriticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)

    def test_baseline_structure(self):
        model = ActorCritic()
        for mlp in (model.actor_mlp, model.critic_mlp):
            self.assertEqual([(x.in_features, x.out_features) for x in mlp if isinstance(x, nn.Linear)],
                             [(104, 512), (512, 256), (256, 128)])
            self.assertEqual([type(x) for x in mlp], [nn.Linear, nn.ELU] * 3)
            self.assertTrue(all(x.alpha == 1.0 for x in mlp if isinstance(x, nn.ELU)))
        self.assertEqual((model.mean_head.in_features, model.mean_head.out_features), (128, 18))
        self.assertEqual((model.value_head.in_features, model.value_head.out_features), (128, 1))

    def test_actor_and_critic_do_not_share_parameters(self):
        model = ActorCritic()
        actor = list(model.actor_mlp.parameters()) + list(model.mean_head.parameters())
        critic = list(model.critic_mlp.parameters()) + list(model.value_head.parameters())
        self.assertTrue({p.data_ptr() for p in actor}.isdisjoint(p.data_ptr() for p in critic))

    def test_different_batch_sizes_and_cpu(self):
        model = ActorCritic().cpu()
        for size in (1, 7, 256):
            with self.subTest(batch_size=size):
                obs = torch.randn(size, 104)
                output = model(obs)
                self.assertEqual(output.means.shape, (size, 18))
                self.assertEqual(output.log_std.shape, (size, 18))
                self.assertEqual(output.values.shape, (size, 1))
                self.assertEqual(output.values.device.type, "cpu")
                self.assertTrue(all(torch.isfinite(t).all() for t in output))
                sample = model.sample_actions(obs)
                self.assertEqual(sample.actions.shape, (size, 18))
                self.assertEqual(sample.log_prob.shape, (size,))
                self.assertEqual(model.entropy(obs).shape, (size,))
                self.assertEqual(model.distribution(obs).batch_shape, (size,))
                self.assertEqual(model.distribution(obs).event_shape, (18,))

    def test_initial_weights_preserve_native_linear_initialization(self):
        config = ActorCriticConfig(observation_dim=3, action_dim=2, hidden_sizes=(5, 4))
        torch.manual_seed(19)
        # Native constructors in the verified hidden/hidden/value/mean order.
        expected = [nn.Linear(i, o) for i, o in [(3, 5), (5, 4), (3, 5), (5, 4), (4, 1), (4, 2)]]
        torch.manual_seed(19)
        actual = ActorCritic(config)
        layers = [x for x in actual.modules() if isinstance(x, nn.Linear)]
        for reference, layer in zip(expected, layers):
            torch.testing.assert_close(layer.weight, reference.weight, rtol=0, atol=0)
            self.assertEqual(torch.count_nonzero(layer.bias).item(), 0)
            bound = 1 / math.sqrt(layer.in_features)
            self.assertTrue((layer.weight.abs() <= bound).all())

    def test_initial_log_std_is_trainable_global_vector(self):
        model = ActorCritic()
        self.assertIsInstance(model.log_std, nn.Parameter)
        self.assertTrue(model.log_std.requires_grad)
        self.assertEqual(model.log_std.shape, (18,))
        self.assertEqual(torch.count_nonzero(model.log_std).item(), 0)
        output = model(torch.randn(3, 104))
        torch.testing.assert_close(output.log_std.exp(), torch.ones(3, 18))
        torch.testing.assert_close(output.log_std[0], output.log_std[2])

    def test_configurable_architecture_and_initial_std(self):
        model = ActorCritic(ActorCriticConfig(3, 2, (6,), "tanh", math.log(0.5)))
        self.assertIsInstance(model.actor_mlp[1], nn.Tanh)
        output = model(torch.ones(4, 3))
        self.assertEqual(output.means.shape, (4, 2))
        torch.testing.assert_close(output.log_std.exp(), torch.full((4, 2), 0.5))

    def test_no_hidden_layers(self):
        model = ActorCritic(ActorCriticConfig(3, 2, ()))
        self.assertEqual(len(model.actor_mlp), 0)
        self.assertEqual(model(torch.ones(1, 3)).values.shape, (1, 1))

    def test_hand_calculated_forward_fixture(self):
        model = ActorCritic(ActorCriticConfig(2, 2, (2,))).double()
        with torch.no_grad():
            model.actor_mlp[0].weight.copy_(torch.tensor([[1., 0.], [0., 2.]]))
            model.mean_head.weight.copy_(torch.tensor([[1., 2.], [-1., 0.5]]))
            model.mean_head.bias.copy_(torch.tensor([3., -4.]))
            model.critic_mlp[0].weight.copy_(torch.tensor([[2., 0.], [0., -1.]]))
            model.value_head.weight.copy_(torch.tensor([[2., -1.]]))
            model.value_head.bias.fill_(0.25)
        output = model(torch.tensor([[-1., 0.5]], dtype=torch.float64))
        actor_hidden = math.exp(-1) - 1
        expected_mu = torch.tensor([[actor_hidden + 5, -actor_hidden - 3.5]], dtype=torch.float64)
        expected_value = 2 * (math.exp(-2) - 1) - (math.exp(-0.5) - 1) + 0.25
        torch.testing.assert_close(output.means, expected_mu, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(output.values.item(), expected_value, places=12)
        self.assertGreater(output.means[0, 0].item(), 1)  # No tanh or clipping.

    def test_deterministic_actions_do_not_sample(self):
        model = ActorCritic()
        obs = torch.randn(4, 104)
        state = torch.get_rng_state().clone()
        action = model.deterministic_actions(obs)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        torch.testing.assert_close(action, model.actor(obs), rtol=0, atol=0)
        self.assertFalse(action.requires_grad)

    def test_seeded_construction_and_sampling(self):
        obs = torch.ones(3, 104)
        torch.manual_seed(73)
        first = ActorCritic()
        first_sample = first.sample_actions(obs)
        torch.manual_seed(73)
        second = ActorCritic()
        second_sample = second.sample_actions(obs)
        for left, right in zip(first_sample, second_sample):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_sampling_matches_normal_sample_and_retains_raw_actions(self):
        model = ActorCritic()
        obs = torch.zeros(3, 104)
        with torch.no_grad():
            model.mean_head.bias.fill_(5.)
        torch.manual_seed(11)
        expected = model.distribution(obs).sample()
        torch.manual_seed(11)
        actual = model.sample_actions(obs)
        torch.testing.assert_close(actual.actions, expected, rtol=0, atol=0)
        self.assertTrue((actual.actions > 1).any())
        self.assertFalse(actual.actions.requires_grad)
        self.assertFalse(actual.log_prob.requires_grad)
        torch.testing.assert_close(actual.log_prob, model.log_prob(obs, actual.actions))

    def test_policy_backpropagation_leaves_critic_untouched(self):
        model = ActorCritic()
        obs = torch.randn(5, 104)
        actions = torch.randn(5, 18) + 2
        (-model.log_prob(obs, actions).mean()).backward()
        for parameter in list(model.actor_mlp.parameters()) + list(model.mean_head.parameters()) + [model.log_std]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(model.log_std.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.critic_mlp.parameters()))
        self.assertTrue(all(p.grad is None for p in model.value_head.parameters()))

    def test_critic_backpropagation_leaves_policy_untouched(self):
        model = ActorCritic()
        model.critic(torch.randn(5, 104)).square().mean().backward()
        for parameter in list(model.critic_mlp.parameters()) + list(model.value_head.parameters()):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertTrue(all(p.grad is None for p in model.actor_mlp.parameters()))
        self.assertTrue(all(p.grad is None for p in model.mean_head.parameters()))
        self.assertIsNone(model.log_std.grad)

    def test_evaluate_actions_consistency(self):
        model = ActorCritic()
        obs, actions = torch.randn(1, 104), torch.randn(1, 18)
        result = model.evaluate_actions(obs, actions)
        self.assertEqual(result.log_prob.shape, (1,))
        torch.testing.assert_close(result.log_prob, model.log_prob(obs, actions))
        torch.testing.assert_close(result.entropy, model.entropy(obs))
        torch.testing.assert_close(result.values, model.critic(obs))

    def test_reject_bad_input_shapes(self):
        model = ActorCritic()
        for shape in ((104,), (2, 103), (2, 1, 104)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.zeros(shape))
        with self.assertRaises(ValueError):
            model.log_prob(torch.zeros(2, 104), torch.zeros(18))

    def test_native_state_dict_round_trip(self):
        first, second = ActorCritic(), ActorCritic()
        with torch.no_grad():
            first.log_std.fill_(-0.7)
        second.load_state_dict(first.state_dict())
        obs = torch.randn(3, 104)
        for left, right in zip(first(obs), second(obs)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_import_and_forward_with_legacy_imports_blocked(self):
        code = '''
import importlib.abc, sys
class BlockLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'isaacgym', 'isaacgymenvs', 'rl_games'}:
            raise AssertionError('Forbidden dependency: ' + fullname)
sys.meta_path.insert(0, BlockLegacy())
import torch
from hexapod_rl.networks import ActorCritic
assert ActorCritic()(torch.zeros(1, 104)).values.shape == (1, 1)
'''
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        result = subprocess.run([sys.executable, "-B", "-c", code], env=environment,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed; numerical validation is blocked")
class GaussianTests(unittest.TestCase):
    def test_standard_normal_18_dimensional_fixture(self):
        means = torch.zeros(1, 18, dtype=torch.float64)
        log_std = torch.zeros(18, dtype=torch.float64)
        self.assertAlmostEqual(gaussian_log_prob(means, means, log_std).item(), -9 * math.log(2 * math.pi), places=12)
        self.assertAlmostEqual(diagonal_gaussian(means, log_std).entropy().item(), 9 * (1 + math.log(2 * math.pi)), places=12)

    def test_nonunit_gaussian_matches_scalar_reference_equation(self):
        means = torch.tensor([[1., -2.], [0.5, 4.]], dtype=torch.float64)
        actions = torch.tensor([[3., -1.5], [-0.5, 3.]], dtype=torch.float64)
        log_std = torch.tensor([math.log(2.), math.log(0.5)], dtype=torch.float64)
        expected = []
        for x, mu in zip(actions.tolist(), means.tolist()):
            neglogp = 0.5 * sum(((a - m) / s) ** 2 for a, m, s in zip(x, mu, (2., 0.5)))
            neglogp += math.log(2 * math.pi) + math.log(2.) + math.log(0.5)
            expected.append(-neglogp)
        actual = gaussian_log_prob(actions, means, log_std)
        torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual, diagonal_gaussian(means, log_std).log_prob(actions), rtol=1e-12, atol=1e-12)

    def test_log_probability_analytic_gradients(self):
        means = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
        log_std = torch.tensor([math.log(2.), math.log(0.5)], dtype=torch.float64, requires_grad=True)
        gaussian_log_prob(torch.tensor([[1., -2.]], dtype=torch.float64), means, log_std).sum().backward()
        torch.testing.assert_close(means.grad, torch.tensor([[0.25, -8.]], dtype=torch.float64))
        torch.testing.assert_close(log_std.grad, torch.tensor([-0.75, 15.], dtype=torch.float64))

    def test_entropy_analytic_gradient(self):
        log_std = torch.tensor([0.3, -0.8], dtype=torch.float64, requires_grad=True)
        diagonal_gaussian(torch.zeros(1, 2, dtype=torch.float64), log_std).entropy().sum().backward()
        torch.testing.assert_close(log_std.grad, torch.ones(2, dtype=torch.float64))

    def test_log_probability_finite_difference_gradcheck(self):
        means = torch.tensor([[0.1, -0.3]], dtype=torch.float64, requires_grad=True)
        log_std = torch.tensor([-0.2, 0.4], dtype=torch.float64, requires_grad=True)
        actions = torch.tensor([[1., 2.]], dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(lambda mu, ls: gaussian_log_prob(actions, mu, ls), (means, log_std)))

    def test_representative_inputs_are_finite(self):
        means = torch.tensor([[0., 10.], [-10., 2.]], dtype=torch.float64)
        log_std = torch.tensor([-5., 3.], dtype=torch.float64)
        result = gaussian_log_prob(torch.full_like(means, 100.), means, log_std)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.isfinite(diagonal_gaussian(means, log_std).entropy()).all())


if __name__ == "__main__":
    unittest.main()
