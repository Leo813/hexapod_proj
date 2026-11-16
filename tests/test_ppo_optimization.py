import math
import unittest
from unittest.mock import patch

from ppo_test_support import HAS_TORCH, small_ppo, collect_mock, MockVecTask

if HAS_TORCH:
    import torch
    from hexapod_rl.config import ActorCriticConfig, PPOConfig
    from hexapod_rl.networks import ActorCritic
    from hexapod_rl.distributions import gaussian_log_prob
    from hexapod_rl.ppo import PPO
    from hexapod_rl.rollout import RolloutBuffer, TrainingBatch
    from hexapod_rl.schedules import policy_kl


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class OptimizationTests(unittest.TestCase):
    def scalar_problem(self, **overrides):
        model = ActorCritic(ActorCriticConfig(observation_dim=1, action_dim=1, hidden_sizes=()))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        settings = dict(num_actors=2, horizon_length=1, minibatch_size=2, mini_epochs=1,
                        normalize_input=False, normalize_value=False, normalize_advantage=False,
                        lr_schedule="constant", truncate_grads=False)
        settings.update(overrides)
        algorithm = PPO(model, PPOConfig(**settings))
        actions, means = torch.ones(2, 1), torch.zeros(2, 1)
        batch = TrainingBatch(dict(observations=torch.zeros(2, 1), actions=actions,
            old_log_probs=gaussian_log_prob(actions, means, means), old_values=torch.zeros(2, 1),
            returns=torch.ones(2, 1), advantages=torch.ones(2), means=means, stds=torch.ones(2, 1)))
        return algorithm, batch

    def test_adam_settings_and_analytical_first_update(self):
        algorithm, batch = self.scalar_problem()
        model = algorithm.model
        algorithm.optimize(batch)
        defaults = algorithm.optimizer.defaults
        self.assertEqual(defaults["betas"], (.9, .999))
        self.assertEqual(defaults["eps"], 1e-8)
        self.assertEqual(defaults["weight_decay"], 0)
        self.assertFalse(defaults["amsgrad"])
        # dL/d mean_bias=-1; dL/d value_bias=-2; Adam's first bias-corrected step.
        self.assertAlmostEqual(model.mean_head.bias.grad.item(), -1)
        self.assertAlmostEqual(model.value_head.bias.grad.item(), -2)
        self.assertAlmostEqual(model.mean_head.bias.item(), 3e-4 / (1 + 1e-8), places=9)
        self.assertAlmostEqual(model.value_head.bias.item(), 3e-4 * 2 / (2 + 1e-8), places=9)
        self.assertEqual(algorithm.optimizer.state[model.value_head.bias]["step"].item(), 1)

    def test_gradient_unscale_clip_step_update_order(self):
        algorithm, batch = self.scalar_problem(truncate_grads=True, grad_norm=.02)
        events = []
        def record(name, original):
            def call(*args, **kwargs):
                events.append(name)
                return original(*args, **kwargs)
            return call
        scaler = algorithm.scaler
        with patch.object(scaler, "scale", side_effect=record("scale", scaler.scale)), \
             patch.object(scaler, "unscale_", side_effect=record("unscale", scaler.unscale_)), \
             patch.object(scaler, "step", side_effect=record("step", scaler.step)), \
             patch.object(scaler, "update", side_effect=record("update", scaler.update)), \
             patch("torch.nn.utils.clip_grad_norm_", side_effect=record("clip", torch.nn.utils.clip_grad_norm_)):
            metrics = algorithm.optimize(batch)
        self.assertEqual(events, ["scale", "unscale", "clip", "step", "update"])
        norm = math.sqrt(sum(p.grad.square().sum().item() for p in algorithm.model.parameters() if p.grad is not None))
        self.assertLessEqual(norm, .020001)
        self.assertGreater(metrics[0]["grad_norm"], .02)

    def test_cpu_disables_requested_cuda_amp(self):
        algorithm, batch = self.scalar_problem(mixed_precision=True)
        self.assertFalse(algorithm.amp_enabled)
        self.assertFalse(algorithm.scaler.is_enabled())
        self.assertTrue(math.isfinite(algorithm.optimize(batch)[0]["total"]))

    def test_kl_uses_pre_update_forward_and_refreshes_each_pass(self):
        algorithm, batch = self.scalar_problem(mini_epochs=2, lr_schedule="adaptive")
        batch.data["means"].fill_(2)
        original_log_probs = batch.data["old_log_probs"].clone()
        snapshots = []
        def capture(module, args, output):
            snapshots.append((output.means.detach().clone(), output.log_std.exp().detach().clone()))
        hook = algorithm.model.register_forward_hook(capture)
        try:
            metrics = algorithm.optimize(batch)
        finally:
            hook.remove()
        mean0, std0 = snapshots[0]
        mean1, std1 = snapshots[1]
        expected0 = policy_kl(mean0, std0, torch.full_like(mean0, 2), torch.ones_like(std0))
        expected1 = policy_kl(mean1, std1, mean0, std0)
        self.assertAlmostEqual(metrics[0]["kl"], expected0.item())
        self.assertAlmostEqual(metrics[1]["kl"], expected1.item())
        self.assertAlmostEqual(metrics[0]["next_learning_rate"], 2e-4)
        self.assertAlmostEqual(metrics[1]["learning_rate"], 2e-4)
        self.assertAlmostEqual(metrics[1]["next_learning_rate"], 3e-4)
        torch.testing.assert_close(batch.data["means"], mean1, rtol=0, atol=0)
        torch.testing.assert_close(batch.data["stds"], std1, rtol=0, atol=0)
        torch.testing.assert_close(batch.data["old_log_probs"], original_log_probs, rtol=0, atol=0)

    def test_prepare_values_then_returns_uses_two_distinct_statistics(self):
        algorithm = small_ppo(horizon_length=1, minibatch_size=2, gamma=0, normalize_advantage=False)
        buffer = RolloutBuffer(1, 2, 3, 2)
        values = torch.tensor([[1.], [3.]])
        buffer.store_policy_step(torch.zeros(2, 3), torch.zeros(2), torch.zeros(2, 2),
            torch.zeros(2), values, torch.zeros(2, 2), torch.ones(2, 2))
        buffer.store_outcome(torch.tensor([5., 7.]), torch.ones(2))
        buffer.finish(torch.zeros(2, 1))
        batch = algorithm.prepare_batch(buffer)
        expected_old = torch.tensor([[(1 - 4/3) / math.sqrt(23/9 + 1e-5)],
                                     [(3 - 4/3) / math.sqrt(23/9 + 1e-5)]])
        expected_returns = torch.tensor([[(5 - 3.2) / math.sqrt(7.56 + 1e-5)],
                                         [(7 - 3.2) / math.sqrt(7.56 + 1e-5)]])
        torch.testing.assert_close(batch.data["old_values"], expected_old)
        torch.testing.assert_close(batch.data["returns"], expected_returns)
        torch.testing.assert_close(batch.data["advantages"], torch.full((2,), 4.))
        self.assertEqual(algorithm.value_normalizer.count.item(), 5)
        self.assertFalse(algorithm.value_normalizer.training)

    def test_rollout_freezes_statistics_and_returns_denormalized_values(self):
        algorithm = small_ppo()
        algorithm.value_normalizer.running_mean.fill_(3)
        algorithm.value_normalizer.running_var.fill_(4)
        with torch.no_grad():
            algorithm.model.value_head.weight.zero_()
            algorithm.model.value_head.bias.fill_(9)  # clipped to 5 BEFORE unnormalizing
        sample = algorithm.sample(torch.zeros(2, 3))
        expected = torch.full((2, 1), 3 + 5 * math.sqrt(4 + 1e-5))
        torch.testing.assert_close(sample.values, expected)
        torch.testing.assert_close(algorithm.predict_values(torch.zeros(2, 3)), expected)
        self.assertEqual(algorithm.value_normalizer.count.item(), 1)
        self.assertEqual(algorithm.observation_normalizer.count.item(), 1)
        self.assertFalse(sample.actions.requires_grad)

    def test_input_moments_only_first_mini_epoch_and_reset_next_update(self):
        algorithm = small_ppo(mini_epochs=3)
        buffer, env = collect_mock(algorithm)
        batch = algorithm.prepare_batch(buffer)
        self.assertEqual(algorithm.value_normalizer.count.item(), 17)
        metrics = algorithm.optimize(batch)
        self.assertEqual([(m["epoch"], m["start"]) for m in metrics],
                         [(0, 0), (0, 4), (1, 0), (1, 4), (2, 0), (2, 4)])
        self.assertEqual(algorithm.observation_normalizer.count.item(), 9)
        self.assertFalse(algorithm.observation_normalizer.training)
        buffer2, _ = collect_mock(algorithm, env)
        algorithm.update(buffer2)
        self.assertEqual(algorithm.observation_normalizer.count.item(), 17)
        self.assertEqual(algorithm.value_normalizer.count.item(), 33)

    def test_mock_deferred_reset_and_action_mutation(self):
        env = MockVecTask()
        actions = torch.tensor([[1., 8.], [2., 9.]])
        env.step(actions)
        torch.testing.assert_close(actions[:, 1], torch.tensor([-1., -2.]))
        env.step(actions)
        self.assertTrue(env.dones[0])
        self.assertEqual(env.age[0].item(), 2)
        self.assertNotEqual(env.observations[0].abs().sum().item(), 0)
        env.step(actions)
        self.assertEqual(env.observations[0].abs().sum().item(), 0)
        self.assertEqual(env.age[0].item(), 0)
        self.assertTrue(env.timeouts[1])
        self.assertEqual(env.reset_count, 1)

    def test_end_to_end_rollout_and_finite_parameter_updates(self):
        algorithm = small_ppo()
        model = algorithm.model
        before = [p.detach().clone() for p in model.parameters()]
        buffer, env = collect_mock(algorithm)
        self.assertGreater(env.reset_count, 0)
        self.assertTrue(buffer.data["timeouts"].any())
        self.assertTrue(buffer.data["next_dones"].any())
        torch.testing.assert_close(buffer.data["dones"][1:], buffer.data["next_dones"][:-1])
        for t in range(buffer.horizon_length):
            expected = gaussian_log_prob(buffer.data["actions"][t], buffer.data["means"][t], buffer.data["stds"][t].log())
            torch.testing.assert_close(buffer.data["log_probs"][t], expected)
        metrics = algorithm.update(buffer)
        self.assertIs(algorithm.model, model)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        self.assertTrue(all(math.isfinite(v) for row in metrics for v in row.values() if v is not None))

    def test_prepare_does_not_alias_reusable_rollout(self):
        algorithm = small_ppo()
        buffer, _ = collect_mock(algorithm)
        batch = algorithm.prepare_batch(buffer)
        saved = {k: v.clone() for k, v in batch.data.items()}
        buffer.reset()
        for value in buffer.data.values():
            value.zero_()
        for key, value in batch.data.items():
            torch.testing.assert_close(value, saved[key], rtol=0, atol=0)

    def test_gradients_are_cleared_before_each_update(self):
        algorithm, batch = self.scalar_problem()
        for parameter in algorithm.model.parameters():
            parameter.grad = torch.full_like(parameter, 999.)
        algorithm.optimize(batch)
        self.assertAlmostEqual(algorithm.model.mean_head.bias.grad.item(), -1)
        self.assertAlmostEqual(algorithm.model.value_head.bias.grad.item(), -2)

    def test_mismatched_rollout_rejected(self):
        algorithm = small_ppo()
        with self.assertRaises(ValueError):
            algorithm.prepare_batch(RolloutBuffer(2, 2, 3, 2))
