import unittest

from ppo_test_support import HAS_TORCH

if HAS_TORCH:
    import torch
    from hexapod_rl.rollout import RolloutBuffer, TrainingBatch, flatten_time_env


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class RolloutTests(unittest.TestCase):
    def fill(self, horizon=3, envs=2):
        buffer = RolloutBuffer(horizon, envs, 2, 2)
        for t in range(horizon):
            tag = torch.arange(envs).float() * 10 + t
            obs = tag[:, None].expand(-1, 2)
            buffer.store_policy_step(obs, torch.zeros(envs), obs, tag, tag[:, None], obs, torch.ones_like(obs))
            buffer.store_outcome(tag, torch.zeros(envs))
        buffer.finish(torch.zeros(envs, 1))
        return buffer

    def test_shapes_and_environment_major_order(self):
        buffer = self.fill()
        self.assertEqual(buffer.data["observations"].shape, (3, 2, 2))
        self.assertEqual(buffer.data["values"].shape, (3, 2, 1))
        self.assertEqual(buffer.data["dones"].dtype, torch.bool)
        expected = torch.tensor([0., 1., 2., 10., 11., 12.])
        torch.testing.assert_close(buffer.flatten()["observations"][:, 0], expected)
        torch.testing.assert_close(buffer.flatten()["log_probs"], expected)

    def test_policy_copied_before_environment_mutation(self):
        buffer = RolloutBuffer(1, 2, 2, 2)
        observations = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        actions = observations.detach().clone()
        log_probs = torch.tensor([-.7, -.8])
        buffer.store_policy_step(observations, torch.zeros(2), actions, log_probs,
                                 torch.ones(2, 1), actions, torch.ones_like(actions))
        actions[:, 1] = -actions[:, 0]
        with torch.no_grad():
            observations.fill_(999)
        log_probs.fill_(0)
        self.assertFalse(buffer.data["observations"].requires_grad)
        torch.testing.assert_close(buffer.data["actions"][0], torch.tensor([[1., 2.], [3., 4.]]))
        torch.testing.assert_close(buffer.data["observations"][0], torch.tensor([[1., 2.], [3., 4.]]))
        torch.testing.assert_close(buffer.data["log_probs"][0], torch.tensor([-.7, -.8]))

    def test_outcomes_and_bootstrap_are_owned(self):
        buffer = RolloutBuffer(1, 2, 1, 1)
        zeros = torch.zeros(2, 1)
        buffer.store_policy_step(zeros, torch.zeros(2), zeros, torch.zeros(2), zeros, zeros, zeros + 1)
        reward, done, timeout = torch.tensor([1., 2.]), torch.tensor([True, False]), torch.tensor([False, True])
        buffer.store_outcome(reward, done, timeout)
        reward.zero_(); done.zero_(); timeout.zero_()
        value = torch.ones(2, 1)
        buffer.finish(value)
        value.zero_()
        torch.testing.assert_close(buffer.data["rewards"][0, :, 0], torch.tensor([1., 2.]))
        self.assertEqual(buffer.last_dones.tolist(), [True, False])
        self.assertEqual(buffer.data["timeouts"][0].tolist(), [False, True])
        torch.testing.assert_close(buffer.last_values, torch.ones(2, 1))

    def test_flatten_owns_storage_even_for_singleton_axes(self):
        for shape in ((1, 2, 1), (2, 1, 1), (2, 2, 1)):
            source = torch.ones(shape)
            flat = flatten_time_env(source)
            source.zero_()
            torch.testing.assert_close(flat, torch.ones_like(flat))

    def test_lifecycle_and_shape_errors(self):
        buffer = RolloutBuffer(1, 2, 1, 1)
        with self.assertRaises(RuntimeError):
            buffer.finish(torch.zeros(2, 1))
        with self.assertRaises(RuntimeError):
            buffer.flatten()
        with self.assertRaises(RuntimeError):
            buffer.store_outcome(torch.zeros(2), torch.zeros(2))
        x = torch.zeros(2, 1)
        with self.assertRaises(ValueError):
            buffer.store_policy_step(x, torch.zeros(2), x, x, x, x, x)
        buffer.store_policy_step(x, torch.zeros(2), x, torch.zeros(2), x, x, x + 1)
        with self.assertRaises(RuntimeError):
            buffer.store_policy_step(x, torch.zeros(2), x, torch.zeros(2), x, x, x + 1)
        buffer.store_outcome(torch.zeros(2), torch.zeros(2))
        buffer.finish(x)
        buffer.reset()
        self.assertEqual(buffer.position, 0)
        self.assertFalse(buffer.finished)

    def test_different_sizes_cover_every_transition_without_shuffle(self):
        for horizon, envs, minibatch in ((3, 2, 2), (1, 4, 4), (4, 1, 2)):
            flat = self.fill(horizon, envs).flatten()
            batch = TrainingBatch(dict(observations=flat["observations"], actions=flat["actions"],
                old_log_probs=flat["log_probs"], old_values=flat["values"], returns=flat["values"].clone(),
                advantages=flat["rewards"].squeeze(-1), means=flat["means"], stds=flat["stds"]))
            expected = [10 * n + t for n in range(envs) for t in range(horizon)]
            for _ in range(3):
                order = [x.item() for _, data in batch.minibatches(minibatch) for x in data["old_log_probs"]]
                self.assertEqual(order, expected)

    def test_mutable_kl_references_do_not_change_old_log_probs(self):
        flat = self.fill().flatten()
        batch = TrainingBatch(dict(observations=flat["observations"], actions=flat["actions"],
            old_log_probs=flat["log_probs"], old_values=flat["values"], returns=flat["values"].clone(),
            advantages=flat["rewards"].squeeze(-1), means=flat["means"], stds=flat["stds"]))
        before = batch.data["old_log_probs"].clone()
        means = torch.full((2, 2), 8., requires_grad=True)
        batch.update_distribution(slice(0, 2), means, torch.full((2, 2), .3))
        torch.testing.assert_close(batch.data["means"][:2], means.detach())
        torch.testing.assert_close(batch.data["old_log_probs"], before)
        self.assertFalse(batch.data["means"].requires_grad)
        with self.assertRaises(ValueError):
            list(batch.minibatches(4))
