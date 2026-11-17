from dataclasses import replace
import importlib.abc
import io
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from integration_test_support import HAS_TORCH, ROOT, MockInstalledTask, mock_adapter, small_ppo

if HAS_TORCH:
    import torch
    from hexapod_rl.checkpoints import load_checkpoint, save_checkpoint
    from hexapod_rl.checkpoint_format import decode_checkpoint, encode_checkpoint
    from hexapod_rl.collector import EpisodeStatistics, RolloutCollector
    from hexapod_rl.evaluation import evaluate
    from hexapod_rl.integration.configuration import TrainingConfig
    from hexapod_rl.integration.isaacgym_adapter import IsaacGymAdapter, close_environment
    from hexapod_rl.trainer import Trainer, TrainingLogger


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class AdapterCollectorTests(unittest.TestCase):
    def test_initial_ones_and_reset_returns_existing_observations(self):
        task = MockInstalledTask()
        initial = task.observations.clone()
        collector = RolloutCollector(small_ppo(), mock_adapter(task))
        self.assertEqual(task.reset_calls, 1)
        self.assertEqual(task.step_calls, 0)
        self.assertTrue(collector.dones.all())
        torch.testing.assert_close(collector.observations, initial, rtol=0, atol=0)
        self.assertEqual(collector.observations.data_ptr(), task.observations.data_ptr())

    def test_action_clipping_and_storage_ownership(self):
        task = MockInstalledTask()
        adapter = mock_adapter(task)
        actions = torch.tensor([[5., -.2], [-5., .4]])
        saved = actions.clone()
        adapter.step(actions)
        torch.testing.assert_close(actions, saved, rtol=0, atol=0)
        torch.testing.assert_close(task.received_actions[0], saved.clamp(-1, 1))
        self.assertEqual(task.received_actions[0].shape, (2, 2))

    def test_adapter_rejects_observation_shape_dtype_and_device(self):
        adapter = mock_adapter()
        for obs in (torch.zeros(2, 4), torch.zeros(2, 3, dtype=torch.float64), torch.zeros(2, 3, device="meta")):
            with self.subTest(shape=obs.shape, device=obs.device), self.assertRaises(ValueError):
                adapter.observations({"obs": obs})
        with self.assertRaises(ValueError):
            adapter.observations({})

    def test_environment_bounds_and_dimensions_checked(self):
        task = MockInstalledTask()
        with self.assertRaisesRegex(ValueError, "dimensions"):
            IsaacGymAdapter(task, "cpu")
        task.action_space.high = [2., 2.]
        with self.assertRaisesRegex(ValueError, "bounds"):
            mock_adapter(task)

    def test_timeout_information_is_required(self):
        task = MockInstalledTask()
        adapter = mock_adapter(task)
        with patch.object(task, "step", return_value=({"obs": task.observations}, torch.zeros(2), torch.zeros(2, dtype=torch.long), {})):
            with self.assertRaisesRegex(ValueError, "time_outs"):
                adapter.step(torch.zeros(2, 2))

    def test_adapter_cleanup_is_idempotent(self):
        task = MockInstalledTask()
        adapter = mock_adapter(task)
        adapter.close()
        adapter.close()
        self.assertEqual(task.close_calls, 1)

    def test_native_cleanup_when_vectask_has_no_close(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        gym, writer = Mock(), Mock()
        env = SimpleNamespace(gym=gym, viewer="viewer", sim="sim", virtual_display=None)
        close_environment(env, writer)
        gym.destroy_viewer.assert_called_once_with("viewer")
        gym.destroy_sim.assert_called_once_with("sim")
        writer.close.assert_called_once()

    def test_raw_samples_and_log_probs_survive_mutating_steps(self):
        algorithm = small_ppo()
        with torch.no_grad():
            algorithm.model.mean_head.weight.zero_()
            algorithm.model.mean_head.bias.fill_(4)
        collector = RolloutCollector(algorithm, mock_adapter())
        buffer = collector.collect()
        self.assertTrue((buffer.data["actions"] > 1).any())
        torch.testing.assert_close(buffer.data["observations"][0], torch.tensor([[.2, 0, 0], [-.3, 0, 0]]))
        for t in range(buffer.horizon_length):
            normalized = algorithm.observation_normalizer(buffer.data["observations"][t])
            expected = algorithm.model.log_prob(normalized, buffer.data["actions"][t])
            torch.testing.assert_close(buffer.data["log_probs"][t], expected)

    def test_persistent_collector_state_and_deferred_resets(self):
        algorithm = small_ppo(horizon_length=2, minibatch_size=4)
        task = MockInstalledTask()
        collector = RolloutCollector(algorithm, mock_adapter(task))
        first = collector.collect()
        self.assertTrue(first.data["next_dones"][-1, 0])
        self.assertEqual(task.reset_count, 0)  # reset waits for the NEXT step
        last_obs, last_dones = collector.observations.clone(), collector.dones.clone()
        second = collector.collect()
        torch.testing.assert_close(second.data["observations"][0], last_obs)
        torch.testing.assert_close(second.data["dones"][0], last_dones)
        self.assertEqual(task.reset_calls, 1)
        self.assertGreater(task.reset_count, 0)

    def test_collector_timeout_uses_stage2_reward_adjustment(self):
        algorithm = small_ppo(normalize_input=False, normalize_value=False, normalize_advantage=False, gamma=.9)
        with torch.no_grad():
            algorithm.model.value_head.weight.zero_()
            algorithm.model.value_head.bias.fill_(2)
        buffer = RolloutCollector(algorithm, mock_adapter()).collect()
        self.assertTrue(buffer.data["timeouts"][2, 1])
        reward = buffer.data["rewards"][2, 1, 0].item()
        batch = algorithm.prepare_batch(buffer)
        # Environment-major index env1*time4 + time2. Done masks all next values.
        self.assertAlmostEqual(batch.data["returns"][6, 0].item(), reward + .9 * 2, places=5)

    def test_episode_meter_raw_rewards_lengths_and_capped_batch_weights(self):
        meter = EpisodeStatistics(2, "cpu", capacity=3)
        meter.update(torch.tensor([1., 2.]), torch.tensor([False, False]))
        meter.update(torch.tensor([3., 4.]), torch.tensor([True, False]))
        self.assertEqual(meter.summary()["episode_reward"], 4)
        self.assertEqual(meter.summary()["episode_length"], 2)
        meter.update(torch.tensor([10., 2.]), torch.tensor([True, True]))
        self.assertAlmostEqual(meter.summary()["episode_reward"], (4 + 10 + 8) / 3, places=6)
        meter.update(torch.tensor([2., 4.]), torch.tensor([True, True]))
        self.assertAlmostEqual(meter.summary()["episode_reward"], ((22 / 3) + 2 + 4) / 3, places=6)
        self.assertEqual(meter.completed, 5)


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class TrainerCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)

    def test_end_to_end_multiple_epochs_save_restore_continue_and_evaluate(self):
        algorithm = small_ppo(mini_epochs=1)
        before = [p.detach().clone() for p in algorithm.model.parameters()]
        task = MockInstalledTask()
        control = TrainingConfig(max_epochs=2, save_best_after=0, print_stats=False)
        trainer = Trainer(algorithm, mock_adapter(task), self.output, control)
        history = trainer.run()
        self.assertEqual([item["frames"] for item in history], [8, 16])
        self.assertEqual(task.train_frames, [0, 8])
        self.assertEqual(task.reset_calls, 1)
        self.assertEqual(task.close_calls, 1)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, algorithm.model.parameters())))
        self.assertTrue(all(torch.isfinite(p).all() for p in algorithm.model.parameters()))
        checkpoint = self.output / "checkpoints/last.pt"
        self.assertTrue(checkpoint.is_file())
        restored = small_ppo(mini_epochs=1)
        fresh_task = MockInstalledTask()
        resumed = Trainer(restored, mock_adapter(fresh_task), self.output, replace(control, max_epochs=3))
        resumed.resume(checkpoint)
        self.assertEqual((resumed.epoch, resumed.frames, resumed.best_reward), (trainer.epoch, trainer.frames, trainer.best_reward))
        self.assertEqual(fresh_task.reset_calls, 0)
        for a, b in zip(algorithm.model.parameters(), restored.model.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(algorithm.observation_normalizer.count, restored.observation_normalizer.count)
        torch.testing.assert_close(algorithm.value_normalizer.running_var, restored.value_normalizer.running_var)
        self.assertEqual(restored.learning_rate, algorithm.learning_rate)
        result = resumed.run()
        self.assertEqual(result[0]["epoch"], 3)
        self.assertEqual(result[0]["frames"], 24)
        self.assertEqual(fresh_task.train_frames, [16])
        evaluated = small_ppo(mini_epochs=1)
        load_checkpoint(checkpoint, evaluated, resume=False)
        report = evaluate(evaluated, mock_adapter(), steps=5)
        self.assertGreater(report["completed_episodes"], 0)
        self.assertEqual(report["frames"], 10)

    def test_checkpoint_restores_adam_moments_and_rng(self):
        algorithm = small_ppo(mini_epochs=1)
        algorithm.update(RolloutCollector(algorithm, mock_adapter()).collect())
        file = self.output / "state.pt"
        counters = dict(epoch=1, frames=8, best_reward=2.)
        save_checkpoint(file, algorithm, counters, TrainingConfig())
        expected_random, expected_tensor = random.random(), torch.rand(3)
        restored = small_ppo(mini_epochs=1)
        returned = load_checkpoint(file, restored)
        self.assertEqual(returned, counters)
        self.assertEqual(random.random(), expected_random)
        torch.testing.assert_close(torch.rand(3), expected_tensor, rtol=0, atol=0)
        for old, new in zip(algorithm.model.parameters(), restored.model.parameters()):
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(algorithm.optimizer.state[old][key], restored.optimizer.state[new][key], rtol=0, atol=0)
        self.assertEqual(algorithm.scaler.state_dict(), restored.scaler.state_dict())

    def test_checkpoint_rejects_architecture_and_ppo_mismatch_before_loading(self):
        algorithm = small_ppo()
        file = self.output / "state.pt"
        save_checkpoint(file, algorithm, dict(epoch=0, frames=0, best_reward=-100500.), TrainingConfig())
        altered = small_ppo(gamma=.8)
        original = [p.detach().clone() for p in altered.model.parameters()]
        with self.assertRaisesRegex(ValueError, "identical PPO"):
            load_checkpoint(file, altered)
        for a, b in zip(original, altered.model.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        state = decode_checkpoint(torch.load(file, map_location="cpu", weights_only=True), torch.Tensor)
        state["network_config"]["action_dim"] = 12
        torch.save(encode_checkpoint(state, torch.Tensor), file, pickle_protocol=2)
        with self.assertRaisesRegex(ValueError, "dimensions/architecture"):
            load_checkpoint(file, algorithm)

    def test_missing_legacy_and_normalization_incompatible_checkpoints(self):
        algorithm = small_ppo()
        file = self.output / "state.pt"
        with self.assertRaises(FileNotFoundError):
            load_checkpoint(file, algorithm)
        torch.save({"model": {}}, file)
        with self.assertRaisesRegex(ValueError, "RL-Games"):
            load_checkpoint(file, algorithm)
        save_checkpoint(file, algorithm, dict(epoch=0, frames=0, best_reward=0.), TrainingConfig())
        with self.assertRaisesRegex(ValueError, "normalization"):
            load_checkpoint(file, small_ppo(normalize_input=False), resume=False)

    def test_save_best_and_win_threshold_are_gated_by_epoch(self):
        trainer = Trainer(small_ppo(mini_epochs=1), mock_adapter(), self.output,
            TrainingConfig(max_epochs=5, save_best_after=2, score_to_win=-1, print_stats=False))
        history = trainer.run()
        self.assertEqual(len(history), 2)
        self.assertTrue((self.output / "checkpoints/best.pt").is_file())
        self.assertTrue((self.output / "checkpoints/winning.pt").is_file())
        self.assertTrue((self.output / "checkpoints/last.pt").is_file())

    def test_periodic_saves_only_when_reward_not_above_record(self):
        trainer = Trainer(small_ppo(mini_epochs=1), mock_adapter(), self.output,
            TrainingConfig(max_epochs=2, save_frequency=1, save_best_after=50, print_stats=False))
        trainer.best_reward = 1e9
        trainer.run()
        self.assertTrue((self.output / "checkpoints/epoch_000001.pt").is_file())
        self.assertTrue((self.output / "checkpoints/epoch_000002.pt").is_file())
        self.assertFalse((self.output / "checkpoints/best.pt").exists())

    def test_no_completed_episodes_still_saves_at_epoch_limit(self):
        algorithm = small_ppo(horizon_length=1, minibatch_size=2, mini_epochs=1)
        trainer = Trainer(algorithm, mock_adapter(), self.output,
                          TrainingConfig(max_epochs=1, print_stats=False))
        history = trainer.run()
        self.assertIsNone(history[0]["episode_reward"])
        self.assertTrue((self.output / "checkpoints/last.pt").is_file())

    def test_cleanup_on_optimization_error_and_no_partial_epoch_count(self):
        algorithm, task = small_ppo(), MockInstalledTask()
        trainer = Trainer(algorithm, mock_adapter(task), self.output, TrainingConfig(print_stats=False))
        with patch.object(algorithm, "update", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                trainer.run()
        self.assertEqual(task.close_calls, 1)
        self.assertEqual((trainer.epoch, trainer.frames), (0, 0))

    def test_resume_at_epoch_limit_does_not_collect(self):
        task = MockInstalledTask()
        trainer = Trainer(small_ppo(), mock_adapter(task), self.output, TrainingConfig(max_epochs=1, print_stats=False))
        trainer.epoch = 1
        self.assertEqual(trainer.run(), [])
        self.assertEqual(task.reset_calls, 0)
        self.assertEqual(task.step_calls, 0)
        self.assertEqual(task.close_calls, 1)

    def test_evaluation_uses_normalized_means_and_freezes_learning_state(self):
        algorithm = small_ppo()
        algorithm.observation_normalizer.running_mean.fill_(1)
        algorithm.observation_normalizer.running_var.fill_(4)
        algorithm.observation_normalizer.eval()
        task = MockInstalledTask()
        expected = algorithm.model.deterministic_actions(algorithm.observation_normalizer(task.observations)).clamp(-1, 1)
        parameters = [p.detach().clone() for p in algorithm.model.parameters()]
        stats = {k: v.clone() for k, v in algorithm.observation_normalizer.state_dict().items()}
        rng = torch.get_rng_state().clone()
        with patch.object(algorithm, "update", side_effect=AssertionError("optimization during evaluation")):
            report = evaluate(algorithm, mock_adapter(task), steps=4)
        torch.testing.assert_close(task.received_actions[0], expected)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for before, after in zip(parameters, algorithm.model.parameters()):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            self.assertIsNone(after.grad)
        for key, value in algorithm.observation_normalizer.state_dict().items():
            torch.testing.assert_close(value, stats[key], rtol=0, atol=0)
        self.assertEqual(task.close_calls, 1)
        self.assertEqual(report["frames"], 8)

    def test_production_imports_and_mock_training_block_simulator_and_rlgames(self):
        script = f"""
import sys, tempfile, importlib.abc
sys.path[:0] = [{str(ROOT / 'src')!r}, {str(ROOT / 'tests')!r}]
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'isaacgym', 'isaacgymenvs', 'rl_games'}}:
            raise AssertionError('forbidden dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from integration_test_support import small_ppo, mock_adapter
from hexapod_rl.trainer import Trainer
from hexapod_rl.integration.configuration import TrainingConfig
with tempfile.TemporaryDirectory() as directory:
    history = Trainer(small_ppo(mini_epochs=1), mock_adapter(), directory,
                      TrainingConfig(max_epochs=2, print_stats=False)).run()
    assert len(history) == 2 and history[-1]['frames'] == 16
"""
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
