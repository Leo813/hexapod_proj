"""Stage 3.5B regressions; no Isaac Gym import or dynamics model."""

from pathlib import Path
import pickletools
import random
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import Mock, patch

from integration_test_support import HAS_OMEGACONF, HAS_TORCH, make_installation, small_ppo
from ppo_test_support import collect_mock
from hexapod_rl.integration.configuration import (TrainingConfig, load_configuration,
    ineffective_setting_notices, validate_clipping)

if HAS_TORCH:
    import torch
    from hexapod_rl.checkpoints import FORMAT_VERSION, load_checkpoint, save_checkpoint
    from hexapod_rl.checkpoint_format import decode_checkpoint, encode_checkpoint
    from hexapod_rl.collector import RolloutCollector
    from hexapod_rl.config import ActorCriticConfig, PPOConfig
    from hexapod_rl.integration.isaacgym_adapter import IsaacGymAdapter
    from hexapod_rl.networks import ActorCritic
    from hexapod_rl.ppo import PPO, PolicyStep


class ClippingConfigurationTests(unittest.TestCase):
    def test_valid_clipping_domains(self):
        for value in (0, .5, 1, 2, float("inf")):
            with self.subTest(value=value):
                validate_clipping(dict(clipActions=value, clipObservations=value))
        validate_clipping({})  # VecTask's unbounded default

    def test_malformed_clipping_limits(self):
        for name in ("clipActions", "clipObservations"):
            for value in (-1, float("-inf"), float("nan"), True, "1", None, [], {}):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    validate_clipping({name: value})

    def test_ineffective_values_are_not_modified_or_rejected(self):
        task = {"env": {"envSpacing": 3, "urdfAsset": {"fixBaseLink": True},
                        "control": {"controlFrequencyInv": 2}, "baseInitState": {"vLinear": [1, 0, 0]}}}
        notices = ineffective_setting_notices(task)
        self.assertEqual(len(notices), 4)
        self.assertTrue(any("VecTask reads env.controlFrequencyInv" in text for text in notices))
        self.assertEqual(task["env"]["envSpacing"], 3)
        self.assertTrue(task["env"]["urdfAsset"]["fixBaseLink"])


@unittest.skipUnless(HAS_OMEGACONF, "OmegaConf is not installed")
class AuditConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.package = make_installation(self.temporary.name)

    def load(self, *args):
        return load_configuration([f"isaacgymenvs_path={self.package}", *args])

    def update_yaml(self, relative, **values):
        from omegaconf import OmegaConf
        path = self.package / "cfg" / relative
        cfg = OmegaConf.load(path)
        for key, value in values.items():
            OmegaConf.update(cfg, key, value)
        OmegaConf.save(cfg, path)

    def test_application_yaml_then_cli_precedence(self):
        self.update_yaml("config.yaml", output_dir="yaml-run", eval_steps=73, tensorboard=True)
        app = self.load()
        self.assertEqual((app.resolved["output_dir"], app.resolved["eval_steps"], app.resolved["tensorboard"]),
                         ("yaml-run", 73, True))
        app = self.load("output_dir=cli-run", "eval_steps=19", "tensorboard=False")
        self.assertEqual((app.resolved["output_dir"], app.resolved["eval_steps"], app.resolved["tensorboard"]),
                         ("cli-run", 19, False))

    def test_absent_defaults_and_baseline_no_notices(self):
        app = self.load()
        self.assertEqual((app.resolved["output_dir"], app.resolved["eval_steps"], app.resolved["tensorboard"]),
                         ("", 1000, False))
        self.assertFalse(app.resolved["trust_legacy_checkpoint"])
        self.assertEqual(app.notices, ())
        self.assertEqual(app.ppo.reward_scale, 1.)

    def test_reward_scale_alias_must_agree(self):
        self.update_yaml("train/hexapodPPO.yaml", **{"params.config.reward_scale": 2.,
                         "params.config.reward_shaper.scale_value": 2.})
        self.assertEqual(self.load().ppo.reward_scale, 2.)
        with self.assertRaisesRegex(ValueError, "Conflicting reward_scale"):
            self.load("train.params.config.reward_scale=3")

    def test_ineffective_yaml_and_cli_values_produce_notices(self):
        self.update_yaml("task/hexapod.yaml", **{"env.envSpacing": 7., "env.baseInitState.rot": [1., 0., 0., 0.]})
        app = self.load("task.env.learn.linearVelocityXYRewardScale=2", "task.env.randomCommandVelocityRanges.yaw=[0,0]")
        self.assertEqual(len(app.notices), 4)
        self.assertEqual(app.resolved["task"]["env"]["envSpacing"], 7.)
        self.assertTrue(any("commands are overwritten" in text for text in app.notices))

    def test_clipping_overrides_reach_validation(self):
        for value in ("0", "0.5", "1", "2", ".inf"):
            with self.subTest(value=value):
                self.load(f"task.env.clipActions={value}", f"task.env.clipObservations={value}")
        for name in ("clipActions", "clipObservations"):
            for value in ("-1", ".nan", "-.inf", "true", "null", "bad"):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    self.load(f"task.env.{name}={value}")

    def test_legacy_trust_flag_requires_boolean(self):
        self.assertTrue(self.load("trust_legacy_checkpoint=True").resolved["trust_legacy_checkpoint"])
        with self.assertRaisesRegex(ValueError, "trust_legacy_checkpoint"):
            self.load("trust_legacy_checkpoint=1")


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class CheckpointCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.file = Path(self.temporary.name) / "checkpoint.pt"
        self.algorithm = small_ppo()
        self.algorithm.update(collect_mock(self.algorithm)[0])
        self.counters = dict(epoch=3, frames=24, best_reward=1.)
        save_checkpoint(self.file, self.algorithm, self.counters, TrainingConfig())

    def read_state(self):
        return decode_checkpoint(torch.load(self.file, map_location="cpu", weights_only=True), torch.Tensor)

    def write_state(self, state):
        torch.save(encode_checkpoint(state, torch.Tensor), self.file, pickle_protocol=2)

    def test_v3_restricted_load_and_numpy_rng_roundtrip(self):
        # NumPy is installed in CI. Do not silently omit RNG validation if absent.
        import numpy as np
        np.random.seed(917)
        np.random.normal(size=1)  # leave a cached Gaussian to verify all MT19937 fields
        save_checkpoint(self.file, self.algorithm, self.counters, TrainingConfig())
        saved = self.read_state()
        self.assertEqual(saved["format_version"], 3)
        self.assertEqual(FORMAT_VERSION, 3)
        self.assertIsInstance(saved["rng"]["numpy"]["keys"], list)
        expected = (np.random.normal(size=5), np.random.rand(5), random.random(), torch.rand(3))
        with patch("hexapod_rl.checkpoints.torch.load", wraps=torch.load) as loader:
            self.assertEqual(load_checkpoint(self.file, small_ppo()), self.counters)
        self.assertTrue(loader.call_args.kwargs["weights_only"])
        np.testing.assert_array_equal(np.random.normal(size=5), expected[0])
        np.testing.assert_array_equal(np.random.rand(5), expected[1])
        self.assertEqual(random.random(), expected[2])
        torch.testing.assert_close(torch.rand(3), expected[3], rtol=0, atol=0)

    def test_v1_requires_explicit_trust_and_restores_rng(self):
        import numpy as np
        state = self.read_state()
        state["format_version"] = 1
        state["rng"]["numpy"] = np.random.get_state()
        torch.save(state, self.file)
        expected = np.random.rand(4)
        destination = small_ppo()
        before = {k: v.clone() for k, v in destination.model.state_dict().items()}
        with self.assertRaisesRegex(ValueError, "trust_legacy"):
            load_checkpoint(self.file, destination)
        for key, value in destination.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        with self.assertWarnsRegex(RuntimeWarning, "Unrestricted checkpoint"):
            load_checkpoint(self.file, destination, trust_legacy=True)
        np.testing.assert_array_equal(np.random.rand(4), expected)

    def test_restricted_failure_never_retries_unrestricted(self):
        import pickle
        with patch("hexapod_rl.checkpoints.torch.load", side_effect=pickle.UnpicklingError("blocked")) as loader:
            with self.assertRaisesRegex(ValueError, "no unrestricted retry"):
                load_checkpoint(self.file, self.algorithm)
        loader.assert_called_once_with(self.file, map_location="cpu", weights_only=True)

    def test_trainer_and_evaluation_loading_forward_explicit_trust(self):
        from hexapod_rl.trainer import Trainer
        trainer = Trainer(self.algorithm, None, self.temporary.name)
        with patch("hexapod_rl.trainer.load_checkpoint", return_value=self.counters) as loader:
            trainer.resume(self.file, trust_legacy=True)
        loader.assert_called_once_with(self.file, self.algorithm, resume=True, trust_legacy=True)
        # Evaluation loading must remain restricted by default, too.
        with patch("hexapod_rl.checkpoints.torch.load", wraps=torch.load) as loader:
            load_checkpoint(self.file, self.algorithm, resume=False)
        self.assertTrue(loader.call_args.kwargs["weights_only"])

    def test_disabled_to_enabled_scaler_uses_fresh_state(self):
        # Models and tensors stay on CPU. These doubles test transition policy,
        # not CUDA execution, autocast, or the native enabled scaler.
        destination = small_ppo()
        old_scaler = Mock()
        old_scaler.is_enabled.return_value = True
        destination.scaler = old_scaler
        fresh = Mock()
        with patch("hexapod_rl.checkpoints.torch.cuda.amp.GradScaler", return_value=fresh) as factory:
            with self.assertWarnsRegex(RuntimeWarning, "fresh enabled scaler"):
                self.assertEqual(load_checkpoint(self.file, destination), self.counters)
        factory.assert_called_once_with(enabled=True)
        self.assertIs(destination.scaler, fresh)
        old_scaler.load_state_dict.assert_not_called()
        fresh.load_state_dict.assert_not_called()
        for old, new in zip(self.algorithm.model.parameters(), destination.model.parameters()):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(self.algorithm.optimizer.state[old][key],
                                           destination.optimizer.state[new][key], rtol=0, atol=0)
        self.assertEqual(destination.learning_rate, self.algorithm.learning_rate)

    def test_nonempty_enabled_scaler_state_is_restored(self):
        state = self.read_state()
        scaler_state = dict(scale=2048., growth_factor=2., backoff_factor=.5,
                            growth_interval=2000, _growth_tracker=7)
        state["scaler"] = scaler_state
        self.write_state(state)
        destination = small_ppo()
        destination.scaler = Mock()
        destination.scaler.is_enabled.return_value = True
        with patch("hexapod_rl.checkpoints.torch.cuda.amp.GradScaler") as factory:
            load_checkpoint(self.file, destination)
        factory.assert_not_called()
        destination.scaler.load_state_dict.assert_called_once_with(scaler_state)

    def test_disabled_scaler_and_evaluation_do_not_create_enabled_scaler(self):
        destination = small_ppo()
        with patch("hexapod_rl.checkpoints.torch.cuda.amp.GradScaler") as factory:
            load_checkpoint(self.file, destination)
            destination.scaler = Mock()
            destination.scaler.is_enabled.return_value = True
            load_checkpoint(self.file, destination, resume=False)
        factory.assert_not_called()
        destination.scaler.load_state_dict.assert_not_called()

    def test_invalid_learning_rate_fails_before_model_restore(self):
        state = self.read_state()
        state["learning_rate"] *= 2
        self.write_state(state)
        with patch.object(self.algorithm.model, "load_state_dict") as restore:
            with self.assertRaisesRegex(ValueError, "learning rate"):
                load_checkpoint(self.file, self.algorithm)
        restore.assert_not_called()

    def test_complete_archive_uses_historical_restricted_opcodes(self):
        # Inspect the actual torch.save payload, including tensor reconstruction.
        # List from PyTorch v1.13.1 torch/_weights_only_unpickler.py.
        supported = set("APPEND APPENDS BINGET BININT BININT1 BININT2 BINPERSID BINPUT "
            "BINUNICODE BUILD EMPTY_DICT EMPTY_LIST EMPTY_SET EMPTY_TUPLE GLOBAL LONG1 "
            "LONG_BINGET LONG_BINPUT MARK NEWFALSE NEWOBJ NEWTRUE NONE PROTO REDUCE "
            "SETITEM SETITEMS SHORT_BINSTRING STOP TUPLE TUPLE1 TUPLE2 TUPLE3".split())
        state = self.read_state()
        state["scaler"] = dict(scale=65536., growth_factor=2., backoff_factor=.5,
                               growth_interval=2000, _growth_tracker=7)
        state["resolved_config"] = {"clip": float("inf"), "negative": float("-inf"),
            "unset": None, "flags": [True, False], "nested": (0.1, -0., 2 ** 100)}
        state["rng"]["python"] = (state["rng"]["python"][0], state["rng"]["python"][1], .125)
        self.write_state(state)
        with zipfile.ZipFile(self.file) as archive:
            name, = [n for n in archive.namelist() if n.endswith("/data.pkl")]
            ops = list(pickletools.genops(archive.read(name)))
        self.assertLessEqual({op.name for op, _, _ in ops}, supported)
        self.assertEqual([(op.name, arg) for op, arg, _ in ops if op.name == "PROTO"], [("PROTO", 2)])
        self.assertNotIn("BINFLOAT", {op.name for op, _, _ in ops})
        restored = self.read_state()  # real restricted loader, not a mock
        self.assertEqual(restored["scaler"], state["scaler"])
        self.assertEqual(restored["resolved_config"], state["resolved_config"])
        self.assertEqual(restored["rng"]["python"], state["rng"]["python"])
        self.assertEqual(restored["model"]._metadata, state["model"]._metadata)
        for name in ("observation_normalizer", "value_normalizer"):
            for key, value in state["normalizers"][name].items():
                torch.testing.assert_close(restored["normalizers"][name][key], value, rtol=0, atol=0)

    def test_v2_explicit_trust_migrates_to_restricted_v3(self):
        state = self.read_state()
        state["format_version"] = 2
        torch.save(state, self.file, pickle_protocol=2)
        destination = small_ppo()
        if torch.__version__.startswith("1.13.1"):
            with self.assertRaisesRegex(ValueError, "no unrestricted retry"):
                load_checkpoint(self.file, destination)
        with self.assertWarnsRegex(RuntimeWarning, "Unrestricted checkpoint"):
            self.assertEqual(load_checkpoint(self.file, destination, trust_legacy=True), self.counters)
        for old, new in zip(self.algorithm.model.parameters(), destination.model.parameters()):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(self.algorithm.optimizer.state[old][key],
                                           destination.optimizer.state[new][key], rtol=0, atol=0)
        save_checkpoint(self.file, destination, self.counters, TrainingConfig())
        self.assertEqual(self.read_state()["format_version"], 3)
        load_checkpoint(self.file, small_ppo())

    def test_old_torch_rejects_untrusted_before_deserialization(self):
        with patch("hexapod_rl.checkpoints._HAS_WEIGHTS_ONLY", False):
            with patch("hexapod_rl.checkpoints.torch.load") as loader:
                for resume in (True, False):
                    with self.subTest(resume=resume), self.assertRaisesRegex(ValueError, "no deserialization attempted"):
                        load_checkpoint(self.file, self.algorithm, resume=resume)
        loader.assert_not_called()

    def test_old_torch_trusted_load_omits_weights_only_for_all_versions(self):
        import numpy as np
        state = self.read_state()
        native_loader = torch.load

        # Enforce the old call signature while using installed torch for tensors.
        # This does not claim execution under native PyTorch 1.10.
        def old_loader(path, map_location=None):
            return native_loader(path, map_location=map_location, weights_only=False)

        for version in (1, 2, 3):
            fixture = dict(state, format_version=version)
            if version == 1:
                fixture["rng"] = dict(state["rng"], numpy=np.random.get_state())
            torch.save(encode_checkpoint(fixture, torch.Tensor) if version == 3 else fixture,
                       self.file, pickle_protocol=2)
            with self.subTest(version=version), patch("hexapod_rl.checkpoints._HAS_WEIGHTS_ONLY", False):
                with patch("hexapod_rl.checkpoints.torch.load", side_effect=old_loader) as loader:
                    with self.assertWarnsRegex(RuntimeWarning, "Unrestricted checkpoint"):
                        self.assertEqual(load_checkpoint(self.file, small_ppo(), trust_legacy=True), self.counters)
                loader.assert_called_once_with(self.file, map_location="cpu")

    def test_restored_state_produces_identical_next_cpu_update(self):
        self.algorithm.update(collect_mock(self.algorithm)[0])
        destination = small_ppo()
        load_checkpoint(self.file, destination)
        destination.update(collect_mock(destination)[0])
        for old, new in zip(self.algorithm.model.parameters(), destination.model.parameters()):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(self.algorithm.optimizer.state[old][key],
                                           destination.optimizer.state[new][key], rtol=0, atol=0)
        for name in ("observation_normalizer", "value_normalizer"):
            for key, value in getattr(self.algorithm, name).state_dict().items():
                torch.testing.assert_close(getattr(destination, name).state_dict()[key], value, rtol=0, atol=0)
        self.assertEqual(destination.learning_rate, self.algorithm.learning_rate)

    def test_unsupported_metadata_does_not_replace_existing_checkpoint(self):
        original = self.file.read_bytes()
        with self.assertRaisesRegex(TypeError, "Unsupported checkpoint metadata"):
            save_checkpoint(self.file, self.algorithm, self.counters, TrainingConfig(), {"custom": object()})
        self.assertEqual(self.file.read_bytes(), original)


class FullShapeTask:
    """Interface-only fixture: no physics, rewards, URDF or sensor simulation."""
    def __init__(self, clip_actions=1.):
        self.num_envs, self.num_obs, self.num_acts = 2, 104, 18
        self.action_space = SimpleNamespace(shape=(18,), low=[-1.] * 18, high=[1.] * 18)
        self.obs = torch.zeros(2, 104)
        self.dones = torch.ones(2, dtype=torch.long)
        self.clip_actions = clip_actions
        self.steps = self.reset_calls = 0
        self.reset_steps = []
        self.received = []

    def reset(self):
        self.reset_calls += 1
        return {"obs": self.obs}

    def step(self, actions):
        self.steps += 1
        pending = self.dones.bool().clone()
        actions.clamp_(-self.clip_actions, self.clip_actions)
        self.received.append(actions.clone())
        actions[:, 2::3] = -actions[:, 1::3]
        # An abstract step marker followed by deferred reset. This does not
        # claim to reproduce native GPU setter/refresh visibility.
        self.obs.fill_(self.steps)
        if pending.any():
            self.reset_steps.append(self.steps)
            self.obs[pending] = 0
        self.dones.fill_(int(self.steps == 2))
        return {"obs": self.obs}, torch.zeros(2), self.dones, {"time_outs": torch.zeros(2, dtype=torch.long)}


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class FullShapeIntegrationTests(unittest.TestCase):
    def algorithm(self):
        return PPO(ActorCritic(ActorCriticConfig(hidden_sizes=(8,))),
                   PPOConfig(num_actors=2, horizon_length=2, minibatch_size=4))

    def test_initial_zero_observations_and_deferred_resets_across_rollouts(self):
        task = FullShapeTask()
        collector = RolloutCollector(self.algorithm(), IsaacGymAdapter(task, "cpu"))
        self.assertEqual((task.reset_calls, task.steps), (1, 0))
        self.assertEqual(tuple(collector.observations.shape), (2, 104))
        self.assertTrue(collector.dones.all())
        self.assertTrue(task.dones.bool().all())
        self.assertEqual(collector.observations.count_nonzero().item(), 0)
        first = collector.collect()
        self.assertEqual(task.reset_steps, [1])
        self.assertTrue(first.last_dones.all())
        second = collector.collect()
        self.assertEqual(task.reset_steps, [1, 3])
        self.assertEqual(task.reset_calls, 1)
        self.assertTrue(second.data["dones"][0].all())
        torch.testing.assert_close(second.data["observations"][0], torch.full((2, 104), 2.))
        self.assertEqual(second.data["observations"][1].count_nonzero().item(), 0)

    def test_full_action_samples_and_densities_survive_mutation(self):
        algorithm = self.algorithm()
        task = FullShapeTask()
        raw = torch.tensor([[2., .25, 7.] * 6, [-2., -.5, -8.] * 6])
        original = raw.clone()
        log_probs = torch.tensor([-10., -20.])
        sample = PolicyStep(raw, log_probs, torch.zeros(2, 1), torch.zeros(2, 18), torch.ones(2, 18))
        with patch.object(algorithm, "sample", return_value=sample):
            buffer = RolloutCollector(algorithm, IsaacGymAdapter(task, "cpu")).collect()
        torch.testing.assert_close(raw, original, rtol=0, atol=0)
        for t in range(2):
            torch.testing.assert_close(buffer.data["actions"][t], original, rtol=0, atol=0)
            torch.testing.assert_close(buffer.data["log_probs"][t], log_probs, rtol=0, atol=0)
        self.assertEqual(tuple(buffer.data["actions"].shape), (2, 2, 18))

    def test_two_stage_clipping_has_independent_expected_values(self):
        raw = torch.tensor([[-2., .25, 2.] * 6] * 2)
        for limit, triple in ((0., [0., 0., 0.]), (.5, [-.5, .25, .5]),
                              (1., [-1., .25, 1.]), (2., [-1., .25, 1.]),
                              (float("inf"), [-1., .25, 1.])):
            with self.subTest(limit=limit):
                task = FullShapeTask(limit)
                IsaacGymAdapter(task, "cpu").step(raw)
                torch.testing.assert_close(task.received[0], torch.tensor([triple * 6] * 2), rtol=0, atol=0)
                torch.testing.assert_close(raw, torch.tensor([[-2., .25, 2.] * 6] * 2), rtol=0, atol=0)
