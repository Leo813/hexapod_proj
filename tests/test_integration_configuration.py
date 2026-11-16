import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from integration_test_support import HAS_OMEGACONF, ROOT, make_installation
from hexapod_rl.integration.configuration import (load_configuration, locate_installation,
    parse_arguments, TrainingConfig, validate_devices, translate_config)
from hexapod_rl.integration.task_factory import initialize_isaac_gym, load_task_class


class LauncherStaticTests(unittest.TestCase):
    def test_supported_argument_syntax(self):
        args = parse_arguments(["task=hexapod", "headless=True", "checkpoint=C:/folder/a=b.pt"])
        self.assertEqual(args["checkpoint"], "C:/folder/a=b.pt")

    def test_invalid_or_unsupported_arguments(self):
        for args in (["--multirun"], ["+x=1"], ["x=1", "x=2"], ["task=Ant"], ["train=hexapodPPO_LSTM"]):
            with self.subTest(args=args), self.assertRaises(ValueError):
                parse_arguments(args)

    def test_missing_installation_without_importing_it(self):
        with patch("hexapod_rl.integration.configuration.importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "isaacgymenvs_path"):
                locate_installation()

    def test_manual_paths_and_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            package = make_installation(directory)
            self.assertEqual(locate_installation(directory), (package, package / "cfg"))
            self.assertEqual(locate_installation(package, package / "cfg"), (package, package / "cfg"))
            for relative in ("tasks/hexapod.py", "cfg/task/hexapod.yaml", "cfg/train/hexapodPPO.yaml", "cfg/config.yaml"):
                path = package / relative
                original = path.read_bytes()
                path.unlink()
                with self.subTest(relative=relative), self.assertRaisesRegex(FileNotFoundError, "Required file missing"):
                    locate_installation(directory)
                path.write_bytes(original)

    def test_device_validation(self):
        root = dict(sim_device="cuda:0", rl_device="cpu", pipeline="gpu", physics_engine="physx",
                    graphics_device_id=-1, headless=True)
        validate_devices(root)
        for changes in ({"sim_device": "cpu"}, {"rl_device": "mps"}, {"physics_engine": "flex"},
                        {"headless": False}, {"graphics_device_id": -2}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_devices(dict(root, **changes))

    def test_training_control_validation(self):
        self.assertEqual(TrainingConfig().max_epochs, 600)
        self.assertEqual(TrainingConfig(max_epochs=-1).max_epochs, -1)
        for kwargs in ({"max_epochs": 0}, {"save_frequency": -1}, {"games_to_track": 0}, {"print_stats": "True"}):
            with self.assertRaises(ValueError):
                TrainingConfig(**kwargs)

    def test_gym_import_order_guard(self):
        with patch.dict(sys.modules, {"torch": ModuleType("torch")}):
            previous = sys.modules.pop("isaacgym", None)
            try:
                with self.assertRaisesRegex(RuntimeError, "before torch"):
                    initialize_isaac_gym()
            finally:
                if previous is not None:
                    sys.modules["isaacgym"] = previous

    def test_narrow_task_loader_skips_registry_initializers(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules):
            package = make_installation(directory)
            for name in list(sys.modules):
                if name == "isaacgymenvs" or name.startswith("isaacgymenvs."):
                    del sys.modules[name]
            sys.modules["isaacgym"] = ModuleType("isaacgym")
            cls, writer = load_task_class(package)
            self.assertEqual(cls.__name__, "Anymal10")
            self.assertEqual(cls.__bases__[0].__name__, "VecTask")
            self.assertIsNone(writer)
            with self.assertRaisesRegex(RuntimeError, "fresh process"):
                load_task_class(package)

    def test_help_and_launcher_import_need_no_simulator_or_torch(self):
        script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'isaacgym', 'isaacgymenvs', 'rl_games', 'torch'}:
            raise AssertionError('forbidden early import: ' + fullname)
sys.meta_path.insert(0, Block())
from hexapod_rl.launcher import main
assert main(['--help']) == 0
"""
        script = f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r})\n" + script
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(HAS_OMEGACONF, "OmegaConf is not installed")
class ConfigurationCompositionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.package = make_installation(self.temporary.name)

    def load(self, *arguments):
        return load_configuration([f"isaacgymenvs_path={self.temporary.name}", "task=hexapod", *arguments])

    def test_original_yaml_defaults_and_relative_resolvers(self):
        app = self.load()
        self.assertEqual((app.network.observation_dim, app.network.action_dim), (104, 18))
        self.assertEqual(app.network.hidden_sizes, (512, 256, 128))
        self.assertEqual((app.ppo.num_actors, app.ppo.horizon_length, app.ppo.mini_epochs), (256, 48, 5))
        self.assertEqual(app.ppo.learning_rate, 3e-4)
        self.assertEqual(app.training.max_epochs, 600)
        self.assertEqual(app.resolved["task_name"], "Anymal10")
        self.assertTrue(app.resolved["task"]["sim"]["use_gpu_pipeline"])
        self.assertEqual(app.adjustments, ())

    def test_common_overrides_and_explicit_minibatch_adjustment(self):
        app = self.load("headless=True", "num_envs=64", "max_iterations=100", "experiment=custom",
                        "sim_device=cpu", "rl_device=cpu", "pipeline=cpu")
        self.assertEqual(app.ppo.batch_size, 3072)
        self.assertEqual(app.ppo.minibatch_size, 3072)
        self.assertEqual(len(app.adjustments), 1)
        self.assertEqual(app.training.max_epochs, 100)
        self.assertEqual(app.resolved["train"]["params"]["config"]["name"], "custom")
        self.assertFalse(app.resolved["task"]["sim"]["physx"]["use_gpu"])

    def test_explicit_training_override_is_not_silently_changed(self):
        app = self.load("num_envs=64", "train.params.config.minibatch_size=1536", "train.params.config.learning_rate=0.001")
        self.assertEqual(app.ppo.minibatch_size, 1536)
        self.assertEqual(app.ppo.learning_rate, .001)
        with self.assertRaises(ValueError):
            self.load("num_envs=64", "train.params.config.minibatch_size=12288")

    def test_missing_dimension_and_unsupported_settings(self):
        app = self.load()
        import copy
        for section, key, value in (("env", "numObservations", 103), ("env", "numActions", 12)):
            root = copy.deepcopy(app.resolved)
            root["task"][section][key] = value
            with self.assertRaisesRegex(ValueError, "requires"):
                translate_config(root)
        for arg in ("train.params.network.separate=False", "train.params.network.mlp.d2rl=True", "multi_gpu=True"):
            with self.subTest(arg=arg), self.assertRaises(ValueError):
                self.load(arg)
        root = copy.deepcopy(app.resolved)
        root["train"]["params"]["config"]["central_value_config"] = {}
        with self.assertRaisesRegex(ValueError, "Unsupported training"):
            translate_config(root)

    def test_unknown_override_and_missing_checkpoint(self):
        for arg in ("nmu_envs=64", "checkpoint=/does/not/exist.pt", "test=True"):
            with self.subTest(arg=arg), self.assertRaises((ValueError, FileNotFoundError)):
                self.load(arg)

    def test_explicit_config_path_works_outside_current_directory(self):
        app = load_configuration([f"isaacgymenvs_path={self.package}", f"config_path={self.package / 'cfg'}"])
        self.assertEqual(app.config_path, self.package / "cfg")
