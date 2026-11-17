"""Readiness/authorization tests with inert native doubles, never a real GPU."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_native_readiness as readiness
import native_smoke
from integration_test_support import HAS_OMEGACONF, make_installation


def inventory():
    return {"os": "Linux test", "os_release": 'ID=ubuntu\nVERSION_ID="20.04"',
        "python_version": "3.8.18", "distributions": {"torch": "1.13.1+cu117", "numpy": "1.24.4",
        "omegaconf": "2.3.0", "gym": "0.23.1", "matplotlib": "3.5.3", "tensorboard": "2.10.0"},
        "package_locations": {"isaacgym": ["/sdk/python/isaacgym"]},
        "torch_cuda_build": {"value": "11.7", "verified_by_import": False}}


class NativeReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()

    def test_final_release_detection_and_unknown_prereleases(self):
        self.assertEqual(readiness.release("1.13.1+cpu"), (1, 13, 1))
        self.assertEqual(readiness.release("1.10"), (1, 10, 0))
        for value in ("missing", None, "1.13.0rc1", "1.13.0.dev20220101"):
            self.assertIsNone(readiness.release(value))

    def test_source_candidate_is_not_reported_as_native_verified(self):
        result = readiness.assess_requirements(inventory())
        self.assertEqual(result["blockers"], [])
        self.assertIn("not native validation", " ".join(result["notices"]))
        self.assertEqual(result["package_requirements"]["torch"], ">=1.13")

    def test_torch_110_and_old_python_conflicts_are_explicit(self):
        info = inventory()
        info["python_version"] = "3.7.12"
        info["distributions"]["torch"] = "1.10.2+cu113"
        text = " ".join(readiness.assess_requirements(info)["blockers"])
        self.assertIn("torch 1.10.2+cu113 does not satisfy", text)
        self.assertIn("Python 3.7.12 does not satisfy", text)

    def test_missing_dependencies_and_isaac_gym_are_reported(self):
        info = inventory()
        info["distributions"]["torch"] = "not found"
        info["distributions"]["omegaconf"] = "not found"
        info["package_locations"]["isaacgym"] = "not found"
        text = " ".join(readiness.assess_requirements(info)["blockers"])
        for expected in ("torch version missing", "omegaconf", "Missing discoverable isaacgym"):
            self.assertIn(expected, text)

    def test_cpu_build_non_linux_and_new_python_block_smoke(self):
        info = inventory()
        info.update(os="Windows", python_version="3.10.11", torch_cuda_build={"value": None})
        text = " ".join(readiness.assess_requirements(info)["blockers"])
        for expected in ("CPU build", "not Linux", "Python 3.8 is the intersection"):
            self.assertIn(expected, text)

    def test_optional_dependency_range_is_checked(self):
        info = inventory()
        info["distributions"]["omegaconf"] = "2.4.0"
        self.assertIn("OmegaConf does not satisfy", " ".join(readiness.assess_requirements(info)["blockers"]))

    def test_sdk_absence_and_present_files_are_not_invented_or_executed(self):
        self.assertIn("unavailable", readiness.sdk_evidence(None)["status"])
        sdk = self.base / "sdk"
        (sdk / "python").mkdir(parents=True)
        setup = sdk / "python/setup.py"
        setup.write_text("raise RuntimeError('do not execute SDK setup')", encoding="utf-8")
        result = readiness.sdk_evidence(sdk)["files"]
        self.assertFalse(result["python/rlgpu_conda_env.yml"]["present"])
        self.assertTrue(result["python/setup.py"]["present"])
        self.assertEqual(len(result["python/setup.py"]["sha256"]), 64)

    def test_missing_or_conflicting_installed_task_blocks_readiness(self):
        plan = SimpleNamespace(complete=False, revision="test", warnings=[], errors=["missing task"], files=[])
        with patch.object(readiness, "inspect_environment", return_value=inventory()), patch.object(readiness, "inspect_installation", return_value=plan):
            report = readiness.readiness(self.base)
        self.assertIn("Required installed", " ".join(report["blockers"]))
        self.assertEqual(report["simulator_compatibility"], "unverified")

    def test_default_mode_never_starts_worker_or_writes_files(self):
        with patch.object(readiness, "inspect_environment", return_value=inventory()), patch.object(readiness, "run_worker") as worker:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(readiness.main([]), 1)  # no selected checkout
        worker.assert_not_called()
        self.assertEqual(list(self.base.iterdir()), [])

    def test_import_probe_needs_explicit_cli_flag(self):
        report = {"blockers": [], "native_execution": "not requested"}
        with patch.object(readiness, "readiness", return_value=report), patch.object(readiness, "run_worker", return_value={"ok": True}) as worker:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(readiness.main(["--check-imports", "--device", "cuda:2"]), 0)
        worker.assert_called_once_with("cuda:2", 180)

    def test_smoke_requires_output_and_refuses_preflight_blockers(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            readiness.main(["--run-smoke"])
        with patch.object(readiness, "readiness", return_value={"blockers": ["torch unsupported"]}), patch.object(readiness, "run_worker") as worker:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(readiness.main(["--run-smoke", "--isaacgymenvs", str(self.base), "--output", str(self.base / "result")]), 1)
        worker.assert_not_called()
        self.assertFalse((self.base / "result").exists())

    def test_worker_mode_also_requires_active_authorization(self):
        with patch.object(readiness, "import_probe") as probe, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            readiness.main(["--_worker"])
        probe.assert_not_called()

    def test_device_and_time_budgets_are_validated_before_execution(self):
        for args in (["--device", "cpu"], ["--device", "cuda:-1"], ["--timeout", "0"], ["--timeout", "601"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                readiness.main(args)

    def test_import_order_guard_rejects_preimported_torch(self):
        with patch.dict(sys.modules, {"torch": object()}), patch.object(readiness.importlib, "import_module") as imports:
            result = readiness.import_probe("cuda:0")
        imports.assert_not_called()
        self.assertEqual(result["stage"], "import-order")
        self.assertFalse(result["ok"])

    def probe_with(self, importer):
        with patch.dict(sys.modules):
            sys.modules.pop("torch", None)
            with patch.object(readiness.importlib, "import_module", side_effect=importer):
                return readiness.import_probe("cuda:0")

    def test_missing_isaac_and_torch_import_errors_have_distinct_stages(self):
        def no_gym(name):
            raise ImportError("SDK binding unavailable")
        self.assertEqual(self.probe_with(no_gym)["stage"], "isaacgym-import")
        def no_torch(name):
            if name == "torch":
                raise ImportError("CUDA library missing")
            return SimpleNamespace(__file__="sdk")
        result = self.probe_with(no_torch)
        self.assertEqual(result["stage"], "torch-import")
        self.assertIn("CUDA library missing", result["error"])

    def test_cuda_absence_and_device_unavailability_are_distinguished(self):
        cuda = Mock()
        cuda.is_available.return_value = False
        def importer(name):
            return SimpleNamespace(cuda=cuda) if name == "torch" else SimpleNamespace(__file__="sdk")
        self.assertEqual(self.probe_with(importer)["stage"], "cuda-availability")
        cuda.is_available.return_value = True
        cuda.device_count.return_value = 0
        self.assertEqual(self.probe_with(importer)["stage"], "cuda-device")

    def test_worker_process_command_and_native_crash_reporting(self):
        completed = SimpleNamespace(returncode=-11, stdout="native crash", stderr="driver diagnostic")
        with patch.object(readiness.subprocess, "run", return_value=completed) as run:
            result = readiness.run_worker("cuda:0", 15)
        command = run.call_args.args[0]
        self.assertEqual(command[0], sys.executable)
        self.assertIn("--check-imports", command)
        self.assertIn("--_worker", command)
        self.assertEqual(run.call_args.kwargs["timeout"], 15)
        self.assertFalse(result["ok"])
        self.assertEqual(result["returncode"], -11)
        self.assertIn("driver diagnostic", result["stderr_tail"])

    def test_worker_timeout_and_invalid_json_fail_clearly(self):
        with patch.object(readiness.subprocess, "run", side_effect=subprocess.TimeoutExpired("worker", 1)):
            self.assertEqual(readiness.run_worker("cuda:0", 1)["stage"], "worker-timeout")
        with patch.object(readiness.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=readiness.MARKER + "bad", stderr="")):
            self.assertEqual(readiness.run_worker("cuda:0", 1)["stage"], "worker-protocol")

    def test_fresh_worker_with_controlled_modules_imports_gym_first(self):
        # A fresh subprocess uses inert stand-in modules, never installed torch/Gym.
        (self.base / "isaacgym.py").write_text("IMPORTED = True\n", encoding="utf-8")
        (self.base / "torch.py").write_text("""
import sys
assert 'isaacgym' in sys.modules
__version__ = 'mock-only'
float32 = 'float32'
class version:
    cuda = 'mock'
class cuda:
    is_available = staticmethod(lambda: True)
    device_count = staticmethod(lambda: 1)
    set_device = staticmethod(lambda x: None)
    synchronize = staticmethod(lambda x: None)
    get_device_name = staticmethod(lambda x: 'CPU fixture, not a GPU')
class Value:
    device = 'cuda:0'
    def square(self): return self
    def sum(self): return self
    def item(self): return 140.0
def arange(*args, **kwargs): return Value()
""", encoding="utf-8")
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/check_native_readiness.py"),
            "--_worker", "--check-imports"], capture_output=True, text=True, timeout=20,
            env=dict(os.environ, PYTHONPATH=str(self.base)))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout.split(readiness.MARKER)[-1])
        self.assertTrue(report["ok"])
        self.assertEqual(report["torch"], "mock-only")
        self.assertEqual(report["simulator_compatibility"], "not tested")

    def test_output_is_exclusive_and_outside_source_and_checkout(self):
        target = self.base / "checkout"
        target.mkdir()
        output = self.base / "new-run"
        self.assertEqual(readiness.reserve_output(output, target), output)
        with self.assertRaises(FileExistsError):
            readiness.reserve_output(output, target)
        for unsafe in (target / "run", ROOT / "run"):
            with self.assertRaises(ValueError):
                readiness.reserve_output(unsafe, target)

    def test_smoke_overrides_are_bounded_and_preserve_hyperparameters(self):
        args = native_smoke.smoke_overrides("/checkout", "/results/run", "cuda:2")
        for option in ("num_envs=4", "max_iterations=1", "pipeline=gpu", "sim_device=cuda:2", "rl_device=cuda:2", "headless=True"):
            self.assertIn(option, args)
        self.assertFalse(any(arg.startswith("train.params.") for arg in args))
        self.assertFalse(any("episodeLength" in arg for arg in args))

    def test_trace_wrapper_calls_original_once_without_state_changes(self):
        obj = SimpleNamespace(step=Mock(return_value="result"))
        original = obj.step
        events = []
        native_smoke.trace_method(obj, "step", events.append)
        self.assertEqual(obj.step(5), "result")
        original.assert_called_once_with(5)
        self.assertEqual(events, ["step:before", "step:after"])

    def test_nonfinite_metrics_fail_without_weakened_assertions(self):
        native_smoke.finite_metrics([{"total": 1., "grad_norm": None}])
        for value in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "Nonfinite PPO metric total"):
                native_smoke.finite_metrics([{"total": value}])
        self.assertEqual(native_smoke.clean_json({"bad": float("nan")}), {"bad": "nan"})

    def test_missing_cuda_tools_remain_readonly_information(self):
        from install_hexapod_task import inspect_environment
        with patch("install_hexapod_task.shutil.which", return_value=None):
            result = inspect_environment(driver_info=True)
        self.assertEqual(result["driver"], "nvidia-smi not found")

    @unittest.skipUnless(HAS_OMEGACONF, "OmegaConf is not installed")
    def test_smoke_composes_original_yaml_with_existing_configuration_loader(self):
        from hexapod_rl.integration.configuration import load_configuration
        target = self.base / "checkout"
        make_installation(target)
        app = load_configuration(native_smoke.smoke_overrides(target, self.base / "output", "cuda:0"))
        self.assertEqual((app.ppo.num_actors, app.ppo.horizon_length, app.ppo.mini_epochs,
                          app.ppo.minibatch_size, app.training.max_epochs), (4, 48, 5, 192, 1))
        self.assertEqual((app.network.observation_dim, app.network.action_dim), (104, 18))
        self.assertTrue(app.resolved["task"]["sim"]["use_gpu_pipeline"])
        self.assertTrue(app.resolved["task"]["sim"]["physx"]["use_gpu"])
        self.assertEqual(app.network.hidden_sizes, (512, 256, 128))
        self.assertFalse(app.resolved["checkpoint"])
        self.assertEqual(len(app.adjustments), 1)
