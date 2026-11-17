"""Bounded, explicitly invoked worker using the existing task/adapter/Trainer.

Importing this module performs no native imports. See check_native_readiness.py.
"""

import json
import math
from pathlib import Path
import sys
import traceback
import xml.etree.ElementTree as ET

from install_hexapod_task import SOURCE, checked_path


def smoke_overrides(target, output, device):
    return [f"isaacgymenvs_path={target}", "task=hexapod", "num_envs=4", "max_iterations=1",
            "pipeline=gpu", f"sim_device={device}", f"rl_device={device}", "physics_engine=physx",
            "headless=True", "graphics_device_id=-1", "force_render=False", "checkpoint=",
            "test=False", "tensorboard=False", f"output_dir={output}"]


def finite_metrics(metrics):
    for row in metrics:
        for key, value in row.items():
            if value is not None and isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"Nonfinite PPO metric {key}={value}")


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)  # retain failed diagnostic values without nonstandard JSON
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(item) for item in value]
    return value


def trace_method(environment, name, record):
    """Observe original method exactly once; do not insert setters/refreshes/resets."""
    original = getattr(environment, name)
    def traced(*args, **kwargs):
        record(name + ":before")
        result = original(*args, **kwargs)
        record(name + ":after")
        return result
    setattr(environment, name, traced)
    return original


def run_smoke(target, output, device):
    target, output = checked_path(target), checked_path(output)
    if any(output == root or root in output.parents or output in root.parents for root in (SOURCE, target)):
        raise ValueError("Smoke output must be separate from source and checkout")
    if not output.is_dir() or any(output.iterdir()) or Path.cwd().resolve() != output.resolve():
        raise ValueError("Worker requires an empty reserved output directory as its working directory")
    report = {"ok": False, "stage": "import-order", "diagnostic_not_baseline": True,
              "num_envs": 4, "epochs": 1, "trace": [], "warnings": []}
    environment = adapter = writer = None
    originals = {}
    try:
        if "isaacgym" not in sys.modules or "torch" not in sys.modules:
            raise RuntimeError("Run through the fresh-process import probe (Isaac Gym before torch)")
        import torch
        sys.path.insert(0, str(SOURCE / "src"))
        from hexapod_rl.integration.configuration import load_configuration
        from hexapod_rl.integration.task_factory import construct_task
        from hexapod_rl.integration.isaacgym_adapter import IsaacGymAdapter, close_environment
        from hexapod_rl.launcher import establish_runtime
        from hexapod_rl.networks import ActorCritic
        from hexapod_rl.ppo import PPO
        from hexapod_rl.trainer import Trainer, TrainingLogger
        from hexapod_rl.checkpoints import load_checkpoint

        report["stage"] = "configuration"
        application = load_configuration(smoke_overrides(target, output, device))
        if (application.ppo.num_actors, application.ppo.horizon_length, application.ppo.mini_epochs,
                application.ppo.minibatch_size, application.training.max_epochs) != (4, 48, 5, 192, 1):
            raise ValueError("Smoke requires 4 environments, horizon 48, five mini-epochs, minibatch 192, one epoch")
        root = application.resolved
        if not root["task"]["sim"]["use_gpu_pipeline"] or not root["task"]["sim"]["physx"]["use_gpu"]:
            raise ValueError("GPU pipeline and GPU PhysX must both be enabled")
        report["adjustments"] = list(application.adjustments)
        report["notices"] = list(application.notices)
        (output / "resolved-config.json").write_text(json.dumps(root, indent=2), encoding="utf-8")
        establish_runtime(root)
        report["stage"] = "task-construction"
        environment, writer = construct_task(application)
        report["stage"] = "task-evidence"
        env0, actor = environment.envs[0], environment.anymal_handles[0]
        body_names = list(environment.gym.get_actor_rigid_body_names(env0, actor))
        joint_names = list(environment.gym.get_actor_dof_names(env0, actor))
        urdf = ET.parse(target / "assets/urdf/hexapod2/urdf/test_spider6.urdf").getroot()
        joints = [j for j in urdf.findall("joint") if j.get("type") == "revolute"]
        required_bodies = {j.find("child").get("link") for j in joints}
        report["asset"] = {"body_names": body_names, "joint_names": joint_names,
            "body_count": environment.num_bodies, "dof_count": environment.num_dof,
            "urdf_joint_names": [j.get("name") for j in joints],
            "missing_moving_bodies": sorted(required_bodies - set(body_names)),
            "sensor_body_index_5": body_names[5] if len(body_names) > 5 else None,
            "hardcoded_contact_bodies": {str(i): body_names[i] if len(body_names) > i else None for i in (0, 3, 6, 9, 12, 15, 18)}}
        if environment.num_dof != 18 or len(joint_names) != 18 or set(joint_names) != {j.get("name") for j in joints}:
            raise ValueError("Imported DOFs do not match the original 18 revolute joints; inspect asset evidence")
        if required_bodies - set(body_names) or len(body_names) <= 18:
            raise ValueError("Expected moving bodies or hard-coded contact indices are missing")
        params = environment.sim_params
        report["gpu_flags"] = {"pipeline": bool(params.use_gpu_pipeline), "physx": bool(params.physx.use_gpu)}
        if not all(report["gpu_flags"].values()):
            raise ValueError("Constructed simulator parameters are not GPU pipeline/PhysX")

        def tensor_evidence(value):
            return {"shape": list(value.shape), "device": str(value.device), "dtype": str(value.dtype),
                    "values": value.detach().cpu().tolist()}

        def record(event):
            entry = {"event": event}
            for name in ("progress_buf", "reset_buf", "dof_pos", "dof_position_targets", "root_states"):
                entry[name] = tensor_evidence(getattr(environment, name))
            # Derived from the unchanged pre_physics_step expression, not a Gym setter hook.
            entry["derived_scaled_targets"] = tensor_evidence(environment.dof_position_targets * environment.action_scale)
            report["trace"].append(entry)

        report["limits"] = {name: tensor_evidence(getattr(environment, name))
                            for name in ("bbot_dof_lower_limits", "bbot_dof_upper_limits", "feet_indices", "knee_indices", "thigh_indices")}
        report["sensors"] = {name: {"shape": list(getattr(environment, name).shape), "device": str(getattr(environment, name).device)}
                             for name in ("sensor_forces", "sensor_torques")}
        if any(item["shape"] != [4, 3] or item["device"] != device for item in report["sensors"].values()):
            raise ValueError("Force-sensor tensor shape/device mismatch")
        record("constructed")
        for name in ("reset_idx", "compute_observations", "pre_physics_step"):
            originals[name] = trace_method(environment, name, record)

        class ObservedAdapter(IsaacGymAdapter):
            steps = 0
            done_count = 0

            def step(self, actions):
                if self.steps >= 48:
                    raise RuntimeError("Diagnostic step budget exhausted")
                values = super().step(actions)
                self.steps += 1
                for name, value in zip(("observations", "rewards", "dones", "timeouts"), values):
                    if not torch.isfinite(value).all().item():
                        raise ValueError(f"Nonfinite native {name} at step {self.steps}")
                for name in ("sensor_forces", "sensor_torques"):
                    if not torch.isfinite(getattr(environment, name)).all().item():
                        raise ValueError(f"Nonfinite {name} at step {self.steps}")
                self.done_count += int(values[2].count_nonzero().item())
                return values

        adapter = ObservedAdapter(environment, device, task_writer=writer)
        model = ActorCritic(application.network).to(device)
        algorithm = PPO(model, application.ppo)
        before = [p.detach().clone() for p in model.parameters()]
        original_update = algorithm.update
        def checked_update(rollout):
            metrics = original_update(rollout)
            report["updates"] = metrics
            finite_metrics(metrics)
            return metrics
        algorithm.update = checked_update
        report["stage"] = "bounded-training"
        trainer = Trainer(algorithm, adapter, output, application.training, TrainingLogger(), root)
        report["history"] = trainer.run()  # existing collection/update/checkpoint/cleanup lifecycle
        report["steps"], report["done_count"] = adapter.steps, adapter.done_count
        if adapter.steps != 48 or not algorithm.optimizer.state:
            raise ValueError("Incomplete rollout or optimizer state not initialized")
        if not any(not torch.equal(a, b) for a, b in zip(before, model.parameters())):
            raise ValueError("No model parameters changed; an optimizer update is not established")
        if not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise ValueError("Nonfinite model parameters after update")
        report["stage"] = "cuda-checkpoint-restore"
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        restored = PPO(ActorCritic(application.network).to(device), application.ppo)
        counters = load_checkpoint(output / "checkpoints/last.pt", restored)  # restricted, never trusted fallback

        def equal_state(left, right):
            if isinstance(left, torch.Tensor):
                return isinstance(right, torch.Tensor) and left.device == right.device and left.dtype == right.dtype and torch.equal(left, right)
            if isinstance(left, dict):
                return left.keys() == right.keys() and all(equal_state(left[k], right[k]) for k in left)
            if isinstance(left, (tuple, list)):
                return len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right))
            return left == right

        for name in ("model", "optimizer", "observation_normalizer", "value_normalizer", "scaler"):
            left, right = getattr(algorithm, name), getattr(restored, name)
            if not equal_state(left.state_dict(), right.state_dict()):
                raise ValueError(f"Same-device checkpoint mismatch in {name}")
        if counters != trainer.counters() or restored.learning_rate != algorithm.learning_rate:
            raise ValueError("Checkpoint counters/learning rate mismatch")
        if not torch.equal(cpu_rng, torch.get_rng_state()) or not all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all())):
            raise ValueError("Checkpoint torch CPU/CUDA RNG mismatch")
        report["checkpoint"] = {"restored": True, "device": device, "scaler_enabled": restored.scaler.is_enabled(), "counters": counters}
        if not adapter.done_count:
            report["warnings"].append("No natural termination observed; deferred initial resets are not a complete episode-lifecycle test")
        report["warnings"].append("Method-boundary snapshots synchronize CUDA; they do not prove asynchronous setter/refresh correctness")
        report.update(ok=True, stage="bounded-smoke-complete")
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        if environment is not None:
            for name, method in originals.items():
                setattr(environment, name, method)
            try:
                if adapter is not None:
                    adapter.close()
                else:
                    close_environment(environment, writer)
            except Exception as error:
                report.update(ok=False, cleanup_error=str(error))
        (output / "smoke-evidence.json").write_text(json.dumps(clean_json(report), indent=2, allow_nan=False), encoding="utf-8")
    return clean_json({key: value for key, value in report.items() if key not in ("trace", "limits", "updates")})
