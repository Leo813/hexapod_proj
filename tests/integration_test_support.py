"""Synthetic installation and VecTask fixtures; never imports real Isaac Gym."""

import importlib.util
from pathlib import Path
import shutil
from types import SimpleNamespace

from ppo_test_support import HAS_TORCH, MockVecTask, small_ppo

HAS_OMEGACONF = importlib.util.find_spec("omegaconf") is not None
ROOT = Path(__file__).resolve().parents[1]

# Minimal independent root configuration with historical defaults/interpolations.
ROOT_YAML = """
task_name: ${task.name}
experiment: ''
num_envs: ''
seed: 42
torch_deterministic: False
max_iterations: ''
physics_engine: physx
pipeline: gpu
sim_device: cuda:0
rl_device: cuda:0
graphics_device_id: 0
num_threads: 4
solver_type: 1
num_subscenes: 4
test: False
checkpoint: ''
sigma: ''
multi_gpu: False
wandb_activate: False
wandb_name: ${train.params.config.name}
capture_video: False
force_render: True
headless: False
defaults:
  - task: Ant
  - train: ${task}PPO
  - _self_
hydra:
  run:
    dir: .
"""


def make_installation(directory):
    package = Path(directory) / "isaacgymenvs"
    for child in ("tasks/base", "cfg/task", "cfg/train", "utils"):
        (package / child).mkdir(parents=True)
    (package / "__init__.py").write_text("raise AssertionError('broad package initializer executed')\n")
    (package / "tasks/__init__.py").write_text("raise AssertionError('task registry executed')\n")
    (package / "tasks/base/vec_task.py").write_text("class VecTask: pass\n")
    (package / "tasks/hexapod.py").write_text(
        "from isaacgymenvs.tasks.base.vec_task import VecTask\nclass Anymal10(VecTask): pass\n")
    (package / "cfg/config.yaml").write_text(ROOT_YAML)
    for source, target in (("hexapod.yaml", "task/hexapod.yaml"), ("hexapodPPO.yaml", "train/hexapodPPO.yaml")):
        shutil.copyfile(ROOT / source, package / "cfg" / target)
    return package


if HAS_TORCH:
    import torch
    from hexapod_rl.integration.isaacgym_adapter import IsaacGymAdapter


class MockInstalledTask(MockVecTask):
    """Adds the actual VecTask surface to the Stage 2 mutation/reset fixture."""
    def __init__(self):
        super().__init__()
        self.num_envs, self.num_obs, self.num_acts = 2, 3, 2
        self.action_space = SimpleNamespace(shape=(2,), low=[-1., -1.], high=[1., 1.])
        self.reset_calls = self.step_calls = self.close_calls = 0
        self.train_frames = []
        self.received_actions = []

    def reset(self):
        self.reset_calls += 1
        return {"obs": self.observations}  # no physics, no recomputation/reset_idx

    def step(self, actions):
        self.step_calls += 1
        self.received_actions.append(actions.detach().clone())
        return super().step(actions)

    def set_train_info(self, frames):
        self.train_frames.append(frames)

    def close(self):
        self.close_calls += 1


def mock_adapter(task=None):
    task = MockInstalledTask() if task is None else task
    return IsaacGymAdapter(task, "cpu", observation_dim=3, action_dim=2)
