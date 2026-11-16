"""Small application launcher; all tensor/simulator imports are deliberately late."""

from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys

from .integration.configuration import load_configuration
from .integration.task_factory import construct_task, initialize_isaac_gym


HELP = """Custom feedforward hexapod PPO (no RL-Games trainer)
  python train.py task=hexapod [key=value ...]
  python train.py task=hexapod headless=True num_envs=64 max_iterations=100
  python train.py task=hexapod test=True checkpoint=/path/custom.pt eval_steps=1000

Paths: isaacgymenvs_path=/absolute/IsaacGymEnvs [config_path=/absolute/cfg]
Devices: sim_device=cuda:0 rl_device=cuda:0 pipeline=gpu graphics_device_id=0
Outputs: output_dir=/path/run tensorboard=False
Resume: checkpoint=/path/custom.pt max_iterations=700
Other overrides: existing task.* / train.* YAML leaves, seed, torch_deterministic,
headless, experiment, force_render. No Hydra sweeps, +key, LSTM, or multi-GPU.
See docs/stage3-integration.md for manual installation and validation limits.
"""


def establish_runtime(root):
    import numpy as np
    import torch

    for name in ("sim_device", "rl_device"):
        device = torch.device(root[name])
        if device.type == "cuda" and (not torch.cuda.is_available() or device.index >= torch.cuda.device_count()):
            raise ValueError(f"{name}={device} is unavailable; use a CUDA-capable torch/environment or explicitly select supported CPU devices")
    seed = root["seed"]
    if seed == -1:
        seed = 42 if root["torch_deterministic"] else int(np.random.randint(0, 10000))
    root["seed"] = seed
    root["train"]["params"]["seed"] = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    deterministic = root["torch_deterministic"]
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)
    print(f"seed={seed} sim_device={root['sim_device']} rl_device={root['rl_device']} graphics_device_id={root['graphics_device_id']}", flush=True)


def run_application(application):
    # initialize_isaac_gym MUST precede this function (including its imports).
    from .checkpoints import load_checkpoint
    from .evaluation import evaluate
    from .integration.isaacgym_adapter import IsaacGymAdapter, close_environment
    from .networks import ActorCritic
    from .ppo import PPO
    from .trainer import Trainer, TrainingLogger

    root = application.resolved
    establish_runtime(root)
    for message in application.adjustments:
        print(f"Configuration adjustment: {message}", flush=True)
    environment, task_writer = construct_task(application)
    adapter = logger = None
    try:
        adapter = IsaacGymAdapter(environment, root["rl_device"], task_writer=task_writer)
        if adapter.num_envs != application.ppo.num_actors:
            raise ValueError("Constructed environment count differs from configured num_actors")
        model = ActorCritic(application.network).to(root["rl_device"])
        algorithm = PPO(model, application.ppo)
        if root["test"]:
            load_checkpoint(root["checkpoint"], algorithm, resume=False)
            report = evaluate(algorithm, adapter, root["eval_steps"], application.training.games_to_track)
            print(f"Evaluation: {report}", flush=True)
            return report
        name = str(root["train"]["params"]["config"]["name"])
        if not name or Path(name).name != name or any(character in name for character in '/\\:'):
            raise ValueError("Experiment name must be a plain directory name")
        output = (Path(root["output_dir"]).expanduser().resolve() if root["output_dir"] else
                  Path("runs", name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")).resolve())
        if output.exists() and any(output.iterdir()) and not root["checkpoint"]:
            raise ValueError(f"Output directory is not empty: {output}; choose a new output_dir or explicitly resume")
        output.mkdir(parents=True, exist_ok=True)
        logger = TrainingLogger(application.training.print_stats, output / "summaries" if root["tensorboard"] else None)
        trainer = Trainer(algorithm, adapter, output, application.training, logger, root)
        if root["checkpoint"]:
            trainer.resume(root["checkpoint"])
            print(f"Restored learning state at epoch={trainer.epoch}, frames={trainer.frames}; simulator episodes are fresh", flush=True)
        (output / "resolved-config.json").write_text(json.dumps(root, indent=2), encoding="utf-8")
        print(f"Custom PPO output: {output}", flush=True)
        return trainer.run()
    finally:
        try:
            if adapter is not None:
                adapter.close()
            else:
                close_environment(environment, task_writer)
        finally:
            if logger is not None:
                logger.close()


def main(arguments=None):
    arguments = sys.argv[1:] if arguments is None else arguments
    if arguments in (["--help"], ["-h"]):
        print(HELP)
        return 0
    try:
        application = load_configuration(arguments)
        initialize_isaac_gym()
        run_application(application)
        return 0
    except KeyboardInterrupt:
        print("Training interrupted; resources closed. Resume from a completed checkpoint.", file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as error:
        print(f"Custom PPO launch failed: {error}", file=sys.stderr)
        return 1
