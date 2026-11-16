"""Versioned custom learning-state checkpoints; no simulator state or RL-Games conversion."""

from dataclasses import asdict
import importlib.util
import os
from pathlib import Path
import random
import tempfile

import torch


FORMAT_VERSION = 1


def rng_state():
    result = {"python": random.getstate(), "torch": torch.get_rng_state(),
              "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    if importlib.util.find_spec("numpy") is not None:
        import numpy as np
        result["numpy"] = np.random.get_state()
    return result


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG device count differs; exact RNG restoration is unavailable")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    if "numpy" in state:
        import numpy as np
        np.random.set_state(state["numpy"])


def save_checkpoint(path, algorithm, counters, training_config, resolved_config=None):
    normalizers = {name: (getattr(algorithm, name).state_dict() if getattr(algorithm, name) is not None else None)
                   for name in ("observation_normalizer", "value_normalizer")}
    state = dict(format_version=FORMAT_VERSION, network_config=asdict(algorithm.model.config),
        ppo_config=asdict(algorithm.config), training_config=asdict(training_config),
        resolved_config=resolved_config, model=algorithm.model.state_dict(), normalizers=normalizers,
        optimizer=algorithm.optimizer.state_dict(), scaler=algorithm.scaler.state_dict(),
        learning_rate=algorithm.learning_rate, counters=dict(counters), rng=rng_state(),
        simulator_state=None)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path, algorithm, resume=True):
    """Load trusted custom files only; torch's pickle format can execute code.

    Evaluation checks network/normalization compatibility but permits different
    batch sizes. Resume checks all PPO settings, restores optimizer/LR/scaler/RNG,
    and returns counters. Construct a NEW environment/collector on resume.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # CPU loading keeps CPU RNG bytes and permits CPU evaluation of CUDA models.
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or state.get("format_version") != FORMAT_VERSION:
        raise ValueError("Not a supported custom PPO checkpoint; RL-Games checkpoint conversion is not supported")
    if state.get("network_config") != asdict(algorithm.model.config):
        raise ValueError("Checkpoint network dimensions/architecture differ from the configured ActorCritic")
    current = asdict(algorithm.config)
    if resume and state.get("ppo_config") != current:
        raise ValueError("Resume requires identical PPO settings; only application controls such as max_epochs may change")
    for key in ("normalize_input", "normalize_value", "normalization_epsilon", "normalization_clip"):
        if state["ppo_config"].get(key) != current[key]:
            raise ValueError(f"Checkpoint normalization setting differs: {key}")
    for name in ("observation_normalizer", "value_normalizer"):
        module = getattr(algorithm, name)
        saved = state["normalizers"][name]
        if (module is None) != (saved is None):
            raise ValueError(f"Checkpoint has incompatible {name}")
    algorithm.model.load_state_dict(state["model"], strict=True)
    for name in ("observation_normalizer", "value_normalizer"):
        module = getattr(algorithm, name)
        if module is not None:
            module.load_state_dict(state["normalizers"][name], strict=True)
            module.eval()
    if resume:
        algorithm.optimizer.load_state_dict(state["optimizer"])
        algorithm.scaler.load_state_dict(state["scaler"])
        if any(group["lr"] != state["learning_rate"] for group in algorithm.optimizer.param_groups):
            raise ValueError("Checkpoint optimizer learning rate does not match saved learning_rate")
        restore_rng(state["rng"])
    algorithm.model.eval()
    return state["counters"]
