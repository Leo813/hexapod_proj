"""Versioned custom learning-state checkpoints; no simulator state or RL-Games conversion."""

from dataclasses import asdict
import importlib.util
import inspect
import os
from pathlib import Path
import pickle
import random
import tempfile
import warnings

import torch

from .checkpoint_format import FORMAT_VERSION, decode_checkpoint, encode_checkpoint


_HAS_WEIGHTS_ONLY = "weights_only" in inspect.signature(torch.load).parameters


def rng_state():
    result = {"python": random.getstate(), "torch": torch.get_rng_state(),
              "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    if importlib.util.find_spec("numpy") is not None:
        import numpy as np
        name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
        # Builtins only: NumPy arrays require unrestricted pickle reconstruction.
        result["numpy"] = dict(name=name, keys=keys.tolist(), position=position,
                               has_gauss=has_gauss, cached_gaussian=cached_gaussian)
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
        saved = state["numpy"]
        if isinstance(saved, dict):
            saved = (saved["name"], np.asarray(saved["keys"], dtype=np.uint32),
                     saved["position"], saved["has_gauss"], saved["cached_gaussian"])
        np.random.set_state(saved)  # tuple form is supported for explicitly trusted v1 files


def save_checkpoint(path, algorithm, counters, training_config, resolved_config=None):
    normalizers = {name: (getattr(algorithm, name).state_dict() if getattr(algorithm, name) is not None else None)
                   for name in ("observation_normalizer", "value_normalizer")}
    state = dict(format_version=FORMAT_VERSION, network_config=asdict(algorithm.model.config),
        ppo_config=asdict(algorithm.config), training_config=asdict(training_config),
        resolved_config=resolved_config, model=algorithm.model.state_dict(), normalizers=normalizers,
        optimizer=algorithm.optimizer.state_dict(), scaler=algorithm.scaler.state_dict(),
        learning_rate=algorithm.learning_rate, counters=dict(counters), rng=rng_state(),
        simulator_state=None)
    wire = encode_checkpoint(state, torch.Tensor)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        # 1.13.1's restricted unpickler implements only a subset of protocol 2.
        # All scalar/container metadata is inside one JSON string, not pickle.
        torch.save(wire, temporary, pickle_protocol=2)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path, algorithm, resume=True, trust_legacy=False):
    """Restricted loading by default; never automatically retry unrestricted.

    trust_legacy=True permits arbitrary pickle execution for trusted files,
    including legacy v1/v2 and runtimes without restricted loading support.
    Restricted loading is not a sandbox: only accept files from known sources.

    Evaluation checks network/normalization compatibility but permits different
    batch sizes. Resume checks all PPO settings, restores optimizer/LR/scaler/RNG,
    and returns counters. Construct a NEW environment/collector on resume.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # CPU loading keeps CPU RNG bytes and permits CPU evaluation of CUDA models.
    if type(trust_legacy) is not bool:
        raise ValueError("trust_legacy must be boolean")
    if not _HAS_WEIGHTS_ONLY and not trust_legacy:
        raise ValueError("This PyTorch has no weights_only restricted loader; no deserialization attempted. "
                         "Only for a trusted checkpoint, explicitly set trust_legacy=True "
                         "(launcher: trust_legacy_checkpoint=True).")
    if trust_legacy:
        warnings.warn("Unrestricted checkpoint loading explicitly enabled; pickle may execute code. "
                      "Only load a checkpoint whose source you trust.", RuntimeWarning, stacklevel=2)
    try:
        options = {"weights_only": not trust_legacy} if _HAS_WEIGHTS_ONLY else {}
        state = torch.load(path, map_location="cpu", **options)
    except (pickle.UnpicklingError, RuntimeError) as error:
        if trust_legacy:
            raise
        raise ValueError("Restricted checkpoint loading failed; no unrestricted retry was attempted. "
                         "For a trusted legacy v1/v2 file only, explicitly set trust_legacy=True "
                         "(launcher: trust_legacy_checkpoint=True).") from error
    if not isinstance(state, dict) or type(state.get("format_version")) is not int or state["format_version"] not in (1, 2, FORMAT_VERSION):
        raise ValueError("Not a supported custom PPO checkpoint; RL-Games checkpoint conversion is not supported")
    if state["format_version"] == 1 and not trust_legacy:
        raise ValueError("Legacy v1 checkpoint requires explicit trust_legacy=True")
    if state["format_version"] == FORMAT_VERSION:
        state = decode_checkpoint(state, torch.Tensor)
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
    fresh_scaler = None
    if resume:
        if any(group["lr"] != state["learning_rate"] for group in state["optimizer"]["param_groups"]):
            raise ValueError("Checkpoint optimizer learning rate does not match saved learning_rate")
        cuda_rng = state["rng"].get("cuda")
        if cuda_rng is not None and torch.cuda.is_available() and len(cuda_rng) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG device count differs; exact RNG restoration is unavailable")
        if algorithm.scaler.is_enabled() and not state["scaler"]:
            # Never keep a possibly used destination scaler's growth history.
            fresh_scaler = torch.cuda.amp.GradScaler(enabled=True)
            warnings.warn("Checkpoint AMP scaler was disabled; initializing a fresh enabled scaler. "
                          "Compatible model, optimizer, normalizers and RNG state will be restored; "
                          "AMP growth history cannot be continued.", RuntimeWarning, stacklevel=2)
    algorithm.model.load_state_dict(state["model"], strict=True)
    for name in ("observation_normalizer", "value_normalizer"):
        module = getattr(algorithm, name)
        if module is not None:
            module.load_state_dict(state["normalizers"][name], strict=True)
            module.eval()
    if resume:
        algorithm.optimizer.load_state_dict(state["optimizer"])
        if fresh_scaler is not None:
            algorithm.scaler = fresh_scaler
        else:
            algorithm.scaler.load_state_dict(state["scaler"])
        restore_rng(state["rng"])
    algorithm.model.eval()
    return state["counters"]
