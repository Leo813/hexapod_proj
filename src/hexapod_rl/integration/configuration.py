"""Read installed YAML with OmegaConf, without importing Isaac Gym or torch.

This deliberately supports one task and dot-list overrides, not Hydra sweeps,
plugins, task registries, or arbitrary training backends.
"""

from dataclasses import dataclass, fields
import importlib.util
import math
from pathlib import Path
import re

from ..config import ActorCriticConfig, PPOConfig


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int = 600
    save_frequency: int = 50
    save_best_after: int = 500
    score_to_win: float = 10000.0
    print_stats: bool = True
    games_to_track: int = 100

    def __post_init__(self):
        if type(self.max_epochs) is not int or (self.max_epochs <= 0 and self.max_epochs != -1):
            raise ValueError("max_epochs must be positive, or -1 for unlimited training")
        for name in ("save_frequency", "save_best_after", "games_to_track"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == "games_to_track" else 0):
                raise ValueError(f"Invalid {name}")
        if type(self.print_stats) is not bool or not math.isfinite(self.score_to_win):
            raise ValueError("print_stats must be boolean and score_to_win finite")


@dataclass
class ApplicationConfig:
    package_path: Path
    config_path: Path
    resolved: dict
    network: ActorCriticConfig
    ppo: PPOConfig
    training: TrainingConfig
    adjustments: tuple = ()
    notices: tuple = ()


def parse_arguments(arguments):
    result = {}
    for arg in arguments:
        if "=" not in arg:
            raise ValueError(f"Expected key=value, got {arg!r}; use --help for supported syntax")
        key, value = arg.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9.]*", key) or key in result:
            raise ValueError(f"Invalid or duplicate override: {key!r}")
        result[key] = value
    for key, expected in (("task", "hexapod"), ("train", "hexapodPPO")):
        if result.get(key, expected) != expected:
            raise ValueError(f"Only {key}={expected} is supported (no LSTM)")
    return result


def locate_installation(explicit=None, config_path=None):
    if explicit:
        package = Path(explicit).expanduser().resolve()
        if (package / "isaacgymenvs").is_dir():
            package /= "isaacgymenvs"
    else:
        spec = importlib.util.find_spec("isaacgymenvs")
        if spec is None or not spec.submodule_search_locations:
            raise FileNotFoundError("IsaacGymEnvs not found. Install the historical checkout separately or set isaacgymenvs_path=/absolute/IsaacGymEnvs")
        package = Path(next(iter(spec.submodule_search_locations))).resolve()
    cfg = Path(config_path).expanduser().resolve() if config_path else package / "cfg"
    required = (package / "tasks/base/vec_task.py", package / "tasks/hexapod.py",
                cfg / "config.yaml", cfg / "task/hexapod.yaml", cfg / "train/hexapodPPO.yaml")
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}. Follow the manual placement in docs/stage3-integration.md")
    return package, cfg


def validate_devices(root):
    for key in ("sim_device", "rl_device"):
        if not isinstance(root[key], str) or not re.fullmatch(r"cpu|cuda:\d+", root[key]):
            raise ValueError(f"{key} must be cpu or cuda:<nonnegative index>")
    if root["pipeline"] not in ("cpu", "gpu"):
        raise ValueError("pipeline must be cpu or gpu")
    if root["pipeline"] == "gpu" and root["sim_device"] == "cpu":
        raise ValueError("GPU pipeline requires a CUDA sim_device; use pipeline=cpu for CPU simulation")
    if root["physics_engine"] != "physx":
        raise ValueError("Only the historical PhysX hexapod configuration is supported")
    graphics = root["graphics_device_id"]
    if type(graphics) is not int or graphics < -1:
        raise ValueError("graphics_device_id must be an integer >= -1")
    if not root["headless"] and graphics < 0:
        raise ValueError("A viewer requires graphics_device_id >= 0")


def _expect(mapping, key, value):
    if mapping.get(key) != value or (type(value) is bool and type(mapping.get(key)) is not bool):
        raise ValueError(f"Unsupported {key}={mapping.get(key)!r}; required {value!r}")


def validate_clipping(env):
    """VecTask accepts zero and positive infinity; reject invalid clamp bounds."""
    for name in ("clipActions", "clipObservations"):
        value = env.get(name, float("inf"))
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or math.isnan(value) or value < 0):
            raise ValueError(f"task.env.{name} must be a nonnegative number (positive infinity is allowed)")


def ineffective_setting_notices(task):
    """Warn on non-baseline values without rejecting historical task YAMLs.

    These fields remain untouched. This table describes the unchanged task,
    not alternative physics settings to apply on its behalf.
    """
    settings = (
        ("envSpacing", 1.0, "the task uses spacing 1.0"),
        ("control.controlFrequencyInv", 1, "VecTask reads env.controlFrequencyInv, not this nested field"),
        ("urdfAsset.collapseFixedJoints", True, "the task always collapses fixed joints"),
        ("urdfAsset.fixBaseLink", False, "the task always uses a free base"),
        ("urdfAsset.defaultDofDriveMode", 4, "the task sets position drive mode directly"),
        ("baseInitState.rot", [0., 0., 0., 1.], "actor creation uses the identity rotation"),
        ("baseInitState.vLinear", [0., 0., 0.], "the configured initial velocity is not applied"),
        ("baseInitState.vAngular", [0., 0., 0.], "the configured initial velocity is not applied"),
        ("randomCommandVelocityRanges.linear_x", [-.5, .5], "final commands are fixed to [0, 1, 0, -1]"),
        ("randomCommandVelocityRanges.linear_y", [-1., 1.], "sampled commands are overwritten with [0, 1, 0, -1]"),
        ("randomCommandVelocityRanges.yaw", [-1., 1.], "sampled commands are overwritten with [0, 1, 0, -1]"),
        ("learn.linearVelocityXYRewardScale", 1.0, "the active linear reward uses hard-coded coefficients"),
        ("learn.angularVelocityZRewardScale", .5, "the active angular reward uses hard-coded coefficients"),
    )
    notices = []
    for path, baseline, explanation in settings:
        value = task["env"]
        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            if value != baseline:
                notices.append(f"task.env.{path}={value!r} does not control the requested behavior: {explanation}.")
    return tuple(notices)


def translate_config(root, shrink_minibatch=False):
    """Translate a resolved mapping; callable without OmegaConf or simulator."""
    try:
        task, params = root["task"], root["train"]["params"]
        network, train = params["network"], params["config"]
        validate_devices(root)
        validate_clipping(task["env"])
        if set(params) - {"seed", "algo", "model", "network", "load_checkpoint", "load_path", "config"}:
            raise ValueError("Unsupported trainer params outside the feedforward baseline")
        if params["seed"] != root["seed"]:
            raise ValueError("Use root seed= to set both task and training seed")
        if params["load_checkpoint"] != bool(root["checkpoint"]) or params["load_path"] != root["checkpoint"]:
            raise ValueError("Use root checkpoint= to control loading; train.params checkpoint overrides are inconsistent")
        _expect(task, "name", "Anymal10")
        _expect(params["algo"], "name", "a2c_continuous")
        _expect(params["model"], "name", "continuous_a2c_logstd")
        _expect(network, "name", "actor_critic")
        _expect(network, "separate", True)
        if set(network) - {"name", "separate", "space", "mlp"}:
            raise ValueError("Unsupported network settings; only separate feedforward MLPs are supported")
        mlp = network["mlp"]
        if set(mlp) - {"units", "activation", "d2rl", "initializer", "regularizer"}:
            raise ValueError("Unsupported MLP settings")
        _expect(mlp, "d2rl", False)
        _expect(mlp["initializer"], "name", "default")
        if mlp["regularizer"].get("name") not in (None, "None"):
            raise ValueError("MLP regularization is unsupported")
        if set(network["space"]) != {"continuous"}:
            raise ValueError("Only a continuous action space is supported")
        continuous = network["space"]["continuous"]
        if set(continuous) - {"mu_activation", "sigma_activation", "mu_init", "sigma_init", "fixed_sigma"}:
            raise ValueError("Unsupported continuous policy settings")
        for name in ("mu_activation", "sigma_activation"):
            if continuous[name] not in (None, "None"):
                raise ValueError(f"{name} must be None")
        _expect(continuous, "fixed_sigma", True)
        _expect(continuous["mu_init"], "name", "default")
        _expect(continuous["sigma_init"], "name", "const_initializer")
        for key, size in (("numObservations", 104), ("numActions", 18)):
            if task["env"].get(key, size) != size:
                raise ValueError(f"Anymal10 requires {key}={size}")
        if task["env"].get("numStates", 0) != 0 or task["env"].get("numAgents", 1) != 1:
            raise ValueError("Privileged observations and multiagent training are unsupported")
        if task["task"].get("randomize", False):
            raise ValueError("Stage 3 supports the baseline with framework randomization disabled")
        if task["physics_engine"] != root["physics_engine"] or task["sim"]["use_gpu_pipeline"] != (root["pipeline"] == "gpu"):
            raise ValueError("Task simulation configuration disagrees with root physics_engine/pipeline")
        if task["sim"]["physx"]["use_gpu"] != root["sim_device"].startswith("cuda:"):
            raise ValueError("Task PhysX use_gpu disagrees with sim_device")
        model = ActorCriticConfig(hidden_sizes=tuple(mlp["units"]), activation=mlp["activation"],
                                 initial_log_std=continuous["sigma_init"]["val"])
        for name, expected in (("ppo", True), ("multi_gpu", False)):
            _expect(train, name, expected)
        for name, expected in (("schedule_type", "legacy"), ("bound_loss_type", "bound"),
                               ("normalize_rms_advantage", False), ("clip_actions", True)):
            if train.get(name, expected) != expected:
                raise ValueError(f"Only {name}={expected!r} is supported")
        ppo_fields = {f.name for f in fields(PPOConfig)}
        controls = {f.name for f in fields(TrainingConfig)}
        metadata = {"name", "full_experiment_name", "env_name", "ppo", "multi_gpu", "reward_shaper",
                    "seq_len", "schedule_type", "bound_loss_type", "normalize_rms_advantage", "clip_actions"}
        unknown = set(train) - ppo_fields - controls - metadata
        if unknown:
            raise ValueError(f"Unsupported training settings: {sorted(unknown)}")
        _expect(train, "env_name", "rlgpu")  # historical metadata, not a registry call
        if train.get("full_experiment_name", train["name"]) != train["name"]:
            raise ValueError("Use experiment= (or config.name) for the run name; a distinct full_experiment_name is unsupported")
        if set(train["reward_shaper"]) != {"scale_value"}:
            raise ValueError("Only reward_shaper.scale_value is supported")
        required = {"num_actors", "horizon_length", "minibatch_size", "mini_epochs", "gamma", "tau",
                    "e_clip", "critic_coef", "entropy_coef", "bounds_loss_coef", "clip_value",
                    "learning_rate", "truncate_grads", "grad_norm", "mixed_precision", "normalize_input",
                    "normalize_value", "normalize_advantage", "value_bootstrap", "lr_schedule", "kl_threshold"}
        missing = required - set(train)
        if missing:
            raise ValueError(f"Missing PPO YAML settings: {sorted(missing)}")
        kwargs = {key: train[key] for key in ppo_fields & set(train)}
        if "reward_scale" in train and (isinstance(train["reward_scale"], bool)
                or not isinstance(train["reward_scale"], (int, float))
                or train["reward_scale"] != train["reward_shaper"]["scale_value"]):
            raise ValueError("Conflicting reward_scale and reward_shaper.scale_value; use the historical reward_shaper.scale_value")
        kwargs["reward_scale"] = train["reward_shaper"]["scale_value"]
        if kwargs["num_actors"] != task["env"]["numEnvs"]:
            raise ValueError("num_actors must match task.env.numEnvs")
        adjustments = []
        batch_size = kwargs["num_actors"] * kwargs["horizon_length"]
        if shrink_minibatch and kwargs["minibatch_size"] > batch_size:
            adjustments.append(f"num_envs override: minibatch_size {kwargs['minibatch_size']} -> {batch_size} (one full rollout)")
            kwargs["minibatch_size"] = batch_size
            train["minibatch_size"] = batch_size
        ppo = PPOConfig(**kwargs)
        # Required local control fields must not silently fall back to defaults.
        control = TrainingConfig(**{key: train[key] for key in controls if key in train})
        for key in controls - {"games_to_track"}:
            if key not in train:
                raise ValueError(f"Missing training control: {key}")
        for key in ("headless", "test", "torch_deterministic", "force_render"):
            if type(root[key]) is not bool:
                raise ValueError(f"{key} must be boolean")
        for key in ("multi_gpu", "capture_video", "wandb_activate"):
            if root.get(key, False) is not False:
                raise ValueError(f"{key} is unsupported")
        if root.get("sigma", "") != "":
            raise ValueError("sigma overrides are unsupported; restore the policy from a custom checkpoint")
        if type(root["seed"]) is not int or root["seed"] < -1:
            raise ValueError("seed must be -1 or a nonnegative integer")
        if type(root["eval_steps"]) is not int or root["eval_steps"] <= 0:
            raise ValueError("eval_steps must be positive")
        if type(root["tensorboard"]) is not bool:
            raise ValueError("tensorboard must be boolean")
        if type(root.get("trust_legacy_checkpoint", False)) is not bool:
            raise ValueError("trust_legacy_checkpoint must be boolean")
        if root["test"] and not root["checkpoint"]:
            raise ValueError("test=True requires checkpoint=/path/to/custom.pt")
        if root["checkpoint"] and not Path(root["checkpoint"]).expanduser().is_file():
            raise FileNotFoundError(f"Checkpoint not found: {root['checkpoint']}")
        return model, ppo, control, tuple(adjustments)
    except KeyError as error:
        raise ValueError(f"Missing required configuration field: {error}") from error


def load_configuration(arguments):
    overrides = parse_arguments(arguments)
    package, cfg_dir = locate_installation(overrides.pop("isaacgymenvs_path", None),
                                          overrides.pop("config_path", None))
    try:
        from omegaconf import OmegaConf
    except ImportError as error:
        raise ImportError("Configuration loading requires OmegaConf; install the project's [integration] extra in your training environment") from error
    resolvers = {"eq": lambda x, y: str(x).lower() == str(y).lower(),
                 "contains": lambda x, y: str(x).lower() in str(y).lower(),
                 "if": lambda pred, a, b: a if pred else b,
                 "resolve_default": lambda default, arg: default if arg == "" else arg}
    for name, resolver in resolvers.items():
        OmegaConf.register_new_resolver(name, resolver, replace=True)
    root = OmegaConf.load(cfg_dir / "config.yaml")
    # Explicitly compose only the two selected groups. No registry/PBT/Hydra jobs.
    for key in ("defaults", "hydra"):
        root.pop(key, None)
    root.task = OmegaConf.load(cfg_dir / "task/hexapod.yaml")
    root.train = OmegaConf.load(cfg_dir / "train/hexapodPPO.yaml")
    for key, value in dict(output_dir="", eval_steps=1000, tensorboard=False,
                           trust_legacy_checkpoint=False).items():
        if key not in root:
            root[key] = value
    overrides.pop("task", None)
    overrides.pop("train", None)
    for key in overrides:
        sentinel = object()
        if OmegaConf.select(root, key, default=sentinel) is sentinel:
            raise ValueError(f"Unknown override: {key}; Hydra sweeps and +key syntax are not supported")
    root = OmegaConf.merge(root, OmegaConf.from_dotlist([f"{key}={value}" for key, value in overrides.items()]))
    resolved = OmegaConf.to_container(root, resolve=True)
    model, ppo, training, adjustments = translate_config(resolved,
        shrink_minibatch="num_envs" in overrides and "train.params.config.minibatch_size" not in overrides)
    return ApplicationConfig(package, cfg_dir, resolved, model, ppo, training, adjustments,
                             ineffective_setting_notices(resolved["task"]))
