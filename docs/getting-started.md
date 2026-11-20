# Getting started: installation, training and evaluation

Follow these steps on your Linux training machine to install the hexapod task
and train or evaluate a PPO policy. Native Isaac Gym/CUDA execution of this
application remains **unverified**; CPU tests do not establish compatibility
or reproduction of the paper's results.

## 1. Prerequisites and the provisional runtime

| Component | Selected target |
| --- | --- |
| Ubuntu | 20.04 |
| Python | 3.8 |
| PyTorch | 1.13.1, CUDA 11.7 build |
| Isaac Gym | Preview 4, installed separately |
| IsaacGymEnvs | `aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3` |
| Simulation | GPU PhysX, GPU pipeline |
| Learning | PyTorch PPO on GPU |

Use one prepared Python environment for all installation and launch commands.
Keep an existing working CUDA-enabled PyTorch installation intact. If it differs
from this target, review compatibility before proceeding; the package's broad
`torch>=1.13` declaration does not certify every newer release with Preview 4.

The [CPU test environment](../.github/workflows/pytorch-tests.yml) is separate
from the native runtime. Do not install its CPU PyTorch wheel into the GPU
environment or copy its NumPy pin without checking SDK compatibility.

You need Git and Python 3.8-compatible build tooling for editable installation:
pip with PEP 660 support, setuptools with editable hooks (version 64 or later),
and wheel as required by the selected backend. The project's declared
`setuptools>=61` floor alone is insufficient for that editable workflow.
Choose versions that still support Python 3.8; do not blindly upgrade to the
latest tools. [Setuptools editable-hook release](https://setuptools.pypa.io/en/latest/history.html#v64-0-0).

## 2. Prepare Isaac Gym Preview 4 separately

Obtain the SDK from the [NVIDIA archive](https://developer.nvidia.com/isaac-gym/download)
and follow its bundled `docs/index.html` and installation instructions. Those
instructions take precedence for native dependencies. Install the SDK,
drivers, CUDA tooling, and PyTorch separately from this repository.

The selected PyTorch CUDA build is available in the
[official historical installation instructions](https://pytorch.org/get-started/previous-versions/).
Resolve any conflicts between the provisional target and the SDK's binding,
driver, or native-extension requirements before training.

Check the SDK's Python and terrain-helper requirements, including SciPy.
Older helpers may use `np.float`, removed in NumPy 1.24, preventing task import.
Select a NumPy version compatible with your unchanged SDK; this guide does
not prescribe a native pin.
[Preview 4 failure report](https://github.com/isaac-sim/IsaacGymEnvs/issues/183),
[NumPy alias removal](https://numpy.org/doc/1.24/release/1.24.0-notes.html#expired-deprecations).

Before proceeding, satisfy these application prerequisites in the prepared
environment, including their own dependencies:

| Used by | Requirements |
| --- | --- |
| Historical VecTask and its imported utilities | Isaac Gym, PyTorch, NumPy, `gym==0.23.1` |
| Original hexapod task | matplotlib, TensorBoard via `torch.utils.tensorboard`, SDK tensor/terrain helpers |
| Custom configuration loader | OmegaConf `>=2.2,<2.4`; 2.3.0 is the offline reference |
| Custom PPO and runtime setup | PyTorch and NumPy |

The task imports matplotlib and creates a TensorBoard writer even when
`tensorboard=False`. Do not omit these dependencies on that basis. The complete
SDK dependency set must come from the SDK itself, not this table.

## 3. Prepare the historical IsaacGymEnvs checkout

For a new workspace, the examples use sibling checkouts under
`$HOME/hexapod-work`. Substitute your own absolute paths consistently.

```bash
mkdir -p "$HOME/hexapod-work"
cd "$HOME/hexapod-work"
git clone https://github.com/isaac-sim/IsaacGymEnvs.git
git -C IsaacGymEnvs checkout aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3
```

These commands are for a new checkout; do not replace an existing modified
installation. Keep the full checkout, including `isaacgymenvs/tasks/base/`,
`isaacgymenvs/utils/` and `isaacgymenvs/cfg/config.yaml`.

Once the required runtime and build prerequisites are already satisfied,
register this checkout with the active interpreter:

```bash
python -m pip install --no-deps --no-build-isolation --editable "$HOME/hexapod-work/IsaacGymEnvs"
```

This registers the external package separately from the PPO package. The flags
skip dependency resolution and build isolation, so prerequisites must already
be installed. Use the same flags for the PPO package below to preserve your
prepared PyTorch stack.

IsaacGymEnvs also declares dependencies for unrelated tasks and trainers that
this launcher does not use. `python -m pip check` may report these omissions
after a `--no-deps` installation. Review reports against the runtime requirements
above; do not automatically install every upstream dependency or dismiss
genuine conflicts.

## 4. Clone this repository

```bash
cd "$HOME/hexapod-work"
git clone https://github.com/Leo813/hexapod_proj.git
cd hexapod_proj
```

Run the remaining commands from this repository root. These optional inventory
and help commands work before native dependencies are installed:

```bash
python -B scripts/install_hexapod_task.py --inspect-environment
python train_agent.py --help
```

Inventory reports do not prove that native libraries can import or execute.

## 5. Install the original task and assets

Choose either method. Both produce the same required layout; neither needs a
change to the IsaacGymEnvs task registry or its original training launcher.

### Method A: task installation utility

Preview the changes, apply them, then verify:

```bash
python -B scripts/install_hexapod_task.py --isaacgymenvs "$HOME/hexapod-work/IsaacGymEnvs"
python -B scripts/install_hexapod_task.py --isaacgymenvs "$HOME/hexapod-work/IsaacGymEnvs" --apply
python -B scripts/install_hexapod_task.py --isaacgymenvs "$HOME/hexapod-work/IsaacGymEnvs" --verify
```

Supply the absolute checkout root, not its `isaacgymenvs` subdirectory. Use a
physical path outside this repository; symlinks and traversal paths are rejected,
and the filesystem must support hard links. Preview writes nothing; apply creates
missing files and preserves identical files. Conflicts block installation with
no force-overwrite option. Resolve them or use a fresh checkout, then preview again.

Use a trusted checkout that is not being changed concurrently. An I/O failure
can leave a partial installation; preview again before retrying. Verification
checks file contents, not native runtime compatibility. A revision mismatch is
a warning rather than proof of incompatibility.

The utility installs the task, two configurations, and complete `hexapod2` asset
tree. It does not install Python packages or the simulator.

### Method B: manual copying

Copy these files and the complete directory, preserving names and contents:

```text
hexapod_proj/hexapod.py
    -> IsaacGymEnvs/isaacgymenvs/tasks/hexapod.py

hexapod_proj/hexapod.yaml
    -> IsaacGymEnvs/isaacgymenvs/cfg/task/hexapod.yaml

hexapod_proj/hexapodPPO.yaml
    -> IsaacGymEnvs/isaacgymenvs/cfg/train/hexapodPPO.yaml

hexapod_proj/hexapod2/
    -> IsaacGymEnvs/assets/urdf/hexapod2/
```

Create missing destination directories as needed. Inspect any existing files
before copying; do not overwrite unrelated changes. Preserve the whole asset
tree, including `urdf/`, `meshes/` and package metadata. Avoid an accidental
nested `hexapod2/hexapod2/` directory.

**Keep the existing `IsaacGymEnvs/isaacgymenvs/cfg/config.yaml` unchanged.**
It supplies root settings used by the task/train YAML interpolation. Do not
replace it with `hexapod.yaml` or copy the LSTM configuration for this workflow.
Do not edit the task or URDF to accommodate a different layout.

Verify manual copying with the same utility:

```bash
python -B scripts/install_hexapod_task.py --isaacgymenvs "$HOME/hexapod-work/IsaacGymEnvs" --verify
```

Verification checks file placement and content. Native mesh loading remains
unverified: the URDF uses `package://test_spider6/meshes/...`, matching its
package name, while the directory is named `hexapod2`. All referenced mesh
filenames are supplied, but importer URI resolution needs a native check.

## 6. Install the custom PPO package and verify discovery

After preparing the dependencies above, install the custom package separately:

```bash
python -m pip install --no-deps --no-build-isolation --editable "$HOME/hexapod-work/hexapod_proj"
```

OmegaConf and the other runtime prerequisites must already be installed;
`--no-deps` also skips the optional integration extra's dependencies.

Check discovery without importing the simulator or tensor libraries:

```bash
python -c "import importlib.util; s = importlib.util.find_spec('isaacgymenvs'); print(s); assert s is not None and s.submodule_search_locations"
python -c "import importlib.util; print(importlib.util.find_spec('hexapod_rl'))"
```

The first result should point to the intended external checkout, with its
installed task and assets. **Cloning and copying alone do not register
IsaacGymEnvs**, and the installer does not remember its path for training.
If discovery fails or selects the wrong checkout, specify it explicitly:

```bash
python train_agent.py isaacgymenvs_path=/absolute/path/to/IsaacGymEnvs
```

Repeat this override for resume and evaluation if needed. Use a fresh launcher
process so Isaac Gym initializes before PyTorch. Do not pre-import torch or
the broad IsaacGymEnvs registry in the same interactive process.

## 7. Start training

After installation and native compatibility checks on your machine:

```bash
python train_agent.py
```

This starts simulation with task `hexapod` and configuration `hexapodPPO`.
Explicit `task=hexapod` remains accepted. Other tasks and LSTM training are
unsupported.

| Default | Value |
| --- | --- |
| Parallel environments | 256 |
| Simulation / learning | GPU PhysX, `pipeline=gpu`, `sim_device=cuda:0`, `rl_device=cuda:0` |
| Maximum epochs | 600 |
| Viewer | `headless=False` |

Displayless GPU servers should disable the viewer:

```bash
python train_agent.py headless=True
```

For a smaller, bounded experiment:

```bash
python train_agent.py num_envs=64 max_iterations=100
```

Add `headless=True` on a displayless server. The smaller example adjusts the
default minibatch to the rollout size (3072), changing the experiment's batch
size. `max_iterations` counts training epochs, not physics steps.

The preserved task has three boxes and rough terrain, but fixes commands to
forward motion. Box-position reward terms and box-fall termination checks are
inactive. It does not implement the paper's full curriculum or establish
successful object balancing. See the [baseline contract](baseline-contract.md).

### Additional configuration

For other supported settings, run `--help`. The loader accepts `key=value`
overrides of existing supported YAML leaves, not Hydra sweeps or `+key` syntax.
Examples include `seed=42`, `torch_deterministic=True`, `force_render=False`,
`experiment=my_run`, and `train.params.config.learning_rate=0.0001` (a changed
experiment). Unknown/duplicate keys, alternate tasks, LSTM, multi-GPU, video,
W&B and framework domain randomization are rejected.

Configuration precedence is CLI > root YAML > absent-key defaults. The latter
are `output_dir=""`, `eval_steps=1000`, `tensorboard=False`, and
`trust_legacy_checkpoint=False`. `config_path=/absolute/custom/cfg` can select
a directory containing `config.yaml`, `task/hexapod.yaml` and
`train/hexapodPPO.yaml`. The loader preserves their relative interpolation;
Hydra does not change the working directory. Explicit relative launch paths
are interpreted from that directory. A specified minibatch override always
wins over the smaller-environment convenience adjustment and must divide the
rollout size. Devices must be `cpu` or `cuda:<index>`; GPU pipeline requires a
CUDA simulator device. CPU simulation still requires Isaac Gym.

## 8. Outputs and resuming training

By default, outputs are relative to the launch working directory:

```text
runs/Anymal_<timestamp>/
  resolved-config.json
  checkpoints/
    last.pt                 # on normal epoch-limit or winning completion
    best.pt                 # only when the best-reward gate is met
    epoch_<number>.pt        # conditional periodic saves
    winning.pt              # only when the winning condition is met
```

Optional custom TensorBoard logging (`tensorboard=True`) writes `summaries/`
inside that run. The original task also creates its own writer under `runs/`.
Use a writable working directory. `output_dir=/absolute/path/to/run` selects a
custom training output; a new run refuses a nonempty directory.

Checkpoint creation is conditional. Periodic saves require completed-episode
statistics, an epoch divisible by `save_frequency`,
and reward no greater than the stored best reward. Best checkpoints require
an improved reward and epoch >=500. Winning requires a best reward >10000
after that gate. Normal completion at the epoch limit saves `last.pt` even if
no episodes have completed. Resuming at an already-reached limit performs no
updates and does not create another final checkpoint.

**`save_frequency=50` does not guarantee a checkpoint every 50 epochs.**
Default runs can have no checkpoint before the best-save gate. Interruptions
and failures do not trigger an emergency save, so an interrupted run may have
no recoverable checkpoint.

Resume an existing custom checkpoint:

```bash
python train_agent.py checkpoint=/path/to/last.pt max_iterations=700
```

The target is **700 total epochs**, not 700 additional epochs. Resume restores
training state but starts fresh simulator episodes and, by default, a new
timestamped output directory. To select an output directory, pass
`output_dir=/path/to/run` explicitly.

Repeat the original PPO overrides, including `num_envs` and any minibatch
override: resume checks all PPO settings and network configuration. The saved
`resolved-config.json` helps recover those settings. Task/physics overrides
are not automatically restored or fully compared; retain them deliberately.
Application controls such as the epoch limit may change.

## 9. Evaluate a trained agent

```bash
python train_agent.py test=True checkpoint=/path/to/last.pt
```

For displayless evaluation with an explicit duration:

```bash
python train_agent.py test=True checkpoint=/path/to/last.pt headless=True eval_steps=1000
```

Evaluation runs the simulator using deterministic policy means and frozen
normalization, without PPO updates. It reports episode statistics; the default
1000 steps may finish before an episode ends. Keep the trained architecture,
normalization settings and intended task configuration compatible.

Use checkpoints produced by this application; historical RL-Games checkpoints
are unsupported. Current checkpoints use restricted loading on PyTorch 1.13.1.
For trusted legacy application checkpoints only, `trust_legacy_checkpoint=True`
allows unrestricted pickle loading, which can execute code. It is never enabled
automatically; do not use it to bypass an unexplained load error. See the
[checkpoint format and restoration](ppo-implementation.md#checkpoint-format-and-restoration).

## 10. Troubleshooting

| Symptom | Check / next action |
| --- | --- |
| IsaacGymEnvs not found | Use the same Python environment as its editable installation, inspect discovery, or supply `isaacgymenvs_path`. |
| Missing task/YAML/assets | Check the four destinations above and run `--verify`; preserve external root `cfg/config.yaml`. |
| Installer conflict | Inspect differing files; use a fresh checkout or deliberately resolve the conflict. The utility has no overwrite switch. |
| Editable installation fails | Check Python 3.8-compatible pip/setuptools/wheel and PEP 660 support. `--no-build-isolation` requires them to exist already. |
| Missing OmegaConf, gym, matplotlib or TensorBoard | Prepare the required dependency in the selected environment without replacing the native stack. |
| SDK raises an `np.float` error | Inspect SDK requirements and NumPy compatibility. The CPU CI pin is not the native prescription. |
| CUDA unavailable / native import fails | Check the selected Python, CUDA-enabled torch build, SDK binaries and driver using SDK instructions. Do not substitute the CPU CI environment. |
| Viewer fails on a remote server | Use `headless=True`; GPU simulation and learning remain enabled. |
| Isaac Gym must initialize before torch | Start a fresh `python train_agent.py` process. |
| Mesh resolution / body or sensor mismatch | Retain the supplied layout; capture native importer diagnostics for review before changing assets or task code. |
| Resume rejects PPO settings | Repeat the original overrides; inspect `resolved-config.json`. Increasing the total epoch limit is allowed. |
| No checkpoint after interruption | Inspect the actual files; the save gates above may not have produced one. |
| Old torch 1.10.x notice in readiness output | This retained diagnostic notice describes an earlier candidate; the runtime table above is the current provisional target. Metadata notices do not certify native compatibility. |

## Native diagnostics

**Advanced, optional checks.** Use these tools after setup to inspect the
environment or explicitly run bounded native checks. They do not install
dependencies or repair the checkout. Use the target environment's interpreter.

### Environment inventory

Read-only inventory, without importing torch or Isaac Gym:

```bash
python -B scripts/check_native_readiness.py --isaacgymenvs /opt/IsaacGymEnvs --isaacgym-sdk /opt/isaacgym
```

Replace the paths with your installations. Inventory reads SDK metadata and
checks task-file hashes without importing tensor libraries or writing files.
Omitting the checkout reports an installation-verification blocker. Add
`--driver-info` for read-only `nvidia-smi` output. Exit 0 means no detected
metadata blockers, not proven compatibility; blockers/failures exit 1 and
invalid syntax exits 2.

### Import and CUDA check

On the prepared GPU machine, explicitly request a fresh-process import/CUDA check:

```bash
python -B scripts/check_native_readiness.py --isaacgymenvs /opt/IsaacGymEnvs --check-imports --device cuda:0 --timeout 60
```

This imports Isaac Gym before torch and checks a tensor operation on the
selected CUDA device without constructing a simulation. Device indices follow
`CUDA_VISIBLE_DEVICES`; SDK imports may use native extension caches. Metadata
blockers can still cause exit 1 even if the probe succeeds. Passing does not
establish PhysX, asset, or training compatibility.

### Bounded training check

After resolving blockers and reviewing import results, a separate flag starts
the bounded native training diagnostic:

```bash
# /data/hexapod-validation must exist; smoke-001 must not exist.
python -B scripts/check_native_readiness.py \
  --isaacgymenvs /opt/IsaacGymEnvs \
  --run-smoke --device cuda:0 \
  --output /data/hexapod-validation/smoke-001 --timeout 180
```

The output must be a new absolute directory outside both checkouts, with an
existing parent. The worker runs there, including the original task's writer.
Use this wrapper; `scripts/native_smoke.py` is its worker implementation, not a
standalone launcher. Active flags are not enabled by default or by CPU CI.

The diagnostic runs four environments for one epoch with GPU PhysX/pipeline,
headless mode, and a minibatch of 192. It checks tensors, sensor/step finiteness,
joint/body names, optimizer updates, and same-device checkpoint restoration.
It does not test resumed updates or evaluation after loading, and terrain/contact
allocation can still be costly.

Artifacts include `readiness.json`, completed-worker `native-stdout.log` and
`native-stderr.log`, `resolved-config.json`, `smoke-evidence.json`,
`checkpoints/last.pt`, and task `runs/` output. Early failures may omit files;
timeout output tails remain in the readiness report. Timeout must be 1..600
seconds. A timeout kill can prevent graceful native cleanup; inspect the report
and use a fresh output directory for a deliberate retry.

Snapshot copies synchronize CUDA and can alter timing; constructor resets
precede instrumentation, and natural episode/reset coverage may be incomplete.
Sensor/body mappings still need review. Passing does not validate asynchronous
execution, next-update equivalence, CPU-to-CUDA AMP transitions, or evaluation.
Retain diagnostic artifacts for review; do not change physics or assets simply
to make a check pass.

## Validation boundaries and further reading

See [GitHub Actions](https://github.com/Leo813/hexapod_proj/actions) for CPU test
results. Local runs with skipped tests are not equivalent to a successful CI
run. Native compatibility, mesh loading, sensor ordering, GPU resets, and CUDA
checkpoint restoration still require target-machine validation.

For technical contracts and attribution, see the [executable baseline](baseline-contract.md),
[PPO implementation](ppo-implementation.md) and [reference mapping](reference-implementation.md).
