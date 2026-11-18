# Getting started: installation, training and evaluation

This is the current installation guide for the independent PyTorch PPO
application. It keeps the original hexapod task, physics, assets and YAML
settings. Stage-specific documents remain historical development records.

The steps below are instructions for your Linux training machine. Native
Isaac Gym/CUDA execution of this application remains **unverified**; offline
tests do not certify the native combination or reproduce the paper's results.

## 1. Prerequisites and the provisional runtime

| Component | Selected target |
| --- | --- |
| Ubuntu | 20.04 |
| Python | 3.8 |
| PyTorch | 1.13.1, CUDA 11.7 build |
| Isaac Gym | Preview 4, installed separately |
| IsaacGymEnvs | `aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3` |
| Simulation | GPU PhysX, GPU pipeline |
| Learning | Independent PyTorch PPO on GPU |

Use one prepared Python environment for all installation and launch commands.
Keep an existing working CUDA-enabled PyTorch installation intact. If it differs
from this target, review compatibility before proceeding; the package's broad
`torch>=1.13` declaration does not certify every newer release with Preview 4.

The separate CPU CI configuration is Ubuntu 22.04, Python 3.10,
`torch==1.13.1+cpu`, NumPy 1.24.4 and OmegaConf 2.3.0. It tests software without
Isaac Gym. Do not install that CPU wheel into the GPU training environment or
copy its NumPy pin into a native environment without checking SDK compatibility.

You need Git and Python 3.8-compatible build tooling for editable installation:
pip with PEP 660 support, setuptools with editable hooks (version 64 or later),
and wheel as required by the selected backend. The project's declared
`setuptools>=61` floor alone is insufficient for that editable workflow.
Choose versions that still support Python 3.8; do not blindly upgrade to the
latest tools. [Setuptools editable-hook release](https://setuptools.pypa.io/en/latest/history.html#v64-0-0).

## 2. Prepare Isaac Gym Preview 4 separately

Obtain the SDK from the [NVIDIA archive](https://developer.nvidia.com/isaac-gym/download)
and follow its bundled `docs/index.html` and installation instructions. Those
instructions take precedence for native dependencies. This repository does not
download or install the SDK, drivers, CUDA toolkits or PyTorch.

The selected PyTorch CUDA build is available in the
[official historical installation instructions](https://pytorch.org/get-started/previous-versions/).
Its availability does not prove compatibility with the SDK bindings, driver or
toolchain used to build native extensions. If SDK requirements conflict with
the proposed target, resolve that conflict before training.

Check the actual SDK's Python requirements and helper imports. Older SDK
helpers can reference `np.float`, which NumPy 1.24 removed. This can prevent
task import even when CPU tests pass. Select a NumPy version compatible with
your unchanged SDK; this guide does not prescribe an unverified native pin.
Check terrain-helper dependencies such as SciPy as well.
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

This is the external package installation, separate from installing our PPO
package. The flags prevent pip from resolving runtime dependencies or fetching
an isolated build environment. They do not supply missing prerequisites.

The upstream package declares dependencies for many other tasks and RL-Games.
Our launcher bypasses those broad initializers and does not use that trainer.
Avoid installing the full dependency set just to satisfy unrelated task
requirements: it can pull in RL-Games, Warp, PySDF, URDFPy and other packages.
`python -m pip check` may report omitted upstream declarations after a deliberate
`--no-deps` installation. Review each report against the required runtime;
do not automatically install everything or dismiss genuine conflicts.

## 4. Clone this repository

```bash
cd "$HOME/hexapod-work"
git clone https://github.com/Leo813/hexapod_proj.git
cd hexapod_proj
```

Run the remaining commands from this repository root. The installer and help
command use the standard library and can run before native dependencies are
available:

```bash
python -B scripts/install_hexapod_task.py --inspect-environment
python train_agent.py --help
```

Inventory reports describe the selected Python environment. Passing metadata
checks does not prove that native libraries can import or execute.

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

Supply the absolute checkout root, not its `isaacgymenvs` subdirectory. Use
physical paths outside this repository; symlinks and traversal paths are
rejected. Preview writes nothing. Apply creates missing files and preserves
identical files. A conflicting file blocks installation; there is no force
overwrite. Inspect conflicts and deliberately resolve them or use a fresh
checkout, then preview again. The filesystem must support hard links.

The manifest contains 35 files: three task/configuration files plus the entire
32-file `hexapod2` tree, including 25 STL meshes. The utility installs task files
only; it does not install Python packages or the simulator.

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

This preserves the selected PyTorch build. OmegaConf and the other runtime
prerequisites must already be installed; `--no-deps` does not install the
optional integration extra's dependencies.

Check discovery without importing the simulator or tensor libraries:

```bash
python -c "import importlib.util; s = importlib.util.find_spec('isaacgymenvs'); print(s); assert s is not None and s.submodule_search_locations"
python -c "import importlib.util; print(importlib.util.find_spec('hexapod_rl'))"
```

The first result should point to the intended external checkout, with its
copied task/YAMLs and adjacent assets. **Cloning and copying alone do not make
IsaacGymEnvs discoverable.** The installer does not remember a checkout path
for later training. The root entry point adds this repository's `src/` to its
Python path, but that does not register the external package.

If discovery is unavailable or selects the wrong checkout, the existing
explicit-path override is:

```bash
python train_agent.py isaacgymenvs_path=/absolute/path/to/IsaacGymEnvs
```

Repeat that override for resume and evaluation if needed. Use a fresh process:
the launcher initializes Isaac Gym before importing PyTorch, then loads only
the installed `Anymal10` task. Do not import the broad IsaacGymEnvs registry or
pre-import torch in the same interactive process.

## 7. Start training

After installation and native compatibility checks on your machine:

```bash
python train_agent.py
```

This starts actual simulation and automatically selects task `hexapod`, class
`Anymal10`, and configuration `hexapodPPO`. It uses our independent PyTorch PPO.
No explicit `task=hexapod` argument is required, although it remains accepted.
Other tasks and LSTM training are unsupported.

| Default | Value |
| --- | --- |
| Parallel environments | 256 |
| Observations / policy actions | 104 / 18; the task couples six mimic-joint actions |
| Actor and critic | Separate MLPs, 512-256-128 hidden layers, ELU |
| Simulation / learning | GPU PhysX, `pipeline=gpu`, `sim_device=cuda:0`, `rl_device=cuda:0` |
| Maximum epochs | 600 |
| Viewer | `headless=False` |
| PPO | Existing YAML hyperparameters: horizon 48, minibatch 12288, five mini-epochs |

Displayless GPU servers should disable the viewer:

```bash
python train_agent.py headless=True
```

For a smaller, bounded experiment:

```bash
python train_agent.py num_envs=64 max_iterations=100
```

Add `headless=True` on a displayless server. With this smaller environment
count, the loader reduces the oversized default minibatch to the rollout size
(3072) and prints the adjustment. This changes the experiment's batch size.
`max_iterations` counts training epochs, not individual physics steps.

The preserved task has three boxes and rough terrain, but fixes commands to
forward motion. Box-position reward terms and box-fall termination checks are
inactive. It does not implement the paper's full curriculum or establish
successful object balancing. See the [baseline contract](baseline-contract.md).

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

Checkpoint creation follows the existing historical gates. Periodic saves
require completed-episode statistics, an epoch divisible by `save_frequency`,
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
model, optimizer/learning rate, normalizers, compatible AMP scaler and RNG
state, and counters. Simulator episodes start fresh. Without `output_dir`,
resume creates a new timestamped output location. To continue writing to a
chosen directory, pass `output_dir=/path/to/run` explicitly.

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

Current custom checkpoints use version 3 and restricted loading on PyTorch
1.13.1. Historical RL-Games checkpoints are not supported. The loader never
automatically falls back to unrestricted deserialization. For a genuinely
trusted legacy custom checkpoint only, `trust_legacy_checkpoint=True` permits
unrestricted pickle loading, which can execute code. Do not enable it simply
to bypass an unexplained load error. See the
[checkpoint compatibility record](stage3-audit-corrections.md).

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
| Old torch 1.10.x notices in stage records/readiness output | Those describe the earlier candidate; this guide records the current provisional target. They do not certify native compatibility. |

## Validation boundaries and further reading

The strict [CPU workflow](../.github/workflows/pytorch-tests.yml) requires at
least 209 tests, all executed, with no failures, errors or skips. Local tests
may skip when dependencies are absent; those results are not full CI success.
The [README testing status](../README.md#testing-status) records the local scope.

Native SDK/PyTorch binary compatibility, mesh URI resolution, rigid-body and
sensor ordering, GPU reset behavior and CUDA checkpoint restoration still
require execution on the target machine. No successful native training or
paper-result reproduction is claimed.

For detailed historical evidence, see [safe installation](stage4-installation.md),
[native validation preparation](stage4-native-validation.md), and
[reference implementations](reference-implementation.md). Older runtime tables
and test counts describe their original stages; this guide is the current
installation entry point.
