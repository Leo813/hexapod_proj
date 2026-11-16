# Stage 3: custom PPO training application

Date: 2026-11-16. Branch: `feat/custom-ppo-stage3`. Starting commit
`4771c170421333e49ea2effad59ba182380158e3` includes the Stage 2 merge and was
verified identical to GitHub `main` before edits. The initial working tree was
clean. No commit, push, PR, external-source edit, or dependency installation was
performed during this implementation.

The application is implemented for review. **Live Isaac Gym compatibility and GPU
training remain unverified.** Both local Python environments lack torch and
OmegaConf; new numerical/composition tests are implemented but were not executed
successfully here. The user reports that Stages 1/2 previously passed CI; those
results do not validate these unpushed Stage 3 changes.

## Architecture and files

```text
train.py -> launcher -> installed YAML / typed configuration
                    -> Isaac Gym initialization BEFORE torch
                    -> installed Anymal10(VecTask), unchanged
                    -> IsaacGymAdapter
                    -> ActorCritic + PPO (Stages 1/2, unchanged)
                    -> RolloutCollector -> Trainer -> custom checkpoints
                                      \-> deterministic evaluation
```

| Created file | Responsibility |
| --- | --- |
| `train.py` | Repository-root command; source path is relative to its own file |
| `src/hexapod_rl/launcher.py` | Application wiring, seed/device checks, run directory, train/resume/evaluate |
| `src/hexapod_rl/integration/__init__.py` | Import-safe optional integration package |
| `src/hexapod_rl/integration/configuration.py` | Installation discovery, OmegaConf composition/overrides, typed translation and controls |
| `src/hexapod_rl/integration/task_factory.py` | Import-order guard and narrowly scoped loading of the installed task |
| `src/hexapod_rl/integration/isaacgym_adapter.py` | VecTask tensor contract, independent clipped actions, native resource cleanup |
| `src/hexapod_rl/collector.py` | Persistent production rollout collection and completed-episode meters |
| `src/hexapod_rl/trainer.py` | Epoch/frame accounting, logging, historical save/win decisions, cleanup |
| `src/hexapod_rl/checkpoints.py` | Versioned native torch checkpoint format and learning-state restoration |
| `src/hexapod_rl/evaluation.py` | Frozen normalization and deterministic mean actions, without optimization |
| `tests/integration_test_support.py` | Temporary fake installation and test-only mutating/deferred-reset VecTask |
| `tests/test_integration_configuration.py` | 9 dependency-independent launcher checks and 6 OmegaConf tests |
| `tests/test_production_integration.py` | 22 numerical adapter/collector/trainer/checkpoint/evaluation tests |
| `docs/stage3-integration.md` | This guide |

Modified files: `README.md`, `pyproject.toml` (optional `integration` dependency),
`.github/workflows/pytorch-tests.yml`, `ci/run_stage1_tests.py`,
`ci/test_run_stage1_tests.py`, and `tests/test_static_contract.py`. The last change
allows application I/O/configuration imports only in the new application layer;
the original tensor-core import allowlist remains unchanged. No existing test
case was removed. All Stage 1/2 mathematical/network implementation files remain
unchanged, as do `hexapod.py`, both PPO YAMLs, `hexapod.yaml`, and all robot assets.

## Verified historical contracts

The behavioral references remain RL-Games v1.6.0 at
`fa1c13cc10158ab42e829a0971a9dc9a4544b3b9` and IsaacGymEnvs at
`aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3`. These are selected reconstruction
targets, not proven records of the original research installation.

The following code was reinspected before integration:

| Pinned source | Required behavior |
| --- | --- |
| [VecTask and Env](https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/isaacgymenvs/tasks/base/vec_task.py) | Constructor/device rules; reset returns existing clipped observations; step returns obs/reward/reset/extras; set_train_info; no native close method |
| [IsaacGymEnvs package initialization](https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/isaacgymenvs/__init__.py) | Resolver definitions; make() routes through rlgames_utils |
| [Task package initializer](https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/isaacgymenvs/tasks/__init__.py) | Importing tasks normally eagerly imports many unrelated tasks and their dependencies |
| [Root configuration](https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/isaacgymenvs/cfg/config.yaml) and [historical launcher](https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/isaacgymenvs/train.py) | Parent defaults, nested task/train interpolation, devices and import ordering |
| [set_seed](https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/isaacgymenvs/utils/utils.py) | Python/NumPy/torch seed, seed=-1 behavior, cuDNN/deterministic settings |
| [A2CBase.init_tensors/play_steps; ContinuousA2CBase.preprocess_actions/train](https://github.com/Denys88/rl_games/blob/fa1c13cc10158ab42e829a0971a9dc9a4544b3b9/rl_games/common/a2c_common.py) | Initial ones, copying before step, raw episode reward accounting, action transform, frame increments, checkpoint/win gates |
| [AverageMeter](https://github.com/Denys88/rl_games/blob/fa1c13cc10158ab42e829a0971a9dc9a4544b3b9/rl_games/algos_torch/torch_ext.py) | Capped batch-weighted completed-episode mean, default capacity 100 |

No RL-Games class or training function is imported or called. The application
loads the actual installed task and base implementation; it does not implement
another simulator or replace `VecTask`. See [baseline-contract.md](baseline-contract.md)
for original action coupling, reward/reset defects and paper discrepancies; none
are corrected here.

## Manual installation and file placement

Use a compatible Linux environment with Isaac Gym Preview 4, the selected
IsaacGymEnvs checkout, drivers, and a suitable CUDA-enabled torch build installed
separately. The CPU-only GitHub Actions torch build is for offline tests, not a
GPU simulator installation recipe. Exact original Python/CUDA/driver versions
are not known. Verify Isaac Gym's supplied examples in that environment first.

Manually place the unchanged files as follows (the launcher never copies them):

```text
IsaacGymEnvs/
  assets/urdf/hexapod2/
    urdf/test_spider6.urdf
    meshes/...
    ...                         # preserve the original hexapod2 tree
  isaacgymenvs/
    tasks/hexapod.py             # this repository's unchanged hexapod.py
    cfg/config.yaml             # installed historical root configuration
    cfg/task/hexapod.yaml        # this repository's unchanged task YAML
    cfg/train/hexapodPPO.yaml     # this repository's unchanged feedforward YAML
```

The installed task's `__file__` remains in `isaacgymenvs/tasks/`, so its existing
`../../assets` lookup is preserved. No URDF/mesh content or path is rewritten.
The LSTM file is not used. No edit to `isaacgym_task_map` is required.

The original task imports NumPy, matplotlib and `torch.utils.tensorboard`, and
creates its own global SummaryWriter at import time. Those original dependencies
must already be available even when custom `tensorboard=False`; the launcher
does not remove or replace task-side logging. The adapter closes that writer.

The project adds only an optional OmegaConf requirement (`>=2.2,<2.4`), compatible
with the selected checkout's `hydra-core>=1.2` dependency family. In a prepared
training environment, manual project installation can be:

```bash
python -m pip install omegaconf==2.3.0
python -m pip install --no-deps --editable /absolute/path/to/hexapod_proj
```

These commands are instructions, not an automatic installer, and were not run
here. `--no-deps` preserves the already selected torch/simulator stack. If using
the `[integration]` extra instead, ensure dependency resolution preserves your
compatible torch build. Simulator packages are not project runtime dependencies.

## Task loading and import order

The launcher first reads configuration with standard-library/OmegaConf code,
then imports `isaacgym`, then imports torch and the custom learning components.
Importing torch first in an interactive process produces an explicit error;
launch a fresh `python train.py` process instead.

To avoid broad registry/factory imports, the loader establishes process-local
package paths for `isaacgymenvs` and `isaacgymenvs.tasks` without executing their
package initializers. It then imports `isaacgymenvs.tasks.hexapod` normally.
The installed `tasks/base/vec_task.py` and utility modules execute unchanged.
The pinned base initializer is license text only and utils initializer is empty.
No imported source is patched, and no external file is edited. The launcher
owns this namespace for the process and rejects an already-imported IsaacGymEnvs
namespace; it must not be mixed with `isaacgymenvs.make()` in the same process.
This deliberate, narrow loading strategy needs a live smoke test against the
selected checkout, especially if using a different revision.

## Configuration and commands

Run commands from this repository (or invoke its `train.py` by absolute path):

```bash
python train.py task=hexapod
python train.py task=hexapod headless=True
python train.py task=hexapod num_envs=64 max_iterations=100
python train.py task=hexapod isaacgymenvs_path=/opt/IsaacGymEnvs
```

Automatic discovery uses the installed top-level package spec without importing
it. `isaacgymenvs_path` accepts the checkout root or its `isaacgymenvs` directory.
`config_path=/absolute/custom/cfg` optionally selects a configuration directory
containing `config.yaml`, `task/hexapod.yaml` and `train/hexapodPPO.yaml`. Discovery
does not search the working directory. Relative explicitly supplied paths, output
paths, and checkpoint paths are interpreted from the invoking working directory;
Hydra does not change that directory.

The implementation uses OmegaConf YAML and dot-list parsing. It composes the
installed root, task and feedforward training configuration explicitly and
registers historical `eq`, `contains`, `if` and `resolve_default` resolvers. Relative
interpolations retain their original nesting. Hydra's job/defaults/PBT groups are
not executed. This supports `key=value`, not the entire Hydra command language:

| Supported input | Meaning |
| --- | --- |
| `task=hexapod`, optional `train=hexapodPPO` | Only supported task/trainer choices |
| `num_envs=64`, `max_iterations=100` | Resolve the existing parent overrides |
| `headless=True`, `force_render=False` | Existing rendering controls |
| `seed=42`, `torch_deterministic=True` | Seed and torch determinism settings |
| `sim_device=cuda:0`, `rl_device=cuda:0`, `graphics_device_id=0`, `pipeline=gpu` | Separate simulator, learning and graphics settings |
| `experiment=my_run`, `output_dir=/path/run` | Run name / output location |
| `train.params.config.learning_rate=0.0001` | Override an existing supported training leaf |
| `train.params.network.mlp.units=[512,256,128]` | Network widths (must match when loading a checkpoint) |
| `checkpoint=/path/custom.pt`, `test=True`, `eval_steps=1000` | Resume or bounded deterministic evaluation |
| `tensorboard=True` | Additional custom trainer scalar logging |

Unknown keys, duplicate keys, `+key`, sweeps, alternate tasks, LSTM, multi-GPU,
central critics, framework randomization, video capture, W&B and sigma overrides
fail explicitly. Unsupported learning settings in the loaded YAML also fail.
`seq_len` is inert feedforward metadata; `env_name=rlgpu` is verified historical
metadata, not a registry lookup. The source-defined 104 observations and 18
actions are validated before/after construction; the policy is never reduced to
12 independent joint effects.

PPO/network values are read from YAML and mapped to the existing typed configs,
not replaced wholesale with defaults. Fields absent from YAML such as Adam
epsilon retain the previously verified Stage 2 defaults. Training controls read
`max_epochs=600`, `save_frequency=50`, `save_best_after=500`, `score_to_win=10000`,
and `print_stats=True` from the original file.

**Explicit convenience adjustment:** when `num_envs` is overridden downward and
the YAML minibatch would exceed the new rollout, the launcher selects one full
rollout minibatch and prints/records the change. Thus 64 environments with horizon
48 give minibatch 3072, not the original 12288. This is a new experiment, not an
equivalent baseline run. An explicit `train.params.config.minibatch_size` override
always wins and is validated rather than changed. Other nondivisible batches
fail. The unmodified baseline remains 256*48, five passes, original coefficients.

Devices must be `cpu` or `cuda:<index>`. GPU pipeline with CPU simulation, missing
CUDA devices, invalid graphics IDs, or contradictory task/root device settings
fail clearly. A deliberate CPU configuration requires all relevant overrides:

```bash
python train.py task=hexapod sim_device=cpu rl_device=cpu pipeline=cpu headless=True
```

That still requires Isaac Gym and is not a simulator-free mode. VecTask handles
its original result transfers to `rl_device`, and Anymal10 transfers prepared
actions to its simulation tensor device. The adapter does not silently move
incorrectly placed observations/rewards. Cross-device live operation is untested.

## Adapter and persistent collection

The adapter requires `[N,104]` float32 `observation_dict['obs']` on the RL device,
18-action continuous bounds [-1,1], float32 rewards `[N]`, integer/bool dones
`[N]`, and `extras['time_outs']` `[N]`. It does not recompute or re-clip observations;
VecTask's existing clipping precedes PPO's running normalization.

The collector creates all-one initial boundary flags, calls `reset()` exactly
once, and performs no eager physics/observation calculation. For every step it
copies observations, done flags, raw actions, log densities, values and Gaussian
parameters into the Stage 2 buffer **before** calling the adapter. The adapter
clamps raw actions into independent storage. For the verified [-1,1] action
space, historical rescaling is identity. In-place femur/tibia coupling cannot
change the stored policy samples or their densities.

Step outcomes are copied immediately, and the returned observations/done flags
persist into the next step and next rollout. Only buffer position is reset each
epoch. No reset_idx/reset_done or Gymnasium autoreset is introduced. Final value
prediction bootstraps the completed horizon. Stage 2 continues to apply the
pre-step-value timeout reward adjustment and next-done GAE masks unchanged.

Completed-episode meters accumulate **raw** reward and length until each returned
done. They then update a capped batch-weighted mean and clear only those running
counters. The default capacity is 100; this matches reference AverageMeter,
including its batch weighting rather than a literal last-100 deque. There is no
full observation/action/reward transfer to NumPy. Small logging scalars and
completed-index extraction can synchronize with a GPU.

## Training controls, logging, and cleanup

Each completed rollout/update increments epoch by one and frames by
`num_actors*horizon_length`. Frames count environment control transitions, not
physics substeps. `set_train_info(frames)` is called before collection. Console
metrics include epoch, frames, completed episode reward/length, actor/critic
loss, entropy, bounds/total loss, KL, starting LR and final adapted LR. Optional
TensorBoard receives the same scalars. Until an episode finishes, episode means
are `None`, not fabricated zero-reward episodes.

Historical saving/termination gates are preserved:

- Record starts at -100500. A strictly better completed-episode mean becomes
  `best_reward` only at `epoch >= save_best_after`, and writes `best.pt`.
- A positive `save_frequency` writes `epoch_XXXXXX.pt` at divisible epochs only
  when an episode mean exists and is <= the saved record. Thus the historical
  setting 50 does **not** imply unconditional saves before epoch 500.
- Winning requires a new eligible record strictly greater than `score_to_win`;
  it writes `winning.pt` and stops. A score threshold is not guaranteed reachable.
- `max_epochs` is the total completed epoch limit, including restored epochs.
  `-1` allows indefinite training. Reaching the limit writes `last.pt`, even when
  no episode has terminated. Winning also writes `last.pt` for convenience.

Run directories default to `runs/<name>_<timestamp>` and contain a resolved JSON
configuration, optional summaries and a `checkpoints/` directory. A nonempty
explicit output directory requires explicit resume, to avoid accidental reuse.
Checkpoint filenames differ from RL-Games; they do not imply format compatibility.

Training errors, Ctrl-C and viewer termination pass through cleanup. The pinned
VecTask lacks `close()`, so cleanup destroys viewer then simulator and closes the
task's global writer; injected environments may supply `close()`. No partial
rollout/update is checkpointed on interruption. Use a completed saved checkpoint;
early interruptions before any save may have none. A failure during the task's
own constructor may leave partially created native handles until process exit,
since the task did not return an object to clean up.

## Checkpoint, resume, and evaluation

Version 1 custom checkpoints contain network and both normalizer states, Adam
moments, AMP scaler, current LR, epoch/frames/best eligible reward, typed network/
PPO/training configs, resolved application config, Python/NumPy/torch CPU RNG and
available CUDA RNG states. Writes use a temporary file and atomic replacement.
Only load trusted custom files: torch's native pickle serialization can execute
code. No RL-Games checkpoint conversion is provided.

Resume validates the network architecture and every PPO setting before restoring
learning state. Application limits/logging/save controls can change. For example:

```bash
python train.py task=hexapod checkpoint=/path/run/checkpoints/last.pt max_iterations=700
```

Repeat any original PPO overrides, including `num_envs` and minibatch settings.
The saved adapted LR is restored from the optimizer rather than reset to the
initial YAML LR. Best eligible reward is also restored; unlike the historical
train() reset of that field, resume does not discard the saved record.

**Physical state is not restored.** The selected VecTask's get_env_state returns
None. The new task initializes fresh episodes, and a new collector begins with
ones/reset-current-observation semantics. Episode-meter history is fresh too.
RNG state is restored after task/model construction, but it cannot make those
new physical episodes continue the old trajectory. CUDA RNG cannot be restored
on a CPU-only process; differing available CUDA device counts are rejected when
restoring CUDA RNG. No uninterrupted-trajectory equivalence is claimed.

For deterministic evaluation:

```bash
python train.py task=hexapod test=True checkpoint=/path/run/checkpoints/best.pt headless=True eval_steps=5000
```

Evaluation restores network and normalizers, validates their compatibility, and
allows a different environment batch size. It applies frozen observation
normalization before deterministic mean actions and the same adapter clipping.
There are no PPO updates, optimizer loads or statistics updates. `eval_steps`
counts vector steps (default 1000), not completed episodes; the final report may
have no completed episodes if the budget is too short. It reports completed-episode
reward/length and total environment frames, then releases resources.

## Offline validation and CI

All existing 87 tests remain. Stage 3 adds 37: nine standard-library launcher/
installation/import checks, six OmegaConf composition tests against copies of the
actual repository YAMLs, and 22 torch integration tests. Tests use temporary fake
installation trees and a deterministic mock with reused tensors, in-place action
mutation, multiple episode lengths, timeouts and deferred resets. They exercise
the production modules, including two training epochs, save/load, continuation
and evaluation. A subprocess explicitly blocks Isaac Gym, IsaacGymEnvs and
RL-Games imports during mock training. These are interface tests, not physics.

Local results on Python 3.9.10 and 3.12.14: **124 discovered, 27 passed, 97 skipped,
zero failures/errors**. Of the skips, 91 need torch and six need OmegaConf. Both
dependencies are absent locally; no local installation was attempted. The nine
separate CI failure-gate tests pass. Launcher `--help` works without torch or
simulator imports. Protected source/assets and mathematical modules have no diff.
Syntax/import checks and whitespace checks pass. The strict CI entrypoint fails
on the absent torch import, and its suite-result gate rejects all skipped tests.

The workflow retains Ubuntu 22.04, Python 3.10, torch **1.13.1+cpu**, NumPy 1.24.4,
official actions and read-only token permissions. It additionally installs
**OmegaConf 2.3.0** for real YAML composition tests and raises the minimum count to
**124**. Any skip/failure/error/expected failure/incomplete execution still fails.
No simulator packages are installed in CI. The optional integration dependency
does not change the base project's torch requirement. The strict runner retains
its historical filename:

```bash
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s ci -p 'test_*.py' -v
python -B ci/run_stage1_tests.py
```

The first command permits local dependency skips; only the last enforces full
validation. After review/push, inspect **Stage 1 to 3 PyTorch validation** in
[GitHub Actions](https://github.com/Leo813/hexapod_proj/actions) or PR Checks. No
successful Stage 3 run has been observed here. All 124 current tests must pass
without skips before claiming CPU numerical/integration validation. A local YAML
schema validator/actionlint is unavailable; the edited workflow was reviewed and
its embedded Python preflight syntax checked, not executed as a GitHub job.

## Troubleshooting and targeted simulator smoke test

| Failure | Action |
| --- | --- |
| IsaacGymEnvs not found | Select the installed checkout with isaacgymenvs_path |
| Required task/YAML missing | Perform the documented manual file placement; config_path must contain all three YAMLs |
| Asset directory missing | Copy the unchanged hexapod2 tree to assets/urdf/hexapod2 |
| Isaac Gym must initialize before torch | Start a fresh process with this repository's train.py |
| Namespace already imported | Do not mix the narrow loader with make()/task-registry imports |
| OmegaConf/torch/matplotlib/TensorBoard import failure | Use the prepared training environment and install required dependencies manually |
| Invalid minibatch | Choose a divisor of num_envs*horizon; explicit overrides are never silently changed |
| Observation/dtype/device mismatch | Inspect the actual returned VecTask tensors; do not bypass validation with a silent CPU transfer |
| Checkpoint mismatch | Use a custom checkpoint and reproduce its network/PPO settings; evaluation only relaxes batch/optimizer settings |

On a simulator-equipped Linux machine, first run all offline tests, then:

```bash
python train.py task=hexapod isaacgymenvs_path=/opt/IsaacGymEnvs headless=True num_envs=8 max_iterations=2 output_dir=runs/stage3-smoke
```

Confirm the printed minibatch adjustment to 384, two completed epochs and 768
environment frames, finite losses/parameters, and `checkpoints/last.pt`. Confirm
one initial reset call and no eager extra physics step when instrumenting the
adapter (without changing task behavior). Inspect actual `[8,104]` observations,
18-dimensional policy samples, timeout flags and deferred reset timing. Then
resume with the same `num_envs=8` and `max_iterations=3` into a new output directory
and verify the final frame count 1152. Evaluate the saved policy with the same
network settings and confirm frozen normalizers and no optimizer steps.

Separately verify native mesh resolution, imported DOF/body ordering, sensor
index 5, contact indices, viewer lifecycle, GPU tensor/PhysX synchronization,
cross-device transfers and CUDA AMP. Check physics/actions/rewards against the
original task before attempting long training. Package/JIT import compatibility
and native simulator cleanup remain untested. No locomotion, object-balancing,
learning-curve equivalence or complete GPU compatibility claim is made.

## Later work (Stage 4, not implemented)

After live Stage 3 validation, consider a separate asset-placement/path-resolution
tool, clearer install/version diagnostics, a minimal distributable task package,
checkpoint format migrations and a supported GPU environment matrix. Preserve
asset contents and environment semantics when designing those improvements.
LSTM and distributed training remain separate work. Stage 4 requires approval.
