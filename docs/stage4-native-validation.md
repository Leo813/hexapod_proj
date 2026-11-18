# Stage 4B.1: native compatibility investigation and validation preparation

For current installation instructions and the selected provisional runtime,
use [Getting started](getting-started.md). This record preserves Stage 4B.1's
historical candidates and evidence; it does not report native validation.

Date: 2026-11-18. Branch: `feat/stage4-native-readiness`. Start:
`bc17da42df3e9f52be413d76b98b047b53a93f0d`, the Stage 4A merge. The working tree
was clean. Local tracking refs were stale; an authorized fetch obtained the
expected merge and the feature branch was created from `origin/main`. Local
`main` was not edited. No external IsaacGymEnvs checkout was updated.

**Preparation is complete; native execution is pending.** No real Isaac Gym
imports, CUDA operations or simulation/training were executed during this
stage. No package requirement, original task/YAML/asset, PPO equation or
network architecture was changed. The tools below never install dependencies,
repair a driver, download a simulator or modify an external checkout.

## Evidence and uncertainty

| Category | Established evidence |
| --- | --- |
| Verified upstream requirements | NVIDIA's [Preview 4 archive](https://developer.nvidia.com/isaac-gym/download) lists Ubuntu 18.04/20.04, Python 3.6/3.7/3.8, NVIDIA driver >=470 (470.74 recommended), and Pascal-or-newer hardware with >=8 GB VRAM. These prerequisites do not establish a torch/CUDA binary pairing. |
| Verified repository requirements | `pyproject.toml` declares Python >=3.8, torch >=1.13, and optional OmegaConf >=2.2,<2.4. Original YAMLs retain GPU PhysX, GPU pipeline and 256 environments through historical root resolvers. |
| Selected upstream reference | IsaacGymEnvs `aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3`: [README](https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/README.md) requests Preview 4; [setup metadata](https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/setup.py) declares Python >=3.6 and unversioned torch, plus other dependencies. This is not a fully pinned environment or the proven original checkout. |
| Provisional project assumptions | Ubuntu 20.04, Preview 4, Python 3.8, torch 1.10.x, compatible CUDA 11.x, GPU simulation/learning and 256 environments. Exact original Python/torch/CUDA versions remain unknown. |
| Unverified hypotheses | SDK binary compatibility with a specific torch build, native task import, GPU tensor behavior, contact/body ordering, sensor semantics and scientific reproduction. No offline test establishes these. |

The original SDK's `python/rlgpu_conda_env.yml`, `python/setup.py` and bundled
documentation were **not available** here. The NVIDIA archive points to
`docs/index.html` inside the package but does not list a numeric PyTorch pin.
No SDK contents or dependency pins have been inferred from third-party copies.
`--isaacgym-sdk` below can collect hashes and text from the actual future SDK
without executing setup.py or accepting/downloading a binary distribution.

## Runtime candidates and recommendation

| Combination or scope | Status | Meaning |
| --- | --- | --- |
| Project Python >=3.8 / torch >=1.13 declarations | **Verified** | Read from repository metadata, not a native execution result. |
| Ubuntu 22.04 / Python 3.10 / torch 1.13.1+cpu / NumPy 1.24.4 / OmegaConf 2.3.0 | **Unverified in full locally**; existing offline CI configuration | Preserved 178-test baseline; no GPU/simulator claim. The new 203-test CI run remains pending; historical reports retain their original scope. |
| Core custom tensor/NN/PPO API calls on torch 1.10.2 | **Source-supported** | Inspected APIs exist with the used signatures; numerical and CUDA behavior not executed. |
| Ubuntu 20.04 / Python 3.8 / torch 1.10.x / CUDA 11.x / Preview 4 | **Provisional; unsupported by current package requirement** | torch <1.13 fails metadata policy. Restricted checkpoint loading is unavailable. No automatic downgrade or requirement change. |
| Ubuntu 20.04 / Python 3.8 / torch 1.13.1 CUDA build / Preview 4 | **Source-supported API candidate; native unverified** | Matches package floor and Python range intersection. SDK/driver/extension compatibility must still be established. |
| torch 1.12.1 | **Unsupported by current package requirement; native unverified** | Historically available CUDA builds, but still below the floor and without weights_only. Not a supported workaround. |
| Arbitrary newer torch releases with Preview 4 | **Unverified** | A permissive package upper range does not certify old native binaries. |

Preserve the existing working GPU installation and inventory it first. If
constructing a separately approved validation matrix later, Python **3.8** is
the overlap between the project and NVIDIA's documented ranges. Torch 1.13.1
is the first practical candidate already represented in the CPU baseline;
its CUDA build remains a **candidate**, not a recommendation to upgrade the
existing machine. The [official historical PyTorch matrix](https://pytorch.org/get-started/previous-versions/)
lists 1.13.1 CUDA 11.6/11.7 builds and historical 1.10/1.12 options. A torch
wheel's bundled CUDA runtime, system toolkit used to compile gymtorch, NVIDIA
driver and SDK binaries are distinct compatibility inputs. A generic “11.x”
label is insufficient to select them. No installation commands are run here.

If the actual machine is torch 1.10.x, retain it and stop before custom PPO
training. Review lowering the requirement only after native/offline testing
of that exact stack and a checkpoint security decision. The diagnostic permits
an explicitly requested import probe of such a stack, but does not let a
successful import override the training prerequisite failure.

## Full custom-package PyTorch API assessment

The review covered networks, distributions, losses/optimizer, normalization,
advantages, rollout buffers/collector, schedules, checkpoint format, evaluation,
trainer, launcher and integration. No additional post-1.10 core mathematical
API requirement was identified in these calls. This is a source inspection,
not proof of exhaustive runtime compatibility.

| Components / used calls | Historical evidence and limits |
| --- | --- |
| Actor/critic: Linear, Sequential, ELU and other supported activations, Parameter, init.zeros_, Module.to/state_dict/load_state_dict | [1.10.2 Linear](https://github.com/pytorch/pytorch/blob/v1.10.2/torch/nn/modules/linear.py) and [Module](https://github.com/pytorch/pytorch/blob/v1.10.2/torch/nn/modules/module.py) support the constructor/device/state APIs. Separate network architecture remains unchanged. |
| Gaussian sampling, log densities, entropy | [Normal](https://github.com/pytorch/pytorch/blob/v1.10.2/torch/distributions/normal.py) and [Independent](https://github.com/pytorch/pytorch/blob/v1.10.2/torch/distributions/independent.py) provide sample/log_prob/entropy and reinterpreted batch dimensions. [broadcast_to, maximum, square](https://github.com/pytorch/pytorch/blob/v1.10.2/torch/_torch_docs.py) are present. |
| PPO optimization and clipping | [1.10.2 optimizer](https://github.com/pytorch/pytorch/blob/v1.10.2/torch/optim/optimizer.py) supports zero_grad(set_to_none=True); [clip_grad_norm_](https://github.com/pytorch/pytorch/blob/v1.10.2/torch/nn/utils/clip_grad.py) accepts the used positional arguments. No foreach/fused/capturable option is supplied by this project. |
| Normalization and GAE | Float64 registered buffers, tensor moments, reductions, masks and in-place copies use existing APIs. [1.10.2 tensor docs](https://github.com/pytorch/pytorch/blob/v1.10.2/torch/_tensor_docs.py) support var/std with unbiased=True; the code does not use newer correction= syntax. Sample variance and update timing remain unchanged. |
| Rollouts, collection, schedules and evaluation | zeros/empty, transpose/reshape, detach/clone/copy_, nonzero, indexing, no_grad, eval/train and deterministic mean inference are source-supported. Native device/shape ownership is still validated by the adapter, not inferred from CPU mocks. |
| CUDA setup and RNG | [1.10.2 CUDA APIs](https://github.com/pytorch/pytorch/blob/v1.10.2/torch/cuda/__init__.py), [CUDA RNG](https://github.com/pytorch/pytorch/blob/v1.10.2/torch/cuda/random.py) and [use_deterministic_algorithms](https://github.com/pytorch/pytorch/blob/v1.10.2/torch/__init__.py) provide the used signatures. Kernel determinism, native extensions and GPU availability remain unverified. |
| AMP | [1.10.2 autocast](https://github.com/pytorch/pytorch/blob/v1.10.2/torch/cuda/amp/autocast_mode.py) accepts enabled=; [GradScaler](https://github.com/pytorch/pytorch/blob/v1.10.2/torch/cuda/amp/grad_scaler.py) supports scale/unscale_/step/update and state restoration. The code uses torch.cuda.amp, not the newer torch.amp API. Actual CUDA overflow/scaling behavior requires execution. |
| Trainer/launcher | Existing control flow adds no newer tensor API requirement. Original task-side matplotlib/TensorBoard imports, gym/VecTask dependencies and native JIT bindings remain separate prerequisites. |

### Checkpoint and optimizer limitations

Torch [1.10.2](https://github.com/pytorch/pytorch/blob/v1.10.2/torch/serialization.py)
and [1.12.1](https://github.com/pytorch/pytorch/blob/v1.12.1/torch/serialization.py)
have no weights_only parameter. The current loader rejects default loading
before deserialization on such runtimes. Only an explicit trust decision
permits the old unrestricted API; it never retries unrestricted automatically.
The smoke test uses default restricted loading and offers no trust bypass.

Version 3 metadata is tagged JSON plus a native tensor table, serialized with
protocol 2. This avoids the unsupported float opcode in 1.13.1's early
restricted loader. The [1.13.1 loader source](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/_weights_only_unpickler.py)
is the relevant security/API baseline; restricted loading is still not a
sandbox for arbitrary hostile files. The file-placement utility does not load
checkpoints at all.

[Adam 1.10.2](https://github.com/pytorch/pytorch/blob/v1.10.2/torch/optim/adam.py)
uses integer step counters; [Adam 1.13.1](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/optim/adam.py)
uses tensor counters and has additional state/options. The optimizer loaders
cast state to parameter devices with version-specific handling. Therefore,
successful tensor deserialization does not establish cross-version Adam
resume equivalence. Test same-version/same-device restoration first. No
cross-version conversion is introduced here.

Same-CUDA nonempty AMP state restores normally. Disabled CPU scaler -> enabled
CUDA scaler explicitly creates a fresh scaler and warns; no missing growth
history or CUDA RNG state is invented. This transition policy is already in
Stage 3.5B and is unchanged. Python 3.8 syntax is checked statically; neither
that nor torch's [1.13.1 Python metadata](https://github.com/pytorch/pytorch/blob/v1.13.1/setup.py)
establishes Preview 4 ABI compatibility.

**Proposed changes for later review:** if native 1.10 proves necessary, run a
separate exact-version matrix, adapt validation fixtures that unconditionally
use weights_only, decide whether trusted-only checkpoints are acceptable or a
separate restricted/non-pickle format is needed, and assess Adam migration
before lowering the package floor. No such policy/package change is made now.

## Read-only readiness

Use the Python interpreter from the environment being investigated:

```bash
python -B scripts/check_native_readiness.py
python -B scripts/check_native_readiness.py --isaacgymenvs /opt/IsaacGymEnvs --isaacgym-sdk /opt/isaacgym
```

The tool reuses Stage 4A metadata/path/hash/revision inspection. It reports
OS/Python, distribution versions, package paths, a statically readable torch
CUDA build, installed-file hashes and missing prerequisites. No tensor/native
package import occurs. No output files or directories are created. SDK setup
and conda YAML files, when supplied, are read as text (up to 64 KiB each), not
executed; presence/hashes do not certify that the package is genuinely Preview 4.
The historical root config remains untouched. The report has explicit blockers
and notices and never declares “native ready” from package names alone.

Add `--driver-info` only to request a read-only nvidia-smi inventory. Missing
tools remain an informative result, not a repair action. The default does not
query a GPU. There are no pip/Conda, driver or external Git mutation commands.
Only this project's own main ref was fetched while preparing the feature branch.

Exit 0 means no metadata blockers (and any requested active check succeeded),
not proven simulator compatibility. Exit 1 reports blockers/check failure;
invalid arguments exit 2. An import probe can succeed while the command exits
1 because torch 1.10 still violates package policy, or no task checkout was
selected. Version strings not understood by the deliberately conservative
final-release parser are reported unknown rather than guessed compatible.

## Explicit native import and tiny CUDA check

Run only when the GPU machine owner is ready to authorize native execution:

```bash
python -B scripts/check_native_readiness.py --isaacgymenvs /opt/IsaacGymEnvs --check-imports --device cuda:0 --timeout 60
```

A fresh Python subprocess imports **isaacgym first, then torch**; importing
torch before the worker starts is rejected. It checks CUDA availability and
the selected visible device, allocates eight float32 values, squares/sums them
and synchronizes. The expected result is 140. Device indices are relative to
CUDA_VISIBLE_DEVICES. Errors are labeled import-order, isaacgym-import,
torch-import, cuda-availability, cuda-device or cuda-operation. Native crashes,
missing result markers and wall-time expiration have separate process errors
with stdout/stderr tails. No simulation is constructed in this mode.

This explicitly requested mode may cause SDK imports/native libraries to
initialize and use their normal extension caches. It does not install or
upgrade anything. Passing it establishes only imports and that tiny operation,
not PhysX/URDF/sensor/training compatibility. Active checks were not executed
against real packages here.

## Explicit bounded smoke test

After resolving the read-only report's blockers and reviewing import results:

```bash
# /data/hexapod-validation must already exist; smoke-001 must NOT exist.
python -B scripts/check_native_readiness.py \
  --isaacgymenvs /opt/IsaacGymEnvs \
  --run-smoke --device cuda:0 \
  --output /data/hexapod-validation/smoke-001 --timeout 180
```

This flag explicitly authorizes native processes and output creation. The output
must be a new absolute directory outside the source and external checkout.
The child runs there so the original task's global SummaryWriter writes its
default runs directory there, not into the source/installed package. Native
extension caches may still use their SDK defaults. There is no automatic
invocation of active flags from CI or default inspection.

The existing configuration loader, task factory, adapter, ActorCritic, PPO,
RolloutCollector and Trainer are reused. Diagnostic overrides are four
environments, one epoch, GPU PhysX/pipeline on the selected device, headless
mode and no viewer graphics device. The original horizon remains 48 and five
mini-epochs remain. The existing documented minibatch adjustment yields 192
(4*48), giving five minibatch updates. Runtime assertions reject a different
budget; no CLI option here enables more environments/epochs. Timeout is bounded
to 1..600 seconds. The unchanged terrain/contact allocation can still be costly
on a small GPU; no automatic physics-memory tuning is performed.

This is **not** the original 256-environment experiment. No episode duration,
reward, network size, PPO coefficient or original YAML is rewritten. The
script is a diagnostic application around Trainer, not a second simulation
loop. A wall-time kill can prevent graceful native cleanup; the parent records
the timeout, and process exit releases process-owned native resources. Do not
reuse a failed output directory or silently extend a failing run.

Expected artifacts (some may be absent if failure occurs early):

- `readiness.json`: preflight and worker result.
- `native-stdout.log`, `native-stderr.log`: completed worker logs; timeout tails
  remain in readiness.json instead.
- `resolved-config.json`: actual diagnostic configuration and device settings.
- `smoke-evidence.json`: stage, finite-update checks, native names/indices,
  sensor shapes/devices, limits, snapshots, counters and checkpoint result.
- `checkpoints/last.pt`: the existing Trainer's version 3 learning checkpoint.
- `runs/...`: the unchanged task's own TensorBoard output.

The worker compares imported DOF names with the original 18 revolute joints,
records imported rigid bodies and checks moving links/hard-coded indices exist.
Construction errors retain their native traceback; missing URDF/meshes are
also caught by the preceding hash check. The adapter enforces [4,104]
observations, [4,18] actions and CUDA device/dtype contracts. GPU sim parameter
flags, sensor tensor shapes/devices, finite step observations/rewards/sensors,
all returned PPO metrics and final parameters are checked. Adam state must be
populated and at least one model parameter must change.

After the existing Trainer saves and closes the environment, a fresh CUDA
actor/critic/PPO instance restores last.pt with restricted loading. Model,
normalizers, optimizer tensors and devices, scaler state, counters, learning
rate and torch CPU/CUDA RNG are compared. The worker does not start another
environment, epoch or policy evaluation loop after restoration. Exact next-
update equivalence and CPU-to-CUDA transitions remain separate validation.

## Interpreting inherited simulation uncertainties

| Evidence collected | What still needs independent review |
| --- | --- |
| Imported DOF and rigid-body names/order, original URDF joint list | Whether native order matches every inherited positional assumption; fixed-joint collapse changes counts. |
| Native body name at index 5, sensor force/torque tensor shape/device/finite values | Whether index 5 is the scientifically intended tray; successful creation alone does not establish correct placement or force meaning. |
| Body names at contact indices 0,3,6,9,12,15,18 and named feet/knee/thigh indices | Whether observation and termination indices refer to the intended links. No remapping occurs. |
| Joint limits, positions, unscaled targets and source-derived scaled targets | Whether randomized resets/coupling/scaling place targets outside intended limits. Derived scaled targets are not a direct Gym setter interception. |
| Before/after reset_idx, compute_observations and pre_physics_step snapshots | GPU setter/refresh visibility and stale state. These are method boundaries, not traces inside C++/PhysX. |

The observer calls each original method exactly once, inserts no resets,
setters or refreshes, and restores method bindings during cleanup. Copying
diagnostic snapshots to CPU synchronizes CUDA and can change timing, so the
trace is evidence for review, not proof of asynchronous correctness. Constructor
resets happen before instrumentation; the initial deferred reset and any
natural terminations during the bounded rollout are observed afterward.
A run without natural termination explicitly warns that full episode/reset
coverage was not established; the test does not force a reset to manufacture it.

Potential mismatches must be reported using the stage, native traceback,
version inventory, original-file hashes and exact values/names from the trace.
Do not change the source simulation, physics or reward to make the smoke pass.

## Further validation after the bounded test

1. Repeat a same-version/same-device CUDA learning-state round trip with an
   identical fixed next rollout/update, controlling CUDA_VISIBLE_DEVICES order
   and RNG. Compare optimizer moments/step counters, metrics and resulting
   parameters with justified tolerances. Do not assume simulator state resumed.
2. Separately test CPU disabled-scaler -> CUDA enabled-scaler restoration with
   the same PPO settings and an explicit expected fresh-scaler warning. Compare
   compatible learning state and complete a finite update; missing CPU-run CUDA
   RNG and AMP history cannot be restored. Existing CPU scaler doubles do not
   establish this native behavior.
3. Test frozen-normalizer evaluation and additional natural termination/reset
   cases through the existing launcher under separately bounded controls.
4. Review diagnostic timing effects, body/sensor/contact ordering and joint
   targets before authorizing a separate 256-environment baseline test. Full
   training, learning curves and research reproduction are not Stage 4B.1.

The original SDK inventory, exact driver/toolkit/runtime pairing, NumPy/native
extension interoperability and full dependency closure remain native blockers.
An installed package's metadata or a successful reduced run does not resolve
all of them. No speculative compatibility patch or dependency-floor change is
included in this preparation.

## Offline tests and current execution status

Twenty-five new tests cover version parsing/policy, missing dependencies and
SDK files, missing installed task files, explicit authorization, import ordering,
CUDA absence/device errors, controlled fresh-process fixtures, timeout/crash
diagnostics, exclusive outputs, bounded configuration, finite metrics and
observer call preservation. One composition test uses the existing OmegaConf
loader and unchanged YAML fixtures; all native modules in subprocess tests are
inert doubles, not actual Isaac Gym or CUDA.

The strict CPU CI minimum is **203** (178 preserved +25 new), with no skips,
failures/errors or incomplete execution accepted. Existing CI dependency pins
remain unchanged; native GPU modes are never part of required CI. Local
validation limitations and results are recorded in the completion report.
The read-only diagnostic here reports Windows 11 / Python 3.12.14, missing
torch/OmegaConf/Isaac Gym and no selected native checkout or SDK. It exits 1
with those blockers as intended. No real active/native check was requested or
run during this task. All native results described above remain pending.

Local results: 181 tests ran inside the sandbox, with 60 passes and 121 skips
(108 torch-dependent, 13 OmegaConf-dependent), no failures/errors. The 22
unchanged Stage 4A installation tests passed separately outside the sandbox
under the existing approval because atomic hard-link tests need that access.
Together: **203 discovered, 82 passed, 121 skipped**. Of the 25 new tests, 24
passed and the OmegaConf composition test skipped. All nine CI-gate tests
passed; the strict CI entrypoint correctly rejects missing torch. All 42
package/test/CI/script Python files parse with Python 3.8 grammar. Existing
test methods are retained, whitespace checks pass, and protected-source diffs
show no simulation, YAML, asset, learning-code or dependency changes. These
local skips are not accepted as full CI validation.
