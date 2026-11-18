# Stage 3.5B: targeted compatibility corrections

Based on merged Stage 3 commit `11287ea55471e9d31fbeed129fa7e63c968085ef`.
Work branch: `fix/stage3-audit-corrections`. These are software compatibility
changes, not changes to the scientific experiment or Stage 4 installation work.

## Changes and policies

### CK-01: AMP checkpoints across devices

When the destination scaler is enabled but the saved scaler state is empty,
resume creates a fresh enabled `torch.cuda.amp.GradScaler` and emits a warning.
Compatible model parameters, normalizers, Adam state, learning rate, counters
and available RNG state are restored normally. A previously used destination
scaler is replaced, so its unrelated growth history cannot leak into resume.
There is no old AMP growth history to restore from a disabled CPU scaler.

Nonempty enabled scaler state is restored normally. A disabled destination
scaler retains PyTorch's no-op loading behavior. Evaluation does not load or
create a scaler. CPU-to-CUDA transitions cannot restore CUDA RNG state that
the CPU run did not save; they do not reproduce uninterrupted training.

The transition regression uses scaler doubles on CPU. It does **not** validate
CUDA, native enabled GradScaler execution, or mixed-precision numerics.
Reference: [PyTorch 1.13.1 GradScaler.state_dict/load_state_dict](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/cuda/amp/grad_scaler.py).

### CK-02: checkpoint format and explicit deserialization

New saves use **format version 3**. The original version 2 correction was
insufficient: [published run 44045723462](https://github.com/Leo813/hexapod_proj/actions/runs/44045723462)
executed 144 tests with two failures, nine errors and no skips. All eleven
tracebacks originate from restricted loading, before restoration or the intended
compatibility checks. No failure in that run demonstrated a PPO equation bug.

**Root cause:** opcode 71 is `BINFLOAT`, used for ordinary Python floats by
pickle protocol 2. The [PyTorch 1.13.1 restricted unpickler](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/_weights_only_unpickler.py)
has no handler for it. Removing NumPy arrays was not enough: learning rates,
optimizer betas/epsilon, configuration values, reward counters, AMP scale/growth
values and cached Python/NumPy Gaussian values still contain floats. Higher
pickle protocols introduce additional unsupported instructions such as FRAME
and MEMOIZE; arbitrary Python objects and NumPy reconstruction globals are
also outside the restricted loader's allowlist. The fix therefore addresses
the entire metadata tree, not one field or one opcode.

**Wire representation:** the torch archive contains only `format_version=3`,
one JSON `metadata` string, and a `tensors` list. All nested metadata uses
explicit tags for literals, hexadecimal float strings, lists, tuples,
dictionaries, ordered dictionaries and tensor indices. This preserves integer
optimizer keys, Python RNG tuples, exact finite Python float values (including
negative zero), infinities/NaN, and state-dict `_metadata`. Model/Adam/normalizer
and torch/CUDA RNG tensors retain native torch storage serialization.
NumPy MT19937 arrays remain integer lists and are reconstructed only after
loading; cached Gaussian values and generator positions are retained.

Saves explicitly use `pickle_protocol=2`. Thus float metadata, custom classes,
sets, bytes and newer-protocol opcodes cannot leak into the pickle stream
through resolved configuration. Unsupported metadata types are rejected before
replacing a checkpoint; malformed envelopes/tags/references are rejected before
model restoration. JSON decoding never imports a named class or executes a
file-provided callable. Container aliases/cycles and arbitrary Python object
graphs are not supported checkpoint metadata. Tensor storage remains managed
by torch; this does not turn the format into a non-pickle archive.

Normal loading explicitly uses `torch.load(..., map_location="cpu",
weights_only=True)`. This argument exists in PyTorch 1.13.1 and avoids relying
on the changed default in PyTorch 2.6. Failure never triggers an automatic
unrestricted retry. Format version, architecture, PPO and normalization checks
remain in place; optimizer LR and CUDA RNG device-count checks occur before
model restoration.

**Use checkpoint files only from known sources.** Restricted loading reduces
pickle reconstruction capabilities; it is not a sandbox or a guarantee against
malicious files, particularly with a historical dependency release.

Existing **version 1** checkpoints require explicit trust because their NumPy
RNG arrays use pickle reconstruction. **Version 2** files containing floats
also need explicit trust on 1.13.1; on newer runtimes a successful restricted
load of version 2 is accepted. No error causes an unrestricted retry. Only for
a file whose origin and contents you trust:

```bash
python train_agent.py checkpoint=/trusted/run/checkpoints/last.pt trust_legacy_checkpoint=True
python train_agent.py test=True checkpoint=/trusted/run/checkpoints/best.pt trust_legacy_checkpoint=True
```

The equivalent API is `load_checkpoint(path, algorithm, trust_legacy=True)` or
`trainer.resume(path, trust_legacy=True)`. This flag selects
`weights_only=False` where available and emits a warning: **arbitrary pickle
code may execute**. On an older runtime without that parameter, it calls the
legacy API without the unsupported keyword, but only after this explicit trust
decision. Without the flag those runtimes reject loading before calling
`torch.load`, for resume and evaluation alike. Capability detection inspects
the API signature; it does not optimistically try loading and catch TypeError.
Trust is never inferred from a filename, error, or checkpoint metadata. Leave
it false for version 3 on runtimes with restricted loading. A subsequent save
writes version 3. RL-Games files remain unsupported; old project code cannot
read the new envelope.

The offline CPU CI target remains Python 3.10 / PyTorch 1.13.1+cpu; it does
not identify the original simulator's Python, PyTorch or CUDA installation.
The new format is designed for restricted loading on 1.13.1 and 2.6.0, but the
broad `torch>=1.13` package declaration is not a claim that every intervening
or future release has been tested. No dependency versions are changed here.
See the exact additional verification procedure below.

References: [1.13.1 serialization API](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/serialization.py),
[2.6 serialization semantics](https://docs.pytorch.org/docs/2.6/notes/serialization.html).

### Original GPU runtime compatibility review (2026-11-17)

The primary runtime remains **Ubuntu 20.04, Isaac Gym Preview 4, GPU PhysX,
GPU PyTorch training and 256 parallel environments**. The selected
[IsaacGymEnvs root configuration](https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/isaacgymenvs/cfg/config.yaml)
defaults to `pipeline=gpu`, `sim_device=cuda:0` and `rl_device=cuda:0`.
Those defaults and the original task's 256 environments are unchanged.
Ubuntu 22.04 / Python 3.10 / torch 1.13.1+cpu is solely the offline CI
environment. Passing it does not qualify that stack for native simulation.

**Documented requirements and remaining evidence gap.** NVIDIA's
[Preview 4 download archive](https://developer.nvidia.com/isaac-gym/download)
lists Ubuntu 18.04/20.04, Python 3.6/3.7/3.8, driver 470 or newer
(470.74 recommended), and Pascal-or-newer hardware with at least 8 GB VRAM.
It does not specify a numeric PyTorch requirement; it directs readers to the
downloaded SDK documentation. The exact SDK dependency files and the user's
original CUDA environment were not available for this review. Consequently,
no exact Preview 4 PyTorch pin or original CUDA build is established here.
Inspect the actual SDK and installed environment before making that claim.

The selected IsaacGymEnvs revision
`aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3` explicitly requests Preview 4 in
its [README](https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/README.md).
Its [package metadata](https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/setup.py)
declares Python `>=3.6` and unversioned `torch`, alongside dependencies such
as `hydra-core>=1.2` and `rl-games>=1.6.0`. This establishes intended Preview 4
support, not a fully pinned, tested dependency combination. It neither proves
that every declared Python version can resolve all dependencies nor certifies
the custom checkpoint implementation. The reference revision is a selected
comparison baseline, not a recovered lockfile from the original experiment.

**Assessment of `torch>=1.13`.** This is the custom package's current software
requirement, not a documented NVIDIA requirement. It is consistent with the
restricted loading API used by CK-02. It would exclude an original installation
using an older torch, if that is what inventory reveals. Keep the requirement
unchanged until that inventory and whole-package compatibility checks exist;
do not upgrade the working CUDA installation to satisfy it automatically.
The package's Python `>=3.8` requirement intersects NVIDIA's documented Python
range at **3.8**. PyTorch 1.13.1's
[metadata](https://github.com/pytorch/pytorch/blob/v1.13.1/setup.py) permits
Python 3.8, but this overlap alone does not establish binary, CUDA or native
Isaac Gym compatibility. The unbounded torch upper range likewise is not a
claim of validation on every modern release.

| Inspected PyTorch source | Checkpoint loading implication |
| --- | --- |
| [1.8.1](https://github.com/pytorch/pytorch/blob/v1.8.1/torch/serialization.py), [1.10.0](https://github.com/pytorch/pytorch/blob/v1.10.0/torch/serialization.py), [1.10.2](https://github.com/pytorch/pytorch/blob/v1.10.2/torch/serialization.py), [1.12.1](https://github.com/pytorch/pytorch/blob/v1.12.1/torch/serialization.py) | No `weights_only` parameter. Default loading is rejected before deserialization; explicit trust selects the old API without this keyword. Native execution remains unverified. |
| [1.13.1](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/serialization.py) | Explicit `weights_only=True/False` is supported, but the restricted opcode subset cannot load v2 floats. Version 3 targets that subset. Actual corrected CI execution remains pending. |
| [2.6](https://docs.pytorch.org/docs/2.6/notes/serialization.html) | Explicit policy avoids dependence on the changed default. Numerical compatibility still requires the separate validation below. |

These older versions were inspected to establish API differences, not to
identify which one was installed originally. The user's provisional target is
Ubuntu 20.04 / Preview 4 / Python 3.8 / torch 1.10.x / compatible CUDA 11.x;
it is not a verified inventory. The 1.10.2 source supports the protocol-2 zip
save format and `map_location`, and its
[optimizer loader](https://github.com/pytorch/pytorch/blob/v1.10.2/torch/optim/optimizer.py)
maps state tensors to parameter devices. Its
[GradScaler](https://github.com/pytorch/pytorch/blob/v1.10.2/torch/cuda/amp/grad_scaler.py)
supports the same state fields and rejects an empty enabled-scaler state.

The compatibility correction now includes an explicitly trusted old-API path,
covered by a strict signature-double regression awaiting CI, not native 1.10 execution. There is no
restricted-load path on 1.10. Supporting untrusted input there would need a
separately reviewed non-pickle tensor format or restricted loader. Further
blockers to declaring full 1.10 support are whole-package/API validation,
native 1.10 CPU/CUDA checkpoint round trips, optimizer state differences across
torch versions, the actual Python/CUDA/driver combination and Preview 4
execution. The package still declares `torch>=1.13`; the narrow loader path
does not lower that requirement or certify cross-version optimizer resume.

**GPU checkpoint behavior from source inspection.** `map_location="cpu"`
controls initial deserialization, not the training device. The launcher moves
the model to the requested RL device and PPO places normalizers there before restoration;
`load_state_dict` copies their tensors into those destination modules. The
[1.13.1 optimizer loader](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/optim/optimizer.py)
casts state tensors to the associated parameter device, with special handling
for the step counter. Thus Adam moments can restore to CUDA without changing
the CPU mapping used for RNG bytes. A native check must verify the installed
version's actual state placement; do not force every optimizer tensor to CUDA.

For same-device CUDA resume, nonempty scaler state is loaded normally; model,
optimizer, normalizers, learning rate, counters and available RNG state are
restored. For disabled-to-enabled AMP, CK-01 explicitly creates a fresh scaler
and warns that growth history is unavailable. This matches the empty-state
rejection in the inspected
[1.8.1](https://github.com/pytorch/pytorch/blob/v1.8.1/torch/cuda/amp/grad_scaler.py)
and [1.13.1](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/cuda/amp/grad_scaler.py)
GradScaler implementations; it does not establish that the complete package
supports 1.8.1. CUDA RNG restoration requires the same visible device count;
device identity/order must also be controlled in native validation.
Simulator state is not saved, so even successful same-device restoration does
not establish an uninterrupted simulation trajectory.

These are **source-level GPU compatibility findings**, not executed CUDA tests.
No CUDA-enabled PyTorch installation was installed, upgraded or replaced.

### CF-02: configuration precedence

Application settings use **CLI > root YAML > absent-key defaults**. Defaults
are `output_dir=""`, `eval_steps=1000`, `tensorboard=False`, and
`trust_legacy_checkpoint=False`. Explicit YAML values survive composition.
The trust flag must be boolean.

Historical `train.params.config.reward_shaper.scale_value` remains the
canonical reward scale, with the unchanged default 1.0. If a custom YAML also
contains `train.params.config.reward_scale`, it must be numeric and equal to
the historical field. Conflicts are rejected instead of silently overwritten.
This validation covers direct translation and CLI overrides of existing keys.

### CF-01: ineffective original task settings

The launcher emits a **configuration notice** when the following fields differ
from their original YAML values. It leaves them untouched and does not reject
otherwise valid historical configurations. The unchanged baseline is quiet.
Notices are printed before task construction for both training and evaluation.

| Field under `task.env` | Original behavior retained |
| --- | --- |
| `envSpacing` | Hard-coded spacing 1.0 |
| `control.controlFrequencyInv` | VecTask reads `env.controlFrequencyInv`, not the existing nested field |
| `urdfAsset.collapseFixedJoints` | Fixed joints always collapsed |
| `urdfAsset.fixBaseLink` | Free base |
| `urdfAsset.defaultDofDriveMode` | Task explicitly sets position drives, despite historical YAML value 4 |
| `baseInitState.rot`, `vLinear`, `vAngular` | These configured values are not applied to initial actors/reset snapshots |
| `randomCommandVelocityRanges.*` | Final commands overwritten with `[0,1,0,-1]` |
| `learn.linearVelocityXYRewardScale`, `angularVelocityZRewardScale` | Active reward uses hard-coded coefficients |

Notices compare the resolved values with the original baseline, so edited YAML
and command-line changes are both covered. A notice is not a statement that
the field has been implemented. Default joint angles still affect randomized
reset state, while position-target reset remains hard-coded; this is also
unchanged. No configuration key is remapped into new simulation behavior.

### CF-03: clipping validation

Both `clipActions` and `clipObservations` must be numeric, nonnegative and not
NaN. Booleans, strings, null, negative values and negative infinity are rejected.
**Zero and positive infinity are allowed**, consistent with tensor clamp:
zero suppresses that input and infinity leaves that clipping stage unbounded.
An omitted field retains VecTask's positive-infinity default.

Actions still pass through policy-side `[-1,1]` clipping followed by VecTask's
configured clipping. Limits 0.5, 1 and 2 therefore give effective bounds 0.5,
1 and 1. Policy sampling and stored log probabilities remain based on the raw
18-dimensional Gaussian samples. Observation clipping remains separate from
PPO's running observation normalization and its own clip limit.

## Regression coverage and CI

The initial `tests/test_audit_corrections.py` change added **20 tests** to the
existing 124 (the version 3 additions below extend this coverage):

- Three dependency-independent clipping/notice tests.
- Six OmegaConf tests for defaults, YAML/CLI precedence, reward-scale conflicts,
  ineffective settings, clip validation and explicit legacy trust.
- Eight checkpoint tests for restricted version 2 loading, NumPy/Python/torch
  RNG restoration, trusted version 1 loading, no automatic unsafe fallback,
  AMP transition policy, same-mode behavior and preflight LR validation.
- Three full-shape interface tests for `[N,104]` observations, initial zero
  observations and all-one done flags, deferred resets across rollouts,
  `[N,18]` raw action/log-probability ownership and two-stage clipping.

The full-shape fixture is an interface probe, not another physics simulator.
It deliberately cannot establish native GPU reset visibility. All old tests
remain. The static import check additionally permits the standard-library
`pickle` and `warnings` modules in the application layer.

The initial correction required 144 tests. The version 3 correction adds six
dependency-independent format tests and six torch-dependent regressions;
CI now requires at least **156 tests**, with every discovered test executed and
no skips, failures, errors or expected failures. The nine separate gate tests
also reject the obsolete 124- and 144-test counts. The workflow retains Ubuntu 22.04,
Python 3.10, torch 1.13.1+cpu, NumPy 1.24.4 and OmegaConf 2.3.0. The `fix/**`
push pattern covers this branch after an explicitly authorized future push.
No corrected workflow was pushed or run on GitHub during this task. The
previously published failed run was read to diagnose this correction.

The new tests cover all metadata types and float precision; ordered state-dict
metadata; opaque tensor references; malformed tags/envelopes; unsupported
objects; actual tensor-archive opcode inspection against the 1.13.1 handler
list; real restricted loading with populated Adam, AMP and cached RNG fields;
version 2 migration; default rejection and explicitly trusted loading under an
old-API signature double; and exact next-update CPU restoration. Existing
version 1, RNG, normalizer, optimizer, AMP and integration tests remain, with
fixture mutation updated to decode/re-encode version 3 rather than editing an
obsolete flat wire format. No failing assertion or tolerance was weakened.

## Local validation and additional verification

Initial local Python 3.12.14 discovery (before version 3): **144 discovered, 30 passed, 114 skipped,
zero failures/errors**. Of the skips, 102 require torch and 12 require
OmegaConf. The nine CI-runner tests pass. Python 3.10.11 is also present but
lacks torch and OmegaConf. No dependencies were installed. The strict CI
entrypoint correctly rejects the missing torch dependency; local skipped
tests do not establish successful numerical validation.

All 35 package/test/CI Python files parse with Python 3.8 grammar. The original
124 test cases remain present. Whitespace and protected-source diff checks pass.
No YAML parser or actionlint is available locally: the workflow's sole change
is the reviewed `fix/**` branch pattern, not a validated GitHub execution.
Historical reports in other documents remain records of their original runs.

The earlier 2026-11-17 GPU compatibility follow-up changed only this document and its
link in `stage3-integration.md`. Repeated local discovery produced the same
144 total / 30 passed / 114 skipped result, and all nine CI-runner tests passed.
The dependency requirement, checkpoint code, workflow and GPU defaults were
left intact. CPU numerical functionality still awaits an actual complete CI
run; CUDA and native simulator functionality await the procedure below.

After the version 3 correction, local discovery finds **156 tests: 36 passed,
120 skipped, no failures/errors**. The skips comprise 108 torch-dependent and
12 OmegaConf-dependent tests. The six metadata-format tests execute without
torch; they do not verify tensor serialization or CUDA. Local dependencies
remain unavailable and were not installed. Full acceptance requires a new CI
run with all 156 tests executed and zero skips/failures/errors. The published
144-test failure is retained as historical evidence, not relabeled a pass.
All nine CI-runner tests pass; the strict entrypoint rejects missing torch.
All 37 package/test/CI Python files parse with Python 3.8 grammar. Existing
test methods remain, with the v2 round-trip test renamed for v3. Whitespace
checks pass, and original simulation/assets/YAMLs, PPO/actor-critic code,
dependency requirements and workflow dependency versions remain unchanged.

Reproduce the offline CPU CI validation using the unchanged dependency commands in
`.github/workflows/pytorch-tests.yml`, then run:

```bash
python -B -m unittest discover -s ci -p 'test_*.py' -v
python -B ci/run_stage1_tests.py
```

To verify the changed PyTorch default independently, use a separate Linux
Python 3.10 environment (commands below are a procedure, **not executed here**):

```bash
python3.10 -m venv /tmp/hexapod-torch26-validation
/tmp/hexapod-torch26-validation/bin/python -m pip install --upgrade pip
/tmp/hexapod-torch26-validation/bin/python -m pip install numpy==1.24.4 omegaconf==2.3.0
/tmp/hexapod-torch26-validation/bin/python -m pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
/tmp/hexapod-torch26-validation/bin/python -m pip install --no-deps --editable .
/tmp/hexapod-torch26-validation/bin/python -m pip check
/tmp/hexapod-torch26-validation/bin/python -B -c "import torch; assert torch.__version__ == '2.6.0+cpu'; assert torch.version.cuda is None"
/tmp/hexapod-torch26-validation/bin/python -B -m unittest discover -s ci -p 'test_*.py' -v
/tmp/hexapod-torch26-validation/bin/python -B ci/run_stage1_tests.py
```

Run in the repository root. Require all 156 tests with zero skips in both
version environments. CUDA resume additionally needs a real CUDA-enabled
installation: save after CPU optimization with `mixed_precision=True`, resume
the same PPO settings onto CUDA, verify the fresh-scaler warning and preserved
Adam state, then perform a finite update and a same-CUDA save/resume. This
check needs CUDA, but does not require Isaac Gym.

### Required validation on the original GPU machine

This procedure is future validation, not an installation task or a record of
successful execution. Preserve the existing working environment throughout.

1. Record Ubuntu, GPU and driver versions (`nvidia-smi`), Python, installed
   torch/CUDA build, NumPy and dependency versions. Read the actual Preview 4
   SDK's `docs/index.html`, `python/setup.py` and
   `python/rlgpu_conda_env.yml`; record the installed IsaacGymEnvs commit.
   Import Isaac Gym before torch when checking them in the same process:

   ```bash
   python -c "import sys; import isaacgym; import torch; print(sys.version); print(isaacgym.__file__); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
   python -m pip check
   ```

   Do not use package installation to discover compatibility. If the installed
   torch is below the declared floor, stop before running or installing the
   custom package and review a targeted compatibility change using that exact
   version. An unrestricted loading retry is not a remedy.
2. In a compatible, already provisioned CUDA environment, first run the complete
   offline suite with no skips. These tests and the scaler doubles remain CPU
   checks even when the host has a GPU; their success alone is insufficient.
3. Exercise checkpoints without Isaac Gym using the real CUDA PPO/scaler:
   optimize a fixed batch until Adam state is populated and AMP state is
   nonempty, then save and restore into a fresh CUDA instance with identical
   settings. Compare parameters, normalizers, Adam moments and step counters,
   LR, scaler state, counters and the next CPU/CUDA RNG draws. Check model and
   moment devices. Run the same next fixed batch from both saved states and
   compare finite losses and updates using justified numerical tolerances.
   Keep visible GPU identity/order fixed. Test both enabled and disabled AMP;
   a same-mode restore must not issue the fresh-scaler transition warning.
4. Separately save a CPU checkpoint with `mixed_precision=True` (disabled
   CPU scaler), then resume on CUDA with the same PPO settings. Require the
   explicit fresh-scaler warning, restored model/normalizers/Adam/LR/counters,
   a fresh enabled scaler, and a finite CUDA update. Do not expect nonexistent
   saved CUDA RNG state or AMP history to appear. Verify that trusted version 1
   files require the explicit flag and malformed restricted loads never retry
   unrestricted. Perform these checks only with controlled checkpoint fixtures.
5. Finally run the unchanged hexapod task under Preview 4 on Ubuntu 20.04 with
   `pipeline=gpu sim_device=cuda:0 rl_device=cuda:0 num_envs=256`, first for a
   short train/save/resume/evaluate cycle in a new output directory. Confirm
   GPU PhysX, CUDA observation/action/model tensors, finite optimization,
   episode/reset behavior and successful checkpoint restoration. Keep the
   original assets, rewards and PPO configuration. This is a native smoke
   test, not evidence of recovered learning curves or scientific reproduction.

Record the complete version inventory and actual results separately for:
**CPU software tests**, **CUDA checkpoint/optimizer execution**, and
**native Isaac Gym training**. Until these steps are executed, only the
dependency-independent local checks and the source review are established.

## Deliberately unresolved

The original task, all original YAMLs, URDFs, meshes and PPO mathematical code
are unchanged. No simulator execution or research reproduction is claimed.
These concerns remain for native verification or separately approved changes:

- GPU reset/setter/refresh ordering and stale-state visibility.
- Force-sensor attachment at body index 5.
- Imported joint/body ordering and hard-coded contact indices.
- Joint limits, scaled targets and randomized reset positions.
- Stale angular-velocity history in the original reward calculation.
- Differences between the paper and executable experiment.
- Actual original task/JIT import, native cleanup and cross-device simulation.
- CK-03: resume still checks learning configuration, not environment/asset
  identity; simulator state and episode history are not restored.

No default learning equations, rewards, observations, actions, coupling or
termination conditions were changed. Intentional differences are the versioned
checkpoint representation, explicit legacy trust and fresh AMP transition
policy, corrected configuration precedence, notices and input validation.
