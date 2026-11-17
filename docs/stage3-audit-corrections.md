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

New saves use **format version 2**. NumPy MT19937 state is stored as a dictionary
of strings, integer lists and scalar values instead of a NumPy array. Torch RNG
state remains tensors; Python RNG state remains built-in tuples. NumPy's cached
Gaussian and generator position are preserved. Application/resolved metadata
must likewise contain ordinary built-in values, not arbitrary Python objects.

Normal loading explicitly uses `torch.load(..., map_location="cpu",
weights_only=True)`. This argument exists in PyTorch 1.13.1 and avoids relying
on the changed default in PyTorch 2.6. Failure never triggers an automatic
unrestricted retry. Format version, architecture, PPO and normalization checks
remain in place; optimizer LR and CUDA RNG device-count checks occur before
model restoration.

**Use checkpoint files only from known sources.** Restricted loading reduces
pickle reconstruction capabilities; it is not a sandbox or a guarantee against
malicious files, particularly with a historical dependency release.

Existing **version 1** checkpoints require an explicit trust decision because
their NumPy RNG arrays use pickle reconstruction. Only for a file whose origin
and contents you trust:

```bash
python train.py task=hexapod checkpoint=/trusted/run/checkpoints/last.pt trust_legacy_checkpoint=True
python train.py task=hexapod test=True checkpoint=/trusted/run/checkpoints/best.pt trust_legacy_checkpoint=True
```

The equivalent API is `load_checkpoint(path, algorithm, trust_legacy=True)` or
`trainer.resume(path, trust_legacy=True)`. This flag selects
`weights_only=False` and emits a warning: **arbitrary pickle code may execute**.
It is never inferred from a filename, error, or checkpoint metadata. Leave it
false for version 2 files. A subsequent save writes version 2. RL-Games files
are still unsupported. Older project code cannot read version 2 files.

The historical CPU validation target remains Python 3.10 / PyTorch 1.13.1+cpu.
The new format is designed for restricted loading on 1.13.1 and 2.6.0, but the
broad `torch>=1.13` package declaration is not a claim that every intervening
or future release has been tested. No dependency versions are changed here.
See the exact additional verification procedure below.

References: [1.13.1 serialization API](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/serialization.py),
[2.6 serialization semantics](https://docs.pytorch.org/docs/2.6/notes/serialization.html).

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

`tests/test_audit_corrections.py` adds **20 tests** to the existing 124:

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

CI now requires at least **144 tests**, with every discovered test executed and
no skips, failures, errors or expected failures. The nine separate gate tests
also reject the obsolete 124-test count. The workflow retains Ubuntu 22.04,
Python 3.10, torch 1.13.1+cpu, NumPy 1.24.4 and OmegaConf 2.3.0. The `fix/**`
push pattern covers this branch after an explicitly authorized future push.
No workflow was pushed or run on GitHub during this task.

## Local validation and additional verification

Local Python 3.12.14 discovery: **144 discovered, 30 passed, 114 skipped,
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

Reproduce historical validation using the unchanged dependency commands in
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

Run in the repository root. Require all 144 tests with zero skips in both
version environments. CUDA resume additionally needs a real CUDA-enabled
installation: save after CPU optimization with `mixed_precision=True`, resume
the same PPO settings onto CUDA, verify the fresh-scaler warning and preserved
Adam state, then perform a finite update and a same-CUDA save/resume. This
check needs CUDA, but does not require Isaac Gym.

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
