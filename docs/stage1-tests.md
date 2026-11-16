# Stage 1 implementation and validation report

Date: 2026-11-12. Branch: `feat/custom-pytorch-ppo`.

Stage 1 adds the independent feedforward network and historical specifications.
No PPO optimizer, normalization implementation, rollout collector, training loop,
simulator adapter, LSTM, or checkpoint manager has been implemented. No original
tracked files were edited. No dependencies were installed, and no commits,
pushes, pull requests, or merges were performed.

## Files added

| File | Purpose |
| --- | --- |
| `pyproject.toml` | Minimal src-layout packaging; torch is the only runtime dependency |
| `src/hexapod_rl/__init__.py` | Lightweight package/config export without importing torch |
| `src/hexapod_rl/config.py` | Immutable validated network configuration |
| `src/hexapod_rl/distributions.py` | Unsquashed diagonal Gaussian and explicit joint log density |
| `src/hexapod_rl/networks/__init__.py` | Network exports |
| `src/hexapod_rl/networks/actor_critic.py` | Independent actor/critic, sampling and evaluation APIs |
| `tests/test_static_contract.py` | Nine dependency-independent checks |
| `tests/test_actor_critic.py` | Twenty-three PyTorch numerical/runtime tests |
| `docs/baseline-contract.md` | Local environment contract and paper discrepancies |
| `docs/reference-implementation.md` | Pinned sources, verified defaults, mapping and remaining unknowns |
| `docs/stage1-tests.md` | This report and execution instructions |

The model defaults to separate 104->512->256->128 ELU networks with an
18-dimensional identity mean head and scalar identity value head. Linear
constructor weights are retained and biases zeroed. A trainable global
18-vector log_std starts at zero. Actions are Normal samples; densities and
entropy are summed across all 18 components. No clipping, squashing, coupling,
running normalization, or environment imports occur inside the network.

Public source retrieval was used to verify RL-Games v1.6.0 commit
`fa1c13cc10158ab42e829a0971a9dc9a4544b3b9` and IsaacGymEnvs commit
`aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3` (2023-10-18). Source files were kept
outside the repository in a temporary inspection directory, not installed or
vendored. The specification contains permanent source links.

## Executed checks

Two existing Windows interpreters were checked using importlib.util.find_spec:

| Interpreter | Version | torch | pytest |
| --- | --- | --- | --- |
| `C:/Users/USER/AppData/Local/Programs/Python/Python39/python.exe` | 3.9.10 | absent | absent |
| `C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe` | 3.12.14 | absent | absent |

pytest is not needed: this suite uses standard-library unittest. No automatic
dependency installation was attempted.

Commands executed from the repository root:

```powershell
& 'C:/Users/USER/AppData/Local/Programs/Python/Python39/python.exe' -B -m unittest discover -s tests -v
& 'C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -m unittest discover -s tests -v
git diff --check
git diff --exit-code -- hexapod.py hexapod.yaml hexapodPPO.yaml hexapodPPO_LSTM.yaml hexapod2
```

Each unittest invocation discovered 32 tests: **9 passed, 23 skipped, 0 failed**.
All skipped tests explicitly report that PyTorch is missing. This is not a
successful numerical validation of the network. It is the dependency-blocked
outcome allowed by Stage 1's completion criteria.

The passing checks cover configuration agreement with selected local fields,
configuration validation/immutability, parsing all new Python files, static
runtime-import restrictions, active URDF joint/mimic counts, active mesh
references, and documentation of selected revisions. The YAML checks are narrow
assertions, not a full YAML parser or resolver implementation.

Additional checks: Python 3.12 tomllib parsed package metadata; a whitespace
scan of the new source/tests/docs found no trailing whitespace; git diff checks
found no edits to tracked baseline files. Source inspection and test syntax
checks do not establish that torch imports or numerical operations succeed.

## Numerical tests included but blocked

The 23 tests cover:

- Baseline layer widths, ELU placement/alpha, head dimensions, and independent
  actor/critic parameter storage.
- Forward/sampling shapes for batches 1, 7, and 256 on CPU.
- Native Linear constructor weights retained exactly in a controlled seed test,
  zero biases, trainable initial log_std, and initial unit standard deviations.
- Configurable widths/activation/log_std and the no-hidden-layer case.
- A hand-calculated nonzero-weight ELU actor/critic fixture with unbounded means.
- Deterministic mean selection without RNG consumption and reproducible
  construction/sampling under fixed seeds.
- Equality to Normal.sample(), raw actions exceeding 1, and agreement of sampled
  action log densities with later evaluation.
- Policy/critic gradient separation, finite gradients, and nonzero log_std
  gradients.
- Evaluation interfaces, invalid shapes, and native state_dict restoration.
  The state_dict test is in-memory; it is not a checkpoint-manager test.
- A fresh-process import/forward test that rejects any attempt to import
  isaacgym, isaacgymenvs, or rl_games.
- Analytical standard-normal and non-unit Gaussian likelihood/entropy fixtures,
  analytical gradients, finite-difference gradcheck, and representative finite
  inputs. These are mathematical fixtures, not calls into RL-Games.

## Running later without simulator packages

Use a Python environment where a compatible CPU torch is already installed.
No package installation is needed for tests: they add the repository's src
directory to sys.path. To avoid mistaking dependency skips for full validation,
first run a torch preflight and require it to succeed:

```text
python -B -c "import torch; print(torch.__version__)"
python -B -m unittest discover -s tests -v
```

A fully validated CPU run should report 32 tests with **zero skips and failures**.
An installed but broken torch import is allowed to fail loudly rather than
being misreported as an absent optional dependency. Static checks alone:

```text
python -B -m unittest discover -s tests -p test_static_contract.py -v
```

For an interactive usage example, with `src` on PYTHONPATH:

```python
import torch
from hexapod_rl import ActorCriticConfig
from hexapod_rl.networks import ActorCritic

model = ActorCritic(ActorCriticConfig()).to("cpu")
observations = torch.zeros(8, 104)  # Caller will supply prepared observations.
sample = model.sample_actions(observations)
evaluation = model.evaluate_actions(observations, sample.actions)
mean_actions = model.deterministic_actions(observations)
```

This example is provided for later execution; it was not run on this host.
Native `.to('cuda')` is supported structurally, but GPU/AMP behavior has not been
tested. Very extreme log_std values may overflow/underflow, as no clipping was
added to the historical parameterization.

## Next-stage recommendations and remaining limits

First execute the blocked numerical tests on a CPU torch environment. Stage 2
then needs observation/value running statistics with verified update timing,
owned rollout storage, timeout reward shaping and GAE, advantage normalization,
clipped actor/value objectives and bounds loss, Adam/AMP gradient handling, and
the legacy adaptive KL schedule. Test each with deterministic fixtures and a
mock vector environment before live integration. See the reference specification
for unusual behaviors that generic PPO implementations often omit.

Actual original dependency versions, simulator import/body ordering, sensor
attachment, physics results, original checkpoints, and equivalent learning
performance remain unverified. Normalization, training, legacy checkpoint
conversion, and live simulator compatibility are not Stage 1 deliverables.
No Stage 2 work should begin without explicit approval.

## Subsequent CI validation setup (2026-11-15)

The report above records the original 2026-11-12 local results and remains
unchanged. The workflow added afterward is **not evidence of a successful GitHub
run**. No successful Actions execution has been observed as part of this setup.

`.github/workflows/pytorch-tests.yml` runs the **Stage 1 PyTorch validation**
workflow on a GitHub-hosted `ubuntu-22.04` x64 runner with Python **3.10**. It uses
the officially documented [actions/checkout@v7](https://github.com/actions/checkout)
and [actions/setup-python@v7](https://github.com/actions/setup-python). Its triggers
are PRs targeting `main`, pushes to `main`, `feat/**`, `feature/**`, and `codex/**`,
and manual `workflow_dispatch`. The token has only `contents: read`; checkout
does not persist credentials. There are no secrets or `pull_request_target`
events. The job has a 15-minute timeout.

### Dependencies and preflight

CI installs **torch==1.13.1+cpu** using the official CPU wheel repository, following
the [historical PyTorch instructions](https://pytorch.org/get-started/previous-versions/#v1131)
with the unneeded torchvision/torchaudio packages omitted. The official wheel
index lists the CPython 3.10 Linux x86_64 CPU wheel. **numpy==1.24.4** keeps NumPy
on the 1.x ABI for this older torch build; its
[release notes](https://numpy.org/doc/stable/release/1.24.4-notes.html) identify
Python 3.8-3.11 support. These are CI validation pins, not claims about the
original developer's installed NumPy or Python versions.

```text
python -m pip install --upgrade pip
python -m pip install numpy==1.24.4
python -m pip install torch==1.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps --editable .
python -m pip check
```

The editable install's `--no-deps` prevents the project's broad `torch>=1.13`
requirement from selecting another torch build. Package metadata is unchanged.
CI does not install Isaac Gym, IsaacGymEnvs, RL-Games, CUDA toolkits, or GPU
training dependencies. A preflight prints and verifies Python 3.10, the exact
torch/NumPy versions, installed project metadata and package imports. It requires
`torch.version.cuda is None` and no available CUDA device, and executes a CPU
tensor reduction plus tensor-to-NumPy conversion. Import or dependency failures
stop the job before test execution.

CPU execution is sufficient for this stage's feedforward tensor operations,
Gaussian equations, deterministic fixtures, and autograd checks. It does not
validate GPU numerical behavior, mixed precision, simulator integration, or
training performance.

### Enforcing complete test execution

The job uses only standard-library unittest:

```text
python -B -m unittest discover -s ci -p 'test_*.py' -v
python -B ci/run_stage1_tests.py
```

The first command tests the CI gate itself with nine dependency-independent
synthetic scenarios. These tests live outside `tests/` so they cannot substitute
for any of the existing 32 Stage 1 tests.

`ci/run_stage1_tests.py` first requires a successful torch import, then uses
`unittest.TestLoader.discover` on `tests/test_*.py`. It captures the discovered
count before execution and runs the full suite with verbose tracebacks. It
prints discovered/executed/failure/error/skip counts and exits nonzero if:

- Fewer than 32 tests were discovered (additional tests are allowed).
- The executed count differs from the discovered count.
- Any test or subtest was skipped, even when unittest otherwise reports success.
- Any failure, error, unexpected success, or expected failure occurred.

Failures also produce GitHub `::error::` annotations. A normal unittest zero exit
status with skipped numerical tests is therefore insufficient to pass this job.
No Stage 1 tests, assertions, tolerances, or neural-network code were changed.

### Viewing results and interpreting success

After these changes are pushed through the normal review process, open the
repository's [Actions page](https://github.com/Leo813/hexapod_proj/actions), select
**Stage 1 PyTorch validation**, and open the run/job logs. PRs also show the job
under **Checks**. Inspect the dependency preflight and **Run all Stage 1 tests
with no skips** steps, not merely the nine CI-gate tests. Manual **Run workflow**
becomes available once the workflow exists on the default branch; select the
desired branch when dispatching it.

For the current suite, final validation requires an actual successful GitHub
run reporting **32 discovered, 32 executed, zero failures/errors/skips**. A green
run establishes that the checked-out Stage 1 tests passed on the selected CPU
stack. It does not establish exact original-version provenance, differential
execution against RL-Games, simulator compatibility, GPU parity, or equivalent
learning curves. Stage 2 still requires explicit approval.

### Local checks for the CI change

On 2026-11-15, the existing Python 3.9.10 interpreter ran both commands below:

```text
python -B -m unittest discover -s ci -p 'test_*.py' -v
python -B -m unittest discover -s tests -p test_static_contract.py -v
```

All nine CI-gate tests and all nine static tests passed. Full discovery under
the existing Python 3.12.14 interpreter still reported 32 tests: nine passed and
23 numerical tests skipped because torch remains absent. Running the new CI
entrypoint returned exit code 1 with a PyTorch-import error, as intended.
Separately exercising its result policy on the actual discovered suite returned
failure with `skipped=23`, verifying that skipped numerical coverage is rejected.
These rejection checks are successful checks of CI behavior, not numerical
test successes.

The new runner scripts and extracted workflow Python preflight passed AST
parsing. Workflow structure/indentation was reviewed against the official action
usage; no YAML parser or actionlint was available locally, so no full local
workflow-schema validation is claimed. `git diff --check` passed. The historical
report prefix was verified unchanged, and the diff confirmed that existing
network/tests, package metadata, original environment/configurations, and robot
assets were untouched. No local dependencies were installed. Dependency
installation on Linux and execution of all 32 tests without skips still require
an actual GitHub Actions run.

## Subsequent Stage 2 coverage (2026-11-15)

The historical reports above are preserved. Stage 2 adds PPO core tests and
renames the workflow display to **Stage 1 and 2 PyTorch validation**. The existing
runner now discovers both stages and requires at least 87 tests, with zero skips
and complete execution. All 32 Stage 1 tests remain unchanged. Dependency pins,
runner OS and permissions remain unchanged. See
[stage2-implementation.md](stage2-implementation.md) for the new algorithms,
coverage, local dependency-blocked results and remaining validation requirements.
No successful Stage 2 GitHub run has been observed during that implementation.
