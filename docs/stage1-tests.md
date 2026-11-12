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
