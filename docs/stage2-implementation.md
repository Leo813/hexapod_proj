# Stage 2: independent feedforward PPO core

Date: 2026-11-15. Branch: `feat/custom-ppo-stage2`.

Stage 2 implements PPO mathematics, data preparation, normalization, and optimizer
updates around the existing Stage 1 `ActorCritic`. It does not construct an Isaac
Gym environment or add a production collection/training loop. Original task code,
configuration YAML, rewards, physics, URDFs, and meshes remain unchanged. The paper
remains scientific background; the source/configuration contract governs behavior.

**Validation status:** numerical execution is unverified on this development host
because PyTorch is unavailable. The tests described below are implemented, not
claimed to have passed numerically. No Stage 2 GitHub Actions run or GPU run has
been observed. Source agreement is not proof of equivalent learning performance.

## Architecture and interfaces

| File | Responsibility |
| --- | --- |
| `src/hexapod_rl/config.py` | Immutable, validated `PPOConfig`; existing network config preserved |
| `src/hexapod_rl/normalization.py` | Vector/scalar `RunningMeanStd`, normalization and denormalization |
| `src/hexapod_rl/rollout.py` | Owned tensor storage, bootstrap state, environment-major flattening, deterministic `TrainingBatch` slices |
| `src/hexapod_rl/advantages.py` | Timeout reward adjustment, GAE, global advantage standardization |
| `src/hexapod_rl/ppo.py` | Individually exposed losses, rollout policy/value evaluation, batch preparation, Adam/AMP optimization |
| `src/hexapod_rl/schedules.py` | Historical Gaussian KL equation and legacy adaptive learning rate |
| `tests/ppo_test_support.py` | Test-only arithmetic vector environment and collection helper |
| `tests/test_{normalization,rollout,advantages,ppo_losses,ppo_optimization,schedules,ppo_config}.py` | Stage 2 numerical fixtures and dependency-independent checks |

`hexapod_rl.__init__` additionally exports `PPOConfig` without requiring torch.
The Stage 1 network and Gaussian implementation are reused without edits. Runtime
imports remain limited to PyTorch, the standard library, and this package; tests
never import `hexapod.py` or install/import simulator or RL-Games packages.

Construct the network on its final device, then pass that same instance to
`PPO(model, config)`. The optimizer owns its parameters, including global log_std.
Do not move the model to another device after creating the optimizer/normalizers.
The baseline uses float32 model/rollout data and float64 running accumulators.

The caller supplies environment-prepared observations. `PPO.sample(obs)` returns
raw actions, their joint **log probabilities**, reward-unit values, means, and
standard deviations. `predict_values(obs)` bootstraps the last state without
sampling or consuming an action RNG draw. Both use evaluation mode and frozen
normalization statistics.

The future collector's sequence is:

```python
# Illustrative interface only; no production simulator adapter exists yet.
policy = algorithm.sample(obs)
buffer.store_policy_step(obs, dones, *policy)  # BEFORE env.step mutates anything
env_actions = policy.actions.clamp(-1, 1).clone()
next_obs_dict, rewards, next_dones, extras = env.step(env_actions)
buffer.store_outcome(rewards, next_dones, extras.get("time_outs"))
obs, dones = next_obs_dict["obs"], next_dones
# After horizon_length steps:
buffer.finish(algorithm.predict_values(obs))
metrics = algorithm.update(buffer)
buffer.reset()
```

Clipping/rescaling and simulator calls belong to the future adapter, not to the
Gaussian likelihood. The example's [-1,1] transform assumes the current action
space; the mock tests use it explicitly. The original 18-dimensional density,
including six subsequently coupled action components, must be retained.

`prepare_batch(buffer)` exposes preparation separately; `optimize(batch)` accepts
prepared, detached `TrainingBatch` data. `update(buffer)` combines them. Prepare a
rollout once: preparation updates value statistics. Calling it twice on the same
rollout intentionally performs two more statistics updates. Each optimization
call uses the configured mini-epochs and returns individual loss/KL/LR/gradient
norm metrics. Reported gradient norm is the norm **before** clipping.

## Historical sources and correspondence

The selected baseline is RL-Games v1.6.0, commit
`fa1c13cc10158ab42e829a0971a9dc9a4544b3b9`, and IsaacGymEnvs commit
`aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3`. This selection does not establish the
original installed versions. The cached, pinned upstream source was reinspected
for Stage 2. The implementation independently expresses its equations and control
flow; no complete upstream module is copied, vendored, or imported.

| Pinned reference | Behavior mapped here |
| --- | --- |
| [RunningMeanStd.forward/update_from_moments](https://github.com/Denys88/rl_games/blob/fa1c13cc10158ab42e829a0971a9dc9a4544b3b9/rl_games/algos_torch/running_mean_std.py) | Initial moments, sample variance, merge equation, clipping and precision |
| [BaseModelNetwork and ModelA2CContinuousLogStd](https://github.com/Denys88/rl_games/blob/fa1c13cc10158ab42e829a0971a9dc9a4544b3b9/rl_games/algos_torch/models.py) | Input normalization, rollout value denormalization, raw Normal policy |
| [A2CBase.play_steps/discount_values/trancate_gradients_and_step; ContinuousA2CBase.prepare_dataset/train_epoch](https://github.com/Denys88/rl_games/blob/fa1c13cc10158ab42e829a0971a9dc9a4544b3b9/rl_games/common/a2c_common.py) | Rollout ownership, timeout shaping, GAE, units/timing, gradient order, legacy scheduling |
| [ExperienceBuffer.update_data](https://github.com/Denys88/rl_games/blob/fa1c13cc10158ab42e829a0971a9dc9a4544b3b9/rl_games/common/experience.py) and [PPODataset](https://github.com/Denys88/rl_games/blob/fa1c13cc10158ab42e829a0971a9dc9a4544b3b9/rl_games/common/datasets.py) | Copies before stepping, contiguous slices, mutable mu/sigma references |
| [A2CAgent.__init__/calc_gradients/bound_loss](https://github.com/Denys88/rl_games/blob/fa1c13cc10158ab42e829a0971a9dc9a4544b3b9/rl_games/algos_torch/a2c_continuous.py) | Adam, total objective, soft bounds, KL argument order |
| [common_losses.critic_loss](https://github.com/Denys88/rl_games/blob/fa1c13cc10158ab42e829a0971a9dc9a4544b3b9/rl_games/common/common_losses.py) and [torch_ext.actor_loss/policy_kl](https://github.com/Denys88/rl_games/blob/fa1c13cc10158ab42e829a0971a9dc9a4544b3b9/rl_games/algos_torch/torch_ext.py) | Clipped losses, reductions and reference-specific KL |
| [AdaptiveScheduler.update](https://github.com/Denys88/rl_games/blob/fa1c13cc10158ab42e829a0971a9dc9a4544b3b9/rl_games/common/schedulers.py) | KL thresholds, LR factors and limits |
| [PyTorch 1.13.1 Adam](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/optim/adam.py) | Inherited beta defaults and disabled AMSGrad |

See [baseline-contract.md](baseline-contract.md) for the environment's deferred
resets, action mutation, observation layout, paper discrepancies, and wrapper
constraints. These have not been silently corrected.

## Normalization equations, units, and timing

For each observation feature (or the single value), initialize accumulated mean
`m=0`, variance `v=1`, count `c=1`, all float64. For a batch of `n` samples, compute
batch mean `b` and **sample variance** `s²` with denominator `n-1` in the input
dtype. With `d=b-m` and `C=c+n`, merge:

```text
m_new = m + d*n/C
v_new = (c*v + n*s² + d²*c*n/C) / C
c_new = C
```

The `n*s²` term and initial count 1 are intentional reference behavior. This is
not the population-variance update often used in other PPO implementations.
Outputs use `float32(m)` and `sqrt(float32(v) + 1e-5)`:

```text
normalize(x)   = clamp((x - m) / sqrt(v + epsilon), -5, 5)
denormalize(z) = clamp(z, -5, 5) * sqrt(v + epsilon) + m
```

The implementation permits configurable positive epsilon and clipping (including
`clip=None` on the normalizer). PPO baseline clipping remains 5. `train()` updates
moments even on a denormalization call; `eval()` never updates them. Accumulators
are registered buffers: native `state_dict`/`load_state_dict` save and restore all
three. Reconstruct with the same size/epsilon/clip settings before loading; these
configuration values and module train/eval mode are not tensor state. Transfer
normalizers by device, without casting their float64 buffers to model dtype.

Timing is part of the algorithm:

1. Rollout input/value statistics are frozen. Network value output is in
   normalized units; stored values and final bootstrap are denormalized.
2. Compute returns and advantages in reward units before value normalization.
3. Train the value normalizer on flattened old values, transforming them with
   those updated statistics. Then update it again on returns, transforming them
   with the newer statistics. Freeze it afterward. These two transformations
   deliberately use different moments.
4. During optimization, update observation moments before each minibatch forward
   in the first mini-epoch only. Freeze for subsequent passes. Re-enable at the
   start of the next rollout's optimization.

## Rollout organization and GAE

Rollouts are tensors `[T,N,...]` on the requested device: observations, raw sampled
actions, positive-sign log densities, reward-unit values, raw rewards, pre-action
done flags, outcome done flags/timeouts, means, and standard deviations. Final
values and final done flags are owned tensors too. Values/rewards have a final
singleton axis; log densities and flags do not. Store methods copy/detach inputs
and validate shape/device. No NumPy or CPU/GPU transfers are used for storage.

Two-phase storage makes the required ordering explicit. The policy fields must
be copied before stepping an environment that may reuse observations or mutate
actions. Outcome fields are copied immediately after stepping. The caller must
carry `next_dones` into the following step's `dones`. `finish()` requires all `T`
steps and records the last outcome's done flags with the supplied bootstrap value.

First shape rewards, using the **pre-action** denormalized value:

```text
r'_t = reward_scale*r_t + gamma*V_t*timeout_t   # if value_bootstrap enabled
```

There is no terminal-observation value substitution. A timeout also remains a
done boundary. For each environment independently, use the next observation's
done (`dones[t+1]`, or final `last_dones`), defining `q_t=1-next_done`:

```text
delta_t = r'_t + gamma*q_t*V_next - V_t
A_t = delta_t + gamma*tau*q_t*A_next
R_t = A_t + V_t
```

At the rollout horizon `A_next=0` and `V_next=last_values`; a continuing horizon
bootstraps, a terminal horizon masks that value. Episode boundaries stop both TD
bootstrap and recursive advantages. This matches deferred-reset data semantics
rather than imposing Gymnasium autoreset/terminated/truncated conventions.

Transpose time/environment axes before flattening: `[T,N,...]` becomes
`[N*T,...]`, with all time steps for environment 0 first. Prepared data owns
storage independent of the reusable rollout. Standardize scalar advantages
globally as `(A-mean(A))/(sample_std(A)+1e-8)`, with epsilon outside the std.

`TrainingBatch.minibatches(size)` yields contiguous slices from the start every
pass, with **no shuffle**. Batch size must divide exactly. Baseline `256*48=12288`
is one minibatch, repeated for five mini-epochs. Old log densities remain fixed;
mean/std references are overwritten after each minibatch with that minibatch's
forward distribution. Callers supplying a TrainingBatch directly must supply
owned, detached data; minibatch slices are views.

## PPO objective

Using positive log densities `l_new`, `l_old`, let `r=exp(l_new-l_old)` and clip
width `e=0.2`. The reference uses negative log densities; reversing the difference
here preserves its ratio. Mean reductions are over the minibatch:

```text
L_policy = mean(max(-A*r, -A*clamp(r, 1-e, 1+e)))
V_clip = V_old + clamp(V_new - V_old, -e, e)
L_critic = mean(max((V_new-R)², (V_clip-R)²))
H = mean(sum_actions(0.5*log(2*pi*euler_constant) + log_std))
L_bounds = mean(sum_actions(max(mu-1.1,0)² + min(mu+1.1,0)²))
L_total = L_policy + 0.5*critic_coef*L_critic
          - entropy_coef*H + bounds_loss_coef*L_bounds
```

Disabling value clipping uses only the unclipped squared error. The critic loss
itself contains no half factor; the half appears in the total objective. Bounds
penalize **means**, not sampled/clipped/coupled actions. `LossTerms` exposes all
five reduced quantities. There is no mean squashing, std clipping, KL loss, or KL
early stopping.

## Optimization and learning rate

Adam uses LR `3e-4`, epsilon `1e-8`, betas `(0.9,0.999)`, weight decay 0, AMSGrad
disabled. These beta values are explicit versions of PyTorch 1.13.1's defaults,
not evidence of the original installed torch version. Each minibatch does:

1. Forward/loss under CUDA autocast when enabled on a CUDA model.
2. Clear parameter gradients to None, scale loss, backward.
3. If truncation enabled: unscale, then clip total gradient norm to 1.
4. `GradScaler.step(optimizer)`, then `GradScaler.update()`.
5. Compute KL with the just-evaluated **pre-optimizer-step** distribution,
   refresh stored mean/std for the slice, then update LR for the next minibatch.

CPU disables CUDA AMP even when the baseline config requests it. The historical
`torch.cuda.amp` APIs are used for compatibility with PyTorch 1.13.1. This CPU
path does not establish CUDA/mixed-precision numerical equivalence. Metrics
conversion to Python scalars entails synchronization on GPU; storage otherwise
stays on the model device.

Let `(m0,s0)` be current forward parameters and `(m1,s1)` stored KL references:

```text
KL = mean_samples(sum_actions(
    log(s1/s0 + 1e-5) + (s0² + (m1-m0)²)/(2*(s1² + 1e-5)) - 0.5))
```

This direction and both epsilon placements follow the reference. Identical
distributions can yield a small nonzero or negative result; it is not clamped.
Legacy adaptation occurs after **every minibatch**, including every pass:

```text
KL > 2*0.008   -> LR = max(LR/1.5, 1e-6)
KL < 0.5*0.008 -> LR = min(LR*1.5, 1e-2)
otherwise     -> unchanged
```

Threshold comparisons are strict. Entropy coefficient remains unchanged. A
constant schedule is an explicit customization, not the baseline.

## Configuration mapping

| Local YAML setting | PPOConfig default |
| --- | --- |
| `task.env.numEnvs` / resolved `num_actors` | `num_actors=256` |
| `horizon_length`, `minibatch_size`, `mini_epochs` | 48, 12288, 5 |
| `gamma`, `tau`, `e_clip` | 0.99, 0.95, 0.2 |
| `critic_coef`, `entropy_coef`, `bounds_loss_coef` | 2, 0, 0.001 |
| `clip_value`, `value_bootstrap` | True, True |
| `normalize_input`, `normalize_value`, `normalize_advantage` | True, True, True |
| `reward_shaper.scale_value` | `reward_scale=1` |
| `learning_rate`, `lr_schedule`, `kl_threshold` | 3e-4, adaptive, 0.008 |
| `truncate_grads`, `grad_norm`, `mixed_precision` | True, 1, True (CPU disables AMP) |

Other defaults come from the pinned reference: Adam settings, normalization
epsilons/clips, LR bounds, legacy schedule frequency and bounds-loss form. The
fixed feedforward PPO/single-value algorithm is implicit. The config rejects
invalid ranges, nonfinite settings and nondivisible minibatches. It does not
parse Hydra, resolve nested YAML interpolation, or manage seeds, experiment
names, epochs, scores, logging, checkpoints, or environment creation. Changing
defaults deliberately defines a new experiment.

## Offline tests and CI

All 32 Stage 1 tests are preserved. Stage 2 adds 55 tests:

| File | Tests | Evidence sought |
| --- | ---: | --- |
| `test_normalization.py` | 8 | Analytical moments/sequential updates, precision, epsilon/clipping, mode, saved state, edge cases |
| `test_rollout.py` | 7 | Ownership before/after mutation, lifecycle, shapes, environment-major order, repeated fixed slices, mutable KL references |
| `test_advantages.py` | 8 | Hand-calculated continuing/terminal/horizon/timeout and multienvironment GAE, sample std |
| `test_ppo_losses.py` | 8 | Both advantage signs/clip branches, value maximum, entropy, bounds/coefficients, analytical gradient and gradcheck |
| `test_ppo_optimization.py` | 12 | Analytical first Adam step, gradient clearing/order/clipping, CPU AMP fallback, normalization timing/units, KL refresh/LR frequency, mock collection and finite parameter updates |
| `test_schedules.py` | 7 | Four dependency-independent threshold/limit tests; three numerical KL direction/epsilon/reduction tests |
| `test_ppo_config.py` | 5 | Dependency-independent YAML mapping, immutable settings and validation |

The deterministic mock reuses output tensors, couples actions in-place, tracks
different per-environment episode lengths, returns timeout information, and
applies pending resets after integrating the next step. It models interface
hazards only, not robot dynamics or locomotion. The end-to-end fixture samples
the existing network, preserves raw-action densities, prepares targets, performs
multiple updates, and asserts changed finite parameters. That assertion remains
unexecuted locally until torch is available.

The existing workflow retains Ubuntu 22.04, Python 3.10, torch **1.13.1+cpu**,
NumPy 1.24.4, editable `--no-deps` installation and restricted token permissions.
Only workflow display labels change. `ci/run_stage1_tests.py` retains its filename
for compatibility but now discovers both stages and requires at least **87**
tests. Any skip, failure, error, expected failure, unexpected success or incomplete
execution fails CI. The nine separate CI-gate tests now exercise the larger
minimum and reject a Stage-1-only 32-test run. They cannot replace package tests.

```text
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s ci -p 'test_*.py' -v
python -B ci/run_stage1_tests.py
```

The first command permits skips for local inspection; the last is the strict CI
entrypoint and requires a successful torch import. Open the repository's
[Actions page](https://github.com/Leo813/hexapod_proj/actions), select **Stage 1 and
2 PyTorch validation**, and inspect the full-suite step and PR Checks after normal
review/push. A successful Stage 2 validation requires all 87 current tests to
execute with zero skips/errors/failures on the pinned CPU stack. No workflow was
dispatched and nothing was committed or pushed as part of this work.

Local validation on the existing Python 3.9.10 and 3.12.14 interpreters discovers
87 package tests: **18 passed, 69 skipped, zero failures/errors**. The 69 are 23
existing Stage 1 and 46 new Stage 2 numerical tests. All nine CI-gate tests pass.
The strict entrypoint rejects the missing torch import; its result gate also
rejects the actual skipped suite. AST/import-contract checks and `git diff
--check` pass. Original environment/configuration/asset paths and Stage 1 network,
distribution and test files have no diff. No local dependencies were installed.
These results do not establish numerical or PyTorch 1.13.1 runtime compatibility;
that still requires the workflow. No YAML parser/actionlint is available locally;
the workflow change is restricted to two display strings.

## Intentional boundaries and remaining Stage 3 work

- Singleton training batches are rejected when sample-variance normalization is
  enabled, instead of allowing the reference's undefined variance to poison state.
  Evaluation on a singleton is supported. Extra shape/device/lifecycle checks
  fail early. These defensive differences do not change the baseline batch.
- Only vector observations, one scalar critic, feedforward policy, legacy
  scheduling and a single device are implemented. No image/per-channel or
  norm-only normalization, RNN masks, centralized critic, multiagent/multi-GPU,
  alternate regularization loss, or distributed synchronization is supplied.
- The typed API uses positive log densities and separate normalizer modules;
  naming/layout do not match RL-Games state dictionaries. Normalizer, network,
  optimizer and scaler expose native state dictionaries, but there is no complete
  checkpoint/resume format or legacy checkpoint converter. Future checkpoints
  must capture all those states, configuration, LR, counters and RNG state.
- The actual original software versions, seeds/resolved configuration, body/DOF
  mapping, sensor attachment and trained checkpoints remain unknown. No upstream
  RL-Games differential run, GPU AMP test, simulator run or learning-curve
  comparison was performed. Extreme unconstrained log_std may overflow as in the
  selected policy parameterization; no speculative clamp was introduced.
- Stage 3 must construct/register the existing task while retaining `VecTask`
  and avoiding RL-Games-dependent convenience imports, resolve configuration and
  seeds/devices, implement the production collector/training launcher, and carry
  correct first-observation, action-copy/clipping, timeout and deferred-reset
  semantics. Add checkpoint/resume/evaluation/logging and validate tiny live
  rollouts before long training. Check simulator asset/body ordering and CPU/GPU
  transfers explicitly. LSTM and distributed training remain separate extensions.

Stage 3 requires explicit approval; it has not been started.
