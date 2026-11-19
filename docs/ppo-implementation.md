# Independent PPO implementation

This document records the maintained learning and application contracts. The
[executable baseline](baseline-contract.md) governs the environment; the
[reference mapping](reference-implementation.md) identifies upstream sources
and attribution. For installation and commands, use [Getting started](getting-started.md).

## Components and data ownership

`train_agent.py` delegates to `launcher.py`: configuration is composed before
Isaac Gym initialization, then tensor-dependent application modules are loaded.
`task_factory.py` establishes process-local package paths and loads the installed
`Anymal10` without executing the broad IsaacGymEnvs package/task initializers.
It rejects an already-imported IsaacGymEnvs namespace. Do not mix this loader
with `isaacgymenvs.make()` in one process; external base/utils code is unchanged.

| Module under `src/hexapod_rl/` | Responsibility |
| --- | --- |
| `config.py`, `integration/configuration.py` | Typed network/PPO controls and installed YAML composition |
| `networks/actor_critic.py`, `distributions.py` | Separate MLPs, trainable global log_std, raw Gaussian actions/densities |
| `normalization.py` | Observation and scalar-value running moments |
| `integration/isaacgym_adapter.py`, `collector.py` | Tensor contract, action clipping, persistent collection, episode statistics |
| `rollout.py`, `advantages.py` | Owned data, minibatches, timeout shaping and GAE |
| `ppo.py`, `schedules.py` | Losses, Adam/AMP updates, historical KL/LR rule |
| `trainer.py`, `evaluation.py` | Epoch lifecycle, logging/saving, frozen deterministic evaluation |
| `checkpoints.py`, `checkpoint_format.py` | Versioned learning-state persistence and metadata codec |

Construct the model on its final device before `PPO(model, config)`. Model and
rollout data use float32; normalizer accumulators use float64. `PPO.sample(obs)`
returns raw actions, positive joint log densities, denormalized values, means
and standard deviations. `predict_values(obs)` bootstraps without sampling.

The adapter accepts float32 `[N,104]` observations and `[N]` rewards on the RL
device, integer/bool `[N]` dones/timeouts, and 18-action bounds [-1,1]. Missing
`extras['time_outs']` is an error. VecTask clips observations; the adapter does
not repeat that transformation or silently move incorrectly placed results.

The collector calls `reset()` once, starts with all-one boundary flags, and
copies policy fields into `[T,N,...]` storage **before** `env.step`. Independently
allocated clipped actions protect raw samples from task-side mimic mutation.
Outcome tensors are copied immediately afterward. Observations/dones persist
across rollouts; no eager physics step, reset_done or immediate autoreset is
introduced. Values/rewards have a trailing singleton axis; log densities/flags
do not. Flattening swaps time/environment first, then copies into owned storage.

Episode statistics use raw rewards and a capped batch-weighted completed-episode
mean (capacity 100), matching reference AverageMeter rather than a last-100 deque.
`set_train_info(frames)` runs before collection. A completed update advances one
epoch and `N*T` frames; incomplete updates do not advance counters. Cleanup closes
the adapter and logger; the adapter destroys viewer/simulator handles when the
pinned VecTask has no `close()`. Partial task-construction failures may leave
native handles until process exit. Save gates are described in the user guide.

## Normalization and target preparation

Initialize running mean `m=0`, variance `v=1`, count `c=1` in float64. Compute
batch mean `b` and sample variance `s2` (denominator `n-1`) in the input dtype.
With `d=b-m` and `C=c+n`, merge:

```text
m_new = m + d*n/C
v_new = (c*v + n*s2 + d*d*c*n/C) / C
c_new = C
normalize(x)   = clamp((x-m)/sqrt(v+1e-5), -5, 5)
denormalize(z) = clamp(z, -5, 5)*sqrt(v+1e-5) + m
```

Transformations use float32 views of the statistics. The initial count and
`n*s2` merge term intentionally retain reference behavior. Registered buffers
hold the moments; size/epsilon/clipping settings must also match on restoration.
Move normalizers by device without casting their buffers to model dtype.

Rollout moments are frozen, and stored values are in reward units. Compute
advantages/returns before value normalization. Then update the value normalizer
on old values and transform them, update again on returns and transform them,
and freeze it. These transformations use different moments. Observation moments
update before each forward in the first optimization mini-epoch only, then
freeze until the next rollout's optimization.

Timeout shaping uses the pre-action value, not a terminal-observation value:

```text
r_shaped[t] = reward_scale*r[t] + gamma*V[t]*timeout[t]  # if bootstrap enabled
q[t] = 1 - next_done[t]
delta[t] = r_shaped[t] + gamma*q[t]*V_next - V[t]
A[t] = delta[t] + gamma*tau*q[t]*A_next
R[t] = A[t] + V[t]
```

The final continuation uses `last_values` with `A_next=0`. Stored `dones[t]`
describe the pre-action state; non-final next masks use `dones[t+1]`. Timeouts
remain done boundaries. Standardize advantages globally using sample standard
deviation: `(A-mean(A))/(std(A)+1e-8)`.

`prepare_batch(rollout)` updates value statistics, so call it once per rollout.
`optimize(batch)` operates on detached prepared data; `update(rollout)` combines
the two. Minibatches are contiguous, environment-major slices, with no shuffling.
Minibatch size must divide the batch exactly; baseline `256*48=12288` gives one minibatch per
pass for five passes. Old log densities stay fixed; KL mean/std references are
updated after each minibatch with its evaluated distribution.

## Losses and optimizer ordering

For positive log densities, `ratio=exp(logp_new-logp_old)` and clip width `e=0.2`:

```text
L_policy = mean(max(-A*ratio, -A*clamp(ratio, 1-e, 1+e)))
V_clip = V_old + clamp(V_new-V_old, -e, e)
L_critic = mean(max((V_new-R)^2, (V_clip-R)^2))
H = mean(sum_actions(log_std + 0.5*log(2*pi*euler_constant)))
L_bounds = mean(sum_actions(max(mu-1.1,0)^2 + min(mu+1.1,0)^2))
L_total = L_policy + 0.5*critic_coef*L_critic
          - entropy_coef*H + bounds_loss_coef*L_bounds
```

Without value clipping, use only unclipped squared error. Bounds penalize means,
not sampled/clipped actions. There is no tanh squashing, std clamp, KL loss or
KL early stopping. All 18 policy components contribute to likelihood and entropy.

Adam defaults are LR `3e-4`, epsilon `1e-8`, betas `(0.9,0.999)`, weight decay 0,
AMSGrad disabled. For each minibatch: forward/loss under enabled CUDA autocast;
clear gradients to None; scale/backward; unscale before clipping gradient norm
to 1; scaler.step; scaler.update; then evaluate KL and adapt the next LR.
Reported gradient norm precedes clipping. CPU disables CUDA AMP. Scalar logging
can synchronize GPU execution; CPU tests do not validate mixed-precision parity.

For current pre-update forward parameters `(m0,s0)` and stored `(m1,s1)`:

```text
KL = mean_samples(sum_actions(
    log(s1/s0 + 1e-5) + (s0^2 + (m1-m0)^2)/(2*(s1^2 + 1e-5)) - 0.5))
KL > 2*0.008   -> LR = max(LR/1.5, 1e-6)
KL < 0.5*0.008 -> LR = min(LR*1.5, 1e-2)
otherwise     -> unchanged
```

Comparisons are strict and adaptation occurs after every minibatch. The KL
epsilon placement is intentional; identical distributions need not give exactly
zero, and the result is not clamped. Current parameters were evaluated before
the optimizer step even though KL is calculated afterward.

## Checkpoint format and restoration

Version 3 archives contain `format_version`, a JSON `metadata` string and a
tensor list, saved with pickle protocol 2 through temporary-file/atomic replacement.
Tags encode literals, hexadecimal floats, tuples/lists, dictionaries (including
integer keys), ordered dictionaries with `_metadata`, and tensor indices.
NumPy RNG arrays become integer lists. Unsupported metadata objects/cycles are
not supported. The codec does not import classes or execute file-provided calls.
This remains a torch/pickle archive, not a non-pickle format.

The representation avoids unsupported Python-float `BINFLOAT` and newer pickle
opcodes in the historical restricted loader. See
[PyTorch 1.13.1 unpickler](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/_weights_only_unpickler.py)
and [serialization API](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/serialization.py).

Resume restores model, both normalizers, Adam moments/LR, AMP scaler, counters
and Python/NumPy/torch CPU/available CUDA RNG. Network and all PPO settings must
match; normalization compatibility, optimizer LR consistency and CUDA RNG device
count are checked. Simulator state and episode-meter history are not restored.
Evaluation loads only model/normalizers, permits a different environment count,
freezes normalization and performs no optimizer updates.

An enabled destination scaler with an empty saved CPU scaler gets a fresh enabled
scaler and warning; nonempty enabled state restores normally. Disabled destinations
use PyTorch's no-op scaler loading. Missing CUDA RNG/AMP history cannot be recovered
from a CPU run. Same-device CUDA and CPU-to-CUDA behavior require native validation.
[GradScaler reference](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/cuda/amp/grad_scaler.py).

Loading explicitly uses CPU mapping and `weights_only=True` where available.
Version 1 requires explicit trust; version 2 float metadata also needs it on
1.13.1, while a successful restricted version 2 load is accepted on capable
runtimes. Without the `weights_only` API, loading is rejected before deserialization
unless trust is explicit. This fallback does not establish support for a torch
version below the package requirement. `trust_legacy_checkpoint=True` permits
unrestricted pickle execution and warns; no failure triggers an automatic retry.
Use known-source files only: restricted loading is not a security sandbox.
New saves are version 3; old code cannot read the envelope. RL-Games checkpoints
are not supported. See the [user commands](getting-started.md#9-evaluate-a-trained-agent).

## Validation boundaries

Tests cover analytical losses/gradients, moment timing, GAE boundaries, data
ownership, optimizer ordering, mocked train/resume/evaluation and serialization.
The strict runner requires all tests with no skips; see
[testing status](../README.md#testing-status). No simulator or RL-Games import is
needed for these fixtures. They do not establish native physics or learning curves.

Supported scope is a feedforward policy, vector observations, scalar critic,
single-device learning and legacy scheduling. Singleton training batches with
sample-variance normalization fail explicitly; singleton evaluation is supported.
LSTM, distributed learning and original-framework checkpoint conversion are absent.
Exact upstream RNG consumption and bitwise initialization are not promised;
the reference document explains those boundaries.
