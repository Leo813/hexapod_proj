# PPO implementation

The application uses a feedforward actor-critic policy and PPO with single-device
learning. See [Getting started](getting-started.md) for installation and commands.

## Software components

`train_agent.py` delegates to `launcher.py`, which composes configuration,
initializes Isaac Gym, then imports tensor-dependent application modules.
`integration/task_factory.py` loads the installed `Anymal10` task directly from
IsaacGymEnvs. It rejects an already-imported IsaacGymEnvs namespace; use a fresh
process rather than mixing this loader with `isaacgymenvs.make()`.

| Module under `src/hexapod_rl/` | Responsibility |
| --- | --- |
| `config.py`, `integration/configuration.py` | Typed settings and installed YAML composition |
| `networks/actor_critic.py`, `distributions.py` | Actor-critic network and Gaussian policy |
| `normalization.py` | Observation and value running statistics |
| `integration/isaacgym_adapter.py`, `collector.py` | Environment interface and persistent rollout collection |
| `rollout.py`, `advantages.py` | Rollout storage, minibatches, timeout shaping, and GAE |
| `ppo.py`, `schedules.py` | PPO losses, Adam/AMP updates, and adaptive learning rate |
| `trainer.py`, `evaluation.py` | Training lifecycle and policy evaluation |
| `checkpoints.py`, `checkpoint_format.py` | Learning-state persistence and metadata encoding |

## Actor-critic network

The default network accepts 104 observations. Separate actor and critic MLPs
have hidden widths 512, 256, and 128, with ELU after each hidden layer. The actor
has an unbounded 18-dimensional mean head; the critic has a scalar value head.
Linear layers retain their PyTorch constructor weights and have zero biases.
A trainable, observation-independent `log_std` vector starts at zero, giving
unit standard deviation. There are no recurrent layers, dropout, or LayerNorm.

The policy is an unsquashed diagonal Gaussian. Log-probabilities and entropy
sum all 18 action dimensions. Raw network calls do not normalize observations,
denormalize values, clip actions, or apply joint coupling; PPO and the environment
adapter supply those transformations. Singleton evaluation is supported, while
sample-variance normalization requires at least two training samples.

Construct the model on its final device before `PPO(model, config)`. Model and
rollout tensors use float32; normalizer buffers use float64. `PPO.sample(obs)`
returns raw actions, positive joint log-probabilities, denormalized values,
means, and standard deviations. `predict_values(obs)` does not sample actions.
Values have shape `[N,1]`, actions/means/stds `[N,18]`, and log-probabilities `[N]`.

## Environment and rollout processing

The adapter requires float32 `[N,104]` observations and `[N]` rewards on the
learning device, integer/bool `[N]` dones/timeouts, and action bounds [-1,1].
`extras['time_outs']` is required. VecTask clips observations; the adapter does
not repeat that clipping or silently move results from the wrong device.

The collector resets once and retains observations and boundary flags between
rollouts. It copies raw policy samples into owned `[T,N,...]` storage before
`env.step`, then immediately copies the resulting observations, rewards, and
flags. A separate action tensor is clipped to [-1,1] before the task mutates it
for joint coupling. Likelihoods always use the stored raw actions.

The task couples six tibia actions to negative femur actions and accumulates
position targets. All 18 policy components remain part of the distribution.
Reset handling is deferred by the task; the collector adds no eager physics
step or immediate autoreset. Its initial reset returns the allocated observation
buffer. Actor-state views assume the robot followed by three boxes. Preserve
these interface conventions when maintaining integration code.

Episode statistics use raw rewards and a capped batch-weighted mean of completed
episodes (capacity 100). Each completed update advances one epoch and `N*T`
frames. Cleanup closes logging and simulator/viewer handles; partial task
construction failures may leave native handles until process exit. Offline
fixtures must not import `hexapod.py`, which imports the simulator and creates
a module-level TensorBoard writer.

Timeout shaping and generalized advantage estimation use reward-unit values:

```text
r_shaped[t] = reward_scale*r[t] + gamma*V[t]*timeout[t]  # if bootstrap enabled
q[t] = 1 - next_done[t]
delta[t] = r_shaped[t] + gamma*q[t]*V_next - V[t]
A[t] = delta[t] + gamma*tau*q[t]*A_next
R[t] = A[t] + V[t]
```

Timeout shaping uses the pre-action value; timeouts remain done boundaries.
Stored `dones[t]` describe the pre-action state, so non-final masks use
`dones[t+1]`. The final continuation uses `last_values` and `A_next=0`.
Defaults are `gamma=0.99`, `tau=0.95`, and a horizon of 48 steps.

Rollouts flatten in environment-major order into contiguous minibatches without
shuffling. Minibatch size must divide the batch exactly. The default
`256*48=12288` samples form one minibatch per pass, repeated for five mini-epochs.
Old log-probabilities stay fixed; stored KL means/stds update after each minibatch
with its evaluated distribution.

## Normalization and target preparation

Running mean, variance, and count start at 0, 1, and 1 in float64. Batch moments
use the input dtype and sample variance (denominator `n-1`). For batch mean `b`,
variance `s2`, count `n`, current moments `m,v,c`, and `d=b-m`, `C=c+n`:

```text
m_new = m + d*n/C
v_new = (c*v + n*s2 + d*d*c*n/C) / C
c_new = C
normalize(x)   = clamp((x-m)/sqrt(v+1e-5), -5, 5)
denormalize(z) = clamp(z, -5, 5)*sqrt(v+1e-5) + m
```

Transformations use float32 views of the statistics. Move normalizers by device
without casting their buffers to the model dtype. Moments are frozen during
rollout collection. Advantages and returns are computed before value normalization;
advantages are standardized using `(A-mean(A))/(std(A)+1e-8)` with sample std.

Batch preparation updates value statistics on old values and transforms them,
then updates again on returns and transforms them. These transformations use
different moments. Observation statistics update before each forward pass only
in the first optimization mini-epoch, then freeze. Call `prepare_batch(rollout)`
once per rollout; `optimize(batch)` consumes its detached output, and
`update(rollout)` combines both operations.

## Losses and optimization

For `ratio=exp(logp_new-logp_old)` and default clip width `e=0.2`:

```text
L_policy = mean(max(-A*ratio, -A*clamp(ratio, 1-e, 1+e)))
V_clip = V_old + clamp(V_new-V_old, -e, e)
L_critic = mean(max((V_new-R)^2, (V_clip-R)^2))
H = mean(sum_actions(log_std + 0.5*log(2*pi*euler_constant)))
L_bounds = mean(sum_actions(max(mu-1.1,0)^2 + min(mu+1.1,0)^2))
L_total = L_policy + 0.5*critic_coef*L_critic
          - entropy_coef*H + bounds_loss_coef*L_bounds
```

Without value clipping, the critic uses unclipped squared error. Default critic,
entropy, and bounds coefficients are 2, 0, and 0.001. Bounds penalize policy means.
There is no tanh squashing, std clamp, KL loss, or KL early stopping.

Adam uses default LR `3e-4`, epsilon `1e-8`, betas `(0.9,0.999)`, zero weight decay,
and no AMSGrad. Each minibatch performs forward/loss under enabled CUDA autocast,
clears gradients, scales/backpropagates the loss, unscales gradients, clips their
norm to 1, and steps/updates the scaler. CPU execution disables CUDA AMP.

The adaptive schedule updates after each minibatch. For current pre-update
forward parameters `(m0,s0)` and stored `(m1,s1)`:

```text
KL = mean_samples(sum_actions(
    log(s1/s0 + 1e-5) + (s0^2 + (m1-m0)^2)/(2*(s1^2 + 1e-5)) - 0.5))
KL > 2*0.008   -> LR = max(LR/1.5, 1e-6)
KL < 0.5*0.008 -> LR = min(LR*1.5, 1e-2)
otherwise     -> unchanged
```

Comparisons are strict. The current distribution was evaluated before the
optimizer step, although KL is calculated afterward. Epsilon placement means
identical distributions need not yield exactly zero; KL is not clamped.

## Checkpoint format and restoration

Version 3 archives contain `format_version`, a JSON `metadata` string, and a
tensor list. Saves use pickle protocol 2 and temporary-file/atomic replacement.
Metadata tags encode literals, hexadecimal floats, tuples/lists, dictionaries
(including integer keys), ordered dictionaries with `_metadata`, and tensor
indices. NumPy RNG arrays become integer lists. Unsupported objects and cycles
are rejected; the codec does not import classes or execute metadata-provided
calls. The file remains a torch/pickle archive. This encoding supports restricted
loading on PyTorch 1.13.1.

Resume restores the model, both normalizers, Adam moments/LR, AMP scaler,
counters, and Python/NumPy/torch CPU/available CUDA RNG state. Network and all
PPO settings must match. Loading checks normalization compatibility, optimizer
LR consistency, and CUDA RNG device count. Simulator state and episode-meter
history are not restored. Task/physics settings must be retained separately.

An enabled destination scaler with an empty saved CPU scaler starts fresh with
a warning; nonempty enabled state restores normally. Disabled destinations use
PyTorch's no-op scaler loading. Missing CUDA RNG/AMP history cannot be recovered
from a CPU run.

Loading uses CPU mapping and `weights_only=True` where available. Version 1
requires explicit trust. Version 2 float metadata also requires trust on
PyTorch 1.13.1; a successful restricted version 2 load is accepted on capable
runtimes. If the `weights_only` API is absent, loading is rejected before
deserialization unless trust is explicit; this does not extend supported PyTorch
versions below the package requirement.

`trust_legacy_checkpoint=True` permits unrestricted pickle execution and emits
a warning. Load known-source files only: restricted loading is not a security
sandbox, and failures never trigger an automatic unrestricted retry. New saves
use version 3, which older application code cannot read. Other applications'
checkpoint formats are unsupported. See [saving and resuming](getting-started.md#8-outputs-and-resuming-training)
for conditional saves, output paths, and required overrides.

## Evaluation

Evaluation restores the model and normalizers, permits a different environment
count, freezes normalization, and selects deterministic policy means without
sampling or optimizer updates. Actions still pass through environment clipping
and coupling. Keep the trained architecture and normalization settings compatible;
see [evaluation commands](getting-started.md#9-evaluate-a-trained-agent).
