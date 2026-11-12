# Historical reference and Stage 1 mapping

## Evidence and revision selection

**Local** means checked-in project evidence; **Reference** means inspected
historical source; **Unverified** means not established by that evidence or not
executed. See [baseline contract](baseline-contract.md) for local behavior and
[validation report](stage1-tests.md) for what actually ran.

| Component | Selected revision | Status |
| --- | --- | --- |
| RL-Games | v1.6.0, `fa1c13cc10158ab42e829a0971a9dc9a4544b3b9` | Tag resolved via GitHub Git ref API; source read at the commit |
| IsaacGymEnvs | `aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3` | 2023-10-18T16:22:21Z, merge release 1.5.1 from dev to main |
| Isaac Gym | Preview 4 | User-selected simulator target; selected IsaacGymEnvs README explicitly requests it |
| PyTorch initializer source | v1.13.1 tag | Supporting historical definition of Linear defaults, not an original installed-version claim |

RL-Games v1.6.0 is the requested reconstruction baseline, not a verified record
of the original installation. IsaacGymEnvs was selected as the latest default-
branch commit returned by the public commits API with
`until=2023-12-31T23:59:59Z&per_page=1`. It lies in the requested period, its
setup metadata identifies 1.5.1 and requires `rl-games>=1.6.0`, and its README
targets Preview 4. This is a suitable historical pairing, not proof that this
exact task ran against it. The `>=` requirement alone does not pin RL-Games.

Revision evidence: [RL-Games tag API](https://api.github.com/repos/Denys88/rl_games/git/ref/tags/v1.6.0),
[RL-Games commit](https://github.com/Denys88/rl_games/commit/fa1c13cc10158ab42e829a0971a9dc9a4544b3b9),
[IsaacGymEnvs commit](https://github.com/isaac-sim/IsaacGymEnvs/commit/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3),
[IsaacGymEnvs setup](https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/setup.py),
[Preview 4 requirement](https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/README.md).

Public source files were downloaded only into a temporary inspection directory.
None were installed or vendored. The new implementation independently expresses
the network structure and equations with native torch. RL-Games' setup declares
MIT licensing; its authors are Denys Makoviichuk and Viktor Makoviichuk.

## Verified source responsibilities

All RL-Games links below are pinned to the selected commit, not the current branch.

| Source | Classes/functions inspected and responsibility |
| --- | --- |
| [algos_torch/model_builder.py](https://github.com/Denys88/rl_games/blob/fa1c13cc10158ab42e829a0971a9dc9a4544b3b9/rl_games/algos_torch/model_builder.py) | NetworkBuilder/ModelBuilder registry maps actor_critic to A2CBuilder and continuous_a2c_logstd to ModelA2CContinuousLogStd |
| [algos_torch/network_builder.py](https://github.com/Denys88/rl_games/blob/fa1c13cc10158ab42e829a0971a9dc9a4544b3b9/rl_games/algos_torch/network_builder.py) | NetworkBuilder.BaseNetwork factories, _build_sequential_mlp; A2CBuilder.Network.__init__, load, forward |
| [algos_torch/models.py](https://github.com/Denys88/rl_games/blob/fa1c13cc10158ab42e829a0971a9dc9a4544b3b9/rl_games/algos_torch/models.py) | BaseModelNetwork normalization; ModelA2CContinuousLogStd.Network.forward and neglogp |
| [algos_torch/a2c_continuous.py](https://github.com/Denys88/rl_games/blob/fa1c13cc10158ab42e829a0971a9dc9a4544b3b9/rl_games/algos_torch/a2c_continuous.py) | A2CAgent.__init__, calc_gradients, bound_loss; Adam, losses, KL, update result |
| [common/a2c_common.py](https://github.com/Denys88/rl_games/blob/fa1c13cc10158ab42e829a0971a9dc9a4544b3b9/rl_games/common/a2c_common.py) | A2CBase defaults/play_steps/discount_values; ContinuousA2CBase preprocess_actions/train_epoch/prepare_dataset |
| [algos_torch/players.py](https://github.com/Denys88/rl_games/blob/fa1c13cc10158ab42e829a0971a9dc9a4544b3b9/rl_games/algos_torch/players.py) | PpoPlayerContinuous.get_action: deterministic mean selection, then optional clipping/rescaling |
| [common/common_losses.py](https://github.com/Denys88/rl_games/blob/fa1c13cc10158ab42e829a0971a9dc9a4544b3b9/rl_games/common/common_losses.py) | actor_loss and critic_loss equations |
| [algos_torch/torch_ext.py](https://github.com/Denys88/rl_games/blob/fa1c13cc10158ab42e829a0971a9dc9a4544b3b9/rl_games/algos_torch/torch_ext.py) | policy_kl, reductions, supporting utilities |
| [algos_torch/running_mean_std.py](https://github.com/Denys88/rl_games/blob/fa1c13cc10158ab42e829a0971a9dc9a4544b3b9/rl_games/algos_torch/running_mean_std.py) | RunningMeanStd moments, normalization and inverse transformation |
| [common/schedulers.py](https://github.com/Denys88/rl_games/blob/fa1c13cc10158ab42e829a0971a9dc9a4544b3b9/rl_games/common/schedulers.py) | AdaptiveScheduler.update |
| [common/datasets.py](https://github.com/Denys88/rl_games/blob/fa1c13cc10158ab42e829a0971a9dc9a4544b3b9/rl_games/common/datasets.py) | PPODataset._get_item and update_mu_sigma: contiguous batches and mutable KL reference parameters |
| [common/experience.py](https://github.com/Denys88/rl_games/blob/fa1c13cc10158ab42e829a0971a9dc9a4544b3b9/rl_games/common/experience.py) | ExperienceBuffer.update_data copies samples into allocated storage |

IsaacGymEnvs references:

- [tasks/base/vec_task.py](https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/isaacgymenvs/tasks/base/vec_task.py):
  Env spaces and clipping settings; VecTask allocation, step, reset, reset_done.
- [utils/rlgames_utils.py](https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/isaacgymenvs/utils/rlgames_utils.py):
  RLGPUEnv delegation and environment information.
- [train.py](https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/isaacgymenvs/train.py):
  rlgpu/RLGPU registration and RL-Games Runner entrypoint.
- [__init__.py](https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/isaacgymenvs/__init__.py):
  configuration resolver definitions; make() depends on rlgames_utils.

## Network and initialization

**Reference:** A2CBuilder.Network constructs separate actor and critic hidden
MLPs when `separate=True`. Each layer is Linear followed by the requested
activation, including the last hidden layer. The actor output is a separate
mean head; the critic output is a separate value head. The default value size
is one, value activation is identity, and the local mean activation is identity.
No LayerNorm, dropout, recurrent layers, or D2RL are requested by the baseline.

The factory entry for `default` is an identity operation on weights. Consequently:

1. Linear constructors initialize weights normally.
2. The general default initializer does not alter those weights.
3. Every linear bias is explicitly zeroed, including both output heads.
4. `mu_init: default` likewise leaves the mean head's weight unchanged.
5. `sigma_init: const_initializer, val: 0` sets the sigma parameter to zero.

This is not orthogonal initialization and not a small special policy-head gain.
[PyTorch v1.13.1 Linear.reset_parameters](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/nn/modules/linear.py#L103)
uses Kaiming uniform with a=sqrt(5), equivalent to weight bounds
[-1/sqrt(fan_in),1/sqrt(fan_in)]. Its initial biases are random before RL-Games
zeros them. The independent implementation retains constructor weights and
zeros biases, preserving those draws rather than creating bias-free layers.

**Reference:** with `fixed_sigma=True`, the builder creates an action-sized
`nn.Parameter` with requires_grad=True. The name "sigma" is misleading for this
model: it is consumed as log standard deviation. It is fixed with respect to
observations, not frozen during optimization. A2CAgent passes all model
parameters to Adam, so it is trained. Initial log_std=0 gives std=1 and variance=1.
The builder broadcasts it using a zero multiple of the means plus the parameter.

**Implementation:** `ActorCriticConfig` holds dimensions, hidden widths,
activation, and initial log_std; its defaults match local YAML. ActorCritic
constructs actor hidden layers, critic hidden layers, value head, then mean head;
the global log_std is a trainable vector. Configurable alternative widths or
activations are experiments, not claims of baseline parity. YAML/Hydra loading
is not implemented in Stage 1.

Same-seed bitwise initialization against the complete upstream builder is not
claimed: upstream's shape probing consumes an additional random observation
tensor before layer construction. We do not reproduce this irrelevant shape
probe. Initialization distributions and biases are matched; deterministic
cross-implementation comparisons should assign identical parameter tensors.
The test suite also checks exact preservation of native constructor weights in
the custom construction order. CPU/GPU, torch versions, and floating arithmetic
can cause further numerical differences.

## Distribution, likelihoods, and output conventions

**Reference:** ModelA2CContinuousLogStd exponentiates log_std, constructs an
elementwise Normal with validate_args=False, and uses sample(), not rsample(),
for rollout actions. It neither squashes nor clips actions or log_std. Its
negative joint log density is:

```text
negative_log_prob(a) = 0.5 * sum(((a - mean) / exp(log_std))**2)
                    + 0.5 * action_dim * log(2*pi)
                    + sum(log_std)
entropy = sum(log_std + 0.5 * log(2*pi*e))
```

There is no standard-deviation epsilon or tanh Jacobian. Entropy uses
Normal.entropy() summed over action components. Mean/std are [N,18], values
are [N,1]. Upstream squeezes its negative-log-probability result and can produce
a scalar for N=1. Training returns prev_neglogp/values/entropy/mus/sigmas;
rollout returns neglogpacs/values/actions/mus/sigmas plus recurrent state metadata.

The independent API deliberately uses positive log-probabilities and keeps a
stable [N] batch shape, including N=1. `Independent(Normal(...),1)` expresses one
18-dimensional event; the explicit gaussian_log_prob helper follows the same
negative-log-density equation with its sign inverted. No wrappers or upstream
runtime imports are used.

| Custom interface | Shape / reference mapping |
| --- | --- |
| actor(obs) | [N,18] means, raw and unbounded |
| critic(obs) | [N,1] raw network values |
| forward(obs) | NamedTuple of means, broadcast log_std, values |
| distribution(obs) | Independent Gaussian; batch [N], event [18] |
| sample_actions(obs) | actions [N,18], log_prob [N], values [N,1], under no_grad |
| log_prob(obs, actions) | -reference neglogp, summed over all 18 dimensions |
| entropy(obs) | [N], summed entropy |
| evaluate_actions(obs, actions) | Differentiable log_prob, entropy, values |
| deterministic_actions(obs) | Mean only, no clipping and no RNG consumption |

The upstream player computes a sampled model result even when it later selects
the mean. The custom deterministic helper returns the same raw mean without
that unused RNG draw; reproducing player-side RNG consumption is outside this
network API. There is no promise of a drop-in RL-Games result dictionary or
checkpoint key layout.

**Reference:** clipping belongs to ContinuousA2CBase.preprocess_actions (default
clip_actions=True) and the environment. Raw actions are stored before clipping.
The independent network intentionally implements neither clipping/rescaling
nor six-joint coupling. All 18 terms remain in likelihood and entropy.

**Stage boundary:** the upstream BaseModelNetwork wraps the network with
observation normalization and rollout value denormalization. Stage 1 supplies
only the neural/distribution core. Therefore direct raw observations and values
do not reproduce a trained upstream model with its running statistics. Later
code must normalize observations and interpret value units explicitly.

## Trainer defaults verified for later stages (not implemented)

The local PPO YAML is authoritative for explicit settings. Source inspection
establishes the following inherited behavior for the selected single-device,
feedforward path:

- Adam uses eps=1e-8 and default weight_decay=0. The betas are not supplied,
  so they are inherited from the installed torch optimizer, not pinned by the
  project. Preserve the original optimizer settings once that torch version is known.
- The PPO ratio uses exp(old_neglogp-new_neglogp). The actor objective is the
  maximum of the two negative clipped/unclipped surrogates.
- Value loss is the maximum of unclipped and clipped squared errors using the
  same e_clip. Total loss is actor + 0.5*critic_coef*critic - entropy_coef*entropy
  + bounds_loss_coef*bounds. Non-recurrent reductions are means over samples.
- Default bound_loss_type is bound. It penalizes squared mean excess beyond
  +/-1.1, summed over action dimensions, rather than penalizing sampled actions.
- Gradient clipping follows AMP unscaling, then scaler.step and scaler.update.
- Raw actions, negative log densities, values, means and standard deviations
  are copied into the rollout buffer before env_step. PPO old log densities
  remain fixed while PPODataset.update_mu_sigma replaces the KL reference mean
  and standard deviation for each minibatch with the latest evaluated values.
- GAE uses next-step done masks: the stored dones belong to the current
  observation, so non-final recursion uses dones[t+1]. Timeout reward shaping
  adds gamma times the PRE-STEP value before GAE, not a terminal-observation value.
- Time/environment axes are swapped before flattening. PPODataset takes
  contiguous slices; this feedforward path does not shuffle each mini-epoch.
  Default 256*48=12288 gives one minibatch per pass, five passes.
- Advantages are computed before value normalization, then standardized across
  the flattened rollout with tensor.std() and denominator epsilon 1e-8.
  normalize_rms_advantage defaults False. Preserve the torch sample-variance
  convention, rather than silently using population variance.
- RunningMeanStd has float64 mean/variance/count buffers initialized to 0/1/1,
  epsilon=1e-5, and uses input.var(axis) with torch's default correction. Outputs
  use float32 statistics and clamp normalized values to [-5,5]. Denormalization
  first clamps its input to [-5,5]. This is not a generic Welford implementation
  with a tiny initial count or population batch variance.
- Rollouts run with the model in eval mode, so observation stats are frozen.
  Training updates observation stats during the first mini-epoch only, then
  freezes that normalizer for remaining mini-epochs.
- prepare_dataset puts the value normalizer in train mode, calls it on old
  values and then returns, updating it twice, then switches it to eval mode.
  The two transformations therefore do not use identical statistics.
- schedule_type defaults legacy, updating LR per minibatch. AdaptiveScheduler
  divides LR by 1.5 for KL>2*threshold and multiplies by 1.5 for KL<0.5*threshold;
  bounds are 1e-6 and 1e-2, with unchanged entropy coefficient.
- policy_kl is called with current distribution as p0 and stored distribution
  as p1. Its formula is sum(log(s1/s0+1e-5) +
  (s0^2+(m1-m0)^2)/(2*(s1^2+1e-5)) - 0.5), averaged over samples. It has deliberate
  epsilon placements and need not be exactly zero at identical inputs. The
  current tensors were evaluated before the optimizer step even though KL is
  calculated afterward. Do not substitute a generic old-to-new analytic KL or
  introduce KL early stopping.

These are specifications for Stage 2 review, not working training features.

## Remaining uncertainty and validation boundaries

- The original installed versions, resolved launch config/seed, checkpoints,
  curriculum history, and trained normalizer statistics are unavailable.
- Isaac Gym Preview 4 binaries, imported body/DOF order, sensor body mapping,
  package mesh resolution, physics, and device synchronization are untested.
- No RL-Games model was instantiated for differential execution; evidence is
  source inspection and independent mathematical fixtures, not an executed
  upstream equivalence test.
- Torch numerical tests could not run in the current environment because torch
  is absent. Static checks are not evidence of gradient or performance parity.
- No normalization, optimizer, scheduler, rollout, GAE, checkpoint manager,
  LSTM, multi-GPU trainer, or simulator adapter exists in Stage 1.
- No claim of identical learning curves, exact RNG stream parity with upstream,
  complete simulator compatibility, or original checkpoint loading is made.

Stage 2 should first implement tests for normalization timing/units and timeout
semantics, then owned rollout storage and GAE, then the verified PPO losses,
Adam/AMP gradient handling, and legacy adaptive schedule. Use deterministic
fixtures and mocked vector environments before attempting live simulation.
Proceed only after explicit approval.
