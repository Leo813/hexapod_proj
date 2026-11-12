# Hexapod executable baseline

Stage 1, 2026-11-12. The original environment, YAML files, URDF, and meshes are
unchanged. This contract records behavior, including suspected defects; it does
not authorize fixes. The research paper is background, not an implementation
override. Reproducing the paper's complete experimental curriculum is a different
objective from replacing the checked-in trainer backend.

Evidence labels used here and in [the reference specification](reference-implementation.md):

- **Local:** established by checked-in source/configuration or parsed asset XML.
- **Reference:** established by the selected historical upstream source. This is
  a reconstruction choice, not proof of the original installed software.
- **Unverified:** requires original installation records, runtime inspection,
  numerical validation, or a simulator-equipped machine.

## Scope and dependencies

**Local:** `README.md` describes installing this task in an external IsaacGymEnvs
tree and registering it in that tree's task mapping. There is no original local
trainer, framework checkout, dependency lockfile, or reference checkpoint.

`hexapod.py` defines `Anymal10(VecTask)`. Isaac Gym owns physics, asset import,
actors, sensors, terrain, and simulator tensors. IsaacGymEnvs supplies `VecTask`
and training integration. RL-Games supplies the original model, PPO, buffers,
normalizers, optimizer, schedule, and checkpoints. PyTorch implements local
tensor calculations and the new independent network.

The original module imports NumPy, torch, Isaac Gym, IsaacGymEnvs, TensorBoard,
and matplotlib. A module-level `SummaryWriter()` can create output at import.
Offline tests must not import that module. ROS/catkin metadata in `hexapod2/`
describes the asset package, not the learning software's dependency versions.

**Historical target (chosen / user supplied):** approximately 2023, Ubuntu 20.04,
Isaac Gym Preview 4, the selected IsaacGymEnvs revision, and RL-Games v1.6.0.
The original Python, PyTorch, CUDA, driver, and exact framework revisions remain
unverified. **Independent package:** Python >=3.8, torch >=1.13 as a declared
development compatibility floor, standard-library unittest, CPU or native torch
device transfer. This is not an original-environment version pin. No Ubuntu
requirement applies to independent network tests. See [test report](stage1-tests.md).

## Environment interface

**Local:** constructor arguments are `cfg, rl_device, sim_device,
graphics_device_id, headless, virtual_screen_capture, force_render`. The task
writes `numObservations=104`, `numActions=18` before calling `VecTask`.
It implements `create_sim`, `pre_physics_step`, `post_physics_step`,
`compute_observations`, `compute_reward`, and `reset_idx`.

**Reference:** the selected `VecTask` exposes an 18-dimensional Box action space
with limits [-1, 1]. `step(actions)` returns `(observation_dict, rewards, resets,
extras)`: `observation_dict['obs']` is [N,104], reward and reset buffers are [N],
and `extras['time_outs']` is [N]. Observations and results are moved to
`rl_device`. Reset buffers are allocated as long tensors, timeout flags are
computed by a Boolean expression, and observations/rewards are floating tensors.
No privileged state space is configured locally. The original `RLGPUEnv` is a
delegating RL-Games adapter, not needed by the Stage 1 package.

`VecTask.reset()` returns the current clipped observation buffer without
computing new observations. The local constructor resets state but never calls
`compute_observations`; combined with the selected base's zero allocation, the
first returned observations are zeros. This is a source-derived reconstruction,
not simulator verification. Do not add an eager observation/physics step casually.

**Reference:** `IsaacGymEnvs.make()` imports `rlgames_utils`, which imports
RL-Games. Future integration must avoid that convenience factory/adapter to
remove the backend dependency while retaining the task and `VecTask`.

## Observations

**Local:** `compute_anymal_observations` (`hexapod.py:767`) concatenates:

| Slice | Width | Data / transformation |
| --- | ---: | --- |
| 0:3 | 3 | Body linear velocity: inverse quaternion rotation, x2 |
| 3:6 | 3 | Body angular velocity: inverse quaternion rotation, x0.25 |
| 6:9 | 3 | Gravity: forward `quat_rotate`, not inverse rotation |
| 9:11 | 2 | Commands 1:3 multiplied by [8,1] |
| 11:14 | 3 | First three force-sensor components |
| 14:32 | 18 | DOF position minus default position, x1 |
| 32:50 | 18 | DOF velocity x0.5 |
| 50:68 | 18 | Unscaled measured DOF torques |
| 68:86 | 18 | Contact-force xyz vectors at body indices 3,6,9,12,15,18 |
| 86:104 | 18 | Current stored coupled action |

Object states, sensor torque, and terrain-height measurements are absent from
the active observation. The action block describes the preceding control action
when this observation is used to choose the next action.

**Local + Reference:** preprocessing order is task feature scaling, then
`VecTask` clipping to [-5,5], then the RL-Games model's running input
normalization/clipping. Thus the fixed forward command's feature 8 becomes 5
before running normalization. Stage 1 implements neither clipping nor running
statistics; callers will supply prepared observations once Stage 2 exists.

## Actions and controls

**Local:** 18 policy outputs are retained. In `pre_physics_step` (`hexapod.py:414`),
indices 2,5,8,11,14,17 are overwritten with the negative of indices
1,4,7,10,13,16, respectively. These are the intended tibia/femur pairs. There
are only 12 independent effects on control, but reducing the policy density to
12 dimensions changes the learning problem and is not part of this replacement.

The input action tensor is mutated before cloning. Position targets are persistent:

1. Couple tibia actions to femur actions.
2. Add `dt * 20 * actions` to the target accumulator.
3. Clamp accumulator to imported joint limits.
4. Overwrite tibia accumulator entries with negative femur entries.
5. Send `actionScale * accumulator` as simulator position targets.

At baseline, dt=0.01, actionScale=0.8, stiffness=200, damping=1.2. Target increments
are therefore 0.2 times the coupled action. Scaling occurs after clamping and
coupling; it is not a stock Anymal position-offset action. Reset accumulator is
[0,1,-1] repeated six times. The accumulator itself is not an observation feature.
Position-drive mode is set in code even though the YAML asset field says effort.

**Reference:** RL-Games first stores raw Gaussian actions and their negative log
densities in owned rollout tensors, then clamps actions to [-1,1] and rescales
to the environment limits. For these limits, rescaling is an identity. `VecTask`
clamps again before calling the task; both clamps are out-of-place. Consequently
the task's input mutation does not alter original stored policy actions in this
selected path. The future collector must preserve this distinction. Do not
evaluate likelihoods on clipped or coupled actions.

## Reward

**Local:** `compute_anymal_reward` (`hexapod.py:638`) uses body-frame linear v
and angular w, commands c, measured torques T, coupled actions a, and histories:

```text
forward = 2.5 * tanh(8 * c[1] * v[y])
turn = 2.0 * tanh(2 * c[2] * w[z])
lateral_cost = 0.5 * (tanh(25 * abs(v[x]) - 3) + 1)
torque_term = -0.00025 * dt * sum(T**2)
joint_term = -0.002 * sum((previous_joint_velocity - joint_velocity)**2)
action_term = -0.02 * sum((previous_action - a)**2)
yaw_term = 0.25 * c[3] * tanh(abs(previous_angular_velocity[z] + w[z])) * forward
reward = max(0, forward + turn + torque_term + joint_term + action_term
                + yaw_term - lateral_cost)
```

The separate linear punishment is multiplied by zero. Only the torque reward
scale from the YAML reward dictionary affects the final sum; its effective
coefficient is -2.5e-6. Other active terms are not multiplied by dt. The joint
velocity difference is not divided by dt. Both femur and overwritten tibia
components contribute to the action penalty. Reward is detached and floored at
zero. Distance and explicit object-position rewards are inactive.

`self.base_ang_vel` is computed once during initialization. Post-step history
updates copy from this stale tensor, not the newly computed local angular
velocity. Preserve this behavior until a separate change is approved.

## Resets, termination, and randomness

**Local:** commands are sampled and discretized, then overridden to [0,1,0,-1].
This means forward locomotion with no commanded yaw. The discarded command
draws still consume RNG state. DOF reset positions are defaults times independent
Uniform(0.5,1.5) samples; velocities are Uniform(-0.1,0.1). There is no explicit
clamp or mimic coupling of randomized reset positions. Robot and three object
root states are restored from the initial actor-state tensor.

Selected histories are cleared, but `self.actions` is not. The target accumulator
is reset, progress is zeroed, and reset flags are set to one. The separately
prepared `initial_root_states` tensor is not used for these actor resets.

Post-step order is: increment progress; reset environments flagged by the
previous step; refresh/construct observations; calculate reward/new reset flags;
update histories. This defers resets to a subsequent post-physics step, after
another action and simulation step. Initialization also leaves reset flags set.
Do not replace this with standard immediate autoreset semantics.

Termination is the OR of:

- Base contact-force norm >10.
- Sum of absolute DOF velocities at indices 1,4,7,10,13,16 <0.01 (femurs under
  the intended ordering, despite the coxa comment).
- Progress >=4999, since max length is round(50 / 0.01)=5000.

There is no active object-fall or knee-contact termination and no special
terminal penalty. **Reference:** `VecTask.step` generates timeout flags after
post-step from `(progress >= max_length-1) & (reset != 0)`. It does not distinguish
a simultaneous physical failure and timeout. RL-Games adds `gamma * V(s_t)`
to timeout rewards using the pre-step value, then applies done masking in GAE;
it does not obtain a terminal-observation value here.

Framework domain randomization is disabled. Terrain randomness and joint-state
reset randomness still apply. Randomization actor name `anymal10` differs from
created actor name `anymal`; enabling it is separate work.

## Assets and simulation

**Local:** `hexapod2/urdf/test_spider6.urdf` has 21 active links, 18 revolute
joints, and two fixed joints. All mimic declarations and six mimic links are
commented out; their six meshes and ROS joint names remain. There are 25 STL
files, of which 19 distinct files are referenced by active XML. Mesh references
use `package://test_spider6/`. Every active reference exists in this checkout.

The task loads `../../assets/urdf/hexapod2/urdf/test_spider6.urdf` relative to its
installed task location, not this repository's `hexapod2/` directly. Fixed joints
are collapsed. Actual imported DOF/body ordering remains unverified; hardcoded
contact indices and force sensor body index 5 must be checked in simulation.
The sensor attachment comment says tray, but a numeric index alone does not
establish that. The ROS joint-name YAML includes inactive mimic names.

There are four actors per environment: robot, then three boxes. Root tensor
views and indexed resets depend on this ordering. Box size is 0.17x0.17x0.1 m,
density 30, friction 1.4. Robot initial position is [6,6,0.01]; boxes start at
x=5.95, y=6.1, z=0.31/0.41/0.51. Objects influence physics even without explicit
balancing rewards. Environment spacing is overwritten to 1 in code.

Simulation dt=0.01, substeps=2, z-up, gravity=-9.81, controlFrequencyInv=1.
Terrain is a 100x100 m random triangle mesh with heights [-0.01,0.01],
horizontal scale 0.1 and vertical scale 0.005. Plane creation is disabled;
plane-friction YAML values do not set triangle-mesh friction. Dormant height
sampling has different scales and indexing concerns but is not observed.

## Network and trainer configuration

**Local:** feedforward actor and critic have independent 104->512->256->128
MLPs, ELU activations, unbounded 18-dimensional mean and scalar value heads.
The config requests `continuous_a2c_logstd`, fixed sigma, zero sigma initializer,
default weight/mean initialization, no regularizer, and no D2RL.
The verified meaning of these settings is documented in the reference contract.

| Setting | Value |
| --- | --- |
| Environments / horizon | 256 default / 48 |
| Minibatch / mini-epochs | 12288 / 5 |
| gamma / tau | 0.99 / 0.95 |
| PPO clip / value clipping | 0.2 / enabled |
| LR / schedule / KL threshold | 3e-4 / adaptive / 0.008 |
| Critic / entropy / bounds coefficients | 2 / 0 / 0.001 |
| Gradient norm / truncation | 1 / enabled |
| Input, value, advantage normalization | all enabled |
| Value bootstrap / mixed precision | enabled / enabled |
| Reward shaper scale | 1 |
| Max epochs / save frequency / best-after | 600 default / 50 / 500 |
| Winning score | 10000 |

Default rollout is exactly one minibatch, with five updates. Relative YAML
interpolations require an enclosing configuration and resolvers `resolve_default`,
`eq`, `contains`, `if`; these are registered in the selected IsaacGymEnvs
`__init__.py`. Seed, devices, experiment, checkpoint, and multi-GPU settings
are inherited. Stage 1's typed network config does not resolve these files or
pretend to implement training configuration.

LSTM is deferred: one 256-unit layer before the MLP, concat_input=True,
seq_len=4, horizon=24, minibatch=1024, entropy=0.001, bounds coefficient=0,
clip_actions=False, max epochs=1200, best-after=100. Environment clipping still
applies even when the trainer's action clipping is disabled.

## Paper discrepancies

**Local comparison:** `Paper_yuhaotsai.pdf`, especially pages 2-5, describes
the research objectives and staged experiments. It does not supersede code.

| Topic | Paper | Executable baseline |
| --- | --- | --- |
| Discount / GAE | Table I: 0.95 / 0.99 | gamma=0.99 / tau=0.95 |
| Curriculum | Commands, terrain, then increasing objects | No curriculum manager; three objects |
| Commands | Bidirectional/turning/randomized | Forced forward after random draws |
| Terrain maximum | 0.1 m described | Generator range +/-0.01 m |
| Randomization | Varied physics and commands discussed | Framework randomization off; joint randomness active |
| Joint velocity / femur upper limit | 3.8*pi / 0.9*pi | URDF 5.969026 / 1.3963 |
| Angular history | Previous angular velocity | Stale initialization-derived history |
| Plate sensor | Described as plate-mounted | Index 5; actual mapping unverified |

No paper-based corrective changes are included. Exact simulator compatibility,
object balance performance, and equivalent learning curves remain unverified.
