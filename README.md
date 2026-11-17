# Hexapod locomotion with custom PyTorch PPO

The original hexapod task was adapted from NVIDIA IsaacGymEnvs' ANYmal example.
The custom trainer replaces the RL-Games learning backend while retaining that
task, Isaac Gym physics, and IsaacGymEnvs `VecTask`.

Use the [Stage 4A task preparation guide](docs/stage4-installation.md) to inspect
your environment, preview unchanged task/configuration/asset copies, explicitly
apply them, and verify their hashes. This utility does not install Isaac Gym,
PyTorch or CUDA; native GPU validation remains Stage 4B work.

From this repository, after manually installing Isaac Gym Preview 4 and the
historical IsaacGymEnvs checkout and placing the task/configuration/assets:

```bash
python train.py task=hexapod
python train.py task=hexapod headless=True num_envs=64 max_iterations=100
```

Use `isaacgymenvs_path=/absolute/path/to/IsaacGymEnvs` when automatic discovery
is insufficient. Run `python train.py --help` for supported overrides. Our
launcher loads `Anymal10` directly; **no task-registry edit is required**. The
original registry-based instructions apply only to the historical RL-Games
launcher, which this command does not use.

Training, checkpoint/resume, deterministic evaluation, and production collector
tests are implemented. **Actual Isaac Gym/GPU training has not been validated.**
Local Stage 3 numerical and OmegaConf tests remain dependency-blocked; the CI
workflow is configured to run them without accepting skips. Do not interpret
mock tests as proof of simulator compatibility or equivalent learning curves.

See [Stage 3 installation and integration](docs/stage3-integration.md) for exact
manual file placement, commands, validation status, and a simulator smoke test.
The [baseline contract](docs/baseline-contract.md),
[historical references](docs/reference-implementation.md), and
[Stage 2 equations](docs/stage2-implementation.md) document the preserved behavior.

Evaluate or resume a **custom** checkpoint:

```bash
python train.py task=hexapod test=True checkpoint=/path/run/checkpoints/last.pt headless=True
python train.py task=hexapod checkpoint=/path/run/checkpoints/last.pt max_iterations=700
```

Resume restores learning state and starts fresh simulator episodes. Historical
RL-Games checkpoints are not supported. If training used overrides such as
`num_envs`, repeat them when resuming; PPO settings must match.

The original research [demo video](https://drive.google.com/file/d/1CU9ajgWKnAI1ainfM-g37qSE5lLipkwa/view?usp=sharing)
predates this independent training implementation.

