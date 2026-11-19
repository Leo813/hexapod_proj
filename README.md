# Hexapod locomotion with independent PyTorch PPO

## Project overview

This repository combines the original Isaac Gym hexapod environment with an
independent PyTorch PPO training application. It retains the robot assets,
physics, observations, rewards and task YAMLs while replacing the RL-Games
learning backend.

**Start with the [installation and training guide](docs/getting-started.md).**
Native Isaac Gym/GPU execution of the custom application remains unverified;
successful training and reproduction of the research results are not claimed.

## Original research objective

The research investigates command-following locomotion and transporting objects
on uneven terrain with a hexapod robot. See
[the paper](Paper_yuhaotsai.pdf) and the original
[demonstration video](https://drive.google.com/file/d/1CU9ajgWKnAI1ainfM-g37qSE5lLipkwa/view?usp=sharing).
The demonstration predates the independent PPO implementation.

The preserved executable task includes rough terrain and three boxes, but fixes
commands to forward motion and leaves box-position reward terms and box-fall
termination checks inactive. It does not implement every curriculum and
payload-balancing objective described in the paper. The
[baseline contract](docs/baseline-contract.md) records the implementation baseline.

## Independent PyTorch PPO

The custom actor-critic, Gaussian policy, normalization, PPO updates, rollout
collection and checkpoint handling are implemented with PyTorch. Their behavior
uses RL-Games v1.6.0 as a historical reference; RL-Games is not the custom
training backend. Isaac Gym and the selected IsaacGymEnvs `VecTask` still supply
the simulation environment. No IsaacGymEnvs task-registry edit is required.

## Features

- Feedforward PPO with separate actor and critic MLPs (512-256-128, ELU).
- Original 104 observations, 18 policy actions and task-side mimic-joint coupling.
- GPU simulation and GPU learning, with 256 environments by default.
- Custom checkpoint saving/resume and deterministic evaluation.
- Task-file installation preview, conflict protection and hash verification.
- Offline tests using CPU fixtures without Isaac Gym or RL-Games.

## Requirements

The provisional native target is **Ubuntu 20.04, Python 3.8, PyTorch 1.13.1
with CUDA 11.7, and Isaac Gym Preview 4**, using IsaacGymEnvs revision
`aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3`. This exact combination has not been
validated natively. Install the SDK separately using its own instructions.

The original task also requires matplotlib, TensorBoard and SDK helper
dependencies; configuration uses OmegaConf. Python 3.8-compatible editable build
tooling is required. The [guide](docs/getting-started.md) explains the full
prerequisites, legacy NumPy concerns and how to preserve an existing CUDA stack.
The CPU CI dependency versions are a separate software-validation environment.

## Installation

Follow [Getting started](docs/getting-started.md) to prepare the historical
external checkout, clone this project, install the task automatically or by
manual copying, and install both Python packages separately without replacing
PyTorch. The guide includes file verification and external package discovery.
Keep the existing IsaacGymEnvs root configuration and the entire asset tree.

## Training

From this repository root, after completing installation and checking native
compatibility on your machine:

```bash
python train_agent.py
```

The launcher automatically selects `hexapod`, class `Anymal10`, and the existing
`hexapodPPO` configuration. Defaults include GPU PhysX, GPU learning, 256
environments, a viewer (`headless=False`), and a maximum of 600 training epochs.

```bash
# Displayless GPU server
python train_agent.py headless=True

# Smaller, bounded experiment
python train_agent.py num_envs=64 max_iterations=100

# Explicit external checkout when automatic discovery is unavailable
python train_agent.py isaacgymenvs_path=/absolute/path/to/IsaacGymEnvs

python train_agent.py --help
```

Add `headless=True` to other commands on displayless servers. Smaller environment
counts can trigger the existing minibatch adjustment. Explicit `task=hexapod`
remains accepted; other tasks and LSTM training are unsupported.

Outputs default to `runs/Anymal_<timestamp>/`. Resume a **custom** checkpoint:

```bash
python train_agent.py checkpoint=/path/to/last.pt max_iterations=700
```

This targets 700 total epochs and starts fresh simulator episodes. Repeat the
original PPO overrides. Resume creates a new timestamped output directory unless
`output_dir` is supplied. Checkpoint saves are conditional: `save_frequency=50`
is not a guarantee of a save every 50 epochs, and interruptions may leave no
recoverable checkpoint. Read the [checkpoint guide](docs/getting-started.md#8-outputs-and-resuming-training).

## Evaluation

```bash
python train_agent.py test=True checkpoint=/path/to/last.pt
```

Use `headless=True` on a displayless server and `eval_steps=1000` to set the
number of evaluation steps. Evaluation uses deterministic means and frozen
normalization without PPO updates. Historical RL-Games checkpoints are not
supported. See [evaluation and checkpoint trust](docs/getting-started.md#9-evaluate-a-trained-agent).

## Repository structure

| Path | Purpose |
| --- | --- |
| `train_agent.py` | Custom training/evaluation entry point |
| `src/hexapod_rl/` | PyTorch learning components and environment integration |
| `hexapod.py` | Preserved original Isaac Gym task |
| `hexapod.yaml`, `hexapodPPO.yaml` | Original task and feedforward PPO settings |
| `hexapodPPO_LSTM.yaml` | Historical configuration; unsupported by the custom launcher |
| `hexapod2/` | Original robot URDFs, meshes and package files |
| `scripts/install_hexapod_task.py` | Prepare unchanged task/configuration/asset files |
| `scripts/check_native_readiness.py` | Inventory and explicitly requested native diagnostics |
| `tests/`, `ci/` | Offline tests and strict test runner |
| `docs/` | Installation guide, baseline contracts, PPO details and upstream references |

## Testing status

The [GitHub Actions workflow](.github/workflows/pytorch-tests.yml) uses Ubuntu
22.04, Python 3.10, PyTorch 1.13.1+cpu, NumPy 1.24.4 and OmegaConf 2.3.0. Its
strict runner requires at least **209 tests**, complete execution and zero
failures, errors or skips. View actual run outcomes in the repository's
[Actions tab](https://github.com/Leo813/hexapod_proj/actions).

Local runs may skip tests when PyTorch or OmegaConf is unavailable; such runs
do not establish full numerical validation. Consult the Actions results for
the revision being used. CPU validation does not establish native GPU compatibility.

In a prepared CPU test environment:

```bash
python -B -m unittest discover -s ci -p 'test_*.py' -v
python -B ci/run_stage1_tests.py
```

Ordinary `unittest discover` allows skips; the second command enforces complete
coverage. Neither CPU tests nor mocks validate native mesh loading, GPU physics,
sensor ordering, CUDA checkpoint restoration or equivalent learning curves.

## References and attribution

- Yu-Hao Tsai, Saeed Saeedvand and Jacky Baltes,
  [A Deep Reinforcement Learning Algorithm for Objects Balance Control with Hexapod Robot](Paper_yuhaotsai.pdf).
- NVIDIA [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym/download)
  and [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs/tree/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3);
  the original task was adapted from the ANYmal example.
- [RL-Games v1.6.0 reference mapping](docs/reference-implementation.md)
  and [independent PPO implementation notes](docs/ppo-implementation.md).
- [Task installation and native diagnostics](docs/getting-started.md).

## License

The independently developed software, including the custom PyTorch PPO
implementation, is provided under the [MIT License](LICENSE), to the extent
the repository owner holds the necessary rights. Third-party code, robot assets
and research materials may have separate terms; see
[Third-party notices](THIRD_PARTY_NOTICES.md). Existing copyright notices remain
applicable, and the root MIT License does not grant rights over robot assets
whose licensing remains uncertain.
