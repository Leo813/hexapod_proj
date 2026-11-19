# Third-party notices

The root [MIT License](LICENSE) applies to independently developed software,
including the custom PyTorch PPO implementation, to the extent the repository
owner holds the necessary rights. It does not replace third-party terms or
assign third-party copyright to Leo813.

## NVIDIA Isaac Gym task

The original [hexapod.py](hexapod.py) was adapted from the ANYmal example in
[NVIDIA IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs). Its existing
Copyright (c) 2018-2023, NVIDIA Corporation notice and BSD 3-Clause terms remain
in the file and must be preserved. The root MIT License does not replace those
copyright and licensing requirements.

## Hexapod robot assets

The robot descriptions and meshes in [hexapod2/](hexapod2/) were provided to the
research project by the ERC laboratory associated with Saeed Saeedvand and
Jacky Baltes, the second and third authors of the paper. The project acknowledges
the laboratory's contribution; it does not claim that the repository owner
created these assets.

The exact origin and ownership of the CAD models, URDF and meshes have not been
independently established. The existing [ROS package metadata](hexapod2/package.xml)
designates the license as BSD but leaves author and maintainer information as
placeholders. It does not conclusively identify the rights holder or precise
terms for every asset. The URDF's SolidWorks exporter attribution is retained;
it does not establish ownership of the robot design.

The root MIT License is not intended to grant rights over third-party robot
assets whose licensing remains uncertain. Attribution alone does not establish
redistribution permission; applicable rights and terms need to be confirmed
with the relevant rights holders.

## External Isaac Gym dependencies

[NVIDIA Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym/download)
and the external IsaacGymEnvs installation remain subject to their respective
terms. This repository does not bundle the complete Isaac Gym SDK. The root
MIT License does not license these external dependencies.

## RL-Games reference

[RL-Games](https://github.com/Denys88/rl_games), by Denys Makoviichuk and Viktor
Makoviichuk, is a historical implementation reference for the independent PPO
implementation. It is not the custom training backend, and its package is not
bundled. The [reference mapping](docs/reference-implementation.md) preserves
upstream attribution and pinned source links; this acknowledgement does not
assert that RL-Games source files were copied into the custom package.

## Research paper

Yu-Hao Tsai, Saeed Saeedvand and Jacky Baltes authored
[A Deep Reinforcement Learning Algorithm for Objects Balance Control with Hexapod Robot](Paper_yuhaotsai.pdf),
included in this repository. The paper retains its original author attribution
and is not placed under the root MIT software license. No Creative Commons
license is introduced; any applicable rights and publication terms remain with
their respective holders.
