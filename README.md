# DRL-MIMO-coupled-tanks-control

MATLAB and Simulink implementation of reinforcement-learning-based control for a nonlinear coupled MIMO tank system.

This repository contains the MATLAB and Simulink files associated with the manuscript:

**Reinforcement learning for control of nonlinear MIMO coupled systems**

The code implements reinforcement-learning-based controllers for a nonlinear coupled MIMO dual-tank process. The evaluated algorithms are:

- Deep Deterministic Policy Gradient (DDPG)
- Twin-Delayed Deep Deterministic Policy Gradient (TD3)
- Soft Actor-Critic (SAC)

The Simulink model defines the coupled-tank plant, the reinforcement learning environment, the observation signals, the reward calculation, the stopping conditions, and the derivative-action implementation used to generate the manipulated variables.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
├── matlab/
│   ├── train_DDPG.m
│   ├── train_TD3.m
│   └── train_SAC.m
└── simulink/
    └── rldepositos.slx
