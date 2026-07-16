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
```
Requirements
The code was developed for MATLAB/Simulink. To run the scripts and Simulink model, the following MathWorks products are required:
MATLAB
Simulink
Reinforcement Learning Toolbox
Deep Learning Toolbox
Additional toolboxes may be required depending on the MATLAB version and local configuration.
Files
`simulink/rldepositos.slx`
Simulink model containing the nonlinear coupled-tank plant and the reinforcement learning environment.
The model includes:
the coupled-tank plant;
reference signals;
observation generation;
reward calculation;
stopping logic;
RL Agent block;
derivative-action implementation for the pump and valve commands.
In the Simulink model, the second controlled tank is internally named tank 4 because the model was derived from a quadruple-tank notation. In the paper, this tank is referred to as tank 2.
`matlab/train_DDPG.m`
Training script for the DDPG controller. The actor is constrained to a deterministic linear feedback structure. After training, the actor weights define the feedback gain matrix used by the final controller.
`matlab/train_TD3.m`
Training script for the TD3 controller. TD3 uses two critic networks and a deterministic actor. The final actor defines a feedback gain matrix in the same way as in the DDPG implementation.
`matlab/train_SAC.m`
Training script for the SAC controller. SAC uses a stochastic Gaussian policy and the same plant and state-action environment used for the comparison in the paper.
State and action definitions
The reinforcement learning state vector is:
```text
s = [-e1, -e2, dy1/dt, dy2/dt]'
```
where `e1` and `e2` are the tracking errors of the controlled tank levels.
The action vector is:
```text
a = [dp/dt, dv/dt]'
```
where `p(t)` is the pump command and `v(t)` is the valve command.
For DDPG and TD3, the deterministic actor learns a linear feedback structure. Since the paper defines the control law as:
```text
u = -Kx
```
the feedback gain matrix can be obtained from the actor weights as described in the training scripts.
How to run
Open MATLAB.
Add the repository folders to the MATLAB path.
Open the Simulink model:
```matlab
open_system('simulink/rldepositos.slx')
```
Run one of the training scripts:
```matlab
run('matlab/train_DDPG.m')
run('matlab/train_TD3.m')
run('matlab/train_SAC.m')
```
Each script contains a `doTraining` flag. If `doTraining = true`, the corresponding agent is trained from scratch. If `doTraining = false`, the script attempts to load a previously saved trained agent file.
The scripts save the trained agents as:
```text
trained_DDPG_agent.mat
trained_TD3_agent.mat
trained_SAC_agent.mat
```
Notes on reproducibility
Training reinforcement learning agents may produce slightly different results depending on MATLAB version, toolbox version, numerical settings, and hardware. The scripts set the random seed using:
```matlab
rng(0)
```
to improve reproducibility.
License
This code is released under the MIT License. See the `LICENSE` file for details.
Citation
If you use this repository, please cite the associated paper. A `CITATION.cff` file is included for convenience.
