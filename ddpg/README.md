[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"


# Project 2: Continuous Control

## Environment Details

This project solves the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment. The observation **state space** consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each **action** is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector is a number between -1 and 1. The environment contains 20 identical agents, each with its own copy of the environment. In order to solve this environment, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,
    - After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
    - This yields an average score for each episode (where the average is over all 20 agents).

![Trained Agent][image1]

## Getting Started

### Set up the environment

1. Download the environment:

    - **_Reacher with Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

2. Place it under the same package as Deep Deterministic Policy Gradients (DDPG).ipynb and unzip it.

### Set up the python environment

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

        - __Linux__ or __Mac__:
        ```bash
        conda create --name drlnd python=3.6
        source activate drlnd
        ```
        - __Windows__:
        ```bash
        conda create --name drlnd python=3.6
        activate drlnd
        ```

2. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

4. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.

### How to Test the Model

`Deep Deterministic Policy Gradients (DDPG).ipynb` is the file you want to run. In the menu bar, hover your mouse cursor over `Kernel` > `Change Kernel` and click on `drlnd` environment you've created in the section above. Then, you should execute the following sections:

1. Set up the environment
2. Insepct the state and action spaces
3. Test-run the model
4. End the environment

## How to Train

`Banana with Double DQN.ipynb` is the file you want to run. In the menu bar, hover your mouse cursor over `Kernel` > `Change Kernel` and click on `drlnd` environment you've created in the section above. Then, you should execute the following sections:

1. Set up the environment
2. Insepct the state and action spaces
3. Train
4. End the environment
