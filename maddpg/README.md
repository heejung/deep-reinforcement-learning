[//]: # (Image References)

[image1]: tennis_play.png "Trained Agent"


# Project 3: Collaboration and Competition

## Project Details

![Trained Agent][image1]

the state and action spaces, and when the environment is considered solved).

The environment is called Tennis based on UnityEnvironment. It is a multi-agent collaborative environment whose goal is to keep the ball from being dropped to the ground or to the net between the two agents. If either of this happens, the agent receives a reward of -0.01. If the agent hits the ball over the net, it receives a reward of +0.1. The state size is 24 per agent. The state consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. The action space size is 2 and consists of two continuous real numbers representing jumping and movement toward or away from the net. 

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Installation & Dependencies

### Unity Environment Set-up

1. Download the environment from one of the links below. You need only select the environment that matches your operating system:

Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

(For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this link to obtain the environment.

2. Place the file in this folder and unzip (or decompress) the file.

### Python Environment Set-up

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
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.

## Instructions

### How to Train a Model

`Tennis with Multi-Agent DDPG.ipynb` is the file you want to run. In the menu bar, hover your mouse cursor over `Kernel` > `Change Kernel` and click on `drlnd` environment you've created in the section above. Then, you should execute the following sections:

1. Start the Environment
2. Examine the State and Action Spaces
3. Performance of a Random Model
4. Hyperparameters
5. Class Definitions

### How to Test the Trained Model

`Tennis with Multi-Agent DDPG.ipynb` is the file you want to run. In the menu bar, hover your mouse cursor over `Kernel` > `Change Kernel` and click on `drlnd` environment you've created in the section above. Then, you should execute the following sections:

1. Start the Environment
2. Examine the State and Action Spaces
3. Performance of a Random Model
4. Hyperparameters
5. Class Definitions
7. Load & Test our Trained Model
6. Train the Model
