# Project 1. Navigation

## Environment

The environment is Banana created using Unity. Banana environment has the state size of 37 and the action size of 4 (left, right, forward, and backward). The environment is considered solved when the average score is 13.0+ for the series of 100 episodes. 

## Installation & Dependencies

### Unity Environment Set-up

1. Download the environment from one of the links below. You need only select the environment that matches your operating system:

Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

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

## How to Test the Model

`Banana with Double DQN.ipynb` is the file you want to run. In the menu bar, hover your mouse cursor over `Kernel` > `Change Kernel` and click on `drlnd` environment you've created in the section above. Then, you should execute the following sections:

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

## The Model

The model algorithm is adopted from Double Deep Q-Learning (Double DQN):

Hasselt, H.; Guez, A.; Silver, D. (2016), 'Deep reinforcement learning with double Q-Learning', In Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence (AAAI'16), AAAI Press 2094-2100.

The trained model weights are stored in `checkpoint_ddqn_banana.pth` file. The model architecture code is stored in `model.py`, `dqn_agent.py` and `ddqn_agent.py`. The model hyper parameters are chosen during a toy example LunarLander-v2 OpenAI gym environment: ReplyBuffer size 100000, batch size 64, discount factor gamma 0.99, learning rate 5e-4, target QNetwork updated every 2 steps, and tau 1e-3. The deep neural network used to represent the Q function has two fully connected layers of size 64 as that worked the best on the toy example evironment LunarLander-v2. Applying the same DDQN algorithm with the same best hyper parameters turned out to work well compared against the benchmark model performance given by the project which is to solve the environment in 1800 episodes. My model solved the environment in 500 episodes which is 1/3 of the benchmark. So, I didn't bother to try out any other hyper parameters. Instead, I tried training for as long as 2000 episodes to observe whether the model can learn more if trained for a longer period of time past the point it solved the environment. However, to my dismay, its average learning curve flattened 650th episode. 

The test performance of the trained model out-performs the minimal requirement when the model is run three times: 20.0, 14.0, and 20.0. The results are recorded and saved in `Banana with Double DQN.ipynb` file. 

## Future Work

During test time, it was having a trouble when it got too close to blue banas on both sides. It should have taken an action to avoid getting any closer to the blue banana or figure out a way to take one blue banana in order to take two more yellow banans to improve the final overall score. Since such is a rare corner case, prioritized replay algorithm might help improve the model's performance.  Probably, adopting all the improvement algorithms, including Dueling DQN, like Deepmind's Rainbow model will improve the model's performance drastically like Deepmind's paper shows.
