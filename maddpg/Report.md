[./]: # (Image References)

[image1]: graph.png "Graph"


# Project 2: Collaboration and Competition

## Implementation Details (Learning Algorithm)

This work is inspired by the original paper [*Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments* by Lowe and Wu et al](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf).  The MADDPG architecture I implemented also rely on the standard DDPG agent implementation with a standard replay buffer.  Two DDPG agents are instantiated with the same hyperparameter values. Actor and Critic both have two hidden layers of size 256 and 128. Both networks have the same learning rate of 1e-4.  There is no reward discount during training.  Soft update coefficient tau is set to 0.001.  Batch size is set to 128. Buffer size is 1e5. Critic's weigh decay rate is explicitly set to 0.  The target networks are updated at every step. Actor outputs 2 real numbers between -1 and 1, representing the action space.  Critic outputs a single real number representing the estimated reward.


## Plot of Rewards

The model solved the problem in 1893 episodes after the first 100th episode is reached with the final average score 0.51 in 1492.98 seconds of training on a GPU.

![Graph][image1]


## Future work

1. Futher hyperparameter tuning.

2. Prioritized Experience Replay (PER): Improve the current memory replay to use PER for the faster learning. 

3. Try out the latest algorithm for Multi-Agent Reinforcement Learning (MARL) like this 
