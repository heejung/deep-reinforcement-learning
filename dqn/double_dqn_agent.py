import numpy as np

import torch
import torch.nn.functional as F
from dqn_agent import DqnAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DoubleDqnAgent(DqnAgent):
    """Interacts with and learns from the environment using Double DQN.
    
    Double DQN is explained further at:

    Hado van Hasselt, Arthur Guez, and David Silver. 2016.
    Deep reinforcement learning with double Q-Learning.
    In Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence (AAAI'16).
    AAAI Press 2094-2100.
    """

    def __init__(self, state_size, action_size, seed, fc1_size=64, fc2_size=64, tau=1e-3):
        """Initialize an DoubleDqnAgent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            fc1_size (int): size of the first fully-connected layer
            fc2_size (int): size of the second fully-connected layer
        """
        super(DoubleDqnAgent, self).__init__(state_size, action_size, seed, fc1_size, fc2_size, tau)
        self.tau = tau
    
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        self.qnetwork_target.eval()
        self.qnetwork_local.eval()
        with torch.no_grad():
            _, max_actions = torch.max(self.qnetwork_local(next_states), dim=1)
            qa_next_s = self.qnetwork_target(next_states)
            max_q_as = torch.gather(qa_next_s, dim=1, index=max_actions.unsqueeze(1))
            q_nextsa = rewards + gamma * max_q_as * (1 - dones)
        self.qnetwork_target.train()
        self.qnetwork_local.train()

        q_as = torch.gather(self.qnetwork_local(states), dim=1, index=actions)
        loss = F.mse_loss(q_as, q_nextsa)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)