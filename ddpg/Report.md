[./]: # (Image References)

[image1]: graph.png "Graph"


# Project 2: Continuous Control

## Implementation Details

This is inspired by the original paper CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING by Lillicrap and Hunt et al: https://arxiv.org/pdf/1509.02971.pdf.  I tried the original architecture and the hyperparameters from the paper.  The training process was extremely slow as you can see in the main ipynb file; the model did not learn much during the initial 60 episodes. It might be due to the huge number of parameters that need to be trained in the original paper. I reduced the architecture size while keeping the batch norm because the paper said that it was important for generalization. And, I also tried the architecture without the batch norm. It turned out that, for this environment, the current architecture was already very good such that the batch norm actually hurts the stability of the learning. Although not shown in the python file, I also tried out an architecture with much fewer parameters to learn to make the learning process faster. And, I found that the hidden layer dimensions of 256 and 128 are the smallest possible dimension as a power of 2. I also tried clip_grad_norm as suggested by the original project instructions from Udacity. However, it turned out that, for my architecture, clip_grad_norm hurts the learning instead. So, I also removed it. Weight decay was also tried out as the original architecture used it. For this environment and the given architecture, weight decay did not help the learning either. Other than the hideen layer sizes, learning rates, weight decay, batch size, buffer size, and batch norm, every thing else is the same as the architecture described in the orignial paper mentioned above.

## Architecture

For the final model, I used two hidden layers of size 256 and 128 for both actor and critic neural networks. Action input is concatenated as the input to the second hidden layer in the critic model. tanh activation layer is used for the output of the actor layer to keep the value between -1 and 1. For all other hidden layers, ReLU activation is applied. Soft update is applied with the tau set to 0.001. Discount rate is set to 0.99. Reply memory buffer size is 1e5. Batch size is 128. Learning rates are set to 1e-4. Weight decay, batch norms, and clip_grad_norm are not applied. Optimizer is Adam for both actor and critic. Ornstein-Uhlenbeck process is used as the noise function for exploration with mu, theta, and sigma set to 0, 0.15, and 0.2, respectively, as described in the original paper mentioned at the top of this page.

## Results

The model solved the problem in 7 episodes after the first 100th episode is reached with the final average score 30.29. 

![Graph][image1]


## Future work

[Distributed Distributional Deterministic Policy Gradients (D4PG)](https://arxiv.org/pdf/1804.08617.pdf) might improve the learning speed even more. 