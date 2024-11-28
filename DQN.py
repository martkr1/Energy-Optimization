import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Multi-Layer Perceptron with one hidden layer.

"""



class DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()

        # Let first try with a "simple" MLP with 1 hidden layer
        self.layer1 = nn.Linear(n_states, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)


    def forward(self, x):
        # used in __call__
        x = torch.tensor(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)