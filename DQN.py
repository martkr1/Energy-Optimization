import torch
import torch.nn as nn
import torch.nn.functional as F


"""
This is a simple model which works for a model having n_states different scalar observations 
but can only adjust one controller n_actions different ways (up, down, stay). When multiple controllers
are included we need some configuration of the network(s) that produces the output 
[(p_1, p_2, p_3)_1, .. , (p_1, p_2, p_3)_n] where n is the number of controllers and p_i is the 
probabilities of each action. 

Why probabilities and not just actions directy? 
with probabilities we allow for exploitation and can continusly adjust the prob of taking a 
particular action. 

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