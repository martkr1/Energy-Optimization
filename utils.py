import numpy as np
import torch

import random 
from collections import deque, namedtuple





def select_action(env, policy_net, state):
    """ To ensure the model explores new spaces we will sometimes choose actions randomly. If not random we choose the action which result in the highest expected reward. 
    Choosing a random action will decay exponientially throughout learning. 
    """
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000

    select_action.steps_done += 1
    sample = np.random.rand(1)
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * select_action.steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            pol_out = policy_net(state).max(1).indices
            return torch.tensor(env.action_pairs[pol_out.item()])
    else:
        return torch.tensor(env.sample("action"), dtype=torch.float32) # random action, unsqueeze such that one action has shape (1,1) i.e. env.step works since we index action[0]
    

class ReplayMemory(object):
#Class to store action state pairs

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        Transition = namedtuple('Transition',
                    ('state', 'action', 'next_state', 'reward'))
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    