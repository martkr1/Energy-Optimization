import torch
import itertools
from collections import namedtuple
import numpy as np

from utils import select_action

def optimize_model(policy_net, target_net, memory, optimizer, BATCH_SIZE, GAMMA):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.

    Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])


    #NOTE state is tuple so we need to reshape to be able to concat
    state_batch = torch.cat(batch.state) 
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch) #gather policy output with action indices along axis 1 

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100) # we dont want too big gradients
    optimizer.step()

    return loss


def train(env, policy_net, target_net, memory, optimizer, plotter, BATCH_SIZE, GAMMA, TAU):
    
    rewards = []
    consumptions = []
    losses = []

    select_action.steps_done = 0

    for i_episode in range(1):
        # Initialize the environment and get its state
        state = env.reset()
        state_flat = np.array(list(itertools.chain.from_iterable(state)))
        state_tensor = torch.tensor(state_flat, dtype=torch.float32).unsqueeze(0) # make torch tensor and add axis
        for t in itertools.count():
            action = select_action(env, policy_net, state_tensor)
            observation, reward, terminated, truncated = env.step(action, step_change=1)
            reward = torch.tensor([reward])
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                obs_flat = np.array(list(itertools.chain.from_iterable(observation)))
                next_state = torch.tensor(obs_flat, dtype=torch.float32).unsqueeze(0)



            action_idx = env.action_pairs.index(tuple(float(i) for i in action))

            # Store the transition in memory
            memory.push(state_tensor, torch.tensor(action_idx, dtype=torch.int64).view(1,1), next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            loss = optimize_model(policy_net, target_net, memory, optimizer, BATCH_SIZE, GAMMA)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if len(memory) >= BATCH_SIZE:
                # need batch size memory to calculate loss
                rewards.append(reward[0])
                losses.append(loss.detach())

                #NOTE we could use "state" but this is flattened tensor where indexing changes dependent on number of controllers
                consumptions.append(env.state[1].item()) # hence we call the simulator and the indexing is consistent. 

                plotter(consumptions, rewards, losses)

            if done: 
                break 