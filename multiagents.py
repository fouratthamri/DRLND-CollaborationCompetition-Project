import numpy as np
import random
import copy
from collections import namedtuple, deque
from utils import OUNoise, Memory
from agents import Agent

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
UPDATE_EVERY = 2        # Udpate every
NB_LEARN = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiAgents():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, n_agents, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        
        self.ma = [Agent(state_size, action_size, i, n_agents, random_seed) for i in range(n_agents)]
        
        # Replay memory
        self.memory = Memory(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.t_step = 0
        
        
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(states, actions, rewards, next_states, dones)
            
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            for _ in range(NB_LEARN):
                for agent in self.ma:
                    experiences = self.memory.sample()
                    self.learn(experiences, agent, GAMMA)
                    
                for agent in self.ma:
                    agent.soft_update(agent.critic_local,
                          agent.critic_target,
                          TAU)
                    agent.soft_update(agent.actor_local,
                          agent.actor_target,
                          TAU)    
              
    def learn(self, experiences, agent, gamma):
        states, actions, _, _, _ = experiences

        actions_target =[agent_j.actor_target(states.index_select(1, torch.tensor([j]).to(device)).squeeze(1)) for j, agent_j in enumerate(self.ma)]
        
        agent_action_pred = agent.actor_local(states.index_select(1, agent.index).squeeze(1))
        actions_pred = [agent_action_pred if j==agent.index.cpu().data.numpy()[0] else actions.index_select(1, torch.tensor([j]).to(device)).squeeze(1) for j, agent_j in enumerate(self.ma)]
        
        agent.learn(experiences,
                    gamma,
                    actions_target,
                    actions_pred)


    def act(self, states, i_episode=0, add_noise=True):
        actions = [np.squeeze(agent.act(np.expand_dims(state, axis=0), add_noise), axis=0) for agent, state in zip(self.ma, states)]
        return np.stack(actions)
       
        
    def reset(self):
        for agent in self.ma:
            agent.reset()
        