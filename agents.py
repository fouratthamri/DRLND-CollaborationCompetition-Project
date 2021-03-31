import numpy as np
import random
import copy
from collections import namedtuple, deque
from utils import OUNoise, Memory
from nets import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # The replaybuffer memory size
GAMMA = 0.99            # The Bellman discount factor
LR_ACTOR = 1e-3         # Actor net learning rate 
LR_CRITIC = 1e-3        # Critic net learning rate 
WEIGHT_DECAY = 0        # L2 weight decay
TAU = 1e-3              # Soft update parameter 
BATCH_SIZE = 512        # minibatch size


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():    
    def __init__(self, state_size, action_size, random_seed):
        """
        Args:
        ======
            state_size (int): state dim
            action_size (int): action dim
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # actor net initialization
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # critic net initialization
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Ornstein-Uhlenbeck Exploration Noise Process
        self.noise = OUNoise(action_space=action_size, seed=random_seed)

        # Replay memory init
        self.memory = Memory(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, learning_memory_array, no_bounce, losing_sequence_adding_prob=0.5):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        if not no_bounce or random.uniform(0, 1)>= losing_sequence_adding_prob:
            for element in learning_memory_array:
                self.memory.add(element[0], element[1], element[2], element[3], element[4])

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            for _ in range(200):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """map action to state"""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.evolve_state()
        return np.clip(action, -1, 1)
    
    def act_on_all_agents(self, states):
        """map action to state to all agents"""
        vectorized_act = np.vectorize(self.act, excluded='self', signature='(n),()->(k)')
        return vectorized_act(states, True)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update actor and critic nets parameters

        Args:
        ======
            experiences (Tuple[torch.Tensor]): experience tuples 
            gamma (float): bellman discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        #Soft update model parameters
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)