import numpy as np 
import torch 
import torch.nn.functional as F 
import torch.optim as optim

from model import QNetwork

import random 
from collections import namedtuple, deque

LR = 5e-4
TAU = 1e-3
GAMMA = 0.99
UPDATE_EVERY = 4
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:

    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size 
        self.seed = random.seed(seed)

        self.q_local = QNetwork(state_size, action_size, seed).to(device)
        self.q_target = QNetwork(state_size, action_size, seed).to(device)
        self.q_target.eval()
        self.optimizer = optim.Adam(self.q_local.parameters(), lr=LR)
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        self.t_size = 0
    
    def step(self, state, action, reward, next_state, done):

        self.memory.add(state, action, reward, next_state, done)

        self.t_size = (self.t_size+1)%UPDATE_EVERY
        if self.t_size==0:
            if len(self.memory) > BATCH_SIZE:
                e = self.memory.sample()
                self.learn(e)

    def act(self, state, epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)        #Get state
        self.q_local.eval()                                         #Set Q_local in evaluate mode
        #Equivalent to q_local.train(False)
        with torch.no_grad():                                       #Get Action values
            action_values = self.q_local(state)
        self.q_local.train()                                        #Train Q_local

        if random.random()>epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size)).astype(np.int_)
    
    def learn(self, experiences, gamma=GAMMA):
        states, actions, rewards, next_states, dones = experiences

        #TD target
        best_actions = self.q_local(next_states).detach().max(1)[1].unsqueeze(1)
        evaluations = self.q_target(next_states).gather(1, best_actions) 
        Q_target = rewards + evaluations*gamma*(1-dones)

        #Currently predicted Q value
        Q_expected = self.q_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.q_local, self.q_target)

    def soft_update(self, local_model, target_model, tau=TAU):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data+(1.0-tau)*target_param.data)



class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.batch_size = batch_size
        self.batch_size = batch_size 
        self.seed = random.seed(seed)

        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        e = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([i.state for i in e if i is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([i.action for i in e if i is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([i.reward for i in e if i is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([i.next_state for i in e if i is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([i.done for i in e if i is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)