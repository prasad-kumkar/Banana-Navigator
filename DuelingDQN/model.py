import torch
import torch.nn.functional as F 
import torch.nn as nn

class QNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=32):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        '''
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        '''
        #Dualing DQN
        self.advantage1 = nn.Linear(state_size, 16)
        self.advantage2 = nn.Linear(16, action_size)

        self.value1 = nn.Linear(state_size, 16)
        self.value2 = nn.Linear(16, 1)

    def forward(self, x):
        ''''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.relu(self.fc3(x))
        '''
        x = x.float()
        x_adv = F.relu(self.advantage1(x))
        x_adv = F.relu(self.advantage2(x_adv))

        x_val = self.value1(x)
        x_val = self.value2(x_val)

        mean_adv = x_adv.mean(1).unsqueeze(1).expand_as(x_adv)

        return x_val + x_adv - mean_adv