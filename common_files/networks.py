import torch
import torch.nn as nn
import numpy as np
from common_files.utils import T, colorize

class Q_net(nn.Module):
    def __init__(self, obs, act, layer_size, device):
        super().__init__()
        self.device = device
        self.layer_size = layer_size
        self.fc1 = nn.Linear(obs+act, self.layer_size)
        self.fc2 = nn.Linear(self.layer_size, self.layer_size)
        self.fc3 = nn.Linear(self.layer_size, 1)
        self.relu = nn.ReLU()
        self.to(self.device)
    
    def forward(self, obs, act):
        x = torch.cat((obs, act), dim=1) 
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        #  print(out[0])
        return torch.squeeze(out, -1)
    
class mu_net(nn.Module):
    def __init__(self, in_features, out_features, act_space, layer_size, device):
        super().__init__()
        self.device = device
        self.act_space = act_space
        self.layer_size = layer_size
        self.fc1 = nn.Linear(in_features, self.layer_size)
        self.fc2 = nn.Linear(self.layer_size, self.layer_size)
        self.fc3 = nn.Linear(self.layer_size, out_features)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.to(self.device)

    def forward(self, x):
        # print(x.shape)
        # print(self.fc1)
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.tanh(self.fc3(out))

        # print(out)

        # map action range to output range
        if np.all(np.abs(self.act_space.low) == self.act_space.high):
            out = out * T(self.act_space.high)
        else:
            print(colorize('Action space has unsymmetrical bounds. Code not yet implemented!', color='red'))
            NotImplementedError

        return out 