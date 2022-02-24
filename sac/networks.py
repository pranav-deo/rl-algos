import torch
import torch.nn as nn
import numpy as np
from common_files.utils import T, colorize, load_hyperparams

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
        return torch.squeeze(out, -1)

class Q_cnn_net(Q_net):
    def __init__(self, obs_img_shape, act, layer_size, device):
        super().__init__(6*6*12, act, layer_size, device)  # Assuming obs_img_shape=3x64x64
        
        hyperparams = load_hyperparams()
        oc = hyperparams['net']['out_channels']
        ss = hyperparams['net']['stride']
        ks = hyperparams['net']['kernel_size']
        self.conv1 = nn.Conv2d(obs_img_shape[0], out_channels=oc[0], kernel_size=ks, stride=ss)
        self.conv2 = nn.Conv2d(in_channels=oc[0], out_channels=oc[1], kernel_size=ks, stride=ss)
        self.to(self.device)

    def forward(self, obs, act):
        out = self.relu(self.conv1(obs))
        out = self.relu(self.conv2(out))
        return super().forward(out.view(-1, 6*6*12), act)
    
class pi_net(nn.Module):
    def __init__(self, in_features, out_features, act_space, layer_size, device):
        super().__init__()
        self.device = device
        self.act_space = act_space
        self.layer_size = layer_size
        self.fc1 = nn.Linear(in_features, self.layer_size)
        self.fc2 = nn.Linear(self.layer_size, self.layer_size)
        self.mu_layer = nn.Linear(self.layer_size, out_features)
        self.log_sigma_layer = nn.Linear(self.layer_size, out_features)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.to(self.device)

    def forward(self, x, deterministic=False):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        mu = self.mu_layer(out)
        log_sigma = self.log_sigma_layer(out)
        sigma = torch.exp(log_sigma)

        action_distribution = torch.distributions.MultivariateNormal(mu, torch.diag_embed(sigma))

        if deterministic:
            action = mu
        else:
            action = action_distribution.rsample()

        log_prob = action_distribution.log_prob(action)
        log_prob = log_prob - (2*(np.log(2) - action - torch.nn.functional.softplus(-2*action))).sum(axis=-1)

        # map action range to output range
        if not np.all(np.abs(self.act_space.low) == self.act_space.high):
            print(colorize('Action space has unsymmetrical bounds. Code not yet implemented!', color='red'))
            raise NotImplementedError

        action = torch.tanh(action)
        action = self.act_space.high[0] * action

        return action, log_prob

class pi_cnn_net(pi_net):
    def __init__(self, obs_img_shape, out_features, act_space, layer_size, device):
        super().__init__(6*6*12, out_features, act_space, layer_size, device)  # Assuming in_channels=3, img_size=64x64
        
        hyperparams = load_hyperparams()
        oc = hyperparams['net']['out_channels']
        ss = hyperparams['net']['stride']
        ks = hyperparams['net']['kernel_size']
        self.conv1 = nn.Conv2d(obs_img_shape[0], out_channels=oc[0], kernel_size=ks, stride=ss)
        self.conv2 = nn.Conv2d(in_channels=oc[0], out_channels=oc[1], kernel_size=ks, stride=ss)
        self.to(self.device)

    def forward(self, x, deterministic=False):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        return super().forward(out.view(-1, 6*6*12), deterministic)
