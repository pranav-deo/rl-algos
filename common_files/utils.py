import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
import json

np.seterr('raise')

def get_resized_img_from_env(env, img_size=(64,64)):
    img = env.render(mode="rgb_array")
    img = cv2.resize(img, img_size, interpolation = cv2.INTER_AREA)
    img = np.moveaxis(img, -1, 0)
    return img

def T(arr, device=None) -> torch.Tensor:
    """ Shorthand for readability and code compactness. Gives float tensor on gpu.
    input - arr: numpy array or list """

    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return torch.tensor(arr, device=device, dtype=torch.float32)    

class ReplayBuffer():
    """ Torch array of the shape: size x (phi_next + phi + a + r + done_flag) """
    def __init__(self, size, obs_len, act_len, rew_len, device) -> None:
        self.buffer_size = size
        self.device = device
        self.replay_buffer = torch.zeros((self.buffer_size, obs_len*2 + act_len + rew_len + 1), dtype=torch.float32, device=self.device)  # One extra for done flag 
        self.buffer_counter = 0

    def add_to_buffer(self, phi_next, phi, aa, rr, done_flag) -> None:
        # phi_next, phi, a, r, done_flag
        self.replay_buffer[self.buffer_counter % self.buffer_size] = T(list(phi_next)+list(phi)+list(aa)+list(rr)+list(done_flag)).to(torch.float32)
        self.buffer_counter += 1

    def sample_random_minibatch(self, minibatch_length) -> torch.Tensor:
        minibatch_length = minibatch_length%(self.buffer_size-1)
        rng = np.random.default_rng()
        arr = np.arange(min(self.buffer_counter, self.buffer_size-1))
        rng.shuffle(arr)
        self.minibatch_idx = arr[:min(self.buffer_counter, minibatch_length)]
        minibatch = self.replay_buffer[self.minibatch_idx]
        return minibatch

    def get_buffer(self) -> torch.Tensor:
        return self.replay_buffer

class VisualReplayBuffer():
    """ Tuple of Torch arrays of the shape: size x (obs_img_size), size x (obs_img_size), size x (a + r + done_flag) """
    def __init__(self, size, obs_size, act_len, rew_len, device) -> None:
        self.buffer_size = size
        self.device = device
        self.replay_buffer = torch.zeros((self.buffer_size, act_len + rew_len + 1), dtype=torch.float32, device=self.device)  # One extra for done flag 
        self.buffer_counter = 0

        self.phi_next_img_buffer = torch.zeros((self.buffer_size, *obs_size), dtype=torch.float32)
        self.phi_img_buffer = torch.zeros((self.buffer_size, *obs_size), dtype=torch.float32)

    def add_to_buffer(self, phi_next_img, phi_img, aa, rr, done_flag) -> None:
        self.phi_img_buffer[self.buffer_counter%self.buffer_size] = T(phi_img, device='cpu')
        self.phi_next_img_buffer[self.buffer_counter%self.buffer_size] = T(phi_next_img, device='cpu')
        self.replay_buffer[self.buffer_counter % self.buffer_size] = T(list(aa)+list(rr)+list(done_flag))
        self.buffer_counter += 1

    def sample_random_minibatch(self, minibatch_length) -> tuple:
        minibatch_length = minibatch_length%(self.buffer_size-1)
        rng = np.random.default_rng()
        arr = np.arange(min(self.buffer_counter, self.buffer_size-1))
        rng.shuffle(arr)
        self.minibatch_idx = arr[:min(self.buffer_counter, minibatch_length)]
        minibatch = self.replay_buffer[self.minibatch_idx]
        return minibatch, self.phi_next_img_buffer[self.minibatch_idx], self.phi_img_buffer[self.minibatch_idx]

    def get_buffer(self) -> tuple:
        return self.replay_buffer, self.phi_next_img_buffer, self.phi_img_buffer

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def load_hyperparams():
    hyperparams = json.load(open('./common_files/hyperparams.json'))
    return hyperparams

class MetricLogger():
    def __init__(self, env_name, algo_name) -> None:
        self.algo_name = algo_name
        self.env_name = env_name
        self.critic_loss = []
        self.actor_loss = []
        self.rewards = []
    
    def log(self, item, name):
        if not (isinstance(item, float) or isinstance(item, int)):
            item = item[0] if isinstance(item, list) else item.item()
        if name == 'actor':
            self.actor_loss.append(item)
        elif name == 'critic':
            self.critic_loss.append(item)
        elif name == 'reward':
            self.rewards.append(item)

    def save_logs(self):
        np.save(f'{self.algo_name}/{self.env_name}/actor_losses.npy', np.array(self.actor_loss))
        np.save(f'{self.algo_name}/{self.env_name}/critic_losses.npy', np.array(self.critic_loss))
        np.save(f'{self.algo_name}/{self.env_name}/rewards.npy', np.array(self.rewards))

    def plot_logs(self):
        plt.figure()
        plt.plot(self.actor_loss, label='actor')
        plt.plot(self.critic_loss, label='critic')
        plt.legend()
        plt.xlabel('Number of Epochs')
        plt.savefig(f'{self.algo_name}/{self.env_name}/actor_critic_losses.png', bbox_inches='tight')
        plt.show()
        plt.figure()
        plt.plot(self.rewards, label='reward')
        plt.legend()
        plt.xlabel('Number of Epochs')
        plt.savefig(f'{self.algo_name}/{self.env_name}/rewards.png', bbox_inches='tight')
        plt.show()

def plot_dic(env_name):
    dic = json.load(open(f'./{env_name}/epi_rewards_dic.json'))
    arr_size = len(dic)
    rew_arr = np.zeros(arr_size)
    for key, val in dic.items():
        rew_arr[int(key)//10 - 1] = val

    plt.figure()
    plt.plot(rew_arr, label='reward')
    plt.xticks(ticks=np.arange(1,51,2), labels=np.arange(10,510,20), rotation=90)
    plt.legend()
    plt.savefig(f'{env_name}/rewards.png', bbox_inches='tight')
    plt.show()
