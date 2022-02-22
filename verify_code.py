import numpy as np
import gym

from common_files.utils import T
import matplotlib.pyplot as plt
from common_files.gym_to_gif import combine_gifs, label_with_episode_number
import seaborn as sns; sns.set()
import os
import imageio
import numpy as np
import json

np.seterr('raise')

AGENT = 'sac'

if AGENT == 'td3':
    from td3.agent import Agent
elif AGENT == 'sac':
    from sac.agent import Agent


def plot_Q_vals(actor):
    # Plot Q function for some action
    Q_vals = np.zeros((200,100))
    ii=0
    for xx in range(-1, 1, 200):
        for yy in range(0, 1, 100):
            Q_vals[ii%100][ii//100] = actor.critic_model_1(T([xx/100, yy/100, 0,0,0,0,0,0]).unsqueeze(0), T([0,0]).unsqueeze(0)).item()
            ii+=1
    ax = sns.heatmap(Q_vals.T)
    plt.show()

def main(epi):
    epi_reward = 0
    frames = []
    env = gym.make(env_name)
    obs_init = env.reset()
    actor = Agent(obs_init.shape[0], env.action_space.sample().shape[0], lambda: gym.make(env_name), max_env_steps=1000, logger=None)
    actor.load_Q_model(best_reward=False, step=None)
    # plot_Q_vals(actor)
    ii = 0
    print(f'Episode: {epi}')
    while True:
        action, _ = actor.actor_model(T(obs_init), deterministic=True)
        Q_val = actor.critic_model_1(T(obs_init).unsqueeze(0), action.unsqueeze(0))
        frame = env.render(mode="rgb_array")
        frames.append(label_with_episode_number(frame, epi, ii))
        obs, rew, done, _ = env.step(action.tolist())
        epi_reward += rew
        obs_init = obs
        print(f"Step: {ii}    Action: {[round(i,2) for i in action.cpu().detach().tolist()]}    Q_val: {Q_val.item():.3f}     Reward: {rew:.3f}")
        
        ii+=1
        if done:
            break
    env.close()
    epi_rewards_dic[epi] = epi_reward
    imageio.mimwrite(f'./{AGENT}/{env_name}/{epi}/result.gif', frames, fps=15)


if __name__ == "__main__":
    env_name = 'LunarLanderContinuous-v2'
    epi_list = [10*i for i in range(1, 6)]
    epi_rewards_dic = {}
    
    for epi in epi_list:
        main(epi)

    json.dump(epi_rewards_dic, open(f"./{AGENT}/{env_name}/epi_rewards_dic.json", 'w'))
    combine_gifs(AGENT, env_name, start_epoch=10, end_epoch=50, interval=10, num_rows=1, num_columns=5)