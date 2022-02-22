from ast import Num
import gym
import time
from common_files.utils import MetricLogger, colorize

AGENT = 'sac'

if AGENT == 'td3':
    from td3.agent import Agent
elif AGENT == 'sac':
    from sac.agent import Agent

# Use only environments with continuous action spaces
env_name = 'LunarLanderContinuous-v2'

env = gym.make(env_name)

obs_init = env.reset()

num_epochs = 50
logger = MetricLogger(env_name, AGENT)
agent = Agent(obs_len=obs_init.shape[0], act_len=env.action_space.sample().shape[0], env_fn=lambda: gym.make(env_name),max_env_steps=1000, logger=logger)

print(colorize(f'Training {AGENT} agent on {env_name} Environment...', color='yellow'))

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print('='*50)
    agent.learn()
    print('='*50)

# Save Q model
agent.save_Q_model()

env.close()

logger.save_logs()
logger.plot_logs()

# Things changed:
# 1. explicitly freezing target params
# 2. torch no grad in test runs
# 3. Change way of writing polyak avg
# 4. Clipping of noisy action
# 5. Changed implementation of Noise in actions

# TODO:
# 1. count number of variables in the network
# 2. improve logging
