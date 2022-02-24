from common_files.utils import MetricLogger, colorize, load_hyperparams

hyperparams = load_hyperparams()
AGENT = hyperparams['algo']['agent']

if 'cnn' in AGENT:
    # For headless rendering
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

import gym

if AGENT == 'td3':
    from td3.agent import Agent
elif AGENT == 'sac':
    from sac.agent import Agent
elif AGENT == 'sac-cnn':
    from sac.agent import VisualAgent

# Use only environments with continuous action spaces before changes
env_name = hyperparams['env_name']
env = gym.make(env_name)
obs_init = env.reset()

num_epochs = hyperparams['num_epochs']
logger = MetricLogger(env_name, AGENT)

if 'cnn' in AGENT:
    # obs_init_img = get_resized_img_from_env(env)
    agent = VisualAgent(obs_size=(3,64,64), act_len=env.action_space.sample().shape[0], env_fn=lambda: gym.make(env_name),max_env_steps=1000, logger=logger)
else:
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
