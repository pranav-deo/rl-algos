import os
import numpy as np
import torch
import os
import pickle
import json

from common_files.utils import T, ReplayBuffer, VisualReplayBuffer, colorize, get_resized_img_from_env, load_hyperparams

class AgentTemplate():
    def __init__(self, obs_len, act_len, env_fn, max_env_steps, logger, algo_name) -> None:
        hyperparams = load_hyperparams()
        self.algo_name = algo_name
        self.obs_len = obs_len  # Assuming observation space is 1D array
        self.act_len = act_len  # Assuming action space is 1D array

        self.env, self.test_env = env_fn(), env_fn()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.buffer_size = int(hyperparams['buffer_size'])
        self.discount_factor = hyperparams['algo']['discount_factor']             # Gamma of Bellmann eqn
        self.num_test_episodes = hyperparams['algo']['num_test_episodes']         # Number of test episodes after each epoch
        self.num_timesteps = hyperparams['algo']['num_timesteps']
        self.max_env_steps = max_env_steps
        self.update_start = int(hyperparams['algo']['update_start'])              # Test and Update target networks only after these number of steps

        self.epi_eq_epoch = hyperparams['Episode=Epoch']
        self.step_counter = 0               # Counter for total number of steps 
        self.save_freq = 10                 # Save actor and crititc models after N epochs
        self.epoch_counter = 0              # Counter for number of epochs


        self.logger = logger
        if 'cnn' not in algo_name:
            self.r_buffer = ReplayBuffer(size=self.buffer_size, obs_len=self.obs_len, rew_len=1, act_len=self.act_len, device=self.device)

        # Init placeholder
        self.loss_actor = T(-float('inf'))
        self.loss_critic = T(-float('inf'))

    def save_Q_model(self, best_reward=False, step_save=None) -> None:
        env_name = self.env.unwrapped.spec.id
        if best_reward:
            print(colorize(f"\n Saving models with best rewards yet for {env_name} environment ...", color='green'))
        elif step_save is not None:
            print(f"\n Saving models and buffer for {env_name} environment ...")
        if not os.path.exists(f'{self.algo_name}/{env_name}'):
            os.makedirs(f'{self.algo_name}/{env_name}')
        save_str = '_bestRew' if best_reward else '_final'

        if step_save is not None:
            if not os.path.exists(f'{self.algo_name}/{env_name}/{step_save}'):
                os.makedirs(f'{self.algo_name}/{env_name}/{step_save}')
            env_name = env_name+f'/{step_save}'
            save_str = ''
        torch.save(self.actor_model.state_dict(), f'{self.algo_name}/{env_name}/actor{save_str}.pt')
        torch.save(self.critic_model_1.state_dict(), f'{self.algo_name}/{env_name}/critic{save_str}.pt')
        torch.save(self.r_buffer.get_buffer(), f'{self.algo_name}/{env_name}/r_buffer{save_str}.pt')

    def load_Q_model(self, best_reward=False, step=None) -> None:
        env_name = self.env.unwrapped.spec.id
        load_str = '_bestRew' if best_reward else '_final'
        if step is not None:
            env_name = env_name+f'/{step}'
            load_str = ''
        self.actor_model.load_state_dict(torch.load(f'{self.algo_name}/{env_name}/actor{load_str}.pt'))        
        self.critic_model_1.load_state_dict(torch.load(f'{self.algo_name}/{env_name}/critic{load_str}.pt'))

    def print_info(self, test_reward, a_loss, c_loss) -> None:
        print(f'Test Reward: {test_reward:.3f}  Actor avg metric: {a_loss:.3f}    Critic avg loss: {c_loss:.3f}')

    def grad_descent_step(self) -> None:
        raise NotImplementedError

    def network_update(self) -> None:
        raise NotImplementedError

    def test_actor(self) -> list:
        with torch.inference_mode():
            rew_list = []
            for _ in range(self.num_test_episodes):
                env_copy = self.test_env
                state = env_copy.reset()
                tot_reward = 0
                done = False

                while not done:
                    with torch.no_grad():
                        action = self.give_action(state, testing=True)
                    obs, rew, done, _ = env_copy.step(action)
                    tot_reward += rew
                    state = obs
                rew_list.append(tot_reward)
            return rew_list

    def give_target_Q(self):
        raise NotImplementedError

    def give_action(self, curr_obs:list, testing:bool) -> list:
        raise NotImplementedError

    def update_policies(self, actor_loss_list, critic_loss_list)-> None:
        """ Append actor and critic losses to the lists passed """
        raise NotImplementedError

    def test_and_save(self):
        rew_avg = -float('inf')
        if self.step_counter >= self.update_start:
            rew_list = self.test_actor()
            rew_avg = np.mean(rew_list)
        l1 = -float('inf') if len(self.actor_loss_list)==0 else np.mean(self.actor_loss_list)
        l2 = -float('inf') if len(self.critic_loss_list)==0 else np.mean(self.critic_loss_list)

        self.logger.log(l1, 'actor')
        self.logger.log(l2, 'critic')
        self.logger.log(rew_avg, 'reward')

        # Save model if current params give best test rewards
        if len(self.logger.rewards)>1 and rew_avg > np.max(self.logger.rewards[:-1]):
            self.save_Q_model(best_reward=True)
        
        if self.epoch_counter%self.save_freq == 0:
            self.save_Q_model(step_save=self.epoch_counter)

        self.print_info(rew_avg, l1, l2)

    def learn(self) -> None:
        self.epoch_counter+=1
        curr_obs = self.env.reset()
        self.actor_loss_list = []
        self.critic_loss_list = []

        _timesteps_in_epi = 0
        for tt in range(self.num_timesteps):
            curr_act = self.give_action(curr_obs)

            next_obs, rew , done, _ =  self.env.step(curr_act)

            # Add transition to buffer
            if _timesteps_in_epi == self.max_env_steps-1:
                done_ = False
            else:
                done_ = done

            self.r_buffer.add_to_buffer(phi_next=next_obs, phi=curr_obs, aa=curr_act, rr=[rew], done_flag=[1 if done_ else 0])
            _timesteps_in_epi += 1

            self.step_counter+=1
            self.update_policies(self.actor_loss_list, self.critic_loss_list)

            if done:
                if self.epi_eq_epoch:
                    break
                else:
                    curr_obs = self.env.reset()
            else:
                curr_obs = list(next_obs)

        self.test_and_save()

class VisualAgentTemplate(AgentTemplate):
    def __init__(self, obs_size, act_len, env_fn, max_env_steps, logger, algo_name) -> None:        
        super().__init__(obs_size, act_len, env_fn, max_env_steps, logger, algo_name)
        self.obs_size = obs_size
        self.r_buffer = VisualReplayBuffer(size=self.buffer_size, obs_size=self.obs_size, rew_len=1, act_len=self.act_len, device=self.device)

        # Init placeholder
        self.loss_actor = T(-float('inf'))
        self.loss_critic = T(-float('inf'))

    def save_Q_model(self, best_reward=False, step_save=None, save_buffer=False) -> None:
        env_name = self.env.unwrapped.spec.id
        if best_reward:
            print(colorize(f"\n Saving models with best rewards yet for {env_name} environment ...", color='green'))
        elif step_save is not None:
            print(f"\n Saving models and buffer for {env_name} environment ...")
        if not os.path.exists(f'{self.algo_name}/{env_name}'):
            os.makedirs(f'{self.algo_name}/{env_name}')
        save_str = '_bestRew' if best_reward else '_final'

        if step_save is not None:
            if not os.path.exists(f'{self.algo_name}/{env_name}/{step_save}'):
                os.makedirs(f'{self.algo_name}/{env_name}/{step_save}')
            env_name = env_name+f'/{step_save}'
            save_str = ''
        torch.save(self.actor_model.state_dict(), f'{self.algo_name}/{env_name}/actor{save_str}.pt')
        torch.save(self.critic_model_1.state_dict(), f'{self.algo_name}/{env_name}/critic{save_str}.pt')

        if save_buffer:
            print(colorize('Saving buffer is toggled on. This will result in high disk usage. Use wisely!', bold=True, color='crimson'))
            # torch.save(self.r_buffer.get_buffer(), f'{self.algo_name}/{env_name}/r_buffer{save_str}.pt')
            with open(f'{self.algo_name}/{env_name}/r_buffer{save_str}.pickle', 'wb') as f:
                pickle.dump(self.r_buffer.get_buffer(), f)

    def test_actor(self) -> list:
        with torch.inference_mode():
            rew_list = []
            for _ in range(self.num_test_episodes):
                env_copy = self.test_env
                _ = env_copy.reset()
                img_obs = get_resized_img_from_env(env_copy)
                tot_reward = 0
                done = False

                while not done:
                    with torch.no_grad():
                        action = self.give_action(img_obs, testing=True)
                    _, rew, done, _ = env_copy.step(action[0])
                    new_img_obs = get_resized_img_from_env(env_copy)
                    tot_reward += rew
                    img_obs = new_img_obs
                rew_list.append(tot_reward)
            return rew_list

    def learn(self) -> None:
        self.epoch_counter+=1
        _ = self.env.reset()
        curr_obs_img = get_resized_img_from_env(self.env)
        self.actor_loss_list = []
        self.critic_loss_list = []

        _timesteps_in_epi = 0
        for tt in range(self.num_timesteps):
            curr_act = self.give_action(curr_obs_img)
            _, rew , done, _ =  self.env.step(curr_act)
            next_obs_img = get_resized_img_from_env(self.env)

            # Add transition to buffer
            if _timesteps_in_epi == self.max_env_steps-1:
                done_ = False
            else:
                done_ = done

            self.r_buffer.add_to_buffer(phi_next_img=next_obs_img, phi_img=curr_obs_img, aa=curr_act, rr=[rew], done_flag=[1 if done_ else 0])
            _timesteps_in_epi += 1

            self.step_counter+=1
            self.update_policies(self.actor_loss_list, self.critic_loss_list)

            if done:
                if self.epi_eq_epoch:
                    self.env.reset()
                    # self.env.close()
                    break
                else:
                    _ = self.env.reset()
                    curr_obs_img = get_resized_img_from_env(self.env)
            else:
                curr_obs_img = next_obs_img

        self.test_and_save()