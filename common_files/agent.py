import os
import numpy as np
import torch
import os

from common_files.utils import T, ReplayBuffer, colorize

class AgentTemplate():
    def __init__(self, obs_len, act_len, env_fn, max_env_steps, logger, algo_name) -> None:
        self.algo_name = algo_name
        self.obs_len = obs_len  # Assuming observation space is 1D array
        self.act_len = act_len  # Assuming action space is 1D array

        self.env, self.test_env = env_fn(), env_fn()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.buffer_size = int(1e6)
        self.discount_factor = 0.99         # Gamma of Bellmann eqn
        self.num_test_episodes = 10         # Number of test episodes after each epoch
        self.num_timesteps = 4000
        self.max_env_steps = max_env_steps

        self.step_counter = 0               # Counter for total number of steps 
        self.save_freq = 10                 # Save actor and crititc models after N epochs
        self.epoch_counter = 0              # Counter for number of epochs


        self.logger = logger
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
            os.mkdir(f'{self.algo_name}/{env_name}')
        save_str = '_bestRew' if best_reward else '_final'

        if step_save is not None:
            if not os.path.exists(f'{self.algo_name}/{env_name}/{step_save}'):
                os.mkdir(f'{self.algo_name}/{env_name}/{step_save}')
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

    def learn(self) -> None:
        self.epoch_counter+=1
        curr_obs = self.env.reset()
        actor_loss_list = []
        critic_loss_list = []

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
            self.update_policies(actor_loss_list, critic_loss_list)

            if done:
                curr_obs = self.env.reset()
            else:
                curr_obs = list(next_obs)

        rew_list = self.test_actor()
        l1 = -float('inf') if len(actor_loss_list)==0 else np.mean(actor_loss_list)
        l2 = -float('inf') if len(critic_loss_list)==0 else np.mean(critic_loss_list)
        rew_avg = np.mean(rew_list)

        self.logger.log(l1, 'actor')
        self.logger.log(l2, 'critic')
        self.logger.log(rew_avg, 'reward')

        # Save model if current params give best test rewards
        if len(self.logger.rewards)>1 and rew_avg > np.max(self.logger.rewards[:-1]):
            self.save_Q_model(best_reward=True)
        
        if self.epoch_counter%self.save_freq == 0:
            self.save_Q_model(step_save=self.epoch_counter)

        self.print_info(rew_avg, l1, l2)