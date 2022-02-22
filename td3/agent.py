import torch
from torch.optim import Adam
import copy

from td3.networks import Q_net, mu_net
from common_files.utils import T
from common_files.agent import AgentTemplate

class Agent(AgentTemplate):
    def __init__(self, obs_len, act_len, env_fn, max_env_steps, logger) -> None:
        super().__init__(obs_len, act_len, env_fn, max_env_steps, logger)

        self.actor_lr = 1e-4
        self.critic_lr = 1e-4

        self.action_noise_std = 0.2         # Std of gaussian noise for actions in critic update 
        self.action_noise_limit = 0.5       # Clip Limit for gaussian noise for actions in critic update
        self.policy_update_freq = 2         # Actor policy update delay

        self.exploration_noise = 0.1        # Noise in actor action selection
        self.update_weight = 0.995          # Target network weight in updation
        self.update_frequency = 50          # Frequency of timesteps for target network updation
        self.update_start = int(1e3)        # Update target networks only after these number of steps
        self.uniform_steps_counter=int(1e5) # Use random action policy instead of actor for these no. of steps
        self.num_updates_per_cycle = 50     # Number of gradient updates per step 
        self.minibatch_length = 100         # minibatch length for the optimizer

        self.critic_model_1 = Q_net(obs=self.obs_len, act=self.act_len, layer_size=256, device=self.device)
        self.critic_model_2 = Q_net(obs=self.obs_len, act=self.act_len, layer_size=256, device=self.device)
        self.actor_model = mu_net(in_features=self.obs_len, out_features=self.act_len, act_space=self.env.action_space, layer_size=256, device=self.device)
        
        self.target_critic_model_1 = copy.deepcopy(self.critic_model_1)        
        self.target_critic_model_2 = copy.deepcopy(self.critic_model_2)        
        self.target_actor_model = copy.deepcopy(self.actor_model)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in list(self.target_critic_model_1.parameters())+list(self.target_critic_model_2.parameters())+list(self.target_actor_model.parameters()):
            p.requires_grad = False
        
        self.optimizer_actor = Adam(self.actor_model.parameters(), lr=self.actor_lr)
        self.optimizer_critic = Adam(list(self.critic_model_1.parameters())+list(self.critic_model_2.parameters()), lr=self.critic_lr)

    def grad_descent_step(self, minibatch, target_Q, policy_update=False) -> None:
        self.optimizer_critic.zero_grad()
        Q_val_1 = self.critic_model_1(obs=minibatch[:, self.obs_len:2*self.obs_len], act=minibatch[:,2*self.obs_len:2*self.obs_len+self.act_len])
        Q_val_2 = self.critic_model_2(obs=minibatch[:, self.obs_len:2*self.obs_len], act=minibatch[:,2*self.obs_len:2*self.obs_len+self.act_len])

        criterion = torch.nn.MSELoss()
        self.loss_critic_1 = criterion(Q_val_1, target_Q)
        self.loss_critic_2 = criterion(Q_val_2, target_Q)
        self.loss_critic = self.loss_critic_1 + self.loss_critic_2
        self.loss_critic.backward()
        self.optimizer_critic.step()
        
        if policy_update:
            for p in self.critic_model_1.parameters():
                p.requires_grad = False

            self.optimizer_actor.zero_grad()
            self.loss_actor = self.critic_model_1(obs=minibatch[:,self.obs_len:2*self.obs_len], act=self.actor_model(minibatch[:,self.obs_len:2*self.obs_len]))
            self.loss_actor = -torch.mean(self.loss_actor)
            self.loss_actor.backward()
            self.optimizer_actor.step()

            for p in self.critic_model_1.parameters():
                p.requires_grad = True

    def network_update(self) -> None:
        def interpolate_weights(target_wt, curr_wt) -> None:
            for p, p_tar in zip(curr_wt, target_wt):
                # p_tar.data.copy_(p_tar*self.update_weight + (1-self.update_weight)*p)
                p_tar.data.mul_(self.update_weight)
                p_tar.data.add_((1 - self.update_weight) * p.data)

        with torch.no_grad():
            interpolate_weights(target_wt=self.target_critic_model_1.parameters(), curr_wt=self.critic_model_1.parameters())
            interpolate_weights(target_wt=self.target_critic_model_2.parameters(), curr_wt=self.critic_model_2.parameters())
            interpolate_weights(target_wt=self.target_actor_model.parameters(), curr_wt=self.actor_model.parameters())

    def give_target_Q(self, minibatch):
        with torch.no_grad():
            act_ = self.target_actor_model(minibatch[:,:self.obs_len])
            noise_ =  torch.clip(torch.normal(0, self.action_noise_std, size=tuple(act_.shape), device=self.device), min=-1*self.action_noise_limit*torch.ones_like(act_), max=self.action_noise_limit*torch.ones_like(act_))
            noisy_act = torch.clip(act_ + noise_, min=T(self.env.action_space.low), max=T(self.env.action_space.high))

            max_Q_1 = self.target_critic_model_1(obs=minibatch[:,:self.obs_len], act=noisy_act)
            max_Q_2 = self.target_critic_model_2(obs=minibatch[:,:self.obs_len], act=noisy_act)
            max_Q = torch.minimum(max_Q_1, max_Q_2)
            target_Q = minibatch[:,-2] + (1-minibatch[:,-1])*self.discount_factor*max_Q
            return target_Q

    def give_action(self, curr_obs) -> list:
        if self.step_counter >= self.uniform_steps_counter:
            curr_act = self.actor_model(T(curr_obs))
            curr_act = curr_act + (2*torch.rand(tuple(curr_act.shape), device=self.device)-1)*self.exploration_noise
            curr_act = torch.clip(curr_act, min=T(self.env.action_space.low), max=T(self.env.action_space.high))
        else:
            # Uniform random action
            curr_act = self.env.action_space.sample()
        curr_act = curr_act.tolist()
        return curr_act

    def update_policies(self, actor_loss_list, critic_loss_list)-> None:
        if self.step_counter % self.update_frequency == 0 and self.step_counter >= self.update_start:
            for tt in range(self.num_updates_per_cycle):
                # Sample random minibatch
                minibatch = self.r_buffer.sample_random_minibatch(self.minibatch_length)
                
                target_Q = self.give_target_Q(minibatch)
                self.policy_update_freq
                policy_update = True if tt%self.policy_update_freq==1 else False
                self.grad_descent_step(minibatch, target_Q, policy_update=policy_update)
                self.network_update()

                actor_loss_list.append(self.loss_actor.item())
                critic_loss_list.append(self.loss_critic.item())

    def learn(self) -> None:
        super().learn()