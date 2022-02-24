import torch
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import copy

from sac.networks import Q_cnn_net, Q_net, pi_net, pi_cnn_net
from common_files.utils import T, load_hyperparams
from common_files.agent import AgentTemplate, VisualAgentTemplate
from itertools import chain

class Agent(AgentTemplate):
    def __init__(self, obs_len, act_len, env_fn, max_env_steps, logger, algo_name=None) -> None:
        super().__init__(obs_len, act_len, env_fn, max_env_steps, logger, algo_name='sac' if not algo_name else algo_name)

        if 'cnn' in algo_name:
            return

        hyperparams = load_hyperparams()

        self.actor_lr = hyperparams['algo']['actor_lr']
        self.critic_lr = hyperparams['algo']['critic_lr']

        self.policy_update_freq = hyperparams['algo']['policy_update_freq']         # Actor policy update frequency

        self.alpha = hyperparams['algo']['alpha']                                   # Entropy weight in target Q values
        self.update_weight = hyperparams['algo']['update_weight']                   # Target network weight in updation
        self.update_frequency = hyperparams['algo']['update_frequency']             # Frequency of timesteps for target network updation
        self.update_start = int(hyperparams['algo']['update_start'])                # Update target networks only after these number of steps
        self.uniform_steps_counter=int(hyperparams['algo']['uniform_steps_counter'])# Use random action policy instead of actor for these no. of steps
        self.num_updates_per_cycle = hyperparams['algo']['num_updates_per_cycle']   # Number of gradient updates per step 
        self.minibatch_length = hyperparams['algo']['minibatch_length']             # minibatch length for the optimizer

        self.critic_model_1 = Q_net(obs=self.obs_len, act=self.act_len, layer_size=hyperparams['net']['linear_dim'], device=self.device)
        self.critic_model_2 = Q_net(obs=self.obs_len, act=self.act_len, layer_size=hyperparams['net']['linear_dim'], device=self.device)
        self.actor_model = pi_net(in_features=self.obs_len, out_features=self.act_len, act_space=self.env.action_space, layer_size=hyperparams['net']['linear_dim'], device=self.device)
        
        self.target_critic_model_1 = copy.deepcopy(self.critic_model_1)        
        self.target_critic_model_2 = copy.deepcopy(self.critic_model_2)        

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in chain(self.target_critic_model_1.parameters(), self.target_critic_model_2.parameters()):
            p.requires_grad = False
        
        self.optimizer_actor = Adam(self.actor_model.parameters(), lr=self.actor_lr)
        self.optimizer_critic = Adam(chain(self.critic_model_1.parameters(), self.critic_model_2.parameters()), lr=self.critic_lr)

    def grad_descent_step(self, minibatch, target_Q, policy_update=False) -> None:
        obs = minibatch[:,self.obs_len:2*self.obs_len]
        act = minibatch[:,2*self.obs_len:2*self.obs_len+self.act_len]

        self.optimizer_critic.zero_grad()
        Q_val_1 = self.critic_model_1(obs=obs, act=act)
        Q_val_2 = self.critic_model_2(obs=obs, act=act)

        criterion = torch.nn.MSELoss()
        loss_critic_1 = criterion(Q_val_1, target_Q)
        loss_critic_2 = criterion(Q_val_2, target_Q)
        self.loss_critic = loss_critic_1 + loss_critic_2
        self.loss_critic.backward()
        self.optimizer_critic.step()
        
        if policy_update:
            for p in chain(self.critic_model_1.parameters(), self.critic_model_2.parameters()):
                p.requires_grad = False

            self.optimizer_actor.zero_grad()
            action, log_prob = self.actor_model(obs)
            
            self.loss_actor = torch.minimum(self.critic_model_1(obs=obs, act=action), self.critic_model_2(obs=obs, act=action))            
            self.loss_actor = self.loss_actor - self.alpha*log_prob
            self.loss_actor = -torch.mean(self.loss_actor)
            
            self.loss_actor.backward()
            self.optimizer_actor.step()

            for p in chain(self.critic_model_1.parameters(), self.critic_model_2.parameters()):
                p.requires_grad = True

    def network_update(self) -> None:
        def interpolate_weights(target_wt, curr_wt) -> None:
            for p, p_tar in zip(curr_wt, target_wt):
                p_tar.data.mul_(self.update_weight)
                p_tar.data.add_((1 - self.update_weight) * p.data)

        with torch.no_grad():
            interpolate_weights(target_wt=self.target_critic_model_1.parameters(), curr_wt=self.critic_model_1.parameters())
            interpolate_weights(target_wt=self.target_critic_model_2.parameters(), curr_wt=self.critic_model_2.parameters())

    def give_target_Q(self, minibatch):
        with torch.no_grad():
            obs_next = minibatch[:,:self.obs_len]
            action, log_prob = self.actor_model(obs_next)

            Q1 = self.target_critic_model_1(obs=obs_next, act=action)
            Q2 = self.target_critic_model_2(obs=obs_next, act=action)

            min_Q = torch.minimum(Q1, Q2)  - self.alpha*log_prob
            target_Q = minibatch[:,-2] + self.discount_factor*(1-minibatch[:,-1])*min_Q
            return target_Q

    def give_action(self, curr_obs, testing=False) -> list:
        if testing:
            with torch.no_grad():
                action, _ = self.actor_model(T(curr_obs).unsqueeze(0), deterministic=True)
        elif self.step_counter >= self.uniform_steps_counter:
            with torch.no_grad():
                action, _ = self.actor_model(T(curr_obs))
        else:
            # Uniform random action
            action = self.env.action_space.sample()
        action = action.tolist()
        return action

    def update_policies(self, actor_loss_list, critic_loss_list)-> None:
        if self.step_counter % self.update_frequency == 0 and self.step_counter >= self.update_start:
            for tt in range(self.num_updates_per_cycle):
                # Sample random minibatch
                minibatch = self.r_buffer.sample_random_minibatch(self.minibatch_length)
                
                target_Q = self.give_target_Q(minibatch)
                policy_update = True if tt%self.policy_update_freq==0 else False
                self.grad_descent_step(minibatch, target_Q, policy_update=policy_update)
                self.network_update()

                actor_loss_list.append(self.loss_actor.item())
                critic_loss_list.append(self.loss_critic.item())

    def learn(self) -> None:
        super().learn()

class VisualAgent(Agent, VisualAgentTemplate):
    def __init__(self, obs_size, act_len, env_fn, max_env_steps, logger, algo_name=None) -> None:
        VisualAgentTemplate.__init__(self, obs_size, act_len, env_fn, max_env_steps, logger, algo_name='sac-cnn')

        hyperparams = load_hyperparams()

        self.actor_lr = hyperparams['algo']['actor_lr']
        self.critic_lr = hyperparams['algo']['critic_lr']

        self.policy_update_freq = hyperparams['algo']['policy_update_freq']         # Actor policy update frequency

        self.alpha = hyperparams['algo']['alpha']                                   # Entropy weight in target Q values
        self.update_weight = hyperparams['algo']['update_weight']                   # Target network weight in updation
        self.update_frequency = hyperparams['algo']['update_frequency']             # Frequency of timesteps for target network updation
        self.update_start = int(hyperparams['algo']['update_start'])                # Update target networks only after these number of steps
        self.uniform_steps_counter=int(hyperparams['algo']['uniform_steps_counter'])# Use random action policy instead of actor for these no. of steps
        self.num_updates_per_cycle = hyperparams['algo']['num_updates_per_cycle']   # Number of gradient updates per step 
        self.minibatch_length = hyperparams['algo']['minibatch_length']             # minibatch length for the optimizer

        self.critic_model_1 = Q_cnn_net(obs_img_shape=self.obs_size, act=self.act_len, layer_size=hyperparams['net']['linear_dim'], device=self.device)
        self.critic_model_2 = Q_cnn_net(obs_img_shape=self.obs_size, act=self.act_len, layer_size=hyperparams['net']['linear_dim'], device=self.device)
        self.actor_model = pi_cnn_net(obs_img_shape=self.obs_size, out_features=self.act_len, act_space=self.env.action_space, layer_size=hyperparams['net']['linear_dim'], device=self.device)
        
        self.target_critic_model_1 = copy.deepcopy(self.critic_model_1)        
        self.target_critic_model_2 = copy.deepcopy(self.critic_model_2)        

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in chain(self.target_critic_model_1.parameters(), self.target_critic_model_2.parameters()):
            p.requires_grad = False
        
        self.optimizer_actor = Adam(self.actor_model.parameters(), lr=self.actor_lr)
        self.optimizer_critic = Adam(chain(self.critic_model_1.parameters(), self.critic_model_2.parameters()), lr=self.critic_lr)

    def grad_descent_step(self, minibatch, target_Q, policy_update=False) -> None:
        obs_img = minibatch[2].to(self.device)
        act = minibatch[0][:,:self.act_len].to(self.device)

        self.optimizer_critic.zero_grad()
        Q_val_1 = self.critic_model_1(obs=obs_img, act=act)
        Q_val_2 = self.critic_model_2(obs=obs_img, act=act)

        criterion = torch.nn.MSELoss()
        loss_critic_1 = criterion(Q_val_1, target_Q)
        loss_critic_2 = criterion(Q_val_2, target_Q)
        self.loss_critic = loss_critic_1 + loss_critic_2
        self.loss_critic.backward()
        self.optimizer_critic.step()
        
        if policy_update:
            for p in chain(self.critic_model_1.parameters(), self.critic_model_2.parameters()):
                p.requires_grad = False

            self.optimizer_actor.zero_grad()
            action, log_prob = self.actor_model(obs_img)
            
            self.loss_actor = torch.minimum(self.critic_model_1(obs=obs_img, act=action), self.critic_model_2(obs=obs_img, act=action))            
            self.loss_actor = self.loss_actor - self.alpha*log_prob
            self.loss_actor = -torch.mean(self.loss_actor)
            
            self.loss_actor.backward()
            self.optimizer_actor.step()

            for p in chain(self.critic_model_1.parameters(), self.critic_model_2.parameters()):
                p.requires_grad = True

    def network_update(self) -> None:
        Agent.network_update(self)

    def give_target_Q(self, minibatch):
        with torch.no_grad():
            obs_next_img = minibatch[1].to(self.device)
            action, log_prob = self.actor_model(obs_next_img)

            Q1 = self.target_critic_model_1(obs=obs_next_img, act=action)
            Q2 = self.target_critic_model_2(obs=obs_next_img, act=action)

            min_Q = torch.minimum(Q1, Q2)  - self.alpha*log_prob
            target_Q = minibatch[0][:,-2].to(self.device) + self.discount_factor*(1-minibatch[0][:,-1].to(self.device))*min_Q
            return target_Q

    def give_action(self, curr_obs_img, testing=False) -> list:
        return Agent.give_action(self, curr_obs_img, testing)

    def update_policies(self, actor_loss_list, critic_loss_list)-> None:
        Agent.update_policies(self, actor_loss_list, critic_loss_list)

    def learn(self) -> None:
        return VisualAgentTemplate.learn(self)