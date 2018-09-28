from core import mod_utils as utils
import torch, fastrand
import torch.nn as nn
from torch.optim import Adam
from core.adaptive_noise import OUNoise
from torch.nn.modules import CosineSimilarity, PairwiseDistance
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
from core.replay_memory import ReplayMemory
import numpy as np
from core.models import Actor, Critic, soft_update, hard_update
from core.uvfa import UVFA

class Actor_Critic(object):
    def __init__(self, state_dim, action_dim, gamma, tau, buffer_size, is_mem_cuda, out_act):

        self.actor = Actor(state_dim, action_dim, is_evo=False, out_act=out_act)
        self.actor_target = Actor(state_dim, action_dim, is_evo=False, out_act=out_act)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = gamma; self.tau = tau
        self.loss = nn.MSELoss()
        self.replay_buffer = ReplayMemory(buffer_size, is_mem_cuda)
        self.exploration_noise = OUNoise(action_dim)

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

    def act(self, state, is_noise):
        state = utils.to_tensor(state).unsqueeze(0)
        action = self.actor.forward(state)
        action = action.detach().numpy().flatten()
        if is_noise: action += self.exploration_noise.noise()
        return action


    def train_from_batch(self, batch):
        env_state_batch = torch.cat(batch.state)
        goal_batch = torch.cat(batch.goal)
        uvfa_states = torch.cat((env_state_batch, goal_batch), dim=1).detach()
        next_env_state_batch = torch.cat(batch.next_state)
        next_uvfa_states = torch.cat((next_env_state_batch, goal_batch), dim=1).detach()
        action_batch = torch.cat(batch.action).detach()
        reward_batch = torch.cat(batch.reward).detach()

        #if self.args.use_done_mask: done_batch = torch.cat(batch.done)


        #Load everything to GPU if not already
        # if self.args.is_memory_cuda and not self.args.is_cuda:
        self.actor.cuda(); self.actor_target.cuda(); self.critic_target.cuda(); self.critic.cuda()
        uvfa_states = uvfa_states.cuda(); next_uvfa_states = next_uvfa_states.cuda(); action_batch = action_batch.cuda(); reward_batch = reward_batch.cuda()
        #     if self.args.use_done_mask: done_batch = done_batch.cuda()


        #Critic Update
        with torch.no_grad():
            next_action_batch = self.actor_target.forward(next_uvfa_states)
            next_q = self.critic_target.forward(next_uvfa_states, next_action_batch)
            #if self.args.use_done_mask: next_q = next_q * ( 1 - done_batch.float()) #Done mask
            target_q = reward_batch + (self.gamma * next_q)

        self.critic_optim.zero_grad()
        current_q = self.critic.forward((uvfa_states.detach()), (action_batch.detach()))
        dt = self.loss(current_q, target_q)
        dt.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
        self.critic_optim.step()

        #Actor Update
        self.actor_optim.zero_grad()
        policy_loss = -self.critic.forward((uvfa_states),self.actor.forward((uvfa_states)))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        #Nets back to CPU if using memory_cuda
        self.actor.cpu(); self.actor_target.cpu(); self.critic_target.cpu(); self.critic.cpu()


class HAC(object):
    def __init__(self, args, core):
        self.args = args; self.core = core
        self.master = Actor_Critic(state_dim=args.state_dim, action_dim=args.goal_dim, gamma=args.gamma, tau=args.tau, buffer_size=args.buffer_size, is_mem_cuda=args.is_mem_cuda, out_act='limit2')
        self.sub = Actor_Critic(state_dim=args.state_dim, action_dim=args.action_dim, gamma=args.gamma, tau=args.tau, buffer_size=args.buffer_size, is_mem_cuda=args.is_mem_cuda, out_act='tanh')
        self.sub_period = 10
        self.vector_distance = PairwiseDistance()

    # def rew_func(self, state, goal):
    #     #Simplest is the cosine similarity
    #     #distance = self.vector_distance.forward(state, goal)
    #     #distance = np.abs(state - goal) > self.success_criteria).float()/self.args.state_dim
    #     achieved_goal = state[0:self.args.goal_dim]
    #     is_success = (np.abs(achieved_goal - goal) < self.success_criteria).all()
    #     return is_success

    def compute_her_reward(self, buffer, env):
        for experience in buffer:
            #reward = float(self.rew_func(experience[3], experience[1]))
            reward = env.compute_reward(experience[3][0:self.args.goal_dim], experience[1], experience[5])
            experience[4] = reward
        return buffer

    def her_augmentation(self, buffer, k):
        her_buffer = []; buffer_dim = len(buffer)-1
        for _ in range(k):
            for i in range(len(buffer)):
                # Chooses an index in the future
                if buffer_dim == i: her_goal_index = i #Edge case of last experience
                else: her_goal_index = i + fastrand.pcg32bounded(buffer_dim-i)

                her_goal = buffer[her_goal_index][3][0:self.args.goal_dim]
                her_buffer.append(buffer[i][:])
                her_buffer[-1][1] = her_goal

        return buffer + her_buffer

    def rollout(self, env, master, sub, master_noise=False, sub_noise=False, memorize=True, her_ratio=4):
        #Initializations
        master.actor.eval(); sub.actor.eval()
        fitness = 0.0; binary_success = 0.0
        master_obs = env.reset(); sub_obs = master_obs[:]

        done = False; steps = 0; master_trajectory = []
        while not done:
            #Master forward
            master_uvfa = np.concatenate((master_obs))
            subgoal = master.act(master_uvfa, master_noise)
            if master_noise == False: print(master_obs[1], subgoal)
            #subgoal = master_obs[1]

            ############### SUB NETWORK ###############
            sub_trajectory = []; master_step_reward = 0
            for time in range(self.sub_period):
                sub_uvfa = np.concatenate((sub_obs[0], subgoal))
                action = sub.act(sub_uvfa, sub_noise)

                sub_next_obs, env_reward, done, info = self.core.env.step(action)  # Simulate one step in environment
                fitness += env_reward; master_step_reward += env_reward; steps+=1
                sub_step_reward = env.compute_reward(sub_next_obs[0][0:self.args.goal_dim], subgoal, info)

                if memorize: sub_trajectory.append([sub_obs[0], subgoal, action, sub_next_obs[0], 0, info]) #[state. goal, action, next_state, reward, done]
                sub_obs[0] = sub_next_obs[0][:]

                #ir_reward = self.rew_func(sub_next_obs[0], subgoal)
                if done or sub_step_reward > -1 or info['is_success']: break

            #Process intrinsic reward for sub (transfer experiences to buffer)
            if memorize: #Sub
                her_trajectory = self.her_augmentation(sub_trajectory, her_ratio)
                her_trajectory = self.compute_her_reward(her_trajectory, env)
                #Fill to actual memory buffer
                for entry in her_trajectory+sub_trajectory: self.sub.replay_buffer.add_experience(entry[0], entry[1], entry[2], entry[3], entry[4], entry[5])

            # ############## BACK TO MASTER ##############
            #master_reward = master_step_reward + sub_step_reward #Count both the environment reward as well as the goal achieved reward
            if memorize: master_trajectory.append([master_obs[0], master_obs[1], subgoal, sub_next_obs[0], None, info])  # [state. goal, action, next_state, reward, done]
            master_obs[0] = sub_obs[0][:]

            if info['is_success']:
                binary_success = 1.0
                break
        ############ END INTERACTION WITH THE ENV #########

        # Process intrinsic reward for MASTER (transfer experiences to buffer)
        if memorize:
            her_trajectory = self.her_augmentation(master_trajectory, her_ratio)
            her_trajectory = self.compute_her_reward(her_trajectory, env)
            master_trajectory = self.compute_her_reward(master_trajectory, env)
            #Fill to actual memory buffer
            for entry in her_trajectory+master_trajectory: self.master.replay_buffer.add_experience(entry[0], entry[1], entry[2], entry[3], entry[4], entry[5])

        return fitness, steps, binary_success

    def train(self, tracker):
        for episode in range(1, self.args.num_games):
            _, steps, _ = self.rollout(self.core.env, self.master, self.sub, master_noise=True, sub_noise=True, memorize=True)

            #Gradient Descent
            if len(self.master.replay_buffer.memory) > self.args.batch_size:  # MASTER
                for _ in range(steps*3):
                    # MASTER
                    batch = self.master.replay_buffer.sample(self.args.batch_size)
                    self.master.train_from_batch(batch)

            if len(self.sub.replay_buffer.memory) > self.args.batch_size * 1: #SUB
                for _ in range(steps*3):
                    #SUB
                    batch = self.sub.replay_buffer.sample(self.args.batch_size)
                    self.sub.train_from_batch(batch)

            if episode % 10 == 0:
                test_score = 0.0; test_steps = 0.0; num_test = 20
                for _ in range(num_test):
                    _, steps, binary_success = self.rollout(self.core.env, self.master, self.sub, master_noise=False,
                                                                  sub_noise=False,
                                                                  memorize=False)
                    test_score += binary_success/num_test; test_steps += steps/num_test

                tracker.update([test_score, test_steps], episode)
                print ('Episode:', episode, 'Test_success_ratio:', '%.2f'%tracker.all_tracker[0][1], 'Avg_test_steps:', '%.2f'%tracker.all_tracker[1][1])






