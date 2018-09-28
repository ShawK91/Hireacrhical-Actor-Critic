from core import mod_utils as utils
import torch
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
    def __init__(self, state_dim, action_dim, gamma, tau, buffer_size, is_mem_cuda):

        self.actor = Actor(state_dim, action_dim, is_evo=False)
        self.actor_target = Actor(state_dim, action_dim, is_evo=False)
        self.actor_optim = Adam(self.actor.parameters(), lr=0.5e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_optim = Adam(self.critic.parameters(), lr=0.5e-3)

        self.gamma = gamma; self.tau = tau
        self.loss = nn.MSELoss()
        self.replay_buffer = ReplayMemory(buffer_size, is_mem_cuda)
        self.exploration_noise = OUNoise(action_dim)

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

    def act(self, state, is_noise):
        action = self.actor.forward(state)
        if is_noise: action += utils.to_tensor(self.exploration_noise.noise()).unsqueeze(0)
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


class HAC_general(object):
    def __init__(self, args, core):
        self.args = args; self.core = core
        self.master = Actor_Critic(state_dim=args.state_dim*2, action_dim=args.state_dim, gamma=args.gamma, tau=args.tau, buffer_size=args.buffer_size, is_mem_cuda=args.is_mem_cuda)
        self.sub = Actor_Critic(state_dim=args.state_dim*2, action_dim=args.action_dim, gamma=args.gamma, tau=args.tau, buffer_size=args.buffer_size, is_mem_cuda=args.is_mem_cuda)
        self.sub_period = 10; self.success_criteria = 0.1
        self.goals_primary = [core.env.reset()] #Initialize the goals primary
        self.vector_distance = PairwiseDistance()

    def compute_intrinsic_reward(self, state, goal):
        #Simplest is the cosine similarity
        #distance = self.vector_distance.forward(state, goal)
        distance = torch.sum(torch.abs(state - goal) > self.success_criteria).float()/self.args.state_dim
        return float(utils.to_numpy(distance))

    def fill_reward(self, buffer, reward):
        for entry in buffer: entry[-2] = reward
        return buffer

    def fill_her_reward(self, buffer, goal, her_ratio):
        count = 0; count_clip = len(buffer) * her_ratio
        for entry in buffer:
            entry[-2] = 1 #Imagined Reward (HER)
            entry[1] = goal #Actual goal achieved
            count += 1
            if count >= count_clip: break
        return buffer


    def rollout(self, env, master, sub, master_noise=False, sub_noise=False, memorize=True, her_ratio=1.0):
        #Initializations
        master.actor.eval(); sub.actor.eval()
        fitness = 0.0
        master_env_state = env.reset()
        sub_env_state = master_env_state

        done = False; steps = 0
        while not done:
            #if memorize: self.num_frames += 1; self.gen_frames += 1

            #Master forward
            master_uvfa = torch.cat((master_env_state, self.goals_primary[0]), dim=1)
            subgoal = master.act(master_uvfa, master_noise)

            #if master_noise == False: print(subgoal)
            ############### SUB NETWORK ###############
            temp_sub_buffer = []; master_env_reward = 0
            binary_sub_reward = 0.0
            for time in range(self.sub_period):
                sub_uvfa = torch.cat((sub_env_state, subgoal), dim=1)
                action = sub.act(sub_uvfa, sub_noise)

                next_sub_state, env_reward, done, info = self.core.env.step(action.detach().cpu().numpy().flatten())  # Simulate one step in environment
                fitness += env_reward; master_env_reward += env_reward; steps+=1

                if memorize: temp_sub_buffer.append([sub_env_state, subgoal, action, next_sub_state, None, done])
                sub_env_state = next_sub_state

                sub_ir = self.compute_intrinsic_reward(subgoal, sub_env_state)
                if sub_ir < self.success_criteria:
                    binary_sub_reward = 1.0
                    break
                elif done:break

            #Process intrinsic reward for sub (transfer experiences to buffer)
            if memorize: #Sub
                real_experiences = self.fill_reward(temp_sub_buffer, binary_sub_reward)
                for entry in real_experiences: self.sub.replay_buffer.add_experience(entry[0], entry[1], entry[2], entry[3], entry[4], entry[5])
                ############# HER #############
                her_experiences = self.fill_her_reward(temp_sub_buffer, sub_env_state, her_ratio)
                for entry in real_experiences: self.sub.replay_buffer.add_experience(entry[0], entry[1], entry[2], entry[3], entry[4], entry[5])


            ############## BACK TO MASTER ##############
            master_reward = master_env_reward #Count both the environment reward as well as the goal achieved reward
            if memorize: self.master.replay_buffer.add_experience(master_env_state, self.goals_primary[0], subgoal, sub_env_state, master_reward, done)
            master_env_state = sub_env_state

        return fitness, steps

    def train(self):
        for episode in range(1, self.args.num_games):
            self.rollout(self.core.env, self.master, self.sub, master_noise=True, sub_noise=True, memorize=True)
            fitness, steps = self.rollout(self.core.env, self.master, self.sub, master_noise=False, sub_noise=False,
                                   memorize=False)


            #Gradient Descent
            if not len(self.master.replay_buffer.memory) < self.args.batch_size * 1: #HEATUP PHASE
                for _ in range(steps):
                    #if episode % 100 > 10:
                        #MASTER
                        batch = self.master.replay_buffer.sample(self.args.batch_size)
                        self.master.train_from_batch(batch)

                    #else:
                        #SUB
                        batch = self.sub.replay_buffer.sample(self.args.batch_size)
                        self.sub.train_from_batch(batch)

            print (episode, fitness, steps)






