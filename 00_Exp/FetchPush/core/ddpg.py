from core import mod_utils as utils
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
from core.models import Actor, Critic, hard_update, soft_update


class DDPG(object):
    def __init__(self, args):

        self.args = args

        self.actor = Actor(args, init=True)
        self.actor_target = Actor(args, init=True)
        self.actor_optim = Adam(self.actor.parameters(), lr=0.5e-4)

        self.critic = Critic(args)
        self.critic_target = Critic(args)
        self.critic_optim = Adam(self.critic.parameters(), lr=0.5e-3)

        self.gamma = args.gamma; self.tau = self.args.tau
        self.loss = nn.MSELoss()

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

    def update_parameters(self, batch):
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        if self.args.use_done_mask: done_batch = torch.cat(batch.done)


        #Load everything to GPU if not already
        if self.args.is_memory_cuda and not self.args.is_cuda:
            self.actor.cuda(); self.actor_target.cuda(); self.critic_target.cuda(); self.critic.cuda()
            state_batch = state_batch.cuda(); next_state_batch = next_state_batch.cuda(); action_batch = action_batch.cuda(); reward_batch = reward_batch.cuda()
            if self.args.use_done_mask: done_batch = done_batch.cuda()




        #Critic Update
        next_action_batch = self.actor_target.forward(next_state_batch)
        with torch.no_grad():
            next_q = self.critic_target.forward(next_state_batch, next_action_batch)
            if self.args.use_done_mask: next_q = next_q * ( 1 - done_batch.float()) #Done mask
            target_q = reward_batch + (self.gamma * next_q)

        self.critic_optim.zero_grad()
        current_q = self.critic.forward((state_batch), (action_batch))
        dt = self.loss(current_q, target_q)
        dt.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
        self.critic_optim.step()

        #Actor Update
        self.actor_optim.zero_grad()
        policy_loss = -self.critic.forward((state_batch),self.actor.forward((state_batch)))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        #Nets back to CPU if using memory_cuda
        if self.args.is_memory_cuda and not self.args.is_cuda: self.actor.cpu(); self.actor_target.cpu(); self.critic_target.cpu(); self.critic.cpu()


