from core import mod_utils as utils
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from core.models import Actor, Critic, hard_update



class A2C(object):
    def __init__(self, args):

        self.args = args

        self.actor = Actor(args)
        self.actor_target = Actor(args)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(args)
        self.critic_target = Critic(args)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = args.gamma; self.tau = self.args.tau
        self.loss = nn.MSELoss()

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

    def update_parameters(self, batch):
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)
        state_batch.volatile = False;
        next_state_batch.volatile = True;
        action_batch.volatile = False

        # Critic Update
        vals = self.critic.forward(state_batch)
        new_vals = self.critic.forward(next_state_batch) * (1 - done_batch)
        targets = reward_batch + self.gamma * new_vals
        self.critic_optim.zero_grad()
        dt = self.loss(vals, targets)
        dt.backward()
        self.critic_optim.step()

        # Actor Update
        self.actor_optim.zero_grad()
        state_batch = utils.to_tensor(utils.to_numpy(state_batch)); targets = utils.to_tensor(utils.to_numpy(targets)); vals = utils.to_tensor(utils.to_numpy(vals))
        action_logs = self.actor.forward(state_batch)
        entropy_loss = torch.mean(entropy(torch.exp(action_logs)))
        action_logs = F.log_softmax(action_logs)
        dt = targets - vals
        alogs = []
        for i, action in enumerate(action_batch):
            action_i = int(action.cpu().data.numpy())
            alogs.append(action_logs[i, action_i])
        alogs = torch.cat(alogs).unsqueeze(0)

        policy_loss = -torch.mean(dt * alogs.t())
        actor_loss = policy_loss - entropy_loss
        actor_loss.backward()
        self.actor_optim.step()

