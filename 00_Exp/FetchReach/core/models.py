from core import mod_utils as utils
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F



def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)




class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, is_evo, out_act):
        super(Actor, self).__init__()
        self.out_act = out_act
        l1 = 128; l2 = 128; l3 = l2

        # Construct Hidden Layer 1
        self.w_l1 = nn.Linear(state_dim, l1)
        self.lnorm1 = LayerNorm(l1)

        #Hidden Layer 2
        self.w_l2 = nn.Linear(l1, l2)
        self.lnorm2 = LayerNorm(l2)

        #Hidden Layer 3
        # self.w_l3 = nn.Linear(l2, l3)
        # if self.args.use_ln: self.lnorm3 = LayerNorm(l2)

        #Out
        self.w_out = nn.Linear(l3, action_dim)

        #Init
        if not is_evo:
            self.w_out.weight.data.mul_(0.1)
            self.w_out.bias.data.mul_(0.1)


    def forward(self, input):


        #Hidden Layer 1
        out = self.w_l1(input)
        out = self.lnorm1(out)
        out = F.tanh(out)

        #Hidden Layer 2
        out = self.w_l2(out)
        self.lnorm2(out)
        out = F.tanh(out)

        # #Hidden Layer 3
        # out = self.w_l3(out)
        # if self.args.use_ln: out = self.lnorm3(out)
        # out = F.tanh(out)

        #Out
        out = self.w_out(out)
        if self.out_act == 'tanh': out = F.tanh(out)
        #out = F.sigmoid(out)
        return out


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        l1 = 128; l2 = 256; l3 = l2

        # Construct input interface (Hidden Layer 1)
        self.w_state_l1 = nn.Linear(state_dim, l1)
        self.w_action_l1 = nn.Linear(action_dim, l1)

        #Hidden Layer 2
        self.w_l2 = nn.Linear(2*l1, l2)
        self.lnorm2 = LayerNorm(l2)

        #Hidden Layer 3
        #self.w_l3 = nn.Linear(l2, l3)
        #if self.args.use_ln: self.lnorm3 = LayerNorm(l3)

        #Out
        self.w_out = nn.Linear(l3, 1)
        self.w_out.weight.data.mul_(0.1)
        self.w_out.bias.data.mul_(0.1)


    def forward(self, input, action):

        #Hidden Layer 1 (Input Interface)
        out_state = F.elu(self.w_state_l1(input))
        out_action = F.elu(self.w_action_l1(action))
        out = torch.cat((out_state, out_action), 1)

        # Hidden Layer 2
        out = self.w_l2(out)
        out = self.lnorm2(out)
        out = F.elu(out)

        # Hidden Layer 3
        # out = self.w_l3(out)
        # if self.args.use_ln: out = self.lnorm3(out)
        # out = F.elu(out)

        # Output interface
        out = self.w_out(out)

        return out




def fanin_init(size, fanin=None):
    v = 0.008
    return torch.Tensor(size).uniform_(-v, v)

def actfn_none(inp): return inp

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

def entropy(p):
    return -torch.sum(p * torch.log(p), 1)

def activation(self, mat):
    if self.args.actfn == 'tanh': return F.tanh(mat)
    elif self.args.actfn == 'sigmoid': return F.sigmoid(mat)
    elif self.actfn == 'none': return mat
    elif self.actfn == 'elu': return F.elu(mat)
    elif self.actfn == 'leaky_relu': return F.leaky_relu(mat)