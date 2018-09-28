from core import mod_utils as utils
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F



class UVFA():

    def __init__(self, state, goal):
        self.env_state = state
        self.goal = goal
        self.uvfa_state = torch.cat((state, goal))
