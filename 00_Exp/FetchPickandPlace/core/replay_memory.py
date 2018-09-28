import random, numpy as np
from collections import namedtuple
from core import mod_utils as utils

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple(
    'Transition', ('state', 'goal', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity, is_mem_cuda):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.is_mem_cuda = is_mem_cuda


    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)


    def add_experience(self, state, goal, action, next_state, reward, done):
        #Format as tensors
        state = utils.to_tensor(state).unsqueeze(0)
        goal = utils.to_tensor(goal).unsqueeze(0)
        action = utils.to_tensor(action).unsqueeze(0)
        next_state = utils.to_tensor(next_state).unsqueeze(0)
        reward = utils.to_tensor(np.array([reward])).unsqueeze(0)


        #done = utils.to_tensor(np.array([done]).astype('uint8')).unsqueeze(0)

        #If storing memory in GPU, upload everything to the GPU
        if self.is_mem_cuda:
            reward.cuda()
            state.cuda
            goal.cuda()
            action.cuda()
            next_state.cuda()


        self.push(state, goal, action, next_state, reward, done)
