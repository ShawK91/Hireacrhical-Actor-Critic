
from torch.autograd import Variable
import random, pickle
import gym, numpy as np
import core.mod_utils as utils


class FetchWrapper(gym.Wrapper):


    def reset(self):
        return utils.to_tensor(self._preprocess_state(self.env.reset())).unsqueeze(0)


    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)

    def get_rendered_image(self):
        return self.env.render(mode='rgb_array')

    def reward(reward):
        return reward

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = utils.to_tensor(self._preprocess_state(next_state)).unsqueeze(0)
        return next_state, reward, done, info

    def _preprocess_state(self, state):
        """
        Do initial state preprocessing such as cropping, rgb2gray, rescale etc.
        Implementing this function is optional.
        :param state: a raw state from the environment
        :return: the preprocessed state
        """
        if np.isinf(self.observation_space.low).any() or np.isinf(self.observation_space.high).any(): return state
        return (state - self.observation_space.low) /(self.observation_space.high)





class FetchWrapper(gym.ObservationWrapper):
    """Flattens selected keys of a Dict observation space into
    an array.
    """
    def __init__(self, env, dict_keys):
        super(FetchWrapper, self).__init__(env)
        self.dict_keys = dict_keys

        # Figure out observation_space dimension.
        size = 0
        for key in dict_keys:
            shape = self.env.observation_space.spaces[key].shape
            size += np.prod(shape)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype='float32')

    def observation(self, observation):
        assert isinstance(observation, dict)
        obs = []
        for key in self.dict_keys:
            obs.append(observation[key].ravel())
        return obs
