from collections import deque

import numpy as np
import gymnasium as gym

class FrameStack(gym.Wrapper):
    """Stack the last 'num_stack' frames so the agent can see motion"""

    def __init__(self, env, num_stack = 4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen = num_stack)
        h, w = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low = 0,
            high = 255,
            shape = (num_stack, h , w), 
            dtype = np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return np.stack(self.frames, axis = 0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return np.stack(self.frames, axis = 0), reward, terminated, truncated, info