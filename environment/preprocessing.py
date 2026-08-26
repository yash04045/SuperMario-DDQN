import cv2
import numpy as np
import gymnasium as gym

class GrayScaleResize(gym.ObservationWrapper):
    """Convert RGB Frames to grayscale and resize to shape x shape"""

    def __init__(self, env, shape = 84):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(low = 0, high = 255, shape = (shape, shape), dtype = np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.shape, self.shape), interpolation = cv2.INTER_AREA)
        return frame

class SkipFrame(gym.Wrapper):
    """Repeat the same action for 'skip' frames, summing the reward."""

    def __init__(self, env, skip = 4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info
