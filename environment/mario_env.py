from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from nes_py.wrappers import JoypadSpace

from .preprocessing import GrayScaleResize, SkipFrame
from .wrappers import FrameStack

# gym-super-mario-bros (unmaintained since 2022) predates NumPy 2.0's
# stricter overflow rules: ram[0x6d] * 0x100 raises because 0x100 itself
# doesn't fit in uint8. Cast to Python int first to sidestep it.

def _x_position_patched(self):
    return int(self.ram[0x6d]) * 0x100 + int(self.ram[0x86])

SuperMarioBrosEnv._x_position = property(_x_position_patched)

def make_mario_env(skip = 4, shape = 84, num_stack = 4):
    # Built directly instead of via gym_super_mario_bros.make(), which is
    # hardcoded to legacy gym.make() and rejects nes-py's Gymnasium-native
    # action/observation spaces.
    env = SuperMarioBrosEnv(rom_mode = "vanilla")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip = skip)
    env = GrayScaleResize(env, shape = shape)
    env = FrameStack(env, num_stack = num_stack)
    return env
