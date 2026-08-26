import time

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from nes_py.wrappers import JoypadSpace


# gym-super-mario-bros 7.4.0 predates NumPy 2.0's stricter scalar-overflow
# rules: `self.ram[0x6d] * 0x100` raises OverflowError because 0x100 (256)
# itself doesn't fit in the uint8 RAM values. Casting to Python int first
# sidesteps it without needing to downgrade NumPy.
def _x_position_patched(self):
    return int(self.ram[0x6d]) * 0x100 + int(self.ram[0x86])


SuperMarioBrosEnv._x_position = property(_x_position_patched)


# 1. Create the raw Mario environment
# NOTE: gym_super_mario_bros.make() is hardcoded to legacy gym.make()
# (see _registration.py), which rejects the Gymnasium-native spaces that
# nes-py 9.0.1 actually produces. Instantiating the env class directly
# skips that broken registry/checker and gives a clean Gymnasium env.
env = SuperMarioBrosEnv(rom_mode="vanilla")

# SuperMarioBrosEnv.__init__ doesn't accept a render_mode kwarg (it never
# forwards one to nes-py's NESEnv), so it's set directly on the instance.
# nes-py's env.render() then pops up a pyglet window each time it's called.
env.render_mode = "human"

# 2. Restrict the controller to a small set of useful actions
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# 3. Start a new episode
obs, info = env.reset()
env.render()

print("Initial observation:")
print("  shape :", obs.shape)
print("  dtype :", obs.dtype)

# 4. Take a few random actions (raised from 10 -> 300 so there's actually
# something to watch; at ~60 fps that's roughly 5 seconds of play)
for step in range(300):

    # Pick a random action from the available actions
    action = env.action_space.sample()

    # Perform the action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(1 / 60)

    print(f"\nStep {step + 1}")
    print("  action     :", action)
    print("  obs.shape  :", obs.shape)
    print("  obs.dtype  :", obs.dtype)
    print("  reward     :", reward)
    print("  terminated :", terminated)
    print("  truncated  :", truncated)
    print("  info.keys  :", info.keys())

    # Stop if the episode has ended
    if terminated or truncated:
        print("\nEpisode ended.")
        break

# 5. Close the environment
env.close()