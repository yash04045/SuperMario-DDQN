import time

from environment import make_mario_env
from agent import DDQNAgent

CHECKPOINT_PATH = "checkpoints/ddqn_ep4700.pt"

env = make_mario_env()
state_shape = env.observation_space.shape
num_actions = env.action_space.n

agent = DDQNAgent(state_shape, num_actions)
agent.load(CHECKPOINT_PATH)

# make_mario_env() wraps the raw SuperMarioBrosEnv inside JoypadSpace ->
# SkipFrame -> GrayScaleResize -> FrameStack. render_mode has to be set on
# that innermost env, so unwrap all the way down to it.
env.unwrapped.render_mode = "human"

state, info = env.reset()
env.render()

done = False
while not done:
    action = agent.select_action(state, greedy=True)
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
    time.sleep(1 / 60)

env.close()
