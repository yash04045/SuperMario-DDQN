import os 

from environment import make_mario_env
from agent import DDQNAgent
from .rewards import shape_reward
from .evaluation import evaluate

def train(num_episodes = 10000, checkpoint_dir = "checkpoints", save_every = 100, log_every = 10):
    os.makedirs(checkpoint_dir, exist_ok = True)

    env = make_mario_env()
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n 

    agent = DDQNAgent(state_shape, num_actions)

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, shape_reward(reward), next_state, done)
            agent.learn()

            state = next_state
            episode_reward += reward

        if episode % log_every == 0:
            print(f"Episode {episode} | Reward {episode_reward:.1f}| Epsilon {agent.exploration.epsilon:.3f}")

        if episode % save_every ==0:
            agent.save(os.path.join(checkpoint_dir, f"ddqn_ep{episode}.pt"))
            stats = evaluate(agent,env)
            print(f" Eval: {stats}")

    env.close()
