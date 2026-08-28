import numpy as np

def evaluate(agent, env, num_episodes = 5):
    """Run the agent greedily (no exploration) and report average reward"""

    rewards = []
    for _ in range(num_episodes):
        state, info = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, greedy = True)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)

    return{
        "mean_reward" : float(np.mean(rewards)),
        "std_reward" : float(np.std(rewards)),
    }