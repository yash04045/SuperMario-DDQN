def shape_reward(reward, clip = 15.0):
    """Clip the reward to keep training stable.
    SuperMarioBrosEnv already computes a decent reward internally
    (x-position progress - time penalty - death penalty), so this doesn't
    add anything extra — it just keeps outlier values from destabilizing
    the loss.
    """

    return max(-clip, min(clip, reward))