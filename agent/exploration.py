import random

class EpsilonGreedy:
    """Linearly decay epsilon from start to end over decay_steps."""

    def __init__(self, start = 1.0, end = 0.02, decay_steps = 100000):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.step_count = 0

    @property
    def epsilon(self):
        fraction = min(self.step_count / self.decay_steps, 1.0)
        return self.start + fraction * (self.end - self.start)

    def step(self):
        self.step_count += 1

    def should_explore(self):
        return random.random() < self.epsilon