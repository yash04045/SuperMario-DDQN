import torch 
import torch.nn as nn
import numpy as np

from .network import DQNNet
from .replay_buffer import ReplayBuffer
from .exploration import EpsilonGreedy

class DDQNAgent:
    """Double DQN: the online network picks the next action, the target
    network evaluates it. Using two networks this way stops DQN's habit of
    overestimating Q-values."""

    def __init__(self, state_shape, num_actions, device = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.num_actions = num_actions

        self.online_net = DQNNet(state_shape, num_actions).to(device)
        self.target_net = DQNNet(state_shape, num_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr = 0.0002)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(capacity = 20000)
        self.exploration = EpsilonGreedy()

        self.gamma = 0.9
        self.batch_size = 32
        self.min_replay_size = 1000
        self.target_update_every = 10000
        self.total_steps = 0

    def select_action(self, state, greedy = False):
        if not greedy and self.exploration.should_explore():
            return np.random.randint(self.num_actions)

        state_t = torch.tensor(state, dtype = torch.float32, device = self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(torch.argmax(q_values, dim = 1).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self):
        self.total_steps += 1
        self.exploration.step()

        if len(self.replay_buffer) < self.min_replay_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype = torch.float32, device = self.device) 
        actions = torch.tensor(actions, dtype = torch.int64, device = self.device) 
        rewards = torch.tensor(rewards, dtype = torch.float32, device = self.device) 
        next_states = torch.tensor(next_states, dtype = torch.float32, device = self.device) 
        dones = torch.tensor(dones, dtype = torch.float32, device = self.device) 

        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_actions = torch.argmax(self.online_net(next_states), dim = 1)
            next_q_values = self.target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + (1-dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.total_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return loss.item()

    def save(self, path):
        torch.save(self.online_net.state_dict(), path)

    def load(self, path):
        self.online_net.load_state_dict(torch.load(path, map_location = self.device))
        self.target_net.load_state_dict(self.online_net.state_dict())


