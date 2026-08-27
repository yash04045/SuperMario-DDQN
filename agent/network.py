import torch 
import torch.nn as nn

class DQNNet(nn.Module):
    """A small CNN that  maps a stack of frames to one Q-value per action."""

    def __init__(self, input_shape, num_actions):
        super().__init__()
        c, h, w = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size = 8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 2, stride = 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out_size = self.conv(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        x = x.float() / 255.0
        return self.fc(self.conv(x))