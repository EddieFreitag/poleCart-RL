import torch
from torch import nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(64, 3)  # 3 actions: left, right, no force
        self.value_head = nn.Linear(64, 1)

    def forward(self, state):
        x = self.net(state)
        probs = torch.softmax(self.policy_head(x), dim=-1)
        value = self.value_head(x)
        return probs, value    