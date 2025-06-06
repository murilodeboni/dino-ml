# model.py

import torch.nn as nn
import torch.nn.functional as F


class DinoNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=16, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activations = {}

        # Register hooks for activations
        self.fc1.register_forward_hook(self._save_activation('fc1'))
        self.fc2.register_forward_hook(self._save_activation('fc2'))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _save_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach().cpu().numpy()

        return hook
