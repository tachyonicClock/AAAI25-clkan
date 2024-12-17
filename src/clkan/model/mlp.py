from typing import List

from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(self, feature_counts: List[int], activation: nn.Module):
        super().__init__()
        layers = [nn.Flatten()]
        for i in range(len(feature_counts) - 1):
            layers.append(nn.Linear(feature_counts[i], feature_counts[i + 1]))
            if i < len(feature_counts) - 2:
                layers.append(activation)
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        return self.network(x)
