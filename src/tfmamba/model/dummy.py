import torch
import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * input_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleConv1DClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, T, 1)
        x = x.transpose(1, 2)  # (B, 1, T)
        x = self.encoder(x)  # (B, C, T')
        x = self.pool(x).squeeze(-1)  # (B, C)
        return self.classifier(x)
