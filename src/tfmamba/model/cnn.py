import torch
import torch.nn as nn
import torch.nn.functional as F


class Simple1DCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )

        self.bn = nn.BatchNorm1d(16)

        self.pool = nn.MaxPool1d(kernel_size=2)

        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len, 1)

        x = x.permute(0, 2, 1)  # (B, 1, L)
        x = self.conv(x)  # (B, 16, L)
        x = self.bn(x)
        x = F.relu(x)

        x = self.pool(x)

        x = x.mean(dim=2)  # (B, 16)

        x = self.fc(x)  # (B, num_classes)
        return x


def show_model_params():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Simple1DCNN(
        num_classes=10,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f} M")


if __name__ == "__main__":
    show_model_params()
