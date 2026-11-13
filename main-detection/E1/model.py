"""Residual 1D CNN, [batch, channels, 32] -> logits (normal, fault). About 230k parameters."""
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(incoming, outgoing, 3, stride=stride, padding=1, bias=False), nn.GroupNorm(8, outgoing), nn.ReLU(),
            nn.Conv1d(outgoing, outgoing, 3, padding=1, bias=False), nn.GroupNorm(8, outgoing))
        self.skip = nn.Identity() if incoming == outgoing and stride == 1 else nn.Sequential(
            nn.Conv1d(incoming, outgoing, 1, stride=stride, bias=False), nn.GroupNorm(8, outgoing))

    def forward(self, x):
        return torch.relu(self.body(x) + self.skip(x))


class BinaryCNN1D(nn.Module):
    def __init__(self, in_channels, hidden_features=64, dropout=.1):
        super().__init__()
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, 5, padding=2, bias=False), nn.GroupNorm(8, 64), nn.ReLU(),
            ResidualBlock(64, 64, 1), ResidualBlock(64, 96, 2), ResidualBlock(96, 128, 2), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, hidden_features), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_features, 2))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.in_channels:
            raise ValueError(f'expected [batch,{self.in_channels},time]')
        return self.head(self.encoder(x))


def feature_dropout(x, probability):
    """Drop each variable's 4 Gaussian + 3 Energy score channels (0..90) together; other channels untouched."""
    if probability == 0:
        return x
    keep = (torch.rand(len(x), 13, 1, device=x.device) >= probability).to(x.dtype)
    keep = torch.cat([keep.repeat_interleave(4, 1), keep.repeat_interleave(3, 1)], 1)
    return torch.cat([x[:, :91] * keep / (1 - probability), x[:, 91:]], 1)


@torch.inference_mode()
def predict_proba(model, x, device, batch=512):
    """p(fault) for every row."""
    model.eval()
    return torch.cat([model(torch.as_tensor(x[i:i + batch], device=device)).softmax(1)[:, 1].cpu()
                      for i in range(0, len(x), batch)]).numpy() if len(x) else x[:0, 0, 0]
