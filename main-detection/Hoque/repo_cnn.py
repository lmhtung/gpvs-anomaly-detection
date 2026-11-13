"""The repo's raw-waveform CNN and training routine, copied from gpvs-fault-detection/code/11_cnn.py (MIT, K. S. Hoque).

Only change: `fit` returns the trained model and the standardization, so it can also produce probabilities.
"""
import numpy as np
import torch
from torch import nn

SIGNALS = ["Ipv", "Vpv", "Vdc", "ia", "ib", "ic", "va", "vb", "vc", "Iabc", "If", "Vabc", "Vf"]
WIN = 200
EPOCHS, BATCH, LR = 15, 128, 1e-3


class CNN(nn.Module):
    def __init__(self, n_in, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_in, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), nn.Linear(64, n_cls))

    def forward(self, x):
        return self.net(x)


def fit(Xtr, ytr, seed, device='cpu'):
    """Standardize channels on the training set, train with class-weighted loss (as fit_predict in 11_cnn.py)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
    Xtr_t = torch.tensor((Xtr - mu) / sd)
    classes = np.unique(ytr)
    ytr_i = torch.tensor(np.searchsorted(classes, ytr))
    counts = np.bincount(ytr_i.numpy(), minlength=len(classes))
    w = torch.tensor(len(ytr) / (len(classes) * counts), dtype=torch.float32)
    model = CNN(Xtr.shape[1], len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(weight=w.to(device))
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=g)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss_fn(model(Xtr_t[b].to(device)), ytr_i[b].to(device)).backward()
            opt.step()
    model.eval()
    return model, mu, sd, classes


@torch.no_grad()
def predict_proba(model, mu, sd, X, device='cpu'):
    Xt = torch.tensor((X - mu) / sd)
    return torch.cat([model(Xt[i:i + 1024].to(device)).softmax(1).cpu()
                      for i in range(0, len(Xt), 1024)]).numpy() if len(X) else np.zeros((0, 2))
