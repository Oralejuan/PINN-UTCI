"""MLP compacto para predecir UTCI a partir de (x, y, hour, season, features)."""

import torch
import torch.nn as nn


class UTCIMLP(nn.Module):
    """MLP shallow con activación tanh (derivables suaves, ideal para PINN)."""

    def __init__(self, in_dim, hidden_dim=128, n_layers=4, out_dim=1):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)