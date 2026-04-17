from __future__ import annotations

import torch
import torch.nn as nn


class SmallCRNNVAD(nn.Module):
    def __init__(self, n_mels: int = 40, hidden_size: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, T, M]
        x = feats.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        return self.classifier(x).squeeze(-1)
