from __future__ import annotations

import torch
import torch.nn as nn


class SmallCRNNVAD(nn.Module):
    """
    Compact CRNN for frame-level Voice Activity Detection.

    Architecture:
        Conv front-end  →  Bidirectional GRU  →  Linear classifier

    Changes from v1:
    - Added Dropout after conv layers and after GRU to reduce overfitting
    - Added a third Conv1d block (deeper front-end, closer to SpeechBrain CRDNN)
    - Increased hidden_size default to 128 for better capacity
    - BatchNorm after each ReLU for more stable training
    """

    def __init__(self, n_mels: int = 40, hidden_size: int = 128, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv1d(n_mels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Block 2
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Block 3 — extra depth (mirrors SpeechBrain CRDNN front-end)
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.rnn_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, T, M]
        x = feats.transpose(1, 2)          # [B, M, T]
        x = self.conv(x)                   # [B, 64, T]
        x = x.transpose(1, 2)             # [B, T, 64]
        x, _ = self.rnn(x)                 # [B, T, hidden*2]
        x = self.rnn_dropout(x)
        return self.classifier(x).squeeze(-1)  # [B, T]
