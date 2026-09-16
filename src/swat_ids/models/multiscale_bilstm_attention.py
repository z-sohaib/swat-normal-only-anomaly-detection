from __future__ import annotations

import torch
from torch import nn


class MultiScaleBiLSTMAttention(nn.Module):
    def __init__(
        self,
        input_features: int,
        num_classes: int,
        conv_channels: int,
        kernel_sizes: tuple[int, ...],
        lstm_hidden: int,
        lstm_layers: int,
        attention_heads: int,
        dense_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("At least one convolution kernel size is required.")

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=input_features,
                        out_channels=conv_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm1d(conv_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for kernel_size in kernel_sizes
            ]
        )
        cnn_features = conv_channels * len(kernel_sizes)
        self.lstm = nn.LSTM(
            input_size=cnn_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        attention_dim = lstm_hidden * 2
        if attention_dim % attention_heads != 0:
            raise ValueError("BiLSTM output dimension must be divisible by attention_heads.")
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.Linear(attention_dim, dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected shape: batch, time, features.
        x = x.transpose(1, 2)
        branch_outputs = [branch(x) for branch in self.branches]
        h = torch.cat(branch_outputs, dim=1)
        h = h.transpose(1, 2)
        h, _ = self.lstm(h)
        attended, _ = self.attention(h, h, h, need_weights=False)
        pooled = attended.mean(dim=1)
        return self.classifier(pooled)

