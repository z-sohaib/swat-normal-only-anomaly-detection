from __future__ import annotations

import torch
from torch import nn


class CNNClassifier(nn.Module):
    def __init__(
        self,
        input_features: int,
        num_classes: int,
        conv_channels: int,
        kernel_sizes: tuple[int, ...],
        dense_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("CNNClassifier requires at least one kernel size.")
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
        classifier_in = conv_channels * len(kernel_sizes)
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_in),
            nn.Linear(classifier_in, dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: batch, time, features.
        x = x.transpose(1, 2)
        branch_outputs = [branch(x) for branch in self.branches]
        h = torch.cat(branch_outputs, dim=1)
        pooled = h.mean(dim=2)
        return self.classifier(pooled)


class RNNClassifier(nn.Module):
    def __init__(
        self,
        input_features: int,
        num_classes: int,
        lstm_hidden: int,
        lstm_layers: int,
        dense_hidden: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        classifier_in = lstm_hidden * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_in),
            nn.Linear(classifier_in, dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        pooled = h.mean(dim=1)
        return self.classifier(pooled)


class CNNRNNClassifier(nn.Module):
    def __init__(
        self,
        input_features: int,
        num_classes: int,
        conv_channels: int,
        kernel_sizes: tuple[int, ...],
        lstm_hidden: int,
        lstm_layers: int,
        dense_hidden: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("CNNRNNClassifier requires at least one kernel size.")
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
        self.rnn = nn.LSTM(
            input_size=cnn_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        classifier_in = lstm_hidden * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_in),
            nn.Linear(classifier_in, dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: batch, time, features.
        x = x.transpose(1, 2)
        branch_outputs = [branch(x) for branch in self.branches]
        h = torch.cat(branch_outputs, dim=1).transpose(1, 2)
        h, _ = self.rnn(h)
        pooled = h.mean(dim=1)
        return self.classifier(pooled)
