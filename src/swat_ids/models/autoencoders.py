from __future__ import annotations

import math

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_features: int,
        lstm_hidden: int,
        lstm_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.output = nn.Linear(lstm_hidden, input_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        latent = hidden[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        decoded, _ = self.decoder(latent)
        return self.output(decoded)


class MultiScaleBiLSTMAttentionAutoencoder(nn.Module):
    def __init__(
        self,
        input_features: int,
        conv_channels: int,
        kernel_sizes: tuple[int, ...],
        lstm_hidden: int,
        lstm_layers: int,
        attention_heads: int,
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
        self.encoder = nn.LSTM(
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
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.Linear(attention_dim, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, input_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        branch_outputs = [branch(h) for branch in self.branches]
        h = torch.cat(branch_outputs, dim=1).transpose(1, 2)
        h, _ = self.encoder(h)
        attended, _ = self.attention(h, h, h, need_weights=False)
        return self.reconstruction_head(attended)


class TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        input_features: int,
        model_dim: int,
        transformer_layers: int,
        attention_heads: int,
        feedforward_dim: int,
        dropout: float,
        max_sequence_length: int = 512,
    ) -> None:
        super().__init__()
        if model_dim % attention_heads != 0:
            raise ValueError("Transformer model dimension must be divisible by attention_heads.")

        self.input_projection = nn.Linear(input_features, model_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(
            model_dim=model_dim,
            dropout=dropout,
            max_sequence_length=max_sequence_length,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.output = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, input_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x)
        h = self.positional_encoding(h)
        h = self.encoder(h)
        return self.output(h)


class CNNTransformerAutoencoder(nn.Module):
    def __init__(
        self,
        input_features: int,
        conv_channels: int,
        kernel_sizes: tuple[int, ...],
        model_dim: int,
        transformer_layers: int,
        attention_heads: int,
        feedforward_dim: int,
        dropout: float,
        max_sequence_length: int = 512,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("At least one convolution kernel size is required.")
        if model_dim % attention_heads != 0:
            raise ValueError("Transformer model dimension must be divisible by attention_heads.")

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
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.input_projection = nn.Linear(conv_channels * len(kernel_sizes), model_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(
            model_dim=model_dim,
            dropout=dropout,
            max_sequence_length=max_sequence_length,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.output = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, input_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        branch_outputs = [branch(h) for branch in self.branches]
        h = torch.cat(branch_outputs, dim=1).transpose(1, 2)
        h = self.input_projection(h)
        h = self.positional_encoding(h)
        h = self.encoder(h)
        return self.output(h)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, dropout: float, max_sequence_length: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_sequence_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / model_dim)
        )
        encoding = torch.zeros(max_sequence_length, model_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError(
                f"Input sequence length {x.shape[1]} exceeds positional encoding length "
                f"{self.encoding.shape[1]}."
            )
        return self.dropout(x + self.encoding[:, : x.shape[1], :])
