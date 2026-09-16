from __future__ import annotations

from torch import nn

from swat_ids.config import ModelConfig
from swat_ids.models.autoencoders import (
    CNNTransformerAutoencoder,
    LSTMAutoencoder,
    MultiScaleBiLSTMAttentionAutoencoder,
    TransformerAutoencoder,
)


def build_anomaly_model(config: ModelConfig, input_features: int) -> nn.Module:
    if config.name == "lstm_autoencoder":
        return LSTMAutoencoder(
            input_features=input_features,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers,
            dropout=config.dropout,
        )
    if config.name == "ms_cnn_bilstm_attention_autoencoder":
        return MultiScaleBiLSTMAttentionAutoencoder(
            input_features=input_features,
            conv_channels=config.conv_channels,
            kernel_sizes=config.kernel_sizes,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers,
            attention_heads=config.attention_heads,
            dropout=config.dropout,
        )
    if config.name == "transformer_autoencoder":
        return TransformerAutoencoder(
            input_features=input_features,
            model_dim=config.lstm_hidden,
            transformer_layers=config.lstm_layers,
            attention_heads=config.attention_heads,
            feedforward_dim=config.dense_hidden,
            dropout=config.dropout,
        )
    if config.name == "cnn_transformer_autoencoder":
        return CNNTransformerAutoencoder(
            input_features=input_features,
            conv_channels=config.conv_channels,
            kernel_sizes=config.kernel_sizes,
            model_dim=config.lstm_hidden,
            transformer_layers=config.lstm_layers,
            attention_heads=config.attention_heads,
            feedforward_dim=config.dense_hidden,
            dropout=config.dropout,
        )
    raise ValueError(
        f"Unsupported anomaly model.name: {config.name!r}. Expected one of: "
        "lstm_autoencoder, ms_cnn_bilstm_attention_autoencoder, "
        "transformer_autoencoder, cnn_transformer_autoencoder."
    )
