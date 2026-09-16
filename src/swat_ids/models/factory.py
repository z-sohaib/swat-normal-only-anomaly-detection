from __future__ import annotations

from torch import nn

from swat_ids.config import ModelConfig
from swat_ids.models.baselines import CNNClassifier, CNNRNNClassifier, RNNClassifier
from swat_ids.models.multiscale_bilstm_attention import MultiScaleBiLSTMAttention


def build_model(
    config: ModelConfig,
    input_features: int,
    num_classes: int,
) -> nn.Module:
    if config.name == "cnn":
        return CNNClassifier(
            input_features=input_features,
            num_classes=num_classes,
            conv_channels=config.conv_channels,
            kernel_sizes=config.kernel_sizes,
            dense_hidden=config.dense_hidden,
            dropout=config.dropout,
        )
    if config.name == "lstm":
        return RNNClassifier(
            input_features=input_features,
            num_classes=num_classes,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers,
            dense_hidden=config.dense_hidden,
            dropout=config.dropout,
            bidirectional=False,
        )
    if config.name == "bilstm":
        return RNNClassifier(
            input_features=input_features,
            num_classes=num_classes,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers,
            dense_hidden=config.dense_hidden,
            dropout=config.dropout,
            bidirectional=True,
        )
    if config.name == "cnn_lstm":
        return CNNRNNClassifier(
            input_features=input_features,
            num_classes=num_classes,
            conv_channels=config.conv_channels,
            kernel_sizes=config.kernel_sizes,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers,
            dense_hidden=config.dense_hidden,
            dropout=config.dropout,
            bidirectional=False,
        )
    if config.name == "cnn_bilstm":
        return CNNRNNClassifier(
            input_features=input_features,
            num_classes=num_classes,
            conv_channels=config.conv_channels,
            kernel_sizes=config.kernel_sizes,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers,
            dense_hidden=config.dense_hidden,
            dropout=config.dropout,
            bidirectional=True,
        )
    if config.name == "ms_cnn_bilstm_attention":
        return MultiScaleBiLSTMAttention(
            input_features=input_features,
            num_classes=num_classes,
            conv_channels=config.conv_channels,
            kernel_sizes=config.kernel_sizes,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers,
            attention_heads=config.attention_heads,
            dense_hidden=config.dense_hidden,
            dropout=config.dropout,
        )
    raise ValueError(
        f"Unsupported model.name: {config.name!r}. Expected one of: "
        "cnn, lstm, bilstm, cnn_lstm, cnn_bilstm, ms_cnn_bilstm_attention."
    )
