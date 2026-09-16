# SWaT W&B Sweeps

These sweep files tune the implemented SWaT model families with Weights &
Biases. Every trial still writes the normal local thesis artifacts under
`runs/wandb/<base_experiment>/<wandb_run_id>/`, including `metrics.json`,
`config.toml`, `resolved_config.toml`, and `model.pt`.

## Objective

The sweep objective is:

```text
test_attack_f1
```

This is stricter than optimizing accuracy, because SWaT is imbalanced and high
accuracy can hide weak attack detection. Use the W&B dashboard or the local
CSV aggregator to inspect secondary metrics:

- `test_accuracy`
- `test_macro_f1`
- `test_attack_precision`
- `test_attack_recall`
- `test_attack_fpr`
- `official_event_recall`
- `false_alarm_windows`
- `mean_detection_delay_seconds`

## Sweep Files

Supervised classifier sweeps:

```text
configs/sweeps/supervised_cnn.yaml
configs/sweeps/supervised_lstm.yaml
configs/sweeps/supervised_bilstm.yaml
configs/sweeps/supervised_cnn_lstm.yaml
configs/sweeps/supervised_cnn_bilstm.yaml
configs/sweeps/supervised_ms_cnn_bilstm_attention.yaml
```

Normal-only anomaly-detection sweeps:

```text
configs/sweeps/anomaly_lstm_autoencoder.yaml
configs/sweeps/anomaly_ms_cnn_bilstm_attention.yaml
configs/sweeps/anomaly_transformer_autoencoder.yaml
configs/sweeps/anomaly_cnn_transformer_autoencoder.yaml
```

## Recommended Counts

Run small sweeps first:

```text
10-15 trials per supervised baseline
20-30 trials for the final LSTM anomaly detector
20-30 trials for the proposed attention anomaly detector
10-15 trials for transformer variants
```

Then take the best configuration per model and repeat it with seeds `42`, `7`,
and `123` before making thesis claims.
