# SWaT Industrial Anomaly Detection

Research code for the SWaT part of the thesis. This project studies deep learning for anomaly detection in a cyber-physical industrial control system using the Secure Water Treatment testbed.

The project evaluates whether the architecture family developed on UNSW-NB15 can transfer to a time-series industrial anomaly-detection setting. It also compares that architecture against a normal-only LSTM autoencoder protocol, which is closer to how SWaT is commonly evaluated in the literature.

## Research Objective

SWaT is different from flow-based intrusion datasets. The data is a multivariate time series collected from sensors and actuators in a physical process. The main research problem is not only classifying rows as normal or attack, but detecting abnormal temporal behavior while keeping false alarms low.

This project therefore focuses on:

- normal-only anomaly detection;
- temporal windowing over sensor and actuator variables;
- preprocessing fitted only on normal training data;
- attack labels used only for validation analysis and final evaluation;
- event-level and window-level metrics;
- precision, recall, F1, false-positive rate, and detection delay;
- comparison between simple recurrent autoencoders and attention-based temporal architectures.

## Dataset

Expected official SWaT files:

```text
data/raw/
  SWaT_Dataset_Normal_v1.csv
  SWaT_Dataset_Attack_v0.csv
  List_of_attacks_Final.csv
```

The attack-list file is useful for event-level evaluation. The dataset is not tracked by Git. Keep raw, converted, and processed files local.

If the official files are downloaded as Excel workbooks, convert them to CSV first using the conversion scripts in `scripts/`, then run the profiling and preprocessing checks.

## Project Layout

```text
.
  configs/                  Supervised, anomaly, final, and sweep configurations.
    sweeps/                 W&B sweep definitions.
  data/                     Local raw and processed SWaT files, ignored by Git.
  plan/                     Research plans, phase notes, and final evaluation notes.
  reports/                  Supervisor-facing summaries and generated figures.
  runs/                     Local experiment outputs, ignored by Git.
  scripts/                  Data checks, conversion helpers, reports, and W&B utilities.
  src/swat_ids/             Installable Python package.
    models/                 LSTM, CNN-LSTM, BiLSTM, attention, and autoencoder models.
    config.py               TOML configuration loading.
    data.py                 Dataset loading, cleaning, scaling, and windowing.
    train.py                Supervised training path.
    train_anomaly.py        Normal-only anomaly-detection training path.
    postprocessing.py       Temporal smoothing, merge-gap, and event logic.
    operational.py          Latency, throughput, and model-size utilities.
  pyproject.toml            Python package metadata and CLI entry points.
  requirements.txt          Project dependencies.
```

## Experimental Protocols

Two protocols exist in this project.

### 1. Supervised Classification

The supervised path was used as a diagnostic baseline. It trains directly on labeled windows. This is useful for checking whether the data pipeline and model implementations behave correctly, but it is not the main SWaT thesis protocol because the literature usually treats SWaT as anomaly detection.

### 2. Normal-Only Anomaly Detection

The final protocol trains only on normal operating data and evaluates on the attack-period file. This better matches the SWaT setting because real industrial attacks are rare, expensive to label, and difficult to represent fully during training.

The final model selection was based on a tradeoff between:

- high macro-F1;
- high attack precision and recall;
- low attack false-positive rate;
- event recall;
- detection delay;
- stable multi-seed behavior.

## Main Models

Implemented model families include:

- LSTM classifier;
- BiLSTM classifier;
- CNN-LSTM classifier;
- temporal CNN-BiLSTM-attention model;
- LSTM autoencoder / forecaster;
- attention-based autoencoder / forecaster.

The architecture transferred from UNSW-NB15 was tested in SWaT as a temporal multi-scale CNN-BiLSTM-attention anomaly model. It improved event sensitivity in some runs, but the final recommended SWaT model is the W&B-tuned LSTM autoencoder because it produced the strongest balance between high precision and low false alarms.

## Setup

Create a local environment from this project root:

```powershell
cd projects/swat
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python scripts/check_setup.py
```

For NVIDIA GPU training, install the CUDA-compatible PyTorch build for the local machine, then reinstall the project:

```powershell
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
python scripts/check_setup.py
```

Confirm GPU availability with:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## Reproducible Workflow

Validate setup and data:

```powershell
python scripts/check_setup.py
python scripts/profile_dataset.py --raw-dir data/raw --output-dir runs/profile
python scripts/validate_preprocessing.py --config configs/anomaly_final_event_sensitive_lstm_stable_top5_robust_z.toml --output-dir runs/preprocessing
```

Run a one-epoch software smoke test:

```powershell
swat-train --config configs/supervised_smoke.toml
```

Run supervised diagnostic baselines:

```powershell
swat-train --config configs/supervised_lstm.toml
swat-train --config configs/supervised_bilstm.toml
swat-train --config configs/supervised_cnn_lstm.toml
swat-train --config configs/supervised_ms_cnn_bilstm_attention.toml
```

Run anomaly-detection candidates:

```powershell
swat-train-anomaly --config configs/anomaly_final_event_sensitive_lstm_stable_top5_robust_z.toml
swat-train-anomaly --config configs/anomaly_final_proposed_attention_stable_max_robust_z.toml
```

Run final fixed-seed verification:

```powershell
swat-train-anomaly --config configs/final_wandb_lstm_seed42.toml
swat-train-anomaly --config configs/final_wandb_lstm_seed7.toml
swat-train-anomaly --config configs/final_wandb_lstm_seed123.toml
swat-train-anomaly --config configs/final_wandb_attention_seed42.toml
swat-train-anomaly --config configs/final_wandb_attention_seed7.toml
swat-train-anomaly --config configs/final_wandb_attention_seed123.toml
```

## W&B Sweeps

Sweep definitions live in:

```text
configs/sweeps/
```

Typical command pattern:

```powershell
wandb sweep --entity <entity> --project pfe-thesis-swat configs\sweeps\anomaly_lstm_autoencoder.yaml
wandb agent <entity>/pfe-thesis-swat/<sweep_id> --count 30
```

Use W&B to search hyperparameters. Final reported values should come from fixed-seed reruns of the selected configuration.

## Final Result Summary

Final recommended SWaT model:

```text
W&B-tuned LSTM autoencoder
Protocol: normal-only anomaly detection
Scoring: top-5 robust-z anomaly score
Post-processing: temporal smoothing, merge gap, minimum consecutive windows
Seeds: 42, 7, 123
```

Three-seed test result:

| Metric | Mean +/- Std |
| --- | ---: |
| Accuracy | 96.47% +/- 0.14% |
| Macro-F1 | 91.08% +/- 0.29% |
| Weighted-F1 | 96.27% +/- 0.13% |
| Attack precision | 97.07% +/- 1.88% |
| Attack recall | 74.27% +/- 0.84% |
| Attack F1 | 84.14% +/- 0.51% |
| Attack false-positive rate | 0.33% +/- 0.22% |
| Event recall | 47.22% +/- 0.00% |
| False-alarm segments | 5.33 +/- 5.13 |
| Mean detection delay | 100.76 s +/- 22.92 s |

Transferred attention model result:

| Model | Accuracy | Macro-F1 | Attack F1 | Event recall | Attack false-positive rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| W&B-tuned temporal MS-CNN-BiLSTM-attention | 94.30% +/- 1.34% | 87.13% +/- 2.28% | 77.52% +/- 3.77% | 58.33% +/- 7.35% | 3.23% +/- 1.92% |

The SWaT result should be framed honestly: the transferred attention architecture improves event sensitivity but increases false alarms, while the final LSTM autoencoder is the stronger operational model for this dataset.

## Git Policy

Do not commit:

- raw, converted, or processed SWaT files;
- local virtual environments;
- local runs and W&B logs;
- trained checkpoints;
- generated cache files.

Keep source code, configs, research notes, and Markdown reports under version control.
