---
title: "SWaT Progress Report"
author: "Thesis Progress Report"
date: "2026-08-08"
---

# SWaT Progress Report

## 1. Objective

This report summarizes the SWaT experimental branch of the thesis in a format
aligned with the UNSW-NB15 report. The purpose is to help decide whether SWaT
should be used as the second dataset in a short research paper centered on the
proposed architecture.

The main gap addressed is:

> SWaT anomaly detection should not be evaluated only with point-level
> accuracy. A useful industrial detector must also report false alarms, event
> detection, delay, and operational cost.

For the paper discussion, the central architecture is:

> Multi-scale CNN-BiLSTM-attention adapted from UNSW-NB15 to SWaT time-series
> anomaly detection.

## 2. Dataset And Protocol

The project uses the official SWaT physical/historian dataset.

| File | Role | Rows |
| --- | --- | ---: |
| `SWaT_Dataset_Normal_v1.csv` | Normal-operation source | 495,000 |
| `SWaT_Dataset_Attack_v0.csv` | Attack-period evaluation source | 449,919 |

Validated attack-period label distribution:

| Label | Rows |
| --- | ---: |
| Normal | 395,298 |
| Attack | 54,621 |

Final evaluation protocol:

- normal-only anomaly detection;
- training on normal-operation windows;
- validation/calibration on held-out normal windows;
- testing on the official attack-period file;
- scaling fitted only on training data;
- labels used only for evaluation;
- event-level evaluation uses the official attack list.

Reported metrics:

- accuracy;
- macro-F1;
- attack precision, recall, and F1;
- attack false-positive rate;
- event recall;
- false-alarm segments;
- false-alarm windows;
- detection delay;
- latency and parameter count.

## 3. Work Completed

| Phase | Work Completed | Status |
| --- | --- | --- |
| Phase 1 | Official data validation, profile, GPU/dependency checks | Done |
| Phase 2 | Chronological preprocessing and leakage checks | Done |
| Phase 3 | Supervised CNN/LSTM/CNN-BiLSTM/proposed baselines | Done |
| Phase 4 | Proposed supervised optimization | Done |
| Phase 5 | Normal-only anomaly-detection pipeline | Done |
| Phase 6 | Alarm-budgeted evaluation | Done |
| Phase 7 | Metric-bias and point-adjustment audit | Done |
| Phase 8 | Multi-seed stability and ablation | Done |
| Phase 9 | Transformer gap exploration | Done |
| Phase 10 | W&B finalization for proposed and comparator models | Done |

## 4. Proposed Architecture

The proposed architecture from UNSW-NB15 was transferred to SWaT by changing
the input representation and output head.

```text
Multivariate SWaT sensor/actuator window
  -> temporal multi-scale Conv1D kernels
  -> concatenate multi-scale temporal features
  -> BiLSTM temporal encoder
  -> multi-head self-attention
  -> reconstruction/forecasting anomaly head
  -> anomaly score + threshold/post-processing
```

Component mapping:

| Component | UNSW-NB15 Form | SWaT Form |
| --- | --- | --- |
| Input | Encoded flow feature sequence | Sensor/actuator time window |
| Multi-scale CNN | Feature-axis Conv1D | Temporal Conv1D |
| BiLSTM | Feature-dependency encoder | Temporal dynamics encoder |
| Attention | Feature-position attention | Time-window attention |
| Output | Multiclass classifier | Anomaly score from reconstruction/forecasting |

This keeps the paper narrative centered on the same architecture family while
respecting the fact that SWaT is a cyber-physical time-series dataset, not a
network-flow multiclass dataset.

## 5. Main Local Results

Values are mean +/- standard deviation across seeds `42`, `7`, and `123` where
available.

| Model | Accuracy | Macro-F1 | Attack Precision | Attack Recall | Attack F1 | Event Recall | False-Alarm Segments | Role |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Supervised CNN-BiLSTM baseline | 96.77% | 80.44% | Not comparable | Not comparable | 62.58% | 42.86% | very low | Supervised context only |
| Pre-W&B proposed MS-CNN-BiLSTM-attention | 82.46% +/- 8.84% | 72.47% +/- 10.01% | Not stable | Not stable | 55.98% +/- 14.10% | 62.04% +/- 4.24% | 73.00 +/- 41.33 | Early proposed version |
| **W&B proposed MS-CNN-BiLSTM-attention** | **94.30% +/- 1.34%** | **87.13% +/- 2.28%** | **78.65% +/- 10.84%** | **77.19% +/- 3.43%** | **77.52% +/- 3.77%** | **58.33% +/- 7.35%** | **25.00 +/- 16.70** | **Main proposed architecture result** |
| LSTM anomaly comparator | 96.47% +/- 0.14% | 91.08% +/- 0.29% | 97.07% +/- 1.88% | 74.27% +/- 0.84% | 84.14% +/- 0.51% | 47.22% +/- 0.00% | 5.33 +/- 5.13 | Operational comparator |
| CNN-transformer diagnostic | 47.40% | 44.21% | Very low | High false positives | 30.86% | 83.33% | 35 | Negative transformer ablation |

Interpretation:

- The proposed architecture improved strongly after W&B tuning.
- It achieved higher event recall and attack recall than the LSTM comparator.
- The LSTM comparator has better strict point-level F1 and false-alarm control.
- For a paper centered on the proposed architecture, SWaT can be used as a
  transfer experiment, but the paper must not claim that the proposed SWaT
  model is the best local detector on every operational metric.

## 6. Proposed Architecture Strengths

The proposed SWaT model is valuable because it:

- transfers the UNSW-NB15 architecture family to a different cybersecurity
  domain;
- improves from 72.47% macro-F1 pre-W&B to 87.13% macro-F1 after tuning;
- achieves the strongest local event recall among the two final candidates:
  58.33% vs 47.22%;
- achieves the strongest local attack recall among the two final candidates:
  77.19% vs 74.27%;
- provides evidence that attention-based temporal modeling can increase attack
  coverage, even when it costs more false alarms.

## 7. Operational Tradeoff

The main tradeoff is false-alarm control.

| Metric | Proposed MS-CNN-BiLSTM-attention | LSTM Comparator | Better |
| --- | ---: | ---: | --- |
| Attack recall | 77.19% | 74.27% | Proposed |
| Event recall | 58.33% | 47.22% | Proposed |
| Attack precision | 78.65% | 97.07% | LSTM comparator |
| Attack F1 | 77.52% | 84.14% | LSTM comparator |
| Attack FPR | 3.23% | 0.33% | LSTM comparator |
| False-alarm segments | 25.00 | 5.33 | LSTM comparator |

This table should be used honestly in discussion:

> The proposed architecture favors attack/event coverage, while the simpler
> LSTM comparator favors precision and false-alarm control.

## 8. Comparison With Deep-Learning SWaT References

The table below avoids classical ML references and avoids treating protocol-
different headline numbers as direct rankings.

| Reference | Deep Architecture | Reported SWaT Result | Protocol Note | Relation To This Project |
| --- | --- | ---: | --- | --- |
| CNN-LSTM + SHAP framework | CNN-LSTM with explainability | Strong point-level accuracy/F1/AUROC reported | Per-point anomaly detection on SWaT/WADI | Strong DL reference; our proposed model adds event/false-alarm reporting. |
| Dynamic CPS hybrid deep-learning framework | CNN + BiLSTM + attention / forecast-deviation family | High point-level performance reported | Forecast/deviation framework; protocol differs | Architecturally close; useful for positioning, not direct ranking. |
| TransEdge | Transformer-style anomaly detector | High F1/recall reported | Binary anomaly detection; protocol differs | Strong transformer reference; our local transformer did not reproduce this strength. |
| **This project** | **MS-CNN-BiLSTM-attention anomaly model** | **94.30% accuracy; 87.13% macro-F1; 77.52% attack F1** | Normal-only training, multi-seed, event/false-alarm reporting | **Main transferred proposed architecture result.** |

## 9. Contribution Summary

The SWaT contribution can be stated as:

> A transfer study of the proposed multi-scale CNN-BiLSTM-attention
> architecture from network-flow IDS to industrial cyber-physical anomaly
> detection, showing that the model achieves competitive attack/event coverage
> under a strict normal-only SWaT protocol while revealing a precision and
> false-alarm tradeoff.

This is useful for the thesis because it shows:

- the proposed architecture can be adapted across data modalities;
- event-level and false-alarm metrics change how models are judged;
- architecture complexity is not automatically sufficient for industrial
  anomaly detection;
- SWaT is scientifically useful, but less clean than CICIoT2023 if the short
  paper needs a simple architecture-centered comparison.

## 10. Limitations And Next Work

Limitations:

- the proposed SWaT model is not the best local model on strict attack F1 or
  false-alarm control;
- SWaT comparisons are protocol-sensitive;
- the local transformer result was weak under strict point metrics;
- drift and sensor-noise robustness are planned but not yet completed.

Next work:

- keep SWaT as strong thesis evidence for operational evaluation;
- use SWaT cautiously in a short paper if the paper title centers on the
  proposed architecture;
- if SWaT is selected for the paper, emphasize transfer and event recall rather
  than claiming best overall SWaT performance.

## References

- SWaT dataset page:
  <https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/swat/>
- CNN-LSTM + SHAP:
  <https://www.scitepress.org/publishedPapers/2026/144148/pdf/index.html>
- Dynamic CPS hybrid deep learning:
  <https://www.sciencedirect.com/science/article/pii/S2773186326001295>
- TransEdge:
  <https://link.springer.com/article/10.1007/s10844-026-01043-w>
