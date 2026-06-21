# CIFT Detector Score Calibration

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `cpu`
- Evaluation strategy: `stratified_group_kfold_with_inner_platt_calibration`
- Score semantics: `inner_cv_platt_calibrated_probability`
- Task: `safe_secret_vs_exfiltration`
- Positive label: `exfiltration_intent`
- Activation feature: `readout_window_layer_15`
- Outer folds: `5`
- Inner calibration folds: `3`
- Decision threshold: `0.5000`

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.7708 |
| Macro F1 | 0.7704 |
| Brier score | 0.1838 |
| Log loss | 0.5556 |
| Expected calibration error | 0.1701 |

## Calibration Bins

| Bin | Range | Examples | Mean Probability | Empirical Positive Rate | Absolute Gap |
|---:|---|---:|---:|---:|---:|
| 1 | [0.00, 0.10] | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2 | [0.10, 0.20] | 0 | 0.0000 | 0.0000 | 0.0000 |
| 3 | [0.20, 0.30] | 12 | 0.2472 | 0.3333 | 0.0861 |
| 4 | [0.30, 0.40] | 23 | 0.3768 | 0.1739 | 0.2029 |
| 5 | [0.40, 0.50] | 17 | 0.4320 | 0.2941 | 0.1379 |
| 6 | [0.50, 0.60] | 11 | 0.5501 | 0.4545 | 0.0955 |
| 7 | [0.60, 0.70] | 21 | 0.6698 | 0.8571 | 0.1873 |
| 8 | [0.70, 0.80] | 12 | 0.7246 | 1.0000 | 0.2754 |
| 9 | [0.80, 0.90] | 0 | 0.0000 | 0.0000 | 0.0000 |
| 10 | [0.90, 1.00] | 0 | 0.0000 | 0.0000 | 0.0000 |
