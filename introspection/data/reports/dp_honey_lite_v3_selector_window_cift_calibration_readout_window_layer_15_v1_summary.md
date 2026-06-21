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
| Accuracy | 0.8125 |
| Macro F1 | 0.8125 |
| Brier score | 0.1485 |
| Log loss | 0.4729 |
| Expected calibration error | 0.1110 |

## Calibration Bins

| Bin | Range | Examples | Mean Probability | Empirical Positive Rate | Absolute Gap |
|---:|---|---:|---:|---:|---:|
| 1 | [0.00, 0.10] | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2 | [0.10, 0.20] | 2 | 0.1470 | 0.0000 | 0.1470 |
| 3 | [0.20, 0.30] | 47 | 0.2499 | 0.0851 | 0.1648 |
| 4 | [0.30, 0.40] | 17 | 0.3539 | 0.2941 | 0.0598 |
| 5 | [0.40, 0.50] | 14 | 0.4463 | 0.4286 | 0.0177 |
| 6 | [0.50, 0.60] | 14 | 0.5315 | 0.5000 | 0.0315 |
| 7 | [0.60, 0.70] | 28 | 0.6728 | 0.8929 | 0.2201 |
| 8 | [0.70, 0.80] | 12 | 0.7617 | 0.7500 | 0.0117 |
| 9 | [0.80, 0.90] | 26 | 0.8574 | 0.9231 | 0.0657 |
| 10 | [0.90, 1.00] | 0 | 0.0000 | 0.0000 | 0.0000 |
