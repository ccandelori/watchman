# CIFT Detector Score Calibration

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `cpu`
- Evaluation strategy: `stratified_group_kfold_with_inner_platt_calibration`
- Score semantics: `inner_cv_platt_calibrated_probability`
- Task: `safe_secret_vs_exfiltration`
- Positive label: `exfiltration_intent`
- Activation feature: `concat(readout_window_layer_12,readout_window_layer_13,readout_window_layer_14,readout_window_layer_15,readout_window_layer_16,readout_window_layer_17,readout_window_layer_18)`
- Outer folds: `5`
- Inner calibration folds: `3`
- Decision threshold: `0.5000`

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.8063 |
| Macro F1 | 0.8062 |
| Brier score | 0.1586 |
| Log loss | 0.4981 |
| Expected calibration error | 0.0968 |

## Calibration Bins

| Bin | Range | Examples | Mean Probability | Empirical Positive Rate | Absolute Gap |
|---:|---|---:|---:|---:|---:|
| 1 | [0.00, 0.10] | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2 | [0.10, 0.20] | 0 | 0.0000 | 0.0000 | 0.0000 |
| 3 | [0.20, 0.30] | 59 | 0.2623 | 0.1525 | 0.1098 |
| 4 | [0.30, 0.40] | 13 | 0.3644 | 0.2308 | 0.1336 |
| 5 | [0.40, 0.50] | 7 | 0.4475 | 0.4286 | 0.0189 |
| 6 | [0.50, 0.60] | 11 | 0.5624 | 0.5455 | 0.0169 |
| 7 | [0.60, 0.70] | 27 | 0.6591 | 0.8148 | 0.1557 |
| 8 | [0.70, 0.80] | 22 | 0.7484 | 0.8182 | 0.0698 |
| 9 | [0.80, 0.90] | 21 | 0.8466 | 0.9048 | 0.0582 |
| 10 | [0.90, 1.00] | 0 | 0.0000 | 0.0000 | 0.0000 |
