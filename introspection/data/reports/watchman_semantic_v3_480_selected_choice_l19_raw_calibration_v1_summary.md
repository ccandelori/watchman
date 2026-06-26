# CIFT Detector Score Calibration

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `mps`
- Evaluation strategy: `stratified_group_kfold_with_inner_platt_calibration`
- Score semantics: `inner_cv_platt_calibrated_probability`
- Task: `safe_secret_vs_exfiltration`
- Positive label: `exfiltration_intent`
- Activation feature: `selected_choice_window_layer_19`
- Outer folds: `4`
- Inner calibration folds: `3`
- Decision threshold: `0.5000`

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 1.0000 |
| Macro F1 | 1.0000 |
| Brier score | 0.0018 |
| Log loss | 0.0402 |
| Expected calibration error | 0.0392 |

## Calibration Bins

| Bin | Range | Examples | Mean Probability | Empirical Positive Rate | Absolute Gap |
|---:|---|---:|---:|---:|---:|
| 1 | [0.00, 0.10] | 239 | 0.0387 | 0.0000 | 0.0387 |
| 2 | [0.10, 0.20] | 1 | 0.1591 | 0.0000 | 0.1591 |
| 3 | [0.20, 0.30] | 0 | 0.0000 | 0.0000 | 0.0000 |
| 4 | [0.30, 0.40] | 0 | 0.0000 | 0.0000 | 0.0000 |
| 5 | [0.40, 0.50] | 0 | 0.0000 | 0.0000 | 0.0000 |
| 6 | [0.50, 0.60] | 0 | 0.0000 | 0.0000 | 0.0000 |
| 7 | [0.60, 0.70] | 1 | 0.6860 | 1.0000 | 0.3140 |
| 8 | [0.70, 0.80] | 0 | 0.0000 | 0.0000 | 0.0000 |
| 9 | [0.80, 0.90] | 2 | 0.8712 | 1.0000 | 0.1288 |
| 10 | [0.90, 1.00] | 237 | 0.9627 | 1.0000 | 0.0373 |
