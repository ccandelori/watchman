# CIFT Detector Score Calibration

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `cpu`
- Evaluation strategy: `stratified_group_kfold_with_inner_platt_calibration`
- Score semantics: `inner_cv_platt_calibrated_probability`
- Task: `safe_secret_vs_exfiltration`
- Positive label: `exfiltration_intent`
- Activation feature: `selected_choice_window_layer_19`
- Outer folds: `5`
- Inner calibration folds: `3`
- Decision threshold: `0.5000`

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.9917 |
| Macro F1 | 0.9917 |
| Brier score | 0.0080 |
| Log loss | 0.0605 |
| Expected calibration error | 0.0419 |

## Calibration Bins

| Bin | Range | Examples | Mean Probability | Empirical Positive Rate | Absolute Gap |
|---:|---|---:|---:|---:|---:|
| 1 | [0.00, 0.10] | 231 | 0.0429 | 0.0087 | 0.0342 |
| 2 | [0.10, 0.20] | 6 | 0.1326 | 0.0000 | 0.1326 |
| 3 | [0.20, 0.30] | 3 | 0.2497 | 0.0000 | 0.2497 |
| 4 | [0.30, 0.40] | 1 | 0.3434 | 0.0000 | 0.3434 |
| 5 | [0.40, 0.50] | 1 | 0.4305 | 1.0000 | 0.5695 |
| 6 | [0.50, 0.60] | 0 | 0.0000 | 0.0000 | 0.0000 |
| 7 | [0.60, 0.70] | 2 | 0.6720 | 0.5000 | 0.1720 |
| 8 | [0.70, 0.80] | 0 | 0.0000 | 0.0000 | 0.0000 |
| 9 | [0.80, 0.90] | 6 | 0.8753 | 1.0000 | 0.1247 |
| 10 | [0.90, 1.00] | 230 | 0.9623 | 1.0000 | 0.0377 |
