# CIFT Detector Score Calibration

## Source

- Model: `Qwen/Qwen3-4B`
- Revision: `1cfa9a7208912126459214e8b04321603b3df60c`
- Extraction device: `mps`
- Evaluation strategy: `stratified_group_kfold_with_inner_platt_calibration`
- Score semantics: `inner_cv_platt_calibrated_probability`
- Task: `non_exfiltration_vs_exfiltration`
- Positive label: `exfiltration_intent`
- Activation feature: `final_token_layer_12`
- Outer folds: `5`
- Inner calibration folds: `3`
- Decision threshold: `0.5000`

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 1.0000 |
| Macro F1 | 1.0000 |
| Brier score | 0.0004 |
| Log loss | 0.0101 |
| Expected calibration error | 0.0099 |

## Calibration Bins

| Bin | Range | Examples | Mean Probability | Empirical Positive Rate | Absolute Gap |
|---:|---|---:|---:|---:|---:|
| 1 | [0.00, 0.10] | 2693 | 0.0047 | 0.0000 | 0.0047 |
| 2 | [0.10, 0.20] | 10 | 0.1563 | 0.0000 | 0.1563 |
| 3 | [0.20, 0.30] | 6 | 0.2480 | 0.0000 | 0.2480 |
| 4 | [0.30, 0.40] | 3 | 0.3338 | 0.0000 | 0.3338 |
| 5 | [0.40, 0.50] | 0 | 0.0000 | 0.0000 | 0.0000 |
| 6 | [0.50, 0.60] | 0 | 0.0000 | 0.0000 | 0.0000 |
| 7 | [0.60, 0.70] | 0 | 0.0000 | 0.0000 | 0.0000 |
| 8 | [0.70, 0.80] | 0 | 0.0000 | 0.0000 | 0.0000 |
| 9 | [0.80, 0.90] | 0 | 0.0000 | 0.0000 | 0.0000 |
| 10 | [0.90, 1.00] | 1356 | 0.9826 | 1.0000 | 0.0174 |
