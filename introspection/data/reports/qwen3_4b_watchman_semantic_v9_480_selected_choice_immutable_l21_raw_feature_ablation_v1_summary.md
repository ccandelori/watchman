# CIFT Feature Ablation

## Source

- Model: `Qwen/Qwen3-4B`
- Revision: `1cfa9a7208912126459214e8b04321603b3df60c`
- Extraction device: `mps`
- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Baseline variant: `selected_choice_l21`
- Baseline feature: `selected_choice_window_layer_21`
- Best variant: `combined_l20`
- Best feature: `combined_readout_window_layer_20`
- Variant count: `12`

## Variant Ranking

| Rank | Variant | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---:|---|---|---:|---:|---:|---:|
| 1 | `combined_l20` | `combined_readout_window_layer_20` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2 | `combined_l21` | `combined_readout_window_layer_21` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 3 | `combined_l22` | `combined_readout_window_layer_22` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 4 | `selected_choice_l19` | `selected_choice_window_layer_19` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 5 | `selected_choice_l20` | `selected_choice_window_layer_20` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 6 | `selected_choice_l21` (baseline) | `selected_choice_window_layer_21` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 7 | `selected_choice_l22` | `selected_choice_window_layer_22` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 8 | `combined_l19` | `combined_readout_window_layer_19` | 0.9975 | 0.9975 | 0.0050 | 0.0050 |
| 9 | `readout_l22` | `readout_window_layer_22` | 0.6558 | 0.6592 | 0.0405 | 0.0384 |
| 10 | `readout_l21` | `readout_window_layer_21` | 0.6193 | 0.6225 | 0.0348 | 0.0348 |
| 11 | `readout_l19` | `readout_window_layer_19` | 0.6092 | 0.6117 | 0.0461 | 0.0452 |
| 12 | `readout_l20` | `readout_window_layer_20` | 0.6044 | 0.6050 | 0.0610 | 0.0605 |

## Top Confusion Matrices

### 1. combined_l20

```text
[240, 0]
[0, 240]
```

### 2. combined_l21

```text
[240, 0]
[0, 240]
```

### 3. combined_l22

```text
[240, 0]
[0, 240]
```

### 4. selected_choice_l19

```text
[240, 0]
[0, 240]
```

### 5. selected_choice_l20

```text
[240, 0]
[0, 240]
```

### 6. selected_choice_l21

```text
[240, 0]
[0, 240]
```

### 7. selected_choice_l22

```text
[240, 0]
[0, 240]
```

### 8. combined_l19

```text
[240, 0]
[1, 239]
```

### 9. readout_l22

```text
[156, 84]
[81, 159]
```

### 10. readout_l21

```text
[142, 98]
[84, 156]
```
