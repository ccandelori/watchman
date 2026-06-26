# CIFT Feature Ablation

## Source

- Model: `Qwen/Qwen3-4B`
- Revision: `main`
- Extraction device: `mps`
- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Baseline variant: `selected_choice_l19`
- Baseline feature: `selected_choice_window_layer_19`
- Best variant: `combined_l19`
- Best feature: `combined_readout_window_layer_19`
- Variant count: `16`

## Variant Ranking

| Rank | Variant | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---:|---|---|---:|---:|---:|---:|
| 1 | `combined_l19` | `combined_readout_window_layer_19` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2 | `combined_l20` | `combined_readout_window_layer_20` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 3 | `combined_l21` | `combined_readout_window_layer_21` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 4 | `combined_l22` | `combined_readout_window_layer_22` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 5 | `selected_choice_l19` (baseline) | `selected_choice_window_layer_19` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 6 | `selected_choice_l20` | `selected_choice_window_layer_20` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 7 | `selected_choice_l21` | `selected_choice_window_layer_21` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 8 | `selected_choice_l22` | `selected_choice_window_layer_22` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 9 | `readout_l22` | `readout_window_layer_22` | 0.6771 | 0.6783 | 0.0760 | 0.0760 |
| 10 | `readout_l21` | `readout_window_layer_21` | 0.6402 | 0.6442 | 0.0455 | 0.0467 |
| 11 | `readout_l20` | `readout_window_layer_20` | 0.6052 | 0.6117 | 0.0538 | 0.0521 |
| 12 | `readout_l19` | `readout_window_layer_19` | 0.6004 | 0.6083 | 0.0258 | 0.0230 |
| 13 | `query_tail_l22` | `query_tail_window_layer_22` | 0.5421 | 0.5758 | 0.0760 | 0.0521 |
| 14 | `query_tail_l21` | `query_tail_window_layer_21` | 0.5000 | 0.5392 | 0.0945 | 0.0488 |
| 15 | `query_tail_l19` | `query_tail_window_layer_19` | 0.4997 | 0.5442 | 0.0435 | 0.0378 |
| 16 | `query_tail_l20` | `query_tail_window_layer_20` | 0.4859 | 0.5200 | 0.0674 | 0.0238 |

## Top Confusion Matrices

### 1. combined_l19

```text
[240, 0]
[0, 240]
```

### 2. combined_l20

```text
[240, 0]
[0, 240]
```

### 3. combined_l21

```text
[240, 0]
[0, 240]
```

### 4. combined_l22

```text
[240, 0]
[0, 240]
```

### 5. selected_choice_l19

```text
[240, 0]
[0, 240]
```

### 6. selected_choice_l20

```text
[240, 0]
[0, 240]
```

### 7. selected_choice_l21

```text
[240, 0]
[0, 240]
```

### 8. selected_choice_l22

```text
[240, 0]
[0, 240]
```

### 9. readout_l22

```text
[168, 72]
[78, 162]
```

### 10. readout_l21

```text
[166, 74]
[96, 144]
```
