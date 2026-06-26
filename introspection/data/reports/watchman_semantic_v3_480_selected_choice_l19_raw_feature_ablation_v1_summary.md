# CIFT Feature Ablation

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `mps`
- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Baseline variant: `selected_choice_l19`
- Baseline feature: `selected_choice_window_layer_19`
- Best variant: `selected_choice_l19`
- Best feature: `selected_choice_window_layer_19`
- Variant count: `7`

## Variant Ranking

| Rank | Variant | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---:|---|---|---:|---:|---:|---:|
| 1 | `selected_choice_l19` (baseline) | `selected_choice_window_layer_19` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2 | `selected_choice_l20` | `selected_choice_window_layer_20` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 3 | `selected_choice_l21` | `selected_choice_window_layer_21` | 0.9958 | 0.9958 | 0.0042 | 0.0042 |
| 4 | `selected_choice_l22` | `selected_choice_window_layer_22` | 0.9917 | 0.9917 | 0.0059 | 0.0059 |
| 5 | `combined_readout_l19` | `combined_readout_window_layer_19` | 0.9520 | 0.9521 | 0.0150 | 0.0149 |
| 6 | `readout_l19` | `readout_window_layer_19` | 0.5649 | 0.5687 | 0.0210 | 0.0180 |
| 7 | `query_tail_l19` | `query_tail_window_layer_19` | 0.4340 | 0.5146 | 0.0852 | 0.0171 |

## Top Confusion Matrices

### 1. selected_choice_l19

```text
[240, 0]
[0, 240]
```

### 2. selected_choice_l20

```text
[240, 0]
[0, 240]
```

### 3. selected_choice_l21

```text
[238, 2]
[0, 240]
```

### 4. selected_choice_l22

```text
[238, 2]
[2, 238]
```

### 5. combined_readout_l19

```text
[226, 14]
[9, 231]
```

### 6. readout_l19

```text
[151, 89]
[118, 122]
```

### 7. query_tail_l19

```text
[133, 107]
[126, 114]
```
