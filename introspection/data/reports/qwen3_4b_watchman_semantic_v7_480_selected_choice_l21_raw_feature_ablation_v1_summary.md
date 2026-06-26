# CIFT Feature Ablation

## Source

- Model: `Qwen/Qwen3-4B`
- Revision: `main`
- Extraction device: `mps`
- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Baseline variant: `selected_choice_l21`
- Baseline feature: `selected_choice_window_layer_21`
- Best variant: `combined_l20`
- Best feature: `combined_readout_window_layer_20`
- Variant count: `16`

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
| 9 | `readout_l22` | `readout_window_layer_22` | 0.6592 | 0.6625 | 0.0368 | 0.0349 |
| 10 | `readout_l19` | `readout_window_layer_19` | 0.6228 | 0.6250 | 0.0460 | 0.0448 |
| 11 | `readout_l21` | `readout_window_layer_21` | 0.6197 | 0.6233 | 0.0402 | 0.0404 |
| 12 | `readout_l20` | `readout_window_layer_20` | 0.6084 | 0.6092 | 0.0655 | 0.0652 |
| 13 | `query_tail_l20` | `query_tail_window_layer_20` | 0.5235 | 0.5325 | 0.0188 | 0.0245 |
| 14 | `query_tail_l21` | `query_tail_window_layer_21` | 0.5130 | 0.5375 | 0.0414 | 0.0306 |
| 15 | `query_tail_l22` | `query_tail_window_layer_22` | 0.5089 | 0.5508 | 0.0620 | 0.0326 |
| 16 | `query_tail_l19` | `query_tail_window_layer_19` | 0.4850 | 0.5158 | 0.0444 | 0.0159 |

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
[158, 82]
[81, 159]
```

### 10. readout_l19

```text
[147, 93]
[87, 153]
```
