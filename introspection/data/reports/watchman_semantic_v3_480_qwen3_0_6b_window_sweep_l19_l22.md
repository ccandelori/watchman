# Binary Layer Sweep

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `mps`
- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Reference feature: `selected_choice_window_layer_21`
- Best feature: `selected_choice_window_layer_19`
- Feature count: `16`

## Feature Ranking

| Rank | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---:|---|---:|---:|---:|---:|
| 1 | `selected_choice_window_layer_19` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2 | `selected_choice_window_layer_20` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 3 | `selected_choice_window_layer_21` (reference) | 0.9975 | 0.9975 | 0.0050 | 0.0050 |
| 4 | `selected_choice_window_layer_22` | 0.9908 | 0.9908 | 0.0049 | 0.0049 |
| 5 | `combined_readout_window_layer_19` | 0.9489 | 0.9492 | 0.0305 | 0.0303 |
| 6 | `combined_readout_window_layer_20` | 0.9104 | 0.9117 | 0.0488 | 0.0468 |
| 7 | `combined_readout_window_layer_21` | 0.8991 | 0.9025 | 0.0887 | 0.0825 |
| 8 | `combined_readout_window_layer_22` | 0.8971 | 0.8992 | 0.0616 | 0.0592 |
| 9 | `readout_window_layer_19` | 0.5524 | 0.5558 | 0.0144 | 0.0133 |
| 10 | `readout_window_layer_22` | 0.5361 | 0.5458 | 0.0241 | 0.0124 |
| 11 | `readout_window_layer_20` | 0.5341 | 0.5392 | 0.0199 | 0.0210 |
| 12 | `readout_window_layer_21` | 0.5322 | 0.5367 | 0.0235 | 0.0227 |
| 13 | `query_tail_window_layer_22` | 0.4982 | 0.5058 | 0.0118 | 0.0133 |
| 14 | `query_tail_window_layer_19` | 0.4765 | 0.5167 | 0.0440 | 0.0139 |
| 15 | `query_tail_window_layer_20` | 0.4341 | 0.5017 | 0.0752 | 0.0033 |
| 16 | `query_tail_window_layer_21` | 0.4290 | 0.5142 | 0.0778 | 0.0148 |

## Top Confusion Matrices

### 1. selected_choice_window_layer_19

```text
[240, 0]
[0, 240]
```

### 2. selected_choice_window_layer_20

```text
[240, 0]
[0, 240]
```

### 3. selected_choice_window_layer_21

```text
[239, 1]
[0, 240]
```

### 4. selected_choice_window_layer_22

```text
[238, 2]
[2, 238]
```

### 5. combined_readout_window_layer_19

```text
[229, 11]
[13, 227]
```

### 6. combined_readout_window_layer_20

```text
[216, 24]
[18, 222]
```

### 7. combined_readout_window_layer_21

```text
[210, 30]
[16, 224]
```

### 8. combined_readout_window_layer_22

```text
[213, 27]
[23, 217]
```

### 9. readout_window_layer_19

```text
[146, 94]
[119, 121]
```

### 10. readout_window_layer_22

```text
[99, 141]
[78, 162]
```
