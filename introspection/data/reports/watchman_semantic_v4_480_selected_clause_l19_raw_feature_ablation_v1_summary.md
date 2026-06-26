# CIFT Feature Ablation

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `cpu`
- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Baseline variant: `selected_choice_l19`
- Baseline feature: `selected_choice_window_layer_19`
- Best variant: `selected_choice_l20`
- Best feature: `selected_choice_window_layer_20`
- Variant count: `16`

## Variant Ranking

| Rank | Variant | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---:|---|---|---:|---:|---:|---:|
| 1 | `selected_choice_l20` | `selected_choice_window_layer_20` | 0.9933 | 0.9933 | 0.0097 | 0.0097 |
| 2 | `selected_choice_l21` | `selected_choice_window_layer_21` | 0.9933 | 0.9933 | 0.0097 | 0.0097 |
| 3 | `selected_choice_l19` (baseline) | `selected_choice_window_layer_19` | 0.9908 | 0.9908 | 0.0146 | 0.0145 |
| 4 | `selected_choice_l22` | `selected_choice_window_layer_22` | 0.9875 | 0.9875 | 0.0083 | 0.0083 |
| 5 | `combined_l19` | `combined_readout_window_layer_19` | 0.9169 | 0.9175 | 0.0207 | 0.0203 |
| 6 | `combined_l20` | `combined_readout_window_layer_20` | 0.9162 | 0.9167 | 0.0279 | 0.0275 |
| 7 | `combined_l22` | `combined_readout_window_layer_22` | 0.8656 | 0.8692 | 0.0715 | 0.0666 |
| 8 | `combined_l21` | `combined_readout_window_layer_21` | 0.8547 | 0.8592 | 0.0747 | 0.0676 |
| 9 | `readout_l21` | `readout_window_layer_21` | 0.5194 | 0.5500 | 0.0505 | 0.0243 |
| 10 | `readout_l19` | `readout_window_layer_19` | 0.5179 | 0.5342 | 0.0222 | 0.0172 |
| 11 | `readout_l22` | `readout_window_layer_22` | 0.5059 | 0.5442 | 0.0584 | 0.0231 |
| 12 | `readout_l20` | `readout_window_layer_20` | 0.4867 | 0.5233 | 0.0238 | 0.0094 |
| 13 | `query_tail_l20` | `query_tail_window_layer_20` | 0.4436 | 0.5083 | 0.0837 | 0.0129 |
| 14 | `query_tail_l19` | `query_tail_window_layer_19` | 0.4409 | 0.5075 | 0.0688 | 0.0100 |
| 15 | `query_tail_l21` | `query_tail_window_layer_21` | 0.4119 | 0.5092 | 0.0724 | 0.0130 |
| 16 | `query_tail_l22` | `query_tail_window_layer_22` | 0.4013 | 0.5017 | 0.0833 | 0.0033 |

## Top Confusion Matrices

### 1. selected_choice_l20

```text
[238, 2]
[1, 239]
```

### 2. selected_choice_l21

```text
[238, 2]
[1, 239]
```

### 3. selected_choice_l19

```text
[237, 3]
[1, 239]
```

### 4. selected_choice_l22

```text
[238, 2]
[4, 236]
```

### 5. combined_l19

```text
[225, 15]
[23, 217]
```

### 6. combined_l20

```text
[219, 21]
[19, 221]
```

### 7. combined_l22

```text
[204, 36]
[23, 217]
```

### 8. combined_l21

```text
[203, 37]
[26, 214]
```

### 9. readout_l21

```text
[121, 119]
[99, 141]
```

### 10. readout_l19

```text
[131, 109]
[114, 126]
```
