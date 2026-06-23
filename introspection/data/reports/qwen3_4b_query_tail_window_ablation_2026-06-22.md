# CIFT Feature Ablation

## Source

- Model: `Qwen/Qwen3-4B`
- Revision: `main`
- Extraction device: `mps`
- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Baseline variant: `selected_choice_concat_19_22`
- Baseline feature: `concat(selected_choice_window_layer_19,selected_choice_window_layer_20,selected_choice_window_layer_21,selected_choice_window_layer_22)`
- Best variant: `selected_choice_concat_19_22`
- Best feature: `concat(selected_choice_window_layer_19,selected_choice_window_layer_20,selected_choice_window_layer_21,selected_choice_window_layer_22)`
- Variant count: `3`

## Variant Ranking

| Rank | Variant | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---:|---|---|---:|---:|---:|---:|
| 1 | `selected_choice_concat_19_22` (baseline) | `concat(selected_choice_window_layer_19,selected_choice_window_layer_20,selected_choice_window_layer_21,selected_choice_window_layer_22)` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2 | `readout_window_concat_19_22` | `concat(readout_window_layer_19,readout_window_layer_20,readout_window_layer_21,readout_window_layer_22)` | 0.6856 | 0.6908 | 0.0420 | 0.0426 |
| 3 | `query_tail_concat_19_22` | `concat(query_tail_window_layer_19,query_tail_window_layer_20,query_tail_window_layer_21,query_tail_window_layer_22)` | 0.5589 | 0.5883 | 0.0277 | 0.0421 |

## Top Confusion Matrices

### 1. selected_choice_concat_19_22

```text
[240, 0]
[0, 240]
```

### 2. readout_window_concat_19_22

```text
[158, 82]
[67, 173]
```

### 3. query_tail_concat_19_22

```text
[163, 77]
[124, 116]
```
