# Binary Layer Sweep

## Source

- Model: `Qwen/Qwen3-4B`
- Revision: `1cfa9a7208912126459214e8b04321603b3df60c`
- Extraction device: `mps`
- Evaluation strategy: `stratified_group_kfold`
- Task: `non_exfiltration_vs_exfiltration`
- Reference feature: `query_tail_window_layer_21`
- Best feature: `final_token_layer_08`
- Feature count: `35`
- Invalid feature count: `1`

## Feature Ranking

| Rank | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---:|---|---:|---:|---:|---:|
| 1 | `final_token_layer_08` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2 | `final_token_layer_12` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 3 | `final_token_layer_16` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 4 | `final_token_layer_20` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 5 | `final_token_layer_21` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 6 | `final_token_layer_24` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 7 | `final_token_layer_28` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 8 | `final_token_layer_32` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 9 | `final_token_layer_35` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 10 | `mean_pool_layer_08` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 11 | `mean_pool_layer_12` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 12 | `mean_pool_layer_16` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 13 | `mean_pool_layer_20` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 14 | `mean_pool_layer_21` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 15 | `mean_pool_layer_24` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 16 | `mean_pool_layer_28` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 17 | `mean_pool_layer_32` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 18 | `query_tail_window_layer_12` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 19 | `query_tail_window_layer_16` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 20 | `query_tail_window_layer_20` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 21 | `query_tail_window_layer_21` (reference) | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 22 | `query_tail_window_layer_24` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 23 | `readout_window_layer_08` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 24 | `readout_window_layer_12` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 25 | `readout_window_layer_16` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 26 | `readout_window_layer_20` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 27 | `readout_window_layer_24` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 28 | `readout_window_layer_28` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 29 | `readout_window_layer_32` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 30 | `readout_window_layer_35` | 0.9998 | 0.9998 | 0.0005 | 0.0004 |
| 31 | `readout_window_layer_21` | 0.9996 | 0.9997 | 0.0007 | 0.0006 |
| 32 | `query_tail_window_layer_08` | 0.9908 | 0.9915 | 0.0185 | 0.0169 |
| 33 | `query_tail_window_layer_35` | 0.9901 | 0.9909 | 0.0198 | 0.0182 |
| 34 | `query_tail_window_layer_32` | 0.9845 | 0.9856 | 0.0309 | 0.0288 |
| 35 | `query_tail_window_layer_28` | 0.9826 | 0.9837 | 0.0348 | 0.0326 |

## Top Confusion Matrices

### 1. final_token_layer_08

```text
[1276, 0]
[0, 2552]
```

### 2. final_token_layer_12

```text
[1276, 0]
[0, 2552]
```

### 3. final_token_layer_16

```text
[1276, 0]
[0, 2552]
```

### 4. final_token_layer_20

```text
[1276, 0]
[0, 2552]
```

### 5. final_token_layer_21

```text
[1276, 0]
[0, 2552]
```

### 6. final_token_layer_24

```text
[1276, 0]
[0, 2552]
```

### 7. final_token_layer_28

```text
[1276, 0]
[0, 2552]
```

### 8. final_token_layer_32

```text
[1276, 0]
[0, 2552]
```

### 9. final_token_layer_35

```text
[1276, 0]
[0, 2552]
```

### 10. mean_pool_layer_08

```text
[1276, 0]
[0, 2552]
```

## Invalid Features

| Feature | Reason |
|---|---|
| `mean_pool_layer_35` | 3863 non-finite values across 2323 rows |
