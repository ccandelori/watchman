# Binary Layer Sweep

## Source

- Model: `Qwen/Qwen3-4B`
- Revision: `1cfa9a7208912126459214e8b04321603b3df60c`
- Extraction device: `mps`
- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
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
| 18 | `query_tail_window_layer_08` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 19 | `query_tail_window_layer_12` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 20 | `query_tail_window_layer_16` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 21 | `query_tail_window_layer_20` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 22 | `query_tail_window_layer_21` (reference) | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 23 | `query_tail_window_layer_24` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 24 | `query_tail_window_layer_28` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 25 | `query_tail_window_layer_32` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 26 | `query_tail_window_layer_35` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 27 | `readout_window_layer_08` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 28 | `readout_window_layer_12` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 29 | `readout_window_layer_16` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 30 | `readout_window_layer_20` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 31 | `readout_window_layer_21` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 32 | `readout_window_layer_24` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 33 | `readout_window_layer_28` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 34 | `readout_window_layer_32` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 35 | `readout_window_layer_35` | 0.9997 | 0.9997 | 0.0006 | 0.0006 |

## Top Confusion Matrices

### 1. final_token_layer_08

```text
[1276, 0]
[0, 1276]
```

### 2. final_token_layer_12

```text
[1276, 0]
[0, 1276]
```

### 3. final_token_layer_16

```text
[1276, 0]
[0, 1276]
```

### 4. final_token_layer_20

```text
[1276, 0]
[0, 1276]
```

### 5. final_token_layer_21

```text
[1276, 0]
[0, 1276]
```

### 6. final_token_layer_24

```text
[1276, 0]
[0, 1276]
```

### 7. final_token_layer_28

```text
[1276, 0]
[0, 1276]
```

### 8. final_token_layer_32

```text
[1276, 0]
[0, 1276]
```

### 9. final_token_layer_35

```text
[1276, 0]
[0, 1276]
```

### 10. mean_pool_layer_08

```text
[1276, 0]
[0, 1276]
```

## Invalid Features

| Feature | Reason |
|---|---|
| `mean_pool_layer_35` | 3863 non-finite values across 2323 rows |
