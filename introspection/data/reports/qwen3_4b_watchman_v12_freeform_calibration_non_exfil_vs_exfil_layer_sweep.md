# Binary Layer Sweep

## Source

- Model: `Qwen/Qwen3-4B`
- Revision: `1cfa9a7208912126459214e8b04321603b3df60c`
- Extraction device: `mps`
- Evaluation strategy: `stratified_group_kfold`
- Task: `non_exfiltration_vs_exfiltration`
- Reference feature: `query_tail_window_layer_21`
- Best feature: `final_token_layer_12`
- Feature count: `35`
- Invalid feature count: `1`

## Feature Ranking

| Rank | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---:|---|---:|---:|---:|---:|
| 1 | `final_token_layer_12` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2 | `final_token_layer_16` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 3 | `final_token_layer_20` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 4 | `final_token_layer_21` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 5 | `final_token_layer_24` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 6 | `final_token_layer_28` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 7 | `final_token_layer_32` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 8 | `final_token_layer_35` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 9 | `mean_pool_layer_12` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 10 | `mean_pool_layer_16` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 11 | `mean_pool_layer_20` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 12 | `mean_pool_layer_21` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 13 | `mean_pool_layer_28` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 14 | `mean_pool_layer_32` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 15 | `query_tail_window_layer_12` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 16 | `query_tail_window_layer_16` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 17 | `query_tail_window_layer_20` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 18 | `query_tail_window_layer_21` (reference) | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 19 | `query_tail_window_layer_24` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 20 | `readout_window_layer_12` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 21 | `readout_window_layer_16` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 22 | `readout_window_layer_20` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 23 | `readout_window_layer_21` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 24 | `readout_window_layer_28` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 25 | `readout_window_layer_32` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 26 | `readout_window_layer_35` | 0.9997 | 0.9997 | 0.0006 | 0.0005 |
| 27 | `mean_pool_layer_24` | 0.9942 | 0.9947 | 0.0117 | 0.0106 |
| 28 | `query_tail_window_layer_08` | 0.9942 | 0.9947 | 0.0117 | 0.0106 |
| 29 | `readout_window_layer_24` | 0.9942 | 0.9947 | 0.0117 | 0.0106 |
| 30 | `final_token_layer_08` | 0.9937 | 0.9943 | 0.0126 | 0.0115 |
| 31 | `query_tail_window_layer_35` | 0.9934 | 0.9940 | 0.0132 | 0.0120 |
| 32 | `query_tail_window_layer_28` | 0.9921 | 0.9928 | 0.0157 | 0.0143 |
| 33 | `readout_window_layer_08` | 0.9899 | 0.9908 | 0.0202 | 0.0185 |
| 34 | `mean_pool_layer_08` | 0.9885 | 0.9894 | 0.0229 | 0.0211 |
| 35 | `query_tail_window_layer_32` | 0.9855 | 0.9865 | 0.0290 | 0.0269 |

## Top Confusion Matrices

### 1. final_token_layer_12

```text
[1356, 0]
[0, 2712]
```

### 2. final_token_layer_16

```text
[1356, 0]
[0, 2712]
```

### 3. final_token_layer_20

```text
[1356, 0]
[0, 2712]
```

### 4. final_token_layer_21

```text
[1356, 0]
[0, 2712]
```

### 5. final_token_layer_24

```text
[1356, 0]
[0, 2712]
```

### 6. final_token_layer_28

```text
[1356, 0]
[0, 2712]
```

### 7. final_token_layer_32

```text
[1356, 0]
[0, 2712]
```

### 8. final_token_layer_35

```text
[1356, 0]
[0, 2712]
```

### 9. mean_pool_layer_12

```text
[1356, 0]
[0, 2712]
```

### 10. mean_pool_layer_16

```text
[1356, 0]
[0, 2712]
```

## Invalid Features

| Feature | Reason |
|---|---|
| `mean_pool_layer_35` | 3883 non-finite values across 2343 rows |
