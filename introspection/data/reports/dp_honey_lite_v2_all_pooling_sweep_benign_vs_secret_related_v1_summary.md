# Binary Layer Sweep

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `cpu`
- Evaluation strategy: `stratified_group_kfold`
- Task: `benign_vs_secret_related`
- Reference feature: `readout_window_layer_11`
- Best feature: `final_token_layer_02`
- Feature count: `87`

## Feature Ranking

| Rank | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---:|---|---:|---:|---:|---:|
| 1 | `final_token_layer_02` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2 | `final_token_layer_03` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 3 | `final_token_layer_04` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 4 | `final_token_layer_05` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 5 | `final_token_layer_11` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 6 | `final_token_layer_12` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 7 | `final_token_layer_18` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 8 | `final_token_layer_19` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 9 | `final_token_layer_20` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 10 | `final_token_layer_21` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 11 | `final_token_layer_22` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 12 | `final_token_layer_23` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 13 | `final_token_layer_24` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 14 | `final_token_layer_25` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 15 | `final_token_layer_26` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 16 | `final_token_layer_27` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 17 | `final_token_layer_28` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 18 | `mean_pool_layer_00` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 19 | `mean_pool_layer_02` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 20 | `mean_pool_layer_03` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 21 | `mean_pool_layer_04` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 22 | `mean_pool_layer_07` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 23 | `mean_pool_layer_08` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 24 | `mean_pool_layer_09` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 25 | `mean_pool_layer_10` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 26 | `mean_pool_layer_11` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 27 | `mean_pool_layer_12` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 28 | `mean_pool_layer_13` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 29 | `mean_pool_layer_14` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 30 | `mean_pool_layer_15` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 31 | `mean_pool_layer_16` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 32 | `mean_pool_layer_17` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 33 | `mean_pool_layer_18` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 34 | `mean_pool_layer_19` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 35 | `mean_pool_layer_20` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 36 | `mean_pool_layer_21` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 37 | `mean_pool_layer_22` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 38 | `mean_pool_layer_23` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 39 | `mean_pool_layer_24` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 40 | `mean_pool_layer_25` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 41 | `mean_pool_layer_26` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 42 | `mean_pool_layer_27` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 43 | `mean_pool_layer_28` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 44 | `readout_window_layer_01` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 45 | `readout_window_layer_02` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 46 | `readout_window_layer_03` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 47 | `readout_window_layer_04` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 48 | `readout_window_layer_05` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 49 | `readout_window_layer_06` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 50 | `readout_window_layer_07` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 51 | `readout_window_layer_08` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 52 | `readout_window_layer_09` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 53 | `readout_window_layer_10` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 54 | `readout_window_layer_11` (reference) | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 55 | `readout_window_layer_12` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 56 | `readout_window_layer_13` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 57 | `readout_window_layer_14` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 58 | `readout_window_layer_16` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 59 | `readout_window_layer_17` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 60 | `readout_window_layer_18` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 61 | `readout_window_layer_19` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 62 | `readout_window_layer_20` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 63 | `readout_window_layer_22` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 64 | `readout_window_layer_23` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 65 | `readout_window_layer_24` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 66 | `readout_window_layer_25` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 67 | `readout_window_layer_26` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 68 | `readout_window_layer_27` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 69 | `readout_window_layer_28` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 70 | `mean_pool_layer_06` | 0.9954 | 0.9958 | 0.0092 | 0.0083 |
| 71 | `final_token_layer_16` | 0.9954 | 0.9958 | 0.0092 | 0.0083 |
| 72 | `final_token_layer_17` | 0.9954 | 0.9958 | 0.0092 | 0.0083 |
| 73 | `mean_pool_layer_01` | 0.9954 | 0.9958 | 0.0092 | 0.0083 |
| 74 | `readout_window_layer_21` | 0.9954 | 0.9958 | 0.0092 | 0.0083 |
| 75 | `final_token_layer_13` | 0.9909 | 0.9917 | 0.0182 | 0.0167 |
| 76 | `final_token_layer_14` | 0.9909 | 0.9917 | 0.0182 | 0.0167 |
| 77 | `final_token_layer_15` | 0.9909 | 0.9917 | 0.0182 | 0.0167 |
| 78 | `final_token_layer_01` | 0.9903 | 0.9917 | 0.0194 | 0.0167 |
| 79 | `readout_window_layer_15` | 0.9865 | 0.9875 | 0.0270 | 0.0250 |
| 80 | `final_token_layer_06` | 0.9822 | 0.9833 | 0.0356 | 0.0333 |
| 81 | `final_token_layer_07` | 0.9822 | 0.9833 | 0.0356 | 0.0333 |
| 82 | `final_token_layer_08` | 0.9822 | 0.9833 | 0.0356 | 0.0333 |
| 83 | `final_token_layer_09` | 0.9822 | 0.9833 | 0.0356 | 0.0333 |
| 84 | `final_token_layer_10` | 0.9822 | 0.9833 | 0.0356 | 0.0333 |
| 85 | `readout_window_layer_00` | 0.9822 | 0.9833 | 0.0356 | 0.0333 |
| 86 | `mean_pool_layer_05` | 0.9780 | 0.9792 | 0.0440 | 0.0417 |
| 87 | `final_token_layer_00` | 0.2500 | 0.3333 | 0.0000 | 0.0000 |

## Top Confusion Matrices

### 1. final_token_layer_02

```text
[80, 0]
[0, 160]
```

### 2. final_token_layer_03

```text
[80, 0]
[0, 160]
```

### 3. final_token_layer_04

```text
[80, 0]
[0, 160]
```

### 4. final_token_layer_05

```text
[80, 0]
[0, 160]
```

### 5. final_token_layer_11

```text
[80, 0]
[0, 160]
```

### 6. final_token_layer_12

```text
[80, 0]
[0, 160]
```

### 7. final_token_layer_18

```text
[80, 0]
[0, 160]
```

### 8. final_token_layer_19

```text
[80, 0]
[0, 160]
```

### 9. final_token_layer_20

```text
[80, 0]
[0, 160]
```

### 10. final_token_layer_21

```text
[80, 0]
[0, 160]
```
