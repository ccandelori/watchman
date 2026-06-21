# Binary Layer Sweep

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `cpu`
- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Reference feature: `readout_window_layer_11`
- Best feature: `final_token_layer_04`
- Feature count: `87`

## Feature Ranking

| Rank | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---:|---|---:|---:|---:|---:|
| 1 | `final_token_layer_04` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2 | `final_token_layer_06` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 3 | `final_token_layer_07` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 4 | `final_token_layer_08` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 5 | `final_token_layer_10` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 6 | `final_token_layer_11` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 7 | `final_token_layer_12` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 8 | `final_token_layer_13` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 9 | `final_token_layer_14` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 10 | `final_token_layer_15` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 11 | `final_token_layer_16` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 12 | `final_token_layer_17` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 13 | `final_token_layer_18` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 14 | `final_token_layer_19` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 15 | `final_token_layer_20` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 16 | `final_token_layer_21` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 17 | `final_token_layer_22` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 18 | `final_token_layer_23` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 19 | `final_token_layer_24` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 20 | `final_token_layer_25` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 21 | `final_token_layer_26` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 22 | `final_token_layer_27` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 23 | `final_token_layer_28` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 24 | `mean_pool_layer_06` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 25 | `mean_pool_layer_08` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 26 | `mean_pool_layer_09` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 27 | `mean_pool_layer_11` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 28 | `mean_pool_layer_12` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 29 | `mean_pool_layer_13` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 30 | `mean_pool_layer_14` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 31 | `mean_pool_layer_15` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 32 | `mean_pool_layer_16` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 33 | `mean_pool_layer_17` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 34 | `mean_pool_layer_18` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 35 | `mean_pool_layer_19` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 36 | `mean_pool_layer_20` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 37 | `mean_pool_layer_21` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 38 | `mean_pool_layer_22` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 39 | `mean_pool_layer_23` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 40 | `mean_pool_layer_24` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 41 | `mean_pool_layer_25` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 42 | `mean_pool_layer_26` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 43 | `mean_pool_layer_27` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 44 | `mean_pool_layer_28` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 45 | `readout_window_layer_09` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 46 | `readout_window_layer_11` (reference) | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 47 | `readout_window_layer_13` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 48 | `readout_window_layer_14` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 49 | `readout_window_layer_15` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 50 | `readout_window_layer_16` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 51 | `readout_window_layer_17` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 52 | `readout_window_layer_18` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 53 | `readout_window_layer_20` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 54 | `readout_window_layer_21` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 55 | `final_token_layer_05` | 0.9937 | 0.9938 | 0.0125 | 0.0125 |
| 56 | `readout_window_layer_12` | 0.9937 | 0.9938 | 0.0125 | 0.0125 |
| 57 | `readout_window_layer_27` | 0.9875 | 0.9875 | 0.0250 | 0.0250 |
| 58 | `readout_window_layer_08` | 0.9875 | 0.9875 | 0.0153 | 0.0153 |
| 59 | `readout_window_layer_10` | 0.9875 | 0.9875 | 0.0153 | 0.0153 |
| 60 | `readout_window_layer_23` | 0.9875 | 0.9875 | 0.0153 | 0.0153 |
| 61 | `mean_pool_layer_07` | 0.9875 | 0.9875 | 0.0251 | 0.0250 |
| 62 | `mean_pool_layer_10` | 0.9875 | 0.9875 | 0.0251 | 0.0250 |
| 63 | `readout_window_layer_22` | 0.9812 | 0.9812 | 0.0153 | 0.0153 |
| 64 | `readout_window_layer_28` | 0.9812 | 0.9812 | 0.0153 | 0.0153 |
| 65 | `final_token_layer_09` | 0.9811 | 0.9812 | 0.0378 | 0.0375 |
| 66 | `mean_pool_layer_01` | 0.9811 | 0.9812 | 0.0378 | 0.0375 |
| 67 | `readout_window_layer_25` | 0.9750 | 0.9750 | 0.0125 | 0.0125 |
| 68 | `readout_window_layer_26` | 0.9749 | 0.9750 | 0.0235 | 0.0234 |
| 69 | `mean_pool_layer_02` | 0.9748 | 0.9750 | 0.0368 | 0.0364 |
| 70 | `mean_pool_layer_03` | 0.9746 | 0.9750 | 0.0508 | 0.0500 |
| 71 | `mean_pool_layer_04` | 0.9746 | 0.9750 | 0.0508 | 0.0500 |
| 72 | `mean_pool_layer_05` | 0.9746 | 0.9750 | 0.0508 | 0.0500 |
| 73 | `readout_window_layer_19` | 0.9746 | 0.9750 | 0.0508 | 0.0500 |
| 74 | `readout_window_layer_24` | 0.9687 | 0.9688 | 0.0198 | 0.0198 |
| 75 | `mean_pool_layer_00` | 0.9683 | 0.9688 | 0.0492 | 0.0484 |
| 76 | `final_token_layer_03` | 0.9561 | 0.9563 | 0.0253 | 0.0250 |
| 77 | `readout_window_layer_06` | 0.9375 | 0.9375 | 0.0280 | 0.0280 |
| 78 | `final_token_layer_02` | 0.9243 | 0.9250 | 0.0430 | 0.0424 |
| 79 | `readout_window_layer_07` | 0.9057 | 0.9062 | 0.0957 | 0.0948 |
| 80 | `readout_window_layer_04` | 0.8996 | 0.9000 | 0.0499 | 0.0500 |
| 81 | `final_token_layer_01` | 0.8929 | 0.8938 | 0.0471 | 0.0468 |
| 82 | `readout_window_layer_05` | 0.8799 | 0.8812 | 0.0247 | 0.0234 |
| 83 | `readout_window_layer_03` | 0.8498 | 0.8500 | 0.0750 | 0.0750 |
| 84 | `readout_window_layer_02` | 0.7612 | 0.7625 | 0.0684 | 0.0673 |
| 85 | `readout_window_layer_01` | 0.7487 | 0.7500 | 0.0535 | 0.0523 |
| 86 | `readout_window_layer_00` | 0.7237 | 0.7250 | 0.0559 | 0.0538 |
| 87 | `final_token_layer_00` | 0.3333 | 0.5000 | 0.0000 | 0.0000 |

## Top Confusion Matrices

### 1. final_token_layer_04

```text
[80, 0]
[0, 80]
```

### 2. final_token_layer_06

```text
[80, 0]
[0, 80]
```

### 3. final_token_layer_07

```text
[80, 0]
[0, 80]
```

### 4. final_token_layer_08

```text
[80, 0]
[0, 80]
```

### 5. final_token_layer_10

```text
[80, 0]
[0, 80]
```

### 6. final_token_layer_11

```text
[80, 0]
[0, 80]
```

### 7. final_token_layer_12

```text
[80, 0]
[0, 80]
```

### 8. final_token_layer_13

```text
[80, 0]
[0, 80]
```

### 9. final_token_layer_14

```text
[80, 0]
[0, 80]
```

### 10. final_token_layer_15

```text
[80, 0]
[0, 80]
```
