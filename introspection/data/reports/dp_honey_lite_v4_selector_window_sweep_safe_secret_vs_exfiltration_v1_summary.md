# Binary Layer Sweep

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `cpu`
- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Reference feature: `readout_window_layer_15`
- Best feature: `readout_window_layer_01`
- Feature count: `29`

## Feature Ranking

| Rank | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---:|---|---:|---:|---:|---:|
| 1 | `readout_window_layer_01` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2 | `readout_window_layer_02` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 3 | `readout_window_layer_03` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 4 | `readout_window_layer_09` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 5 | `readout_window_layer_10` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 6 | `readout_window_layer_11` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 7 | `readout_window_layer_12` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 8 | `readout_window_layer_13` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 9 | `readout_window_layer_14` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 10 | `readout_window_layer_15` (reference) | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 11 | `readout_window_layer_16` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 12 | `readout_window_layer_18` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 13 | `readout_window_layer_24` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 14 | `readout_window_layer_25` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 15 | `readout_window_layer_26` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 16 | `readout_window_layer_27` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 17 | `readout_window_layer_28` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 18 | `readout_window_layer_17` | 0.9875 | 0.9875 | 0.0251 | 0.0250 |
| 19 | `readout_window_layer_23` | 0.9811 | 0.9812 | 0.0378 | 0.0375 |
| 20 | `readout_window_layer_06` | 0.9749 | 0.9750 | 0.0307 | 0.0306 |
| 21 | `readout_window_layer_08` | 0.9746 | 0.9750 | 0.0508 | 0.0500 |
| 22 | `readout_window_layer_19` | 0.9746 | 0.9750 | 0.0508 | 0.0500 |
| 23 | `readout_window_layer_20` | 0.9357 | 0.9375 | 0.0815 | 0.0791 |
| 24 | `readout_window_layer_22` | 0.9341 | 0.9375 | 0.1033 | 0.0968 |
| 25 | `readout_window_layer_21` | 0.9223 | 0.9250 | 0.0952 | 0.0919 |
| 26 | `readout_window_layer_07` | 0.9127 | 0.9250 | 0.1745 | 0.1500 |
| 27 | `readout_window_layer_05` | 0.8928 | 0.9000 | 0.1285 | 0.1159 |
| 28 | `readout_window_layer_04` | 0.8435 | 0.8625 | 0.1938 | 0.1696 |
| 29 | `readout_window_layer_00` | 0.3333 | 0.5000 | 0.0000 | 0.0000 |

## Top Confusion Matrices

### 1. readout_window_layer_01

```text
[48, 0]
[0, 48]
```

### 2. readout_window_layer_02

```text
[48, 0]
[0, 48]
```

### 3. readout_window_layer_03

```text
[48, 0]
[0, 48]
```

### 4. readout_window_layer_09

```text
[48, 0]
[0, 48]
```

### 5. readout_window_layer_10

```text
[48, 0]
[0, 48]
```

### 6. readout_window_layer_11

```text
[48, 0]
[0, 48]
```

### 7. readout_window_layer_12

```text
[48, 0]
[0, 48]
```

### 8. readout_window_layer_13

```text
[48, 0]
[0, 48]
```

### 9. readout_window_layer_14

```text
[48, 0]
[0, 48]
```

### 10. readout_window_layer_15

```text
[48, 0]
[0, 48]
```
