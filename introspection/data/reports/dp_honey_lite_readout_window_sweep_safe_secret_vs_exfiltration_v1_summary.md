# Binary Layer Sweep

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `cpu`
- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Reference feature: `readout_window_layer_11`
- Best feature: `readout_window_layer_07`
- Feature count: `29`

## Feature Ranking

| Rank | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---:|---|---:|---:|---:|---:|
| 1 | `readout_window_layer_07` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2 | `readout_window_layer_08` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 3 | `readout_window_layer_10` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 4 | `readout_window_layer_19` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 5 | `readout_window_layer_14` | 0.9365 | 0.9375 | 0.0635 | 0.0625 |
| 6 | `readout_window_layer_20` | 0.7333 | 0.7500 | 0.0000 | 0.0000 |
| 7 | `readout_window_layer_00` | 0.6667 | 0.7500 | 0.3333 | 0.2500 |
| 8 | `readout_window_layer_01` | 0.6667 | 0.7500 | 0.3333 | 0.2500 |
| 9 | `readout_window_layer_03` | 0.6667 | 0.7500 | 0.3333 | 0.2500 |
| 10 | `readout_window_layer_05` | 0.6667 | 0.7500 | 0.3333 | 0.2500 |
| 11 | `readout_window_layer_06` | 0.6667 | 0.7500 | 0.3333 | 0.2500 |
| 12 | `readout_window_layer_15` | 0.6667 | 0.7500 | 0.3333 | 0.2500 |
| 13 | `readout_window_layer_16` | 0.6667 | 0.7500 | 0.3333 | 0.2500 |
| 14 | `readout_window_layer_21` | 0.6667 | 0.7500 | 0.3333 | 0.2500 |
| 15 | `readout_window_layer_26` | 0.6032 | 0.6875 | 0.2698 | 0.1875 |
| 16 | `readout_window_layer_17` | 0.5333 | 0.6250 | 0.2000 | 0.1250 |
| 17 | `readout_window_layer_02` | 0.3333 | 0.5000 | 0.0000 | 0.0000 |
| 18 | `readout_window_layer_04` | 0.3333 | 0.5000 | 0.0000 | 0.0000 |
| 19 | `readout_window_layer_11` (reference) | 0.3333 | 0.5000 | 0.0000 | 0.0000 |
| 20 | `readout_window_layer_12` | 0.3333 | 0.5000 | 0.0000 | 0.0000 |
| 21 | `readout_window_layer_13` | 0.3333 | 0.5000 | 0.0000 | 0.0000 |
| 22 | `readout_window_layer_18` | 0.3333 | 0.5000 | 0.0000 | 0.0000 |
| 23 | `readout_window_layer_22` | 0.3333 | 0.5000 | 0.0000 | 0.0000 |
| 24 | `readout_window_layer_23` | 0.3333 | 0.5000 | 0.0000 | 0.0000 |
| 25 | `readout_window_layer_24` | 0.3333 | 0.5000 | 0.0000 | 0.0000 |
| 26 | `readout_window_layer_25` | 0.3333 | 0.5000 | 0.0000 | 0.0000 |
| 27 | `readout_window_layer_27` | 0.3333 | 0.5000 | 0.0000 | 0.0000 |
| 28 | `readout_window_layer_28` | 0.3333 | 0.5000 | 0.0000 | 0.0000 |
| 29 | `readout_window_layer_09` | 0.1667 | 0.2500 | 0.1667 | 0.2500 |

## Top Confusion Matrices

### 1. readout_window_layer_07

```text
[8, 0]
[0, 8]
```

### 2. readout_window_layer_08

```text
[8, 0]
[0, 8]
```

### 3. readout_window_layer_10

```text
[8, 0]
[0, 8]
```

### 4. readout_window_layer_19

```text
[8, 0]
[0, 8]
```

### 5. readout_window_layer_14

```text
[8, 0]
[1, 7]
```

### 6. readout_window_layer_20

```text
[4, 4]
[0, 8]
```

### 7. readout_window_layer_00

```text
[8, 0]
[4, 4]
```

### 8. readout_window_layer_01

```text
[8, 0]
[4, 4]
```

### 9. readout_window_layer_03

```text
[4, 4]
[0, 8]
```

### 10. readout_window_layer_05

```text
[8, 0]
[4, 4]
```
