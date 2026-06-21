# Binary Layer Sweep

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `cpu`
- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Reference feature: `readout_window_layer_21`
- Best feature: `readout_window_layer_00`
- Feature count: `29`

## Feature Ranking

| Rank | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---:|---|---:|---:|---:|---:|
| 1 | `readout_window_layer_00` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2 | `readout_window_layer_01` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 3 | `readout_window_layer_02` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 4 | `readout_window_layer_03` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 5 | `readout_window_layer_04` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 6 | `readout_window_layer_05` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 7 | `readout_window_layer_06` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 8 | `readout_window_layer_07` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 9 | `readout_window_layer_08` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 10 | `readout_window_layer_09` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 11 | `readout_window_layer_10` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 12 | `readout_window_layer_11` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 13 | `readout_window_layer_12` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 14 | `readout_window_layer_13` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 15 | `readout_window_layer_14` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 16 | `readout_window_layer_15` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 17 | `readout_window_layer_16` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 18 | `readout_window_layer_17` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 19 | `readout_window_layer_18` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 20 | `readout_window_layer_19` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 21 | `readout_window_layer_20` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 22 | `readout_window_layer_21` (reference) | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 23 | `readout_window_layer_22` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 24 | `readout_window_layer_23` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 25 | `readout_window_layer_24` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 26 | `readout_window_layer_25` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 27 | `readout_window_layer_26` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 28 | `readout_window_layer_27` | 0.9244 | 0.9250 | 0.0734 | 0.0729 |
| 29 | `readout_window_layer_28` | 0.9213 | 0.9250 | 0.1061 | 0.1000 |

## Top Confusion Matrices

### 1. readout_window_layer_00

```text
[80, 0]
[0, 80]
```

### 2. readout_window_layer_01

```text
[80, 0]
[0, 80]
```

### 3. readout_window_layer_02

```text
[80, 0]
[0, 80]
```

### 4. readout_window_layer_03

```text
[80, 0]
[0, 80]
```

### 5. readout_window_layer_04

```text
[80, 0]
[0, 80]
```

### 6. readout_window_layer_05

```text
[80, 0]
[0, 80]
```

### 7. readout_window_layer_06

```text
[80, 0]
[0, 80]
```

### 8. readout_window_layer_07

```text
[80, 0]
[0, 80]
```

### 9. readout_window_layer_08

```text
[80, 0]
[0, 80]
```

### 10. readout_window_layer_09

```text
[80, 0]
[0, 80]
```
