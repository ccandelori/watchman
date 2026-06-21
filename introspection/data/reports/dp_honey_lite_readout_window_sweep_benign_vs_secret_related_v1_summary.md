# Binary Layer Sweep

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `cpu`
- Evaluation strategy: `stratified_group_kfold`
- Task: `benign_vs_secret_related`
- Reference feature: `readout_window_layer_11`
- Best feature: `readout_window_layer_16`
- Feature count: `29`

## Feature Ranking

| Rank | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---:|---|---:|---:|---:|---:|
| 1 | `readout_window_layer_16` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2 | `readout_window_layer_13` | 0.8333 | 0.8333 | 0.1667 | 0.1667 |
| 3 | `readout_window_layer_02` | 0.7000 | 0.8333 | 0.3000 | 0.1667 |
| 4 | `readout_window_layer_09` | 0.7000 | 0.8333 | 0.3000 | 0.1667 |
| 5 | `readout_window_layer_17` | 0.7000 | 0.8333 | 0.3000 | 0.1667 |
| 6 | `readout_window_layer_20` | 0.7000 | 0.8333 | 0.3000 | 0.1667 |
| 7 | `readout_window_layer_23` | 0.7000 | 0.8333 | 0.3000 | 0.1667 |
| 8 | `readout_window_layer_27` | 0.7000 | 0.8333 | 0.3000 | 0.1667 |
| 9 | `readout_window_layer_10` | 0.6556 | 0.7917 | 0.2556 | 0.1250 |
| 10 | `readout_window_layer_15` | 0.6143 | 0.7500 | 0.2143 | 0.0833 |
| 11 | `readout_window_layer_18` | 0.6143 | 0.7500 | 0.2143 | 0.0833 |
| 12 | `readout_window_layer_19` | 0.6143 | 0.7500 | 0.2143 | 0.0833 |
| 13 | `readout_window_layer_08` | 0.5741 | 0.7083 | 0.1741 | 0.0417 |
| 14 | `readout_window_layer_06` | 0.5333 | 0.6667 | 0.1333 | 0.0000 |
| 15 | `readout_window_layer_11` (reference) | 0.5333 | 0.6667 | 0.1333 | 0.0000 |
| 16 | `readout_window_layer_12` | 0.5333 | 0.6667 | 0.1333 | 0.0000 |
| 17 | `readout_window_layer_14` | 0.5333 | 0.6667 | 0.1333 | 0.0000 |
| 18 | `readout_window_layer_24` | 0.5333 | 0.6667 | 0.1333 | 0.0000 |
| 19 | `readout_window_layer_25` | 0.5333 | 0.6667 | 0.1333 | 0.0000 |
| 20 | `readout_window_layer_28` | 0.5333 | 0.6667 | 0.1333 | 0.0000 |
| 21 | `readout_window_layer_01` | 0.4000 | 0.6667 | 0.0000 | 0.0000 |
| 22 | `readout_window_layer_03` | 0.4000 | 0.6667 | 0.0000 | 0.0000 |
| 23 | `readout_window_layer_05` | 0.4000 | 0.6667 | 0.0000 | 0.0000 |
| 24 | `readout_window_layer_22` | 0.4000 | 0.6667 | 0.0000 | 0.0000 |
| 25 | `readout_window_layer_26` | 0.4000 | 0.6667 | 0.0000 | 0.0000 |
| 26 | `readout_window_layer_04` | 0.3250 | 0.5000 | 0.0750 | 0.1667 |
| 27 | `readout_window_layer_07` | 0.3250 | 0.5000 | 0.0750 | 0.1667 |
| 28 | `readout_window_layer_21` | 0.3250 | 0.5000 | 0.0750 | 0.1667 |
| 29 | `readout_window_layer_00` | 0.2500 | 0.3333 | 0.0000 | 0.0000 |

## Top Confusion Matrices

### 1. readout_window_layer_16

```text
[8, 0]
[0, 16]
```

### 2. readout_window_layer_13

```text
[8, 0]
[4, 12]
```

### 3. readout_window_layer_02

```text
[4, 4]
[0, 16]
```

### 4. readout_window_layer_09

```text
[4, 4]
[0, 16]
```

### 5. readout_window_layer_17

```text
[4, 4]
[0, 16]
```

### 6. readout_window_layer_20

```text
[4, 4]
[0, 16]
```

### 7. readout_window_layer_23

```text
[4, 4]
[0, 16]
```

### 8. readout_window_layer_27

```text
[4, 4]
[0, 16]
```

### 9. readout_window_layer_10

```text
[4, 4]
[1, 15]
```

### 10. readout_window_layer_15

```text
[4, 4]
[2, 14]
```
