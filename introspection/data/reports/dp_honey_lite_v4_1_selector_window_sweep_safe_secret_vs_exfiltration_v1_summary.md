# Binary Layer Sweep

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `cpu`
- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Reference feature: `readout_window_layer_15`
- Best feature: `readout_window_layer_15`
- Feature count: `29`

## Feature Ranking

| Rank | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---:|---|---:|---:|---:|---:|
| 1 | `readout_window_layer_15` (reference) | 0.7646 | 0.7875 | 0.1607 | 0.1225 |
| 2 | `readout_window_layer_17` | 0.6485 | 0.6813 | 0.0885 | 0.0696 |
| 3 | `readout_window_layer_16` | 0.6469 | 0.6937 | 0.1621 | 0.1016 |
| 4 | `readout_window_layer_20` | 0.5818 | 0.6438 | 0.1665 | 0.1075 |
| 5 | `readout_window_layer_18` | 0.5727 | 0.6250 | 0.1279 | 0.0791 |
| 6 | `readout_window_layer_19` | 0.5718 | 0.6250 | 0.0660 | 0.0395 |
| 7 | `readout_window_layer_14` | 0.5579 | 0.6312 | 0.1435 | 0.0893 |
| 8 | `readout_window_layer_21` | 0.5397 | 0.6000 | 0.1061 | 0.0500 |
| 9 | `readout_window_layer_12` | 0.4759 | 0.5813 | 0.1608 | 0.0960 |
| 10 | `readout_window_layer_07` | 0.4682 | 0.5563 | 0.0829 | 0.0415 |
| 11 | `readout_window_layer_23` | 0.4649 | 0.5312 | 0.0915 | 0.0484 |
| 12 | `readout_window_layer_06` | 0.4583 | 0.5437 | 0.1585 | 0.1075 |
| 13 | `readout_window_layer_22` | 0.4544 | 0.5062 | 0.0650 | 0.0125 |
| 14 | `readout_window_layer_13` | 0.4542 | 0.5625 | 0.1289 | 0.0685 |
| 15 | `readout_window_layer_09` | 0.4361 | 0.4875 | 0.0995 | 0.0829 |
| 16 | `readout_window_layer_24` | 0.4291 | 0.4875 | 0.0542 | 0.0250 |
| 17 | `readout_window_layer_10` | 0.4270 | 0.5125 | 0.0931 | 0.0468 |
| 18 | `readout_window_layer_05` | 0.4158 | 0.5188 | 0.0819 | 0.0250 |
| 19 | `readout_window_layer_11` | 0.3968 | 0.5000 | 0.1118 | 0.0791 |
| 20 | `readout_window_layer_04` | 0.3709 | 0.4813 | 0.0636 | 0.0250 |
| 21 | `readout_window_layer_28` | 0.3697 | 0.5125 | 0.0727 | 0.0250 |
| 22 | `readout_window_layer_27` | 0.3671 | 0.4750 | 0.0489 | 0.0637 |
| 23 | `readout_window_layer_25` | 0.3651 | 0.5000 | 0.0635 | 0.0000 |
| 24 | `readout_window_layer_03` | 0.3640 | 0.4125 | 0.0820 | 0.0750 |
| 25 | `readout_window_layer_26` | 0.3600 | 0.5000 | 0.0533 | 0.0000 |
| 26 | `readout_window_layer_02` | 0.3467 | 0.4125 | 0.0886 | 0.0848 |
| 27 | `readout_window_layer_01` | 0.3289 | 0.3500 | 0.1055 | 0.0935 |
| 28 | `readout_window_layer_08` | 0.3154 | 0.4625 | 0.0241 | 0.0500 |
| 29 | `readout_window_layer_00` | 0.0667 | 0.1000 | 0.1333 | 0.2000 |

## Top Confusion Matrices

### 1. readout_window_layer_15

```text
[34, 14]
[6, 42]
```

### 2. readout_window_layer_17

```text
[32, 16]
[15, 33]
```

### 3. readout_window_layer_16

```text
[32, 16]
[12, 36]
```

### 4. readout_window_layer_20

```text
[27, 21]
[15, 33]
```

### 5. readout_window_layer_18

```text
[26, 22]
[15, 33]
```

### 6. readout_window_layer_19

```text
[26, 22]
[14, 34]
```

### 7. readout_window_layer_14

```text
[29, 19]
[16, 32]
```

### 8. readout_window_layer_21

```text
[30, 18]
[20, 28]
```

### 9. readout_window_layer_12

```text
[25, 23]
[18, 30]
```

### 10. readout_window_layer_07

```text
[22, 26]
[17, 31]
```
