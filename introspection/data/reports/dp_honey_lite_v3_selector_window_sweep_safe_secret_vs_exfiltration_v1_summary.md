# Binary Layer Sweep

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `cpu`
- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Reference feature: `readout_window_layer_21`
- Best feature: `readout_window_layer_15`
- Feature count: `29`

## Feature Ranking

| Rank | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---:|---|---:|---:|---:|---:|
| 1 | `readout_window_layer_15` | 0.7736 | 0.7812 | 0.0709 | 0.0656 |
| 2 | `readout_window_layer_17` | 0.7662 | 0.7750 | 0.1322 | 0.1272 |
| 3 | `readout_window_layer_16` | 0.7625 | 0.7688 | 0.0879 | 0.0875 |
| 4 | `readout_window_layer_18` | 0.7504 | 0.7625 | 0.1336 | 0.1259 |
| 5 | `readout_window_layer_13` | 0.7014 | 0.7188 | 0.1145 | 0.1027 |
| 6 | `readout_window_layer_14` | 0.7001 | 0.7188 | 0.1126 | 0.1027 |
| 7 | `readout_window_layer_20` | 0.6508 | 0.6750 | 0.1178 | 0.1093 |
| 8 | `readout_window_layer_21` (reference) | 0.6443 | 0.6687 | 0.1385 | 0.1179 |
| 9 | `readout_window_layer_19` | 0.6157 | 0.6625 | 0.2002 | 0.1523 |
| 10 | `readout_window_layer_23` | 0.5494 | 0.5938 | 0.1123 | 0.0523 |
| 11 | `readout_window_layer_07` | 0.5468 | 0.5750 | 0.0697 | 0.0424 |
| 12 | `readout_window_layer_12` | 0.5432 | 0.6125 | 0.2103 | 0.1637 |
| 13 | `readout_window_layer_10` | 0.5423 | 0.5750 | 0.0349 | 0.0468 |
| 14 | `readout_window_layer_11` | 0.5365 | 0.5750 | 0.0662 | 0.0612 |
| 15 | `readout_window_layer_22` | 0.5255 | 0.5813 | 0.1112 | 0.0673 |
| 16 | `readout_window_layer_25` | 0.5191 | 0.5875 | 0.1406 | 0.0871 |
| 17 | `readout_window_layer_09` | 0.5029 | 0.5750 | 0.1745 | 0.1196 |
| 18 | `readout_window_layer_08` | 0.4980 | 0.5563 | 0.1431 | 0.0976 |
| 19 | `readout_window_layer_24` | 0.4918 | 0.5437 | 0.0886 | 0.0319 |
| 20 | `readout_window_layer_06` | 0.4788 | 0.5188 | 0.0871 | 0.0508 |
| 21 | `readout_window_layer_28` | 0.4700 | 0.5312 | 0.0847 | 0.0280 |
| 22 | `readout_window_layer_01` | 0.4698 | 0.5062 | 0.0955 | 0.0848 |
| 23 | `readout_window_layer_26` | 0.4580 | 0.5437 | 0.1053 | 0.0424 |
| 24 | `readout_window_layer_27` | 0.4550 | 0.5250 | 0.1071 | 0.0459 |
| 25 | `readout_window_layer_05` | 0.4166 | 0.4625 | 0.0623 | 0.0606 |
| 26 | `readout_window_layer_04` | 0.4105 | 0.4688 | 0.0812 | 0.0625 |
| 27 | `readout_window_layer_03` | 0.3822 | 0.4062 | 0.0574 | 0.0559 |
| 28 | `readout_window_layer_02` | 0.3327 | 0.4000 | 0.0627 | 0.1072 |
| 29 | `readout_window_layer_00` | 0.2000 | 0.3000 | 0.1633 | 0.2449 |

## Top Confusion Matrices

### 1. readout_window_layer_15

```text
[59, 21]
[14, 66]
```

### 2. readout_window_layer_17

```text
[61, 19]
[17, 63]
```

### 3. readout_window_layer_16

```text
[57, 23]
[14, 66]
```

### 4. readout_window_layer_18

```text
[58, 22]
[16, 64]
```

### 5. readout_window_layer_13

```text
[58, 22]
[23, 57]
```

### 6. readout_window_layer_14

```text
[58, 22]
[23, 57]
```

### 7. readout_window_layer_20

```text
[54, 26]
[26, 54]
```

### 8. readout_window_layer_21

```text
[47, 33]
[20, 60]
```

### 9. readout_window_layer_19

```text
[47, 33]
[21, 59]
```

### 10. readout_window_layer_23

```text
[48, 32]
[33, 47]
```
