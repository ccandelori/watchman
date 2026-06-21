# Binary Layer Sweep

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `cpu`
- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Reference feature: `mean_pool_layer_18`
- Best feature: `final_token_layer_11`
- Feature count: `58`

## Feature Ranking

| Rank | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---:|---|---:|---:|---:|---:|
| 1 | `final_token_layer_11` | 0.9657 | 0.9667 | 0.0686 | 0.0667 |
| 2 | `final_token_layer_09` | 0.9321 | 0.9333 | 0.0640 | 0.0624 |
| 3 | `final_token_layer_12` | 0.9161 | 0.9167 | 0.0919 | 0.0913 |
| 4 | `final_token_layer_13` | 0.9154 | 0.9167 | 0.1067 | 0.1054 |
| 5 | `final_token_layer_14` | 0.9154 | 0.9167 | 0.1067 | 0.1054 |
| 6 | `final_token_layer_18` | 0.9154 | 0.9167 | 0.1067 | 0.1054 |
| 7 | `final_token_layer_08` | 0.9146 | 0.9167 | 0.0767 | 0.0745 |
| 8 | `mean_pool_layer_00` | 0.8995 | 0.9000 | 0.0977 | 0.0972 |
| 9 | `mean_pool_layer_03` | 0.8993 | 0.9000 | 0.0822 | 0.0816 |
| 10 | `final_token_layer_10` | 0.8981 | 0.9000 | 0.1367 | 0.1333 |
| 11 | `final_token_layer_07` | 0.8979 | 0.9000 | 0.0643 | 0.0624 |
| 12 | `final_token_layer_16` | 0.8828 | 0.8833 | 0.0854 | 0.0850 |
| 13 | `final_token_layer_17` | 0.8825 | 0.8833 | 0.1138 | 0.1130 |
| 14 | `final_token_layer_15` | 0.8657 | 0.8667 | 0.1007 | 0.1000 |
| 15 | `mean_pool_layer_06` | 0.8628 | 0.8667 | 0.1043 | 0.1000 |
| 16 | `mean_pool_layer_04` | 0.8477 | 0.8500 | 0.1005 | 0.0972 |
| 17 | `mean_pool_layer_15` | 0.8462 | 0.8500 | 0.1010 | 0.0972 |
| 18 | `mean_pool_layer_16` | 0.8462 | 0.8500 | 0.1010 | 0.0972 |
| 19 | `mean_pool_layer_05` | 0.8442 | 0.8500 | 0.1537 | 0.1434 |
| 20 | `final_token_layer_19` | 0.8326 | 0.8333 | 0.1756 | 0.1748 |
| 21 | `mean_pool_layer_13` | 0.8300 | 0.8333 | 0.1207 | 0.1179 |
| 22 | `mean_pool_layer_02` | 0.8267 | 0.8333 | 0.1494 | 0.1394 |
| 23 | `final_token_layer_06` | 0.8235 | 0.8333 | 0.1687 | 0.1581 |
| 24 | `mean_pool_layer_20` | 0.8095 | 0.8167 | 0.1149 | 0.1106 |
| 25 | `final_token_layer_22` | 0.7986 | 0.8000 | 0.0847 | 0.0850 |
| 26 | `final_token_layer_20` | 0.7979 | 0.8000 | 0.1151 | 0.1130 |
| 27 | `mean_pool_layer_14` | 0.7969 | 0.8000 | 0.1148 | 0.1130 |
| 28 | `mean_pool_layer_11` | 0.7967 | 0.8000 | 0.0877 | 0.0850 |
| 29 | `mean_pool_layer_21` | 0.7935 | 0.8000 | 0.1167 | 0.1130 |
| 30 | `final_token_layer_24` | 0.7825 | 0.7833 | 0.0851 | 0.0850 |
| 31 | `final_token_layer_23` | 0.7816 | 0.7833 | 0.0845 | 0.0850 |
| 32 | `final_token_layer_26` | 0.7804 | 0.7833 | 0.1016 | 0.1000 |
| 33 | `mean_pool_layer_12` | 0.7801 | 0.7833 | 0.1263 | 0.1247 |
| 34 | `mean_pool_layer_17` | 0.7741 | 0.7833 | 0.1447 | 0.1354 |
| 35 | `mean_pool_layer_22` | 0.7688 | 0.7833 | 0.1352 | 0.1247 |
| 36 | `final_token_layer_21` | 0.7660 | 0.7667 | 0.1232 | 0.1225 |
| 37 | `mean_pool_layer_01` | 0.7527 | 0.7667 | 0.1377 | 0.1225 |
| 38 | `final_token_layer_25` | 0.7473 | 0.7500 | 0.0929 | 0.0913 |
| 39 | `mean_pool_layer_18` (reference) | 0.7225 | 0.7333 | 0.1499 | 0.1434 |
| 40 | `mean_pool_layer_19` | 0.7191 | 0.7333 | 0.1537 | 0.1434 |
| 41 | `mean_pool_layer_07` | 0.7118 | 0.7167 | 0.1151 | 0.1130 |
| 42 | `mean_pool_layer_08` | 0.7116 | 0.7167 | 0.0678 | 0.0667 |
| 43 | `final_token_layer_27` | 0.6972 | 0.7000 | 0.0997 | 0.1000 |
| 44 | `final_token_layer_05` | 0.6961 | 0.7000 | 0.1348 | 0.1354 |
| 45 | `final_token_layer_01` | 0.6934 | 0.7000 | 0.0758 | 0.0667 |
| 46 | `mean_pool_layer_10` | 0.6883 | 0.7000 | 0.1434 | 0.1354 |
| 47 | `mean_pool_layer_09` | 0.6851 | 0.7000 | 0.1332 | 0.1247 |
| 48 | `final_token_layer_28` | 0.6790 | 0.6833 | 0.1438 | 0.1434 |
| 49 | `mean_pool_layer_23` | 0.6767 | 0.7000 | 0.1897 | 0.1716 |
| 50 | `mean_pool_layer_24` | 0.6648 | 0.6833 | 0.1722 | 0.1616 |
| 51 | `mean_pool_layer_25` | 0.6543 | 0.6667 | 0.1461 | 0.1394 |
| 52 | `final_token_layer_03` | 0.6200 | 0.6333 | 0.1912 | 0.1871 |
| 53 | `final_token_layer_04` | 0.6101 | 0.6167 | 0.1467 | 0.1453 |
| 54 | `final_token_layer_02` | 0.6068 | 0.6167 | 0.1505 | 0.1453 |
| 55 | `mean_pool_layer_26` | 0.5608 | 0.5833 | 0.2013 | 0.1900 |
| 56 | `mean_pool_layer_27` | 0.5331 | 0.5500 | 0.1678 | 0.1633 |
| 57 | `mean_pool_layer_28` | 0.4487 | 0.4667 | 0.1579 | 0.1546 |
| 58 | `final_token_layer_00` | 0.3333 | 0.5000 | 0.0000 | 0.0000 |

## Top Confusion Matrices

### 1. final_token_layer_11

```text
[30, 0]
[2, 28]
```

### 2. final_token_layer_09

```text
[29, 1]
[3, 27]
```

### 3. final_token_layer_12

```text
[29, 1]
[4, 26]
```

### 4. final_token_layer_13

```text
[29, 1]
[4, 26]
```

### 5. final_token_layer_14

```text
[29, 1]
[4, 26]
```

### 6. final_token_layer_18

```text
[27, 3]
[2, 28]
```

### 7. final_token_layer_08

```text
[27, 3]
[2, 28]
```

### 8. mean_pool_layer_00

```text
[27, 3]
[3, 27]
```

### 9. mean_pool_layer_03

```text
[27, 3]
[3, 27]
```

### 10. final_token_layer_10

```text
[28, 2]
[4, 26]
```
