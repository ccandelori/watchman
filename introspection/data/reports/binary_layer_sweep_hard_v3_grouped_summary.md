# Binary Layer Sweep

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `cpu`
- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Reference feature: `mean_pool_layer_18`
- Best feature: `final_token_layer_16`
- Feature count: `58`

## Feature Ranking

| Rank | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---:|---|---:|---:|---:|---:|
| 1 | `final_token_layer_16` | 0.9321 | 0.9333 | 0.0640 | 0.0624 |
| 2 | `mean_pool_layer_20` | 0.9163 | 0.9167 | 0.0527 | 0.0527 |
| 3 | `mean_pool_layer_02` | 0.8830 | 0.8833 | 0.1129 | 0.1130 |
| 4 | `mean_pool_layer_03` | 0.8823 | 0.8833 | 0.1251 | 0.1247 |
| 5 | `final_token_layer_11` | 0.8818 | 0.8833 | 0.0860 | 0.0850 |
| 6 | `final_token_layer_15` | 0.8813 | 0.8833 | 0.0681 | 0.0667 |
| 7 | `final_token_layer_17` | 0.8813 | 0.8833 | 0.0681 | 0.0667 |
| 8 | `mean_pool_layer_00` | 0.8813 | 0.8833 | 0.0681 | 0.0667 |
| 9 | `final_token_layer_08` | 0.8662 | 0.8667 | 0.0998 | 0.1000 |
| 10 | `final_token_layer_12` | 0.8650 | 0.8667 | 0.0675 | 0.0667 |
| 11 | `mean_pool_layer_21` | 0.8636 | 0.8667 | 0.1166 | 0.1130 |
| 12 | `final_token_layer_14` | 0.8623 | 0.8667 | 0.0900 | 0.0850 |
| 13 | `final_token_layer_18` | 0.8623 | 0.8667 | 0.0900 | 0.0850 |
| 14 | `mean_pool_layer_04` | 0.8489 | 0.8500 | 0.1106 | 0.1106 |
| 15 | `final_token_layer_13` | 0.8475 | 0.8500 | 0.0632 | 0.0624 |
| 16 | `final_token_layer_19` | 0.8452 | 0.8500 | 0.1012 | 0.0972 |
| 17 | `mean_pool_layer_22` | 0.8324 | 0.8333 | 0.0751 | 0.0745 |
| 18 | `mean_pool_layer_18` (reference) | 0.8324 | 0.8333 | 0.1054 | 0.1054 |
| 19 | `final_token_layer_22` | 0.8307 | 0.8333 | 0.1084 | 0.1054 |
| 20 | `final_token_layer_20` | 0.8302 | 0.8333 | 0.0946 | 0.0913 |
| 21 | `mean_pool_layer_13` | 0.8161 | 0.8167 | 0.1344 | 0.1333 |
| 22 | `mean_pool_layer_06` | 0.8156 | 0.8167 | 0.0813 | 0.0816 |
| 23 | `mean_pool_layer_15` | 0.8149 | 0.8167 | 0.1233 | 0.1225 |
| 24 | `mean_pool_layer_16` | 0.8149 | 0.8167 | 0.1233 | 0.1225 |
| 25 | `mean_pool_layer_01` | 0.8139 | 0.8167 | 0.1231 | 0.1225 |
| 26 | `final_token_layer_09` | 0.8129 | 0.8167 | 0.1245 | 0.1225 |
| 27 | `final_token_layer_07` | 0.8080 | 0.8167 | 0.1330 | 0.1225 |
| 28 | `mean_pool_layer_12` | 0.7991 | 0.8000 | 0.2029 | 0.2014 |
| 29 | `mean_pool_layer_05` | 0.7986 | 0.8000 | 0.0847 | 0.0850 |
| 30 | `mean_pool_layer_17` | 0.7967 | 0.8000 | 0.0877 | 0.0850 |
| 31 | `final_token_layer_10` | 0.7946 | 0.8000 | 0.1385 | 0.1354 |
| 32 | `mean_pool_layer_19` | 0.7825 | 0.7833 | 0.0851 | 0.0850 |
| 33 | `mean_pool_layer_14` | 0.7823 | 0.7833 | 0.1140 | 0.1130 |
| 34 | `mean_pool_layer_08` | 0.7818 | 0.7833 | 0.1458 | 0.1453 |
| 35 | `mean_pool_layer_11` | 0.7797 | 0.7833 | 0.1764 | 0.1716 |
| 36 | `final_token_layer_21` | 0.7767 | 0.7833 | 0.0884 | 0.0850 |
| 37 | `mean_pool_layer_07` | 0.7650 | 0.7667 | 0.1227 | 0.1225 |
| 38 | `mean_pool_layer_09` | 0.7638 | 0.7667 | 0.1745 | 0.1700 |
| 39 | `final_token_layer_06` | 0.7576 | 0.7667 | 0.0854 | 0.0816 |
| 40 | `mean_pool_layer_10` | 0.7490 | 0.7500 | 0.1185 | 0.1179 |
| 41 | `final_token_layer_23` | 0.7461 | 0.7500 | 0.1192 | 0.1179 |
| 42 | `final_token_layer_24` | 0.7461 | 0.7500 | 0.1192 | 0.1179 |
| 43 | `final_token_layer_25` | 0.7461 | 0.7500 | 0.1192 | 0.1179 |
| 44 | `final_token_layer_26` | 0.7461 | 0.7500 | 0.1192 | 0.1179 |
| 45 | `final_token_layer_03` | 0.7461 | 0.7500 | 0.1192 | 0.1179 |
| 46 | `mean_pool_layer_23` | 0.7324 | 0.7333 | 0.1326 | 0.1333 |
| 47 | `mean_pool_layer_25` | 0.7319 | 0.7333 | 0.1425 | 0.1434 |
| 48 | `final_token_layer_27` | 0.7310 | 0.7333 | 0.1441 | 0.1434 |
| 49 | `mean_pool_layer_24` | 0.7166 | 0.7167 | 0.1451 | 0.1453 |
| 50 | `mean_pool_layer_26` | 0.6972 | 0.7000 | 0.1732 | 0.1716 |
| 51 | `final_token_layer_28` | 0.6621 | 0.6667 | 0.1028 | 0.1054 |
| 52 | `final_token_layer_05` | 0.6343 | 0.6500 | 0.1470 | 0.1333 |
| 53 | `mean_pool_layer_27` | 0.6276 | 0.6333 | 0.0887 | 0.0850 |
| 54 | `final_token_layer_04` | 0.6212 | 0.6333 | 0.0622 | 0.0667 |
| 55 | `mean_pool_layer_28` | 0.5897 | 0.6000 | 0.2238 | 0.2068 |
| 56 | `final_token_layer_01` | 0.5662 | 0.5833 | 0.0963 | 0.0913 |
| 57 | `final_token_layer_02` | 0.5116 | 0.5167 | 0.0991 | 0.0972 |
| 58 | `final_token_layer_00` | 0.3333 | 0.5000 | 0.0000 | 0.0000 |

## Top Confusion Matrices

### 1. final_token_layer_16

```text
[28, 2]
[2, 28]
```

### 2. mean_pool_layer_20

```text
[26, 4]
[1, 29]
```

### 3. mean_pool_layer_02

```text
[26, 4]
[3, 27]
```

### 4. mean_pool_layer_03

```text
[26, 4]
[3, 27]
```

### 5. final_token_layer_11

```text
[28, 2]
[5, 25]
```

### 6. final_token_layer_15

```text
[27, 3]
[4, 26]
```

### 7. final_token_layer_17

```text
[27, 3]
[4, 26]
```

### 8. mean_pool_layer_00

```text
[26, 4]
[3, 27]
```

### 9. final_token_layer_08

```text
[26, 4]
[4, 26]
```

### 10. final_token_layer_12

```text
[27, 3]
[5, 25]
```
