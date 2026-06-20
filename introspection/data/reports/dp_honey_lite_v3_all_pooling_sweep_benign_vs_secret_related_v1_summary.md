# Binary Layer Sweep

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `cpu`
- Evaluation strategy: `stratified_group_kfold`
- Task: `benign_vs_secret_related`
- Reference feature: `readout_window_layer_11`
- Best feature: `final_token_layer_01`
- Feature count: `87`

## Feature Ranking

| Rank | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---:|---|---:|---:|---:|---:|
| 1 | `final_token_layer_01` | 0.9906 | 0.9917 | 0.0115 | 0.0102 |
| 2 | `mean_pool_layer_00` | 0.9412 | 0.9500 | 0.0579 | 0.0468 |
| 3 | `mean_pool_layer_01` | 0.9312 | 0.9375 | 0.0572 | 0.0527 |
| 4 | `mean_pool_layer_02` | 0.9247 | 0.9333 | 0.0737 | 0.0664 |
| 5 | `mean_pool_layer_17` | 0.9180 | 0.9208 | 0.1065 | 0.1074 |
| 6 | `mean_pool_layer_14` | 0.9158 | 0.9208 | 0.0871 | 0.0878 |
| 7 | `mean_pool_layer_15` | 0.9107 | 0.9125 | 0.1188 | 0.1189 |
| 8 | `final_token_layer_11` | 0.9090 | 0.9167 | 0.0759 | 0.0757 |
| 9 | `mean_pool_layer_12` | 0.9023 | 0.9167 | 0.0503 | 0.0373 |
| 10 | `mean_pool_layer_03` | 0.8893 | 0.9042 | 0.0902 | 0.0797 |
| 11 | `mean_pool_layer_16` | 0.8868 | 0.8875 | 0.1458 | 0.1453 |
| 12 | `final_token_layer_02` | 0.8743 | 0.8792 | 0.1164 | 0.1181 |
| 13 | `final_token_layer_12` | 0.8722 | 0.8792 | 0.0831 | 0.0806 |
| 14 | `mean_pool_layer_09` | 0.8707 | 0.8917 | 0.1016 | 0.0761 |
| 15 | `mean_pool_layer_21` | 0.8691 | 0.8750 | 0.1032 | 0.1054 |
| 16 | `readout_window_layer_02` | 0.8664 | 0.8667 | 0.1657 | 0.1654 |
| 17 | `readout_window_layer_17` | 0.8636 | 0.8750 | 0.0393 | 0.0437 |
| 18 | `mean_pool_layer_19` | 0.8589 | 0.8667 | 0.1078 | 0.1107 |
| 19 | `final_token_layer_13` | 0.8494 | 0.8583 | 0.0900 | 0.0888 |
| 20 | `mean_pool_layer_10` | 0.8493 | 0.8625 | 0.0823 | 0.0808 |
| 21 | `mean_pool_layer_20` | 0.8450 | 0.8500 | 0.1310 | 0.1333 |
| 22 | `mean_pool_layer_13` | 0.8425 | 0.8542 | 0.1339 | 0.1337 |
| 23 | `mean_pool_layer_18` | 0.8408 | 0.8458 | 0.1379 | 0.1404 |
| 24 | `final_token_layer_10` | 0.8359 | 0.8667 | 0.1180 | 0.0752 |
| 25 | `final_token_layer_03` | 0.8292 | 0.8458 | 0.0424 | 0.0429 |
| 26 | `mean_pool_layer_11` | 0.8271 | 0.8458 | 0.0525 | 0.0583 |
| 27 | `mean_pool_layer_22` | 0.8207 | 0.8375 | 0.0736 | 0.0738 |
| 28 | `readout_window_layer_16` | 0.8157 | 0.8417 | 0.0920 | 0.0626 |
| 29 | `final_token_layer_14` | 0.8085 | 0.8208 | 0.1113 | 0.1091 |
| 30 | `final_token_layer_17` | 0.8013 | 0.8167 | 0.0946 | 0.0888 |
| 31 | `final_token_layer_18` | 0.7976 | 0.8083 | 0.0811 | 0.0848 |
| 32 | `readout_window_layer_18` | 0.7970 | 0.8083 | 0.1226 | 0.1273 |
| 33 | `readout_window_layer_03` | 0.7901 | 0.8042 | 0.1520 | 0.1465 |
| 34 | `final_token_layer_05` | 0.7883 | 0.8042 | 0.0899 | 0.0880 |
| 35 | `readout_window_layer_14` | 0.7879 | 0.8000 | 0.1623 | 0.1638 |
| 36 | `mean_pool_layer_07` | 0.7877 | 0.8292 | 0.1868 | 0.1464 |
| 37 | `readout_window_layer_13` | 0.7846 | 0.8125 | 0.1296 | 0.1156 |
| 38 | `readout_window_layer_09` | 0.7839 | 0.8042 | 0.0918 | 0.0899 |
| 39 | `readout_window_layer_11` (reference) | 0.7675 | 0.7833 | 0.1059 | 0.1153 |
| 40 | `readout_window_layer_19` | 0.7658 | 0.7833 | 0.1233 | 0.1240 |
| 41 | `final_token_layer_15` | 0.7630 | 0.7833 | 0.0786 | 0.0680 |
| 42 | `final_token_layer_16` | 0.7584 | 0.7833 | 0.1427 | 0.1404 |
| 43 | `readout_window_layer_01` | 0.7500 | 0.7708 | 0.0985 | 0.1003 |
| 44 | `final_token_layer_19` | 0.7436 | 0.7542 | 0.0737 | 0.0806 |
| 45 | `final_token_layer_04` | 0.7423 | 0.7625 | 0.1071 | 0.1123 |
| 46 | `readout_window_layer_12` | 0.7407 | 0.7583 | 0.0384 | 0.0537 |
| 47 | `readout_window_layer_10` | 0.7378 | 0.7542 | 0.0915 | 0.1057 |
| 48 | `mean_pool_layer_08` | 0.7346 | 0.7708 | 0.1533 | 0.1324 |
| 49 | `mean_pool_layer_24` | 0.7337 | 0.7625 | 0.1293 | 0.1254 |
| 50 | `mean_pool_layer_04` | 0.7309 | 0.7542 | 0.1217 | 0.1232 |
| 51 | `mean_pool_layer_25` | 0.7262 | 0.7542 | 0.1841 | 0.1775 |
| 52 | `mean_pool_layer_06` | 0.7248 | 0.7625 | 0.1044 | 0.0982 |
| 53 | `final_token_layer_20` | 0.7203 | 0.7375 | 0.0562 | 0.0486 |
| 54 | `mean_pool_layer_23` | 0.7165 | 0.7417 | 0.1548 | 0.1506 |
| 55 | `readout_window_layer_06` | 0.7064 | 0.7458 | 0.0588 | 0.0482 |
| 56 | `mean_pool_layer_26` | 0.6917 | 0.7375 | 0.1728 | 0.1721 |
| 57 | `final_token_layer_07` | 0.6902 | 0.7125 | 0.0611 | 0.0534 |
| 58 | `final_token_layer_22` | 0.6880 | 0.7167 | 0.1025 | 0.0991 |
| 59 | `final_token_layer_06` | 0.6781 | 0.7000 | 0.0945 | 0.0991 |
| 60 | `final_token_layer_09` | 0.6769 | 0.6958 | 0.0929 | 0.1083 |
| 61 | `final_token_layer_25` | 0.6761 | 0.7125 | 0.0917 | 0.0827 |
| 62 | `readout_window_layer_20` | 0.6747 | 0.7083 | 0.1535 | 0.1542 |
| 63 | `readout_window_layer_15` | 0.6718 | 0.7208 | 0.0970 | 0.0991 |
| 64 | `readout_window_layer_04` | 0.6705 | 0.7083 | 0.1640 | 0.1576 |
| 65 | `readout_window_layer_07` | 0.6678 | 0.7208 | 0.1151 | 0.1146 |
| 66 | `final_token_layer_21` | 0.6661 | 0.6833 | 0.0903 | 0.0998 |
| 67 | `final_token_layer_27` | 0.6642 | 0.6958 | 0.1196 | 0.1161 |
| 68 | `final_token_layer_24` | 0.6629 | 0.6875 | 0.1087 | 0.1215 |
| 69 | `final_token_layer_23` | 0.6610 | 0.6958 | 0.0659 | 0.0890 |
| 70 | `readout_window_layer_05` | 0.6604 | 0.7000 | 0.0521 | 0.0626 |
| 71 | `mean_pool_layer_05` | 0.6537 | 0.7167 | 0.1959 | 0.1685 |
| 72 | `final_token_layer_26` | 0.6470 | 0.6750 | 0.0714 | 0.0775 |
| 73 | `final_token_layer_08` | 0.6348 | 0.6750 | 0.1158 | 0.1282 |
| 74 | `readout_window_layer_21` | 0.6323 | 0.6708 | 0.1564 | 0.1567 |
| 75 | `readout_window_layer_24` | 0.6322 | 0.6708 | 0.1222 | 0.1353 |
| 76 | `readout_window_layer_26` | 0.6213 | 0.6583 | 0.0525 | 0.0850 |
| 77 | `final_token_layer_28` | 0.6209 | 0.6458 | 0.0856 | 0.1003 |
| 78 | `readout_window_layer_22` | 0.6131 | 0.6625 | 0.1394 | 0.1440 |
| 79 | `readout_window_layer_28` | 0.5988 | 0.6208 | 0.0576 | 0.0878 |
| 80 | `readout_window_layer_25` | 0.5886 | 0.6583 | 0.1564 | 0.1341 |
| 81 | `readout_window_layer_23` | 0.5721 | 0.6125 | 0.1150 | 0.1335 |
| 82 | `readout_window_layer_08` | 0.5699 | 0.6667 | 0.1433 | 0.1087 |
| 83 | `mean_pool_layer_27` | 0.5584 | 0.6375 | 0.1423 | 0.1423 |
| 84 | `readout_window_layer_27` | 0.5409 | 0.5792 | 0.1479 | 0.1440 |
| 85 | `mean_pool_layer_28` | 0.5138 | 0.5833 | 0.1419 | 0.1514 |
| 86 | `final_token_layer_00` | 0.2500 | 0.3333 | 0.0000 | 0.0000 |
| 87 | `readout_window_layer_00` | 0.2500 | 0.3333 | 0.0000 | 0.0000 |

## Top Confusion Matrices

### 1. final_token_layer_01

```text
[79, 1]
[1, 159]
```

### 2. mean_pool_layer_00

```text
[73, 7]
[5, 155]
```

### 3. mean_pool_layer_01

```text
[75, 5]
[10, 150]
```

### 4. mean_pool_layer_02

```text
[73, 7]
[9, 151]
```

### 5. mean_pool_layer_17

```text
[77, 3]
[16, 144]
```

### 6. mean_pool_layer_14

```text
[74, 6]
[13, 147]
```

### 7. mean_pool_layer_15

```text
[79, 1]
[20, 140]
```

### 8. final_token_layer_11

```text
[71, 9]
[11, 149]
```

### 9. mean_pool_layer_12

```text
[68, 12]
[8, 152]
```

### 10. mean_pool_layer_03

```text
[67, 13]
[10, 150]
```
