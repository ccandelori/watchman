# Binary Layer Sweep

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `cpu`
- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Reference feature: `readout_window_layer_11`
- Best feature: `readout_window_layer_21`
- Feature count: `87`

## Feature Ranking

| Rank | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---:|---|---:|---:|---:|---:|
| 1 | `readout_window_layer_21` | 0.5284 | 0.5375 | 0.0638 | 0.0606 |
| 2 | `final_token_layer_19` | 0.5155 | 0.5188 | 0.0457 | 0.0424 |
| 3 | `mean_pool_layer_06` | 0.5126 | 0.5250 | 0.0499 | 0.0500 |
| 4 | `final_token_layer_16` | 0.5111 | 0.5375 | 0.0510 | 0.0500 |
| 5 | `mean_pool_layer_05` | 0.5073 | 0.5500 | 0.0390 | 0.0153 |
| 6 | `mean_pool_layer_22` | 0.5048 | 0.5437 | 0.0895 | 0.0643 |
| 7 | `final_token_layer_20` | 0.5009 | 0.5125 | 0.0232 | 0.0250 |
| 8 | `readout_window_layer_20` | 0.5008 | 0.5188 | 0.0417 | 0.0250 |
| 9 | `readout_window_layer_16` | 0.4993 | 0.5188 | 0.0665 | 0.0508 |
| 10 | `final_token_layer_18` | 0.4988 | 0.5188 | 0.0426 | 0.0319 |
| 11 | `readout_window_layer_04` | 0.4958 | 0.5250 | 0.0385 | 0.0459 |
| 12 | `mean_pool_layer_01` | 0.4954 | 0.5125 | 0.0671 | 0.0673 |
| 13 | `mean_pool_layer_07` | 0.4937 | 0.5000 | 0.0536 | 0.0523 |
| 14 | `final_token_layer_17` | 0.4930 | 0.5188 | 0.0561 | 0.0375 |
| 15 | `mean_pool_layer_00` | 0.4909 | 0.5000 | 0.0412 | 0.0342 |
| 16 | `mean_pool_layer_03` | 0.4905 | 0.5125 | 0.0440 | 0.0424 |
| 17 | `readout_window_layer_15` | 0.4902 | 0.5188 | 0.0481 | 0.0468 |
| 18 | `final_token_layer_23` | 0.4891 | 0.5125 | 0.0358 | 0.0319 |
| 19 | `final_token_layer_21` | 0.4879 | 0.4938 | 0.0290 | 0.0234 |
| 20 | `final_token_layer_25` | 0.4821 | 0.5125 | 0.0452 | 0.0250 |
| 21 | `mean_pool_layer_09` | 0.4804 | 0.5062 | 0.0414 | 0.0459 |
| 22 | `readout_window_layer_05` | 0.4802 | 0.5188 | 0.0782 | 0.0250 |
| 23 | `final_token_layer_27` | 0.4779 | 0.5062 | 0.0366 | 0.0125 |
| 24 | `mean_pool_layer_04` | 0.4757 | 0.5500 | 0.0540 | 0.0250 |
| 25 | `final_token_layer_08` | 0.4747 | 0.5188 | 0.0913 | 0.0805 |
| 26 | `mean_pool_layer_14` | 0.4745 | 0.5188 | 0.0862 | 0.0319 |
| 27 | `mean_pool_layer_25` | 0.4731 | 0.5312 | 0.0836 | 0.0625 |
| 28 | `final_token_layer_22` | 0.4715 | 0.4875 | 0.0519 | 0.0319 |
| 29 | `final_token_layer_15` | 0.4693 | 0.4875 | 0.0410 | 0.0319 |
| 30 | `final_token_layer_13` | 0.4679 | 0.4750 | 0.0322 | 0.0306 |
| 31 | `final_token_layer_11` | 0.4671 | 0.4875 | 0.0431 | 0.0424 |
| 32 | `mean_pool_layer_26` | 0.4664 | 0.5312 | 0.0653 | 0.0342 |
| 33 | `final_token_layer_02` | 0.4628 | 0.4938 | 0.0897 | 0.0459 |
| 34 | `readout_window_layer_18` | 0.4619 | 0.5250 | 0.0777 | 0.0606 |
| 35 | `mean_pool_layer_10` | 0.4616 | 0.4875 | 0.0600 | 0.0580 |
| 36 | `readout_window_layer_13` | 0.4578 | 0.4750 | 0.0357 | 0.0364 |
| 37 | `mean_pool_layer_12` | 0.4566 | 0.4813 | 0.0767 | 0.0673 |
| 38 | `mean_pool_layer_02` | 0.4562 | 0.4875 | 0.0595 | 0.0508 |
| 39 | `readout_window_layer_27` | 0.4552 | 0.5188 | 0.0776 | 0.0153 |
| 40 | `readout_window_layer_14` | 0.4518 | 0.4750 | 0.0246 | 0.0234 |
| 41 | `final_token_layer_05` | 0.4514 | 0.4813 | 0.0805 | 0.0643 |
| 42 | `readout_window_layer_01` | 0.4514 | 0.5062 | 0.0643 | 0.0306 |
| 43 | `mean_pool_layer_28` | 0.4509 | 0.5250 | 0.1035 | 0.0538 |
| 44 | `final_token_layer_14` | 0.4506 | 0.4688 | 0.0179 | 0.0198 |
| 45 | `mean_pool_layer_11` | 0.4503 | 0.5062 | 0.0355 | 0.0306 |
| 46 | `mean_pool_layer_20` | 0.4502 | 0.4813 | 0.0359 | 0.0250 |
| 47 | `final_token_layer_24` | 0.4501 | 0.4875 | 0.0630 | 0.0250 |
| 48 | `mean_pool_layer_08` | 0.4500 | 0.4750 | 0.0532 | 0.0637 |
| 49 | `final_token_layer_28` | 0.4500 | 0.4938 | 0.0596 | 0.0125 |
| 50 | `readout_window_layer_09` | 0.4475 | 0.5188 | 0.0725 | 0.0153 |
| 51 | `readout_window_layer_07` | 0.4470 | 0.4875 | 0.0418 | 0.0375 |
| 52 | `readout_window_layer_28` | 0.4462 | 0.5000 | 0.0732 | 0.0198 |
| 53 | `readout_window_layer_25` | 0.4448 | 0.5000 | 0.0640 | 0.0342 |
| 54 | `final_token_layer_09` | 0.4447 | 0.4688 | 0.0619 | 0.0713 |
| 55 | `readout_window_layer_22` | 0.4433 | 0.4938 | 0.0553 | 0.0125 |
| 56 | `mean_pool_layer_15` | 0.4392 | 0.5000 | 0.0968 | 0.0484 |
| 57 | `readout_window_layer_08` | 0.4372 | 0.4938 | 0.0561 | 0.0234 |
| 58 | `final_token_layer_03` | 0.4369 | 0.4938 | 0.0694 | 0.0364 |
| 59 | `mean_pool_layer_24` | 0.4359 | 0.5000 | 0.0439 | 0.0198 |
| 60 | `readout_window_layer_03` | 0.4350 | 0.4562 | 0.1033 | 0.0805 |
| 61 | `mean_pool_layer_13` | 0.4334 | 0.4875 | 0.1052 | 0.0940 |
| 62 | `mean_pool_layer_16` | 0.4331 | 0.4750 | 0.0691 | 0.0606 |
| 63 | `readout_window_layer_26` | 0.4328 | 0.5062 | 0.0837 | 0.0125 |
| 64 | `readout_window_layer_17` | 0.4301 | 0.5062 | 0.0585 | 0.0415 |
| 65 | `final_token_layer_12` | 0.4243 | 0.4688 | 0.0757 | 0.0559 |
| 66 | `mean_pool_layer_21` | 0.4232 | 0.4750 | 0.0245 | 0.0234 |
| 67 | `final_token_layer_26` | 0.4228 | 0.4813 | 0.0685 | 0.0250 |
| 68 | `mean_pool_layer_27` | 0.4225 | 0.5188 | 0.0587 | 0.0250 |
| 69 | `mean_pool_layer_17` | 0.4224 | 0.4625 | 0.0288 | 0.0538 |
| 70 | `mean_pool_layer_23` | 0.4217 | 0.4875 | 0.0647 | 0.0545 |
| 71 | `mean_pool_layer_18` | 0.4169 | 0.4813 | 0.0734 | 0.0319 |
| 72 | `final_token_layer_10` | 0.4163 | 0.4437 | 0.0758 | 0.0750 |
| 73 | `readout_window_layer_24` | 0.4147 | 0.5000 | 0.0786 | 0.0342 |
| 74 | `readout_window_layer_11` (reference) | 0.4138 | 0.4625 | 0.0619 | 0.0696 |
| 75 | `final_token_layer_07` | 0.4121 | 0.4688 | 0.0522 | 0.0280 |
| 76 | `final_token_layer_06` | 0.4115 | 0.4875 | 0.0447 | 0.0319 |
| 77 | `final_token_layer_01` | 0.4107 | 0.4437 | 0.0764 | 0.0824 |
| 78 | `readout_window_layer_10` | 0.4077 | 0.4813 | 0.0334 | 0.0545 |
| 79 | `final_token_layer_04` | 0.3990 | 0.4437 | 0.0453 | 0.0306 |
| 80 | `readout_window_layer_23` | 0.3986 | 0.4938 | 0.0696 | 0.0125 |
| 81 | `readout_window_layer_19` | 0.3964 | 0.5125 | 0.0772 | 0.0153 |
| 82 | `readout_window_layer_12` | 0.3948 | 0.4125 | 0.0928 | 0.0776 |
| 83 | `mean_pool_layer_19` | 0.3888 | 0.4562 | 0.0845 | 0.0508 |
| 84 | `readout_window_layer_02` | 0.3772 | 0.4437 | 0.0655 | 0.0459 |
| 85 | `readout_window_layer_06` | 0.3751 | 0.4500 | 0.0339 | 0.0424 |
| 86 | `final_token_layer_00` | 0.3333 | 0.5000 | 0.0000 | 0.0000 |
| 87 | `readout_window_layer_00` | 0.3333 | 0.5000 | 0.0000 | 0.0000 |

## Top Confusion Matrices

### 1. readout_window_layer_21

```text
[40, 40]
[34, 46]
```

### 2. final_token_layer_19

```text
[38, 42]
[35, 45]
```

### 3. mean_pool_layer_06

```text
[36, 44]
[32, 48]
```

### 4. final_token_layer_16

```text
[44, 36]
[38, 42]
```

### 5. mean_pool_layer_05

```text
[34, 46]
[26, 54]
```

### 6. mean_pool_layer_22

```text
[42, 38]
[35, 45]
```

### 7. final_token_layer_20

```text
[33, 47]
[31, 49]
```

### 8. readout_window_layer_20

```text
[32, 48]
[29, 51]
```

### 9. readout_window_layer_16

```text
[35, 45]
[32, 48]
```

### 10. final_token_layer_18

```text
[45, 35]
[42, 38]
```
