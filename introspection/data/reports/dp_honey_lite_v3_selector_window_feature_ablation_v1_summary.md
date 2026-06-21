# CIFT Feature Ablation

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `cpu`
- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Baseline variant: `baseline_layer_15`
- Baseline feature: `readout_window_layer_15`
- Best variant: `selector_band_12_18`
- Best feature: `concat(readout_window_layer_12,readout_window_layer_13,readout_window_layer_14,readout_window_layer_15,readout_window_layer_16,readout_window_layer_17,readout_window_layer_18)`
- Variant count: `7`

## Variant Ranking

| Rank | Variant | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---:|---|---|---:|---:|---:|---:|
| 1 | `selector_band_12_18` | `concat(readout_window_layer_12,readout_window_layer_13,readout_window_layer_14,readout_window_layer_15,readout_window_layer_16,readout_window_layer_17,readout_window_layer_18)` | 0.8138 | 0.8187 | 0.1068 | 0.1035 |
| 2 | `local_concat_14_15_16` | `concat(readout_window_layer_14,readout_window_layer_15,readout_window_layer_16)` | 0.7793 | 0.7875 | 0.0492 | 0.0459 |
| 3 | `baseline_layer_15` (baseline) | `readout_window_layer_15` | 0.7736 | 0.7812 | 0.0709 | 0.0656 |
| 4 | `late_band_15_21` | `concat(readout_window_layer_15,readout_window_layer_16,readout_window_layer_17,readout_window_layer_18,readout_window_layer_19,readout_window_layer_20,readout_window_layer_21)` | 0.7680 | 0.7812 | 0.1271 | 0.1186 |
| 5 | `all_readout_layers_00_28` | `concat(readout_window_layer_00,readout_window_layer_01,readout_window_layer_02,readout_window_layer_03,readout_window_layer_04,readout_window_layer_05,readout_window_layer_06,readout_window_layer_07,readout_window_layer_08,readout_window_layer_09,readout_window_layer_10,readout_window_layer_11,readout_window_layer_12,readout_window_layer_13,readout_window_layer_14,readout_window_layer_15,readout_window_layer_16,readout_window_layer_17,readout_window_layer_18,readout_window_layer_19,readout_window_layer_20,readout_window_layer_21,readout_window_layer_22,readout_window_layer_23,readout_window_layer_24,readout_window_layer_25,readout_window_layer_26,readout_window_layer_27,readout_window_layer_28)` | 0.7109 | 0.7312 | 0.1502 | 0.1275 |
| 6 | `single_layer_21` | `readout_window_layer_21` | 0.6443 | 0.6687 | 0.1385 | 0.1179 |
| 7 | `last_quarter_22_28` | `concat(readout_window_layer_22,readout_window_layer_23,readout_window_layer_24,readout_window_layer_25,readout_window_layer_26,readout_window_layer_27,readout_window_layer_28)` | 0.5428 | 0.6062 | 0.1474 | 0.0781 |

## Top Confusion Matrices

### 1. selector_band_12_18

```text
[65, 15]
[14, 66]
```

### 2. local_concat_14_15_16

```text
[60, 20]
[14, 66]
```

### 3. baseline_layer_15

```text
[59, 21]
[14, 66]
```

### 4. late_band_15_21

```text
[61, 19]
[16, 64]
```

### 5. all_readout_layers_00_28

```text
[58, 22]
[21, 59]
```

### 6. single_layer_21

```text
[47, 33]
[20, 60]
```

### 7. last_quarter_22_28

```text
[54, 26]
[37, 43]
```
