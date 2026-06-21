# CIFT Meta-Head Regularization Sweep

## Source

- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Method: `activation_probe`
- Baseline feature: `concat(final_token_layer_11,final_token_layer_16)`
- Source-head C: `1.0`
- Dataset count: `4`
- Variant count: `9`
- Best variant: `meta_c_10`

## Variant Summary

| Variant | Meta C | Source Count | Calibration Labels | Candidate Errors | Fixed | Persistent | Introduced | Net Error Delta | Mean Accuracy |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `meta_c_0_03` | 0.03 | 14 | `secret_present_safe` | 42 | 5 | 15 | 27 | 22 | 0.8250 |
| `meta_c_0_05` | 0.05 | 14 | `secret_present_safe` | 40 | 5 | 15 | 25 | 20 | 0.8333 |
| `meta_c_0_1` | 0.1 | 14 | `secret_present_safe` | 39 | 4 | 16 | 23 | 19 | 0.8375 |
| `meta_c_0_25` | 0.25 | 14 | `secret_present_safe` | 34 | 4 | 16 | 18 | 14 | 0.8583 |
| `meta_c_0_5` | 0.5 | 14 | `secret_present_safe` | 31 | 6 | 14 | 17 | 11 | 0.8708 |
| `meta_c_1` | 1 | 14 | `secret_present_safe` | 27 | 7 | 13 | 14 | 7 | 0.8875 |
| `meta_c_2` | 2 | 14 | `secret_present_safe` | 28 | 8 | 12 | 16 | 8 | 0.8833 |
| `meta_c_5` | 5 | 14 | `secret_present_safe` | 25 | 10 | 10 | 15 | 5 | 0.8958 |
| `meta_c_10` | 10 | 14 | `secret_present_safe` | 23 | 11 | 9 | 14 | 3 | 0.9042 |

## Dataset Variant Results

| Dataset | Variant | Candidate Errors | Fixed | Persistent | Introduced | Candidate Accuracy |
|---|---|---:|---:|---:|---:|---:|
| `baseline_prompts_v1` | `meta_c_0_03` | 8 | 2 | 5 | 3 | 0.8667 |
| `hard_prompts_v1` | `meta_c_0_03` | 11 | 1 | 3 | 8 | 0.8167 |
| `hard_prompts_v2` | `meta_c_0_03` | 10 | 0 | 2 | 8 | 0.8333 |
| `hard_prompts_v3` | `meta_c_0_03` | 13 | 2 | 5 | 8 | 0.7833 |
| `baseline_prompts_v1` | `meta_c_0_05` | 8 | 2 | 5 | 3 | 0.8667 |
| `hard_prompts_v1` | `meta_c_0_05` | 11 | 1 | 3 | 8 | 0.8167 |
| `hard_prompts_v2` | `meta_c_0_05` | 9 | 0 | 2 | 7 | 0.8500 |
| `hard_prompts_v3` | `meta_c_0_05` | 12 | 2 | 5 | 7 | 0.8000 |
| `baseline_prompts_v1` | `meta_c_0_1` | 9 | 2 | 5 | 4 | 0.8500 |
| `hard_prompts_v1` | `meta_c_0_1` | 10 | 1 | 3 | 7 | 0.8333 |
| `hard_prompts_v2` | `meta_c_0_1` | 8 | 0 | 2 | 6 | 0.8667 |
| `hard_prompts_v3` | `meta_c_0_1` | 12 | 1 | 6 | 6 | 0.8000 |
| `baseline_prompts_v1` | `meta_c_0_25` | 8 | 2 | 5 | 3 | 0.8667 |
| `hard_prompts_v1` | `meta_c_0_25` | 8 | 1 | 3 | 5 | 0.8667 |
| `hard_prompts_v2` | `meta_c_0_25` | 7 | 0 | 2 | 5 | 0.8833 |
| `hard_prompts_v3` | `meta_c_0_25` | 11 | 1 | 6 | 5 | 0.8167 |
| `baseline_prompts_v1` | `meta_c_0_5` | 8 | 2 | 5 | 3 | 0.8667 |
| `hard_prompts_v1` | `meta_c_0_5` | 8 | 2 | 2 | 6 | 0.8667 |
| `hard_prompts_v2` | `meta_c_0_5` | 7 | 0 | 2 | 5 | 0.8833 |
| `hard_prompts_v3` | `meta_c_0_5` | 8 | 2 | 5 | 3 | 0.8667 |
| `baseline_prompts_v1` | `meta_c_1` | 8 | 2 | 5 | 3 | 0.8667 |
| `hard_prompts_v1` | `meta_c_1` | 7 | 2 | 2 | 5 | 0.8833 |
| `hard_prompts_v2` | `meta_c_1` | 7 | 0 | 2 | 5 | 0.8833 |
| `hard_prompts_v3` | `meta_c_1` | 5 | 3 | 4 | 1 | 0.9167 |
| `baseline_prompts_v1` | `meta_c_2` | 7 | 3 | 4 | 3 | 0.8833 |
| `hard_prompts_v1` | `meta_c_2` | 8 | 2 | 2 | 6 | 0.8667 |
| `hard_prompts_v2` | `meta_c_2` | 8 | 0 | 2 | 6 | 0.8667 |
| `hard_prompts_v3` | `meta_c_2` | 5 | 3 | 4 | 1 | 0.9167 |
| `baseline_prompts_v1` | `meta_c_5` | 8 | 3 | 4 | 4 | 0.8667 |
| `hard_prompts_v1` | `meta_c_5` | 8 | 2 | 2 | 6 | 0.8667 |
| `hard_prompts_v2` | `meta_c_5` | 6 | 1 | 1 | 5 | 0.9000 |
| `hard_prompts_v3` | `meta_c_5` | 3 | 4 | 3 | 0 | 0.9500 |
| `baseline_prompts_v1` | `meta_c_10` | 7 | 4 | 3 | 4 | 0.8833 |
| `hard_prompts_v1` | `meta_c_10` | 7 | 2 | 2 | 5 | 0.8833 |
| `hard_prompts_v2` | `meta_c_10` | 6 | 1 | 1 | 5 | 0.9000 |
| `hard_prompts_v3` | `meta_c_10` | 3 | 4 | 3 | 0 | 0.9500 |