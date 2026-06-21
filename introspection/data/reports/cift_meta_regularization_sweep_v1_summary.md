# CIFT Meta-Head Regularization Sweep

## Source

- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Method: `activation_probe`
- Baseline feature: `concat(final_token_layer_11,final_token_layer_16)`
- Source-head C: `1.0`
- Dataset count: `2`
- Variant count: `9`
- Best variant: `meta_c_5`

## Variant Summary

| Variant | Meta C | Source Count | Calibration Labels | Candidate Errors | Fixed | Persistent | Introduced | Net Error Delta | Mean Accuracy |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `meta_c_0_03` | 0.03 | 14 | `secret_present_safe` | 23 | 2 | 7 | 16 | 14 | 0.8083 |
| `meta_c_0_05` | 0.05 | 14 | `secret_present_safe` | 21 | 2 | 7 | 14 | 12 | 0.8250 |
| `meta_c_0_1` | 0.1 | 14 | `secret_present_safe` | 20 | 1 | 8 | 12 | 11 | 0.8333 |
| `meta_c_0_25` | 0.25 | 14 | `secret_present_safe` | 18 | 1 | 8 | 10 | 9 | 0.8500 |
| `meta_c_0_5` | 0.5 | 14 | `secret_present_safe` | 15 | 2 | 7 | 8 | 6 | 0.8750 |
| `meta_c_1` | 1 | 14 | `secret_present_safe` | 12 | 3 | 6 | 6 | 3 | 0.9000 |
| `meta_c_2` | 2 | 14 | `secret_present_safe` | 13 | 3 | 6 | 7 | 4 | 0.8917 |
| `meta_c_5` | 5 | 14 | `secret_present_safe` | 9 | 5 | 4 | 5 | 0 | 0.9250 |
| `meta_c_10` | 10 | 14 | `secret_present_safe` | 9 | 5 | 4 | 5 | 0 | 0.9250 |

## Dataset Variant Results

| Dataset | Variant | Candidate Errors | Fixed | Persistent | Introduced | Candidate Accuracy |
|---|---|---:|---:|---:|---:|---:|
| `hard_prompts_v2` | `meta_c_0_03` | 10 | 0 | 2 | 8 | 0.8333 |
| `hard_prompts_v3` | `meta_c_0_03` | 13 | 2 | 5 | 8 | 0.7833 |
| `hard_prompts_v2` | `meta_c_0_05` | 9 | 0 | 2 | 7 | 0.8500 |
| `hard_prompts_v3` | `meta_c_0_05` | 12 | 2 | 5 | 7 | 0.8000 |
| `hard_prompts_v2` | `meta_c_0_1` | 8 | 0 | 2 | 6 | 0.8667 |
| `hard_prompts_v3` | `meta_c_0_1` | 12 | 1 | 6 | 6 | 0.8000 |
| `hard_prompts_v2` | `meta_c_0_25` | 7 | 0 | 2 | 5 | 0.8833 |
| `hard_prompts_v3` | `meta_c_0_25` | 11 | 1 | 6 | 5 | 0.8167 |
| `hard_prompts_v2` | `meta_c_0_5` | 7 | 0 | 2 | 5 | 0.8833 |
| `hard_prompts_v3` | `meta_c_0_5` | 8 | 2 | 5 | 3 | 0.8667 |
| `hard_prompts_v2` | `meta_c_1` | 7 | 0 | 2 | 5 | 0.8833 |
| `hard_prompts_v3` | `meta_c_1` | 5 | 3 | 4 | 1 | 0.9167 |
| `hard_prompts_v2` | `meta_c_2` | 8 | 0 | 2 | 6 | 0.8667 |
| `hard_prompts_v3` | `meta_c_2` | 5 | 3 | 4 | 1 | 0.9167 |
| `hard_prompts_v2` | `meta_c_5` | 6 | 1 | 1 | 5 | 0.9000 |
| `hard_prompts_v3` | `meta_c_5` | 3 | 4 | 3 | 0 | 0.9500 |
| `hard_prompts_v2` | `meta_c_10` | 6 | 1 | 1 | 5 | 0.9000 |
| `hard_prompts_v3` | `meta_c_10` | 3 | 4 | 3 | 0 | 0.9500 |