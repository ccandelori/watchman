# CIFT Meta-Head Ablation

## Source

- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Method: `activation_probe`
- Baseline feature: `concat(final_token_layer_11,final_token_layer_16)`
- Dataset count: `2`
- Variant count: `7`
- Best variant: `full_dual_readout`

## Variant Summary

| Variant | Calibration Labels | Source Count | Decision Rule | Candidate Errors | Fixed | Persistent | Introduced | Net Error Delta | Mean Accuracy |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| `full_dual_readout` | `secret_present_safe` | 14 | `logistic_default` | 12 | 3 | 6 | 6 | 3 | 0.9000 |
| `drop_last_mean_pool` | `secret_present_safe` | 13 | `logistic_default` | 12 | 4 | 5 | 7 | 3 | 0.9000 |
| `drop_last_two_mean_pool` | `secret_present_safe` | 12 | `logistic_default` | 15 | 2 | 7 | 8 | 6 | 0.8750 |
| `drop_last_final_token` | `secret_present_safe` | 13 | `logistic_default` | 14 | 3 | 6 | 8 | 5 | 0.8833 |
| `drop_last_dual_readout_layer` | `secret_present_safe` | 12 | `logistic_default` | 15 | 2 | 7 | 8 | 6 | 0.8750 |
| `final_token_only` | `secret_present_safe` | 7 | `logistic_default` | 20 | 1 | 8 | 12 | 11 | 0.8333 |
| `mean_pool_only` | `secret_present_safe` | 7 | `logistic_default` | 17 | 7 | 2 | 15 | 8 | 0.8583 |

## Dataset Variant Results

| Dataset | Variant | Candidate Errors | Fixed | Persistent | Introduced | Candidate Accuracy |
|---|---|---:|---:|---:|---:|---:|
| `hard_prompts_v2` | `full_dual_readout` | 7 | 0 | 2 | 5 | 0.8833 |
| `hard_prompts_v3` | `full_dual_readout` | 5 | 3 | 4 | 1 | 0.9167 |
| `hard_prompts_v2` | `drop_last_mean_pool` | 7 | 0 | 2 | 5 | 0.8833 |
| `hard_prompts_v3` | `drop_last_mean_pool` | 5 | 4 | 3 | 2 | 0.9167 |
| `hard_prompts_v2` | `drop_last_two_mean_pool` | 8 | 0 | 2 | 6 | 0.8667 |
| `hard_prompts_v3` | `drop_last_two_mean_pool` | 7 | 2 | 5 | 2 | 0.8833 |
| `hard_prompts_v2` | `drop_last_final_token` | 7 | 0 | 2 | 5 | 0.8833 |
| `hard_prompts_v3` | `drop_last_final_token` | 7 | 3 | 4 | 3 | 0.8833 |
| `hard_prompts_v2` | `drop_last_dual_readout_layer` | 7 | 0 | 2 | 5 | 0.8833 |
| `hard_prompts_v3` | `drop_last_dual_readout_layer` | 8 | 2 | 5 | 3 | 0.8667 |
| `hard_prompts_v2` | `final_token_only` | 10 | 0 | 2 | 8 | 0.8333 |
| `hard_prompts_v3` | `final_token_only` | 10 | 1 | 6 | 4 | 0.8333 |
| `hard_prompts_v2` | `mean_pool_only` | 10 | 2 | 0 | 10 | 0.8333 |
| `hard_prompts_v3` | `mean_pool_only` | 7 | 5 | 2 | 5 | 0.8833 |