# CIFT Meta-Head Readout Family Comparison

## Source

- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Method: `activation_probe`
- Baseline feature: `concat(final_token_layer_11,final_token_layer_16)`
- Source-head C: `1.0`
- Meta-head C: `10.0`
- Dataset count: `2`
- Variant count: `3`
- Best variant: `full_dual_readout`

## Variant Summary

| Variant | Source Family | Meta C | Source Count | Calibration Labels | Candidate Errors | Fixed | Persistent | Introduced | Net Error Delta | Mean Accuracy |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `full_dual_readout` | `full_dual_readout` | 10 | 14 | `secret_present_safe` | 9 | 5 | 4 | 5 | 0 | 0.9250 |
| `final_token_only` | `final_token_only` | 10 | 7 | `secret_present_safe` | 23 | 1 | 8 | 15 | 14 | 0.8083 |
| `mean_pool_only` | `mean_pool_only` | 10 | 7 | `secret_present_safe` | 11 | 8 | 1 | 10 | 2 | 0.9083 |

## Dataset Variant Results

| Dataset | Variant | Candidate Errors | Fixed | Persistent | Introduced | Candidate Accuracy |
|---|---|---:|---:|---:|---:|---:|
| `hard_prompts_v2` | `full_dual_readout` | 6 | 1 | 1 | 5 | 0.9000 |
| `hard_prompts_v3` | `full_dual_readout` | 3 | 4 | 3 | 0 | 0.9500 |
| `hard_prompts_v2` | `final_token_only` | 12 | 0 | 2 | 10 | 0.8000 |
| `hard_prompts_v3` | `final_token_only` | 11 | 1 | 6 | 5 | 0.8167 |
| `hard_prompts_v2` | `mean_pool_only` | 7 | 2 | 0 | 7 | 0.8833 |
| `hard_prompts_v3` | `mean_pool_only` | 4 | 6 | 1 | 3 | 0.9333 |