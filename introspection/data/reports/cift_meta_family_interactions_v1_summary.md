# CIFT Meta-Head Family Interactions

## Source

- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Method: `activation_probe`
- Baseline feature: `concat(final_token_layer_11,final_token_layer_16)`
- Source-head C: `1.0`
- Meta-head C: `10.0`
- Dataset count: `2`
- Variant count: `3`
- Best variant: `raw_scores`

## Variant Summary

| Variant | Interaction Rule | Meta C | Source Count | Added Features | Meta Features | Calibration Labels | Candidate Errors | Fixed | Persistent | Introduced | Net Error Delta | Mean Accuracy |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `raw_scores` | `raw_scores` | 10 | 14 | 0 | 14 | `secret_present_safe` | 9 | 5 | 4 | 5 | 0 | 0.9250 |
| `family_means` | `family_means` | 10 | 14 | 2 | 16 | `secret_present_safe` | 10 | 5 | 4 | 6 | 1 | 0.9167 |
| `family_mean_gaps` | `family_mean_gaps` | 10 | 14 | 4 | 18 | `secret_present_safe` | 15 | 5 | 4 | 11 | 6 | 0.8750 |

## Dataset Variant Results

| Dataset | Variant | Candidate Errors | Fixed | Persistent | Introduced | Candidate Accuracy |
|---|---|---:|---:|---:|---:|---:|
| `hard_prompts_v2` | `raw_scores` | 6 | 1 | 1 | 5 | 0.9000 |
| `hard_prompts_v3` | `raw_scores` | 3 | 4 | 3 | 0 | 0.9500 |
| `hard_prompts_v2` | `family_means` | 7 | 1 | 1 | 6 | 0.8833 |
| `hard_prompts_v3` | `family_means` | 3 | 4 | 3 | 0 | 0.9500 |
| `hard_prompts_v2` | `family_mean_gaps` | 10 | 1 | 1 | 9 | 0.8333 |
| `hard_prompts_v3` | `family_mean_gaps` | 5 | 4 | 3 | 2 | 0.9167 |