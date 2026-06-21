# CIFT Meta-Head Source-Score Calibration

## Source

- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Method: `activation_probe`
- Baseline feature: `concat(final_token_layer_11,final_token_layer_16)`
- Source-head C: `1.0`
- Meta-head C: `10.0`
- Dataset count: `2`
- Variant count: `3`
- Best variant: `raw_probability`

## Variant Summary

| Variant | Calibration Rule | Meta C | Source Count | Calibration Labels | Candidate Errors | Fixed | Persistent | Introduced | Net Error Delta | Mean Accuracy |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `raw_probability` | `raw_probability` | 10 | 14 | `secret_present_safe` | 9 | 5 | 4 | 5 | 0 | 0.9250 |
| `clipped_logit` | `clipped_logit` | 10 | 14 | `secret_present_safe` | 11 | 7 | 2 | 9 | 2 | 0.9083 |
| `platt_probability` | `platt_probability` | 10 | 14 | `secret_present_safe` | 10 | 5 | 4 | 6 | 1 | 0.9167 |

## Dataset Variant Results

| Dataset | Variant | Candidate Errors | Fixed | Persistent | Introduced | Candidate Accuracy |
|---|---|---:|---:|---:|---:|---:|
| `hard_prompts_v2` | `raw_probability` | 6 | 1 | 1 | 5 | 0.9000 |
| `hard_prompts_v3` | `raw_probability` | 3 | 4 | 3 | 0 | 0.9500 |
| `hard_prompts_v2` | `clipped_logit` | 9 | 1 | 1 | 8 | 0.8500 |
| `hard_prompts_v3` | `clipped_logit` | 2 | 6 | 1 | 1 | 0.9667 |
| `hard_prompts_v2` | `platt_probability` | 7 | 1 | 1 | 6 | 0.8833 |
| `hard_prompts_v3` | `platt_probability` | 3 | 4 | 3 | 0 | 0.9500 |