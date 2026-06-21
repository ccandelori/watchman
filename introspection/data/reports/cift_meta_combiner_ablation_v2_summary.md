# CIFT Meta-Head Combiner Ablation

## Source

- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Method: `activation_probe`
- Baseline feature: `concat(final_token_layer_11,final_token_layer_16)`
- Dataset count: `2`
- Variant count: `7`
- Best variant: `logistic_meta_head`

## Variant Summary

| Variant | Combiner Rule | Source Count | Calibration Labels | Candidate Errors | Fixed | Persistent | Introduced | Net Error Delta | Mean Accuracy |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| `logistic_meta_head` | `logistic_meta_head` | 14 | `secret_present_safe` | 12 | 3 | 6 | 6 | 3 | 0.9000 |
| `mean_score` | `mean_score` | 14 | `secret_present_safe` | 26 | 2 | 7 | 19 | 17 | 0.7833 |
| `max_score` | `max_score` | 14 | `secret_present_safe` | 42 | 1 | 8 | 34 | 33 | 0.6500 |
| `top_two_mean` | `top_two_mean` | 14 | `secret_present_safe` | 41 | 1 | 8 | 33 | 32 | 0.6583 |
| `majority_vote` | `majority_vote` | 14 | `secret_present_safe` | 27 | 2 | 7 | 20 | 18 | 0.7750 |
| `positive_logistic` | `positive_logistic` | 14 | `secret_present_safe` | 23 | 3 | 6 | 17 | 14 | 0.8083 |
| `simplex_logistic` | `simplex_logistic` | 14 | `secret_present_safe` | 23 | 3 | 6 | 17 | 14 | 0.8083 |

## Dataset Variant Results

| Dataset | Variant | Candidate Errors | Fixed | Persistent | Introduced | Candidate Accuracy |
|---|---|---:|---:|---:|---:|---:|
| `hard_prompts_v2` | `logistic_meta_head` | 7 | 0 | 2 | 5 | 0.8833 |
| `hard_prompts_v3` | `logistic_meta_head` | 5 | 3 | 4 | 1 | 0.9167 |
| `hard_prompts_v2` | `mean_score` | 13 | 0 | 2 | 11 | 0.7833 |
| `hard_prompts_v3` | `mean_score` | 13 | 2 | 5 | 8 | 0.7833 |
| `hard_prompts_v2` | `max_score` | 21 | 0 | 2 | 19 | 0.6500 |
| `hard_prompts_v3` | `max_score` | 21 | 1 | 6 | 15 | 0.6500 |
| `hard_prompts_v2` | `top_two_mean` | 20 | 0 | 2 | 18 | 0.6667 |
| `hard_prompts_v3` | `top_two_mean` | 21 | 1 | 6 | 15 | 0.6500 |
| `hard_prompts_v2` | `majority_vote` | 13 | 0 | 2 | 11 | 0.7833 |
| `hard_prompts_v3` | `majority_vote` | 14 | 2 | 5 | 9 | 0.7667 |
| `hard_prompts_v2` | `positive_logistic` | 11 | 0 | 2 | 9 | 0.8167 |
| `hard_prompts_v3` | `positive_logistic` | 12 | 3 | 4 | 8 | 0.8000 |
| `hard_prompts_v2` | `simplex_logistic` | 11 | 0 | 2 | 9 | 0.8167 |
| `hard_prompts_v3` | `simplex_logistic` | 12 | 3 | 4 | 8 | 0.8000 |