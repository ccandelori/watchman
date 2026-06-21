# Binary Feature Stability

## Source

- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Dataset count: `4`
- Feature count: `3`
- Fold count: `5`

## Feature Summary

| Rank | Feature | Wins | Mean Macro F1 | Mean Accuracy | Min Macro F1 | Max Macro F1 | Macro F1 Range |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `final_token_layer_11` | 2 | 0.8978 | 0.9000 | 0.8445 | 0.9657 | 0.1212 |
| 2 | `final_token_layer_16` | 2 | 0.8902 | 0.8917 | 0.8655 | 0.9321 | 0.0667 |
| 3 | `mean_pool_layer_18` | 0 | 0.8239 | 0.8292 | 0.7225 | 0.8788 | 0.1563 |

## Macro F1 by Dataset

| Dataset | Winner | `mean_pool_layer_18` | `final_token_layer_11` | `final_token_layer_16` |
|---|---|---:|---:|---:|
| `baseline_prompts_v1` | `final_token_layer_16` | 0.8620 | 0.8445 | 0.8804 |
| `hard_prompts_v1` | `final_token_layer_11` | 0.8788 | 0.8993 | 0.8655 |
| `hard_prompts_v2` | `final_token_layer_11` | 0.7225 | 0.9657 | 0.8828 |
| `hard_prompts_v3` | `final_token_layer_16` | 0.8324 | 0.8818 | 0.9321 |

## Accuracy by Dataset

| Dataset | Winner | `mean_pool_layer_18` | `final_token_layer_11` | `final_token_layer_16` |
|---|---|---:|---:|---:|
| `baseline_prompts_v1` | `final_token_layer_16` | 0.8667 | 0.8500 | 0.8833 |
| `hard_prompts_v1` | `final_token_layer_11` | 0.8833 | 0.9000 | 0.8667 |
| `hard_prompts_v2` | `final_token_layer_11` | 0.7333 | 0.9667 | 0.8833 |
| `hard_prompts_v3` | `final_token_layer_16` | 0.8333 | 0.8833 | 0.9333 |
