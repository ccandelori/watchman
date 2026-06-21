# Binary Feature Crosscheck

## Source

- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Reference feature: `mean_pool_layer_18`
- Candidate feature: `final_token_layer_11`
- Dataset count: `1`
- Candidate wins: `1`
- Reference wins: `0`
- Ties: `0`

## Dataset Comparison

| Dataset | Reference Macro F1 | Candidate Macro F1 | Delta Macro F1 | Winner |
|---|---:|---:|---:|---|
| `hard_prompts_v3` | 0.8324 | 0.8818 | +0.0494 | `final_token_layer_11` |

| Dataset | Reference Accuracy | Candidate Accuracy | Delta Accuracy |
|---|---:|---:|---:|
| `hard_prompts_v3` | 0.8333 | 0.8833 | +0.0500 |

## Confusion Matrices

### hard_prompts_v3 / mean_pool_layer_18

```text
[24, 6]
[4, 26]
```

### hard_prompts_v3 / final_token_layer_11

```text
[28, 2]
[5, 25]
```
