# Binary Feature Crosscheck

## Source

- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Reference feature: `mean_pool_layer_18`
- Candidate feature: `final_token_layer_11`
- Dataset count: `3`
- Candidate wins: `2`
- Reference wins: `1`
- Ties: `0`

## Dataset Comparison

| Dataset | Reference Macro F1 | Candidate Macro F1 | Delta Macro F1 | Winner |
|---|---:|---:|---:|---|
| `baseline_prompts_v1` | 0.8620 | 0.8445 | -0.0175 | `mean_pool_layer_18` |
| `hard_prompts_v1` | 0.8788 | 0.8993 | +0.0205 | `final_token_layer_11` |
| `hard_prompts_v2` | 0.7225 | 0.9657 | +0.2432 | `final_token_layer_11` |

| Dataset | Reference Accuracy | Candidate Accuracy | Delta Accuracy |
|---|---:|---:|---:|
| `baseline_prompts_v1` | 0.8667 | 0.8500 | -0.0167 |
| `hard_prompts_v1` | 0.8833 | 0.9000 | +0.0167 |
| `hard_prompts_v2` | 0.7333 | 0.9667 | +0.2333 |

## Confusion Matrices

### baseline_prompts_v1 / mean_pool_layer_18

```text
[26, 4]
[4, 26]
```

### baseline_prompts_v1 / final_token_layer_11

```text
[25, 5]
[4, 26]
```

### hard_prompts_v1 / mean_pool_layer_18

```text
[27, 3]
[4, 26]
```

### hard_prompts_v1 / final_token_layer_11

```text
[27, 3]
[3, 27]
```

### hard_prompts_v2 / mean_pool_layer_18

```text
[21, 9]
[7, 23]
```

### hard_prompts_v2 / final_token_layer_11

```text
[30, 0]
[2, 28]
```
