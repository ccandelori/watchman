# CIFT Feature Ablation

## Source

- Model: `Qwen/Qwen3-4B`
- Revision: `1cfa9a7208912126459214e8b04321603b3df60c`
- Extraction device: `mps`
- Evaluation strategy: `stratified_group_kfold`
- Task: `non_exfiltration_vs_exfiltration`
- Baseline variant: `candidate_final_token_l12`
- Baseline feature: `final_token_layer_12`
- Best variant: `candidate_final_token_l12`
- Best feature: `final_token_layer_12`
- Variant count: `4`

## Variant Ranking

| Rank | Variant | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---:|---|---|---:|---:|---:|---:|
| 1 | `candidate_final_token_l12` (baseline) | `final_token_layer_12` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2 | `final_token_l16` | `final_token_layer_16` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 3 | `reference_query_tail_l21` | `query_tail_window_layer_21` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 4 | `early_final_token_l08` | `final_token_layer_08` | 0.9937 | 0.9943 | 0.0126 | 0.0115 |

## Top Confusion Matrices

### 1. candidate_final_token_l12

```text
[1356, 0]
[0, 2712]
```

### 2. final_token_l16

```text
[1356, 0]
[0, 2712]
```

### 3. reference_query_tail_l21

```text
[1356, 0]
[0, 2712]
```

### 4. early_final_token_l08

```text
[1356, 0]
[20, 2692]
```
