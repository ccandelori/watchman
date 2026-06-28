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
- Variant count: `1`

## Variant Ranking

| Rank | Variant | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---:|---|---|---:|---:|---:|---:|
| 1 | `candidate_final_token_l12` (baseline) | `final_token_layer_12` | 0.9790 | 0.9811 | 0.0189 | 0.0172 |

## Top Confusion Matrices

### 1. candidate_final_token_l12

```text
[1456, 20]
[60, 2892]
```
