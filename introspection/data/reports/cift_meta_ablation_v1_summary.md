# CIFT Meta-Head Ablation

## Source

- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Method: `activation_probe`
- Baseline feature: `concat(final_token_layer_11,final_token_layer_16)`
- Dataset count: `2`
- Variant count: `12`
- Best variant: `full_dual_readout_safe_secret_logistic_default`

## Variant Summary

| Variant | Calibration Labels | Source Count | Decision Rule | Candidate Errors | Fixed | Persistent | Introduced | Net Error Delta | Mean Accuracy |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| `full_dual_readout_safe_secret_logistic_default` | `secret_present_safe` | 14 | `logistic_default` | 12 | 3 | 6 | 6 | 3 | 0.9000 |
| `full_dual_readout_safe_secret_train_f1_threshold` | `secret_present_safe` | 14 | `train_f1_threshold` | 14 | 3 | 6 | 8 | 5 | 0.8833 |
| `full_dual_readout_nonleaking_logistic_default` | `benign`, `secret_present_safe` | 14 | `logistic_default` | 12 | 3 | 6 | 6 | 3 | 0.9000 |
| `full_dual_readout_nonleaking_train_f1_threshold` | `benign`, `secret_present_safe` | 14 | `train_f1_threshold` | 14 | 3 | 6 | 8 | 5 | 0.8833 |
| `early_dual_readout_safe_secret_logistic_default` | `secret_present_safe` | 8 | `logistic_default` | 22 | 1 | 8 | 14 | 13 | 0.8167 |
| `early_dual_readout_safe_secret_train_f1_threshold` | `secret_present_safe` | 8 | `train_f1_threshold` | 25 | 1 | 8 | 17 | 16 | 0.7917 |
| `early_dual_readout_nonleaking_logistic_default` | `benign`, `secret_present_safe` | 8 | `logistic_default` | 22 | 1 | 8 | 14 | 13 | 0.8167 |
| `early_dual_readout_nonleaking_train_f1_threshold` | `benign`, `secret_present_safe` | 8 | `train_f1_threshold` | 25 | 1 | 8 | 17 | 16 | 0.7917 |
| `early_final_token_safe_secret_logistic_default` | `secret_present_safe` | 4 | `logistic_default` | 21 | 1 | 8 | 13 | 12 | 0.8250 |
| `early_final_token_safe_secret_train_f1_threshold` | `secret_present_safe` | 4 | `train_f1_threshold` | 27 | 1 | 8 | 19 | 18 | 0.7750 |
| `early_final_token_nonleaking_logistic_default` | `benign`, `secret_present_safe` | 4 | `logistic_default` | 21 | 1 | 8 | 13 | 12 | 0.8250 |
| `early_final_token_nonleaking_train_f1_threshold` | `benign`, `secret_present_safe` | 4 | `train_f1_threshold` | 27 | 1 | 8 | 19 | 18 | 0.7750 |

## Dataset Variant Results

| Dataset | Variant | Candidate Errors | Fixed | Persistent | Introduced | Candidate Accuracy |
|---|---|---:|---:|---:|---:|---:|
| `hard_prompts_v2` | `full_dual_readout_safe_secret_logistic_default` | 7 | 0 | 2 | 5 | 0.8833 |
| `hard_prompts_v3` | `full_dual_readout_safe_secret_logistic_default` | 5 | 3 | 4 | 1 | 0.9167 |
| `hard_prompts_v2` | `full_dual_readout_safe_secret_train_f1_threshold` | 8 | 0 | 2 | 6 | 0.8667 |
| `hard_prompts_v3` | `full_dual_readout_safe_secret_train_f1_threshold` | 6 | 3 | 4 | 2 | 0.9000 |
| `hard_prompts_v2` | `full_dual_readout_nonleaking_logistic_default` | 7 | 0 | 2 | 5 | 0.8833 |
| `hard_prompts_v3` | `full_dual_readout_nonleaking_logistic_default` | 5 | 3 | 4 | 1 | 0.9167 |
| `hard_prompts_v2` | `full_dual_readout_nonleaking_train_f1_threshold` | 8 | 0 | 2 | 6 | 0.8667 |
| `hard_prompts_v3` | `full_dual_readout_nonleaking_train_f1_threshold` | 6 | 3 | 4 | 2 | 0.9000 |
| `hard_prompts_v2` | `early_dual_readout_safe_secret_logistic_default` | 9 | 0 | 2 | 7 | 0.8500 |
| `hard_prompts_v3` | `early_dual_readout_safe_secret_logistic_default` | 13 | 1 | 6 | 7 | 0.7833 |
| `hard_prompts_v2` | `early_dual_readout_safe_secret_train_f1_threshold` | 12 | 0 | 2 | 10 | 0.8000 |
| `hard_prompts_v3` | `early_dual_readout_safe_secret_train_f1_threshold` | 13 | 1 | 6 | 7 | 0.7833 |
| `hard_prompts_v2` | `early_dual_readout_nonleaking_logistic_default` | 9 | 0 | 2 | 7 | 0.8500 |
| `hard_prompts_v3` | `early_dual_readout_nonleaking_logistic_default` | 13 | 1 | 6 | 7 | 0.7833 |
| `hard_prompts_v2` | `early_dual_readout_nonleaking_train_f1_threshold` | 12 | 0 | 2 | 10 | 0.8000 |
| `hard_prompts_v3` | `early_dual_readout_nonleaking_train_f1_threshold` | 13 | 1 | 6 | 7 | 0.7833 |
| `hard_prompts_v2` | `early_final_token_safe_secret_logistic_default` | 9 | 0 | 2 | 7 | 0.8500 |
| `hard_prompts_v3` | `early_final_token_safe_secret_logistic_default` | 12 | 1 | 6 | 6 | 0.8000 |
| `hard_prompts_v2` | `early_final_token_safe_secret_train_f1_threshold` | 15 | 0 | 2 | 13 | 0.7500 |
| `hard_prompts_v3` | `early_final_token_safe_secret_train_f1_threshold` | 12 | 1 | 6 | 6 | 0.8000 |
| `hard_prompts_v2` | `early_final_token_nonleaking_logistic_default` | 9 | 0 | 2 | 7 | 0.8500 |
| `hard_prompts_v3` | `early_final_token_nonleaking_logistic_default` | 12 | 1 | 6 | 6 | 0.8000 |
| `hard_prompts_v2` | `early_final_token_nonleaking_train_f1_threshold` | 15 | 0 | 2 | 13 | 0.7500 |
| `hard_prompts_v3` | `early_final_token_nonleaking_train_f1_threshold` | 12 | 1 | 6 | 6 | 0.8000 |