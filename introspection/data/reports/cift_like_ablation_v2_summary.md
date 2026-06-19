# CIFT-Like Ablation

## Source

- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Baseline feature: `concat(final_token_layer_11,final_token_layer_16)`
- Dataset count: `4`
- Variant count: `8`
- Ablation wins: `0`
- Baseline wins: `4`
- Ties: `0`

## Best Variant by Dataset

| Dataset | Baseline Macro F1 | Best Variant | Best Variant Macro F1 | Delta Macro F1 | Winner |
|---|---:|---|---:|---:|---|
| `baseline_prompts_v1` | 0.8804 | `cift_residual_safe_secret_mean_pool_last_quarter` | 0.8623 | -0.0181 | `concat(final_token_layer_11,final_token_layer_16)` |
| `hard_prompts_v1` | 0.9331 | `cift_residual_safe_secret_final_token_last_quarter` | 0.8132 | -0.1199 | `concat(final_token_layer_11,final_token_layer_16)` |
| `hard_prompts_v2` | 0.9657 | `cift_residual_safe_secret_final_token_last_quarter` | 0.7808 | -0.1849 | `concat(final_token_layer_11,final_token_layer_16)` |
| `hard_prompts_v3` | 0.8811 | `cift_residual_safe_secret_final_token_last_quarter` | 0.7461 | -0.1350 | `concat(final_token_layer_11,final_token_layer_16)` |

## Aggregate by Variant

| Variant | Representation | Calibration Labels | Mean Macro F1 | Min Macro F1 |
|---|---|---|---:|---:|
| `cift_diag_safe_secret_final_token_last_quarter` | `diagonal_distance` | `secret_present_safe` | 0.5446 | 0.4076 |
| `cift_diag_nonleaking_final_token_last_quarter` | `diagonal_distance` | `benign`, `secret_present_safe` | 0.5783 | 0.4255 |
| `cift_residual_safe_secret_final_token_last_quarter` | `standardized_residual_concat` | `secret_present_safe` | 0.7924 | 0.7461 |
| `cift_residual_nonleaking_final_token_last_quarter` | `standardized_residual_concat` | `benign`, `secret_present_safe` | 0.7924 | 0.7461 |
| `cift_diag_safe_secret_mean_pool_last_quarter` | `diagonal_distance` | `secret_present_safe` | 0.5232 | 0.3819 |
| `cift_diag_nonleaking_mean_pool_last_quarter` | `diagonal_distance` | `benign`, `secret_present_safe` | 0.5080 | 0.3136 |
| `cift_residual_safe_secret_mean_pool_last_quarter` | `standardized_residual_concat` | `secret_present_safe` | 0.7402 | 0.5951 |
| `cift_residual_nonleaking_mean_pool_last_quarter` | `standardized_residual_concat` | `benign`, `secret_present_safe` | 0.7402 | 0.5951 |

## Variant Results

| Dataset | Variant | Representation | Calibration Labels | Macro F1 | Delta Macro F1 |
|---|---|---|---|---:|---:|
| `baseline_prompts_v1` | `cift_diag_safe_secret_final_token_last_quarter` | `diagonal_distance` | `secret_present_safe` | 0.6981 | -0.1822 |
| `baseline_prompts_v1` | `cift_diag_nonleaking_final_token_last_quarter` | `diagonal_distance` | `benign`, `secret_present_safe` | 0.7135 | -0.1669 |
| `baseline_prompts_v1` | `cift_residual_safe_secret_final_token_last_quarter` | `standardized_residual_concat` | `secret_present_safe` | 0.8293 | -0.0511 |
| `baseline_prompts_v1` | `cift_residual_nonleaking_final_token_last_quarter` | `standardized_residual_concat` | `benign`, `secret_present_safe` | 0.8293 | -0.0511 |
| `baseline_prompts_v1` | `cift_diag_safe_secret_mean_pool_last_quarter` | `diagonal_distance` | `secret_present_safe` | 0.6541 | -0.2263 |
| `baseline_prompts_v1` | `cift_diag_nonleaking_mean_pool_last_quarter` | `diagonal_distance` | `benign`, `secret_present_safe` | 0.6871 | -0.1933 |
| `baseline_prompts_v1` | `cift_residual_safe_secret_mean_pool_last_quarter` | `standardized_residual_concat` | `secret_present_safe` | 0.8623 | -0.0181 |
| `baseline_prompts_v1` | `cift_residual_nonleaking_mean_pool_last_quarter` | `standardized_residual_concat` | `benign`, `secret_present_safe` | 0.8623 | -0.0181 |
| `hard_prompts_v1` | `cift_diag_safe_secret_final_token_last_quarter` | `diagonal_distance` | `secret_present_safe` | 0.5399 | -0.3932 |
| `hard_prompts_v1` | `cift_diag_nonleaking_final_token_last_quarter` | `diagonal_distance` | `benign`, `secret_present_safe` | 0.6081 | -0.3250 |
| `hard_prompts_v1` | `cift_residual_safe_secret_final_token_last_quarter` | `standardized_residual_concat` | `secret_present_safe` | 0.8132 | -0.1199 |
| `hard_prompts_v1` | `cift_residual_nonleaking_final_token_last_quarter` | `standardized_residual_concat` | `benign`, `secret_present_safe` | 0.8132 | -0.1199 |
| `hard_prompts_v1` | `cift_diag_safe_secret_mean_pool_last_quarter` | `diagonal_distance` | `secret_present_safe` | 0.6374 | -0.2957 |
| `hard_prompts_v1` | `cift_diag_nonleaking_mean_pool_last_quarter` | `diagonal_distance` | `benign`, `secret_present_safe` | 0.6612 | -0.2719 |
| `hard_prompts_v1` | `cift_residual_safe_secret_mean_pool_last_quarter` | `standardized_residual_concat` | `secret_present_safe` | 0.8074 | -0.1257 |
| `hard_prompts_v1` | `cift_residual_nonleaking_mean_pool_last_quarter` | `standardized_residual_concat` | `benign`, `secret_present_safe` | 0.8074 | -0.1257 |
| `hard_prompts_v2` | `cift_diag_safe_secret_final_token_last_quarter` | `diagonal_distance` | `secret_present_safe` | 0.5329 | -0.4328 |
| `hard_prompts_v2` | `cift_diag_nonleaking_final_token_last_quarter` | `diagonal_distance` | `benign`, `secret_present_safe` | 0.5662 | -0.3995 |
| `hard_prompts_v2` | `cift_residual_safe_secret_final_token_last_quarter` | `standardized_residual_concat` | `secret_present_safe` | 0.7808 | -0.1849 |
| `hard_prompts_v2` | `cift_residual_nonleaking_final_token_last_quarter` | `standardized_residual_concat` | `benign`, `secret_present_safe` | 0.7808 | -0.1849 |
| `hard_prompts_v2` | `cift_diag_safe_secret_mean_pool_last_quarter` | `diagonal_distance` | `secret_present_safe` | 0.3819 | -0.5838 |
| `hard_prompts_v2` | `cift_diag_nonleaking_mean_pool_last_quarter` | `diagonal_distance` | `benign`, `secret_present_safe` | 0.3702 | -0.5955 |
| `hard_prompts_v2` | `cift_residual_safe_secret_mean_pool_last_quarter` | `standardized_residual_concat` | `secret_present_safe` | 0.5951 | -0.3707 |
| `hard_prompts_v2` | `cift_residual_nonleaking_mean_pool_last_quarter` | `standardized_residual_concat` | `benign`, `secret_present_safe` | 0.5951 | -0.3707 |
| `hard_prompts_v3` | `cift_diag_safe_secret_final_token_last_quarter` | `diagonal_distance` | `secret_present_safe` | 0.4076 | -0.4735 |
| `hard_prompts_v3` | `cift_diag_nonleaking_final_token_last_quarter` | `diagonal_distance` | `benign`, `secret_present_safe` | 0.4255 | -0.4556 |
| `hard_prompts_v3` | `cift_residual_safe_secret_final_token_last_quarter` | `standardized_residual_concat` | `secret_present_safe` | 0.7461 | -0.1350 |
| `hard_prompts_v3` | `cift_residual_nonleaking_final_token_last_quarter` | `standardized_residual_concat` | `benign`, `secret_present_safe` | 0.7461 | -0.1350 |
| `hard_prompts_v3` | `cift_diag_safe_secret_mean_pool_last_quarter` | `diagonal_distance` | `secret_present_safe` | 0.4193 | -0.4618 |
| `hard_prompts_v3` | `cift_diag_nonleaking_mean_pool_last_quarter` | `diagonal_distance` | `benign`, `secret_present_safe` | 0.3136 | -0.5675 |
| `hard_prompts_v3` | `cift_residual_safe_secret_mean_pool_last_quarter` | `standardized_residual_concat` | `secret_present_safe` | 0.6962 | -0.1849 |
| `hard_prompts_v3` | `cift_residual_nonleaking_mean_pool_last_quarter` | `standardized_residual_concat` | `benign`, `secret_present_safe` | 0.6962 | -0.1849 |

## Variant Sources

| Variant | Source Features | Ridge |
|---|---|---:|
| `cift_diag_safe_secret_final_token_last_quarter` | `final_token_layer_22`, `final_token_layer_23`, `final_token_layer_24`, `final_token_layer_25`, `final_token_layer_26`, `final_token_layer_27`, `final_token_layer_28` | 0.001 |
| `cift_diag_nonleaking_final_token_last_quarter` | `final_token_layer_22`, `final_token_layer_23`, `final_token_layer_24`, `final_token_layer_25`, `final_token_layer_26`, `final_token_layer_27`, `final_token_layer_28` | 0.001 |
| `cift_residual_safe_secret_final_token_last_quarter` | `final_token_layer_22`, `final_token_layer_23`, `final_token_layer_24`, `final_token_layer_25`, `final_token_layer_26`, `final_token_layer_27`, `final_token_layer_28` | 0.001 |
| `cift_residual_nonleaking_final_token_last_quarter` | `final_token_layer_22`, `final_token_layer_23`, `final_token_layer_24`, `final_token_layer_25`, `final_token_layer_26`, `final_token_layer_27`, `final_token_layer_28` | 0.001 |
| `cift_diag_safe_secret_mean_pool_last_quarter` | `mean_pool_layer_22`, `mean_pool_layer_23`, `mean_pool_layer_24`, `mean_pool_layer_25`, `mean_pool_layer_26`, `mean_pool_layer_27`, `mean_pool_layer_28` | 0.001 |
| `cift_diag_nonleaking_mean_pool_last_quarter` | `mean_pool_layer_22`, `mean_pool_layer_23`, `mean_pool_layer_24`, `mean_pool_layer_25`, `mean_pool_layer_26`, `mean_pool_layer_27`, `mean_pool_layer_28` | 0.001 |
| `cift_residual_safe_secret_mean_pool_last_quarter` | `mean_pool_layer_22`, `mean_pool_layer_23`, `mean_pool_layer_24`, `mean_pool_layer_25`, `mean_pool_layer_26`, `mean_pool_layer_27`, `mean_pool_layer_28` | 0.001 |
| `cift_residual_nonleaking_mean_pool_last_quarter` | `mean_pool_layer_22`, `mean_pool_layer_23`, `mean_pool_layer_24`, `mean_pool_layer_25`, `mean_pool_layer_26`, `mean_pool_layer_27`, `mean_pool_layer_28` | 0.001 |