# CIFT-Like Ablation

## Source

- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Baseline feature: `concat(final_token_layer_11,final_token_layer_16)`
- Dataset count: `4`
- Variant count: `24`
- Ablation wins: `0`
- Baseline wins: `4`
- Ties: `0`

## Best Variant by Dataset

| Dataset | Baseline Macro F1 | Best Variant | Best Variant Macro F1 | Delta Macro F1 | Winner |
|---|---:|---|---:|---:|---|
| `baseline_prompts_v1` | 0.8804 | `cift_residual_raw_safe_secret_mean_pool_last_quarter` | 0.8788 | -0.0015 | `concat(final_token_layer_11,final_token_layer_16)` |
| `hard_prompts_v1` | 0.9331 | `cift_residual_raw_nonleaking_mean_pool_last_quarter` | 0.8236 | -0.1095 | `concat(final_token_layer_11,final_token_layer_16)` |
| `hard_prompts_v2` | 0.9657 | `cift_residual_scaled_safe_secret_final_token_last_quarter` | 0.7808 | -0.1849 | `concat(final_token_layer_11,final_token_layer_16)` |
| `hard_prompts_v3` | 0.8811 | `cift_residual_scaled_safe_secret_final_token_last_quarter` | 0.7461 | -0.1350 | `concat(final_token_layer_11,final_token_layer_16)` |

## Aggregate by Variant

| Variant | Representation | Classifier Mode | Calibration Labels | Mean Macro F1 | Min Macro F1 |
|---|---|---|---|---:|---:|
| `cift_diag_scaled_safe_secret_final_token_last_quarter` | `diagonal_distance` | `standard_scaled_logreg` | `secret_present_safe` | 0.5446 | 0.4076 |
| `cift_diag_scaled_nonleaking_final_token_last_quarter` | `diagonal_distance` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.5783 | 0.4255 |
| `cift_diag_raw_safe_secret_final_token_last_quarter` | `diagonal_distance` | `raw_logreg` | `secret_present_safe` | 0.4961 | 0.4076 |
| `cift_diag_raw_nonleaking_final_token_last_quarter` | `diagonal_distance` | `raw_logreg` | `benign`, `secret_present_safe` | 0.5429 | 0.4234 |
| `cift_residual_scaled_safe_secret_final_token_last_quarter` | `standardized_residual_concat` | `standard_scaled_logreg` | `secret_present_safe` | 0.7924 | 0.7461 |
| `cift_residual_scaled_nonleaking_final_token_last_quarter` | `standardized_residual_concat` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.7924 | 0.7461 |
| `cift_residual_raw_safe_secret_final_token_last_quarter` | `standardized_residual_concat` | `raw_logreg` | `secret_present_safe` | 0.7880 | 0.7461 |
| `cift_residual_raw_nonleaking_final_token_last_quarter` | `standardized_residual_concat` | `raw_logreg` | `benign`, `secret_present_safe` | 0.7792 | 0.7461 |
| `cift_abs_residual_scaled_safe_secret_final_token_last_quarter` | `absolute_standardized_residual_concat` | `standard_scaled_logreg` | `secret_present_safe` | 0.6017 | 0.4659 |
| `cift_abs_residual_scaled_nonleaking_final_token_last_quarter` | `absolute_standardized_residual_concat` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.6859 | 0.5746 |
| `cift_abs_residual_raw_safe_secret_final_token_last_quarter` | `absolute_standardized_residual_concat` | `raw_logreg` | `secret_present_safe` | 0.5913 | 0.4304 |
| `cift_abs_residual_raw_nonleaking_final_token_last_quarter` | `absolute_standardized_residual_concat` | `raw_logreg` | `benign`, `secret_present_safe` | 0.7021 | 0.6125 |
| `cift_diag_scaled_safe_secret_mean_pool_last_quarter` | `diagonal_distance` | `standard_scaled_logreg` | `secret_present_safe` | 0.5232 | 0.3819 |
| `cift_diag_scaled_nonleaking_mean_pool_last_quarter` | `diagonal_distance` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.5080 | 0.3136 |
| `cift_diag_raw_safe_secret_mean_pool_last_quarter` | `diagonal_distance` | `raw_logreg` | `secret_present_safe` | 0.5232 | 0.3716 |
| `cift_diag_raw_nonleaking_mean_pool_last_quarter` | `diagonal_distance` | `raw_logreg` | `benign`, `secret_present_safe` | 0.4775 | 0.3011 |
| `cift_residual_scaled_safe_secret_mean_pool_last_quarter` | `standardized_residual_concat` | `standard_scaled_logreg` | `secret_present_safe` | 0.7402 | 0.5951 |
| `cift_residual_scaled_nonleaking_mean_pool_last_quarter` | `standardized_residual_concat` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.7402 | 0.5951 |
| `cift_residual_raw_safe_secret_mean_pool_last_quarter` | `standardized_residual_concat` | `raw_logreg` | `secret_present_safe` | 0.7718 | 0.6682 |
| `cift_residual_raw_nonleaking_mean_pool_last_quarter` | `standardized_residual_concat` | `raw_logreg` | `benign`, `secret_present_safe` | 0.7486 | 0.6104 |
| `cift_abs_residual_scaled_safe_secret_mean_pool_last_quarter` | `absolute_standardized_residual_concat` | `standard_scaled_logreg` | `secret_present_safe` | 0.4359 | 0.1672 |
| `cift_abs_residual_scaled_nonleaking_mean_pool_last_quarter` | `absolute_standardized_residual_concat` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.5143 | 0.2773 |
| `cift_abs_residual_raw_safe_secret_mean_pool_last_quarter` | `absolute_standardized_residual_concat` | `raw_logreg` | `secret_present_safe` | 0.4355 | 0.2008 |
| `cift_abs_residual_raw_nonleaking_mean_pool_last_quarter` | `absolute_standardized_residual_concat` | `raw_logreg` | `benign`, `secret_present_safe` | 0.5082 | 0.2948 |

## Variant Results

| Dataset | Variant | Representation | Classifier Mode | Calibration Labels | Macro F1 | Delta Macro F1 |
|---|---|---|---|---|---:|---:|
| `baseline_prompts_v1` | `cift_diag_scaled_safe_secret_final_token_last_quarter` | `diagonal_distance` | `standard_scaled_logreg` | `secret_present_safe` | 0.6981 | -0.1822 |
| `baseline_prompts_v1` | `cift_diag_scaled_nonleaking_final_token_last_quarter` | `diagonal_distance` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.7135 | -0.1669 |
| `baseline_prompts_v1` | `cift_diag_raw_safe_secret_final_token_last_quarter` | `diagonal_distance` | `raw_logreg` | `secret_present_safe` | 0.6286 | -0.2518 |
| `baseline_prompts_v1` | `cift_diag_raw_nonleaking_final_token_last_quarter` | `diagonal_distance` | `raw_logreg` | `benign`, `secret_present_safe` | 0.6494 | -0.2309 |
| `baseline_prompts_v1` | `cift_residual_scaled_safe_secret_final_token_last_quarter` | `standardized_residual_concat` | `standard_scaled_logreg` | `secret_present_safe` | 0.8293 | -0.0511 |
| `baseline_prompts_v1` | `cift_residual_scaled_nonleaking_final_token_last_quarter` | `standardized_residual_concat` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.8293 | -0.0511 |
| `baseline_prompts_v1` | `cift_residual_raw_safe_secret_final_token_last_quarter` | `standardized_residual_concat` | `raw_logreg` | `secret_present_safe` | 0.8293 | -0.0511 |
| `baseline_prompts_v1` | `cift_residual_raw_nonleaking_final_token_last_quarter` | `standardized_residual_concat` | `raw_logreg` | `benign`, `secret_present_safe` | 0.8125 | -0.0679 |
| `baseline_prompts_v1` | `cift_abs_residual_scaled_safe_secret_final_token_last_quarter` | `absolute_standardized_residual_concat` | `standard_scaled_logreg` | `secret_present_safe` | 0.7801 | -0.1002 |
| `baseline_prompts_v1` | `cift_abs_residual_scaled_nonleaking_final_token_last_quarter` | `absolute_standardized_residual_concat` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.7979 | -0.0825 |
| `baseline_prompts_v1` | `cift_abs_residual_raw_safe_secret_final_token_last_quarter` | `absolute_standardized_residual_concat` | `raw_logreg` | `secret_present_safe` | 0.7737 | -0.1067 |
| `baseline_prompts_v1` | `cift_abs_residual_raw_nonleaking_final_token_last_quarter` | `absolute_standardized_residual_concat` | `raw_logreg` | `benign`, `secret_present_safe` | 0.7725 | -0.1078 |
| `baseline_prompts_v1` | `cift_diag_scaled_safe_secret_mean_pool_last_quarter` | `diagonal_distance` | `standard_scaled_logreg` | `secret_present_safe` | 0.6541 | -0.2263 |
| `baseline_prompts_v1` | `cift_diag_scaled_nonleaking_mean_pool_last_quarter` | `diagonal_distance` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.6871 | -0.1933 |
| `baseline_prompts_v1` | `cift_diag_raw_safe_secret_mean_pool_last_quarter` | `diagonal_distance` | `raw_logreg` | `secret_present_safe` | 0.6926 | -0.1877 |
| `baseline_prompts_v1` | `cift_diag_raw_nonleaking_mean_pool_last_quarter` | `diagonal_distance` | `raw_logreg` | `benign`, `secret_present_safe` | 0.6732 | -0.2072 |
| `baseline_prompts_v1` | `cift_residual_scaled_safe_secret_mean_pool_last_quarter` | `standardized_residual_concat` | `standard_scaled_logreg` | `secret_present_safe` | 0.8623 | -0.0181 |
| `baseline_prompts_v1` | `cift_residual_scaled_nonleaking_mean_pool_last_quarter` | `standardized_residual_concat` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.8623 | -0.0181 |
| `baseline_prompts_v1` | `cift_residual_raw_safe_secret_mean_pool_last_quarter` | `standardized_residual_concat` | `raw_logreg` | `secret_present_safe` | 0.8788 | -0.0015 |
| `baseline_prompts_v1` | `cift_residual_raw_nonleaking_mean_pool_last_quarter` | `standardized_residual_concat` | `raw_logreg` | `benign`, `secret_present_safe` | 0.8620 | -0.0183 |
| `baseline_prompts_v1` | `cift_abs_residual_scaled_safe_secret_mean_pool_last_quarter` | `absolute_standardized_residual_concat` | `standard_scaled_logreg` | `secret_present_safe` | 0.7927 | -0.0876 |
| `baseline_prompts_v1` | `cift_abs_residual_scaled_nonleaking_mean_pool_last_quarter` | `absolute_standardized_residual_concat` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.7793 | -0.1011 |
| `baseline_prompts_v1` | `cift_abs_residual_raw_safe_secret_mean_pool_last_quarter` | `absolute_standardized_residual_concat` | `raw_logreg` | `secret_present_safe` | 0.7905 | -0.0899 |
| `baseline_prompts_v1` | `cift_abs_residual_raw_nonleaking_mean_pool_last_quarter` | `absolute_standardized_residual_concat` | `raw_logreg` | `benign`, `secret_present_safe` | 0.7457 | -0.1346 |
| `hard_prompts_v1` | `cift_diag_scaled_safe_secret_final_token_last_quarter` | `diagonal_distance` | `standard_scaled_logreg` | `secret_present_safe` | 0.5399 | -0.3932 |
| `hard_prompts_v1` | `cift_diag_scaled_nonleaking_final_token_last_quarter` | `diagonal_distance` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.6081 | -0.3250 |
| `hard_prompts_v1` | `cift_diag_raw_safe_secret_final_token_last_quarter` | `diagonal_distance` | `raw_logreg` | `secret_present_safe` | 0.4951 | -0.4380 |
| `hard_prompts_v1` | `cift_diag_raw_nonleaking_final_token_last_quarter` | `diagonal_distance` | `raw_logreg` | `benign`, `secret_present_safe` | 0.6354 | -0.2977 |
| `hard_prompts_v1` | `cift_residual_scaled_safe_secret_final_token_last_quarter` | `standardized_residual_concat` | `standard_scaled_logreg` | `secret_present_safe` | 0.8132 | -0.1199 |
| `hard_prompts_v1` | `cift_residual_scaled_nonleaking_final_token_last_quarter` | `standardized_residual_concat` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.8132 | -0.1199 |
| `hard_prompts_v1` | `cift_residual_raw_safe_secret_final_token_last_quarter` | `standardized_residual_concat` | `raw_logreg` | `secret_present_safe` | 0.8132 | -0.1199 |
| `hard_prompts_v1` | `cift_residual_raw_nonleaking_final_token_last_quarter` | `standardized_residual_concat` | `raw_logreg` | `benign`, `secret_present_safe` | 0.7957 | -0.1374 |
| `hard_prompts_v1` | `cift_abs_residual_scaled_safe_secret_final_token_last_quarter` | `absolute_standardized_residual_concat` | `standard_scaled_logreg` | `secret_present_safe` | 0.6711 | -0.2620 |
| `hard_prompts_v1` | `cift_abs_residual_scaled_nonleaking_final_token_last_quarter` | `absolute_standardized_residual_concat` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.7291 | -0.2040 |
| `hard_prompts_v1` | `cift_abs_residual_raw_safe_secret_final_token_last_quarter` | `absolute_standardized_residual_concat` | `raw_logreg` | `secret_present_safe` | 0.6536 | -0.2795 |
| `hard_prompts_v1` | `cift_abs_residual_raw_nonleaking_final_token_last_quarter` | `absolute_standardized_residual_concat` | `raw_logreg` | `benign`, `secret_present_safe` | 0.7813 | -0.1518 |
| `hard_prompts_v1` | `cift_diag_scaled_safe_secret_mean_pool_last_quarter` | `diagonal_distance` | `standard_scaled_logreg` | `secret_present_safe` | 0.6374 | -0.2957 |
| `hard_prompts_v1` | `cift_diag_scaled_nonleaking_mean_pool_last_quarter` | `diagonal_distance` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.6612 | -0.2719 |
| `hard_prompts_v1` | `cift_diag_raw_safe_secret_mean_pool_last_quarter` | `diagonal_distance` | `raw_logreg` | `secret_present_safe` | 0.6447 | -0.2884 |
| `hard_prompts_v1` | `cift_diag_raw_nonleaking_mean_pool_last_quarter` | `diagonal_distance` | `raw_logreg` | `benign`, `secret_present_safe` | 0.6276 | -0.3055 |
| `hard_prompts_v1` | `cift_residual_scaled_safe_secret_mean_pool_last_quarter` | `standardized_residual_concat` | `standard_scaled_logreg` | `secret_present_safe` | 0.8074 | -0.1257 |
| `hard_prompts_v1` | `cift_residual_scaled_nonleaking_mean_pool_last_quarter` | `standardized_residual_concat` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.8074 | -0.1257 |
| `hard_prompts_v1` | `cift_residual_raw_safe_secret_mean_pool_last_quarter` | `standardized_residual_concat` | `raw_logreg` | `secret_present_safe` | 0.8074 | -0.1257 |
| `hard_prompts_v1` | `cift_residual_raw_nonleaking_mean_pool_last_quarter` | `standardized_residual_concat` | `raw_logreg` | `benign`, `secret_present_safe` | 0.8236 | -0.1095 |
| `hard_prompts_v1` | `cift_abs_residual_scaled_safe_secret_mean_pool_last_quarter` | `absolute_standardized_residual_concat` | `standard_scaled_logreg` | `secret_present_safe` | 0.5703 | -0.3628 |
| `hard_prompts_v1` | `cift_abs_residual_scaled_nonleaking_mean_pool_last_quarter` | `absolute_standardized_residual_concat` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.6622 | -0.2709 |
| `hard_prompts_v1` | `cift_abs_residual_raw_safe_secret_mean_pool_last_quarter` | `absolute_standardized_residual_concat` | `raw_logreg` | `secret_present_safe` | 0.5372 | -0.3959 |
| `hard_prompts_v1` | `cift_abs_residual_raw_nonleaking_mean_pool_last_quarter` | `absolute_standardized_residual_concat` | `raw_logreg` | `benign`, `secret_present_safe` | 0.6447 | -0.2884 |
| `hard_prompts_v2` | `cift_diag_scaled_safe_secret_final_token_last_quarter` | `diagonal_distance` | `standard_scaled_logreg` | `secret_present_safe` | 0.5329 | -0.4328 |
| `hard_prompts_v2` | `cift_diag_scaled_nonleaking_final_token_last_quarter` | `diagonal_distance` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.5662 | -0.3995 |
| `hard_prompts_v2` | `cift_diag_raw_safe_secret_final_token_last_quarter` | `diagonal_distance` | `raw_logreg` | `secret_present_safe` | 0.4531 | -0.5126 |
| `hard_prompts_v2` | `cift_diag_raw_nonleaking_final_token_last_quarter` | `diagonal_distance` | `raw_logreg` | `benign`, `secret_present_safe` | 0.4634 | -0.5024 |
| `hard_prompts_v2` | `cift_residual_scaled_safe_secret_final_token_last_quarter` | `standardized_residual_concat` | `standard_scaled_logreg` | `secret_present_safe` | 0.7808 | -0.1849 |
| `hard_prompts_v2` | `cift_residual_scaled_nonleaking_final_token_last_quarter` | `standardized_residual_concat` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.7808 | -0.1849 |
| `hard_prompts_v2` | `cift_residual_raw_safe_secret_final_token_last_quarter` | `standardized_residual_concat` | `raw_logreg` | `secret_present_safe` | 0.7633 | -0.2024 |
| `hard_prompts_v2` | `cift_residual_raw_nonleaking_final_token_last_quarter` | `standardized_residual_concat` | `raw_logreg` | `benign`, `secret_present_safe` | 0.7626 | -0.2031 |
| `hard_prompts_v2` | `cift_abs_residual_scaled_safe_secret_final_token_last_quarter` | `absolute_standardized_residual_concat` | `standard_scaled_logreg` | `secret_present_safe` | 0.4659 | -0.4998 |
| `hard_prompts_v2` | `cift_abs_residual_scaled_nonleaking_final_token_last_quarter` | `absolute_standardized_residual_concat` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.5746 | -0.3912 |
| `hard_prompts_v2` | `cift_abs_residual_raw_safe_secret_final_token_last_quarter` | `absolute_standardized_residual_concat` | `raw_logreg` | `secret_present_safe` | 0.4304 | -0.5354 |
| `hard_prompts_v2` | `cift_abs_residual_raw_nonleaking_final_token_last_quarter` | `absolute_standardized_residual_concat` | `raw_logreg` | `benign`, `secret_present_safe` | 0.6125 | -0.3532 |
| `hard_prompts_v2` | `cift_diag_scaled_safe_secret_mean_pool_last_quarter` | `diagonal_distance` | `standard_scaled_logreg` | `secret_present_safe` | 0.3819 | -0.5838 |
| `hard_prompts_v2` | `cift_diag_scaled_nonleaking_mean_pool_last_quarter` | `diagonal_distance` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.3702 | -0.5955 |
| `hard_prompts_v2` | `cift_diag_raw_safe_secret_mean_pool_last_quarter` | `diagonal_distance` | `raw_logreg` | `secret_present_safe` | 0.3840 | -0.5817 |
| `hard_prompts_v2` | `cift_diag_raw_nonleaking_mean_pool_last_quarter` | `diagonal_distance` | `raw_logreg` | `benign`, `secret_present_safe` | 0.3011 | -0.6646 |
| `hard_prompts_v2` | `cift_residual_scaled_safe_secret_mean_pool_last_quarter` | `standardized_residual_concat` | `standard_scaled_logreg` | `secret_present_safe` | 0.5951 | -0.3707 |
| `hard_prompts_v2` | `cift_residual_scaled_nonleaking_mean_pool_last_quarter` | `standardized_residual_concat` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.5951 | -0.3707 |
| `hard_prompts_v2` | `cift_residual_raw_safe_secret_mean_pool_last_quarter` | `standardized_residual_concat` | `raw_logreg` | `secret_present_safe` | 0.6682 | -0.2975 |
| `hard_prompts_v2` | `cift_residual_raw_nonleaking_mean_pool_last_quarter` | `standardized_residual_concat` | `raw_logreg` | `benign`, `secret_present_safe` | 0.6104 | -0.3553 |
| `hard_prompts_v2` | `cift_abs_residual_scaled_safe_secret_mean_pool_last_quarter` | `absolute_standardized_residual_concat` | `standard_scaled_logreg` | `secret_present_safe` | 0.2133 | -0.7524 |
| `hard_prompts_v2` | `cift_abs_residual_scaled_nonleaking_mean_pool_last_quarter` | `absolute_standardized_residual_concat` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.2773 | -0.6884 |
| `hard_prompts_v2` | `cift_abs_residual_raw_safe_secret_mean_pool_last_quarter` | `absolute_standardized_residual_concat` | `raw_logreg` | `secret_present_safe` | 0.2135 | -0.7522 |
| `hard_prompts_v2` | `cift_abs_residual_raw_nonleaking_mean_pool_last_quarter` | `absolute_standardized_residual_concat` | `raw_logreg` | `benign`, `secret_present_safe` | 0.2948 | -0.6709 |
| `hard_prompts_v3` | `cift_diag_scaled_safe_secret_final_token_last_quarter` | `diagonal_distance` | `standard_scaled_logreg` | `secret_present_safe` | 0.4076 | -0.4735 |
| `hard_prompts_v3` | `cift_diag_scaled_nonleaking_final_token_last_quarter` | `diagonal_distance` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.4255 | -0.4556 |
| `hard_prompts_v3` | `cift_diag_raw_safe_secret_final_token_last_quarter` | `diagonal_distance` | `raw_logreg` | `secret_present_safe` | 0.4076 | -0.4735 |
| `hard_prompts_v3` | `cift_diag_raw_nonleaking_final_token_last_quarter` | `diagonal_distance` | `raw_logreg` | `benign`, `secret_present_safe` | 0.4234 | -0.4577 |
| `hard_prompts_v3` | `cift_residual_scaled_safe_secret_final_token_last_quarter` | `standardized_residual_concat` | `standard_scaled_logreg` | `secret_present_safe` | 0.7461 | -0.1350 |
| `hard_prompts_v3` | `cift_residual_scaled_nonleaking_final_token_last_quarter` | `standardized_residual_concat` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.7461 | -0.1350 |
| `hard_prompts_v3` | `cift_residual_raw_safe_secret_final_token_last_quarter` | `standardized_residual_concat` | `raw_logreg` | `secret_present_safe` | 0.7461 | -0.1350 |
| `hard_prompts_v3` | `cift_residual_raw_nonleaking_final_token_last_quarter` | `standardized_residual_concat` | `raw_logreg` | `benign`, `secret_present_safe` | 0.7461 | -0.1350 |
| `hard_prompts_v3` | `cift_abs_residual_scaled_safe_secret_final_token_last_quarter` | `absolute_standardized_residual_concat` | `standard_scaled_logreg` | `secret_present_safe` | 0.4897 | -0.3913 |
| `hard_prompts_v3` | `cift_abs_residual_scaled_nonleaking_final_token_last_quarter` | `absolute_standardized_residual_concat` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.6419 | -0.2392 |
| `hard_prompts_v3` | `cift_abs_residual_raw_safe_secret_final_token_last_quarter` | `absolute_standardized_residual_concat` | `raw_logreg` | `secret_present_safe` | 0.5074 | -0.3737 |
| `hard_prompts_v3` | `cift_abs_residual_raw_nonleaking_final_token_last_quarter` | `absolute_standardized_residual_concat` | `raw_logreg` | `benign`, `secret_present_safe` | 0.6419 | -0.2392 |
| `hard_prompts_v3` | `cift_diag_scaled_safe_secret_mean_pool_last_quarter` | `diagonal_distance` | `standard_scaled_logreg` | `secret_present_safe` | 0.4193 | -0.4618 |
| `hard_prompts_v3` | `cift_diag_scaled_nonleaking_mean_pool_last_quarter` | `diagonal_distance` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.3136 | -0.5675 |
| `hard_prompts_v3` | `cift_diag_raw_safe_secret_mean_pool_last_quarter` | `diagonal_distance` | `raw_logreg` | `secret_present_safe` | 0.3716 | -0.5094 |
| `hard_prompts_v3` | `cift_diag_raw_nonleaking_mean_pool_last_quarter` | `diagonal_distance` | `raw_logreg` | `benign`, `secret_present_safe` | 0.3079 | -0.5731 |
| `hard_prompts_v3` | `cift_residual_scaled_safe_secret_mean_pool_last_quarter` | `standardized_residual_concat` | `standard_scaled_logreg` | `secret_present_safe` | 0.6962 | -0.1849 |
| `hard_prompts_v3` | `cift_residual_scaled_nonleaking_mean_pool_last_quarter` | `standardized_residual_concat` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.6962 | -0.1849 |
| `hard_prompts_v3` | `cift_residual_raw_safe_secret_mean_pool_last_quarter` | `standardized_residual_concat` | `raw_logreg` | `secret_present_safe` | 0.7329 | -0.1482 |
| `hard_prompts_v3` | `cift_residual_raw_nonleaking_mean_pool_last_quarter` | `standardized_residual_concat` | `raw_logreg` | `benign`, `secret_present_safe` | 0.6983 | -0.1827 |
| `hard_prompts_v3` | `cift_abs_residual_scaled_safe_secret_mean_pool_last_quarter` | `absolute_standardized_residual_concat` | `standard_scaled_logreg` | `secret_present_safe` | 0.1672 | -0.7139 |
| `hard_prompts_v3` | `cift_abs_residual_scaled_nonleaking_mean_pool_last_quarter` | `absolute_standardized_residual_concat` | `standard_scaled_logreg` | `benign`, `secret_present_safe` | 0.3386 | -0.5425 |
| `hard_prompts_v3` | `cift_abs_residual_raw_safe_secret_mean_pool_last_quarter` | `absolute_standardized_residual_concat` | `raw_logreg` | `secret_present_safe` | 0.2008 | -0.6803 |
| `hard_prompts_v3` | `cift_abs_residual_raw_nonleaking_mean_pool_last_quarter` | `absolute_standardized_residual_concat` | `raw_logreg` | `benign`, `secret_present_safe` | 0.3476 | -0.5335 |

## Variant Sources

| Variant | Source Features | Ridge |
|---|---|---:|
| `cift_diag_scaled_safe_secret_final_token_last_quarter` | `final_token_layer_22`, `final_token_layer_23`, `final_token_layer_24`, `final_token_layer_25`, `final_token_layer_26`, `final_token_layer_27`, `final_token_layer_28` | 0.001 |
| `cift_diag_scaled_nonleaking_final_token_last_quarter` | `final_token_layer_22`, `final_token_layer_23`, `final_token_layer_24`, `final_token_layer_25`, `final_token_layer_26`, `final_token_layer_27`, `final_token_layer_28` | 0.001 |
| `cift_diag_raw_safe_secret_final_token_last_quarter` | `final_token_layer_22`, `final_token_layer_23`, `final_token_layer_24`, `final_token_layer_25`, `final_token_layer_26`, `final_token_layer_27`, `final_token_layer_28` | 0.001 |
| `cift_diag_raw_nonleaking_final_token_last_quarter` | `final_token_layer_22`, `final_token_layer_23`, `final_token_layer_24`, `final_token_layer_25`, `final_token_layer_26`, `final_token_layer_27`, `final_token_layer_28` | 0.001 |
| `cift_residual_scaled_safe_secret_final_token_last_quarter` | `final_token_layer_22`, `final_token_layer_23`, `final_token_layer_24`, `final_token_layer_25`, `final_token_layer_26`, `final_token_layer_27`, `final_token_layer_28` | 0.001 |
| `cift_residual_scaled_nonleaking_final_token_last_quarter` | `final_token_layer_22`, `final_token_layer_23`, `final_token_layer_24`, `final_token_layer_25`, `final_token_layer_26`, `final_token_layer_27`, `final_token_layer_28` | 0.001 |
| `cift_residual_raw_safe_secret_final_token_last_quarter` | `final_token_layer_22`, `final_token_layer_23`, `final_token_layer_24`, `final_token_layer_25`, `final_token_layer_26`, `final_token_layer_27`, `final_token_layer_28` | 0.001 |
| `cift_residual_raw_nonleaking_final_token_last_quarter` | `final_token_layer_22`, `final_token_layer_23`, `final_token_layer_24`, `final_token_layer_25`, `final_token_layer_26`, `final_token_layer_27`, `final_token_layer_28` | 0.001 |
| `cift_abs_residual_scaled_safe_secret_final_token_last_quarter` | `final_token_layer_22`, `final_token_layer_23`, `final_token_layer_24`, `final_token_layer_25`, `final_token_layer_26`, `final_token_layer_27`, `final_token_layer_28` | 0.001 |
| `cift_abs_residual_scaled_nonleaking_final_token_last_quarter` | `final_token_layer_22`, `final_token_layer_23`, `final_token_layer_24`, `final_token_layer_25`, `final_token_layer_26`, `final_token_layer_27`, `final_token_layer_28` | 0.001 |
| `cift_abs_residual_raw_safe_secret_final_token_last_quarter` | `final_token_layer_22`, `final_token_layer_23`, `final_token_layer_24`, `final_token_layer_25`, `final_token_layer_26`, `final_token_layer_27`, `final_token_layer_28` | 0.001 |
| `cift_abs_residual_raw_nonleaking_final_token_last_quarter` | `final_token_layer_22`, `final_token_layer_23`, `final_token_layer_24`, `final_token_layer_25`, `final_token_layer_26`, `final_token_layer_27`, `final_token_layer_28` | 0.001 |
| `cift_diag_scaled_safe_secret_mean_pool_last_quarter` | `mean_pool_layer_22`, `mean_pool_layer_23`, `mean_pool_layer_24`, `mean_pool_layer_25`, `mean_pool_layer_26`, `mean_pool_layer_27`, `mean_pool_layer_28` | 0.001 |
| `cift_diag_scaled_nonleaking_mean_pool_last_quarter` | `mean_pool_layer_22`, `mean_pool_layer_23`, `mean_pool_layer_24`, `mean_pool_layer_25`, `mean_pool_layer_26`, `mean_pool_layer_27`, `mean_pool_layer_28` | 0.001 |
| `cift_diag_raw_safe_secret_mean_pool_last_quarter` | `mean_pool_layer_22`, `mean_pool_layer_23`, `mean_pool_layer_24`, `mean_pool_layer_25`, `mean_pool_layer_26`, `mean_pool_layer_27`, `mean_pool_layer_28` | 0.001 |
| `cift_diag_raw_nonleaking_mean_pool_last_quarter` | `mean_pool_layer_22`, `mean_pool_layer_23`, `mean_pool_layer_24`, `mean_pool_layer_25`, `mean_pool_layer_26`, `mean_pool_layer_27`, `mean_pool_layer_28` | 0.001 |
| `cift_residual_scaled_safe_secret_mean_pool_last_quarter` | `mean_pool_layer_22`, `mean_pool_layer_23`, `mean_pool_layer_24`, `mean_pool_layer_25`, `mean_pool_layer_26`, `mean_pool_layer_27`, `mean_pool_layer_28` | 0.001 |
| `cift_residual_scaled_nonleaking_mean_pool_last_quarter` | `mean_pool_layer_22`, `mean_pool_layer_23`, `mean_pool_layer_24`, `mean_pool_layer_25`, `mean_pool_layer_26`, `mean_pool_layer_27`, `mean_pool_layer_28` | 0.001 |
| `cift_residual_raw_safe_secret_mean_pool_last_quarter` | `mean_pool_layer_22`, `mean_pool_layer_23`, `mean_pool_layer_24`, `mean_pool_layer_25`, `mean_pool_layer_26`, `mean_pool_layer_27`, `mean_pool_layer_28` | 0.001 |
| `cift_residual_raw_nonleaking_mean_pool_last_quarter` | `mean_pool_layer_22`, `mean_pool_layer_23`, `mean_pool_layer_24`, `mean_pool_layer_25`, `mean_pool_layer_26`, `mean_pool_layer_27`, `mean_pool_layer_28` | 0.001 |
| `cift_abs_residual_scaled_safe_secret_mean_pool_last_quarter` | `mean_pool_layer_22`, `mean_pool_layer_23`, `mean_pool_layer_24`, `mean_pool_layer_25`, `mean_pool_layer_26`, `mean_pool_layer_27`, `mean_pool_layer_28` | 0.001 |
| `cift_abs_residual_scaled_nonleaking_mean_pool_last_quarter` | `mean_pool_layer_22`, `mean_pool_layer_23`, `mean_pool_layer_24`, `mean_pool_layer_25`, `mean_pool_layer_26`, `mean_pool_layer_27`, `mean_pool_layer_28` | 0.001 |
| `cift_abs_residual_raw_safe_secret_mean_pool_last_quarter` | `mean_pool_layer_22`, `mean_pool_layer_23`, `mean_pool_layer_24`, `mean_pool_layer_25`, `mean_pool_layer_26`, `mean_pool_layer_27`, `mean_pool_layer_28` | 0.001 |
| `cift_abs_residual_raw_nonleaking_mean_pool_last_quarter` | `mean_pool_layer_22`, `mean_pool_layer_23`, `mean_pool_layer_24`, `mean_pool_layer_25`, `mean_pool_layer_26`, `mean_pool_layer_27`, `mean_pool_layer_28` | 0.001 |