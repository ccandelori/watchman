# CIFT OOF Meta-Head

## Source

- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Baseline feature: `concat(final_token_layer_11,final_token_layer_16)`
- Dataset count: `4`
- Variant count: `2`
- Meta-head wins: `1`
- Baseline wins: `3`
- Ties: `0`

## Best Variant by Dataset

| Dataset | Baseline Macro F1 | Best Variant | Best Variant Macro F1 | Delta Macro F1 | Winner |
|---|---:|---|---:|---:|---|
| `baseline_prompts_v1` | 0.8804 | `cift_meta_oof_final_token_mean_pool_signed_residual` | 0.8598 | -0.0206 | `concat(final_token_layer_11,final_token_layer_16)` |
| `hard_prompts_v1` | 0.9331 | `cift_meta_oof_final_token_mean_pool_signed_residual` | 0.8788 | -0.0543 | `concat(final_token_layer_11,final_token_layer_16)` |
| `hard_prompts_v2` | 0.9657 | `cift_meta_oof_final_token_mean_pool_signed_residual` | 0.8823 | -0.0834 | `concat(final_token_layer_11,final_token_layer_16)` |
| `hard_prompts_v3` | 0.8811 | `cift_meta_oof_final_token_mean_pool_signed_residual` | 0.9156 | +0.0345 | `cift_meta_oof_final_token_mean_pool_signed_residual` |

## Aggregate by Variant

| Variant | Source Count | Inner Folds | Mean Macro F1 | Min Macro F1 |
|---|---:|---:|---:|---:|
| `cift_meta_oof_final_token_signed_residual` | 7 | 3 | 0.8209 | 0.7927 |
| `cift_meta_oof_final_token_mean_pool_signed_residual` | 14 | 3 | 0.8841 | 0.8598 |

## Mean Risk-Oriented Coefficients

| Dataset | Variant | Source Feature | Mean Coefficient |
|---|---|---|---:|
| `baseline_prompts_v1` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_22` | +1.0074 |
| `baseline_prompts_v1` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_23` | +0.6810 |
| `baseline_prompts_v1` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_24` | +0.5410 |
| `baseline_prompts_v1` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_25` | +0.3126 |
| `baseline_prompts_v1` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_26` | +0.0375 |
| `baseline_prompts_v1` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_27` | -0.2334 |
| `baseline_prompts_v1` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_28` | -0.5318 |
| `baseline_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_22` | +0.7054 |
| `baseline_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_23` | +0.5261 |
| `baseline_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_24` | +0.3502 |
| `baseline_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_25` | +0.1186 |
| `baseline_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_26` | -0.1069 |
| `baseline_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_27` | -0.3176 |
| `baseline_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_28` | -0.6294 |
| `baseline_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_22` | +0.7678 |
| `baseline_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_23` | +0.3175 |
| `baseline_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_24` | +0.4759 |
| `baseline_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_25` | +0.5386 |
| `baseline_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_26` | -0.0163 |
| `baseline_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_27` | -0.1761 |
| `baseline_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_28` | -0.3646 |
| `hard_prompts_v1` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_22` | +1.0228 |
| `hard_prompts_v1` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_23` | +0.1624 |
| `hard_prompts_v1` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_24` | +0.3591 |
| `hard_prompts_v1` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_25` | +0.2964 |
| `hard_prompts_v1` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_26` | +0.3266 |
| `hard_prompts_v1` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_27` | -0.5154 |
| `hard_prompts_v1` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_28` | +0.2800 |
| `hard_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_22` | +0.7338 |
| `hard_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_23` | +0.1431 |
| `hard_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_24` | +0.2996 |
| `hard_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_25` | +0.2545 |
| `hard_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_26` | +0.1553 |
| `hard_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_27` | -0.4637 |
| `hard_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_28` | +0.1892 |
| `hard_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_22` | +1.1554 |
| `hard_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_23` | +0.2873 |
| `hard_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_24` | +0.4680 |
| `hard_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_25` | +0.3363 |
| `hard_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_26` | +0.1832 |
| `hard_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_27` | -0.5544 |
| `hard_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_28` | -1.1130 |
| `hard_prompts_v2` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_22` | +0.9963 |
| `hard_prompts_v2` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_23` | +0.3884 |
| `hard_prompts_v2` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_24` | +0.6814 |
| `hard_prompts_v2` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_25` | +0.1530 |
| `hard_prompts_v2` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_26` | -0.3425 |
| `hard_prompts_v2` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_27` | -0.6850 |
| `hard_prompts_v2` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_28` | -0.1274 |
| `hard_prompts_v2` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_22` | +0.8668 |
| `hard_prompts_v2` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_23` | +0.3410 |
| `hard_prompts_v2` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_24` | +0.4246 |
| `hard_prompts_v2` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_25` | +0.2076 |
| `hard_prompts_v2` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_26` | -0.0907 |
| `hard_prompts_v2` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_27` | -0.3677 |
| `hard_prompts_v2` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_28` | -0.1447 |
| `hard_prompts_v2` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_22` | +1.4366 |
| `hard_prompts_v2` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_23` | +0.1389 |
| `hard_prompts_v2` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_24` | +0.2507 |
| `hard_prompts_v2` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_25` | +0.0726 |
| `hard_prompts_v2` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_26` | -0.2093 |
| `hard_prompts_v2` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_27` | -0.7737 |
| `hard_prompts_v2` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_28` | -1.3059 |
| `hard_prompts_v3` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_22` | +1.0704 |
| `hard_prompts_v3` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_23` | +0.7974 |
| `hard_prompts_v3` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_24` | +0.5960 |
| `hard_prompts_v3` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_25` | +0.3659 |
| `hard_prompts_v3` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_26` | -0.2355 |
| `hard_prompts_v3` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_27` | -0.3916 |
| `hard_prompts_v3` | `cift_meta_oof_final_token_signed_residual` | `final_token_layer_28` | -0.5802 |
| `hard_prompts_v3` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_22` | +0.7735 |
| `hard_prompts_v3` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_23` | +0.4831 |
| `hard_prompts_v3` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_24` | +0.4784 |
| `hard_prompts_v3` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_25` | +0.4146 |
| `hard_prompts_v3` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_26` | -0.0484 |
| `hard_prompts_v3` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_27` | -0.2920 |
| `hard_prompts_v3` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `final_token_layer_28` | -0.4923 |
| `hard_prompts_v3` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_22` | +1.2803 |
| `hard_prompts_v3` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_23` | +0.1891 |
| `hard_prompts_v3` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_24` | +0.3850 |
| `hard_prompts_v3` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_25` | +0.3409 |
| `hard_prompts_v3` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_26` | -0.1864 |
| `hard_prompts_v3` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_27` | -0.7677 |
| `hard_prompts_v3` | `cift_meta_oof_final_token_mean_pool_signed_residual` | `mean_pool_layer_28` | -1.0577 |

## Variant Results

| Dataset | Variant | Source Count | Macro F1 | Delta Macro F1 |
|---|---|---:|---:|---:|
| `baseline_prompts_v1` | `cift_meta_oof_final_token_signed_residual` | 7 | 0.8277 | -0.0526 |
| `baseline_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | 14 | 0.8598 | -0.0206 |
| `hard_prompts_v1` | `cift_meta_oof_final_token_signed_residual` | 7 | 0.7927 | -0.1404 |
| `hard_prompts_v1` | `cift_meta_oof_final_token_mean_pool_signed_residual` | 14 | 0.8788 | -0.0543 |
| `hard_prompts_v2` | `cift_meta_oof_final_token_signed_residual` | 7 | 0.8331 | -0.1326 |
| `hard_prompts_v2` | `cift_meta_oof_final_token_mean_pool_signed_residual` | 14 | 0.8823 | -0.0834 |
| `hard_prompts_v3` | `cift_meta_oof_final_token_signed_residual` | 7 | 0.8300 | -0.0511 |
| `hard_prompts_v3` | `cift_meta_oof_final_token_mean_pool_signed_residual` | 14 | 0.9156 | +0.0345 |