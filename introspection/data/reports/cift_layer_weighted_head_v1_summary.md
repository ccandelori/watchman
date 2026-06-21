# CIFT Layer-Weighted Head

## Source

- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Baseline feature: `concat(final_token_layer_11,final_token_layer_16)`
- Head feature: `cift_layer_weighted_final_token_signed_residual`
- Source features: `final_token_layer_22`, `final_token_layer_23`, `final_token_layer_24`, `final_token_layer_25`, `final_token_layer_26`, `final_token_layer_27`, `final_token_layer_28`
- Calibration source labels: `secret_present_safe`
- Risk label: `exfiltration_intent`
- Dataset count: `4`
- Head wins: `0`
- Baseline wins: `4`
- Ties: `0`

## Dataset Comparison

| Dataset | Baseline Macro F1 | Head Macro F1 | Delta Macro F1 | Winner |
|---|---:|---:|---:|---|
| `baseline_prompts_v1` | 0.8804 | 0.8293 | -0.0511 | `concat(final_token_layer_11,final_token_layer_16)` |
| `hard_prompts_v1` | 0.9331 | 0.8132 | -0.1199 | `concat(final_token_layer_11,final_token_layer_16)` |
| `hard_prompts_v2` | 0.9657 | 0.7633 | -0.2024 | `concat(final_token_layer_11,final_token_layer_16)` |
| `hard_prompts_v3` | 0.8811 | 0.7461 | -0.1350 | `concat(final_token_layer_11,final_token_layer_16)` |

## Mean Layer Weights

| Dataset | Source Feature | Mean Weight |
|---|---|---:|
| `baseline_prompts_v1` | `final_token_layer_22` | 0.1429 |
| `baseline_prompts_v1` | `final_token_layer_23` | 0.1429 |
| `baseline_prompts_v1` | `final_token_layer_24` | 0.1429 |
| `baseline_prompts_v1` | `final_token_layer_25` | 0.1429 |
| `baseline_prompts_v1` | `final_token_layer_26` | 0.1429 |
| `baseline_prompts_v1` | `final_token_layer_27` | 0.1429 |
| `baseline_prompts_v1` | `final_token_layer_28` | 0.1429 |
| `hard_prompts_v1` | `final_token_layer_22` | 0.1429 |
| `hard_prompts_v1` | `final_token_layer_23` | 0.1429 |
| `hard_prompts_v1` | `final_token_layer_24` | 0.1429 |
| `hard_prompts_v1` | `final_token_layer_25` | 0.1429 |
| `hard_prompts_v1` | `final_token_layer_26` | 0.1429 |
| `hard_prompts_v1` | `final_token_layer_27` | 0.1429 |
| `hard_prompts_v1` | `final_token_layer_28` | 0.1429 |
| `hard_prompts_v2` | `final_token_layer_22` | 0.1429 |
| `hard_prompts_v2` | `final_token_layer_23` | 0.1429 |
| `hard_prompts_v2` | `final_token_layer_24` | 0.1429 |
| `hard_prompts_v2` | `final_token_layer_25` | 0.1429 |
| `hard_prompts_v2` | `final_token_layer_26` | 0.1429 |
| `hard_prompts_v2` | `final_token_layer_27` | 0.1429 |
| `hard_prompts_v2` | `final_token_layer_28` | 0.1429 |
| `hard_prompts_v3` | `final_token_layer_22` | 0.1429 |
| `hard_prompts_v3` | `final_token_layer_23` | 0.1429 |
| `hard_prompts_v3` | `final_token_layer_24` | 0.1429 |
| `hard_prompts_v3` | `final_token_layer_25` | 0.1429 |
| `hard_prompts_v3` | `final_token_layer_26` | 0.1429 |
| `hard_prompts_v3` | `final_token_layer_27` | 0.1429 |
| `hard_prompts_v3` | `final_token_layer_28` | 0.1429 |