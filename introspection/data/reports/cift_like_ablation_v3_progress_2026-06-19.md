# CIFT-Like Ablation V3 Progress

## Question

Does preserving calibrated residual magnitude improve the CIFT-like monitor
signal?

## Method

The V3 ablation extends the V2 sweep with two new axes:

| Axis | Values |
|---|---|
| Representation | `diagonal_distance`, `standardized_residual_concat`, `absolute_standardized_residual_concat` |
| Classifier mode | `standard_scaled_logreg`, `raw_logreg` |

The fixed reference remains:

```text
concat(final_token_layer_11,final_token_layer_16)
```

Each variant still uses the last quarter of available hidden-state layers
`22..28`, grouped cross-validation, and the `safe_secret_vs_exfiltration` task.

## Result

No CIFT-like variant beats the combined static feature, but the best raw
residual variants close the gap on easier checkpoints.

| Dataset | Combined Macro F1 | Best CIFT-like Variant | Best CIFT-like Macro F1 | Delta Macro F1 |
|---|---:|---|---:|---:|
| `baseline_prompts_v1` | 0.8804 | `cift_residual_raw_safe_secret_mean_pool_last_quarter` | 0.8788 | -0.0015 |
| `hard_prompts_v1` | 0.9331 | `cift_residual_raw_nonleaking_mean_pool_last_quarter` | 0.8236 | -0.1095 |
| `hard_prompts_v2` | 0.9657 | `cift_residual_scaled_safe_secret_final_token_last_quarter` | 0.7808 | -0.1849 |
| `hard_prompts_v3` | 0.8811 | `cift_residual_scaled_safe_secret_final_token_last_quarter` | 0.7461 | -0.1350 |

Top aggregate variants:

| Variant Family | Mean Macro F1 | Min Macro F1 |
|---|---:|---:|
| Final-token signed residual, scaled classifier | 0.7924 | 0.7461 |
| Final-token signed residual, raw classifier | 0.7880 | 0.7461 |
| Mean-pool signed residual, raw classifier | 0.7718 | 0.6682 |
| Final-token absolute residual, raw nonleaking calibration | 0.7021 | 0.6125 |

## Interpretation

The strongest CIFT-like direction remains signed standardized residuals, not
diagonal distance and not absolute residual magnitude alone. Raw logistic
regression helps mean-pool residuals on the baseline and Hard V1, but it does
not improve the hardest checkpoints. Standard-scaled final-token residuals
remain the best aggregate CIFT-like variant.

Absolute residuals are informative but weaker. They improve when calibrated on
`benign` plus `secret_present_safe`, which suggests calibration-set choice can
matter when the representation preserves magnitude in a nonlinear way.

The next CIFT step should stop expanding generic feature ablations and move
closer to the paper method: learn nonnegative layer weights or a small CFS-like
head over the strongest signed residual representation.

Machine-readable report:

```text
data/reports/cift_like_ablation_v3.json
```
