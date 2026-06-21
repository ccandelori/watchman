# CIFT-Like Ablation V2 Progress

## Question

Which part of the first CIFT-like failure is most responsible: readout proxy,
score compression, or calibration set?

## Method

The V2 ablation compares eight CIFT-like variants against the current fixed
combined-feature reference:

```text
concat(final_token_layer_11,final_token_layer_16)
```

The ablated axes are:

| Axis | Values |
|---|---|
| Readout proxy | `final_token`, `mean_pool` |
| Source layers | Last quarter of the hidden-state stack, layers 22 through 28 |
| Representation | `diagonal_distance`, `standardized_residual_concat` |
| Calibration labels | `secret_present_safe`, or `benign` plus `secret_present_safe` |

The evaluation uses grouped cross-validation on `safe_secret_vs_exfiltration`
across all four registered checkpoint datasets.

## Result

The combined static feature still wins every checkpoint, but the best
residual-concat variants are much stronger than the first diagonal-distance
version.

| Dataset | Combined Macro F1 | Best CIFT-like Variant | Best CIFT-like Macro F1 | Delta Macro F1 |
|---|---:|---|---:|---:|
| `baseline_prompts_v1` | 0.8804 | `cift_residual_safe_secret_mean_pool_last_quarter` | 0.8623 | -0.0181 |
| `hard_prompts_v1` | 0.9331 | `cift_residual_safe_secret_final_token_last_quarter` | 0.8132 | -0.1199 |
| `hard_prompts_v2` | 0.9657 | `cift_residual_safe_secret_final_token_last_quarter` | 0.7808 | -0.1849 |
| `hard_prompts_v3` | 0.8811 | `cift_residual_safe_secret_final_token_last_quarter` | 0.7461 | -0.1350 |

Aggregate variant means:

| Variant Family | Mean Macro F1 | Min Macro F1 |
|---|---:|---:|
| Best final-token residual variants | 0.7924 | 0.7461 |
| Best mean-pool residual variants | 0.7402 | 0.5951 |
| Best final-token diagonal variant | 0.5783 | 0.4255 |
| Best mean-pool diagonal variant | 0.5232 | 0.3819 |

## Interpretation

The first diagonal-distance result was too compressed. Keeping per-dimension
standardized residuals gives the classifier much more useful signal, although
not enough to beat the current combined static feature.

The calibration-label axis did not change residual-concat results. That is
expected in hindsight because residual-concat feeds into the standard activation
classifier, which applies `StandardScaler`; the downstream standardization can
wash out calibration-set mean and scale differences. Calibration-set effects
remain visible for diagonal-distance variants because distance construction is
nonlinear.

The next CIFT-like step should test a classifier path that preserves calibrated
residual magnitudes, such as absolute residual features or a no-`StandardScaler`
linear head, before judging calibration-set choices.

Machine-readable report:

```text
data/reports/cift_like_ablation_v2.json
```
