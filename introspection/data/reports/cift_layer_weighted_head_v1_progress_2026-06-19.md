# CIFT Layer-Weighted Head V1 Progress

## Question

Does a nonnegative layer-weighted head improve the strongest signed-residual
CIFT-like signal?

## Method

This checkpoint moves beyond generic residual-feature ablations toward a
CFS-like layer-combination step. For each grouped cross-validation fold:

1. Calibrate final-token last-quarter residuals on `secret_present_safe`
   training rows.
2. Train one signed-residual classifier per source layer.
3. Convert each layer into an exfiltration-risk probability.
4. Learn nonnegative layer weights from training-fold layer AUCs.
5. Learn a training-fold risk threshold and evaluate on held-out prompt
   families.

The source layers are:

```text
final_token_layer_22
final_token_layer_23
final_token_layer_24
final_token_layer_25
final_token_layer_26
final_token_layer_27
final_token_layer_28
```

The fixed reference remains:

```text
concat(final_token_layer_11,final_token_layer_16)
```

## Result

The layer-weighted head does not beat the combined static feature.

| Dataset | Combined Macro F1 | Layer-Weighted Head Macro F1 | Delta Macro F1 |
|---|---:|---:|---:|
| `baseline_prompts_v1` | 0.8804 | 0.8293 | -0.0511 |
| `hard_prompts_v1` | 0.9331 | 0.8132 | -0.1199 |
| `hard_prompts_v2` | 0.9657 | 0.7633 | -0.2024 |
| `hard_prompts_v3` | 0.8811 | 0.7461 | -0.1350 |

The learned mean layer weights are uniform across all four checkpoint datasets:

```text
1 / 7 = 0.1429 per source layer
```

## Interpretation

This negative result is still useful. Under this AUC-weighted head, the
final-token last-quarter layers look redundant: no single late layer receives
more weight than the others. The head therefore behaves like an ensemble average
of similar late-layer residual classifiers rather than a meaningful CFS-style
layer selector.

This suggests the next CIFT step should not be another simple layer-weighting
heuristic over final-token layers. Better next moves are:

1. learn a richer CFS-like head over per-layer residual scores,
2. train that head with out-of-fold layer scores to reduce train-fold optimism,
3. test readout positions beyond the final prompt token.

Machine-readable report:

```text
data/reports/cift_layer_weighted_head_v1.json
```
