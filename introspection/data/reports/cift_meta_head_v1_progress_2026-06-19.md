# CIFT OOF Meta-Head V1 Progress

## Question

Does a small CFS-like meta-head improve the CIFT-like residual signal when it is
trained on out-of-fold per-layer scores rather than same-fold layer scores?

## Method

This checkpoint keeps the outer evaluation as grouped cross-validation over
prompt families. Inside each outer training fold, every source layer gets its
own inner grouped cross-validation loop:

1. Calibrate signed residuals on `secret_present_safe` rows available to that
   inner train fold.
2. Train one residual classifier per source feature.
3. Emit out-of-fold exfiltration-risk probabilities for the outer train rows.
4. Fit a logistic meta-head on those out-of-fold source scores.
5. Evaluate the meta-head on the outer held-out prompt families.

Two source sets are compared:

```text
cift_meta_oof_final_token_signed_residual
cift_meta_oof_final_token_mean_pool_signed_residual
```

The second source set adds `mean_pool_layer_22` through `mean_pool_layer_28`
alongside `final_token_layer_22` through `final_token_layer_28`. This is still
a proxy for richer readout positions, not the full paper readout-position
mechanism.

The fixed reference remains:

```text
concat(final_token_layer_11,final_token_layer_16)
```

## Result

The OOF meta-head does not beat the combined static feature in aggregate, but
it wins one checkpoint and improves substantially over the final-token-only
CIFT meta-head.

| Dataset | Combined Macro F1 | Best Meta-Head Macro F1 | Delta Macro F1 | Winner |
|---|---:|---:|---:|---|
| `baseline_prompts_v1` | 0.8804 | 0.8598 | -0.0206 | combined static |
| `hard_prompts_v1` | 0.9331 | 0.8788 | -0.0543 | combined static |
| `hard_prompts_v2` | 0.9657 | 0.8823 | -0.0834 | combined static |
| `hard_prompts_v3` | 0.8811 | 0.9156 | +0.0345 | OOF meta-head |

Across all four checkpoints:

| Variant | Mean Macro F1 | Minimum Macro F1 |
|---|---:|---:|
| `cift_meta_oof_final_token_signed_residual` | 0.8209 | 0.7927 |
| `cift_meta_oof_final_token_mean_pool_signed_residual` | 0.8841 | 0.8598 |

## Interpretation

This is the first CIFT-like checkpoint that beats the combined static feature
on any registered dataset, but it is not ready for promotion. The static
combined feature still wins three of four checkpoints and remains stronger on
Hard V2, the most targeted pressure test.

The useful signal is that readout expansion matters. Adding mean-pooled
last-quarter residual source scores improves every checkpoint relative to the
final-token-only meta-head. That suggests the earlier final-token-only CIFT
attempts were leaving useful information outside the source set, even though
the current mean-pool proxy is still not the paper's full readout-position
method.

The next CIFT move should be residual analysis of the Hard V3 win and the Hard
V2 deficit, followed by a calibration/readout ablation for the OOF meta-head.

Machine-readable report:

```text
data/reports/cift_meta_head_v1.json
```
