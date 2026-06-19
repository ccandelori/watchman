# CIFT-Like Probe Progress

## Question

Does a first paper-aligned calibrated-deviation score improve on the current
best static activation feature for `safe_secret_vs_exfiltration`?

## Method

The comparison adds `cift_diag_final_token_last_quarter`, a fold-local derived
feature computed from final-token readout features in the last quarter of the
available hidden-state stack:

- `final_token_layer_22`
- `final_token_layer_23`
- `final_token_layer_24`
- `final_token_layer_25`
- `final_token_layer_26`
- `final_token_layer_27`
- `final_token_layer_28`

Inside each grouped cross-validation fold, the score calibrates a diagonal
mean/variance distribution using only training rows whose source label is
`secret_present_safe`. Each row is then represented by one deviation score per
source layer. A logistic activation classifier is trained on those fold-local
scores and compared against the current combined static feature:

```text
concat(final_token_layer_11,final_token_layer_16)
```

This is CIFT-like, not full paper CIFT. It does not yet implement CCI/CFS,
learned nonnegative layer weighting, or true readout positions beyond the final
prompt token.

## Result

| Dataset | Combined Macro F1 | CIFT-like Macro F1 | Delta Macro F1 |
|---|---:|---:|---:|
| `baseline_prompts_v1` | 0.8804 | 0.6981 | -0.1822 |
| `hard_prompts_v1` | 0.9331 | 0.5399 | -0.3932 |
| `hard_prompts_v2` | 0.9657 | 0.5329 | -0.4328 |
| `hard_prompts_v3` | 0.8811 | 0.4076 | -0.4735 |

The CIFT-like score wins zero of four checkpoint comparisons. The combined
static feature remains the stronger current monitor candidate.

## Interpretation

This is a useful negative checkpoint. The result says the naive version of
calibrated last-quarter distance is not enough by itself. It does not falsify
the paper-aligned direction because the implementation is missing several
important parts of the target method.

The next CIFT-like refinement should test whether the failure comes from:

1. using final prompt token positions instead of richer readout positions,
2. compressing each layer into a single diagonal distance before learning,
3. omitting learned layer weighting, or
4. calibrating only on `secret_present_safe` examples instead of a broader
   benign/non-leaking calibration distribution.

Machine-readable report:

```text
data/reports/cift_like_probe_comparison.json
```
