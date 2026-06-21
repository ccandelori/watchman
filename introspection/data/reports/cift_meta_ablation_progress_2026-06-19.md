# CIFT Meta-Head Ablation V1 Progress

## Question

Can a compact calibration/readout/threshold ablation reduce the Hard V2
introduced errors while preserving the Hard V3 fixed cases from the OOF
meta-head?

## Method

The ablation compares 12 variants over three axes:

1. Source set:
   - `full_dual_readout`: final-token and mean-pool layers 22 through 28.
   - `early_dual_readout`: final-token and mean-pool layers 22 through 25.
   - `early_final_token`: final-token layers 22 through 25.
2. Calibration set:
   - `safe_secret`: `secret_present_safe`.
   - `nonleaking`: `benign` plus `secret_present_safe`.
3. Decision rule:
   - `logistic_default`.
   - `train_f1_threshold`.

The comparison remains focused on Hard V2 and Hard V3, with the combined static
feature as the reference:

```text
concat(final_token_layer_11,final_token_layer_16)
```

## Result

The best variant is the current OOF meta-head configuration:

```text
full_dual_readout_safe_secret_logistic_default
```

It preserves the previous residual profile:

| Candidate Errors | Fixed | Persistent | Introduced | Net Error Delta |
|---:|---:|---:|---:|---:|
| 12 | 3 | 6 | 6 | +3 |

No tested variant reduces the Hard V2 introduced-error count. The best variants
still introduce 5 Hard V2 errors. Source trimming makes the result worse:

| Source Set | Best Candidate Errors | Best Introduced Errors | Best Fixed Errors |
|---|---:|---:|---:|
| `full_dual_readout` | 12 | 6 | 3 |
| `early_dual_readout` | 22 | 14 | 1 |
| `early_final_token` | 21 | 13 | 1 |

Threshold tuning also worsens the result. For the full source set, switching
from `logistic_default` to `train_f1_threshold` increases introduced errors
from 6 to 8.

Calibration choice has no measured effect in this grid. `safe_secret` and
`nonleaking` variants produce identical residual counts for matching source
sets and decision rules.

## Interpretation

This ablation rules out the simplest fixes. The Hard V2 regression is not
resolved by broadening calibration, dropping late last-quarter layers, removing
mean-pool sources, or tuning a train-fold threshold.

The useful finding is that the full dual-readout source set is still the best
of the tested CIFT-like options. The Hard V2 problem is more likely tied to
feature semantics or training signal than to these coarse source/threshold
choices.

The next CIFT move should inspect the Hard V2 introduced examples at the score
level: per-source risk scores, meta-head coefficients, and fold thresholds for
the five introduced Hard V2 cases. That should tell us whether the issue is
specific misleading source layers, fold instability, or an irreducible dataset
contrast for this representation.

Machine-readable report:

```text
data/reports/cift_meta_ablation_v1.json
```
