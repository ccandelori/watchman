# Feature Stability Progress - 2026-06-19

## Purpose

This checkpoint compares three features across the four registered
`safe_secret_vs_exfiltration` checkpoints:

- `mean_pool_layer_18`, the fixed historical reference.
- `final_token_layer_11`, the current hard-case candidate.
- `final_token_layer_16`, the strongest feature in the Hard V3 sweep.

The goal is to test feature stability rather than promote whichever feature won
the newest dataset.

## Result

| Rank | Feature | Wins | Mean Macro F1 | Mean Accuracy | Min Macro F1 | Max Macro F1 | Macro F1 Range |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `final_token_layer_11` | 2 | 0.8978 | 0.9000 | 0.8445 | 0.9657 | 0.1212 |
| 2 | `final_token_layer_16` | 2 | 0.8902 | 0.8917 | 0.8655 | 0.9321 | 0.0667 |
| 3 | `mean_pool_layer_18` | 0 | 0.8239 | 0.8292 | 0.7225 | 0.8788 | 0.1563 |

| Dataset | Winner | `mean_pool_layer_18` | `final_token_layer_11` | `final_token_layer_16` |
|---|---|---:|---:|---:|
| `baseline_prompts_v1` | `final_token_layer_16` | 0.8620 | 0.8445 | 0.8804 |
| `hard_prompts_v1` | `final_token_layer_11` | 0.8788 | 0.8993 | 0.8655 |
| `hard_prompts_v2` | `final_token_layer_11` | 0.7225 | 0.9657 | 0.8828 |
| `hard_prompts_v3` | `final_token_layer_16` | 0.8324 | 0.8818 | 0.9321 |

## Interpretation

`final_token_layer_11` remains the best average performer. It wins Hard V1 and
Hard V2, and its Hard V2 advantage is large enough to dominate the mean.

`final_token_layer_16` is the stability challenger. It wins the original
baseline and Hard V3, has a higher minimum macro F1 than layer 11, and has a
much smaller cross-checkpoint range. That makes it less brittle-looking, even
though its average is lower.

`mean_pool_layer_18` no longer wins any checkpoint in this three-feature
comparison. It should still remain the historical reference because it anchors
the earlier experiment trail.

## Next Step

Do not silently promote either final-token feature. Define a feature-selection
rule first. Two reasonable next checks are:

1. Evaluate a small combined-feature probe using `final_token_layer_11` and
   `final_token_layer_16` together.
2. Create one more held-out checkpoint and evaluate the three features without
   further feature discovery on that checkpoint.
