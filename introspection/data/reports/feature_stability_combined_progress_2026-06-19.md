# Combined Feature Stability Progress

## Question

Does a derived probe feature that concatenates `final_token_layer_11` and
`final_token_layer_16` outperform either single feature across the existing
checkpoint suite?

## Method

The comparison used grouped cross-validation on `safe_secret_vs_exfiltration`
across four registered datasets:

- `baseline_prompts_v1`
- `hard_prompts_v1`
- `hard_prompts_v2`
- `hard_prompts_v3`

The candidate set was:

- `mean_pool_layer_18`
- `final_token_layer_11`
- `final_token_layer_16`
- `concat(final_token_layer_11,final_token_layer_16)`

The concatenated feature is derived at evaluation time from the registered
activation artifacts. No activation artifact was rewritten.

## Result

| Rank | Feature | Wins | Mean Macro F1 | Min Macro F1 | Range |
|---:|---|---:|---:|---:|---:|
| 1 | `concat(final_token_layer_11,final_token_layer_16)` | 3 | 0.9151 | 0.8804 | 0.0854 |
| 2 | `final_token_layer_11` | 1 | 0.8978 | 0.8445 | 0.1212 |
| 3 | `final_token_layer_16` | 2 | 0.8902 | 0.8655 | 0.0667 |
| 4 | `mean_pool_layer_18` | 0 | 0.8239 | 0.7225 | 0.1563 |

The combined feature wins or ties on baseline, Hard V1, and Hard V2. It does
not win Hard V3; `final_token_layer_16` remains the strongest local feature on
that held-out checkpoint.

## Interpretation

The combined feature is now the leading candidate by average performance, but
the Hard V3 regression relative to `final_token_layer_16` matters. This is a
candidate checkpoint, not a replacement for the fixed historical reference.

The next useful test is residual analysis: determine whether the combined
feature fixes errors from `mean_pool_layer_18`, `final_token_layer_11`, and
`final_token_layer_16`, and whether any new errors cluster in prompt families
that indicate overfitting.
