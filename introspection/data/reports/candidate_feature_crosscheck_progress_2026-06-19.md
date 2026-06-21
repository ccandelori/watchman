# Candidate Feature Crosscheck Progress - 2026-06-19

## Purpose

The Hard V2 grouped layer sweep identified `final_token_layer_11` as a strong
post-hoc candidate for `safe_secret_vs_exfiltration`. This checkpoint tests
whether that candidate only won the sweep it came from, or whether it also holds
up against the existing baseline and Hard V1 checkpoints.

The comparison keeps `mean_pool_layer_18` as the fixed historical reference
feature and evaluates both features with grouped cross-validation.

## Result

| Dataset | Reference Macro F1 | Candidate Macro F1 | Delta Macro F1 | Winner |
|---|---:|---:|---:|---|
| `baseline_prompts_v1` | 0.8620 | 0.8445 | -0.0175 | `mean_pool_layer_18` |
| `hard_prompts_v1` | 0.8788 | 0.8993 | +0.0205 | `final_token_layer_11` |
| `hard_prompts_v2` | 0.7225 | 0.9657 | +0.2432 | `final_token_layer_11` |

| Dataset | Reference Accuracy | Candidate Accuracy | Delta Accuracy |
|---|---:|---:|---:|
| `baseline_prompts_v1` | 0.8667 | 0.8500 | -0.0167 |
| `hard_prompts_v1` | 0.8833 | 0.9000 | +0.0167 |
| `hard_prompts_v2` | 0.7333 | 0.9667 | +0.2333 |

`final_token_layer_11` wins two of the three checkpoints. The baseline loss is
small, while the Hard V2 gain is large enough that the candidate deserves
follow-up analysis.

## Interpretation

This does not justify silently replacing `mean_pool_layer_18`. The candidate was
discovered by sweeping Hard V2, so its Hard V2 advantage is post-hoc until it is
tested against additional data. However, the result is stronger than a narrow
Hard V2 artifact: it also improves Hard V1 and does not collapse on the original
baseline.

The practical reading is that `mean_pool_layer_18` should remain the fixed
regression checkpoint, while `final_token_layer_11` becomes a candidate
hard-case checkpoint. That separation keeps historical comparisons honest while
letting the project investigate a feature that appears better aligned with the
harder exfiltration contrasts.

## Next Step

Run Hard V2 residual-error analysis for `final_token_layer_11` and compare its
remaining misses against the `mean_pool_layer_18` misses. If the residual errors
are cleaner and the candidate continues to hold up on new prompt families, then
promotion can be considered as an explicit checkpoint transition rather than an
implicit replacement.
