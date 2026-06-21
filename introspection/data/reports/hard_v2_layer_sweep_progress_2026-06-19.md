# Hard V2 Layer Sweep Progress

## Purpose

Hard V2 caused the fixed `mean_pool_layer_18` activation checkpoint to struggle
on grouped `safe_secret_vs_exfiltration`. The adjudication pass found the
missed examples label-defensible, so this sweep asks whether the signal
disappeared or moved to another representation.

This is analysis-only. It does not replace the fixed regression checkpoint.

## Key Result

The signal did not disappear. It moved strongly toward final-token features in
mid layers.

| Rank | Feature | Macro F1 | Accuracy |
|---:|---|---:|---:|
| 1 | `final_token_layer_11` | 0.9657 | 0.9667 |
| 2 | `final_token_layer_09` | 0.9321 | 0.9333 |
| 3 | `final_token_layer_12` | 0.9161 | 0.9167 |
| 4 | `final_token_layer_13` | 0.9154 | 0.9167 |
| 5 | `final_token_layer_14` | 0.9154 | 0.9167 |
| 6 | `final_token_layer_18` | 0.9154 | 0.9167 |
| 39 | `mean_pool_layer_18` | 0.7225 | 0.7333 |

The best feature, `final_token_layer_11`, produced this grouped confusion
matrix for labels ordered as `exfiltration_intent, secret_present_safe`:

```text
[30, 0]
[2, 28]
```

The fixed reference feature, `mean_pool_layer_18`, produced:

```text
[21, 9]
[7, 23]
```

## Interpretation

Hard V2 does not appear to eliminate the activation signal. Instead, it exposes
that the earlier fixed feature was not the strongest representation for this
harder contrast set.

Two patterns stand out:

1. Final-token features dominate the top ranks, especially layers 8-18.
2. The selected mean-pooled late-layer checkpoint ranks much lower on Hard V2.

This suggests the harder output-contract and summary cases may depend more on
the model's terminal prompt representation than on averaged hidden states.

## Caveat

The best-feature result is selection-biased because the sweep evaluates 58
features and then reports the best. Treat `final_token_layer_11` as a candidate
feature, not a validated replacement checkpoint.

Before changing the default probe feature, run at least one confirmation step:

1. Evaluate `final_token_layer_11` on baseline and Hard V1.
2. Compare it against `mean_pool_layer_18` across all existing datasets.
3. Prefer a prespecified feature only after it performs consistently across
   checkpoints, not just because it won a post-hoc sweep on Hard V2.

## Next Move

Run a cross-check report comparing `mean_pool_layer_18` and
`final_token_layer_11` across baseline, Hard V1, and Hard V2 using the same
grouped `safe_secret_vs_exfiltration` task.
