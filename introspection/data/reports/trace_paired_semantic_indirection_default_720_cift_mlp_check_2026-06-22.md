# Paired Semantic Indirection CIFT MLP Check

## Question

Does a more paper-shaped CIFT head improve on the current linear readout-layer
probe for the `paired_semantic_indirection_default_720` corpus?

## Artifact

- Activation artifact:
  `data/trace_collection/qwen3_0_6b_trace_paired_semantic_indirection_default_720_readout.pt`
- Task: `safe_secret_vs_exfiltration`
- Examples: 480 total, balanced 240 / 240.
- Available feature keys:
  `readout_window_layer_00`, `readout_window_layer_07`,
  `readout_window_layer_14`, `readout_window_layer_21`,
  `readout_window_layer_28`.

## Caveats

This is not a fully paper-faithful CIFT reproduction. The current artifact has
only sparse sampled layers rather than all last-quarter layers, and it does not
include benign no-secret calibration rows. The check therefore uses the closest
available approximation: safe-secret calibration, per-layer diagonal CIFT-style
scores, grouped cross-validation, and a small MLP head.

## Result

| Method | Source Features | Feature Width | Macro F1 | Accuracy |
|---|---|---:|---:|---:|
| Linear readout probe | `readout_window_layer_14` | 1024 | 0.5518 | 0.5700 |
| CIFT scalar diagonal + logreg | layers 21, 28 | 2 | 0.4918 | 0.4983 |
| CIFT scalar diagonal + MLP(4) | layers 21, 28 | 2 | 0.4846 | 0.4958 |
| CIFT scalar diagonal + MLP(8) | layers 21, 28 | 2 | 0.4827 | 0.5008 |
| CIFT scalar diagonal + logreg | layers 0, 7, 14, 21, 28 | 5 | 0.5024 | 0.5092 |
| CIFT scalar diagonal + MLP(4) | layers 0, 7, 14, 21, 28 | 5 | 0.4921 | 0.5075 |
| CIFT scalar diagonal + MLP(8) | layers 0, 7, 14, 21, 28 | 5 | 0.4850 | 0.4942 |
| Residual-concat + MLP(4) | layers 21, 28 | 2048 | 0.5147 | 0.5308 |
| Residual-concat + MLP(4) | layers 0, 7, 14, 21, 28 | 5120 | 0.4912 | 0.5342 |

## Interpretation

The MLP does not rescue the CIFT-style scalar representation on this corpus.
The scalar diagonal variants sit near chance, while the existing linear
readout-layer probe remains stronger, though still weak in absolute terms.

The diagnostic residual-concat MLP also fails to beat the linear readout probe.
That suggests the current weakness is not simply that the classifier is too
linear. The stronger hypothesis is that this corpus/artifact combination is
not exposing a stable, separable CIFT signal under the current readout window,
calibration set, and sparse layer sampling.

## Next Move

Before investing in a full neural CIFT head, improve the data/artifact geometry:

1. Include benign no-secret calibration rows where readout positions are still
   well-defined.
2. Extract denser late-layer readout-window activations instead of only
   five sampled layers.
3. Re-run the scalar CIFT/MLP check after the artifact has true paper-shaped
   calibration and layer coverage.
