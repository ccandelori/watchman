# Paired Semantic Indirection Dense Readout Tuning

## Question

Does denser readout-window extraction improve CIFT-style detection on the
`paired_semantic_indirection_default_720` corpus?

## Data Shape

The structured prompt converter now preserves benign no-secret rows for
calibration:

| Label | Rows | Secret Span |
|---|---:|---|
| `benign` | 240 | `null` |
| `secret_present_safe` | 240 | DP-HONEY span |
| `exfiltration_intent` | 240 | DP-HONEY span |

The binary evaluation task remains `safe_secret_vs_exfiltration`, so classifier
metrics are computed over 480 safe/exfiltration rows. The 240 benign rows are
available for CIFT calibration.

## Artifacts

Generated local artifacts:

- `data/trace_collection/structured_prompts.paired_semantic_indirection_default_720_with_benign.jsonl`
- `data/trace_collection/qwen3_0_6b_trace_paired_semantic_indirection_default_720_with_benign_dense_late_readout.pt`
- `data/trace_collection/qwen3_0_6b_trace_paired_semantic_indirection_default_720_with_benign_mid_late_readout.pt`

These files are generated research artifacts and remain ignored by git.

## Layer Sweep

Late band `20-28`:

| Feature | Macro F1 | Accuracy |
|---|---:|---:|
| `readout_window_layer_20` | 0.5204 | 0.5392 |
| `readout_window_layer_21` | 0.4785 | 0.5275 |
| `readout_window_layer_22` | 0.4806 | 0.5208 |
| `readout_window_layer_23` | 0.4794 | 0.5250 |
| `readout_window_layer_24` | 0.4629 | 0.5200 |
| `readout_window_layer_25` | 0.5065 | 0.5433 |
| `readout_window_layer_26` | 0.5131 | 0.5333 |
| `readout_window_layer_27` | 0.4819 | 0.5225 |
| `readout_window_layer_28` | 0.4951 | 0.5208 |

Mid-late band `12-20`:

| Feature | Macro F1 | Accuracy |
|---|---:|---:|
| `readout_window_layer_12` | 0.5422 | 0.5475 |
| `readout_window_layer_13` | 0.5392 | 0.5575 |
| `readout_window_layer_14` | 0.5518 | 0.5700 |
| `readout_window_layer_15` | 0.5364 | 0.5500 |
| `readout_window_layer_16` | 0.5527 | 0.5617 |
| `readout_window_layer_17` | 0.5502 | 0.5625 |
| `readout_window_layer_18` | 0.5201 | 0.5408 |
| `readout_window_layer_19` | 0.5229 | 0.5417 |
| `readout_window_layer_20` | 0.5204 | 0.5392 |

## CIFT-Style Head Check

The scalar diagonal CIFT-style representation remains weak even after adding
benign calibration rows and extracting denser layer bands.

Best scalar diagonal results:

| Source Band | Calibration Labels | Head | Macro F1 | Accuracy |
|---|---|---|---:|---:|
| `20-28` | `benign` | logistic | 0.4971 | 0.5008 |
| `20-28` | `benign` | MLP(8) | 0.4946 | 0.4967 |
| `12-20` | `benign` | logistic | 0.5080 | 0.5092 |
| `12-20` | `benign + secret_present_safe` | MLP(8) | 0.5074 | 0.5158 |
| `14-17` | `benign + secret_present_safe` | MLP(8) | 0.5121 | 0.5133 |
| `14-20` | `benign + secret_present_safe` | logistic | 0.5066 | 0.5075 |

Residual-concat diagnostics preserve more of the activation vector and perform
better:

| Source Band | Calibration Labels | Head | Macro F1 | Accuracy |
|---|---|---|---:|---:|
| `14-17` | `benign` | MLP(4) | 0.5445 | 0.5617 |
| `14-17` | `benign + secret_present_safe` | MLP(4) | 0.5446 | 0.5633 |
| `14-20` | `benign` | MLP(4) | 0.5501 | 0.5783 |
| `14-20` | `benign + secret_present_safe` | MLP(4) | 0.5809 | 0.5900 |

For comparison, grouped text baselines remain below the best activation result:

| Text Baseline | Macro F1 | Accuracy |
|---|---:|---:|
| word TF-IDF 1-2 | 0.4749 | 0.4775 |
| word TF-IDF 1-3 | 0.4669 | 0.4675 |
| word TF-IDF 1-4 | 0.4667 | 0.4675 |
| word TF-IDF 1-5 | 0.4438 | 0.4442 |
| word TF-IDF 1-6 | 0.4575 | 0.4583 |
| char TF-IDF 3-5 | 0.4963 | 0.5050 |

## Interpretation

The useful layer neighborhood for this corpus is not the final late band. It is
roughly `14-17`, with `16` slightly ahead as a single-layer probe. This means
the previous sparse layer choice was not arbitrary luck: it happened to include
the strongest part of the current signal.

The more important finding is that scalar CCI-style compression still loses the
signal. A small MLP over scalar diagonal distances does not improve the monitor.
When the representation preserves signed residual vectors, the MLP improves to
0.5809 macro F1, but that representation is no longer the paper's compact
scalar CCI/CFS path.

The current best hypothesis is therefore:

```text
this corpus contains weak activation signal in mid-late readout layers,
but scalar CIFT compression is too lossy for this data shape.
```

## Next Move

Do not tune the scalar MLP further yet. The better next experiment is a
readout-window/corpus change:

1. Create a variant where the readout window sits inside the user query tail,
   not only the safe tool payload field.
2. Keep benign calibration rows.
3. Extract dense layers `12-20`.
4. Rerun the same layer sweep, text baselines, scalar CIFT heads, and
   residual-concat diagnostic.
