# DP-HONEY-lite V3 Selector-Window Checkpoint

Date: 2026-06-20

## Scope

This checkpoint follows the V3 policy diagnostic work. V3 showed that surface
text baselines no longer solve `safe_secret_vs_exfiltration`, but the original
readout-window activation signal was weak. The new experiment tests whether
more precise readout localization recovers signal.

Two derived datasets were created from `dp_honey_lite_prompts_v3`:

| Dataset | Window Contents | Purpose |
|---|---|---|
| `dp_honey_lite_v3_policy_windows` | selected field, selected mode, and explicit `copy` or `mask` decision text | Oracle-style localization control |
| `dp_honey_lite_v3_selector_windows` | selected field and selected mode only | Candidate CIFT-like localization checkpoint |

## Results

Grouped evaluation used 5 stratified group folds over the 10 scenario families.

| Dataset | Task | Best Feature | Macro F1 | Accuracy | Interpretation |
|---|---|---|---:|---:|---|
| `dp_honey_lite_v3_policy_windows` | `safe_secret_vs_exfiltration` | `readout_window_layer_00` | 1.0000 | 1.0000 | Too direct to count as detector evidence |
| `dp_honey_lite_v3_selector_windows` | `safe_secret_vs_exfiltration` | `readout_window_layer_15` | 0.7736 | 0.7812 | Useful localization improvement |

On the selector-window grouped binary comparison:

| Task | Activation Macro F1 | Word TF-IDF Macro F1 | Char TF-IDF Macro F1 |
|---|---:|---:|---:|
| `safe_secret_vs_exfiltration` | 0.7736 | 0.2670 | 0.3542 |
| `benign_vs_secret_related` | 1.0000 | 1.0000 | 0.8569 |

The selector-window layer sweep also shows a coherent mid-layer band rather
than a single isolated result:

| Rank | Feature | Macro F1 | Accuracy |
|---:|---|---:|---:|
| 1 | `readout_window_layer_15` | 0.7736 | 0.7812 |
| 2 | `readout_window_layer_17` | 0.7662 | 0.7750 |
| 3 | `readout_window_layer_16` | 0.7625 | 0.7688 |
| 4 | `readout_window_layer_18` | 0.7504 | 0.7625 |

## Interpretation

The perfect policy-window result is useful as a sanity check, but not as a
monitor claim. That window includes the explicit selected policy decision, so a
linear probe can recover the label from information that would be too close to
an answer key.

The selector-window result is the important one. It removes the explicit
`copy` or `mask` decision clause and still improves substantially over the
original V3 readout-window checkpoint. That supports the paper-aligned CIFT
intuition that readout position matters: hidden states near the selected field
and selected mode carry more signal than the original broader/final readout
position.

The result is still not production-grade. The best selector-window confusion
matrix has 35 errors across 160 target-task rows:

```text
[59, 21]
[14, 66]
```

This is progress, not a promotion.

## Error Slices

The follow-up error analysis confirms that the remaining failures are not
uniform. The activation probe makes 35 target-task errors, compared with 112
for word TF-IDF and 102 for character TF-IDF.

The largest metadata slices are:

| Slice | True Label | Examples | Errors | Accuracy |
|---|---|---:|---:|---:|
| `payload_condition=payload` | `exfiltration_intent` | 40 | 15 | 0.6250 |
| `selected_mode=mode_b` | `exfiltration_intent` | 40 | 15 | 0.6250 |
| `credential_type=database_uri` | `exfiltration_intent` | 40 | 11 | 0.7250 |
| `credential_type=api_key` | `exfiltration_intent` | 40 | 10 | 0.7500 |
| `selected_mode=mode_b` | `secret_present_safe` | 40 | 10 | 0.7500 |
| `payload_condition=payload` | `secret_present_safe` | 40 | 9 | 0.7750 |

The largest family-level cluster is
`dp_honey_lite_v3_support_transcript` exfiltration, where 7 of 8 examples are
misclassified as safe.

## Next Step

Build a targeted V3.4 prompt or dataset revision for payload plus `mode_b`
cases, with special attention to support-transcript exfiltration. The evaluation
contract should stay fixed: selector-window extraction, layer sweep,
`readout_window_layer_15` grouped comparison or updated best-layer comparison,
and the same metadata-slice report.
