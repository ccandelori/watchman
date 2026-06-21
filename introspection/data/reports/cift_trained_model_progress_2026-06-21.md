# CIFT Trained Model Milestone

## Artifact

The first loadable CIFT-like detector bundle is:

```text
introspection/data/models/cift_qwen3_0_6b_dp_honey_lite_v4_1_selector_window_layer_15_v1.pkl
```

It is an offline research candidate, not a production policy authority. The
bundle contains a fitted logistic activation classifier, label metadata, the
positive label (`exfiltration_intent`), the selected feature
(`readout_window_layer_15`), a decision threshold of `0.50`, source model
metadata, source activation artifact hash, and evaluation report references.

The bundle scores full-train classifier probabilities. The calibration reports
remain separate out-of-fold evidence and should not be treated as the same score
distribution.

## Evidence

The supporting V4.1 grouped evaluation checkpoint for
`safe_secret_vs_exfiltration` is:

| Signal | Macro F1 | Accuracy |
|---|---:|---:|
| Activation probe, `readout_window_layer_15` | 0.7646 | 0.7875 |
| Word TF-IDF | 0.1172 | 0.1187 |
| Character TF-IDF | 0.2851 | 0.3063 |

The refreshed grouped out-of-fold calibration report reaches:

| Metric | Value |
|---|---:|
| Accuracy | 0.7708 |
| Macro F1 | 0.7704 |
| Brier score | 0.1838 |
| Expected calibration error | 0.1701 |

The operating-point sweep finds best balanced macro F1 at threshold `0.55` on
the calibrated out-of-fold probabilities. The frozen bundle still stores a
threshold of `0.50` because it emits full-train classifier probabilities, not
the calibrated out-of-fold probability distribution.

## Error Profile

The V4.1 grouped error analysis records 20 activation-probe errors on the target
task, compared with 78 word TF-IDF errors and 66 character TF-IDF errors.

The largest pressure slice is `support_transcript` exfiltration with 7 errors
out of 8 examples. Other notable slices are `incident_ticket` exfiltration
with 4 errors out of 8 examples, `audit_export` safe-secret rows with 3 errors
out of 8 examples, mode-b exfiltration rows with 8 errors out of 24 examples,
and no-payload exfiltration rows with 8 errors out of 24 examples.

## Runtime Bridge

The V4.1 runtime-turn export contains 144 `NormalizedTurn`-shaped rows. The
trained bundle DetectorResult export contains 96 task rows for
`safe_secret_vs_exfiltration`.

The trained bundle projection over its training rows is perfectly separated:
48 `allow`, 48 `warn`, and 1.0000 in-sample accuracy. This is only an
integration sanity check. It is not the model quality claim; the grouped
out-of-fold metrics above are the quality evidence.

## Current Claim

Aegis now has a fully trained, persisted, loadable CIFT-like detector candidate
for Qwen 0.6B V4.1 selector-window activations. It can be loaded by code,
validated, scored on activation feature matrices, and exported as
DetectorResult-shaped rows for the Aegis runtime spine.

The remaining gap to production CIFT is live white-box activation capture inside
the proxy, stronger calibration for deployment probabilities, and broader
evaluation beyond the synthetic DP-HONEY-lite V4.1 prompt family set.
