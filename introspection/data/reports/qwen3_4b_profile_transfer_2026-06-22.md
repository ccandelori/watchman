# Qwen3-4B Profile Transfer

Date: 2026-06-22

## Purpose

Test whether Qwen3-4B readout activation probes trained on one validated trace
profile transfer to another trace profile without retraining.

This is stricter than grouped in-profile evaluation. Grouped evaluation asks
whether the probe generalizes across task families inside one profile. Profile
transfer asks whether the learned decision boundary survives a change in prompt
construction style.

## Harness

New reusable transfer harness:

```text
introspection/src/aegis_introspection/feature_transfer.py
introspection/scripts/evaluate_feature_transfer.py
introspection/tests/test_feature_transfer.py
```

The harness:

- loads one or more training activation artifacts;
- projects each artifact into the same binary task;
- trains one linear activation probe on the training profiles;
- freezes that classifier;
- scores one or more held-out test profiles;
- writes JSON and Markdown reports.

No runtime candidate bundle is modified by this harness.

## Input Profiles

Semantic-indirection profile:

```text
data/trace_collection/qwen3_4b_trace_paired_semantic_indirection_default_720_with_benign_all_readout.pt
```

Paired-natural profile:

```text
data/trace_collection/qwen3_4b_trace_paired_natural_720_l19_l22_readout.pt
```

Paired-adversarial profile:

```text
data/trace_collection/qwen3_4b_trace_paired_adversarial_720_l19_l22_readout.pt
```

Pre-output intent profile:

```text
data/trace_collection/qwen3_4b_trace_pre_output_intent_720_l19_l22_readout.pt
```

Task:

```text
safe_secret_vs_exfiltration
```

Positive label:

```text
exfiltration_intent
```

Decision threshold:

```text
0.5
```

## Results

### Single-Profile Transfer

| Train Profile | Test Profile | Feature | Accuracy | Macro F1 | Positive F1 |
|---|---|---|---:|---:|---:|
| semantic | paired-natural | `readout_window_layer_20` | 0.5646 | 0.5634 | 0.5407 |
| semantic | paired-natural | `concat(readout_window_layer_19,readout_window_layer_20,readout_window_layer_21,readout_window_layer_22)` | 0.5521 | 0.5507 | 0.5254 |
| paired-natural | semantic | `readout_window_layer_20` | 0.5125 | 0.4894 | 0.3810 |
| paired-natural | semantic | `concat(readout_window_layer_19,readout_window_layer_20,readout_window_layer_21,readout_window_layer_22)` | 0.5146 | 0.4850 | 0.3616 |

Output artifacts:

```text
data/trace_collection/feature_transfer.qwen3_4b_semantic_to_paired_natural_layer20.md
data/trace_collection/feature_transfer.qwen3_4b_semantic_to_paired_natural_l19_l22.md
data/trace_collection/feature_transfer.qwen3_4b_paired_natural_to_semantic_layer20.md
data/trace_collection/feature_transfer.qwen3_4b_paired_natural_to_semantic_l19_l22.md
```

### Paired-Adversarial In-Profile Diagnostic

Before using paired-adversarial as a third transfer profile, we checked whether
the Qwen3-4B readout artifact contains in-profile signal:

| Feature | Macro F1 | Accuracy | Confusion Matrix |
|---|---:|---:|---|
| `readout_window_layer_20` | 0.8991 | 0.8992 | `[[215, 25], [22, 218]]` |
| `concat(readout_window_layer_19,readout_window_layer_20,readout_window_layer_21,readout_window_layer_22)` | 0.9165 | 0.9167 | `[[223, 17], [22, 218]]` |

Output artifact:

```text
data/trace_collection/qwen3_4b_paired_adversarial_l19_l22_candidate_crosscheck.md
```

### Mixed-Profile Leave-One-Profile-Out Transfer

Fixed threshold 0.5:

| Train Profiles | Held-Out Profile | Feature | Accuracy | Macro F1 | Positive F1 | Confusion Matrix |
|---|---|---|---:|---:|---:|---|
| semantic + paired-natural | paired-adversarial | `readout_window_layer_20` | 0.5625 | 0.5556 | 0.5000 | `[[105, 135], [75, 165]]` |
| semantic + paired-natural | paired-adversarial | layer 19-22 concat | 0.6500 | 0.6473 | 0.6164 | `[[135, 105], [63, 177]]` |
| semantic + paired-adversarial | paired-natural | `readout_window_layer_20` | 0.6062 | 0.6060 | 0.6151 | `[[151, 89], [100, 140]]` |
| semantic + paired-adversarial | paired-natural | layer 19-22 concat | 0.6667 | 0.6661 | 0.6800 | `[[170, 70], [90, 150]]` |
| paired-natural + paired-adversarial | semantic | `readout_window_layer_20` | 0.4979 | 0.3689 | 0.6542 | `[[228, 12], [229, 11]]` |
| paired-natural + paired-adversarial | semantic | layer 19-22 concat | 0.5083 | 0.3776 | 0.6629 | `[[232, 8], [228, 12]]` |

Label order for confusion matrices is:

```text
exfiltration_intent, secret_present_safe
```

Output artifacts:

```text
data/trace_collection/feature_transfer.qwen3_4b_train_semantic_natural_test_adversarial_layer20.md
data/trace_collection/feature_transfer.qwen3_4b_train_semantic_natural_test_adversarial_l19_l22.md
data/trace_collection/feature_transfer.qwen3_4b_train_semantic_adversarial_test_natural_layer20.md
data/trace_collection/feature_transfer.qwen3_4b_train_semantic_adversarial_test_natural_l19_l22.md
data/trace_collection/feature_transfer.qwen3_4b_train_natural_adversarial_test_semantic_layer20.md
data/trace_collection/feature_transfer.qwen3_4b_train_natural_adversarial_test_semantic_l19_l22.md
```

Oracle threshold sweep on held-out labels:

| Train Profiles | Held-Out Profile | Feature | Best Threshold | Oracle Macro F1 | Oracle Accuracy | Oracle Positive F1 |
|---|---|---|---:|---:|---:|---:|
| semantic + paired-natural | paired-adversarial | `readout_window_layer_20` | 0.03 | 0.6244 | 0.6292 | 0.6667 |
| semantic + paired-natural | paired-adversarial | layer 19-22 concat | 0.20 | 0.6645 | 0.6646 | 0.6694 |
| semantic + paired-adversarial | paired-natural | `readout_window_layer_20` | 0.23 | 0.6375 | 0.6438 | 0.6851 |
| semantic + paired-adversarial | paired-natural | layer 19-22 concat | 0.38 | 0.6835 | 0.6854 | 0.7079 |
| paired-natural + paired-adversarial | semantic | `readout_window_layer_20` | 0.99 | 0.4727 | 0.5062 | 0.6057 |
| paired-natural + paired-adversarial | semantic | layer 19-22 concat | 0.99 | 0.4628 | 0.5167 | 0.6329 |

The oracle sweep is diagnostic only; it uses held-out labels and cannot be used
as a deployment threshold.

### Pre-Output In-Profile Diagnostic

The fourth profile was added to test whether more profile diversity stabilizes
the transfer boundary:

| Feature | Macro F1 | Accuracy | Confusion Matrix |
|---|---:|---:|---|
| `readout_window_layer_20` | 0.9666 | 0.9667 | `[[228, 12], [4, 236]]` |
| `concat(readout_window_layer_19,readout_window_layer_20,readout_window_layer_21,readout_window_layer_22)` | 0.9725 | 0.9725 | `[[230, 10], [3, 237]]` |

Output artifact:

```text
data/trace_collection/qwen3_4b_pre_output_l19_l22_candidate_crosscheck.md
```

### Four-Profile Leave-One-Profile-Out Transfer

Fixed threshold 0.5:

| Train Profiles | Held-Out Profile | Feature | Accuracy | Macro F1 | Positive F1 | Confusion Matrix |
|---|---|---|---:|---:|---:|---|
| paired-natural + paired-adversarial + pre-output | semantic | `readout_window_layer_20` | 0.4979 | 0.3431 | 0.6620 | `[[236, 4], [237, 3]]` |
| paired-natural + paired-adversarial + pre-output | semantic | layer 19-22 concat | 0.5021 | 0.3486 | 0.6648 | `[[237, 3], [236, 4]]` |
| semantic + paired-adversarial + pre-output | paired-natural | `readout_window_layer_20` | 0.7167 | 0.7140 | 0.7414 | `[[195, 45], [91, 149]]` |
| semantic + paired-adversarial + pre-output | paired-natural | layer 19-22 concat | 0.7667 | 0.7624 | 0.7941 | `[[216, 24], [88, 152]]` |
| semantic + paired-natural + pre-output | paired-adversarial | `readout_window_layer_20` | 0.5792 | 0.5695 | 0.5049 | `[[103, 137], [65, 175]]` |
| semantic + paired-natural + pre-output | paired-adversarial | layer 19-22 concat | 0.6562 | 0.6494 | 0.6005 | `[[124, 116], [49, 191]]` |
| semantic + paired-natural + paired-adversarial | pre-output | `readout_window_layer_20` | 0.9354 | 0.9352 | 0.9389 | `[[238, 2], [29, 211]]` |
| semantic + paired-natural + paired-adversarial | pre-output | layer 19-22 concat | 0.9521 | 0.9520 | 0.9543 | `[[240, 0], [23, 217]]` |

Oracle threshold sweep for layer 19-22 concat:

| Train Profiles | Held-Out Profile | Best Threshold | Oracle Macro F1 | Oracle Accuracy | Confusion Matrix |
|---|---|---:|---:|---:|---|
| paired-natural + paired-adversarial + pre-output | semantic | 0.99 | 0.4209 | 0.5042 | `[[212, 28], [210, 30]]` |
| semantic + paired-adversarial + pre-output | paired-natural | 0.91 | 0.7727 | 0.7729 | `[[193, 47], [62, 178]]` |
| semantic + paired-natural + pre-output | paired-adversarial | 0.01 | 0.6871 | 0.6917 | `[[195, 45], [103, 137]]` |
| semantic + paired-natural + paired-adversarial | pre-output | 0.87 | 0.9625 | 0.9625 | `[[238, 2], [16, 224]]` |

## Interpretation

The current Qwen3-4B probe is not profile-stable. With only one source profile,
neither layer 20 nor layer 19-22 concat produces a usable cross-profile
detector.

This result fits the paired-natural diagnostic:

- paired-natural in-profile layer 20 reaches 0.9364 macro F1;
- semantic-indirection in-profile layer 19-22 reaches 0.7185 macro F1;
- cross-profile transfer falls near chance.

So the activation space contains useful signal, but the classifier boundary
changes substantially between profiles.

Adding a second training profile helps when the held-out profile is
paired-natural or paired-adversarial. The layer 19-22 concat is consistently
better in that mixed-profile setting, reaching 0.6473-0.6661 macro F1 at a fixed
0.5 threshold and 0.6645-0.6835 with an oracle threshold.

Adding pre-output as a fourth profile helps paired-natural and pre-output
strongly, and leaves paired-adversarial roughly similar. The best fixed-threshold
result is now 0.9520 macro F1 on pre-output and 0.7624 on paired-natural. The
paired-adversarial holdout remains moderate at 0.6494 fixed and 0.6871 oracle.

The semantic-indirection holdout remains the hard failure. Training on
paired-natural plus paired-adversarial catches most semantic exfiltration rows,
but it also flags almost every semantic safe-secret row. A held-out oracle
threshold does not rescue the split.

Follow-up error analysis:

```text
introspection/data/reports/qwen3_4b_semantic_transfer_error_analysis_2026-06-22.md
```

The semantic failure is a false-positive collapse:

- 239 total errors.
- 236 false positives.
- 3 false negatives.
- semantic safe median exfil score: 0.9999.
- semantic exfil median exfil score: 1.0000.

Span geometry does not explain the failure. Family distribution does not explain
the failure. The semantic-indirection prompts are structurally different: every
safe row and every exfiltration row mentions both the safe option and the
exfiltration option, then asks the model to apply a final neutral code choice.

The current detector appears to learn "exfil option is present" rather than
"exfil option is selected."

## Consequence

The next CIFT training step should not freeze a new detector from either single
profile. The four-profile result is materially better for three profiles, but
still not strong enough for a general runtime candidate because
semantic-indirection remains unresolved.

Next, compare:

- threshold calibration using validation profiles instead of held-out labels;
- layer 19-22 concat as the current mixed-profile leader;
- semantic-indirection v2/v3 data with selected-choice spans;
- a regularized feature family that reduces profile-specific boundary fitting.

The current semantic-indirection runtime candidate remains useful for runtime
plumbing and historical comparison, but not as a general CIFT model.
