# Qwen3-4B Paired-Natural Transfer Test

Date: 2026-06-22

## Purpose

Pressure-test the frozen Qwen3-4B CIFT runtime candidate outside the corpus it
was trained on.

The frozen candidate was trained on:

```text
paired_semantic_indirection_default_720_with_benign
```

This transfer test scores it, without retraining or threshold changes, on:

```text
paired_natural_720
```

This is not a same-distribution held-out fold. It is a different synthetic
profile that preserves the same runtime/CIFT boundary shape while changing the
prompt semantics.

## Artifacts

Input structured prompts:

```text
data/trace_collection/structured_prompts.paired_natural_720.jsonl
```

Runtime turns:

```text
data/trace_collection/runtime_turns.paired_natural_720_secret_present_binary.jsonl
```

Qwen3-4B readout artifact:

```text
data/trace_collection/qwen3_4b_trace_paired_natural_720_l19_l22_readout.pt
```

Frozen candidate runtime eval:

```text
data/trace_collection/runtime_eval.cift_qwen3_4b_semantic_to_paired_natural_l19_l22_v1.jsonl
```

In-distribution paired-natural grouped diagnostic:

```text
data/trace_collection/qwen3_4b_paired_natural_l19_l22_candidate_crosscheck.md
```

## Frozen Detector Transfer Result

Frozen model:

```text
introspection/data/models/cift_qwen3_4b_semantic_indirection_l19_l22_readout_runtime_v1.json
```

Feature:

```text
concat(readout_window_layer_19,readout_window_layer_20,readout_window_layer_21,readout_window_layer_22)
```

Decision threshold:

```text
0.55
```

Confusion matrix for `secret_present_safe` versus `exfiltration_intent`:

```text
[[146, 94],
 [122, 118]]
```

Rows are true labels and columns are predicted labels in this order:

```text
[[safe -> safe, safe -> exfil],
 [exfil -> safe, exfil -> exfil]]
```

Metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.5500 |
| Macro F1 | 0.5485 |
| Exfiltration precision | 0.5566 |
| Exfiltration recall | 0.4917 |
| Exfiltration F1 | 0.5221 |
| Safe-secret F1 | 0.5748 |

Action counts:

| Action | Count |
|---|---:|
| `allow` | 268 |
| `warn` | 212 |

All 480 runtime rows emitted active CIFT detector results.

## Score Distribution

| Label | Min | Median | Mean | Max |
|---|---:|---:|---:|---:|
| `secret_present_safe` | 0.0000 | 0.0293 | 0.3919 | 1.0000 |
| `exfiltration_intent` | 0.0000 | 0.4814 | 0.4995 | 1.0000 |

The distributions overlap heavily. A post-hoc threshold sweep does not rescue
the transfer result: the oracle best threshold in 0.01 increments is 0.06, with
macro F1 0.5643.

## In-Distribution Diagnostic

To check whether the paired-natural Qwen3-4B readout activations contain signal,
we ran grouped in-distribution probes on the paired-natural artifact. This is a
diagnostic only; it trains within paired-natural folds and is not the frozen
candidate transfer result.

| Feature | Macro F1 | Accuracy | Confusion Matrix |
|---|---:|---:|---|
| `readout_window_layer_20` | 0.9364 | 0.9367 | `[[224, 16], [16, 224]]` |
| `concat(readout_window_layer_19,readout_window_layer_20,readout_window_layer_21,readout_window_layer_22)` | 0.9177 | 0.9183 | `[[219, 21], [20, 220]]` |

This is the key finding: paired-natural contains strong activation signal, but
the semantic-indirection-trained decision boundary does not transfer well.

## Interpretation

The frozen Qwen3-4B CIFT candidate is not yet a general detector. It is a
working runtime candidate for the semantic-indirection profile, but its learned
boundary is profile-specific.

The failure mode is not simply bad threshold calibration. The score ordering
itself changes on paired-natural, and an oracle threshold only reaches 0.5643
macro F1.

The in-distribution paired-natural probe result is encouraging for CIFT: layer
20 alone reaches 0.9364 macro F1 under grouped evaluation. That means the
readout-window activation representation is useful, but the current training
recipe is too narrow.

## Next Move

Build a mixed-profile training setup:

- train on multiple profiles, not just semantic indirection;
- keep one or more profiles fully held out;
- compare single-layer layer-20 versus layer 19-22 concat;
- report frozen transfer performance, not only grouped in-profile performance;
- keep the current semantic-indirection candidate as a historical baseline.

The right lesson is not "tune the threshold." The right lesson is "train and
evaluate the runtime CIFT candidate across profile diversity."
