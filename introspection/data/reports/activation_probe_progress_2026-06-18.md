# Aegis Model Introspection Progress Note

Date: 2026-06-18

Status: exploratory checkpoint, not a validated research finding.

## Purpose

This note records the first working model-introspection loop for Aegis. The
goal was not to prove that activation-based inspection is robust. The goal was
to create a reproducible path from labeled prompts to hidden-state features,
train a simple activation probe, compare it against a surface-text baseline,
and preserve the result as a checkpoint for future progress or regression.

## Context

Aegis is exploring whether model-internal signals can help detect credential
misuse or exfiltration-oriented behavior in agent contexts. Deterministic
secret scanners remain necessary for obvious credential strings. The narrower
question here is whether hidden states can separate cases where a secret is
present but should be protected from cases where the prompt asks for leakage or
unsafe handling.

## Artifacts

- Prompt dataset: `introspection/data/prompts.jsonl`
- Activation artifact: `introspection/data/activations/qwen3_0_6b_features.pt`
- Activation probe report: `introspection/data/reports/probe_baseline.json`
- Text baseline report: `introspection/data/reports/text_baseline.json`
- Human-readable artifact inspector: `introspection/scripts/inspect_activation_artifact.py`
- Activation-probe trainer: `introspection/scripts/train_probe.py`
- Text-baseline trainer: `introspection/scripts/train_text_baseline.py`

## Dataset

The current dataset contains 90 hand-authored examples:

| Label | Count | Meaning |
|---|---:|---|
| `benign` | 30 | No credential handling or exfiltration request. |
| `secret_present_safe` | 30 | Secret-like value appears, but instructions say to protect, redact, or classify it safely. |
| `exfiltration_intent` | 30 | Prompt asks to reveal, transmit, encode, route, or otherwise leak secret-like material. |

The examples are intentionally small and early-stage. They include obvious
surface cues, so results should be treated as a pipeline checkpoint rather than
evidence of generalization.

## Method

1. Loaded `Qwen/Qwen3-0.6B` through PyTorch/Transformers.
2. Verified hidden-state access with `output_hidden_states=True`.
3. Extracted hidden-state features for layers `0`, `7`, `14`, `21`, and `28`.
4. For each selected layer, saved two feature views:
   - `final_token`
   - `mean_pool`
5. Trained logistic-regression activation probes with 5-fold stratified cross-validation.
6. Trained a separate TF-IDF logistic-regression text baseline using the raw prompt text.

The activation probe and text baseline used the same labels, same fold count,
same random seed, and same cross-validation strategy. The text baseline kept
TF-IDF vectorization inside the cross-validation pipeline so each vectorizer was
fit only on the training fold.

## Results

Label order for confusion matrices:

```text
benign, exfiltration_intent, secret_present_safe
```

### Activation Probe

Best feature:

```text
mean_pool_layer_14
```

Metrics:

| Metric | Value |
|---|---:|
| Macro F1 | 0.9554 |
| Accuracy | 0.9556 |

Confusion matrix:

```text
[[30,  0,  0],
 [ 0, 28,  2],
 [ 0,  2, 28]]
```

### Text Baseline

Baseline:

```text
tfidf_logistic_regression
```

Metrics:

| Metric | Value |
|---|---:|
| Macro F1 | 0.7984 |
| Accuracy | 0.8000 |

Confusion matrix:

```text
[[26,  2,  2],
 [ 1, 21,  8],
 [ 2,  3, 25]]
```

## Interpretation

The activation probe outperformed the simple TF-IDF baseline on this dataset.
The most interesting difference is between `secret_present_safe` and
`exfiltration_intent`: the text baseline mislabeled 8 exfiltration examples as
safe-secret examples, while the activation probe mislabeled 2.

This suggests that Qwen's middle-layer mean-pooled representation may encode
some distinction between safe secret handling and exfiltration-oriented
instructions that is not captured as well by simple word and bigram counts.

This does not establish that activation introspection is robust, causal, or
production-ready. It establishes that the pipeline works and that the first
activation signal is strong enough to justify harder tests.

## Caveats

- The dataset is small: 90 examples total.
- The examples are hand-authored and may share stylistic artifacts.
- TF-IDF is a weak text baseline compared with character n-grams, sentence
  embeddings, or a fine-tuned text classifier.
- The cross-validation split is random, not grouped by template or attack type.
- The activation probe may be learning dataset construction artifacts rather
  than a stable safety-relevant representation.
- The result is exploratory and should not be used as a product claim.

## Current Position

Where we have been:

- Set up PyTorch/Transformers access to Qwen hidden states.
- Created a small labeled prompt dataset.
- Extracted activation features into a reusable `.pt` artifact.
- Built an artifact inspector for debugging.
- Trained an activation probe.
- Trained a separate surface-text baseline.

Where we are:

- The activation-probe loop is reproducible.
- The first activation probe beats the first text baseline.
- The result is a useful curiosity and progress marker, not a confirmed finding.

Where we are going:

- Make the dataset harder and less template-driven.
- Add stronger text baselines.
- Compare binary tasks separately.
- Track whether the activation advantage survives more realistic evaluation.

## Next Steps

1. Add a character n-gram TF-IDF baseline.
2. Add grouped splits by prompt family or attack pattern.
3. Add binary evaluations:
   - `benign` vs any secret-related prompt
   - `secret_present_safe` vs `exfiltration_intent`
4. Generate paraphrased examples and hold them out as an evaluation set.
5. Add a comparison report script that summarizes activation and text baselines
   side by side.
6. Preserve this report as the first checkpoint for future regression tracking.

## Reproduction Commands

Inspect the activation artifact:

```bash
.venv-introspection/bin/python introspection/scripts/inspect_activation_artifact.py --examples 2
```

Train the activation probe:

```bash
.venv-introspection/bin/python introspection/scripts/train_probe.py
```

Train the text baseline:

```bash
.venv-introspection/bin/python introspection/scripts/train_text_baseline.py
```

Run tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  .venv-introspection/bin/python -m unittest discover -s introspection/tests
```
