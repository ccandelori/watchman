# Aegis Hard Baseline V1 Progress Note

Date: 2026-06-18

Status: harder exploratory checkpoint, not a validated research finding.

## Purpose

This note records the first hard-dataset pressure test for the Aegis
introspection experiment. The baseline dataset showed that `mean_pool_layer_18`
could separate safe secret handling from exfiltration-oriented prompts better
than text baselines under grouped evaluation. Hard Baseline V1 asks whether
that signal survives more contrastive prompt construction.

## Dataset

Hard Baseline V1 is stored at:

```text
introspection/data/prompts_hard.jsonl
```

It keeps the same shape as the baseline:

| Label | Count | Family Count | Examples per Family |
|---|---:|---:|---:|
| `benign` | 30 | 10 | 3 |
| `secret_present_safe` | 30 | 10 | 3 |
| `exfiltration_intent` | 30 | 10 | 3 |

The examples are harder because safe and exfiltration prompts use more parallel
operational framing. Benign examples also use Aegis-relevant vocabulary such as
proxy policy, metadata, logging, handles, detectors, and agent context.

## Artifacts

- Hard dataset: `introspection/data/prompts_hard.jsonl`
- Hard activation artifact: `introspection/data/activations/qwen3_0_6b_hard_all_layers.pt`
- Hard random binary report: `introspection/data/reports/binary_tasks_hard.json`
- Hard random binary summary: `introspection/data/reports/binary_tasks_hard_summary.md`
- Hard grouped binary report: `introspection/data/reports/binary_tasks_hard_grouped.json`
- Hard grouped binary summary: `introspection/data/reports/binary_tasks_hard_grouped_summary.md`

These machine-readable artifacts are registered in `introspection/data/lineage.json`.

## Random Split Results

### benign_vs_secret_related

| Method | Macro F1 | Accuracy | Confusion Matrix |
|---|---:|---:|---|
| `activation_probe` | 1.0000 | 1.0000 | `[[30, 0], [0, 60]]` |
| `char_tfidf` | 0.9738 | 0.9778 | `[[28, 2], [0, 60]]` |
| `word_tfidf` | 0.9341 | 0.9444 | `[[26, 4], [1, 59]]` |

### safe_secret_vs_exfiltration

| Method | Macro F1 | Accuracy | Confusion Matrix |
|---|---:|---:|---|
| `activation_probe` | 0.9163 | 0.9167 | `[[28, 2], [3, 27]]` |
| `word_tfidf` | 0.5572 | 0.5667 | `[[17, 13], [13, 17]]` |
| `char_tfidf` | 0.5329 | 0.5500 | `[[17, 13], [14, 16]]` |

## Grouped Split Results

### benign_vs_secret_related

| Method | Macro F1 | Accuracy | Confusion Matrix |
|---|---:|---:|---|
| `activation_probe` | 1.0000 | 1.0000 | `[[30, 0], [0, 60]]` |
| `char_tfidf` | 0.9357 | 0.9556 | `[[26, 4], [0, 60]]` |
| `word_tfidf` | 0.9226 | 0.9444 | `[[25, 5], [0, 60]]` |

### safe_secret_vs_exfiltration

| Method | Macro F1 | Accuracy | Confusion Matrix |
|---|---:|---:|---|
| `activation_probe` | 0.8788 | 0.8833 | `[[27, 3], [4, 26]]` |
| `word_tfidf` | 0.5726 | 0.5833 | `[[16, 14], [11, 19]]` |
| `char_tfidf` | 0.5571 | 0.5667 | `[[16, 14], [12, 18]]` |

## Baseline vs Hard V1

For the most important task, `safe_secret_vs_exfiltration`, grouped evaluation
now compares as follows:

| Dataset | Method | Macro F1 | Accuracy |
|---|---|---:|---:|
| Baseline | `activation_probe` | 0.8620 | 0.8667 |
| Baseline | `char_tfidf` | 0.7649 | 0.7833 |
| Baseline | `word_tfidf` | 0.7054 | 0.7167 |
| Hard V1 | `activation_probe` | 0.8788 | 0.8833 |
| Hard V1 | `char_tfidf` | 0.5571 | 0.5667 |
| Hard V1 | `word_tfidf` | 0.5726 | 0.5833 |

The activation probe remains strong under the harder grouped split. The text
baselines degrade sharply on the intent-sensitive task, which is useful: Hard
V1 appears to reduce the value of surface lexical cues.

## Interpretation

The hard-v1 result is encouraging, but it should be read carefully. The
activation probe did not collapse under more contrastive examples; in fact, its
grouped macro F1 on `safe_secret_vs_exfiltration` is slightly higher than the
baseline grouped score. The more important pattern is that TF-IDF baselines
lost substantial ground while the activation probe retained most of its
performance.

This suggests that `mean_pool_layer_18` is capturing information beyond simple
token or character overlap. It does not prove robustness. The dataset is still
hand-authored, prompt-only, and small.

## Caveats

- Hard V1 is still only 90 examples.
- The examples were authored in one pass and may contain construction artifacts.
- The examples are prompt-only, not full tool-call traces or multi-turn agent contexts.
- Grouped cross-validation is stricter than random splitting, but it is still not a separately authored holdout set.
- The activation probe remains a lightweight logistic-regression probe.

## Current Position

Where we have been:

- Built the hidden-state extraction pipeline.
- Established baseline random and grouped binary checkpoints.
- Added lineage validation to preserve experiment state.
- Created a living README for the introspection project.

Where we are:

- Added Hard Baseline V1 as a lineage-tracked dataset.
- Extracted hard-v1 all-layer activation features.
- Ran random and grouped hard-v1 binary evaluations.
- Found that the activation probe still leads on the intent-sensitive task.

Where we are going:

- Inspect hard-v1 errors by family.
- Decide whether Hard Baseline V2 should expand weak families or introduce structured tool-call examples.
- Add a comparison summary that reads registered reports and tracks metric movement across datasets.
- Build a separately authored holdout set after the next data iteration.

## Reproduction Commands

Validate hard dataset shape:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python -m unittest introspection.tests.test_hard_dataset
```

Extract hard all-layer features:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/extract_activations.py \
  --prompts introspection/data/prompts_hard.jsonl \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28 \
  --pooling final_token,mean_pool \
  --output introspection/data/activations/qwen3_0_6b_hard_all_layers.pt
```

Run hard grouped binary tasks:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/train_grouped_binary_tasks.py \
  --artifact introspection/data/activations/qwen3_0_6b_hard_all_layers.pt \
  --output-json introspection/data/reports/binary_tasks_hard_grouped.json \
  --output-md introspection/data/reports/binary_tasks_hard_grouped_summary.md
```

Validate lineage:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/validate_lineage.py
```
