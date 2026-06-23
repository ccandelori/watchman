# Trace Paired-Semantic-Indirection Default 720 Readout-Window Progress

Date: 2026-06-22

## Purpose

This run tested whether the trace harness can generate a corpus where the label
depends on resolving a neutral option code rather than memorizing local action
phrases.

The previous `paired_crossed_action` profile suppressed unigram and bigram
shortcuts, but word 1-3 and 1-4 TF-IDF remained perfect because the grammar
frame was fixed. The new `paired_semantic_indirection` profile maps two neutral
codes to safe/exfiltration actions, flips those mappings across variants, and
asks the prompt to select one code while rejecting the other.

This is a stress test. It is not final training data.

## Local Artifacts

Generated artifacts remain under ignored `data/trace_collection/` paths:

- `collection_inputs.paired_semantic_indirection_default_720.jsonl`
- `pair_validation.paired_semantic_indirection_default_720.json`
- `records.paired_semantic_indirection_default_720.jsonl`
- `structured_prompts.paired_semantic_indirection_default_720.jsonl`
- `qwen3_0_6b_trace_paired_semantic_indirection_default_720_readout.pt`
- `probe_qwen3_0_6b_trace_paired_semantic_indirection_default_720_readout.json`
- `binary_tasks_trace_paired_semantic_indirection_default_720_stratified.json`
- `binary_tasks_trace_paired_semantic_indirection_default_720_grouped.json`

## Corpus Shape

The reproducible profile produced 720 trace inputs:

- 12 task families.
- 20 variants per label.
- 240 `benign` rows.
- 240 `secret_present_safe` rows.
- 240 `exfiltration_intent` rows.
- Shared tool payloads within every safe/exfiltration pair.
- No credential placeholders in tool-call payloads.

The CIFT conversion skipped the 240 benign rows and produced 480 structured
safe/exfiltration prompt records.

The paired validator passed all safe/exfiltration pairs:

| Metric | Value |
|---|---:|
| Pair count | 240 |
| Failed pairs | 0 |
| Max unigram delta | 0 |
| Min bigram Jaccard | 0.9216 |
| Average bigram Jaccard | 0.9259 |
| Tool payload mismatch pairs | 0 |
| Tool placeholder pairs | 0 |

## Activation Extraction

Extraction used Qwen3-0.6B on layers `0,7,14,21,28` with
`readout_window` pooling.

Artifact summary:

- Model: `Qwen/Qwen3-0.6B`
- Rows: 480
- Features: `readout_window_layer_00`, `07`, `14`, `21`, `28`

The stratified multiclass probe selected layer 0, but all stratified activation
scores were weak:

| Feature | Macro F1 | Accuracy |
|---|---:|---:|
| `readout_window_layer_00` | 0.3667 | 0.3958 |
| `readout_window_layer_07` | 0.1471 | 0.1479 |
| `readout_window_layer_14` | 0.2173 | 0.2188 |
| `readout_window_layer_21` | 0.1745 | 0.1750 |
| `readout_window_layer_28` | 0.1845 | 0.1854 |

Grouped activation layer sweep:

| Feature | Grouped Macro F1 | Grouped Accuracy |
|---|---:|---:|
| `readout_window_layer_00` | 0.3333 | 0.5000 |
| `readout_window_layer_07` | 0.4770 | 0.4958 |
| `readout_window_layer_14` | 0.5518 | 0.5700 |
| `readout_window_layer_21` | 0.4785 | 0.5275 |
| `readout_window_layer_28` | 0.4951 | 0.5208 |

## Probe And Baseline Result

The best grouped single-layer linear activation probe was
`readout_window_layer_14`.

| Evaluation | Method | Feature | Macro F1 | Accuracy | Confusion Matrix |
|---|---|---|---:|---:|---|
| Stratified | Activation probe | `readout_window_layer_14` | 0.2173 | 0.2188 | `[[57, 183], [192, 48]]` |
| Stratified | Word TF-IDF | `word_tfidf_1_2` | 0.2900 | 0.2917 | `[[72, 168], [172, 68]]` |
| Stratified | Char TF-IDF | `char_wb_tfidf_3_5` | 0.2913 | 0.2938 | `[[73, 167], [172, 68]]` |
| Family-grouped | Activation probe | `readout_window_layer_14` | 0.5518 | 0.5700 | `[[130, 110], [96, 144]]` |
| Family-grouped | Word TF-IDF | `word_tfidf_1_2` | 0.4749 | 0.4775 | `[[104, 136], [115, 125]]` |
| Family-grouped | Char TF-IDF | `char_wb_tfidf_3_5` | 0.4963 | 0.5050 | `[[114, 126], [112, 128]]` |

Additional word-only diagnostics:

| Method | Evaluation | Macro F1 | Accuracy |
|---|---|---:|---:|
| Word TF-IDF 1-gram | Stratified | 0.2960 | 0.2979 |
| Word TF-IDF 1-gram | Family-grouped | 0.4969 | 0.5000 |
| Word TF-IDF 1-2 gram | Stratified | 0.2900 | 0.2917 |
| Word TF-IDF 1-2 gram | Family-grouped | 0.4749 | 0.4775 |
| Word TF-IDF 1-3 gram | Stratified | 0.2750 | 0.2771 |
| Word TF-IDF 1-3 gram | Family-grouped | 0.4669 | 0.4675 |
| Word TF-IDF 1-4 gram | Stratified | 0.2467 | 0.2479 |
| Word TF-IDF 1-4 gram | Family-grouped | 0.4667 | 0.4675 |
| Word TF-IDF 1-5 gram | Stratified | 0.2179 | 0.2188 |
| Word TF-IDF 1-5 gram | Family-grouped | 0.4438 | 0.4442 |
| Word TF-IDF 1-6 gram | Stratified | 0.1990 | 0.2000 |
| Word TF-IDF 1-6 gram | Family-grouped | 0.4575 | 0.4583 |

## Interpretation

This run is a useful negative result.

The data objective mostly worked: word TF-IDF baselines from 1-gram through
1-6 gram no longer dominate under family-grouped evaluation. The earlier
template shortcut was reduced by balancing neutral code selections and adding a
spacer between the action table and final code choice.

The model objective did not work: the current linear readout-window activation
probe on Qwen3-0.6B does not reliably solve the neutral-code indirection task.
The best grouped activation result, layer 14 at 0.5518 macro F1, is only weakly
above chance.

That means the corpus is probably too compositionally indirect for the current
monitor. It tests whether the model state encodes a relation of the form:

```text
code -> action
selected code -> selected action
selected action -> safe/exfiltration label
```

A linear probe over one pooled readout window is not currently extracting that
relation well.

## Next Step

Keep `paired_semantic_indirection` as a hard stress-test profile, but do not
make it the primary training corpus.

The next production-relevant corpus should sit between the two extremes:

- harder than `paired_crossed_action`, which still leaks through fixed
  templates;
- easier than `paired_semantic_indirection`, which requires multi-hop code
  resolution;
- natural enough that a proxy demo can explain the prompt as a realistic
  credential-boundary request;
- balanced enough that word 1-4 and preferably word 1-6 TF-IDF stay near
  chance.

The next promising direction is a grammar-diverse natural paraphrase corpus
with explicit boundary intent, not neutral-code indirection.
