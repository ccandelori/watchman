# Trace Paraphrase 120 Readout-Window Progress

Date: 2026-06-22

## Purpose

This run exercises the paired paraphrase work-item flow end to end. The goal
was to test whether a small, validator-clean safe/exfiltration paraphrase
corpus produces a stronger CIFT-style readout-window signal than text-only
baselines.

The corpus is still synthetic. It should be treated as a pipeline and data
quality diagnostic, not final evidence.

## Local Artifacts

Generated artifacts remain under ignored `data/trace_collection/` paths:

- `pair_work_items.paraphrase_60.jsonl`
- `pair_completions.paraphrase_60.jsonl`
- `collection_inputs.paraphrase_120.jsonl`
- `pair_validation.paraphrase_120.json`
- `records.paraphrase_120.jsonl`
- `structured_prompts.paraphrase_120.jsonl`
- `qwen3_0_6b_trace_paraphrase_120_readout.pt`
- `probe_qwen3_0_6b_trace_paraphrase_120_readout.json`
- `text_baseline_trace_paraphrase_120_word.json`
- `binary_tasks_trace_paraphrase_120_stratified.json`
- `binary_tasks_trace_paraphrase_120_grouped.json`

## Corpus Shape

The paired work-item flow produced 60 safe/exfiltration pairs:

- 12 task families.
- 5 variants per family.
- 60 `secret_present_safe` rows.
- 60 `exfiltration_intent` rows.
- Shared tool payloads within every pair.
- No credential placeholders in tool-call payloads.

The pair validator passed all pairs:

| Metric | Value |
|---|---:|
| Pair count | 60 |
| Failed pairs | 0 |
| Max unigram delta | 6 |
| Min weighted bigram Jaccard | 0.6667 |
| Tool payload mismatch pairs | 0 |
| Tool placeholder pairs | 0 |

Trace conversion produced 120 structured prompt records and skipped 0 records.

## Activation Extraction

Extraction used Qwen3-0.6B on layers `0,7,14,21,28` with
`readout_window` pooling.

Artifact summary:

- Model: `Qwen/Qwen3-0.6B`
- Rows: 120
- Features: `readout_window_layer_00`, `07`, `14`, `21`, `28`

## Probe Result

The best single-layer linear activation probe was again
`readout_window_layer_28`.

| Evaluation | Method | Feature | Macro F1 | Accuracy | Confusion Matrix |
|---|---|---|---:|---:|---|
| Stratified | Activation probe | `readout_window_layer_28` | 0.9078 | 0.9083 | `[[55, 5], [6, 54]]` |
| Stratified | Word TF-IDF | `word_tfidf_1_2` | 0.7537 | 0.7583 | `[[41, 19], [10, 50]]` |
| Stratified | Char TF-IDF | `char_wb_tfidf_3_5` | 0.4308 | 0.4333 | `[[27, 33], [35, 25]]` |
| Family-grouped | Activation probe | `readout_window_layer_28` | 0.8786 | 0.8800 | `[[53, 7], [8, 52]]` |
| Family-grouped | Word TF-IDF | `word_tfidf_1_2` | 1.0000 | 1.0000 | `[[60, 0], [0, 60]]` |
| Family-grouped | Char TF-IDF | `char_wb_tfidf_3_5` | 1.0000 | 1.0000 | `[[60, 0], [0, 60]]` |

## Interpretation

This is a productive but not-yet-clean result.

The positive result is that readout-window activations remain highly predictive
on a smaller corpus that passes paired validation. Under random stratified
folds, activation features outperform both word and character TF-IDF baselines.

The limiting result is that family-grouped text baselines are still perfect.
The current synthetic paired completions use stable label-marker substitutions
across all task families, so a text model can learn those substitutions and
generalize to held-out families. That means this corpus still does not prove an
activation-over-text advantage under the strongest comparison.

## Next Step

Create a larger paraphrase corpus where label-marker terms are deliberately
crossed and balanced:

- safe prompts should sometimes use words like `handoff`, `continue`, and
  `downstream` in safe contexts;
- exfiltration prompts should sometimes use words like `review`, `internal`,
  and `boundary` while still expressing credential-authority transfer;
- prompt pair wording should vary by family and variant instead of reusing the
  same substitution set globally;
- every run should continue reporting both stratified and family-grouped text
  baselines.

This result supports the pipeline, but the next data pass must attack lexical
marker reuse directly.
