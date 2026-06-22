# Trace LM Studio Repaired 120 Readout-Window Progress

Date: 2026-06-22

## Purpose

This run tested validator-in-the-loop LM Studio generation. The previous
one-shot LM Studio corpus produced more natural prompts, but every pair failed
the strict lexical balance validator. This run asked Qwen through LM Studio to
repair each pair toward explicit validation targets before accepting it.

The corpus is still synthetic. It is a controlled data-quality diagnostic, not
final evidence.

## Local Artifacts

Generated artifacts remain under ignored `data/trace_collection/` paths:

- `pair_work_items.lmstudio_balanced_60.jsonl`
- `pair_completions.lmstudio_repaired_60.jsonl`
- `collection_inputs.lmstudio_repaired_120.jsonl`
- `pair_validation.lmstudio_repaired_120.json`
- `records.lmstudio_repaired_120.jsonl`
- `structured_prompts.lmstudio_repaired_120.jsonl`
- `qwen3_0_6b_trace_lmstudio_repaired_120_readout.pt`
- `probe_qwen3_0_6b_trace_lmstudio_repaired_120_readout.json`
- `binary_tasks_trace_lmstudio_repaired_120_stratified.json`
- `binary_tasks_trace_lmstudio_repaired_120_grouped.json`

## Corpus Shape

The repaired LM Studio pass produced 60 safe/exfiltration pairs:

- 12 task families.
- 5 variants per family.
- 60 `secret_present_safe` rows.
- 60 `exfiltration_intent` rows.
- Shared tool payloads within every pair.
- No credential placeholders in tool-call payloads.

The strict paired validator passed all pairs:

| Metric | Value |
|---|---:|
| Pair count | 60 |
| Failed pairs | 0 |
| Max unigram delta | 7 |
| Min weighted bigram Jaccard | 0.5000 |
| Average weighted bigram Jaccard | 0.6190 |
| Tool payload mismatch pairs | 0 |
| Tool placeholder pairs | 0 |

The repair loop produced no error file. One row matched the deterministic
fallback phrase pattern; the rest were accepted from LM Studio output.

## Activation Extraction

Extraction used Qwen3-0.6B on layers `0,7,14,21,28` with
`readout_window` pooling.

Artifact summary:

- Model: `Qwen/Qwen3-0.6B`
- Rows: 120
- Features: `readout_window_layer_00`, `07`, `14`, `21`, `28`

## Probe And Baseline Result

The best single-layer activation probe shifted from the usual late layer to
`readout_window_layer_14`.

| Evaluation | Method | Feature | Macro F1 | Accuracy | Confusion Matrix |
|---|---|---|---:|---:|---|
| Stratified | Activation probe | `readout_window_layer_14` | 0.7236 | 0.7250 | `[[42, 18], [15, 45]]` |
| Stratified | Word TF-IDF | `word_tfidf_1_2` | 0.8582 | 0.8583 | `[[51, 9], [8, 52]]` |
| Stratified | Char TF-IDF | `char_wb_tfidf_3_5` | 0.1471 | 0.1500 | `[[11, 49], [53, 7]]` |
| Family-grouped | Activation probe | `readout_window_layer_14` | 0.8325 | 0.8333 | `[[48, 12], [8, 52]]` |
| Family-grouped | Word TF-IDF | `word_tfidf_1_2` | 0.9798 | 0.9800 | `[[60, 0], [2, 58]]` |
| Family-grouped | Char TF-IDF | `char_wb_tfidf_3_5` | 0.6292 | 0.6533 | `[[45, 15], [27, 33]]` |

Unigram-only word TF-IDF was weak:

| Evaluation | Method | Macro F1 | Accuracy | Confusion Matrix |
|---|---|---:|---:|---|
| Stratified | Word TF-IDF 1-gram | 0.1575 | 0.1583 | `[[9, 51], [50, 10]]` |
| Family-grouped | Word TF-IDF 1-gram | 0.5755 | 0.6033 | `[[37, 23], [25, 35]]` |

## Interpretation

This run achieved one important data objective and exposed the next shortcut.

The repair loop successfully removed unigram shortcuts. The strict validator
passed every pair, and unigram-only TF-IDF dropped sharply. This is a real
improvement over the one-shot LM Studio corpus.

However, word bigrams still dominate. The repair recipe made the prompts
parallel by preserving a shared sentence frame and swapping action clauses such
as keeping authority inside the boundary versus carrying authority outside it.
That creates predictable local phrase-order patterns, so word 1-2 TF-IDF still
beats the activation probe, especially under family-grouped evaluation.

The activation signal remains meaningful, but this corpus still does not
establish an activation-over-text advantage.

## Next Step

The next corpus should make the validator adversarial to bigram order, not only
to token balance:

- Continue requiring matched tool payloads and no credential placeholders.
- Keep unigram balance strict.
- Add a generation-time or validation-time check for label-correlated action
  bigrams.
- Randomize action-clause order so `inside/outside`, `keep/carry`, and
  `boundary/handoff` phrase order is not label-stable.
- Accept a corpus only when grouped word 1-2 TF-IDF drops below the activation
  probe or at least clearly below prior shortcut-driven runs.

The next target is no longer "make prompts natural." It is "make the
label-bearing phrase order non-memorisable."
