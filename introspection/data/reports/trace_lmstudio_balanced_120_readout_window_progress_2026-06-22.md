# Trace LM Studio Balanced 120 Readout-Window Progress

Date: 2026-06-22

## Purpose

This run tested whether local LM Studio generation could produce a more natural
paired safe/exfiltration corpus than deterministic phrase templates. The
generator was Qwen via LM Studio's local OpenAI-compatible API, using the
previous paired work-item contract.

This is synthetic data. It is useful for data-quality diagnosis, not final
evidence.

## Local Artifacts

Generated artifacts remain under ignored `data/trace_collection/` paths:

- `pair_work_items.lmstudio_balanced_60.jsonl`
- `pair_completions.lmstudio_balanced_60.jsonl`
- `collection_inputs.lmstudio_balanced_120.jsonl`
- `pair_validation.lmstudio_balanced_120.json`
- `records.lmstudio_balanced_120.jsonl`
- `structured_prompts.lmstudio_balanced_120.jsonl`
- `qwen3_0_6b_trace_lmstudio_balanced_120_readout.pt`
- `probe_qwen3_0_6b_trace_lmstudio_balanced_120_readout.json`
- `binary_tasks_trace_lmstudio_balanced_120_stratified.json`
- `binary_tasks_trace_lmstudio_balanced_120_grouped.json`

## Corpus Shape

The LM Studio pass generated 60 paired completions:

- 12 task families.
- 5 variants per family.
- 60 `secret_present_safe` rows.
- 60 `exfiltration_intent` rows.
- Shared tool payloads within every pair.
- No credential placeholders in tool-call payloads.

The strict paired validator rejected all 60 pairs:

| Metric | Value |
|---|---:|
| Pair count | 60 |
| Failed pairs | 60 |
| Max unigram delta | 34 |
| Min weighted bigram Jaccard | 0.0000 |
| Tool payload mismatch pairs | 0 |
| Tool placeholder pairs | 0 |

The failures were lexical-balance failures only. Tool payloads remained matched
and no credential placeholders leaked. The LM-generated prompts were more
natural, but they were not close enough in length or phrase overlap for the
strict validator.

## Activation Extraction

Extraction used Qwen3-0.6B on layers `0,7,14,21,28` with
`readout_window` pooling.

Artifact summary:

- Model: `Qwen/Qwen3-0.6B`
- Rows: 120
- Features: `readout_window_layer_00`, `07`, `14`, `21`, `28`

## Probe And Baseline Result

The best single-layer activation probe remained `readout_window_layer_28`, but
text baselines were stronger on this corpus.

| Evaluation | Method | Feature | Macro F1 | Accuracy | Confusion Matrix |
|---|---|---|---:|---:|---|
| Stratified | Activation probe | `readout_window_layer_28` | 0.7727 | 0.7750 | `[[49, 11], [16, 44]]` |
| Stratified | Word TF-IDF | `word_tfidf_1_2` | 0.8318 | 0.8333 | `[[47, 13], [7, 53]]` |
| Stratified | Char TF-IDF | `char_wb_tfidf_3_5` | 0.8410 | 0.8417 | `[[48, 12], [7, 53]]` |
| Family-grouped | Activation probe | `readout_window_layer_28` | 0.8174 | 0.8200 | `[[47, 13], [8, 52]]` |
| Family-grouped | Word TF-IDF | `word_tfidf_1_2` | 0.9699 | 0.9700 | `[[59, 1], [3, 57]]` |
| Family-grouped | Char TF-IDF | `char_wb_tfidf_3_5` | 0.9531 | 0.9533 | `[[56, 4], [2, 58]]` |

## Interpretation

LM Studio helped with naturalness, but not enough with adversarial balance.
The prompts are less templated than the previous synthetic pair set, and the
activation probe still carries signal, especially under grouped evaluation.
However, word and character baselines remain stronger, which means the labels
are still visible in surface text.

This result argues against scaling this exact prompt recipe to 240+ rows. More
rows with the same lexical asymmetry would make the text shortcut more stable.

## Next Step

Use LM Studio in a repair loop instead of a one-shot generation loop:

1. Generate a paired prompt.
2. Compute unigram delta and bigram overlap immediately.
3. If the pair fails, ask the model to rewrite only the weaker side while
   preserving the semantic label.
4. Accept pairs only when they pass the validator or when a deliberately
   relaxed validation profile is recorded.

The next corpus should be generated with validator-in-the-loop feedback before
running activation extraction.
