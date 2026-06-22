# Trace Crossed-Action 120 Readout-Window Progress

Date: 2026-06-22

## Purpose

This run tested a deliberately adversarial prompt construction against the
strongest text shortcut from the repaired LM Studio corpus: word bigrams.

Each safe/exfiltration pair uses the same unigram and bigram inventory. The
label is carried by which longer action phrase is marked as allowed versus
denied. This is a controlled diagnostic, not a final training corpus.

## Local Artifacts

Generated artifacts remain under ignored `data/trace_collection/` paths:

- `pair_completions.crossed_action_60.jsonl`
- `collection_inputs.crossed_action_120.jsonl`
- `pair_validation.crossed_action_120.json`
- `records.crossed_action_120.jsonl`
- `structured_prompts.crossed_action_120.jsonl`
- `qwen3_0_6b_trace_crossed_action_120_readout.pt`
- `probe_qwen3_0_6b_trace_crossed_action_120_readout.json`
- `binary_tasks_trace_crossed_action_120_stratified.json`
- `binary_tasks_trace_crossed_action_120_grouped.json`

## Corpus Shape

The crossed-action diagnostic produced 60 safe/exfiltration pairs:

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
| Max unigram delta | 0 |
| Min bigram Jaccard | 1.0000 |
| Average bigram Jaccard | 1.0000 |
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

The multiclass linear probe again showed a late-layer trend:

| Feature | Macro F1 | Accuracy |
|---|---:|---:|
| `readout_window_layer_00` | 0.3337 | 0.3417 |
| `readout_window_layer_07` | 0.5702 | 0.5750 |
| `readout_window_layer_14` | 0.7813 | 0.7833 |
| `readout_window_layer_21` | 0.8492 | 0.8500 |
| `readout_window_layer_28` | 0.8997 | 0.9000 |

## Probe And Baseline Result

The best single-layer linear activation probe was
`readout_window_layer_28`.

| Evaluation | Method | Feature | Macro F1 | Accuracy | Confusion Matrix |
|---|---|---|---:|---:|---|
| Stratified | Activation probe | `readout_window_layer_28` | 0.8997 | 0.9000 | `[[55, 5], [7, 53]]` |
| Stratified | Word TF-IDF | `word_tfidf_1_2` | 0.1955 | 0.2000 | `[[13, 47], [49, 11]]` |
| Stratified | Char TF-IDF | `char_wb_tfidf_3_5` | 0.2111 | 0.2167 | `[[14, 46], [48, 12]]` |
| Family-grouped | Activation probe | `readout_window_layer_28` | 0.9231 | 0.9233 | `[[54, 6], [3, 57]]` |
| Family-grouped | Word TF-IDF | `word_tfidf_1_2` | 0.4848 | 0.5000 | `[[26, 34], [26, 34]]` |
| Family-grouped | Char TF-IDF | `char_wb_tfidf_3_5` | 0.4896 | 0.5367 | `[[41, 19], [37, 23]]` |

Additional word-only diagnostics:

| Method | Evaluation | Macro F1 | Accuracy |
|---|---|---:|---:|
| Word TF-IDF 1-gram | Stratified | 0.2063 | 0.2083 |
| Word TF-IDF 1-gram | Family-grouped | 0.4946 | 0.5000 |
| Word TF-IDF 1-2 gram | Stratified | 0.1955 | 0.2000 |
| Word TF-IDF 1-2 gram | Family-grouped | 0.4848 | 0.5000 |
| Word TF-IDF 1-3 gram | Stratified | 0.3198 | 0.3250 |
| Word TF-IDF 1-3 gram | Family-grouped | 1.0000 | 1.0000 |

## Interpretation

This is the first corpus where the readout-window activation probe clearly
beats word 1-2 and character text baselines under family-grouped evaluation.
That is the intended diagnostic result: the previous bigram shortcut was
removed while the activation signal remained strong.

The result is still not a final CIFT claim. Word 1-3 TF-IDF is perfect under
family-grouped evaluation because the deterministic construction leaves a
trigram-level template cue. Phrases such as the allowed/denied action frame
remain too stable across families.

The useful finding is narrower and still important: when unigram and bigram
surface cues are neutralized, late readout-window activations continue to carry
predictive signal. The next data pass should preserve this pressure while
removing fixed trigram grammar.

## Next Step

Generate a grammar-varied crossed-action corpus:

- Keep matched tool payloads and no credential placeholders.
- Preserve strict unigram balance within each pair.
- Keep high bigram overlap, but vary the grammar around allowed and denied
  action clauses.
- Avoid a fixed phrase such as `allowed action is ... denied action is ...`.
- Require grouped word 1-2 and word 1-3 baselines before interpreting any
  activation improvement.

The next target is not simply "more examples." It is a corpus where the same
semantic contrast is expressed through many syntactic frames.
