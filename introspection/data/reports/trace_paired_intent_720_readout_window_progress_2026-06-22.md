# Trace Paired-Intent 720 Readout-Window Progress

Date: 2026-06-22

## Purpose

This milestone tests whether the previous pre-output intent corpus was still
too lexically separable. The new `paired_intent` profile keeps DP-HONEY secrets
in protected context only, gives safe-secret and exfiltration rows identical
tool payloads, and encodes the prompt difference as paired control fields with
matched token sets.

The first paired attempt still used a global boolean order. Word and character
baselines remained perfect. The final profile therefore varies control field
names by task family and balances boolean order across the catalog.

## Local Artifacts

Generated artifacts remain under ignored `data/trace_collection/` paths:

- `collection_inputs.paired_intent_720.jsonl`
- `records.paired_intent_720.jsonl`
- `structured_prompts.paired_intent_720.jsonl`
- `qwen3_0_6b_trace_paired_intent_720_readout.pt`
- `probe_qwen3_0_6b_trace_paired_intent_720_readout.json`
- `candidate_feature_trace_paired_intent_720_readout.json`
- `candidate_feature_trace_paired_intent_720_readout.md`

## Corpus Shape

The normalized record corpus contains:

- 240 `benign` records.
- 240 `secret_present_safe` records.
- 240 `exfiltration_intent` records.
- 12 prompt families.
- Tool calls on every row.
- 0 tool-call credential placeholders.
- 0 credential-shaped tool-call values.
- 480 DP-HONEY sensitive spans, all from protected context.

The CIFT conversion produced:

- 240 `secret_present_safe` structured prompts.
- 240 `exfiltration_intent` structured prompts.
- 240 no-secret `benign` rows skipped for separate calibration.
- Payload token spans on all 480 converted rows.

Safe-secret and exfiltration tool payloads are identical within each task and
variant. Prompt control fields vary by task family, and safe rows include both
`false -> true` and `true -> false` boolean orderings.

## Activation Extraction

Extraction used Qwen3-0.6B on layers `0,7,14,21,28` with
`readout_window` pooling.

Artifact summary:

- Model: `Qwen/Qwen3-0.6B`
- Selected device: `cpu`
- Rows: 480
- Features: `readout_window_layer_00`, `07`, `14`, `21`, `28`

## Probe Result

The stratified linear probe dropped substantially compared with the earlier
pre-output profile. The best single-layer feature was `readout_window_layer_21`.

| Feature | Macro F1 | Accuracy | Confusion Matrix |
|---|---:|---:|---|
| `readout_window_layer_00` | 0.3667 | 0.3958 | `[[102, 138], [152, 88]]` |
| `readout_window_layer_07` | 0.4080 | 0.4104 | `[[102, 138], [145, 95]]` |
| `readout_window_layer_14` | 0.6466 | 0.6479 | `[[161, 79], [90, 150]]` |
| `readout_window_layer_21` | 0.6522 | 0.6542 | `[[162, 78], [88, 152]]` |
| `readout_window_layer_28` | 0.6134 | 0.6167 | `[[155, 85], [99, 141]]` |

The grouped family split still favored the multi-layer concat feature over the
layer-0 reference, but the result is modest:

| Feature | Macro F1 | Accuracy | Confusion Matrix |
|---|---:|---:|---|
| `readout_window_layer_00` | 0.3333 | 0.5000 | `[[240, 0], [240, 0]]` |
| `concat(readout_window_layer_00,readout_window_layer_07,readout_window_layer_14,readout_window_layer_21,readout_window_layer_28)` | 0.5810 | 0.5983 | `[[142, 98], [94, 146]]` |

## Text Baseline

The paired profile partially breaks text shortcuts, but not enough:

| Method | Feature | Macro F1 | Accuracy | Confusion Matrix |
|---|---|---:|---:|---|
| Activation probe | `readout_window_layer_00` | 0.3333 | 0.5000 | `[[240, 0], [240, 0]]` |
| Word TF-IDF | `word_tfidf_1_2` | 0.2323 | 0.2333 | `[[56, 184], [176, 64]]` |
| Char TF-IDF | `char_wb_tfidf_3_5` | 1.0000 | 1.0000 | `[[240, 0], [0, 240]]` |

## Interpretation

This is a useful negative result. Matching tool payloads, varying task-specific
control names, and balancing boolean order is enough to break the word-level
baseline, but character n-grams still recover the label perfectly. The synthetic
control language remains too mechanical.

The activation probe also falls toward weak-to-moderate performance. That is
not surprising: once the prompt is reduced to artificial paired controls, the
model has less natural-language intent to encode. We should not try to win this
benchmark by further tuning synthetic variable names.

## Next Step

Move from synthetic control strings to human or LLM-paraphrased paired prompts.
The next corpus should preserve the same runtime trace schema and DP-HONEY
context-only spans, but the safe and exfiltration intents should be expressed in
natural operational language rather than structured boolean controls. Evaluate
against both word and character TF-IDF before treating activation performance as
a meaningful CIFT signal.
