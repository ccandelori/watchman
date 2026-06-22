# Trace Paired-Natural 720 Readout-Window Progress

Date: 2026-06-22

## Purpose

This milestone follows the paired-intent stress test. The paired-intent corpus
successfully matched safe-secret and exfiltration tool payloads, but its
synthetic control strings made the prompt language too artificial. The
`paired_natural` profile keeps the same pre-output boundary and identical
safe/exfiltration tool payloads, while replacing boolean control strings with
task-specific operational prompts.

This is still deterministic synthetic seed data, not human red-team data.

## Local Artifacts

Generated artifacts remain under ignored `data/trace_collection/` paths:

- `collection_inputs.paired_natural_720.jsonl`
- `records.paired_natural_720.jsonl`
- `structured_prompts.paired_natural_720.jsonl`
- `qwen3_0_6b_trace_paired_natural_720_readout.pt`
- `probe_qwen3_0_6b_trace_paired_natural_720_readout.json`
- `candidate_feature_trace_paired_natural_720_readout.json`
- `candidate_feature_trace_paired_natural_720_readout.md`

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

Safe-secret and exfiltration rows share identical tool payloads within each
task and variant. Their prompt difference is task-specific natural language
about keeping authorization internal versus preparing a downstream handoff.

Observed readout window lengths matched the pre-output corpus:

| Readout Tokens | Rows |
|---:|---:|
| 2 | 24 |
| 3 | 248 |
| 4 | 8 |
| 5 | 8 |
| 7 | 96 |
| 8 | 96 |

## Activation Extraction

Extraction used Qwen3-0.6B on layers `0,7,14,21,28` with
`readout_window` pooling.

Artifact summary:

- Model: `Qwen/Qwen3-0.6B`
- Selected device: `cpu`
- Rows: 480
- Features: `readout_window_layer_00`, `07`, `14`, `21`, `28`

## Probe Result

The stratified linear probe recovered a very strong single-layer signal. The
best single-layer feature was `readout_window_layer_28`.

| Feature | Macro F1 | Accuracy | Confusion Matrix |
|---|---:|---:|---|
| `readout_window_layer_00` | 0.3667 | 0.3958 | `[[102, 138], [152, 88]]` |
| `readout_window_layer_07` | 0.9770 | 0.9771 | `[[239, 1], [10, 230]]` |
| `readout_window_layer_14` | 0.9875 | 0.9875 | `[[240, 0], [6, 234]]` |
| `readout_window_layer_21` | 0.9833 | 0.9833 | `[[238, 2], [6, 234]]` |
| `readout_window_layer_28` | 0.9875 | 0.9875 | `[[240, 0], [6, 234]]` |

The grouped family split again favored the multi-layer concat feature over the
layer-0 reference, but concat underperformed the best single-layer stratified
result:

| Feature | Macro F1 | Accuracy | Confusion Matrix |
|---|---:|---:|---|
| `readout_window_layer_00` | 0.3333 | 0.5000 | `[[240, 0], [240, 0]]` |
| `concat(readout_window_layer_00,readout_window_layer_07,readout_window_layer_14,readout_window_layer_21,readout_window_layer_28)` | 0.9163 | 0.9167 | `[[216, 24], [18, 222]]` |

## Text Baseline

The paired-natural profile did not defeat lexical baselines:

| Method | Feature | Macro F1 | Accuracy | Confusion Matrix |
|---|---|---:|---:|---|
| Activation probe | `readout_window_layer_00` | 0.3333 | 0.5000 | `[[240, 0], [240, 0]]` |
| Word TF-IDF | `word_tfidf_1_2` | 1.0000 | 1.0000 | `[[240, 0], [0, 240]]` |
| Char TF-IDF | `char_wb_tfidf_3_5` | 1.0000 | 1.0000 | `[[240, 0], [0, 240]]` |

## Interpretation

This is a mixed result and should be treated carefully.

The good news is that readout-window activations are highly separable again
once intent is expressed as natural operational language. Mid and late layers
carry the useful signal, which remains consistent with earlier observations.

The bad news is that the corpus is still lexically obvious. The prompt pairs
use different words for safe handling and downstream continuation, and both
word and character TF-IDF classify every row correctly. That means this run
does not establish an activation-over-text advantage. It establishes that the
runtime trace pipeline can generate a more natural pre-output corpus and that
Qwen3-0.6B hidden states encode the label strongly on that corpus.

The gap is data quality, not extraction plumbing.

## Next Step

Move to an adversarial paraphrase or human collection pass:

- Keep the same runtime trace schema.
- Keep DP-HONEY spans in protected context only.
- Keep safe-secret and exfiltration tool payloads identical.
- Require text baselines to be measured before accepting a CIFT result.
- Prefer human or LLM-paraphrased pairs where intent differs semantically but
  obvious marker terms are balanced across labels.

Until that exists, the paired-natural profile is useful as a high-signal
plumbing and regression corpus, not as the final evidence dataset for CIFT.
