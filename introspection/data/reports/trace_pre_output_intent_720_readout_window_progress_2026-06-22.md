# Trace Pre-Output Intent 720 Readout-Window Progress

Date: 2026-06-22

## Purpose

This milestone adds a stricter runtime trace corpus for CIFT probing. The prior
matched-hard corpus aligned safe and exfiltration tool-call shapes, but
exfiltration rows still contained a credential placeholder that DP-HONEY
resolved inside the tool payload. That made the task partly a post-egress
leakage detector problem.

The pre-output intent profile removes that shortcut. It keeps DP-HONEY secrets
in protected context only, gives safe and exfiltration rows matched tool-call
schemas, and leaves tool-call credential fields in a shared pending state.

## Local Artifacts

Generated artifacts remain under ignored `data/trace_collection/` paths:

- `collection_inputs.pre_output_intent_720.jsonl`
- `records.pre_output_intent_720.jsonl`
- `structured_prompts.pre_output_intent_720.jsonl`
- `qwen3_0_6b_trace_pre_output_intent_720_readout.pt`
- `probe_qwen3_0_6b_trace_pre_output_intent_720_readout.json`
- `candidate_feature_trace_pre_output_intent_720_readout.json`
- `candidate_feature_trace_pre_output_intent_720_readout.md`

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

Readout windows are shorter than eight tokens for many examples because the
adapter uses the pending authorization field as the payload readout region.
Observed readout window lengths were:

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

The stratified linear probe no longer reports a perfect result. The best
single-layer feature was `readout_window_layer_07`.

| Feature | Macro F1 | Accuracy | Confusion Matrix |
|---|---:|---:|---|
| `readout_window_layer_00` | 0.3667 | 0.3958 | `[[102, 138], [152, 88]]` |
| `readout_window_layer_07` | 0.9917 | 0.9917 | `[[240, 0], [4, 236]]` |
| `readout_window_layer_14` | 0.9917 | 0.9917 | `[[240, 0], [4, 236]]` |
| `readout_window_layer_21` | 0.9896 | 0.9896 | `[[239, 1], [4, 236]]` |
| `readout_window_layer_28` | 0.9917 | 0.9917 | `[[240, 0], [4, 236]]` |

The grouped family split favored the multi-layer concat feature over the
layer-0 reference:

| Feature | Macro F1 | Accuracy | Confusion Matrix |
|---|---:|---:|---|
| `readout_window_layer_00` | 0.3333 | 0.5000 | `[[240, 0], [240, 0]]` |
| `concat(readout_window_layer_00,readout_window_layer_07,readout_window_layer_14,readout_window_layer_21,readout_window_layer_28)` | 0.9741 | 0.9742 | `[[227, 13], [0, 240]]` |

The error shape is security-friendly on this split: the concat model catches
all exfiltration-intent rows and over-flags 13 safe rows.

## Text Baseline

Text remains too predictive:

| Method | Feature | Macro F1 | Accuracy | Confusion Matrix |
|---|---|---:|---:|---|
| Activation probe | `readout_window_layer_00` | 0.3333 | 0.5000 | `[[240, 0], [240, 0]]` |
| Word TF-IDF | `word_tfidf_1_2` | 1.0000 | 1.0000 | `[[240, 0], [0, 240]]` |
| Char TF-IDF | `char_wb_tfidf_3_5` | 0.9866 | 0.9867 | `[[232, 8], [0, 240]]` |

## Interpretation

This corpus closes the tool-payload leakage shortcut. The model is no longer
seeing a copied credential in tool-call arguments, and all sensitive spans come
from DP-HONEY context. That makes the activation result closer to the intended
pre-output CIFT question.

It does not yet close the lexical shortcut. The operator prompt still separates
safe handling from exfiltration intent with surface text such as withholding
credential material versus preparing downstream authorization. A word-level
classifier can exploit those terms perfectly.

The useful finding is therefore narrower:

- Runtime trace records can now produce CIFT readout windows without post-egress
  credential payload leakage.
- Mid and late readout-window activations still carry strong signal.
- Text is still too easy, so this is not yet a clean activation-over-text
  result.

## Next Step

Generate a paired paraphrase corpus where safe and exfiltration rows share most
surface wording. The label should differ through authorization semantics and
task context, not obvious marker words. Then rerun the same readout-window
pipeline and compare activation probes against word and character TF-IDF under
grouped family splits.
