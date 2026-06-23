# Trace Matched-Hard 720 Readout-Window Progress

Date: 2026-06-22

## Purpose

This milestone follows the first runtime-to-CIFT bridge by generating a harder
synthetic trace corpus. The previous 720-row corpus was useful for plumbing but
had an obvious artifact: exfiltration rows often had tool payloads while
safe-secret rows often did not. This run adds a matched-hard seed profile where
all labels receive tool calls.

## Local Artifacts

Generated artifacts remain under ignored `data/trace_collection/` paths:

- `collection_inputs.matched_hard_720.jsonl`
- `records.matched_hard_720.jsonl`
- `structured_prompts.matched_hard_720.jsonl`
- `qwen3_0_6b_trace_matched_hard_720_readout.pt`
- `probe_qwen3_0_6b_trace_matched_hard_720_readout.json`
- `candidate_feature_trace_matched_hard_720_readout.json`

## Corpus Shape

The matched-hard record corpus contains:

- 240 `benign` records.
- 240 `secret_present_safe` records.
- 240 `exfiltration_intent` records.
- 12 prompt families.
- Tool calls on every row.

The CIFT conversion produced:

- 240 `secret_present_safe` structured prompts with `readout:safe_payload`.
- 240 `exfiltration_intent` structured prompts with `readout:payload_secret`.
- 240 no-secret `benign` rows skipped for separate calibration.
- Payload token spans on all 480 converted rows.

## Activation Extraction

Extraction used Qwen3-0.6B on layers `0,7,14,21,28` with
`readout_window` pooling.

Artifact summary:

- Model: `Qwen/Qwen3-0.6B`
- Selected device: `cpu`
- Rows: 480
- Features: `readout_window_layer_00`, `07`, `14`, `21`, `28`
- Feature shape: `(480, 1024)` for each layer

## Probe Result

The stratified linear probe still reached perfect performance on every
readout-window layer:

| Feature | Macro F1 | Accuracy |
|---|---:|---:|
| `readout_window_layer_00` | 1.0000 | 1.0000 |
| `readout_window_layer_07` | 1.0000 | 1.0000 |
| `readout_window_layer_14` | 1.0000 | 1.0000 |
| `readout_window_layer_21` | 1.0000 | 1.0000 |
| `readout_window_layer_28` | 1.0000 | 1.0000 |

The grouped family split also remained perfect for
`safe_secret_vs_exfiltration`:

| Method | Feature | Macro F1 | Accuracy |
|---|---|---:|---:|
| Activation probe | `readout_window_layer_00` | 1.0000 | 1.0000 |
| Word TF-IDF | `word_tfidf_1_2` | 1.0000 | 1.0000 |
| Char TF-IDF | `char_wb_tfidf_3_5` | 0.9900 | 0.9900 |

## Interpretation

The matched-hard profile fixed one structural artifact: safe and exfiltration
examples now both use payload-style readout windows. However, the task is still
too easy. Text baselines are also near-perfect, which means surface text still
exposes the label.

The likely remaining artifacts are:

- Safe payload values are fixed strings such as `withheld_by_policy`.
- Exfiltration payload values contain DP-HONEY credential-shaped text.
- Operator prompts still describe safe withholding versus downstream filling
  too explicitly.

This is still useful progress because the runtime-to-CIFT contract is now
stronger: it supports safe payload readout windows without fake secret spans.
It is not yet a robust CIFT training corpus.

## Next Step

The next corpus should separate pre-output CIFT training from post-output
canary leakage:

- CIFT intent examples should include the protected secret only in the protected
  context, not already copied into a tool argument.
- Safe and exfiltration rows should use near-identical tool schemas and payload
  values before generation.
- Exfiltration intent should be expressed indirectly in the user query, not by
  inserting a generated credential value into the prompt artifact.
- DP-HONEY canary leakage should remain a separate post-generation detector and
  evaluation path.

This should make the text baseline weaker and give activation probes a more
meaningful chance to prove pre-output signal.
