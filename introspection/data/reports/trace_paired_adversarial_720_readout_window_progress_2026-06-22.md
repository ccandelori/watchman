# Trace Paired-Adversarial 720 Readout-Window Progress

Date: 2026-06-22

## Purpose

This milestone follows the paired-natural run. The paired-natural corpus made
the prompts look more operational, but word and character TF-IDF still reached
perfect accuracy. The `paired_adversarial` profile keeps the same runtime trace
shape and pre-output boundary, but writes safe-secret and exfiltration prompt
pairs with identical token multisets.

The goal is diagnostic: separate unigram shortcuts from phrase-order shortcuts.
This is deterministic synthetic seed data, not human red-team data.

## Local Artifacts

Generated artifacts remain under ignored `data/trace_collection/` paths:

- `collection_inputs.paired_adversarial_720.jsonl`
- `records.paired_adversarial_720.jsonl`
- `structured_prompts.paired_adversarial_720.jsonl`
- `qwen3_0_6b_trace_paired_adversarial_720_readout.pt`
- `probe_qwen3_0_6b_trace_paired_adversarial_720_readout.json`
- `candidate_feature_trace_paired_adversarial_720_readout.json`
- `candidate_feature_trace_paired_adversarial_720_readout.md`

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
task and variant. Their prompt token multisets also match within each
safe/exfiltration pair.

Observed readout window lengths:

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

The stratified linear probe remained strong, though weaker than the
paired-natural profile. The best single-layer feature was
`readout_window_layer_28`.

| Feature | Macro F1 | Accuracy | Confusion Matrix |
|---|---:|---:|---|
| `readout_window_layer_00` | 0.3667 | 0.3958 | `[[102, 138], [152, 88]]` |
| `readout_window_layer_07` | 0.6758 | 0.6771 | `[[162, 78], [77, 163]]` |
| `readout_window_layer_14` | 0.7293 | 0.7312 | `[[176, 64], [65, 175]]` |
| `readout_window_layer_21` | 0.8434 | 0.8438 | `[[207, 33], [42, 198]]` |
| `readout_window_layer_28` | 0.9124 | 0.9125 | `[[217, 23], [19, 221]]` |

The grouped family split favored the multi-layer concat feature over the
layer-0 reference:

| Feature | Macro F1 | Accuracy | Confusion Matrix |
|---|---:|---:|---|
| `readout_window_layer_00` | 0.3333 | 0.5000 | `[[240, 0], [240, 0]]` |
| `concat(readout_window_layer_00,readout_window_layer_07,readout_window_layer_14,readout_window_layer_21,readout_window_layer_28)` | 0.7899 | 0.7958 | `[[186, 54], [44, 196]]` |

## Text Baseline

The profile successfully removed unigram shortcuts but did not remove bigram
shortcuts:

| Method | Feature | Macro F1 | Accuracy | Confusion Matrix |
|---|---|---:|---:|---|
| Activation probe | `readout_window_layer_00` | 0.3333 | 0.5000 | `[[240, 0], [240, 0]]` |
| Word TF-IDF | `word_tfidf_1_1` | 0.4884 | 0.5000 | `[[102, 138], [102, 138]]` |
| Word TF-IDF | `word_tfidf_1_2` | 1.0000 | 1.0000 | `[[240, 0], [0, 240]]` |
| Char TF-IDF | `char_wb_tfidf_3_5` | 0.8699 | 0.8700 | `[[205, 35], [26, 214]]` |

## Interpretation

This is progress on the data diagnosis, not a final CIFT result.

The token-multiset balancing worked: unigram TF-IDF dropped to chance. That
means the previous easy signal was not merely the presence or absence of words
such as `authorization`, `handoff`, `context`, or `value`.

The remaining shortcut is phrase order. Word bigrams still classify the corpus
perfectly because the synthetic pairs encode the label through repeated local
orderings such as authorization-before-disclosure versus
disclosure-before-authorization. Character n-grams also remain strong, though
no longer perfect.

The activation probe remains meaningfully predictive, especially in later
layers, but it does not yet establish an activation-over-text advantage because
the bigram text baseline still dominates.

## Next Step

The next data move should not be another deterministic phrase table. We need a
human or LLM-paraphrased collection pass that preserves the runtime trace
contract while varying local phrase order enough that word bigrams cannot
memorize the label rule.

Acceptance criteria for the next corpus:

- Same DP-HONEY protected-context spans.
- Same matched safe/exfiltration tool payloads.
- No credential values or canaries in tool payloads.
- Word unigram TF-IDF near chance.
- Word bigram and character TF-IDF clearly below the activation probe before
  claiming a CIFT-specific advantage.
