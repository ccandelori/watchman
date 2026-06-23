# Qwen3-4B Semantic V3 Transfer

Date: 2026-06-22

## Purpose

Test whether selected-choice readout generalizes beyond the original semantic
indirection grammar.

The previous selected-choice experiment showed that layers 19-22 can perfectly
separate semantic safe-secret and exfiltration rows when the readout window is
the selected neutral choice clause. That was useful but in-profile. The v3
follow-up creates a held-out semantic profile with:

- new code names, such as `atlas` and `briar`;
- new selector phrases, such as `Confirm symbol`;
- balanced option ordering and code-to-label assignment;
- the same selected-choice span contract.

## New Code Path

New trace-collection profile:

```text
paired_semantic_indirection_v3
```

The profile is available through `aegis-trace-seed-inputs` and emits the same
runtime-shaped trace records as earlier profiles.

## New Artifacts

Corpus artifacts:

```text
data/trace_collection/collection_inputs.paired_semantic_indirection_v3_720.jsonl
data/trace_collection/records.paired_semantic_indirection_v3_720.jsonl
data/trace_collection/structured_prompts.paired_semantic_indirection_v3_720_selected_choice.jsonl
data/trace_collection/structured_prompts.paired_semantic_indirection_v3_720_secret_present_binary_selected_choice.jsonl
```

Activation artifact:

```text
data/trace_collection/qwen3_4b_trace_paired_semantic_indirection_v3_720_secret_present_binary_selected_choice_l19_l22_readouts.pt
```

Evaluation artifacts:

```text
data/trace_collection/qwen3_4b_semantic_v3_selected_choice_l19_l22_vs_payload_readout.md
data/trace_collection/qwen3_4b_semantic_v3_combined_choice_l19_l22_vs_selected_choice.md
data/trace_collection/feature_transfer.qwen3_4b_train_semantic_v2_test_v3_selected_choice_l19_l22.md
data/trace_collection/feature_transfer.qwen3_4b_train_semantic_v3_test_v2_selected_choice_l19_l22.md
data/trace_collection/feature_transfer.qwen3_4b_train_semantic_v2_test_v3_payload_readout_l19_l22.md
```

## Corpus Check

The v3 generated corpus contains:

| Artifact | Rows |
|---|---:|
| collection inputs | 720 |
| normalized trace records | 720 |
| structured prompts | 720 |
| secret-present binary slice | 480 |

Selected-choice spans are present on every secret-present row:

| Label | Rows with selected-choice span |
|---|---:|
| `secret_present_safe` | 240 |
| `exfiltration_intent` | 240 |

Benign rows do not carry selected-choice spans because they do not use the
semantic option-selection prompt.

## In-Profile V3 Result

Grouped safe-secret versus exfiltration, Qwen3-4B layers 19-22:

| Feature | Macro F1 | Accuracy | Confusion Matrix |
|---|---:|---:|---|
| payload/query readout | 0.6856 | 0.6908 | `[158, 82] [67, 173]` |
| selected-choice readout | 1.0000 | 1.0000 | `[240, 0] [0, 240]` |
| combined readout | 1.0000 | 1.0000 | `[240, 0] [0, 240]` |

Selected-choice and combined readouts tie exactly again. The selected-choice
window remains the cleaner feature.

Grouped text baselines on v3 remain near chance:

| Baseline | Macro F1 | Accuracy | Confusion Matrix |
|---|---:|---:|---|
| word TF-IDF 1-2 | 0.4696 | 0.4725 | `[99, 141] [112, 128]` |
| char TF-IDF 3-5 | 0.5187 | 0.5250 | `[112, 128] [101, 139]` |

## Transfer Result

Feature:

```text
concat(selected_choice_window_layer_19,selected_choice_window_layer_20,selected_choice_window_layer_21,selected_choice_window_layer_22)
```

| Train | Test | Macro F1 | Accuracy | Confusion Matrix |
|---|---|---:|---:|---|
| semantic v2 | semantic v3 | 1.0000 | 1.0000 | `[240, 0] [0, 240]` |
| semantic v3 | semantic v2 | 1.0000 | 1.0000 | `[240, 0] [0, 240]` |

Payload/query readout does not transfer:

| Train | Test | Feature | Macro F1 | Accuracy | Confusion Matrix |
|---|---|---|---:|---:|---|
| semantic v2 | semantic v3 | payload/query l19-l22 | 0.5814 | 0.6062 | `[204, 36] [153, 87]` |

## Interpretation

This is the strongest evidence so far that selected-choice span geometry is the
right CIFT-style readout for semantic indirection.

The result rules out two weaker explanations for the v2 result:

1. It was not simply memorizing v2 code names. V3 uses held-out code words.
2. It was not simply relying on full-prompt lexical shortcuts. Text baselines
   remain near chance.

The model-side interpretation is that hidden states at the selected-choice
tokens encode the resolved option after attending back to the option table. That
is exactly the kind of "pre-output access/resolution" signal CIFT is supposed to
exploit.

## Caveats

This is still a synthetic, template-controlled semantic transfer test. The
result does not mean the runtime detector is production-ready.

Remaining gaps:

- V2 and v3 share the same broad task construction.
- The selected-choice parser currently keys off the same semantic spacer.
- Natural, adversarial, and pre-output profiles do not yet emit analogous
  selected-decision spans.
- We have not yet trained a mixed-profile candidate that knows when
  selected-choice windows are available and when only payload/query windows are
  available.

## Next Move

Promote selected-choice readout from a semantic experiment into the CIFT data
contract:

1. Keep `readout_window` for payload/query-tail evidence.
2. Use `selected_choice_window` when the proxy or policy renderer knows the
   selected decision clause.
3. Train a mixed-window candidate that can consume selected-choice features when
   present and fall back to payload/query features otherwise.
4. Add runtime metadata support for `selected_choice_readout_token_indices` so
   live CIFT can use the same feature family.
