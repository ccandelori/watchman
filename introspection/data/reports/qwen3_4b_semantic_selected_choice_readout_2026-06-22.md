# Qwen3-4B Semantic Selected-Choice Readout

Date: 2026-06-22

## Purpose

Test the hypothesis from semantic transfer error analysis:

> The detector was responding to the exfiltration option being mentioned, not to
> the final neutral option being selected.

The experiment adds selected-choice span metadata to semantic-indirection
structured prompts and extracts three Qwen3-4B readout families over layers
19-22:

- `readout_window`: the existing payload/query readout window.
- `selected_choice_window`: the final selected neutral choice clause.
- `combined_readout_window`: the union of payload/query and selected-choice
  windows.

## New Artifacts

Structured prompt artifacts:

```text
data/trace_collection/structured_prompts.paired_semantic_indirection_default_720_selected_choice_v2.jsonl
data/trace_collection/structured_prompts.paired_semantic_indirection_default_720_secret_present_binary_selected_choice_v2.jsonl
```

Activation artifact:

```text
data/trace_collection/qwen3_4b_trace_paired_semantic_indirection_default_720_secret_present_binary_selected_choice_v2_l19_l22_readouts.pt
```

Comparison reports:

```text
data/trace_collection/qwen3_4b_semantic_selected_choice_v2_l19_l22_vs_payload_readout.md
data/trace_collection/qwen3_4b_semantic_combined_choice_v2_l19_l22_vs_payload_readout.md
data/trace_collection/qwen3_4b_semantic_combined_choice_v2_l19_l22_vs_selected_choice.md
```

## Result

Task: grouped `safe_secret_vs_exfiltration` on semantic-indirection secret rows.

Feature: Qwen3-4B layers 19-22 concatenated.

| Feature | Macro F1 | Accuracy | Confusion Matrix |
|---|---:|---:|---|
| payload/query readout | 0.6991 | 0.7042 | `[155, 85] [58, 182]` |
| selected-choice readout | 1.0000 | 1.0000 | `[240, 0] [0, 240]` |
| combined readout | 1.0000 | 1.0000 | `[240, 0] [0, 240]` |

Selected-choice and combined readouts tie exactly. The selected-choice-only
feature is therefore the cleaner interpretation: the signal is concentrated in
the selected neutral choice clause, not in the payload readout.

Grouped text baselines on the same binary slice remain near chance:

| Baseline | Macro F1 | Accuracy | Confusion Matrix |
|---|---:|---:|---|
| word TF-IDF 1-2 | 0.4749 | 0.4775 | `[104, 136] [115, 125]` |
| char TF-IDF 3-5 | 0.4963 | 0.5050 | `[114, 126] [112, 128]` |

## Interpretation

This closes the immediate semantic error-analysis loop. The old readout window
was looking in the wrong place for this profile. When both labels mention both
safe and exfiltration options, the payload/query readout confuses "option
mentioned" with "option selected." Pooling hidden states over the selected
choice clause lets a linear activation probe recover the intended distinction.

The result is also project-aligned: a proxy can know or derive these structural
windows when it renders policy/options for a model. This supports the broader
CIFT direction of using proxy-owned span geometry rather than only final-token
or payload-tail pooling.

## Caveats

The perfect score should not be treated as production-ready evidence.

- This is an in-profile grouped evaluation over a controlled semantic template.
- The held-out groups are task families, not entirely new semantic grammars.
- The current selected-choice window exists only for semantic-indirection rows.
  Natural, adversarial, and pre-output profiles need analogous decision-span
  metadata before mixed-profile transfer can use this feature.

The strongest honest claim is narrower and useful: selected-choice span geometry
fixes the specific semantic-indirection failure mode that payload/query readout
could not fix.

## Next Move

Create a true semantic v3 transfer split:

1. Generate semantic rows with held-out code names, selector phrases, and option
   orderings.
2. Keep both safe and exfiltration action text in every row.
3. Emit selected-choice spans for every semantic row.
4. Train on semantic v2 plus existing profiles where compatible.
5. Hold out semantic v3 and score selected-choice, payload/query, and combined
   readouts separately.

This is the next real test of whether selected-choice CIFT generalizes beyond
the current controlled grammar.

## Follow-Up: Semantic V3 Transfer

The semantic v3 holdout was completed:

```text
introspection/data/reports/qwen3_4b_semantic_v3_transfer_2026-06-22.md
```

Selected-choice layers 19-22 transferred perfectly from semantic v2 to semantic
v3 and from semantic v3 back to semantic v2:

| Train | Test | Macro F1 | Accuracy |
|---|---|---:|---:|
| semantic v2 | semantic v3 | 1.0000 | 1.0000 |
| semantic v3 | semantic v2 | 1.0000 | 1.0000 |

The payload/query readout did not transfer from v2 to v3, reaching only 0.5814
macro F1. This strengthens the conclusion that the selected-choice window is
the useful geometry for semantic-indirection prompts.
