# Qwen3-4B Query-Tail Fallback Experiment

## Purpose

The live CIFT selector benchmark showed that the selected-choice route remained strong, but the fallback route was weak. The fallback route previously pooled over tool payload/readout positions, which often contain operational placeholder text rather than the user-side intent signal. This experiment tested whether a query-tail readout window would be a better fallback.

## Artifacts

- Structured prompts: `data/trace_collection/structured_prompts.paired_semantic_indirection_v3_720_secret_present_binary_query_tail.jsonl`
- Activation artifact: `introspection/data/activations/qwen3_4b_paired_semantic_indirection_v3_secret_present_binary_query_tail_l19_l22.pt`
- Ablation report: `introspection/data/reports/qwen3_4b_query_tail_window_ablation_2026-06-22.md`
- Query-tail error analysis: `introspection/data/reports/qwen3_4b_query_tail_error_analysis_2026-06-22.md`
- Readout-window error analysis: `introspection/data/reports/qwen3_4b_readout_window_l19_l22_error_analysis_2026-06-22.md`

## Result

Grouped CV on the 480-row secret-present binary slice:

| Variant | Macro F1 | Accuracy | Errors |
|---|---:|---:|---:|
| `selected_choice_concat_19_22` | 1.0000 | 1.0000 | 0 |
| `readout_window_concat_19_22` | 0.6856 | 0.6908 | 149 |
| `query_tail_concat_19_22` | 0.5589 | 0.5883 | 201 |

The query-tail fallback is worse than the existing readout-window fallback. It also fails in broad family-level blocks rather than in a narrow set of isolated examples. Several families flip almost entirely, which suggests the query tail is not consistently carrying the intent signal.

## Interpretation

The query-tail hypothesis was plausible because the tail of the user query is after the secret context and near the end of the request. On this corpus, however, the query tail often lands on task scaffolding or response-format language instead of the semantic decision cue. The selected-choice window remains strong because it points directly at the disambiguating cue in the prompt.

This means the selected-choice route is still the best current CIFT readout route for the semantic-indirection corpus. Query-tail should not be promoted as the fallback. The fallback path needs either a better structural cue, a route-specific classifier with conservative policy semantics, or an explicit unavailable/degraded evidence result when the runtime lacks a reliable readout span.

## Decision

- Keep `selected_choice_window` as the strong route for this corpus.
- Keep `readout_window` as the fallback baseline for comparison.
- Do not promote `query_tail_window`.
- Train a Qwen3-4B selected-choice concat bundle as the next runtime candidate.
