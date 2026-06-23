# Watchman Semantic v3 Selected-Choice CIFT Milestone

## Scope

This run evaluates the fresh Watchman semantic-indirection v3 trace corpus with real hidden-state extraction from `Qwen/Qwen3-0.6B` on MPS. It focuses on the binary CIFT task `safe_secret_vs_exfiltration`: distinguish safe credential handling from exfiltration-oriented credential handling before output generation.

The full structured prompt corpus has 720 rows:

- `benign`: 240 rows
- `secret_present_safe`: 240 rows
- `exfiltration_intent`: 240 rows

Selected-choice CIFT requires an explicit selected-choice readout span, so this run uses the 480 secret-present rows. Benign rows remain valid for other tasks, but they intentionally have no selected-choice target.

## Artifacts

- Structured source: `data/trace_collection/structured_prompts.paired_semantic_indirection_v3_720.jsonl`
- Derived secret-present slice: `data/trace_collection/structured_prompts.paired_semantic_indirection_v3_480_secret_present_binary.jsonl`
- Activation artifact: `data/trace_collection/qwen3_0_6b_watchman_semantic_v3_480_windows_l19_l22.pt`
- Activation artifact SHA-256: `30d61ad3b08d493fd695d6a964760e09f3b25fa712676024f59d2eace8724fce`
- Local runtime candidate: `introspection/data/models/cift_qwen3_0_6b_watchman_semantic_v3_selected_choice_layer_19_runtime_candidate_v1.pkl`
- Runtime candidate SHA-256: `56af251cba7c7c1c55ffe00350ad66894669debd5f825039e851ca19f421a333`

The `.pt`, `.jsonl`, and model bundle artifacts are local experiment outputs and are intentionally not committed.

## Extraction

Hidden-state extraction used late layers 19 through 22 and four readout families:

- `selected_choice_window`
- `readout_window`
- `query_tail_window`
- `combined_readout_window`

The local smoke test confirmed `Qwen/Qwen3-0.6B` exposes 29 hidden-state tensors and selected `mps` as the execution device.

## Results

Evaluation used stratified group k-fold cross-validation, grouped by domain family.

| Rank | Feature | Macro F1 | Accuracy |
|---:|---|---:|---:|
| 1 | `selected_choice_window_layer_19` | 1.0000 | 1.0000 |
| 2 | `selected_choice_window_layer_20` | 1.0000 | 1.0000 |
| 3 | `selected_choice_window_layer_21` | 0.9975 | 0.9975 |
| 4 | `selected_choice_window_layer_22` | 0.9908 | 0.9908 |
| 5 | `combined_readout_window_layer_19` | 0.9489 | 0.9492 |
| 9 | `readout_window_layer_19` | 0.5524 | 0.5558 |
| 13 | `query_tail_window_layer_22` | 0.4982 | 0.5058 |

Grouped text baselines on the same task were near chance:

| Method | Macro F1 | Accuracy |
|---|---:|---:|
| `word_tfidf` | 0.4908 | 0.4925 |
| `char_tfidf` | 0.4955 | 0.4967 |

## Interpretation

The selected-choice readout window is the decisive signal on this corpus. Plain payload/readout windows and query-tail windows remain weak, while selected-choice hidden states separate safe handling from exfiltration intent even when grouped by family. This supports the current direction: CIFT should consume explicit semantic-choice geometry emitted by the trace generator or runtime, not rely on broad payload windows.

The result is strong enough to promote `selected_choice_window_layer_19` as a local runtime candidate for integration testing, but it is not final evidence of production robustness. The corpus is synthetic and semantically structured by design. The next validation step is transfer: run the same selected-choice candidate against a held-out corpus with new task families, different credential formats, and adversarial paraphrases.

## Follow-Up

- Add a reproducible single-task grouped baseline command or script so future selected-choice experiments do not rely on ad hoc text-baseline commands.
- Evaluate transfer on a fresh v3 corpus generated with a different participant id and seed profile.
- Compare the selected-choice candidate with the paper-faithful CCI/CFS implementation on the same 480-row slice.
- Wire the local runtime candidate into the Aegis CIFT detector behind an explicit artifact path configuration.
