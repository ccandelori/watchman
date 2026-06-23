# CIFT Data Generation Journey

Date: 2026-06-22

## Why This Exists

The CIFT work has not been a straight line from "make data" to "train model."
Each dataset exposed a shortcut, artifact, or missing runtime assumption. The
next dataset was designed to attack that specific weakness.

This document captures the story of the data generation work so the final
project can explain not only what worked, but why the dataset evolved.

## Guiding Principle

The target is not a high score on an easy synthetic corpus. The target is a
proxy-shaped corpus where:

- DP-HONEY-like protected context exists before the query;
- readout windows occur after both secret context and user/tool intent;
- safe-secret and exfiltration examples share task and tool shape;
- raw production secrets never cross the boundary;
- text baselines are measured before interpreting activation-probe results.

## Journey So Far

| Stage | Purpose | What We Learned | Next Pressure |
|---|---|---|---|
| Initial prompt sets | Smoke-test hidden-state extraction and simple probes. | Mid/late layers often carry useful signal, but early data was too small and too easy. | Move toward proxy-shaped prompts with spans. |
| DP-HONEY-lite structured prompts | Add secret spans, query spans, and readout-window geometry. | Token-position geometry is the key contract for CIFT-style extraction. | Replace lite stopgap with runtime/DP-HONEY-compatible traces. |
| Runtime trace collection harness | Generate `NormalizedTurn` records through the Aegis spine and DP-HONEY-backed protected context. | The introspection data can now match the runtime boundary instead of living as isolated notebook examples. | Remove artifacts where tool payload shape reveals the label. |
| Matched-hard corpus | Give labels more similar tool-call surfaces. | Performance dropped when easy tool-call artifacts were reduced. | Test pre-output intent without credential placeholders in tool payloads. |
| Pre-output intent corpus | Keep credential material in protected context only. | This better matches CIFT's purpose: pre-output credential-access intent. | Remove differences in safe/exfiltration tool payloads. |
| Paired-intent corpus | Make safe-secret and exfiltration rows share identical tool payloads. | The monitor still found signal, but control strings were artificial. | Use natural operational language. |
| Paired-natural corpus | Replace control strings with task-specific natural prompts. | Activations were strong, but word and char TF-IDF were perfect. | Attack lexical shortcuts directly. |
| Paired-adversarial corpus | Match token multisets inside each pair. | Unigram TF-IDF fell near chance, but word bigrams and char n-grams remained strong. | Vary local phrase order instead of relying on deterministic phrase tables. |
| Paired paraphrase work-item flow | Create a clean contract for human or LLM-written safe/exfil pairs. | The team now has a reusable way to collect better data without changing CIFT extraction. | Generate and evaluate paraphrased completions. |
| Synthetic paraphrase 120 | Exercise the new work-item flow end to end. | Activation beat text on random folds, but grouped text baselines were still perfect. | Use an independent paraphrase generator. |
| LM Studio one-shot 120 | Use local Qwen through LM Studio for more natural paired completions. | Naturalness improved, but all pairs failed strict lexical balance; text baselines still beat activation. | Put the validator inside the generation loop. |
| LM Studio repaired 120 | Repair pairs until they pass strict unigram/bigram validation. | Unigram shortcuts were removed, char baseline weakened, but word bigrams still dominated. | Make phrase order itself non-memorisable. |
| Crossed-action 120 | Force each pair to share identical unigram and bigram inventory while swapping allowed versus denied action semantics. | Activation beat word 1-2 and char baselines under grouped evaluation, but word 1-3 TF-IDF was perfect because the grammar frame stayed fixed. | Vary syntactic frames around the same allowed/denied semantic contrast. |
| Paired-crossed-action default 360 | Promote crossed-action generation into the trace harness and run it on the full default task catalog. | The result reproduced the activation-over-word-1-2 diagnostic at 240 CIFT rows, while confirming word 1-3 and 1-4 template shortcuts remain perfect. | Treat this profile as a regression diagnostic and build grammar-diverse traces next. |
| Paired-semantic-indirection default 720 | Use neutral option codes and flipped code-to-action mappings to prevent local n-grams from directly carrying the label. | Word TF-IDF through 1-6 fell near chance, but the linear readout activation probe was also weak. | Keep this as a hard stress test; build natural grammar-diverse boundary prompts next. |
| Dense readout tuning with benign calibration | Preserve benign no-secret rows and extract dense late/mid-late readout layers. | The signal concentrates around layers 14-17; scalar CIFT/MLP remains near chance, while residual-concat diagnostics reach 0.5809 macro F1. | Change the readout window/corpus shape before further scalar-head tuning. |
| Hidden-state source check | Verify whether the larger LM Studio Qwen3.6-27B model can serve as the PyTorch/Transformers activation source. | The local MLX 4-bit checkpoint has tokenizer/config files, but Transformers rejects its quantization metadata before model load. | Use a Transformers-compatible larger checkpoint or build a separate MLX hidden-state backend. |
| Qwen3-4B readout extraction | Move from Qwen3-0.6B CPU extraction to a Transformers-compatible Qwen3-4B MPS activation source. | The hard semantic-indirection corpus now shows a clear activation-over-text result: layers 19-22 readout concat reaches 0.7262 macro F1 while word 1-6 and char 3-5 remain near chance. | Calibrate the focused band and export a runtime-shaped candidate. |
| Qwen3-4B candidate detector | Calibrate the best readout band and freeze a first runtime-exportable CIFT candidate. | Grouped calibrated evaluation holds at 0.7185 macro F1 with ECE 0.0716; the best threshold sweep point is 0.55 with 0.7327 macro F1. A full-train bundle and `aegis.cift_runtime_linear/v1` JSON export now exist for integration tests. | Test the exported runtime model through the Aegis spine and pressure it with held-out or newly generated traces. |
| Aegis runtime-spine offline eval | Run the exported CIFT runtime JSON through `AegisRuntime` with artifact-backed feature-vector annotation. | The candidate now crosses the detector and policy boundaries: 480 active CIFT results, with 240 `allow` and 240 `warn` policy decisions on the training corpus. | Replace artifact-backed features with live model-host features, then evaluate on held-out traces. |
| Live model-host CIFT smoke | Replace the saved `.pt` feature lookup with live Qwen3-4B hidden-state extraction on MPS. | A two-row smoke run reproduced the expected safe/exfiltration decisions through `AegisRuntime`: safe-secret scored 0.062049 and exfiltration scored 0.941469. | Scale the live extractor beyond smoke testing and measure latency/cost, then run held-out traces. |
| Live model-host latency benchmark | Measure live CIFT cost over 20 runtime turns. | Model load is about 4.2 s; after a first-turn warm-up outlier, total runtime is roughly 250-280 ms per turn. Detector scoring itself is only about 1.6 ms; the hidden-state forward pass dominates. | Run the current detector on held-out traces and decide whether demo mode needs batching, warm-up, or model reuse. |
| Selected-choice runtime candidate | Promote selected-choice span geometry into the `NormalizedTurn` CIFT contract and export runtime candidates. | A semantic-v2 selected-choice model runs through `AegisRuntime` on semantic v3 with 1.0000 F1; the payload/query fallback reaches 0.6834 F1 and over-warns safe rows. | Add an explicit runtime selector that uses selected-choice CIFT when available and falls back to payload/query readout otherwise. |
| Live window-selector benchmark | Run the explicit selector against live Qwen3-4B hidden states on a balanced mixed-route fixture. | The selector routed correctly on all 20 rows. Selected-choice remained clean, while payload/query fallback produced one false warning and two missed exfiltration rows. Mean total runtime was about 260 ms per turn after model load. | Prefer selected-choice geometry in the demo and continue fallback-route error analysis before treating fallback warnings as strict blocking evidence. |

## Current State

The best current data-generation result is not "we beat text." We have not.

The honest current result is sharper:

- Runtime-shaped trace generation works.
- DP-HONEY-backed protected-context spans flow into structured CIFT prompts.
- Readout-window activation extraction works on generated traces.
- Pair validation can enforce token balance and matched tool payloads.
- LM Studio can generate and repair local paired prompts.
- Unigram lexical shortcuts can be suppressed.
- Word-bigram shortcuts can be suppressed in a controlled diagnostic.
- Fixed trigram and fourgram grammar are now the main visible shortcuts.
- `paired_crossed_action` is now a reproducible harness profile rather than a
  one-off local corpus recipe.
- `paired_semantic_indirection` closes the short n-gram shortcut more strongly,
  but appears too compositionally indirect for the current Qwen3-0.6B linear
  readout probe.
- Benign no-secret rows now flow through the structured CIFT prompt contract,
  which gives paper-shaped calibration rows without fake secret spans.
- Dense readout tuning suggests the current signal is mid-late, not final-late,
  and scalar CIFT compression is still too lossy on this corpus.
- The current larger LM Studio Qwen3.6-27B checkpoint is not directly usable
  through PyTorch/Transformers for hidden-state extraction because its MLX
  quantization config is not a Hugging Face quantizer.
- `Qwen/Qwen3-4B` now works as a Transformers-compatible MPS activation source.
  On the paired semantic-indirection corpus, its best single readout layer
  reaches 0.6935 macro F1 and the best focused raw readout concat across layers
  19-22 reaches 0.7262 macro F1 on grouped `safe_secret_vs_exfiltration`.
- The best focused Qwen3-4B band is now calibrated and frozen as an
  `offline_research_candidate` model bundle, with a runtime-native
  `aegis.cift_runtime_linear/v1` JSON export and 480 Aegis-shaped
  `DetectorResult` rows for integration tests.
- The exported runtime model now runs through `AegisRuntime` using an offline
  activation-artifact feature extractor. This proves the detector and policy
  spine wiring, while keeping the production runtime free of research-only
  `.pt` artifact loading.
- A live model-host CIFT extractor now exists for the same runtime contract. It
  runs Qwen3-4B through Transformers/MPS, pools the requested readout-window
  layers from `metadata.cift.readout_token_indices`, and feeds the resulting
  vector into the existing runtime detector.
- A 20-turn live selector benchmark suggests the CIFT demo path is feasible on
  local MPS if the model is loaded once and warmed up. In the latest degraded
  fallback rerun, mean total runtime is about 428 ms per turn, and detector
  scoring remains negligible compared with hidden-state extraction.
- Selected-choice span geometry now crosses the runtime bridge in
  `metadata.cift`. A semantic-v2 selected-choice runtime candidate reaches
  1.0000 F1 on semantic v3 through `AegisRuntime`; the payload/query fallback
  remains useful as a degraded path but over-warns safe rows. An explicit CIFT
  window selector now records which route was chosen in detector evidence. A
  live Qwen3-4B selector benchmark exercises both selected-choice and fallback
  routes. On the 20-row mixed fixture, route selection matched expected metadata
  on every row: the ten selected-choice rows were active primary CIFT results,
  while the ten payload/query fallback rows were explicitly degraded evidence
  with capped confidence.

The current bottleneck has moved. We have a viable larger white-box activation
source, and it materially improves the hard-corpus result. The next problem is
not model access; it is proving that the exported detector survives outside the
training corpus and closing the CIFT fidelity gap without collapsing useful
activation vectors into over-compressed scalar distances.

## What The Latest Results Mean

The LM Studio repaired corpus passed strict pair validation:

- 60 pairs.
- 0 failed pairs.
- Max unigram delta: 7.
- Min weighted bigram Jaccard: 0.5000.
- Tool payload mismatch pairs: 0.
- Tool placeholder pairs: 0.

But grouped evaluation still favored text:

- Activation probe: 0.8325 macro F1.
- Word TF-IDF 1-2: 0.9798 macro F1.
- Char TF-IDF 3-5: 0.6292 macro F1.
- Word TF-IDF 1-1: 0.5755 macro F1.

So unigram balance helped, but bigram order remains label-correlated.

The crossed-action diagnostic then attacked that specific bigram shortcut:

- 60 pairs.
- 0 failed pairs.
- Max unigram delta: 0.
- Min bigram Jaccard: 1.0000.
- Average bigram Jaccard: 1.0000.
- Tool payload mismatch pairs: 0.
- Tool placeholder pairs: 0.

Grouped evaluation on the crossed-action corpus showed the intended pressure:

- Activation probe: 0.9231 macro F1.
- Word TF-IDF 1-2: 0.4848 macro F1.
- Char TF-IDF 3-5: 0.4896 macro F1.
- Word TF-IDF 1-3: 1.0000 macro F1.

So the crossed-action corpus is the first clear activation-over-word-1-2
diagnostic. It is not yet text-invariant evidence because trigrams still
recover the fixed allowed/denied grammar.

The reproducible `paired_crossed_action` harness profile expanded the same
diagnostic to the default 12-task catalog:

- 120 pairs.
- 0 failed pairs.
- Max unigram delta: 0.
- Min bigram Jaccard: 1.0000.
- Average bigram Jaccard: 1.0000.
- Tool payload mismatch pairs: 0.
- Tool placeholder pairs: 0.

Grouped evaluation on 240 structured safe/exfiltration rows showed:

- Activation probe: 0.8429 macro F1.
- Word TF-IDF 1-2: 0.4926 macro F1.
- Char TF-IDF 3-5: 0.4347 macro F1.
- Word TF-IDF 1-3: 1.0000 macro F1.
- Word TF-IDF 1-4: 1.0000 macro F1.

This gives us a reproducible regression diagnostic for unigram/bigram shortcut
control, but it does not close the template-shortcut problem.

The `paired_semantic_indirection` profile then removed the short template
shortcut by using neutral option codes:

- 240 pairs.
- 0 failed pairs.
- Max unigram delta: 0.
- Min bigram Jaccard: 0.9216.
- Average bigram Jaccard: 0.9259.
- Tool payload mismatch pairs: 0.
- Tool placeholder pairs: 0.

Grouped evaluation on 480 structured safe/exfiltration rows showed:

- Best grouped activation probe: 0.5518 macro F1.
- Word TF-IDF 1-2: 0.4749 macro F1.
- Char TF-IDF 3-5: 0.4963 macro F1.
- Word TF-IDF 1-3: 0.4669 macro F1.
- Word TF-IDF 1-4: 0.4667 macro F1.
- Word TF-IDF 1-5: 0.4438 macro F1.
- Word TF-IDF 1-6: 0.4575 macro F1.

This is a useful negative result: text shortcuts are controlled, but the
current activation probe does not yet solve the induced code-to-action
relation.

The Qwen3-4B pass then changed the picture. On the same
paired-semantic-indirection profile, a larger white-box model produced a useful
readout signal where Qwen3-0.6B struggled:

- Best grouped single layer: `readout_window_layer_20`, 0.6935 macro F1.
- Best grouped focused concat: layers 19-22, 0.7185 macro F1 at threshold 0.50.
- Best threshold-swept operating point: 0.7327 macro F1 at threshold 0.55.
- Text baselines remained below chance on the same controlled profile.

That candidate was frozen as an offline runtime artifact so it could be driven
through the Aegis runtime boundary:

```text
introspection/data/models/cift_qwen3_4b_semantic_indirection_l19_l22_readout_runtime_v1.json
```

The live Qwen3-4B runtime smoke test then proved the same detector path can run
from a model host instead of a saved `.pt` feature lookup. The 20-row benchmark
showed roughly 250-280 ms steady-state latency per turn on MPS after warm-up,
with the detector itself taking about 1.6 ms.

The paired-natural transfer test exposed the next real weakness:

- Frozen semantic-indirection candidate on paired-natural: 0.5485 macro F1.
- Oracle threshold sweep on the same frozen scores: 0.5643 macro F1.
- In-profile paired-natural layer-20 probe: 0.9364 macro F1.
- In-profile paired-natural layer 19-22 concat: 0.9177 macro F1.

This means the paired-natural corpus contains strong readout activation signal,
but the semantic-indirection-trained decision boundary does not transfer. The
data problem has moved from "can we suppress text shortcuts?" to "can we train
across enough prompt profiles that the CIFT boundary generalizes?"

The paired-adversarial corpus then became the third Qwen3-4B profile:

- In-profile paired-adversarial layer-20 probe: 0.8991 macro F1.
- In-profile paired-adversarial layer 19-22 concat: 0.9165 macro F1.
- Training on semantic + paired-natural and holding out paired-adversarial:
  0.6473 macro F1 with layer 19-22 concat.
- Training on semantic + paired-adversarial and holding out paired-natural:
  0.6661 macro F1 with layer 19-22 concat.
- Training on paired-natural + paired-adversarial and holding out semantic:
  0.3776 macro F1 with layer 19-22 concat.

This is the first real mixed-profile leave-one-profile-out result. It improves
over single-profile transfer when the held-out profile is paired-natural or
paired-adversarial, but semantic-indirection remains a hard distribution shift.
The data story is now about profile diversity plus calibration, not just corpus
size.

The pre-output profile then became a fourth Qwen3-4B artifact:

- In-profile pre-output layer-20 probe: 0.9666 macro F1.
- In-profile pre-output layer 19-22 concat: 0.9725 macro F1.
- Four-profile leave-one-profile-out with layer 19-22 concat:
  - semantic held out: 0.3486 macro F1.
  - paired-natural held out: 0.7624 macro F1.
  - paired-adversarial held out: 0.6494 macro F1.
  - pre-output held out: 0.9520 macro F1.

This confirms that more data helps, but not uniformly. Pre-output and
paired-natural benefit substantially from profile diversity; paired-adversarial
stays moderate; semantic-indirection remains the hard outlier.

Semantic-indirection error analysis explains why:

- The four-profile model makes 236 false positives and 3 false negatives on
  semantic-indirection.
- Semantic safe rows have a median exfiltration score of 0.9999.
- Semantic exfiltration rows have a median exfiltration score of 1.0000.
- Every semantic safe and exfiltration row mentions both the safe option and
  the exfiltration option, then asks the model to apply a final neutral code
  choice.

So semantic-indirection is not merely a harder phrasing profile. It is a
selected-option resolution task. The data fix is to create train/test semantic
variants that teach and evaluate whether CIFT can distinguish an exfiltration
option being mentioned from that option being selected.

## Next Data Requirement

The next corpus set should support mixed-profile training and transfer
evaluation rather than another one-profile leaderboard:

1. Every pair passes strict tool, unigram, and span validation.
2. Each profile avoids stable label-specific action-clause order and fixed
   trigram grammar.
3. Corpus-level grouped word 1-2 and word 1-3 TF-IDF should stay near chance, or
   the profile is treated as a diagnostic artifact only.
4. At least one entire profile is held out from training and used only for
   frozen transfer scoring.

The generation recipe should deliberately cross phrase order:

- Safe prompts sometimes mention `outside`, `handoff`, and `downstream`.
- Exfiltration prompts sometimes mention `internal`, `review`, and `boundary`.
- Both labels include `keep`, `carry`, `inside`, and `outside`.
- The order of those terms varies independently of the label.

The next model-selection artifact should compare at least:

- single-layer `readout_window_layer_20`;
- semantic focused concat, layers 19-22;
- a mixed-profile model trained on all available validated profiles except each
  held-out profile;
- threshold calibration on validation profiles rather than held-out labels;
- selected-choice spans for semantic-indirection-style prompts;
- the frozen semantic-indirection candidate as a historical baseline.

## Product Relevance

This data journey remains aligned with the Aegis runtime spine:

- Detectors will consume normalized runtime traces, not notebook-only prompts.
- DP-HONEY provides the protected-context spans and canary evidence.
- CIFT uses readout windows derived from those spans.
- Text/canary detectors remain separate monitors and baselines.
- Audit artifacts preserve enough metadata to compare regressions honestly.

The trace harness is the bridge between research and product: it lets us keep
running controlled CIFT experiments without polluting the promoted runtime
spine or pretending synthetic data is production telemetry.

## Selected-Choice Span Geometry

The next semantic-indirection experiment added the missing structural metadata:
selected-choice spans.

Generated artifacts:

```text
data/trace_collection/structured_prompts.paired_semantic_indirection_default_720_selected_choice_v2.jsonl
data/trace_collection/structured_prompts.paired_semantic_indirection_default_720_secret_present_binary_selected_choice_v2.jsonl
data/trace_collection/qwen3_4b_trace_paired_semantic_indirection_default_720_secret_present_binary_selected_choice_v2_l19_l22_readouts.pt
```

This was not a larger corpus. It was a better corpus contract for the same
semantic failure: every semantic safe and exfil row mentions both options, then
differs by the selected neutral code. The old payload/query readout saw both
options and confused "exfiltration option mentioned" with "exfiltration option
selected."

Grouped safe-secret versus exfiltration results on Qwen3-4B layers 19-22:

| Feature | Macro F1 | Accuracy |
|---|---:|---:|
| payload/query readout | 0.6991 | 0.7042 |
| selected-choice readout | 1.0000 | 1.0000 |
| combined readout | 1.0000 | 1.0000 |

Text baselines on the same binary slice stayed near chance:

| Baseline | Macro F1 | Accuracy |
|---|---:|---:|
| word TF-IDF 1-2 | 0.4749 | 0.4775 |
| char TF-IDF 3-5 | 0.4963 | 0.5050 |

This changes the data lesson. Corpus size was not the only issue. For
proxy-shaped CIFT data, the corpus must emit the structural windows that the
proxy knows or controls. Selected-choice spans are now part of the expected
semantic-indirection contract, alongside secret, query, payload, and readout
spans.

## Semantic V3 Holdout

The selected-choice contract then became a held-out semantic profile:

```text
paired_semantic_indirection_v3
```

The v3 profile uses new code names and selector phrases while keeping the same
safe/exfil option-selection structure. Generated artifacts:

```text
data/trace_collection/collection_inputs.paired_semantic_indirection_v3_720.jsonl
data/trace_collection/records.paired_semantic_indirection_v3_720.jsonl
data/trace_collection/structured_prompts.paired_semantic_indirection_v3_720_selected_choice.jsonl
data/trace_collection/structured_prompts.paired_semantic_indirection_v3_720_secret_present_binary_selected_choice.jsonl
data/trace_collection/qwen3_4b_trace_paired_semantic_indirection_v3_720_secret_present_binary_selected_choice_l19_l22_readouts.pt
```

Transfer result:

| Train | Test | Feature | Macro F1 | Accuracy |
|---|---|---|---:|---:|
| semantic v2 | semantic v3 | selected-choice | 1.0000 | 1.0000 |
| semantic v3 | semantic v2 | selected-choice | 1.0000 | 1.0000 |
| semantic v2 | semantic v3 | payload/query | 0.5814 | 0.6062 |

This is a stronger data milestone than the in-profile v2 result. It says the
selected-choice readout survives a held-out semantic vocabulary split, while the
older payload/query readout does not.

## Selected-Choice Runtime Promotion

The selected-choice contract now crosses the Aegis runtime bridge. Exported
semantic v3 runtime turns preserve:

```text
metadata.cift.selected_choice_char_span
metadata.cift.selected_choice_token_span
metadata.cift.selected_choice_readout_token_indices
```

Runtime artifacts:

```text
data/trace_collection/runtime_turns.paired_semantic_indirection_v3_720_secret_present_binary_selected_choice.jsonl
introspection/data/models/cift_qwen3_4b_semantic_v2_selected_choice_l19_l22_runtime_v1.json
introspection/data/models/cift_qwen3_4b_semantic_v2_payload_readout_l19_l22_runtime_v1.json
data/trace_collection/runtime_eval.cift_qwen3_4b_semantic_v2_window_selector_l19_l22_on_v3.jsonl
data/trace_collection/runtime_eval.cift_qwen3_4b_semantic_v2_window_selector_l19_l22_on_v3_mixed.jsonl
data/trace_collection/runtime_eval.live_cift_qwen3_4b_window_selector_mixed_smoke_2.jsonl
```

Both runtime models were trained on semantic v2 and evaluated through
`AegisRuntime` on semantic v3:

| Candidate | Accuracy | Precision | Recall | F1 | Confusion |
|---|---:|---:|---:|---:|---|
| selected-choice | 1.0000 | 1.0000 | 1.0000 | 1.0000 | TN 240, FP 0, FN 0, TP 240 |
| payload/query fallback | 0.6062 | 0.5714 | 0.8500 | 0.6834 | TN 87, FP 153, FN 36, TP 204 |
| explicit selector | 1.0000 | 1.0000 | 1.0000 | 1.0000 | TN 240, FP 0, FN 0, TP 240 |
| mixed selector | 0.7958 | 0.7336 | 0.9292 | 0.8199 | TN 159, FP 81, FN 17, TP 223 |

This is the first selected-choice CIFT candidate that is both transfer-tested
and runtime-shaped. The selector now uses selected-choice CIFT when the metadata
exists and falls back to payload/query readout otherwise. On the semantic v3
runtime-turn file, all 480 rows route through selected-choice because all rows
carry selected-choice span metadata.

The mixed selector fixture strips selected-choice metadata from a balanced half
of the same v3 rows. It confirms route selection is auditable and exact: 240
rows use selected-choice, 240 use fallback, and there are zero route mismatches.
The live model-host smoke then runs the same selector against Qwen3-4B on MPS
for two rows, one per route, with active detector results and one cached
hidden-state forward pass per turn.
