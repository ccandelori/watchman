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

The current bottleneck is data design, not extraction plumbing.

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

## Next Data Requirement

The next corpus should be generated with a stronger acceptance condition:

1. Pair passes strict tool and unigram validation.
2. Pair avoids stable label-specific action-clause order and fixed trigram
   grammar.
3. Corpus-level grouped word 1-2 and word 1-3 TF-IDF must fall below the activation probe,
   or the corpus is treated as a diagnostic artifact only.

The next generation recipe should deliberately cross phrase order:

- Safe prompts sometimes mention `outside`, `handoff`, and `downstream`.
- Exfiltration prompts sometimes mention `internal`, `review`, and `boundary`.
- Both labels include `keep`, `carry`, `inside`, and `outside`.
- The order of those terms varies independently of the label.

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
