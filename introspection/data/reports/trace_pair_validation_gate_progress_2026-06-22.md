# Trace Pair Validation Gate Progress

Date: 2026-06-22

## Purpose

The paired-adversarial corpus showed that token-level balancing is necessary
but insufficient. Word unigram TF-IDF dropped to chance, while word bigram
TF-IDF still classified the corpus perfectly. That means future CIFT data needs
a local quality gate before activation extraction.

This milestone adds a paired prompt validation gate to the runtime trace
collection harness.

## Added Contract

The validator consumes:

- Assignment packets.
- Collection input submissions.
- A maximum unigram delta threshold.
- A minimum weighted bigram Jaccard threshold.

It groups `secret_present_safe` and `exfiltration_intent` submissions by
participant, task, and variant, then emits `trace_pair_validation/v1` JSON.

Each pair report records:

- `unigram_delta`
- `bigram_jaccard`
- `tool_calls_match`
- `tool_call_placeholder_present`
- `passed`
- `failures`

## CLI

The new command is:

```bash
uv run aegis-trace-validate-pairs \
  --assignments data/trace_collection/assignments.jsonl \
  --inputs data/trace_collection/collection_inputs.paired_adversarial_720.jsonl \
  --maximum-unigram-delta 0 \
  --minimum-bigram-jaccard 0.1 \
  --output data/trace_collection/pair_validation.paired_adversarial_720.json
```

The command writes the report and exits non-zero if any pair fails.

## Interpretation

This does not replace text baselines. It is an earlier gate that prevents
obviously flawed paired corpora from reaching the expensive activation
extraction stage.

For deterministic adversarial seed data, strict unigram balance is appropriate.
For human or LLM paraphrase data, thresholds should be chosen deliberately and
reported with the corpus. Passing this validator only means the corpus satisfies
basic structural and lexical constraints; it does not prove CIFT-specific
signal.

## Next Step

Use this validator on the next human or LLM-paraphrased safe/exfiltration
collection before converting records into structured CIFT prompts. Then rerun
the text baselines and activation probes. A real CIFT claim still requires text
baselines to fall below the activation probe.
