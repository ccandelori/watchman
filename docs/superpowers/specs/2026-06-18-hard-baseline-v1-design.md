# Hard Baseline V1 Design

## Purpose

Create `introspection/data/prompts_hard.jsonl` as a harder, balanced successor
to the current baseline prompt dataset. The goal is to test whether the current
activation signal, especially `mean_pool_layer_18`, still helps when surface
text cues become less generous.

This is an evaluation dataset design, not a production detector design. The
result should be treated as an exploratory checkpoint.

## Background

The current baseline dataset has 90 hand-authored examples:

| Label | Count |
|---|---:|
| `benign` | 30 |
| `secret_present_safe` | 30 |
| `exfiltration_intent` | 30 |

Grouped binary evaluation on the baseline produced the strongest current result
for the `safe_secret_vs_exfiltration` task:

| Method | Grouped Macro F1 | Grouped Accuracy |
|---|---:|---:|
| `activation_probe` using `mean_pool_layer_18` | 0.8620 | 0.8667 |
| `char_tfidf` | 0.7649 | 0.7833 |
| `word_tfidf` | 0.7054 | 0.7167 |

That result is promising, but the dataset is still small and likely contains
construction artifacts. Hard Baseline V1 should pressure-test the signal with
more contrastive examples while preserving direct comparability.

## Dataset Scope

Create one new dataset:

```text
introspection/data/prompts_hard.jsonl
```

Do not modify `introspection/data/prompts.jsonl`.

Hard Baseline V1 uses the same schema as the baseline:

```json
{"id":"hard_safe_001","label":"secret_present_safe","family":"safe_log_redaction","text":"...","tags":["secret","logging","hard"]}
```

Required fields:

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Stable unique example identifier. |
| `label` | string | One of `benign`, `secret_present_safe`, `exfiltration_intent`. |
| `family` | string | Prompt-pattern group used for grouped cross-validation. |
| `text` | string | Prompt text sent through the model. |
| `tags` | string array | Lightweight descriptors for inspection and future filtering. |

Dataset size:

| Label | Count | Families | Examples per Family |
|---|---:|---:|---:|
| `benign` | 30 | 10 | 3 |
| `secret_present_safe` | 30 | 10 | 3 |
| `exfiltration_intent` | 30 | 10 | 3 |

## Hardness Criteria

Hardness should come from contrast, not from longer prompts or more dramatic
language.

The dataset should follow these rules:

- Safe and exfiltration examples should often reuse similar secret formats.
- Safe prompts may mention risky channels, but only to reject, redact, mask, or
  constrain them.
- Exfiltration prompts should often sound operational, administrative, or
  benign, avoiding obvious words such as "leak" when possible.
- Benign prompts should include Aegis-relevant vocabulary, such as proxy,
  tool calls, metadata, logs, credentials, policies, allowlists, handles, and
  detectors, but without actual secret-like values.
- Families should be semantically parallel where possible, so grouped splits
  test transfer across related but held-out prompt patterns.
- The dataset should avoid near-duplicate phrasing inside a family. The family
  relationship should be conceptual, not just template substitution.

## Proposed Families

### Benign Families

Benign examples should contain security and agent-infrastructure vocabulary
without credential-like values or exfiltration instructions.

| Family | Intent |
|---|---|
| `hard_benign_proxy_policy` | Discuss proxy enforcement, allowlists, and routing policy. |
| `hard_benign_tool_observability` | Discuss tool-call logging and structured observability. |
| `hard_benign_metadata_design` | Discuss metadata schemas without secret material. |
| `hard_benign_redaction_policy` | Discuss redaction policy abstractly without concrete secrets. |
| `hard_benign_handle_architecture` | Discuss opaque handles and credential brokers conceptually. |
| `hard_benign_detector_eval` | Discuss detector metrics, thresholds, and false positives. |
| `hard_benign_incident_review` | Discuss incident review workflows without sensitive values. |
| `hard_benign_encoding_concepts` | Discuss encodings and transformations without secret values. |
| `hard_benign_logging_design` | Discuss safe logging architecture without credential content. |
| `hard_benign_agent_context` | Discuss agent context boundaries and tool permissions. |

### Safe-Secret Families

Safe-secret examples include secret-like material but ask for protection,
classification, redaction, masking, or refusal.

| Family | Intent |
|---|---|
| `hard_safe_log_redaction` | Keep secrets out of logs or replace them with placeholders. |
| `hard_safe_metadata_filtering` | Prevent secret-like values from entering metadata fields. |
| `hard_safe_tool_argument_review` | Review tool-call arguments and remove sensitive values. |
| `hard_safe_encoding_refusal` | Refuse encoding, transforming, or copying secret values. |
| `hard_safe_handle_nonresolution` | Treat handles as opaque and do not resolve them. |
| `hard_safe_summary_without_secret` | Summarize security issues without quoting secrets. |
| `hard_safe_partial_disclosure_refusal` | Refuse partial, chunked, or last-four disclosure. |
| `hard_safe_policy_classification` | Classify risky content without reproducing values. |
| `hard_safe_broker_boundary` | Explain that only a broker may access underlying credentials. |
| `hard_safe_output_contract` | Return structured safe output that excludes the secret. |

### Exfiltration Families

Exfiltration examples request unsafe handling while often using operational or
administrative framing.

| Family | Intent |
|---|---|
| `hard_exfil_log_inclusion` | Put secret-like values into logs, audit records, or diagnostics. |
| `hard_exfil_metadata_forwarding` | Place secret-like values into metadata or provider context. |
| `hard_exfil_tool_argument_injection` | Insert secrets into tool-call arguments or payloads. |
| `hard_exfil_encoding_request` | Encode, transform, or preserve secrets in another representation. |
| `hard_exfil_handle_resolution` | Resolve opaque handles and reveal or transmit underlying values. |
| `hard_exfil_summary_with_secret` | Summarize while preserving exact secret material. |
| `hard_exfil_partial_disclosure` | Reveal secrets in chunks, suffixes, or staged disclosures. |
| `hard_exfil_policy_override` | Override redaction or classification boundaries. |
| `hard_exfil_broker_impersonation` | Pretend to act as the broker and access credentials. |
| `hard_exfil_output_contract_abuse` | Abuse structured output fields to carry secret material. |

## Expected Outputs

Create hard-dataset activation and report artifacts without overwriting baseline
artifacts:

```text
introspection/data/activations/qwen3_0_6b_hard_all_layers.pt
introspection/data/reports/binary_tasks_hard.json
introspection/data/reports/binary_tasks_hard_summary.md
introspection/data/reports/binary_tasks_hard_grouped.json
introspection/data/reports/binary_tasks_hard_grouped_summary.md
```

Add a narrative checkpoint:

```text
introspection/data/reports/hard_baseline_probe_progress_2026-06-18.md
```

Update the living README:

```text
introspection/README.md
```

Update lineage:

```text
introspection/data/lineage.json
```

The lineage manifest should register:

- `hard_prompts_v1`
- `qwen3_0_6b_hard_all_layers_v1`
- `hard_binary_random_v1`
- `hard_binary_grouped_v1`

## Evaluation

Use the existing extraction and evaluation scripts with explicit paths.

Extract all-layer features:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/extract_activations.py \
  --prompts introspection/data/prompts_hard.jsonl \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28 \
  --pooling final_token,mean_pool \
  --output introspection/data/activations/qwen3_0_6b_hard_all_layers.pt
```

Run random binary evaluation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/train_binary_tasks.py \
  --artifact introspection/data/activations/qwen3_0_6b_hard_all_layers.pt \
  --output-json introspection/data/reports/binary_tasks_hard.json \
  --output-md introspection/data/reports/binary_tasks_hard_summary.md
```

Run grouped binary evaluation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/train_grouped_binary_tasks.py \
  --artifact introspection/data/activations/qwen3_0_6b_hard_all_layers.pt \
  --output-json introspection/data/reports/binary_tasks_hard_grouped.json \
  --output-md introspection/data/reports/binary_tasks_hard_grouped_summary.md
```

Validate lineage:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/validate_lineage.py
```

Run tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python -m unittest discover -s introspection/tests
```

## Success Criteria

The implementation is complete when:

- `prompts_hard.jsonl` has exactly 90 valid examples.
- Each label has exactly 30 examples.
- Each label has exactly 10 families.
- Each family has exactly 3 examples.
- Existing baseline files are unchanged.
- Hard activation and binary report artifacts are written to new paths.
- Hard random and grouped reports are registered in lineage.
- Lineage validation passes.
- The full test suite passes.
- The README describes both baseline and hard-v1 checkpoints.

Metric success is intentionally not defined as "the probe must win." The useful
outcome is an honest pressure test. A drop in activation-probe performance is a
valid result if the dataset is harder and lineage is preserved.

## Larger Hard Baseline Plan

Hard Baseline V2 should not be created until V1 results are inspected.

The intended V2 shape is:

```text
150 examples total
50 examples per label
at least 15 families per label
```

V2 should be informed by V1 failures. If V1 shows weakness on metadata
forwarding, log inclusion, handle resolution, or output-contract abuse, V2
should expand those families instead of adding generic examples.

V2 may introduce structured tool-call-style examples, but that should be a
deliberate decision. If tool-call examples dominate V2, create a separate
dataset such as `prompts_tool_calls.jsonl` to keep the experimental story
interpretable.

## Non-Goals

- Do not replace `introspection/data/prompts.jsonl`.
- Do not overwrite baseline activation artifacts or reports.
- Do not change model architecture, probe type, or selected feature in this
  slice.
- Do not add a new dependency.
- Do not claim production readiness from V1 results.

## Open Implementation Notes

The current scripts already support explicit prompt, artifact, JSON report, and
Markdown report paths. The first implementation plan should mostly create data,
run existing scripts, update lineage, update README, and write the checkpoint
report.
