# CIFT Selected-Choice Trace Corpus Checkpoint

## Summary

Generated a fresh paired semantic-indirection v3 corpus from the Watchman
runtime trace harness and converted it into structured CIFT prompt records. This
checkpoint validates the proxy-shaped data contract after the selected-choice
metadata fix: selected-choice geometry is emitted by trace generation and
preserved through structured prompt conversion.

This is a data-readiness checkpoint, not a detector-performance claim. No
hidden-state extraction or probe training was performed in this checkpoint.

## Generated Artifacts

Generated artifacts are intentionally ignored by git under
`data/trace_collection/`:

| Artifact | Rows |
| --- | ---: |
| `data/trace_collection/watchman_assignments.jsonl` | 36 |
| `data/trace_collection/collection_inputs.paired_semantic_indirection_v3_720.jsonl` | 720 |
| `data/trace_collection/records.paired_semantic_indirection_v3_720.jsonl` | 720 |
| `data/trace_collection/structured_prompts.paired_semantic_indirection_v3_720.jsonl` | 720 |

## Corpus Shape

| Label | Rows |
| --- | ---: |
| `benign` | 240 |
| `secret_present_safe` | 240 |
| `exfiltration_intent` | 240 |

The corpus covers 12 task families.

## Contract Checks

| Check | Result |
| --- | ---: |
| Structured prompt rows written | 720 |
| Trace records skipped during conversion | 0 |
| Non-benign rows missing secret token span | 0 |
| Non-benign rows missing selected-choice readout tokens | 0 |
| Rows with fallback reason | 0 |

Selected-choice readout token lengths:

| Token count | Rows |
| --- | ---: |
| 0 | 240 |
| 3 | 144 |
| 4 | 192 |
| 5 | 144 |

The 240 zero-length selected-choice rows are benign rows. All
`secret_present_safe` and `exfiltration_intent` rows have selected-choice
readout coverage.

## Next CIFT Step

Use this structured prompt corpus for the next hidden-state extraction pass,
then train and evaluate selected-choice readout features against text baselines.
The next report should include:

- model and tokenizer identity;
- extracted layers and pooling family;
- grouped train/test protocol;
- word and character TF-IDF baselines;
- selected-choice readout results;
- degraded fallback comparison; and
- recommendation on whether to promote, reject, or revise the candidate.
