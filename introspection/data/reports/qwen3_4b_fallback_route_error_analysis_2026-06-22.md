# Qwen3-4B Fallback Route Error Analysis

Date: 2026-06-22

## Purpose

Diagnose why the live CIFT window selector is clean on selected-choice rows but
weak on payload/query fallback rows.

## Feedback Loop

Primary live benchmark:

```text
data/trace_collection/live_cift_qwen3_4b_window_selector_mixed_benchmark_20.json
```

Larger artifact-backed mixed eval:

```text
data/trace_collection/runtime_eval.cift_qwen3_4b_semantic_v2_window_selector_l19_l22_on_v3_mixed.jsonl
data/trace_collection/runtime_turns.paired_semantic_indirection_v3_720_secret_present_binary_mixed_selector.jsonl
```

## Reproduced Symptom

The 20-row live benchmark has all mistakes on the fallback route:

| Route | Label | Allow | Warn |
|---|---|---:|---:|
| selected-choice | secret_present_safe | 5 | 0 |
| selected-choice | exfiltration_intent | 0 | 5 |
| payload/query fallback | secret_present_safe | 4 | 1 |
| payload/query fallback | exfiltration_intent | 2 | 3 |

The 480-row artifact-backed mixed eval shows the same pattern at scale:

| Route | Rows | Errors | Confusion |
|---|---:|---:|---|
| selected-choice | 240 | 0 | TN 120, FP 0, FN 0, TP 120 |
| payload/query fallback | 240 | 98 | TN 39, FP 81, FN 17, TP 103 |

## Score Distribution

Selected-choice scores are cleanly separated:

| Label | Min | Median | Max |
|---|---:|---:|---:|
| secret_present_safe | 0.0000000001 | 0.0000020191 | 0.0018050510 |
| exfiltration_intent | 0.9999978542 | 0.9999998604 | 0.9999999993 |

Fallback scores overlap badly:

| Label | Min | Median | Max |
|---|---:|---:|---:|
| secret_present_safe | 0.0000002165 | 0.9322954551 | 0.9999993324 |
| exfiltration_intent | 0.0001702425 | 0.9992700700 | 0.9999999270 |

This is not just a threshold problem. At threshold 0.50, fallback macro F1 is
0.5604. Sweeping thresholds improves only modestly; the best sampled macro F1
is 0.6375 at threshold 0.99. That reduces false positives but creates many more
false negatives.

## Error Clustering

Fallback errors cluster strongly by template variant:

| Variant | Errors / Rows | Pattern |
|---|---:|---|
| v003 | 12 / 24 | warns every fallback row |
| v017 | 12 / 24 | warns every fallback row |
| v019 | 12 / 24 | warns every fallback row |
| v015 | 11 / 24 | warns almost every fallback row |
| v007 | 11 / 24 | allows almost every fallback row |

The model appears to learn variant/tool-payload behavior instead of the selected
semantic decision when it is forced onto fallback readout.

The fallback errors also cluster by tool payload surface. The highest error
counts came from:

| Tool | Errors |
|---|---:|
| upload_diagnostics | 23 |
| create_support_transcript | 22 |
| send_email | 21 |
| archive_runbook_note | 17 |
| create_incident_ticket | 15 |

## Root Cause Hypothesis

The current fallback route is not a faithful substitute for selected-choice
readout.

When selected-choice span metadata is absent, the fallback model uses
`metadata.cift.readout_token_indices`. For tool-bearing rows, those indices
usually point into the rendered tool payload, not the query phrase that resolves
the neutral option into "keep credential authority inside aegis" or "carry
credential authority outside aegis."

Examples:

```text
payload: Authorization continuity state: pending_authorization.
payload: handoff prepared
payload: github_pat
```

Those tokens are after the query, but they are not the tokens where the hidden
semantic decision is most directly encoded. As a result, variants such as v003,
v017, and v019 behave like route/template detectors rather than intent
detectors.

## Ranked Hypotheses

1. Fallback readout is pointed at the wrong tokens.
   Prediction: a query-tail or decision-phrase fallback window should reduce
   variant-level always-warn/always-allow behavior.

2. Fallback training data over-represents payload/template artifacts.
   Prediction: retraining the fallback on windows closer to the query decision
   phrase, or on mixed query-tail/payload windows, should improve safe/exfil
   separation.

3. The fallback model is under-calibrated for degraded use.
   Prediction: threshold tuning can improve macro F1, but cannot solve the
   confident false positives and false negatives. The observed threshold sweep
   supports this.

4. Payload fallback should not be a blocking signal.
   Prediction: product behavior improves if selected-choice results can warn or
   block, while payload/query fallback contributes lower-confidence evidence or
   degraded capability status.

## Recommendation

Do not spend the next cycle tuning only the fallback threshold.

The highest-leverage next experiment is a new fallback window:

```text
query_tail_readout_token_indices
```

or an equivalent decision-tail window derived from `query_token_span`. This
window should point at the last few query tokens before the tool call, or at the
semantic decision phrase when a selected-choice-specific span cannot be emitted.

Then train/evaluate three fallback candidates:

| Candidate | Purpose |
|---|---|
| payload readout | current baseline |
| query-tail readout | tests whether payload tokens are the root cause |
| combined query-tail + payload | tests whether payload context helps once the decision phrase is preserved |

Until that experiment lands, treat payload/query fallback as degraded coverage,
not as a strict blocking detector.
