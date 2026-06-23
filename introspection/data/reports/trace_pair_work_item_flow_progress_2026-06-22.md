# Trace Pair Work-Item Flow Progress

Date: 2026-06-22

## Purpose

The CIFT probe results degraded as the synthetic corpora became less
shortcut-friendly. That is useful signal: the monitor needs better paired data,
not just more deterministic variants. The new paired paraphrase work-item flow
creates a controlled collection path for human or LLM-written safe/exfiltration
pairs while preserving the runtime-spine data contract.

## What Changed

The runtime trace harness now emits `trace_pair_work_item/v1` rows. Each row
links a `secret_present_safe` assignment and an `exfiltration_intent`
assignment for the same participant and task, shares one matched tool payload,
and gives the writer explicit goals and constraints for both prompts.

Completed rows use `trace_pair_completion/v1`. The compiler turns each
completion into the ordinary `TraceCollectionSubmission` rows consumed by the
existing trace builder, DP-HONEY injection, structured prompt converter, and
CIFT extraction path.

## Why It Matters

This creates a better data path for testing pre-output intent:

- safe and exfiltration examples share task, participant, variant, and tool
  payload shape;
- prompts can be natural rather than deterministic control strings;
- the generated collection inputs still flow through DP-HONEY-backed
  `NormalizedTurn` records;
- paired validation remains available before CIFT feature extraction.

The flow does not replace synthetic seed profiles. It gives the project a
cleaner next corpus for experiments where lexical shortcuts are the primary
risk to honest CIFT evaluation.

## Current Boundary

This work does not train a new CIFT model by itself. It creates the collection
mechanism needed to gather better paired prompts, compile them into trace
records, and run the established extraction and probe-training pipeline.
