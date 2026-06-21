# Project-Aligned Workflow Checkpoint

## Purpose

The introspection work is now explicitly scoped as the CIFT-like monitor inside
the larger Aegis/AIS system. The activation probe is a pre-output monitor
signal, not a standalone defense.

## System Components

| Component | Role | Current Status |
|---|---|---|
| DP-HONEY | Inject format-matched honeytokens before model access. | Not implemented here yet. |
| CIFT-like activation monitor | Detect hidden-state evidence of credential access before output. | Current focus. |
| Text leakage detector | Scan generated text for canary or secret leakage. | Not implemented here yet. |
| NIMBUS-like accumulator | Track cumulative leakage risk across turns. | Not implemented here yet. |

## Workflow Rule

Every future experiment should state which component it advances, what monitor
event it would emit, what evidence supports it, what residual or held-out check
guards against overfit, and how it composes with the eventual gateway.

## Current Implication

The combined activation feature remains a CIFT candidate. It should not be
promoted until the Hard V3 regression adjudication cases are reviewed and a
feature-selection rule is written.

After re-reading the paper method, the current probe should be described as
CIFT-like rather than full CIFT. The paper's CIFT monitor uses readout positions
after both secret context and query/payload, monitors the last-quarter layers,
and converts benign-calibrated per-layer deviations into a learned Causal Flow
Score. The current work uses final-token/readout-style activation features and
grouped probe evaluation, but has not yet implemented that calibrated CCI/CFS
path.

The next CIFT-aligned implementation step is therefore not only feature
promotion. It is deciding whether to:

1. Promote the current empirical probe as a temporary CIFT-like signal.
2. Build the paper-aligned calibrated readout-position monitor next.
3. Pause CIFT promotion and start DP-HONEY or text detector scaffolding so the
   full AIS pipeline shape is represented earlier.
