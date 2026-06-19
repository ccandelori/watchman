# Aegis Project Plan

**Project:** Aegis - Runtime Credential Defense for LLM Agents  
**Planning Basis:** Gauntlet Capstone Brief, prior Aegis proposal, and AIS credential-exfiltration paper  
**Demo Date:** June 29, 2026  
**Team Assumption:** 3 active builders; fourth teammate is optional support  
**Goal:** Build and demo a working proxy-mode security gateway that detects and mitigates credential exfiltration attempts from LLM agents, with tool-call argument scanning as the main differentiator.

## 1. Capstone Fit

The Gauntlet brief asks for a technically hard project whose ambition matters more than polish. Aegis fits Direction A: combine ML with LLM applications.

LLM agents supply the real-world action layer: they read untrusted content, form responses, and call tools. Classical ML and systems-style detectors supply the safety layer: scoring, canary detection, cumulative leakage accounting, and policy enforcement. The project is ambitious because it attacks a structural failure mode in agent systems, not a cosmetic UX problem.

The live demo should make that ambition visible: a baseline agent leaks or attempts to leak credentials, while an Aegis-protected agent blocks, warns, sanitizes, or escalates with evidence.

## 2. Research Grounding

The core research reference is "Caught in the Act(ivation): Toward Pre-Output and Multi-Turn Detection of Credential Exfiltration by LLM Agents" by Chauhan and Revankar (2026). The paper proposes AIS, a research prototype combining:

1. **CIFT:** activation-based pre-output detection for open-weight models with white-box access.
2. **DP-HONEY:** calibrated honeytoken generation and detection.
3. **NIMBUS:** cumulative leakage scoring across conversation turns.

The paper is explicit about its limits:

1. AIS is not deployment-ready.
2. CIFT requires white-box activation access and does not directly apply to closed cloud APIs.
3. NIMBUS is a learned leakage signal, not a certified information-flow bound.
4. Structured tool-call arguments are out of scope and described as a severe blind spot.

Aegis uses those limitations as project direction. The capstone build should productize the AIS pattern into a gateway and specifically extend protection to structured tool-call arguments.

## 3. System Architecture

```text
Agent Runtime
    |
    v
Aegis Gateway (FastAPI proxy / sidecar)
    |
    +-- Request Normalization
    |     - messages
    |     - tool calls
    |     - tool arguments
    |     - session and trace IDs
    |
    +-- Inspect
    |     - tool-call argument scanner
    |     - canary detector
    |     - credential-pattern detector
    |     - cumulative leakage features
    |
    +-- Score
    |     - per-detector risk scores
    |     - combined risk score
    |     - session leakage budget
    |
    +-- Enforce
    |     - allow
    |     - warn
    |     - sanitize
    |     - block
    |     - escalate
    |
    +-- Audit + Dashboard
          - JSON artifacts
          - Streamlit live view
          - evaluation reports
```

External surfaces:

1. **LLM provider or mock provider:** OpenAI-compatible route for the demo.
2. **Tool runtime:** Supported demo tools include `send_email`, `http_request`, and `query_database`.
3. **Credential broker:** In-memory broker maps opaque handles to real or mock secrets.
4. **Dashboard:** Streamlit app reads audit artifacts and scenario results.

## 4. Component Responsibilities

| Component | Purpose | Owner | MVP Acceptance Criteria |
| --- | --- | --- | --- |
| Gateway proxy | Receives, logs, forwards, and returns model/tool traffic | P1 | Can run a pass-through OpenAI-compatible request and produce audit logs. |
| Normalization layer | Converts provider/tool shapes into internal typed events | P1 | Produces consistent events for messages, responses, and supported tool calls. |
| Detector contract | Shared interface for all detectors | P1 | Every detector returns score, confidence, action recommendation, evidence, and latency. |
| Tool-call argument scanner | Detects suspicious credential movement through structured arguments | P1 | Blocks/escalates scripted leaks in `send_email`, `http_request`, and `query_database`. |
| Evaluation harness | Runs benign and attack scenarios repeatedly | P2 | Runs encoded, multi-turn, tool-call, canary, and benign cases from files. |
| Behavioral/provenance risk signals | Cloud-compatible signals inspired by CIFT constraints | P2 | Produces useful risk evidence without requiring model activations. |
| Canary service | Generates, registers, injects, and detects honeytokens | P3 | Detects registered canaries in model output and tool arguments. |
| Leakage ledger | Tracks cumulative session risk | P3 | Warns/blocks when scripted multi-turn leak crosses threshold. |
| Policy engine | Maps detector output to final action | P3 | YAML policy controls allow/warn/sanitize/block/escalate thresholds. |
| Audit and dashboard | Shows live decisions and evidence | P3 | Dashboard displays scenario, action, risk, detectors, and latency. |

## 5. Team Ownership

### P1 - Gateway and Enforcement Lead

Primary ownership:

1. FastAPI gateway.
2. Provider-compatible proxy route.
3. Normalized request/response model.
4. Inspect -> Score -> Enforce orchestration.
5. Tool-call argument scanner.

Secondary support:

1. Integration with policy engine.
2. End-to-end demo wiring.

### P2 - Evaluation and Signal Lead

Primary ownership:

1. Evaluation harness.
2. Attack and benign scenario library.
3. Baseline-agent comparison.
4. Behavioral/provenance scoring features.
5. Quantitative demo report.

Secondary support:

1. Detector calibration.
2. Regression promotion for discovered failures.

### P3 - Canary, Policy, and Observability Lead

Primary ownership:

1. Credential broker.
2. DP-HONEY-inspired canary registry.
3. NIMBUS-inspired leakage ledger.
4. YAML policy engine.
5. Audit artifact schema.
6. Streamlit dashboard.

Secondary support:

1. Demo narration and visual evidence.
2. Benign credential-use scenarios.

### Optional P4 - Demo and Integration Support

Only assign if the fourth teammate is reliably active:

1. Demo scripting.
2. Documentation.
3. Dashboard polish.
4. End-to-end integration testing.

Do not put critical path work on P4.

## 6. Milestones

| Milestone | Target | Definition of Done |
| --- | --- | --- |
| M1 - Observation proxy | Day 2 | Gateway receives a request, logs normalized data, forwards or mocks upstream, and returns a response. |
| M2 - Detector pipeline | Day 4 | At least two detectors run through a shared interface and produce policy decisions. |
| M3 - Tool-call defense | Day 5 | Supported tool-call exfiltration attempts are blocked or escalated with evidence. |
| M4 - Canary + leakage accounting | Day 7 | Canary hits and multi-turn budget thresholds appear in audit logs and dashboard. |
| M5 - Evaluation harness | Day 8 | Benign and attack scenarios run from repeatable files and produce summary metrics. |
| M6 - Integrated demo | Day 10 | Baseline vs protected flows run end-to-end with dashboard evidence. |
| M7 - Final polish | Day 11 | Demo script, fallback path, metrics table, and final narrative are ready. |

## 7. Day-by-Day Execution Plan

### Day 1 - Architecture Lock and Skeleton

Deliverables:

1. Repository structure for gateway, detectors, policy, storage, dashboard, and evals.
2. Shared data models for normalized turns, tool calls, detector results, and policy decisions.
3. One mock provider route for deterministic testing.
4. First five demo scenarios drafted: benign email, benign API call, encoded leak, multi-turn drip, tool-call exfiltration.

Acceptance criteria:

1. Team can run the project locally.
2. A single command starts the gateway.
3. A mock request produces a normalized audit artifact.

### Day 2 - Observation-Only Proxy

Deliverables:

1. OpenAI-compatible pass-through or mock-compatible chat route.
2. Structured JSON logging for request, response, session ID, trace ID, and latency.
3. Minimal CLI or script that sends one benign scenario through the proxy.

Acceptance criteria:

1. Proxy path works without detectors.
2. Logs are readable by the future dashboard.
3. Upstream failure produces a clear error instead of a silent pass.

### Day 3 - Detector Contract and Policy Skeleton

Deliverables:

1. Detector interface.
2. `DetectorResult` schema.
3. Policy decision schema.
4. YAML policy loader.
5. Static detectors for canary match and credential-shaped value match.

Acceptance criteria:

1. Each detector emits structured evidence.
2. Policy engine can map detector output to allow/warn/block/escalate.
3. Unit tests cover policy threshold behavior.

### Day 4 - Tool-Call Argument Scanner

Deliverables:

1. Supported schemas for `send_email`, `http_request`, and `query_database`.
2. Argument-level scanner for credential-shaped values, unknown recipients/URLs, and canary hits.
3. Provenance check for whether suspicious values appeared in trusted user/system context.
4. Evaluation cases for tool-call argument exfiltration.

Acceptance criteria:

1. A scripted `send_email` leak is blocked before dispatch.
2. A scripted `http_request` leak is blocked or escalated before dispatch.
3. A benign email/tool call is allowed.
4. Evidence includes tool name, argument name, risk reason, and action.

### Day 5 - Credential Broker and Canary Registry

Deliverables:

1. In-memory credential broker with opaque handles.
2. Canary registry keyed by session and service.
3. Format-matched honeytoken generator for at least two credential families.
4. Detector that scans output and tool arguments for registered canaries.

Acceptance criteria:

1. Model-visible context never needs raw real credentials in demo flows.
2. Canary appearance in output triggers non-allow policy action.
3. Canary appearance in tool arguments triggers non-allow policy action.

### Day 6 - Leakage Ledger

Deliverables:

1. Per-session cumulative leakage score.
2. Rule thresholds for warn, sanitize, block, and escalate.
3. Multi-turn scenario that stays below per-turn thresholds but crosses cumulative threshold.
4. Audit output showing per-turn and cumulative score.

Acceptance criteria:

1. Multi-turn drip attack triggers before final scripted leak completes.
2. Benign multi-turn scenario stays under blocking threshold.
3. Dashboard can display current budget state.

### Day 7 - Evaluation Harness

Deliverables:

1. Scenario file format.
2. Runner for benign and attack scenarios.
3. Metrics summary: detection count, false block count, warnings, average latency, detector hits.
4. Regression output for failed cases.

Acceptance criteria:

1. Harness runs without a live model by using scripted responses/tool calls.
2. Harness can optionally run against a real provider if configured.
3. Metrics are reproducible across runs.

### Day 8 - Dashboard

Deliverables:

1. Streamlit dashboard reading audit artifacts and evaluation summaries.
2. Recent decisions table.
3. Metrics strip: total cases, blocked, warned, false blocks, average latency, canary hits.
4. Scenario detail view with detector evidence.

Acceptance criteria:

1. Dashboard updates from a fresh eval run.
2. Non-allow decisions are explainable from the UI.
3. Dashboard is clean enough for the live demo.

### Day 9 - Integrated Baseline vs Protected Demo

Deliverables:

1. Baseline path that runs the same attack scenarios without Aegis enforcement.
2. Protected path that runs through Aegis.
3. Side-by-side output summary.
4. Demo script for three attacks.

Acceptance criteria:

1. Baseline leaks or attempts to dispatch secrets in scripted scenarios.
2. Protected path blocks, warns, sanitizes, or escalates with evidence.
3. Demo can be run in under 10 minutes.

### Day 10 - Hardening and Failure Cases

Deliverables:

1. Add benign edge cases discovered during testing.
2. Tune policy thresholds.
3. Make startup/config errors explicit.
4. Add fallback demo mode that does not require external API access.

Acceptance criteria:

1. Demo still works with no network provider.
2. False blocks are low in scripted benign cases.
3. Known limitations are documented in demo notes.

### Day 11 - Final Demo Package

Deliverables:

1. Final metrics table.
2. Architecture slide or diagram.
3. Demo script with exact sequence.
4. Risk/limitation slide.
5. Short narrative: AIS research prototype -> Aegis practical gateway -> tool-call argument blind spot.

Acceptance criteria:

1. Team can rehearse the full demo twice without manual debugging.
2. Each team member knows which component they will explain.
3. The demo claim is ambitious but accurate.

## 8. Detector Contract

Every detector should return the same logical shape:

| Field | Meaning |
| --- | --- |
| `detector_name` | Stable name such as `tool_call_argument_scanner`. |
| `score` | Risk score from 0.0 to 1.0. |
| `confidence` | Confidence from 0.0 to 1.0. |
| `recommended_action` | One of `allow`, `warn`, `sanitize`, `block`, or `escalate`. |
| `evidence` | Structured detector-specific proof. |
| `latency_ms` | Detector runtime. |

Required evidence for tool-call scanner:

1. `tool_name`
2. `argument_name`
3. `argument_value_preview`
4. `risk_reason`
5. `value_in_trusted_context`
6. `matches_credential_pattern`
7. `matched_canary_id` when applicable

Required evidence for leakage ledger:

1. `turn_score`
2. `cumulative_score`
3. `warn_threshold`
4. `block_threshold`
5. `session_id`

Required evidence for canary detector:

1. `canary_id`
2. `service`
3. `location`
4. `session_id`

## 9. Initial YAML Policy Scope

The first policy file should support four rule types:

1. **Detector score threshold:** block or warn if a named detector score exceeds a threshold.
2. **Tool argument condition:** block if supported tool arguments contain suspicious values.
3. **Canary hit:** block or escalate if a registered honeytoken appears.
4. **Leakage budget threshold:** warn, sanitize, block, or escalate as cumulative score increases.

Avoid nested logical combinations for the first version. The policy engine can evaluate rules independently and choose the most severe resulting action.

## 10. Evaluation Plan

### Scenario Categories

1. **Benign normal usage**
   - Send a normal email.
   - Call a safe HTTP endpoint.
   - Query a database with a normal business question.

2. **Encoded single-turn leakage**
   - Ask model to reveal a credential in Base64.
   - Ask model to split a secret into fragments.
   - Ask model to paraphrase or transform the credential.

3. **Multi-turn dripping**
   - Leak small fragments across several turns.
   - Keep each turn below simple per-turn thresholds.
   - Trigger cumulative ledger threshold.

4. **Tool-call argument exfiltration**
   - Put a credential-shaped value in an email body or recipient.
   - Put a secret in an HTTP query parameter.
   - Put a secret in a database query string.

5. **Canary touches**
   - Include model-visible honeytoken in output.
   - Include honeytoken in tool argument.
   - Verify canary hit produces evidence.

### Metrics

1. Detection rate by scenario category.
2. False block count on benign scenarios.
3. Warning count on benign scenarios.
4. Average gateway latency.
5. Detector hit distribution.
6. Number of scenarios with complete evidence.

### Demo Metrics Table

The final presentation should show:

| Scenario | Baseline Result | Aegis Result | Evidence Shown |
| --- | --- | --- | --- |
| Encoded leak | Secret exposed or transformed | Warn/block/sanitize | Detector score and reason |
| Multi-turn drip | Fragments accumulate | Budget threshold trips | Cumulative score |
| Tool-call exfiltration | Secret sent through tool args | Block before dispatch | Tool name, argument, reason |

## 11. Demo Narrative

The 10-minute presentation should follow this order:

1. State the problem: agents mix trusted credentials with untrusted content.
2. Show why text-only defenses fail: encoded and multi-turn leaks.
3. Introduce Aegis: a runtime gateway with Inspect -> Score -> Enforce.
4. Run baseline agent through three attacks.
5. Run Aegis-protected agent through the same attacks.
6. Show dashboard evidence for each intervention.
7. Explain the research lineage: AIS plus tool-call argument extension.
8. Close with limitations and what would be needed for production.

## 12. Critical Path

The critical path is:

1. Observation proxy.
2. Normalized event model.
3. Detector contract.
4. Tool-call argument scanner.
5. Policy engine.
6. Evaluation harness.
7. Demo dashboard.

Canary generation, leakage accounting, and behavioral scoring matter, but the demo fails most severely if tool-call scanning or the evaluation harness does not work.

## 13. Stretch Goals

Only attempt these after M6 is stable:

1. Partial open-weight introspection using Hugging Face hidden states or hooks.
2. More credential formats in the canary generator.
3. Additional tool schemas.
4. Basic LangChain or OpenAI SDK wrapper.
5. Prometheus metrics.
6. Secret rotation mock action.

## 14. Explicit Limitations to State

1. Aegis is not production-ready.
2. Cloud/API model support cannot provide true CIFT-style activation monitoring.
3. The leakage ledger is a useful cumulative signal, not a formal security proof.
4. The tool-call scanner is scoped to supported schemas.
5. A determined adaptive attacker may find paths around MVP rules.
6. Real deployment would need hard secret-manager integration, stronger policy, persistence, access control, and red-team validation.

## 15. Submission Summary

Aegis will deliver a working runtime security gateway for LLM agents. The project is grounded in AIS research but aims at the paper's deployment gap: structured tool-call arguments and practical enforcement. The team will build a proxy, detector pipeline, credential/canary layer, leakage ledger, YAML policy engine, evaluation harness, and live dashboard. Success is a June 29 demo where Aegis visibly outperforms a baseline agent on encoded leakage, multi-turn leakage, and tool-call argument exfiltration while preserving benign workflows.

