# Aegis Project Plan

**Project:** Aegis - Runtime Credential Defense for LLM Agents  
**Planning Basis:** Gauntlet Capstone Brief, prior Aegis proposal, and AIS credential-exfiltration paper  
**Demo Date:** June 29, 2026  
**Team Assumption:** 3 active builders; fourth teammate is optional support  
**Goal:** Build and demo a self-hosted-first runtime security layer for LLM agents, exposed through both proxy and SDK surfaces, with activation-aware introspection as the strongest path and black-box model support as graceful degradation.

## 1. Capstone Fit

The Gauntlet brief asks for a technically hard project whose ambition matters more than polish. Aegis fits Direction A: combine ML with LLM applications.

LLM agents supply the real-world action layer: they read untrusted content, form responses, and call tools. Classical ML and systems-style detectors supply the safety layer: activation probing, scoring, canary detection, cumulative leakage accounting, and policy enforcement. The project is ambitious because it attacks a structural failure mode in agent systems, not a cosmetic UX problem.

The live demo should make that ambition visible: a baseline agent leaks or attempts to leak credentials, while an Aegis-protected agent blocks, warns, sanitizes, or escalates with evidence. The demo should also show the product distinction between introspection-capable self-hosted mode and black-box mode.

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

Aegis uses those limitations as project direction. The capstone build should productize the AIS pattern into a runtime security layer, prioritize self-hosted model introspection where the AIS signal is strongest, degrade honestly for black-box models, and specifically extend protection to structured tool-call arguments.

## 3. System Architecture

```text
Agent Runtime
    |
    +-------------------------+
    |                         |
    v                         v
Aegis Proxy              Aegis SDK
OpenAI-compatible        Python wrapper
HTTP surface             embedded surface
    |                         |
    +-----------+-------------+
                |
                v
        Aegis Runtime Core
        - normalized events
        - detector contract
        - policy engine
        - credential broker
        - session ledger
        - audit artifacts
                |
       +--------+---------+----------------+
       |                  |                |
       v                  v                v
Self-hosted          Black-box         Tool Runtime
Introspection        Provider          Enforcement
- PyTorch hooks      - request scan     - send_email
- CUDA/MPS/CPU       - output scan      - http_request
- CIFT-like probe    - canaries         - query_database
- pre-output event   - leakage ledger
                |
                v
        Dashboard + Evaluation Reports
```

External surfaces:

1. **Self-hosted model runtime:** Hugging Face/PyTorch causal LMs with hidden-state access on CUDA, MPS, or CPU.
2. **Black-box LLM provider or mock provider:** OpenAI-compatible route for the demo.
3. **Tool runtime:** Supported demo tools include `send_email`, `http_request`, and `query_database`.
4. **Credential broker:** In-memory broker maps opaque handles to real or mock secrets.
5. **Dashboard:** Streamlit app reads audit artifacts and scenario results.

Capability modes:

| Mode | Meaning | Required Demo Evidence |
| --- | --- | --- |
| Self-hosted introspection | Aegis can capture hidden-state readouts and run a probe before output release. | Audit event includes model ID, selected device, probe/artifact ID, and activation detector result. |
| Black-box proxy | Aegis cannot see activations but still inspects prompts, outputs, canaries, tools, and leakage state. | Audit event explicitly records activation monitor unavailable. |
| SDK embedded | Aegis runs inside the agent process for deeper tool and credential-broker integration. | SDK example wraps a model call or scripted model call and a tool dispatch. |
| Offline training/evaluation | Aegis trains or evaluates probe artifacts outside the request path. | Evaluation report or artifact metadata is versioned and loadable by runtime. |

## 4. Component Responsibilities

| Component | Purpose | Owner | MVP Acceptance Criteria |
| --- | --- | --- | --- |
| Runtime core | Shared typed pipeline used by proxy and SDK | P1 | Proxy and SDK both call the same detector, policy, audit, and ledger code. |
| Proxy adapter | Receives, logs, forwards, and returns model/tool traffic | P1 | Can run an OpenAI-compatible request and produce audit logs. |
| SDK adapter | Embeds Aegis into a Python agent process | P1 | Example wraps model generation and tool dispatch without HTTP. |
| Normalization layer | Converts provider/tool shapes into internal typed events | P1 | Produces consistent events for messages, responses, supported tool calls, model identity, and capability mode. |
| Detector contract | Shared interface for all detectors | P1 | Every detector returns score, confidence, action recommendation, evidence, latency, and capability requirements. |
| Tool-call argument scanner | Detects suspicious credential movement through structured arguments | P1 | Blocks/escalates scripted leaks in `send_email`, `http_request`, and `query_database`. |
| Introspection runtime | Captures hidden-state readouts for self-hosted models | P2 | Runs on `auto` device selection across CUDA/MPS/CPU semantics and emits a pre-output monitor event. |
| Probe training/evaluation | Produces and validates CIFT-like probe artifacts | P2 | Trains or loads a versioned probe artifact and reports grouped/held-out evidence. |
| Evaluation harness | Runs benign and attack scenarios repeatedly | P2 | Runs encoded, multi-turn, tool-call, canary, introspection, and benign cases from files. |
| Canary service | Generates, registers, injects, and detects honeytokens | P3 | Detects registered canaries in model output and tool arguments. |
| Leakage ledger | Tracks cumulative session risk | P3 | Warns/blocks when scripted multi-turn leak crosses threshold. |
| Policy engine | Maps detector output to final action | P3 | YAML policy controls allow/warn/sanitize/block/escalate thresholds. |
| Audit and dashboard | Shows live decisions, capabilities, and evidence | P3 | Dashboard displays scenario, capability mode, action, risk, detectors, and latency. |

## 5. Team Ownership

### P1 - Runtime Core, Proxy, and Enforcement Lead

Primary ownership:

1. Shared runtime core.
2. FastAPI proxy.
3. Provider-compatible proxy route.
4. Minimal Python SDK wrapper.
5. Normalized request/response model.
6. Inspect -> Score -> Enforce orchestration.
7. Tool-call argument scanner.

Secondary support:

1. Integration with policy engine.
2. End-to-end demo wiring.

### P2 - Introspection and Evaluation Lead

Primary ownership:

1. Self-hosted introspection runtime.
2. Probe artifact loading and runtime event shape.
3. Offline probe training/evaluation workflow.
4. Evaluation harness.
5. Attack and benign scenario library.
6. Baseline-agent comparison.
7. Quantitative demo report.

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
| M1 - Runtime core skeleton | Day 1 | Shared event, detector, policy, audit, and capability models exist and are used by both adapters. |
| M2 - Proxy and SDK observation | Day 2 | Proxy and SDK can each produce a normalized audit artifact from a scripted or mock model call. |
| M3 - Detector pipeline | Day 4 | At least two detectors run through the shared interface and produce policy decisions. |
| M4 - Tool-call defense | Day 5 | Supported tool-call exfiltration attempts are blocked or escalated with evidence. |
| M5 - Self-hosted introspection event | Day 6 | A PyTorch/Transformers path emits an activation-risk or capability-unavailable event with device metadata. |
| M6 - Canary + leakage accounting | Day 7 | Canary hits and multi-turn budget thresholds appear in audit logs and dashboard. |
| M7 - Evaluation harness | Day 8 | Benign and attack scenarios run from repeatable files and produce summary metrics. |
| M8 - Integrated demo | Day 10 | Baseline vs protected flows run end-to-end with dashboard evidence in self-hosted and black-box modes. |
| M9 - Final polish | Day 12 | Demo script, fallback path, metrics table, and final narrative are ready. |

## 7. Day-by-Day Execution Plan

### Day 1 - Runtime Core and Architecture Lock

Deliverables:

1. Repository structure for core, proxy, SDK, detectors, policy, introspection, storage, dashboard, and evals.
2. Shared data models for normalized turns, model capabilities, tool calls, detector results, policy decisions, and audit events.
3. Minimal in-process SDK wrapper around a scripted model call.
4. One mock provider route for deterministic proxy testing.
5. First six demo scenarios drafted: benign email, benign API call, encoded leak, multi-turn drip, tool-call exfiltration, introspection-detectable credential access.

Acceptance criteria:

1. Team can run the project locally.
2. A single command starts the proxy.
3. A mock proxy request produces a normalized audit artifact.
4. A local SDK call produces the same normalized audit artifact shape.

### Day 2 - Observation-Only Proxy and SDK

Deliverables:

1. OpenAI-compatible pass-through or mock-compatible chat route.
2. SDK wrapper that evaluates a scripted model response.
3. Structured JSON logging for request, response, session ID, trace ID, capability mode, model identity, and latency.
4. Minimal CLI or script that sends one benign scenario through the proxy and SDK.

Acceptance criteria:

1. Proxy path works without detectors.
2. SDK path works without detectors.
3. Logs are readable by the future dashboard.
4. Upstream failure produces a clear error instead of a silent pass.

### Day 3 - Detector Contract and Policy Skeleton

Deliverables:

1. Detector interface.
2. `DetectorResult` schema.
3. Policy decision schema.
4. YAML policy loader.
5. Capability report schema.
6. Static detectors for canary match and credential-shaped value match.

Acceptance criteria:

1. Each detector emits structured evidence.
2. Policy engine can map detector output to allow/warn/block/escalate.
3. Black-box mode records activation monitoring as unavailable.
4. Unit tests cover policy threshold behavior.

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

### Day 5 - Self-Hosted Introspection Runtime

Deliverables:

1. Device-selection wrapper for `auto`, `cuda`, `mps`, and `cpu`.
2. Runtime adapter that can call a supported PyTorch/Transformers causal LM with hidden-state output.
3. Activation monitor event shape that includes model ID, revision, selected device, feature/probe ID, risk score, and evidence.
4. Black-box unavailable event with the same policy/audit compatibility.

Acceptance criteria:

1. Existing MPS development environment can select MPS through `auto`.
2. CPU fallback is explicit and testable.
3. Runtime can emit either an activation-risk event or an activation-unavailable event.
4. Probe implementation is clearly labeled as CIFT-like unless paper-aligned CCI/CFS is actually implemented.

### Day 6 - Credential Broker and Canary Registry

Deliverables:

1. In-memory credential broker with opaque handles.
2. Canary registry keyed by session and service.
3. Format-matched honeytoken generator for at least two credential families.
4. Detector that scans output and tool arguments for registered canaries.

Acceptance criteria:

1. Model-visible context never needs raw real credentials in demo flows.
2. Canary appearance in output triggers non-allow policy action.
3. Canary appearance in tool arguments triggers non-allow policy action.

### Day 7 - Leakage Ledger

Deliverables:

1. Per-session cumulative leakage score.
2. Rule thresholds for warn, sanitize, block, and escalate.
3. Multi-turn scenario that stays below per-turn thresholds but crosses cumulative threshold.
4. Audit output showing per-turn and cumulative score.

Acceptance criteria:

1. Multi-turn drip attack triggers before final scripted leak completes.
2. Benign multi-turn scenario stays under blocking threshold.
3. Dashboard can display current budget state.

### Day 8 - Evaluation Harness

Deliverables:

1. Scenario file format.
2. Runner for benign and attack scenarios.
3. Metrics summary: detection count, false block count, warnings, average latency, detector hits.
4. Regression output for failed cases.
5. Capability-mode comparison for self-hosted introspection versus black-box mode.

Acceptance criteria:

1. Harness runs without a live model by using scripted responses/tool calls.
2. Harness can optionally run against a real provider if configured.
3. Metrics are reproducible across runs.
4. At least one scenario records the difference between activation-capable and activation-unavailable modes.

### Day 9 - Dashboard

Deliverables:

1. Streamlit dashboard reading audit artifacts and evaluation summaries.
2. Recent decisions table.
3. Metrics strip: total cases, blocked, warned, false blocks, average latency, canary hits, active capability modes.
4. Scenario detail view with detector evidence.

Acceptance criteria:

1. Dashboard updates from a fresh eval run.
2. Non-allow decisions are explainable from the UI.
3. Dashboard is clean enough for the live demo.
4. Dashboard makes black-box degradation visible.

### Day 10 - Integrated Baseline vs Protected Demo

Deliverables:

1. Baseline path that runs the same attack scenarios without Aegis enforcement.
2. Protected path that runs through Aegis.
3. Side-by-side output summary.
4. Demo script for three attacks plus one introspection-capability comparison.

Acceptance criteria:

1. Baseline leaks or attempts to dispatch secrets in scripted scenarios.
2. Protected path blocks, warns, sanitizes, or escalates with evidence.
3. Demo can be run in under 10 minutes.
4. Demo distinguishes self-hosted introspection from black-box degradation without overstating black-box coverage.

### Day 11 - Hardening and Failure Cases

Deliverables:

1. Add benign edge cases discovered during testing.
2. Tune policy thresholds.
3. Make startup/config errors explicit.
4. Add fallback demo mode that does not require external API access.

Acceptance criteria:

1. Demo still works with no network provider.
2. False blocks are low in scripted benign cases.
3. Known limitations are documented in demo notes.

### Day 12 - Final Demo Package

Deliverables:

1. Final metrics table.
2. Architecture slide or diagram.
3. Demo script with exact sequence.
4. Risk/limitation slide.
5. Short narrative: AIS research prototype -> Aegis runtime security layer -> tool-call argument blind spot.

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

Required evidence for activation monitor:

1. `capability_mode`
2. `model_id`
3. `selected_device`
4. `probe_id`
5. `feature_key`
6. `risk_score`
7. `unavailable_reason` when activation access is not available

## 9. Initial YAML Policy Scope

The first policy file should support four rule types:

1. **Detector score threshold:** block or warn if a named detector score exceeds a threshold.
2. **Tool argument condition:** block if supported tool arguments contain suspicious values.
3. **Canary hit:** block or escalate if a registered honeytoken appears.
4. **Leakage budget threshold:** warn, sanitize, block, or escalate as cumulative score increases.
5. **Capability condition:** warn or annotate when a requested detector is unavailable in black-box mode.

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

6. **Self-hosted introspection**
   - Run an activation-capable scenario through a local model or stored activation/probe artifact.
   - Run the same scenario in black-box mode.
   - Verify the audit shows the extra pre-output activation signal only in self-hosted mode.

### Metrics

1. Detection rate by scenario category.
2. False block count on benign scenarios.
3. Warning count on benign scenarios.
4. Average runtime latency.
5. Detector hit distribution.
6. Capability mode distribution.
7. Number of scenarios with complete evidence.

### Demo Metrics Table

The final presentation should show:

| Scenario | Baseline Result | Aegis Result | Evidence Shown |
| --- | --- | --- | --- |
| Encoded leak | Secret exposed or transformed | Warn/block/sanitize | Detector score and reason |
| Multi-turn drip | Fragments accumulate | Budget threshold trips | Cumulative score |
| Tool-call exfiltration | Secret sent through tool args | Block before dispatch | Tool name, argument, reason |
| Self-hosted introspection | Model approaches credential access | Pre-output risk event contributes to policy | Model ID, device, probe, feature evidence |
| Black-box degradation | Same model call without activation access | Text/canary/tool/ledger only | Capability report shows activation unavailable |

## 11. Demo Narrative

The 10-minute presentation should follow this order:

1. State the problem: agents mix trusted credentials with untrusted content.
2. Show why text-only defenses fail: encoded and multi-turn leaks.
3. Introduce Aegis: a runtime security layer with proxy and SDK surfaces.
4. Run baseline agent through three attacks.
5. Run Aegis-protected agent through the same attacks.
6. Show self-hosted introspection evidence and the black-box degraded equivalent.
7. Show dashboard evidence for each intervention.
8. Explain the research lineage: AIS plus tool-call argument extension.
9. Close with limitations and what would be needed for production.

## 12. Critical Path

The critical path is:

1. Shared runtime core.
2. Normalized event model.
3. Proxy adapter.
4. Minimal SDK adapter.
5. Detector contract.
6. Tool-call argument scanner.
7. Policy engine.
8. Self-hosted introspection event path.
9. Evaluation harness.
10. Demo dashboard.

Canary generation and leakage accounting matter, but the demo fails most severely if the shared core, tool-call scanning, introspection capability evidence, or evaluation harness does not work.

## 13. Stretch Goals

Only attempt these after M6 is stable:

1. Paper-aligned CCI/CFS readout-position implementation if the demo is already stable.
2. More credential formats in the canary generator.
3. Additional tool schemas.
4. LangChain or OpenAI SDK framework plugin beyond the plain Python SDK.
5. Prometheus metrics.
6. Secret rotation mock action.

## 14. Explicit Limitations to State

1. Aegis is not production-ready.
2. Cloud/API model support cannot provide true CIFT-style activation monitoring.
3. The leakage ledger is a useful cumulative signal, not a formal security proof.
4. The tool-call scanner is scoped to supported schemas.
5. A determined adaptive attacker may find paths around MVP rules.
6. Real deployment would need hard secret-manager integration, stronger policy, persistence, access control, and red-team validation.
7. The current activation work is CIFT-like unless the paper-aligned calibrated CCI/CFS path is completed and independently validated.

## 15. Submission Summary

Aegis will deliver a working runtime security layer for LLM agents. The project is grounded in AIS research but productizes it around two practical deployment surfaces: a proxy for drop-in protection and an SDK for deeper self-hosted integration. The strongest mode uses PyTorch/Transformers introspection on CUDA, MPS, or CPU; black-box mode degrades gracefully to canaries, text scanning, tool-call scanning, policy, and cumulative leakage accounting. Success is a June 29 demo where Aegis visibly outperforms a baseline agent on encoded leakage, multi-turn leakage, tool-call argument exfiltration, and introspection-capable credential-access scenarios while preserving benign workflows.
