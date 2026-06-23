# Aegis Product Requirements Document

**Project:** Aegis - Runtime Credential Defense for LLM Agents  
**Capstone Direction:** Combine ML with LLM applications  
**Team Size:** Plan for 3 active builders, with a fourth person as optional support  
**Demo Date:** June 29, 2026  
**Source Context:** Gauntlet Capstone Brief, prior Aegis proposal, and Chauhan & Revankar's 2026 AIS paper on pre-output and multi-turn credential exfiltration detection

## Problem Statement

LLM agents are becoming useful because they can call tools, query databases, send messages, browse documents, and act across external systems. To do that work, they often need access to real credentials such as API keys, OAuth tokens, database passwords, or service-specific secrets.

That creates a structural security problem: the same agent context can contain trusted credentials and untrusted content. An attacker can place indirect prompt-injection instructions inside a webpage, email, retrieved document, tool result, or user-provided artifact and steer the agent toward leaking secrets. Existing text-level filters are useful but incomplete because attackers can encode secrets, leak them slowly across turns, or route them through structured tool-call arguments instead of natural-language output.

The AIS paper shows that credential exfiltration should be monitored at multiple levels: model-internal access, planted canaries, and cumulative leakage over time. It also explicitly states that structured tool-call arguments remain a severe blind spot. Aegis turns that research direction into a practical runtime security layer whose capstone contribution is first-class detection and enforcement around agent context, model output, and tool-call arguments.

The product should be strongest for self-hosted models, where Aegis can observe activations and train model-specific probes. It should still support black-box models through the same gateway and SDK surfaces, but black-box mode must be represented honestly as degraded coverage because model-internal signals are unavailable.

## Product Vision

Aegis is a runtime security layer for LLM agents. It can run as a proxy in front of an OpenAI-compatible model endpoint or as an SDK embedded directly in an agent runtime. In both modes, it observes data flow, scores exfiltration risk, and enforces configurable policy before sensitive data leaves the system.

The north-star product is a drop-in defense layer for teams building agentic applications:

- An AI platform team can run Aegis as a local sidecar, gateway, or SDK wrapper in front of an agent.
- A self-hosted-model team can enable introspection hooks and train a model-specific credential-access monitor.
- A cloud/API-model team can run Aegis in black-box mode with canaries, tool-call scanning, text scanning, and cumulative leakage accounting.
- A security engineer can define rules for risky tools and credential-shaped values.
- A red team can replay attack cases and see which detector fired, why it fired, and whether the policy response was appropriate.
- A product team can compare a baseline agent against an Aegis-protected agent in a live demo.

The capstone version should not claim to solve credential exfiltration. It should demonstrate that a focused runtime security layer can make credential leaks more visible, measurable, and harder to execute, especially when self-hosted model introspection is available and when attackers attempt to bypass output filters through tool-call arguments.

## Capability Modes

| Mode | Target | Available Signals | Product Claim |
| --- | --- | --- | --- |
| Self-hosted introspection | Local or self-hosted Hugging Face compatible causal LMs | Activation features, trained probes, canaries, text scans, tool-call scans, leakage ledger | Strongest mode; supports pre-output credential-access detection. |
| Black-box proxy | Cloud APIs or local servers without activation access | Canaries, text scans, tool-call scans, leakage ledger, request/response provenance | Graceful degradation; no CIFT-style claim. |
| SDK embedded | Agent runtimes that can call Aegis libraries directly | Same as self-hosted or black-box, plus deeper tool/runtime context when available | Best integration path for tool enforcement and credential brokerage. |
| Offline training/evaluation | Research and security teams | Activation artifacts, grouped evaluation, calibration reports, scenario results | Produces probe artifacts and operating points for runtime use. |

## Goals

1. Build a shared runtime core that powers both proxy mode and SDK mode.
2. Build a working proxy that can sit in front of an agent and observe model requests, model responses, and selected tool-call arguments.
3. Build a Python SDK path that lets self-hosted agent developers attach Aegis directly to model and tool execution.
4. Implement an Inspect -> Score -> Enforce pipeline with modular detectors.
5. Make self-hosted model introspection a first-class path using PyTorch and device-adaptive execution on CUDA, MPS, or CPU.
6. Treat tool-call argument scanning as a first-class defense, not a future add-on.
7. Add calibrated canary detection and session-level leakage accounting inspired by DP-HONEY and NIMBUS.
8. Provide a live dashboard and audit trail that explain each decision.
9. Run an evaluation harness across benign flows and four attack classes: encoded leakage, low-rate multi-turn leakage, tool-call argument exfiltration, and introspection-detectable credential access.
10. Deliver a compelling June 29 demo showing baseline-agent failure versus Aegis-protected mitigation, with visible differences between introspection-capable and black-box coverage.

## Non-Goals

1. Do not claim production-grade prevention of all credential exfiltration.
2. Do not claim full paper-equivalent CIFT until the implementation has paper-aligned readout positions, calibration, and held-out validation.
3. Do not support every agent framework, model provider, and tool schema.
4. Do not build production secret storage, rotation, tenancy, billing, access control, or long-term compliance workflows.
5. Do not create a complex policy DSL or visual policy editor.
6. Do not optimize for Rust/Go-level gateway performance before proving the detection pipeline works.
7. Do not imply that black-box mode provides model-internal pre-output detection.

## Users

1. **AI platform owner:** Wants to deploy agentic workflows while reducing the chance that secrets leak through model output or tool calls.
2. **Security engineer:** Wants a configurable, inspectable policy layer that logs why an agent action was allowed, warned, blocked, sanitized, or escalated.
3. **Agent developer:** Wants a minimal integration path that does not require rewriting the application.
4. **Red teamer:** Wants replayable attack cases and artifacts that show exactly where defenses succeeded or failed.
5. **Capstone evaluator:** Wants to see a technically ambitious system working live, with honest claims and measurable outcomes.

## User Stories

1. As an agent developer, I want to point my agent at a local Aegis proxy, so that I can test defenses without redesigning the agent.
2. As a self-hosted model developer, I want an SDK wrapper around my local model call, so that Aegis can inspect activations before output is released.
3. As a black-box model developer, I want Aegis to run without activation access, so that I still get canaries, output scanning, tool-call scanning, and leakage accounting.
4. As an agent developer, I want Aegis to pass through normal requests unchanged when risk is low, so that legitimate agent work remains usable.
5. As a security engineer, I want Aegis to parse tool-call arguments, so that secrets cannot bypass output filters by moving into structured fields.
6. As a security engineer, I want detector results to include structured evidence, so that I can understand why a decision was made.
7. As a security engineer, I want a capability report for each configured model, so that I know whether CIFT-style introspection is active or unavailable.
8. As a security engineer, I want configurable YAML rules, so that I can tune actions and thresholds without changing code.
9. As a red teamer, I want encoded attack cases, so that I can test whether simple output filters are being bypassed.
10. As a red teamer, I want multi-turn drip attacks, so that I can test whether cumulative leakage is caught even when each turn looks harmless.
11. As a red teamer, I want tool-call argument exfiltration cases, so that I can test the blind spot identified in the AIS paper.
12. As a red teamer, I want self-hosted introspection cases, so that I can compare pre-output activation detection against black-box defenses.
13. As an AI platform owner, I want a dashboard with recent decisions, risk scores, active capabilities, and detector hits, so that I can see the gateway working during a live session.
14. As an evaluator, I want a baseline-vs-protected demo, so that the improvement is visible without reading the code.
15. As a team member, I want clear component ownership, so that three people can work in parallel without blocking each other.
16. As a future maintainer, I want detector interfaces to be small and typed, so that new defenses can be added without rewriting the proxy or SDK.

## Functional Requirements

| ID | Requirement | Priority | Acceptance Criteria |
| --- | --- | --- | --- |
| FR-1 | Provide a shared runtime core | P0 | Proxy and SDK both call the same typed detector, policy, audit, and ledger services. |
| FR-2 | Provide a proxy-mode gateway | P0 | A local service receives an OpenAI-compatible request, logs it, forwards it to a configured upstream or mock provider, logs the response, and returns a valid response to the caller. |
| FR-3 | Provide a Python SDK mode | P0 | An agent can wrap model generation, tool dispatch, and credential-broker calls without going through HTTP proxy mode. |
| FR-4 | Normalize requests and responses | P0 | Aegis creates a normalized internal representation for messages, tool calls, tool arguments, model output, session ID, trace ID, capability mode, and model identity. |
| FR-5 | Define a detector contract | P0 | Each detector returns name, score, confidence, recommended action, latency, capability requirements, and structured evidence. |
| FR-6 | Provide self-hosted introspection hooks | P0 | For supported PyTorch/Transformers causal LMs, Aegis can capture configured hidden-state readouts before output release on CUDA, MPS, or CPU. |
| FR-7 | Train and load introspection probes | P0 | Offline training can produce a versioned probe artifact, and runtime mode can load it to emit a pre-output activation-risk event. |
| FR-8 | Scan tool-call arguments | P0 | At least three high-risk tool schemas are supported: `send_email`, `http_request`, and `query_database`. Suspicious credential-shaped values in outbound arguments are flagged before dispatch. |
| FR-9 | Implement canary generation and detection | P0 | Aegis can register format-matched honeytokens, inject or expose them only in model-visible context, and detect their appearance in output or tool arguments. |
| FR-10 | Implement session leakage accounting | P0 | Aegis maintains a per-session cumulative leakage score and triggers warning/block/escalation thresholds. |
| FR-11 | Implement policy decisions | P0 | Policy engine maps detector results, capability mode, and cumulative state to `allow`, `warn`, `sanitize`, `block`, or `escalate`. |
| FR-12 | Preserve legitimate credential use | P0 | Real credentials are resolved by a credential broker or tool runtime path, not copied into model-visible context as raw secrets. |
| FR-13 | Provide audit artifacts | P0 | Each evaluated turn produces structured JSON containing trace ID, capability mode, detector results, policy decision, and final action. |
| FR-14 | Provide demo dashboard | P1 | Streamlit dashboard shows recent decisions, fired detectors, active capabilities, risk scores, latency, and scenario outcome. |
| FR-15 | Provide evaluation harness | P1 | Harness runs benign flows plus encoded leak, multi-turn drip, tool-call exfiltration, canary, and introspection scenarios with repeatable outputs. |
| FR-16 | Support capability-adaptive operation | P1 | Black-box mode runs without activation access and records that CIFT-style detection is unavailable rather than silently omitting it. |

## Non-Functional Requirements

1. **Latency:** Detector and policy overhead should target under 50ms per gateway turn for the capstone demo. A stricter sub-10ms scoring target remains an optimization goal for simple detectors.
2. **Device portability:** Introspection code must support CUDA, MPS, and CPU through explicit configuration and `auto` selection.
3. **Local-first operation:** The system must run on commodity hardware for the demo, including the current Mac/MPS development environment.
4. **Explainability:** Every block, warning, or escalation must include structured evidence and a human-readable reason.
5. **Safety of claims:** The product must distinguish demo-grade defense from production-grade guarantees and introspection-capable mode from black-box mode.
6. **Modularity:** Detectors must be replaceable without changing the proxy or SDK entry points.
7. **Testability:** Core detectors and policy logic must be unit-testable without a live LLM provider.
8. **Demo reliability:** The live demo must be runnable with deterministic mock or scripted scenarios if external provider access fails.
9. **Reproducibility:** Dependencies for runtime, introspection, dashboard, and development must live in project configuration rather than only in an ad hoc virtual environment.

## Implementation Decisions

1. **Language and framework:** Use Python because it keeps the gateway, SDK, and PyTorch introspection stack in one environment.
2. **Runtime package:** Build a shared `aegis` core package first, then expose it through FastAPI proxy mode and Python SDK mode.
3. **Entry points:** Proxy and SDK are both first-class. Framework plugins remain future work.
4. **Pipeline:** Use a Headroom-inspired Inspect -> Score -> Enforce structure.
5. **Detector contract:** Use a shared `DetectorResult` shape with structured evidence and declared capability requirements.
6. **Introspection:** Use PyTorch/Transformers for self-hosted models, with device selection supporting `auto`, `cuda`, `mps`, and `cpu`.
7. **Probe status:** The current activation work may be packaged as an experimental CIFT-like detector; paper-aligned CCI/CFS should be a named follow-up unless completed and validated.
8. **Black-box degradation:** Black-box mode should keep the same policy/audit surface but emit `activation_monitor_unavailable` capability evidence.
9. **Policy:** Use a small YAML policy loaded at startup.
10. **Credential broker:** Implement an in-memory broker for the demo. Real secrets are represented by opaque handles in model-visible context.
11. **Canaries:** Implement format-matched honeytokens for a small set of credential families, with deterministic registration and matching.
12. **NIMBUS-inspired ledger:** Implement a practical cumulative leakage score. Do not claim a formal information-flow bound.
13. **Dashboard:** Use Streamlit for fastest demo-quality observability.
14. **Evaluation:** Use replayable YAML/JSON scenarios and promote failures into regression cases.

## Success Metrics

| Metric | Target for Demo | Notes |
| --- | --- | --- |
| Self-hosted capability activation | Demonstrate at least one introspection-capable model path | Show active device, model ID, probe/artifact version, and detector event. |
| Black-box graceful degradation | Same scenario runs without activation access | Audit should show activation unavailable, not silently absent. |
| Encoded single-turn attack detection | Detect most scripted cases | Include Base64/hex/paraphrase-style attacks; avoid claiming benchmark generality. |
| Multi-turn drip detection | Trigger cumulative warning or block before final scripted leak completes | Demonstrates temporal accounting. |
| Tool-call exfiltration detection | Block or escalate suspicious arguments in supported tools | Main capstone differentiator. |
| Benign false blocks | Keep false blocks rare in scripted benign cases | Track false warnings separately. |
| Explainability | 100% of non-allow decisions have evidence | Required for dashboard and audit logs. |
| Demo readiness | One command or documented script runs the end-to-end demo | Include fallback scripted path. |

## Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Shared core takes longer than expected | Proxy and SDK may diverge or slip | Keep the core contract small: normalized events, detectors, policy, audit, and ledger only. |
| Proxy compatibility takes longer than expected | Core demo may slip | Start with OpenAI-compatible chat and tool-call shapes only; use mocks for nonessential providers. |
| SDK scope expands too far | Team loses time to framework-specific integration | Keep SDK to plain Python wrappers around model calls, tool calls, and credential broker access. |
| Tool-call scanner becomes too broad | Detector quality drops | Scope to three high-risk tool schemas and exact structured fields. |
| False positives make demo look brittle | Users distrust the system | Include benign credential-use cases and show warnings separately from blocks. |
| Paper-aligned CIFT implementation exceeds timeline | ML component underwhelms | Ship the current CIFT-like path as experimental with explicit capability labels, and keep canary/tool/ledger defenses complete. |
| MPS/CUDA differences cause runtime failures | Self-hosted path becomes brittle | Test device selection independently and keep CPU as the explicit slow fallback. |
| NIMBUS score is overclaimed | Technical credibility suffers | Call it cumulative leakage scoring, not a formal leakage bound. |
| Dashboard consumes too much time | Core defenses suffer | Keep dashboard to recent decisions, metrics, and scenario selector. |
| Fourth teammate is unavailable | Team capacity shrinks | Plan for three owners; assign optional tasks only to a fourth person. |

## Testing Decisions

1. Unit tests should cover detector behavior from normalized inputs, not implementation details.
2. Policy tests should verify action selection for detector combinations and cumulative thresholds.
3. Credential broker tests should prove real secrets are not exposed through model-visible outputs in supported flows.
4. Device-selection tests should cover `auto`, `cuda`, `mps`, and `cpu`, with unavailable accelerator errors tested by mocks.
5. Introspection tests should cover artifact loading, probe event shape, and capability reporting without requiring a large model.
6. Evaluation tests should include benign cases, direct leak attempts, encoded leak attempts, multi-turn leakage, canary hits, tool-call argument exfiltration, and self-hosted introspection cases.
7. End-to-end tests should run without a live provider by using deterministic mock model responses and tool calls.

## Out of Scope

1. Production secret manager integration beyond a minimal interface or mock.
2. Full multi-tenant identity, audit retention, or compliance controls.
3. Generic parsing for arbitrary tool schemas.
4. Claiming full CIFT Mahalanobis/CCI/CFS parity unless the implementation matches the paper method and passes held-out validation.
5. Formal information-flow security guarantees.
6. Browser extension, SaaS deployment, billing, or user management.
7. Automated secret rotation beyond a logged policy action or mock.

## Further Notes

The strongest framing for Aegis is not "we solved credential exfiltration." The strongest framing is: "AIS shows that text-only defenses are insufficient. Aegis productizes that pattern as a self-hosted-first security layer with activation-aware monitoring when available, graceful black-box degradation when it is not, and practical enforcement around the structured tool-call blind spot."
