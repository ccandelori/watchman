# Aegis Product Requirements Document

**Project:** Aegis - Runtime Credential Defense for LLM Agents  
**Capstone Direction:** Combine ML with LLM applications  
**Team Size:** Plan for 3 active builders, with a fourth person as optional support  
**Demo Date:** June 29, 2026  
**Source Context:** Gauntlet Capstone Brief, prior Aegis proposal, and Chauhan & Revankar's 2026 AIS paper on pre-output and multi-turn credential exfiltration detection

## Problem Statement

LLM agents are becoming useful because they can call tools, query databases, send messages, browse documents, and act across external systems. To do that work, they often need access to real credentials such as API keys, OAuth tokens, database passwords, or service-specific secrets.

That creates a structural security problem: the same agent context can contain trusted credentials and untrusted content. An attacker can place indirect prompt-injection instructions inside a webpage, email, retrieved document, tool result, or user-provided artifact and steer the agent toward leaking secrets. Existing text-level filters are useful but incomplete because attackers can encode secrets, leak them slowly across turns, or route them through structured tool-call arguments instead of natural-language output.

The AIS paper shows that credential exfiltration should be monitored at multiple levels: model-internal access, planted canaries, and cumulative leakage over time. It also explicitly states that structured tool-call arguments remain a severe blind spot. Aegis turns that research direction into a practical runtime gateway whose capstone contribution is first-class detection and enforcement around agent context, model output, and tool-call arguments.

## Product Vision

Aegis is a security gateway for LLM agents. It sits between an agent runtime and external model/tool providers, observes the data flow, scores exfiltration risk, and enforces configurable policy before sensitive data leaves the system.

The north-star product is a drop-in defense layer for teams building agentic applications:

- An AI platform team can run Aegis as a local sidecar or gateway in front of an agent.
- A security engineer can define rules for risky tools and credential-shaped values.
- A red team can replay attack cases and see which detector fired, why it fired, and whether the policy response was appropriate.
- A product team can compare a baseline agent against an Aegis-protected agent in a live demo.

The capstone version should not claim to solve credential exfiltration. It should demonstrate that a focused runtime gateway can make credential leaks more visible, measurable, and harder to execute, especially through tool-call arguments.

## Goals

1. Build a working runtime gateway that can sit in front of an agent and observe model requests, model responses, and selected tool-call arguments.
2. Implement an Inspect -> Score -> Enforce pipeline with modular detectors.
3. Treat tool-call argument scanning as a first-class defense, not a future add-on.
4. Add calibrated canary detection and session-level leakage accounting inspired by DP-HONEY and NIMBUS.
5. Provide a live dashboard and audit trail that explain each decision.
6. Run an evaluation harness across benign flows and three attack classes: encoded leakage, low-rate multi-turn leakage, and tool-call argument exfiltration.
7. Deliver a compelling June 29 demo showing baseline-agent failure versus Aegis-protected mitigation.

## Non-Goals

1. Do not claim production-grade prevention of all credential exfiltration.
2. Do not implement full research-grade CIFT activation probing as a required MVP dependency.
3. Do not support every agent framework, model provider, and tool schema.
4. Do not build production secret storage, rotation, tenancy, billing, access control, or long-term compliance workflows.
5. Do not create a complex policy DSL or visual policy editor.
6. Do not optimize for Rust/Go-level gateway performance before proving the detection pipeline works.

## Users

1. **AI platform owner:** Wants to deploy agentic workflows while reducing the chance that secrets leak through model output or tool calls.
2. **Security engineer:** Wants a configurable, inspectable policy layer that logs why an agent action was allowed, warned, blocked, sanitized, or escalated.
3. **Agent developer:** Wants a minimal integration path that does not require rewriting the application.
4. **Red teamer:** Wants replayable attack cases and artifacts that show exactly where defenses succeeded or failed.
5. **Capstone evaluator:** Wants to see a technically ambitious system working live, with honest claims and measurable outcomes.

## User Stories

1. As an agent developer, I want to point my agent at a local Aegis proxy, so that I can test defenses without redesigning the agent.
2. As an agent developer, I want Aegis to pass through normal requests unchanged when risk is low, so that legitimate agent work remains usable.
3. As a security engineer, I want Aegis to parse tool-call arguments, so that secrets cannot bypass output filters by moving into structured fields.
4. As a security engineer, I want detector results to include structured evidence, so that I can understand why a decision was made.
5. As a security engineer, I want configurable YAML rules, so that I can tune actions and thresholds without changing code.
6. As a red teamer, I want encoded attack cases, so that I can test whether simple output filters are being bypassed.
7. As a red teamer, I want multi-turn drip attacks, so that I can test whether cumulative leakage is caught even when each turn looks harmless.
8. As a red teamer, I want tool-call argument exfiltration cases, so that I can test the blind spot identified in the AIS paper.
9. As an AI platform owner, I want a dashboard with recent decisions, risk scores, and detector hits, so that I can see the gateway working during a live session.
10. As an evaluator, I want a baseline-vs-protected demo, so that the improvement is visible without reading the code.
11. As a team member, I want clear component ownership, so that three people can work in parallel without blocking each other.
12. As a future maintainer, I want detector interfaces to be small and typed, so that new defenses can be added without rewriting the gateway.

## Functional Requirements

| ID | Requirement | Priority | Acceptance Criteria |
| --- | --- | --- | --- |
| FR-1 | Provide a proxy-mode gateway | P0 | A local service receives an OpenAI-compatible request, logs it, forwards it to a configured upstream or mock provider, logs the response, and returns a valid response to the caller. |
| FR-2 | Normalize requests and responses | P0 | Gateway creates a normalized internal representation for messages, tool calls, tool arguments, model output, session ID, and trace ID. |
| FR-3 | Define a detector contract | P0 | Each detector returns name, score, confidence, recommended action, latency, and structured evidence. |
| FR-4 | Scan tool-call arguments | P0 | At least three high-risk tool schemas are supported: `send_email`, `http_request`, and `query_database`. Suspicious credential-shaped values in outbound arguments are flagged before dispatch. |
| FR-5 | Implement canary generation and detection | P0 | Gateway can register format-matched honeytokens, inject or expose them only in model-visible context, and detect their appearance in output or tool arguments. |
| FR-6 | Implement session leakage accounting | P0 | Gateway maintains a per-session cumulative leakage score and triggers warning/block/escalation thresholds. |
| FR-7 | Implement policy decisions | P0 | Policy engine maps detector results and cumulative state to `allow`, `warn`, `sanitize`, `block`, or `escalate`. |
| FR-8 | Preserve legitimate credential use | P0 | Real credentials are resolved by a credential broker or tool runtime path, not copied into model-visible context as raw secrets. |
| FR-9 | Provide audit artifacts | P0 | Each evaluated turn produces structured JSON containing trace ID, detector results, policy decision, and final action. |
| FR-10 | Provide demo dashboard | P1 | Streamlit dashboard shows recent decisions, fired detectors, risk scores, latency, and scenario outcome. |
| FR-11 | Provide evaluation harness | P1 | Harness runs benign flows plus encoded leak, multi-turn drip, and tool-call exfiltration attacks with repeatable outputs. |
| FR-12 | Support capability-adaptive operation | P1 | Cloud/API mode runs without activation access; open-weight mode is documented as stronger or stretch if model hooks are implemented. |

## Non-Functional Requirements

1. **Latency:** Detector and policy overhead should target under 50ms per gateway turn for the capstone demo. A stricter sub-10ms scoring target remains an optimization goal for simple detectors.
2. **Local-first operation:** The system must run on commodity hardware for the demo.
3. **Explainability:** Every block, warning, or escalation must include structured evidence and a human-readable reason.
4. **Safety of claims:** The product must distinguish demo-grade defense from production-grade guarantees.
5. **Modularity:** Detectors must be replaceable without changing the proxy entry point.
6. **Testability:** Core detectors and policy logic must be unit-testable without a live LLM provider.
7. **Demo reliability:** The live demo must be runnable with deterministic mock or scripted scenarios if external provider access fails.

## Implementation Decisions

1. **Language and framework:** Use Python and FastAPI for the gateway because Python maximizes team speed and ML integration within the capstone timeline.
2. **Entry point:** Build proxy mode first. SDK mode and framework plugins are future work.
3. **Pipeline:** Use a Headroom-inspired Inspect -> Score -> Enforce structure.
4. **Detector contract:** Use a shared `DetectorResult` shape with structured evidence.
5. **Policy:** Use a small YAML policy loaded at startup.
6. **Credential broker:** Implement an in-memory broker for the demo. Real secrets are represented by opaque handles in model-visible context.
7. **Canaries:** Implement format-matched honeytokens for a small set of credential families, with deterministic registration and matching.
8. **NIMBUS-inspired ledger:** Implement a practical cumulative leakage score. Do not claim a formal information-flow bound.
9. **CIFT positioning:** Treat full activation probing as stretch work. MVP uses cloud-compatible signals plus clear extension points for open-weight introspection.
10. **Dashboard:** Use Streamlit for fastest demo-quality observability.
11. **Evaluation:** Use replayable YAML/JSON scenarios and promote failures into regression cases.

## Success Metrics

| Metric | Target for Demo | Notes |
| --- | --- | --- |
| Encoded single-turn attack detection | Detect most scripted cases | Include Base64/hex/paraphrase-style attacks; avoid claiming benchmark generality. |
| Multi-turn drip detection | Trigger cumulative warning or block before final scripted leak completes | Demonstrates temporal accounting. |
| Tool-call exfiltration detection | Block or escalate suspicious arguments in supported tools | Main capstone differentiator. |
| Benign false blocks | Keep false blocks rare in scripted benign cases | Track false warnings separately. |
| Explainability | 100% of non-allow decisions have evidence | Required for dashboard and audit logs. |
| Demo readiness | One command or documented script runs the end-to-end demo | Include fallback scripted path. |

## Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Proxy compatibility takes longer than expected | Core demo may slip | Start with OpenAI-compatible chat and tool-call shapes only; use mocks for nonessential providers. |
| Tool-call scanner becomes too broad | Detector quality drops | Scope to three high-risk tool schemas and exact structured fields. |
| False positives make demo look brittle | Users distrust the system | Include benign credential-use cases and show warnings separately from blocks. |
| CIFT implementation exceeds timeline | ML component underwhelms | Position full activation probing as stretch; ship cloud-compatible behavioral/provenance signals. |
| NIMBUS score is overclaimed | Technical credibility suffers | Call it cumulative leakage scoring, not a formal leakage bound. |
| Dashboard consumes too much time | Core defenses suffer | Keep dashboard to recent decisions, metrics, and scenario selector. |
| Fourth teammate is unavailable | Team capacity shrinks | Plan for three owners; assign optional tasks only to a fourth person. |

## Testing Decisions

1. Unit tests should cover detector behavior from normalized inputs, not implementation details.
2. Policy tests should verify action selection for detector combinations and cumulative thresholds.
3. Credential broker tests should prove real secrets are not exposed through model-visible outputs in supported flows.
4. Evaluation tests should include benign cases, direct leak attempts, encoded leak attempts, multi-turn leakage, canary hits, and tool-call argument exfiltration.
5. End-to-end tests should run without a live provider by using deterministic mock model responses and tool calls.

## Out of Scope

1. Production secret manager integration beyond a minimal interface or mock.
2. Full multi-tenant identity, audit retention, or compliance controls.
3. Generic parsing for arbitrary tool schemas.
4. Full CIFT Mahalanobis/probe training pipeline.
5. Formal information-flow security guarantees.
6. Browser extension, SaaS deployment, billing, or user management.
7. Automated secret rotation beyond a logged policy action or mock.

## Further Notes

The strongest framing for Aegis is not "we solved credential exfiltration." The strongest framing is: "AIS shows that text-only defenses are insufficient and that structured tool-call arguments remain a severe blind spot. Aegis builds the practical gateway and evaluation loop needed to attack that blind spot in a working agent system."

