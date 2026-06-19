# Aegis — Next Iteration Outline (High-Level Plan)

## Core Philosophy
- Proxy-first architecture (Headroom-inspired) for easy adoption
- Inspect → Score → Enforce pipeline with plugin-style detectors
- Tool-call argument scanning treated as first-class from Day 1
- Lightweight but credible evaluation harness (AgentForge influence)
- Keep scope tight for 3 weeks: deliver a working, demonstrable system

## High-Level Architecture

```
Agent / App
    │
    ▼
Aegis Gateway (FastAPI sidecar / proxy)
    │
    ├── Request Normalization + Provenance Labeling
    │
    └── Defense Pipeline (plugin detectors)
            ├── Tool-Call Argument Scanner
            ├── DP-Honey Canary Service
            ├── NIMBUS Leakage Accountant
            ├── CIFT-ML Pre-Output Monitor (MVP: one open-weight model)
            └── Text / Canary Detector
                    │
                    ▼
            Policy Engine
                    │
                    ├── Allow / Warn / Sanitize / Block / Rotate Secret
                    └── Audit + Observability
```

## Key Technical Decisions (Locked)

- **Entry point**: Proxy mode first (SDK mode noted as future work)
- **Detector interface**: Standard contract (score, evidence, confidence, recommended action)
- **Policy**: Simple YAML file loaded at startup
- **ML constraints**: Commodity hardware, <10ms inference target → classical ML only for risk scoring
- **Tool-call argument scanning**: Included in MVP (narrow scope: 2–3 high-risk tools)
- **Secrets**: Credential broker + opaque handles; real credentials never model-visible by default
- **Evaluation**: Minimal but structured harness with replayable cases

## Component Breakdown (MVP Scope)

| Component                    | Priority | Owner Suggestion | Notes |
|-----------------------------|----------|------------------|-------|
| Gateway + Proxy layer       | High     | P1               | FastAPI, OpenAI/Anthropic compatible routes |
| Request Normalization       | High     | P1               | Provenance labels, argument parsing |
| Tool-Call Argument Scanner  | High     | P1               | Core differentiator |
| DP-Honey Canary Service     | High     | P3               | Generation + validation |
| NIMBUS Leakage Accountant   | High     | P3               | Session + cross-session budget |
| CIFT-ML Risk Model          | Medium   | P2               | One open-weight model for demo |
| Policy Engine + YAML        | High     | P3               | Simple rule evaluation |
| Audit / Logging             | High     | P3               | Structured JSON artifacts |
| Minimal Dashboard           | Medium   | P3               | Streamlit or basic FastAPI page |
| Evaluation Harness          | High     | P2               | Encoded leak, multi-turn drip, argument exfil cases |

## Team Partition (3-Person)

- **P1**: Gateway, proxy, normalization, Tool-Call Argument Scanner, overall pipeline integration
- **P2**: ML Risk Model (CIFT-ML), evaluation harness, red-teaming cases
- **P3**: Canary service, NIMBUS, Policy YAML, audit/logging, minimal dashboard

## Demo Definition (June 29)

Live comparison of baseline agent vs Aegis-protected agent across three attack types:
1. Encoded single-turn credential leakage
2. Low-rate multi-turn dripping
3. Tool-call argument exfiltration

Include real-time signals from the defense pipeline and a simple quantitative summary.

## Open Questions to Resolve This Week

1. Exact detector plugin interface definition
2. Scope of initial policy YAML (how many rule types?)
3. Depth of CIFT integration for the demo (full activation probing or lighter behavioral signals?)
4. Dashboard minimal viable surface

---
This outline is intentionally kept at the decision + ownership level. Ready to expand into a week-by-week task plan once these are confirmed.