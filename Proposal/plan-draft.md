# Aegis Project Plan — High-Level Decisions & Team Partition

## 1. Proxy Mode First?

**Recommendation: Yes — start with Proxy Mode.**

**Why:**
- Zero code changes for users → dramatically higher chance of real adoption and testing.
- Easier to debug and iterate during the 3-week sprint (you control the full request/response).
- Still allows us to demonstrate the core value (Inspect → Score → Enforce pipeline).

**How hard is it to add SDK mode later?**
- Moderately easy. The core `Inspect → Score → Enforce` logic lives in a shared library. 
- Proxy mode just becomes one caller of that library.
- SDK mode would be a thin wrapper around the user's existing client (e.g., wrapping `openai.ChatCompletion.create` or LangChain tools).
- Estimated effort: 2–3 days once the core pipeline is stable.

**Conclusion**: Proxy first is the right call for speed and impact.

---

## 2. ML Performance & Hardware Constraints

**Hard requirements**:
- Must run on commodity hardware (laptop / small server, no GPU assumed for inference).
- Sub-10ms inference latency per decision (ideally <5ms) so it doesn't destroy agent responsiveness.
- Model must be small enough to load quickly and stay in memory.

**Implications**:
- Classical ML only for the risk model (scikit-learn, XGBoost, LightGBM, or tiny ONNX model).
- No heavy transformers at inference time for scoring.
- Feature engineering must be extremely lightweight.

This is achievable. Many production security models run this way.

---

## 3. Policy Configuration — How Users Define Rules

We need a simple but extensible way for users to express policy.

**Recommended approach for 3 weeks**:

Use a **YAML policy file** loaded at startup. Example structure:

```yaml
version: 1
rules:
  - name: "block_unknown_recipients"
    when:
      tool: "send_email"
      argument: "to"
      condition: "not_in_context"
    action: "block"

  - name: "warn_high_risk"
    when:
      risk_score: "> 0.7"
    action: "warn"

  - name: "escalate_credential_tools"
    when:
      tool_category: "credential_access"
      budget_used: "> 0.8"
    action: "escalate"
```

**Why YAML?**
- Human-readable and version-controllable.
- Easy to start with.
- Can later be extended to a small DSL or UI if needed.

For the demo we can ship with 4–5 hardcoded-but-configurable example policies and let users tweak thresholds in the YAML.

---

## 4. Tool-Call Argument Scanning — Defer or Not?

**Strong recommendation: Do NOT defer.**

This is one of the clearest differentiators mentioned in the proposal and in the literature. If we ship without it, the demo loses a lot of its "this is new" power.

**How to make it feasible in 3 weeks**:
- Scope it narrowly at first (focus on 2–3 high-risk tool types, e.g., `send_email`, `query_db`, `http_request`).
- Start with simple provenance + pattern checks (was the argument value present in the original user context? Does it match known credential shapes?).
- Use the same risk model as the rest of the pipeline.

We can treat full semantic understanding of arbitrary tool schemas as future work. For the capstone, proving the *concept* on a few tools is enough.

---

## 5. Team Partition Recommendations

### 3-Person Team (Core)

| Person | Primary Ownership                          | Secondary / Support          | Notes |
|--------|--------------------------------------------|------------------------------|-------|
| **P1** | Core Pipeline (Inspect + Score + Enforce) + Proxy mode | Tool-call argument scanning | Highest technical complexity |
| **P2** | ML Risk Model (CIFT-ML) + training pipeline + evaluation harness | Feature engineering support | Needs to deliver a working model fast |
| **P3** | Canary service (DP-Honey+) + Nimbus ledger + Audit logging + basic dashboard | Policy YAML loader | Dashboard can be minimal (Streamlit or FastAPI + HTML) |

**Demo owner**: Shared, but P3 owns the live dashboard + metrics view.

### 4-Person Team (with buffer)

Add a 4th person with one of these focuses (choose based on team strengths):

- **Option A (Recommended)**: Dedicated Dashboard + Observability person (P3 above splits into two).
- **Option B**: Someone focused purely on evaluation, red-teaming harness, and demo scripting.
- **Option C**: Someone who owns integration testing + end-to-end scenarios.

---

## Summary of Key Decisions

| Question                        | Decision                          | Rationale |
|--------------------------------|-----------------------------------|---------|
| Start with Proxy or SDK?       | Proxy first                       | Adoption + speed |
| Tool-call argument scanning?   | Include from Week 1               | Major differentiator |
| Policy language                | YAML file                         | Simple, versionable |
| ML constraints                 | Commodity hardware, <10ms         | Realistic for demo |
| Team split (3-person)          | Pipeline / ML / Canary+Dashboard  | Clear ownership |
| Defer anything major?          | No — keep scope tight but complete| 3 weeks is short |

---

Next step: Once you confirm or adjust the above decisions, I can produce a detailed Week-by-Week task breakdown with concrete deliverables.
