# Aegis/Watchman Release Readiness

Date: 2026-06-25

## Verdict

Aegis/Watchman is stronger than a bounded demo, but it is not ready for an
unqualified production claim.

The honest claim today is narrower:

> Aegis has a certified model-specific CIFT reference for Qwen/Qwen3-4B on MPS,
> a default mock-provider sentinel smoke path, durable redacted JSONL audit with
> per-turn runtime evidence, and deterministic
> DP-HONEY/credential-slot/canary/NIMBUS runtime controls. Real provider mode is
> configured and hardened, but live real-provider evidence is still missing in
> this environment because provider credentials are unset.

## Current Evidence

| Evidence | Status | Path or command |
| --- | --- | --- |
| Focused hardening regression suite | Pass, 221 tests | `PYTHONPATH=src:introspection/src uv run pytest --no-cov tests/aegis/test_contracts.py tests/aegis/test_loopback_openai_provider.py tests/aegis/test_loopback_real_provider_smoke.py tests/aegis/test_provider_preflight.py tests/aegis/test_provider_smoke_verify.py tests/aegis/test_proxy_config.py tests/aegis/test_proxy.py tests/aegis/test_proxy_http_app.py tests/aegis/test_audit_explain_cli.py tests/aegis/test_nimbus_runtime.py tests/aegis/test_nimbus_eval.py tests/aegis/test_nimbus_training_corpus.py tests/aegis/test_nimbus_infonce_training.py tests/aegis/test_proxy_smoke.py tests/aegis/test_proxy_smoke_contract.py tests/aegis/test_proxy_cift_smoke.py tests/aegis/test_cift_extractor_client.py introspection/tests/test_runtime_bridge.py introspection/tests/test_grok_redteam_corpus.py introspection/tests/test_watchman_synthetic_corpus.py` |
| Provider/smoke CLI regression | Pass, 63 tests after loopback provider, provider preflight, provider evidence verifier, and readiness provider-mode hardening | `PYTHONPATH=src:introspection/src uv run pytest --no-cov tests/aegis/test_loopback_openai_provider.py tests/aegis/test_loopback_real_provider_smoke.py tests/aegis/test_provider_preflight.py tests/aegis/test_provider_smoke_verify.py tests/aegis/test_proxy_config.py tests/aegis/test_proxy_smoke.py tests/aegis/test_proxy_smoke_contract.py` |
| Focused lint | Pass | `PYTHONPATH=src:introspection/src uv run ruff check src/aegis/core/contracts.py src/aegis/audit/explain.py src/aegis/audit/explain_cli.py src/aegis/audit/jsonl.py src/aegis/replay/nimbus_training.py src/aegis/replay/nimbus_infonce.py src/aegis/proxy/provider_smoke_verify.py tests/aegis/test_contracts.py tests/aegis/test_audit_explain_cli.py tests/aegis/test_loopback_real_provider_smoke.py tests/aegis/test_provider_smoke_verify.py tests/aegis/test_nimbus_training_corpus.py tests/aegis/test_nimbus_infonce_training.py src/aegis/proxy/mock_app.py src/aegis/proxy/smoke.py src/aegis/proxy/smoke_contract.py tests/aegis/test_proxy.py tests/aegis/test_proxy_http_app.py tests/aegis/test_proxy_smoke.py tests/aegis/test_proxy_smoke_contract.py README.md docs/aegis-runtime-spine.md docs/aegis-watchman-release-readiness-2026-06-25.md docs/nimbus-training-corpus.md docs/watchman-open-pr-harvest.md pyproject.toml` |
| Default mock-provider smoke | Pass, saved with provider readiness and ambiguous protected workflow fail-closed evidence | `introspection/data/reports/aegis_default_mock_provider_smoke_ambiguous_protected_v1.json` |
| Default mock-provider NIMBUS/DP-HONEY refresh smoke | Pass, saved after NIMBUS grouped-CV, DP-HONEY scanner-eval, and tool-argument leakage-accounting updates. `/ready` reports `dp_honey_status=ready`, `nimbus_status=deterministic_beta`; benign protected slot allows with `honeytoken_substituted`; planted canary in tool arguments is blocked before provider dispatch with `tool_call_canary=escalate` and `nimbus_tool_egress=block`; partial leak sanitizes before block threshold. | `introspection/data/reports/aegis_default_mock_provider_smoke_nimbus_dp_honey_refresh_v2.json`, SHA-256 `2ad28da2089f682bdef4fbc1e6745e029141ce7f6e711de6ab388e0fb4e74fb1`; audit JSONL `introspection/data/reports/aegis_default_mock_provider_smoke_nimbus_dp_honey_refresh_audit_v2.jsonl`, SHA-256 `e3fe46d844edf315b9ce34d847ff3c7b3d1bc7f73c817f2d79f125b415557134` |
| Default mock smoke audit JSONL | Pass, redacted, 5 events | `introspection/data/reports/aegis_default_mock_provider_smoke_ambiguous_protected_audit_v1.jsonl` |
| Default mock smoke offline audit explanation | Pass, provider skipped before completion with 9-stage reconstruction | `introspection/data/reports/aegis_default_mock_provider_smoke_ambiguous_protected_audit_explain_v1.json`, SHA-256 `0b4132257f1e97d87b5ff6f8f8fae9ed667341f4890082cd5c105d8487242eb2` |
| Default mock smoke runtime-evidence artifact | Pass, saved with `aegis.audit_runtime_evidence/v1` audit evidence | `introspection/data/reports/aegis_default_mock_provider_smoke_runtime_evidence_v1.json`, SHA-256 `20d7a4c45d23c0838d1a13209597e656f9c15f46077705b414b34032c377c22d` |
| Default mock smoke runtime-evidence audit JSONL | Pass, redacted, 5 events, raw smoke secret and credential placeholders absent | `introspection/data/reports/aegis_default_mock_provider_smoke_runtime_evidence_audit_v1.jsonl`, SHA-256 `da9d6f720bf33897b3481ce9fa2d165b61d788f55085c86887ebec078cf0d13b` |
| Default mock smoke runtime-evidence audit explanation | Pass, provider skipped before completion, fail-closed event recorded, 9-stage reconstruction | `introspection/data/reports/aegis_default_mock_provider_smoke_runtime_evidence_audit_explain_v1.json`, SHA-256 `82c8a91a0e4e8930912c84180b826d4dbc4af4fca1735ed9eee04ded5cd7b2bc` |
| Mock smoke artifact SHA-256 | Recorded | `882e4c15f608beb2cc5889978adcb7d5f0c1f22cae442d8e983ce8a83cbcda2a` |
| Mock smoke audit SHA-256 | Recorded | `d53fc0a0f93a63a2d1ae894877301a935f366b431fd0e9717f17fab953bcbe6a` |
| Loopback real-provider config preflight | Pass, saved, no network attempted | `introspection/data/reports/aegis_loopback_real_provider_preflight_v3.json`, SHA-256 `309a229f313f27b633dcf031813935c0f8a57d6362c79feac65a23ac2b78a793` |
| Loopback real-provider smoke | Pass, saved, adapter path exercised with mock controls disabled | `introspection/data/reports/aegis_loopback_real_provider_smoke_v3.json`, SHA-256 `70b1c95291ffba24fd900a0751889fc4e8a7d7997fd2f3c40ae30825ad66f433` |
| Loopback real-provider audit JSONL | Pass, redacted, 2 events with `aegis.audit_runtime_evidence/v1` receipts | `introspection/data/reports/aegis_loopback_real_provider_smoke_audit_v3.jsonl`, SHA-256 `d4df7f49771c9c419b8b3b661e70390c8d13d7d3c1bb0448960bcf2b8f2f2ea4` |
| Loopback provider request receipt | Pass, 1 provider request, bearer token matched, forbidden markers absent | `introspection/data/reports/aegis_loopback_openai_provider_request_log_v3.jsonl`, SHA-256 `71c85b820ac8bb3ddc488bce3bdd01d005fe8f991f484ac1d940cae9161cbe15` |
| Loopback provider evidence verification | Pass, machine-verified local adapter evidence chain with audit runtime-evidence receipts | `introspection/data/reports/aegis_loopback_real_provider_smoke_evidence_verification_v2.json`, SHA-256 `964ea69a8a11a96655ce88d1c3a207f0c9ea1ebad162c5cbc5c2c3bf7cffaccd` |
| Real-provider config preflight | Not ready in current shell, saved, no network attempted | `introspection/data/reports/aegis_real_provider_preflight_current_env_v1.json`, SHA-256 `9f76f5429e8d1c0813ecc1b91d54dad33b747fd974d6a5ead6398102d4c502e5` |
| Deterministic NIMBUS labeled eval | Pass, deterministic beta only: TP=1, TN=1, FP=0, FN=0 | `introspection/data/reports/aegis_nimbus_deterministic_beta_eval_v1.json`, SHA-256 `61f797ebda1c8749a8d6df66fca4602e3ee52a18a3a25e7dbecbc66e8030d380` |
| Learned NIMBUS training corpus scaffold | Pass, not promotable: calibration and sealed-holdout profiles each have 19 records, 9 split groups, 7 labels, named scenario-family coverage gates passing, and distinct synthetic secret contexts/session groups. | calibration `introspection/data/reports/aegis_nimbus_training_corpus_v0.jsonl`, SHA-256 `8cc69fdc4b70f96fd095c9562b9165211555e1be083b2c11cb30c8062e11d1b6`; manifest `introspection/data/reports/aegis_nimbus_training_corpus_manifest_v0.json`, SHA-256 `3cf17fbdf0727f4cf6aa8511f364ca6fb7cad924dc9466e7c813261233a1571e`; sealed holdout `introspection/data/reports/aegis_nimbus_sealed_holdout_corpus_v0.jsonl`, SHA-256 `8f2fb1ed2ebde95bf6cdbf5e158b73e08bf046315e1bfbc71217e1ee37bd3507`; sealed manifest `introspection/data/reports/aegis_nimbus_sealed_holdout_corpus_manifest_v0.json`, SHA-256 `9f7e3efc83e435f0fd6cde427fcc29e7d3669ccbb0670f51a4fd0258575406f8` |
| Offline NIMBUS InfoNCE scaffold | Pass, not promotable. Training-diagnostic eval explicitly marks `training_eval_reused=true`: attack top-1=0.7857, turn FP=0, turn FN=3, turn FP rate=0.0, turn FN rate=0.2143, session FP=0, session FN=0. Grouped CV and sealed holdout both report attack top-1=0.7857, turn FP=0, turn FN=3, turn FP rate=0.0, turn FN rate=0.2143, session FP=0, session FN=0; partial-drip turns still miss even though held-out sessions score positive cumulatively. | model `introspection/data/reports/aegis_nimbus_infonce_model_v0.json`, SHA-256 `819467feca9650fc37cc96a6c83035da3fb6ebacb773deed9bb3ee02f426ab06`; eval `introspection/data/reports/aegis_nimbus_infonce_eval_v0.json`, SHA-256 `5b275831491b156d12fb585b46541109305b37d6a2b13ccc20f0e91d51894afc`; markdown `introspection/data/reports/aegis_nimbus_infonce_eval_v0.md`, SHA-256 `a55e5e2f707c2045e0520c16d943a7c4996e3fb6dd3fb62eb81c281f6b2f2404`; grouped CV `introspection/data/reports/aegis_nimbus_infonce_grouped_cv_v0.json`, SHA-256 `a20b1793b6a89868eb7b53e9d86952b7d1d68f602d67747c54e2513133c37c16`; sealed eval `introspection/data/reports/aegis_nimbus_infonce_sealed_holdout_eval_v0.json`, SHA-256 `b57be94e7256675dd8f9d95eff885c7e801d9261fc42eb8a8a86cb36ce8a7e0a`; sealed markdown `introspection/data/reports/aegis_nimbus_infonce_sealed_holdout_eval_v0.md`, SHA-256 `87cf2282ab1fcc4d89f426139e5086c4acf77453e2a7311942f154225577bda3` |
| DP-HONEY scanner held-out eval | Pass for registry-shaped scanner evidence with split-conformal confidence calibration: target alpha=0.1, calibration benign count=16, threshold=0.35, recommended minimum confidence=`medium`, 500 positive examples, 10 benign negatives, TP=500, TN=10, FP=0, FN=0, FP rate=0.0, FN rate=0.0, one-token detection true. | `introspection/data/reports/dp_honey_scanner_eval_v1.json`, SHA-256 `a2f0bf121607e13ca320c0b0bc1f46968a02c3fff25678bdf414349a63b3f13e` |
| DP-HONEY paper-faithfulness evidence report | Pass as operational beta, not promotable as paper-faithful+: 7 checklist requirements met, 1 partial, 0 missing. Met: DP-noised bigram provenance, split-conformal scanner calibration, scanner FP/FN reporting, gateway substitution and ledger evidence, output leak detection, serialized tool-argument canary detection plus NIMBUS pre-dispatch accounting, redacted audit. Partial: statistical distinguisher realism. | `introspection/data/reports/dp_honey_paper_evidence_v2.json`, SHA-256 `2c96ee6a847c4965a6d39b27d5e6c95b4c18d4e7d4f2893b18a340c8a536540e` |
| Qwen3-4B certification verification | Certified | `introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_linear_promoted_runtime_mps_receipt_recheck_certification_verification_current_turn_v1.json` |
| Qwen3-4B strict gateway smoke artifact | Existing artifact, not re-run this pass | `introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_gateway_smoke_mps_receipt_recheck_v1.json` |
| Real-provider live smoke | Not run | `AEGIS_OPENAI_API_KEY` and provider env are unset in the current shell |

## Component Readiness

| Component | Current status | What works | Release blocker or gap |
| --- | --- | --- | --- |
| CIFT strict Qwen3-4B/MPS | Certified reference | Verification report binds `Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c`, MPS, hidden size 2560, 36 layers, tokenizer/special-token/chat-template hashes, runtime SHA, feature key `selected_choice_window_layer_21`, and release-gate eligibility. | Re-run strict gateway/sidecar smoke in the target release environment before external release signoff. |
| CIFT general certification workflow | Real but model-specific | The workflow supports model-specific certification; unsupported models fail the support claim until their own calibration, sealed holdout, live runtime, gateway smoke, and release gate pass. | Do not claim universal model support. A second model run is useful portability evidence but not required for the Qwen3-4B reference claim. |
| DP-HONEY substitution/scanner | Paper-aligned operational beta | Default smoke shows `credential_slot_status=honeytoken_substituted` on benign protected-slot flow, with canary registration feeding canary detectors and NIMBUS. Scanner evidence reports TP=500, TN=10, FP=0, FN=0 with raw values absent from the report and split-conformal confidence calibration for scanner findings. Generator reports include DP-noised bigram parameters, registry version, and spec hash. Tool-argument smoke blocks both raw tool payloads and planted-canary tool payloads before provider dispatch, with NIMBUS pre-dispatch leakage bits recorded. The paper evidence checklist is 7 met, 1 partial, 0 missing. | Not a full autonomous credential broker. A stronger statistical-distinguisher realism gate is still missing, so do not claim full DP-HONEY paper-faithful+. Real honeytoken registration/substitution with external secret managers remains a production integration task. |
| Credential-slot detection v1 | Bounded deterministic v1 | Supports explicit slots, tool schemas, secret-like tool fields, env/config-shaped inputs, and protected workflow metadata. Raw secret presence is distinguished from credential-needed substitution. Ambiguous protected workflows fail closed with structured `credential_slot_status=ambiguous_protected_workflow` error details. | No broad semantic credential-need inference. Ambiguous protected workflows need more operator policy and examples before broad release. |
| Provider gateway | Hardened local/real-provider boundary | Gateway wires DP-HONEY, CIFT, provider egress guard, provider, canary detectors, NIMBUS, policy, and audit. `/ready` reports provider name and mock-control state. Mock controls are rejected for real providers. Provider URLs now require HTTPS unless loopback HTTP. Loopback OpenAI-compatible smoke proves the non-mock adapter path locally, and `aegis-provider-smoke-verify` machine-checks the saved evidence chain plus audit runtime-evidence receipts. | Needs a credentialed external-provider smoke run. |
| Provider egress guard | RC-quality for known sensitive payloads | Saved mock and loopback real-provider smokes show `provider_status=skipped`, `provider_reason=pre_generation_policy_block`, and `guard_reason=blocked_sensitive_value_before_provider_egress` for raw-secret tool payloads. Loopback provider receipt has exactly one benign provider request and `forbidden_substring_present=false`; the evidence verifier rejects artifacts if attack traffic reaches the provider or if audit runtime-evidence receipts are missing. | Expand sink coverage and provider-specific payload shapes over time. |
| Canary detection | Deterministic positive controls work | Text/encoded canary detectors run post-provider; smoke exercises encoded leak and metadata-slot canary leak paths. | Production canary registry/storage and rotation policy remain bounded/local. |
| NIMBUS | Deterministic beta plus non-promotable learned scaffold | Canary-aware deterministic critic accumulates exact, encoded, and partial leakage risk. Runtime evidence now exposes `paper_faithful_learned_critic=false` and `promotion_status=deterministic_canary_beta`. `aegis-nimbus-training-corpus` covers benign, exact canary, encoded, partial/multi-turn, paraphrased, tool-output, and delayed leak cases with split group keys. `aegis-nimbus-eval-infonce` blocks same-corpus eval unless `--allow-training-eval` is explicitly set and emits turn-level/session-level grouped-CV and sealed-holdout evidence. | Paper-faithful learned/session leakage critic is not implemented or wired into runtime. Grouped CV and sealed holdout are still small and lexical: turn FP=0, turn FN=3, turn FNR=0.2143, session FP=0, session FN=0. It still needs a larger corpus, a runtime learned adapter, live FN/FP metrics, and promotion evidence. |
| Observability/audit | Evidence-grade local durability | JSONL audit sink writes redacted local artifacts. Audit records include `aegis.audit_runtime_evidence/v1` with policy mode, final action, provider state, credential-slot status, detector versions, detector latencies, whitelisted artifact hashes, CIFT summary fields when present, fail-closed events, and total latency. `/audit/explain` and `aegis-audit-explain` reconstruct normalize, DP-HONEY, CIFT, provider egress guard, provider, canary, NIMBUS, policy, and audit stages. Saved audit does not contain the raw secret marker and does include `[REDACTED_SENSITIVE]`. | Not operator-grade durable storage yet: no retention policy, index, dashboard, authz, or multi-process store. |
| Real-provider mode | Configured, hardened, loopback-proven locally, unproven against external credentials | README documents no-network provider preflight, loopback OpenAI-compatible smoke, real-provider startup, local evidence verification, and external-provider smoke. URL validation rejects public plaintext HTTP, embedded credentials, query strings, fragments, missing hosts, and non-HTTP schemes. Current-shell external preflight reports provider env unset and attempts no network access. | No external provider credentials in current shell, so credentialed provider completion and provider-side latency/error evidence are missing. |

## Acceptance Check

| Requirement | Status | Evidence |
| --- | --- | --- |
| Existing strict Qwen3-4B/MPS sentinel smoke still passes | Partially current | Certification verifies now; existing strict gateway smoke artifact is `status=ok`. A fresh strict sidecar/gateway smoke was not re-run in this pass. The default mock-provider smoke was freshly re-run and passed after NIMBUS/DP-HONEY updates. |
| Documented default mock-provider smoke command | Met | README documents `uv run aegis-proxy-smoke --url http://127.0.0.1:8000 --timeout 5 --output introspection/data/reports/aegis_default_mock_provider_smoke_ambiguous_protected_v1.json`. |
| Documented real-provider smoke config path | Met | README documents `aegis-provider-preflight --require-real-provider`, the loopback OpenAI-compatible proof path, `AEGIS_PROVIDER=openai_compatible`, and `--provider-mode real-provider --output introspection/data/reports/aegis_real_provider_smoke_v1.json`. |
| Loopback real-provider evidence is machine-verifiable | Met for local adapter evidence | `aegis-provider-smoke-verify` validates preflight, smoke, provider request-log, and audit JSONL agreement, including `aegis.audit_runtime_evidence/v1` benign/blocked trace receipts, and emits `aegis.provider_smoke_evidence/v1`. Current artifact passes with SHA-256 `964ea69a8a11a96655ce88d1c3a207f0c9ea1ebad162c5cbc5c2c3bf7cffaccd`. |
| Durable audit writes redacted local artifacts | Met for local JSONL | Saved mock audit JSONL has 5 events and saved loopback audit JSONL has 2 events. Both include `[REDACTED_SENSITIVE]` and omit the synthetic raw secret marker used by smoke. The fresh mock and loopback runtime-evidence audit JSONL artifacts add `aegis.audit_runtime_evidence/v1` and also omit raw credential placeholders. |
| Trace can be reconstructed stage by stage | Met for local smoke | Saved mock and loopback smokes report `/ready` provider identity and `/audit/explain` for `smoke-egress-guard-trace`, `stage_count=9`, provider skipped. The saved offline `aegis-audit-explain` artifact reconstructs the same blocked provider-egress trace from JSONL and exposes the fail-closed runtime-evidence summary. |
| Deterministic credential-slot coverage beyond explicit metadata | Met for v1 | Tests cover tool schema, env/config-shaped message references, secret-like tool arguments, and structured fail-closed evidence for ambiguous protected workflows. |
| NIMBUS upgraded or explicitly labeled beta with plan | Met as beta plus offline scaffold | Capabilities and runtime evidence label deterministic beta; labeled deterministic eval reports TP/TN/FP/FN separately. The learned-NIMBUS calibration and sealed-holdout corpora each cover 19 records and 9 grouped sessions across 7 labels. Grouped-CV and sealed-holdout evidence report turn FP=0, turn FN=3, turn FNR=0.2143, session FP=0, session FN=0, session FNR=0.0. Learned NIMBUS remains non-promotable and is not wired into runtime. |
| DP-HONEY quantitative scanner evidence exists | Met as paper-aligned beta | `dp_honey_scanner_eval_v1.json` reports TP=500, TN=10, FP=0, FN=0, FP rate=0.0, FN rate=0.0, raw values absent, and split-conformal confidence calibration with alpha=0.1. `dp_honey_paper_evidence_v2.json` records the paper-faithfulness checklist as 7 met, 1 partial, 0 missing, with promotion disabled until statistical distinguisher realism is closed. |
| Final report avoids production overclaim | Met | This report explicitly marks real-provider live evidence, learned NIMBUS, broad credential inference, and operator-grade observability as not complete. |

## Commands To Close The Remaining Release Evidence Gap

Run these only with an intended provider credential and billing context:

```bash
AEGIS_PROVIDER=openai_compatible \
AEGIS_OPENAI_BASE_URL=https://api.openai.com \
AEGIS_OPENAI_API_KEY="$OPENAI_API_KEY" \
AEGIS_OPENAI_MODEL=gpt-4.1-mini \
uv run aegis-provider-preflight \
  --require-real-provider \
  --output introspection/data/reports/aegis_real_provider_preflight_v1.json
```

Then start the gateway:

```bash
AEGIS_PROVIDER=openai_compatible \
AEGIS_OPENAI_BASE_URL=https://api.openai.com \
AEGIS_OPENAI_API_KEY="$OPENAI_API_KEY" \
AEGIS_OPENAI_MODEL=gpt-4.1-mini \
AEGIS_AUDIT_JSONL_PATH=introspection/data/reports/aegis_real_provider_smoke_audit.jsonl \
uv run aegis-proxy --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
uv run aegis-proxy-smoke \
  --url http://127.0.0.1:8000 \
  --timeout 30 \
  --provider-mode real-provider \
  --output introspection/data/reports/aegis_real_provider_smoke_v1.json
```

The signoff evidence should include:

- `/ready` status with provider name, mock-control state, and CIFT capability.
- benign allow with provider completion.
- provider egress guard block before provider call.
- CIFT pre-generation block if strict self-hosted CIFT is enabled.
- durable audit JSONL redaction check.
- `/audit/explain` trace reconstruction for the blocked egress trace.
- explicit statement of model/provider used and whether any mock-only probes were skipped.
