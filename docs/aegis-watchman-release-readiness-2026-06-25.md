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
| Default mock-provider NIMBUS/DP-HONEY refresh smoke | Pass, saved after NIMBUS grouped-CV and DP-HONEY scanner-eval updates. `/ready` reports `dp_honey_status=ready`, `nimbus_status=deterministic_beta`; benign protected slot allows with `honeytoken_substituted`; partial leak sanitizes before block threshold. | `introspection/data/reports/aegis_default_mock_provider_smoke_nimbus_dp_honey_refresh_v1.json`, SHA-256 `bb5c67ff0800f09ec6b0fc34fdfcc05b53998bb207129f888e807f696019fc16`; audit JSONL `introspection/data/reports/aegis_default_mock_provider_smoke_nimbus_dp_honey_refresh_audit_v1.jsonl`, SHA-256 `76013f8c304d6ab1f9debf13c19753b72197bdaa7f896100b1627944c5b7647d` |
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
| Learned NIMBUS training corpus scaffold | Pass, not promotable: 14 records, 7 split groups, 7 labels, named scenario-family coverage gate passing | `introspection/data/reports/aegis_nimbus_training_corpus_v0.jsonl`, SHA-256 `1db1e9f2c59ff842dd8444368c14919458ecc3fa925d86bbe3762d54ff0ff25c`; manifest `introspection/data/reports/aegis_nimbus_training_corpus_manifest_v0.json`, SHA-256 `a734315d888119efc0c59bae1cf82a8a77b4c4201c4807377fb61be348c11746` |
| Offline NIMBUS InfoNCE scaffold | Pass, not promotable. Training-diagnostic eval explicitly marks `training_eval_reused=true`: attack top-1=0.7778, FP=0, FN=2, FP rate=0.0, FN rate=0.2222. Grouped CV: attack top-1=0.6667, FP=0, FN=3, FP rate=0.0, FN rate=0.3333; encoded and partial-drip holdouts fail. | model `introspection/data/reports/aegis_nimbus_infonce_model_v0.json`, SHA-256 `3ae40fba04f803e18bd20cd65a3c06394248948c5b860fbf861ec9df3335ef4e`; eval `introspection/data/reports/aegis_nimbus_infonce_eval_v0.json`, SHA-256 `9cf24c99cadf0e54458f05bb3dc340344bf85de555c4264968e605bca4808e96`; markdown `introspection/data/reports/aegis_nimbus_infonce_eval_v0.md`, SHA-256 `27a6ac0a0a88de2ae41fc7f1afac74144442c2957618be85ee54500db1bd9458`; grouped CV `introspection/data/reports/aegis_nimbus_infonce_grouped_cv_v0.json`, SHA-256 `8d3d74330be4da508a0b697ad4495ea8944be4b53cc1eb06d426b9abfd471ed5` |
| DP-HONEY scanner held-out eval | Pass for registry-shaped scanner evidence with split-conformal confidence calibration: target alpha=0.1, calibration benign count=16, threshold=0.35, recommended minimum confidence=`medium`, 500 positive examples, 10 benign negatives, TP=500, TN=10, FP=0, FN=0, FP rate=0.0, FN rate=0.0, one-token detection true. | `introspection/data/reports/dp_honey_scanner_eval_v1.json`, SHA-256 `a2f0bf121607e13ca320c0b0bc1f46968a02c3fff25678bdf414349a63b3f13e` |
| Qwen3-4B certification verification | Certified | `introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_linear_promoted_runtime_mps_receipt_recheck_certification_verification_current_turn_v1.json` |
| Qwen3-4B strict gateway smoke artifact | Existing artifact, not re-run this pass | `introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_gateway_smoke_mps_receipt_recheck_v1.json` |
| Real-provider live smoke | Not run | `AEGIS_OPENAI_API_KEY` and provider env are unset in the current shell |

## Component Readiness

| Component | Current status | What works | Release blocker or gap |
| --- | --- | --- | --- |
| CIFT strict Qwen3-4B/MPS | Certified reference | Verification report binds `Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c`, MPS, hidden size 2560, 36 layers, tokenizer/special-token/chat-template hashes, runtime SHA, feature key `selected_choice_window_layer_21`, and release-gate eligibility. | Re-run strict gateway/sidecar smoke in the target release environment before external release signoff. |
| CIFT general certification workflow | Real but model-specific | The workflow supports model-specific certification; unsupported models fail the support claim until their own calibration, sealed holdout, live runtime, gateway smoke, and release gate pass. | Do not claim universal model support. A second model run is useful portability evidence but not required for the Qwen3-4B reference claim. |
| DP-HONEY substitution/scanner | Paper-aligned operational beta | Default smoke shows `credential_slot_status=honeytoken_substituted` on benign protected-slot flow, with canary registration feeding canary detectors and NIMBUS. Scanner evidence reports TP=500, TN=10, FP=0, FN=0 with raw values absent from the report and split-conformal confidence calibration for scanner findings. Generator reports include DP-noised bigram parameters, registry version, and spec hash. | Not a full autonomous credential broker. A stronger statistical-distinguisher realism gate is still missing, so do not claim full DP-HONEY paper-faithful+. Real honeytoken registration/substitution with external secret managers remains a production integration task. |
| Credential-slot detection v1 | Bounded deterministic v1 | Supports explicit slots, tool schemas, secret-like tool fields, env/config-shaped inputs, and protected workflow metadata. Raw secret presence is distinguished from credential-needed substitution. Ambiguous protected workflows fail closed with structured `credential_slot_status=ambiguous_protected_workflow` error details. | No broad semantic credential-need inference. Ambiguous protected workflows need more operator policy and examples before broad release. |
| Provider gateway | Hardened local/real-provider boundary | Gateway wires DP-HONEY, CIFT, provider egress guard, provider, canary detectors, NIMBUS, policy, and audit. `/ready` reports provider name and mock-control state. Mock controls are rejected for real providers. Provider URLs now require HTTPS unless loopback HTTP. Loopback OpenAI-compatible smoke proves the non-mock adapter path locally, and `aegis-provider-smoke-verify` machine-checks the saved evidence chain plus audit runtime-evidence receipts. | Needs a credentialed external-provider smoke run. |
| Provider egress guard | RC-quality for known sensitive payloads | Saved mock and loopback real-provider smokes show `provider_status=skipped`, `provider_reason=pre_generation_policy_block`, and `guard_reason=blocked_sensitive_value_before_provider_egress` for raw-secret tool payloads. Loopback provider receipt has exactly one benign provider request and `forbidden_substring_present=false`; the evidence verifier rejects artifacts if attack traffic reaches the provider or if audit runtime-evidence receipts are missing. | Expand sink coverage and provider-specific payload shapes over time. |
| Canary detection | Deterministic positive controls work | Text/encoded canary detectors run post-provider; smoke exercises encoded leak and metadata-slot canary leak paths. | Production canary registry/storage and rotation policy remain bounded/local. |
| NIMBUS | Deterministic beta plus non-promotable learned scaffold | Canary-aware deterministic critic accumulates exact, encoded, and partial leakage risk. Runtime evidence now exposes `paper_faithful_learned_critic=false` and `promotion_status=deterministic_canary_beta`. `aegis-nimbus-training-corpus` covers benign, exact canary, encoded, partial/multi-turn, paraphrased, tool-output, and delayed leak cases with split group keys. `aegis-nimbus-eval-infonce` blocks same-corpus eval unless `--allow-training-eval` is explicitly set and can emit grouped-CV evidence. | Paper-faithful learned/session leakage critic is not implemented or wired into runtime. Grouped CV remains weak: FP=0 but FN=3 and FN rate=0.3333. It still needs a larger corpus, sealed holdout, learned runtime adapter, live FN/FP metrics, and promotion evidence. |
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
| NIMBUS upgraded or explicitly labeled beta with plan | Met as beta plus offline scaffold | Capabilities and runtime evidence label deterministic beta; labeled deterministic eval reports TP/TN/FP/FN separately. The learned-NIMBUS corpus now covers 7 scenario families, and grouped-CV evidence reports FP=0, FN=3, FNR=0.3333. Learned NIMBUS remains non-promotable and is not wired into runtime. |
| DP-HONEY quantitative scanner evidence exists | Met as paper-aligned beta | `dp_honey_scanner_eval_v1.json` reports TP=500, TN=10, FP=0, FN=0, FP rate=0.0, FN rate=0.0, raw values absent, and split-conformal confidence calibration with alpha=0.1. |
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
