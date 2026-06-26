# Aegis

Aegis is a runtime credential-defense spine for LLM agents. It is designed to
sit between agent traffic and model/tool execution, normalize each turn, run
independent detector stages, apply policy in one place, and record auditable
events.

The repository is intentionally small right now. The current code establishes
the shared contracts and quality gates that future DP-HONEY, CIFT, NIMBUS,
proxy, SDK, dashboard, and evaluation work will plug into.

## Current Status

The first runtime spine is implemented and CI-enforced:

- Typed request, detector, policy, capability, and audit contracts.
- A minimal orchestrator that normalizes a turn, runs detector stages, calls a
  mock model provider, applies policy, and writes an audit event.
- A honeytoken ledger that replaces credential placeholders with registered
  canaries and emits `SensitiveSpan` metadata without exposing real secrets.
- First detector seams:
  - `ActivationUnavailableDetector` for explicit CIFT capability reporting in
    black-box/mock mode.
  - `NoopCanaryDetector` as the unconfigured DP-HONEY/text-canary boundary.
  - `TextCanaryDetector` for exact post-generation detection of registered
    honeytoken leaks.
  - `EncodedCanaryDetector` for base64, hex, ROT13, leet, reverse,
    fragmentation, and partial-overlap canary leakage.
- A fixture-backed `cift_selector_probe_v0` candidate monitor replay path that
  consumes promoted calibrated CIFT scores without importing research code.
- A runtime-native `aegis.detectors.cift_runtime.CiftRuntimeDetector` that loads
  an exported JSON linear CIFT model artifact, consumes feature vectors from
  `NormalizedTurn.metadata`, and emits active, degraded, or unavailable CIFT
  evidence without importing the introspection package.
- A `TurnAnnotator` seam that can attach derived metadata before pre-generation
  detectors run, plus a CIFT feature-vector annotator that preserves the
  runtime/import boundary while preparing self-hosted activation features.
- A CIFT window selector that treats selected-choice readout geometry as
  primary coverage and payload/query readout as degraded fallback evidence.
- A runtime-native `NimbusDetector` with a canary-aware critic that estimates
  exact, encoded, and partial canary leakage bits into a per-session budget.
- A development OpenAI-compatible proxy surface for `/health`,
  `/v1/chat/completions`, and `/audit/recent`, with a mock provider by
  default and an OpenAI-compatible HTTP provider adapter behind env config.
- Mandatory CI gates for linting, formatting, strict typing, import-boundary
  checks, and tests with coverage.

This is not yet a production security system. It is the enforced skeleton that
keeps future detector and proxy work compatible.

For the current hardening assessment and component-by-component release table,
see [`docs/aegis-watchman-release-readiness-2026-06-25.md`](docs/aegis-watchman-release-readiness-2026-06-25.md).

## Runtime Shape

```text
chat request
  -> NormalizedTurn
  -> turn annotators
  -> pre-generation detectors
  -> provider egress guard
  -> model provider
  -> post-generation detectors
  -> session detectors
  -> policy engine
  -> audit sink
  -> response
```

The core invariant is:

```text
Detectors produce evidence. Policy makes decisions. Audit records both.
```

Detectors return `DetectorResult` values. They do not call each other and do not
emit final enforcement decisions. The policy layer is the only layer that emits
`PolicyDecision`.

See [docs/aegis-runtime-spine.md](docs/aegis-runtime-spine.md) for the runtime
contract details.

## Quickstart

Install the project and development dependencies with `uv`:

```bash
uv sync --extra dev
```

Run the full local quality gate:

```bash
make quality
```

Run the built-in demo scenarios:

```bash
uv run python scripts/run_demo.py
```

Run the development HTTP proxy:

```bash
uv run aegis-proxy --host 127.0.0.1 --port 8000
```

The development proxy exposes the current OpenAI-compatible mock surface:

```text
GET  /health
GET  /ready
GET  /aegis/capabilities
POST /v1/chat/completions
GET  /audit/recent
GET  /audit/explain
POST /test/reset
POST /test/seed-canary   # mock provider only
```

Run the local operator console after the gateway is running:

```bash
uv run aegis-console \
  --gateway-url http://127.0.0.1:8000 \
  --host 127.0.0.1 \
  --port 8780 \
  --smoke-report introspection/data/reports/aegis_default_mock_provider_smoke_runtime_evidence_v1.json \
  --sample-audit-jsonl introspection/data/reports/aegis_default_mock_provider_smoke_runtime_evidence_audit_v1.jsonl
```

Open `http://127.0.0.1:8780` to see gateway readiness, protection state,
active model/CIFT certification binding, recent allow/block decisions, detector
activity, NIMBUS critic kind/promotion status, and setup commands. The optional
sample audit path is used only when the live gateway audit is empty, so the
console can still show the strict smoke stage timeline without replaying it as
production evidence.

`GET /aegis/capabilities` returns the machine-readable development contract:
provider kind, whether mock controls are enabled, supported mock response modes,
route list, schema versions, and mock-only test controls. Redteam tooling
should call this route before assuming that `mock_response_mode` or
`/test/seed-canary` is accepted.

By default, `aegis-proxy` runs with the deterministic mock provider. To point
the development proxy at an OpenAI-compatible model endpoint, configure the
provider explicitly:

```bash
AEGIS_PROVIDER=openai_compatible \
AEGIS_OPENAI_BASE_URL=https://api.openai.com \
AEGIS_OPENAI_API_KEY="$OPENAI_API_KEY" \
AEGIS_OPENAI_MODEL=gpt-4.1-mini \
uv run aegis-provider-preflight \
  --require-real-provider \
  --output introspection/data/reports/aegis_real_provider_preflight_v1.json

AEGIS_PROVIDER=openai_compatible \
AEGIS_OPENAI_BASE_URL=https://api.openai.com \
AEGIS_OPENAI_API_KEY="$OPENAI_API_KEY" \
AEGIS_OPENAI_MODEL=gpt-4.1-mini \
uv run aegis-proxy --host 127.0.0.1 --port 8000
```

`AEGIS_OPENAI_BASE_URL` may be the service root, `/v1`, or the full
`/v1/chat/completions` endpoint. Non-local provider URLs must use `https`;
`http` is accepted only for loopback development providers such as
`127.0.0.1`, `localhost`, or `::1`, and URLs with embedded credentials,
query strings, or fragments are rejected. Mock controls such as `mock_response` and
`mock_response_mode` are accepted only when `AEGIS_PROVIDER=mock`; the proxy
rejects them for real providers so redteam-only controls cannot cross the
provider boundary.

The default smoke path is mock-provider and does not require network access:

```bash
uv run aegis-proxy-smoke \
  --url http://127.0.0.1:8000 \
  --timeout 5 \
  --output introspection/data/reports/aegis_default_mock_provider_smoke_ambiguous_protected_v1.json
```

To exercise the OpenAI-compatible adapter without external network access or
provider credentials, run the local loopback provider. This is still local
evidence, but it runs the gateway with `AEGIS_PROVIDER=openai_compatible`,
disables mock controls, requires a bearer token, and writes a redacted provider
request receipt showing whether forbidden synthetic markers reached the
provider.

```bash
uv run aegis-loopback-openai-provider \
  --host 127.0.0.1 \
  --port 8776 \
  --response-content "Loopback provider completed." \
  --request-log introspection/data/reports/aegis_loopback_openai_provider_request_log_v3.jsonl \
  --expected-bearer-token loopback-test-token \
  --forbidden-substring ghp_realLookingToolSecret1234567890 \
  --forbidden-substring fake-
```

In another terminal, validate the real-provider config without network I/O and
then start the gateway against that loopback endpoint:

```bash
AEGIS_PROVIDER=openai_compatible \
AEGIS_OPENAI_BASE_URL=http://127.0.0.1:8776/v1 \
AEGIS_OPENAI_API_KEY=loopback-test-token \
AEGIS_OPENAI_MODEL=loopback-model \
uv run aegis-provider-preflight \
  --require-real-provider \
  --output introspection/data/reports/aegis_loopback_real_provider_preflight_v3.json

AEGIS_PROVIDER=openai_compatible \
AEGIS_OPENAI_BASE_URL=http://127.0.0.1:8776/v1 \
AEGIS_OPENAI_API_KEY=loopback-test-token \
AEGIS_OPENAI_MODEL=loopback-model \
AEGIS_AUDIT_JSONL_PATH=introspection/data/reports/aegis_loopback_real_provider_smoke_audit_v3.jsonl \
uv run aegis-proxy --host 127.0.0.1 --port 8777
```

Then run real-provider smoke against the gateway:

```bash
uv run aegis-proxy-smoke \
  --url http://127.0.0.1:8777 \
  --timeout 10 \
  --provider-mode real-provider \
  --output introspection/data/reports/aegis_loopback_real_provider_smoke_v3.json
```

Finally, verify the saved loopback evidence chain. This verifier checks that
preflight, smoke, provider request-log, and audit JSONL artifacts agree; that
only the benign request reached the provider; that the raw credential-shaped
tool payload was blocked before provider generation; and that forbidden markers
were not persisted in saved evidence. It also requires
`aegis.audit_runtime_evidence/v1` receipts proving the benign provider
completion and pre-provider egress block.

```bash
uv run aegis-provider-smoke-verify \
  --preflight introspection/data/reports/aegis_loopback_real_provider_preflight_v3.json \
  --smoke introspection/data/reports/aegis_loopback_real_provider_smoke_v3.json \
  --provider-request-log introspection/data/reports/aegis_loopback_openai_provider_request_log_v3.jsonl \
  --audit-jsonl introspection/data/reports/aegis_loopback_real_provider_smoke_audit_v3.jsonl \
  --forbidden-marker ghp_realLookingToolSecret1234567890 \
  --forbidden-marker fake- \
  --output introspection/data/reports/aegis_loopback_real_provider_smoke_evidence_verification_v2.json
```

For an explicitly configured OpenAI-compatible provider, run the gateway with
real-provider env and use the real-provider smoke mode. This mode still checks
readiness, benign provider completion, CIFT pre-generation blocking when
configured, provider egress blocking before outbound provider calls, audit
readback, and trace explanation; it skips mock-only leak probes.

```bash
AEGIS_PROVIDER=openai_compatible \
AEGIS_OPENAI_BASE_URL=https://api.openai.com \
AEGIS_OPENAI_API_KEY="$OPENAI_API_KEY" \
AEGIS_OPENAI_MODEL=gpt-4.1-mini \
AEGIS_AUDIT_JSONL_PATH=/tmp/aegis-audit.jsonl \
uv run aegis-proxy --host 127.0.0.1 --port 8000

uv run aegis-proxy-smoke \
  --url http://127.0.0.1:8000 \
  --timeout 30 \
  --provider-mode real-provider \
  --output introspection/data/reports/aegis_real_provider_smoke_v1.json
```

Self-hosted CIFT requires a trusted activation extractor sidecar. The proxy does
not accept client-supplied `metadata.cift`; the sidecar is registered from env
and writes proxy-owned feature vectors and selected-choice readout metadata
before pre-generation detectors run:

```bash
export AEGIS_CIFT_EXTRACTOR_API_KEY="set-a-deployment-secret"
source introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_linear_promoted_runtime_mps_receipt_recheck_strict_deployment_env.sh
uv run aegis-proxy --host 127.0.0.1 --port 8000
```

The strict deployment env above identifies the current MPS-bound certified
runtime for `Qwen/Qwen3-4B` at immutable revision
`1cfa9a7208912126459214e8b04321603b3df60c`. The env materializer computes the
certification manifest/report and release-gate report SHA-256 values, runs the
hardened release gate, and refuses to emit startup env if the exact evidence
chain no longer verifies.
Production CIFT certification rejects mutable revisions such as `main`;
`source_revision` must be a resolved lowercase 40-character Git commit SHA or a
`sha256:<64 lowercase hex digest>` content revision.

Aegis CIFT supports model-specific certification for local models that expose
hidden states. `Qwen/Qwen3-4B` is the certified reference model. Other models
are unsupported until they pass their own calibration, sealed holdout, live
runtime, gateway smoke, and hardened release-gate certification. The nested
runtime-candidate promotion gate is intentionally scoped to candidate promotion;
final release eligibility comes only from the certification-bound release gate.

Use the local-model certification wrapper as the single operator entry point.
For the certified Qwen3-4B MPS reference, verify the existing model-bound
evidence chain without replaying offline evidence:

```bash
PYTHONPATH=src:introspection/src \
.venv-mps313/bin/python introspection/scripts/certify_cift_local_model.py verify-existing \
  --repository-root . \
  --runtime-model introspection/data/models/cift_qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_linear_promoted_runtime_mps_receipt_recheck_v1.json \
  --expected-runtime-sha256 b7efb486c369d0745533c49b608970f5e5b3a5c12a1ecee343856b13b4a02d60 \
  --certification-manifest introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_linear_promoted_runtime_mps_receipt_recheck_certification_workflow_v1.json \
  --certification-report introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_linear_promoted_runtime_mps_receipt_recheck_certification_workflow_run_v1.json \
  --certification-artifact-root . \
  --release-gate-report introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_linear_promoted_runtime_mps_receipt_recheck_release_gate_v1.json \
  --verification-report introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_linear_promoted_runtime_mps_receipt_recheck_certification_verification_v1.json \
  --certification-manifest-sha256 61fb5cc75d79dd1e6dddfa01e9c6cdfa87c5b19d0e909af5a97d09d58141a50f \
  --certification-report-sha256 374e8fc3823e4b1f149ed7c0ad16098c4037956bf47dd062a015692f33d5928c \
  --release-gate-report-sha256 306ad21e911087ecd1e25ff7dde7491f761351cb58bc52f5f23ac99c53c7d536 \
  --model-id Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --required-device mps \
  --expected-hidden-size 2560 \
  --expected-layer-count 36 \
  --expected-tokenizer-sha256 41e00eccf531cffc2e562d38bdd879d41e5044ea279af5b73c6a32aabcc8fe04 \
  --expected-special-tokens-sha256 edcb2fc2acbbe77f858a9c4fe51295ffdb84711efba5703ec5906b3d67282569 \
  --expected-chat-template-sha256 a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8 \
  --expected-feature-key selected_choice_window_layer_21 \
  --expected-pooling-method selected_choice_window \
  --expected-dtype-name device \
  --expected-detector-name cift_runtime \
  --expected-extractor-id trusted-activation-sidecar \
  --expected-feature-source self_hosted_activation_extractor \
  --expected-prompt-renderer aegis_trace_bridge_v1 \
  --expected-selected-choice-geometry semantic_indirection_v1 \
  --expected-selected-choice-readout-token-count 4
```

For a new hidden-state-capable model, the same wrapper discovers
model/tokenizer/template metadata, builds a model-bound workflow manifest, runs
or verifies the workflow, and fails release certification unless every required
artifact is materialized and the hardened gate passes:

```bash
PYTHONPATH=src:introspection/src \
.venv-mps313/bin/python introspection/scripts/certify_cift_local_model.py \
  --repository-root . \
  --certification-id <certification-id> \
  --model-id <model-id> \
  --revision <immutable-revision-or-content-hash> \
  --corpus <calibration-structured-prompts.jsonl> \
  --runtime-turns <sealed-runtime-turns.jsonl> \
  --fallback-runtime-model <fallback-runtime-artifact.json> \
  --output-dir introspection/data \
  --workflow-manifest <workflow-manifest.json> \
  --run-report <workflow-run-report.json> \
  --training-dataset-id <dataset-id> \
  --task safe_secret_vs_exfiltration \
  --positive-label exfiltration_intent \
  --behavior-id selected_choice_secret_exfiltration_intent \
  --behavior-description selected_choice_secret_exfiltration_intent \
  --layers <comma-separated-layer-indices> \
  --pooling selected_choice_window \
  --candidate-feature <selected-choice-feature-key> \
  --device <certified-device> \
  --prompt-renderer aegis_trace_bridge_v1 \
  --selected-choice-geometry semantic_indirection_v1 \
  --selected-choice-readout-token-count 4 \
  --dtype device \
  --metric-threshold 1.0 \
  --ablation-delta-threshold 0.0 \
  --created-at <iso-8601-timestamp> \
  --command-timeout-seconds 120 \
  --execute \
  --allow-sealed-holdout-execution \
  --template-value gateway_base_url=http://127.0.0.1:8000 \
  --template-value sidecar_base_url=http://127.0.0.1:9000 \
  --template-value gateway_model=mock-model \
  --template-value extractor_id=trusted-activation-sidecar
```

`AEGIS_CIFT_CERTIFICATION_MANIFEST_PATH` and
`AEGIS_CIFT_CERTIFICATION_REPORT_PATH` must match their pinned SHA-256 values,
bind the selected runtime artifact by SHA-256, and report
`certification_eligible=true` with a finite positive `command_timeout_seconds`;
otherwise the self-hosted CIFT profile fails at startup. Every release-required
artifact in the certification manifest must resolve under
`AEGIS_CIFT_CERTIFICATION_ARTIFACT_ROOT`, including activation tensors, probe
bundles, calibration reports, head-to-head evidence, gateway smoke, and
evidence-chain reports. Self-hosted CIFT fails closed when the artifact root is
not configured or when any materialized artifact is missing or hash-drifted. The
configured CIFT detector name must match the certified gateway smoke report.
`AEGIS_CIFT_REQUIRED_DEVICE` must match the model-specific certification device,
for example `mps` for the Qwen3-4B runtime above. Do not set
`AEGIS_CIFT_FALLBACK_MODEL_PATH` in strict self-hosted mode; startup rejects
fallback scoring because it is not production activation evidence.
`AEGIS_CIFT_SELECTED_CHOICE_READOUT_TOKEN_COUNT` must match the selected-choice
calibration contract, for example `4` for the promoted Qwen3-4B runtime above.

To regenerate the strict deployment env after rebuilding the certification
chain, run the materializer explicitly and commit/review the resulting env file:

```bash
PYTHONPATH=src:introspection/src \
.venv-mps313/bin/python introspection/scripts/materialize_cift_deployment_env.py \
  introspection/data/models/cift_qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_linear_promoted_runtime_mps_receipt_recheck_v1.json \
  --repository-root . \
  --certification-manifest introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_linear_promoted_runtime_mps_receipt_recheck_certification_workflow_v1.json \
  --certification-report introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_linear_promoted_runtime_mps_receipt_recheck_certification_workflow_run_v1.json \
  --certification-artifact-root . \
  --required-device mps \
  --expected-detector-name cift_runtime \
  --expected-extractor-id trusted-activation-sidecar \
  --expected-feature-source self_hosted_activation_extractor \
  --expected-selected-choice-readout-token-count 4 \
  --extractor-base-url http://127.0.0.1:9000 \
  --extractor-timeout-seconds 30.0 \
  --extractor-api-key-env-var AEGIS_CIFT_EXTRACTOR_API_KEY \
  --release-gate-report-output introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_linear_promoted_runtime_mps_receipt_recheck_release_gate_v1.json \
  --output introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_linear_promoted_runtime_mps_receipt_recheck_strict_deployment_env.sh
```

Run the release gate with the same certification binding before starting the
proxy. The env materializer above runs this check automatically; this explicit
command is useful for CI and manual audits:

```bash
PYTHONPATH=src:introspection/src \
.venv-mps313/bin/python introspection/scripts/check_cift_release_gate.py \
  introspection/data/models/cift_qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_linear_promoted_runtime_mps_receipt_recheck_v1.json \
  --repository-root . \
  --required-runtime-prevention-device mps \
  --certification-manifest introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_linear_promoted_runtime_mps_receipt_recheck_certification_workflow_v1.json \
  --certification-report introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_linear_promoted_runtime_mps_receipt_recheck_certification_workflow_run_v1.json \
  --certification-artifact-root . \
  --certification-manifest-sha256 61fb5cc75d79dd1e6dddfa01e9c6cdfa87c5b19d0e909af5a97d09d58141a50f \
  --certification-report-sha256 374e8fc3823e4b1f149ed7c0ad16098c4037956bf47dd062a015692f33d5928c \
  --expected-detector-name cift_runtime \
  --expected-extractor-id trusted-activation-sidecar \
  --expected-feature-source self_hosted_activation_extractor \
  --expected-selected-choice-readout-token-count 4
```

The sidecar endpoint is `POST /v1/cift/features`. It receives
`schema_version=aegis.cift_feature_extract_request/v1`, the requested
`feature_key`, and the normalized turn, then returns
`schema_version=aegis.cift_feature_extract_response/v1`, the same `feature_key`,
`feature_vector`, `selected_choice_readout_token_indices`, and
`model_attestation`. The attestation must use
`schema_version=aegis.cift_model_attestation/v1` and must match the certified
runtime artifact's `source_model_id`, `source_revision`, configured
`AEGIS_CIFT_REQUIRED_DEVICE`, prompt renderer, selected-choice geometry method,
and selected-choice readout token count; otherwise the gateway treats activation
extraction as unavailable and blocks before provider generation. If
selected-choice geometry or feature extraction is unavailable in the production
profile, CIFT fails closed before provider generation.

Before running live Qwen3-4B extraction or certification evidence, preflight the
same Python environment that will run the model. On this Mac, use the
MPS-capable environment and require the smoke tensor to allocate on `mps:0`:

```bash
PYTHONPATH=src:introspection/src \
.venv-mps313/bin/python introspection/scripts/check_cift_device_preflight.py \
  --device mps
```

The live introspection sidecar can then be run from the repository with that
same MPS-capable environment:

```bash
PYTHONPATH=src:introspection/src \
.venv-mps313/bin/python introspection/scripts/run_cift_extractor_sidecar.py \
  --model-id Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --device mps \
  --dtype device \
  --feature-key selected_choice_window_layer_21 \
  --selected-choice-readout-token-count 4 \
  --host 127.0.0.1 \
  --port 9000 \
  --api-key-env-var AEGIS_CIFT_EXTRACTOR_API_KEY
```

That sidecar runs the trusted model forward pass and extracts live hidden-state
features. It renders structured proxy turns into the same prompt format used by
the calibration bridge, then derives selected-choice readout tokens for the
semantic-indirection prompt family using the loaded tokenizer offsets and the
configured readout count. It does not trust client-supplied `metadata.cift`.
Explicit `--device mps` fails closed if Torch cannot allocate an MPS tensor; CPU
fallback is not acceptable evidence for an MPS-certified runtime.
When selected-choice geometry cannot be derived, it returns
`feature_vector=null`, which the production CIFT policy treats as a fail-closed
block. The sidecar attests the actual loaded model revision and selected device
plus its prompt renderer, selected-choice geometry method, and selected-choice
readout token count. It also attests the loaded hidden size, layer count,
tokenizer fingerprint, special-token map hash, and chat-template hash; these
must match the promoted runtime artifact before the gateway can treat the
features as certified.

After the sidecar and gateway are both running, use the CIFT-specific gateway
smoke before treating the deployment as operational. This smoke requires
self-hosted CIFT capabilities, sends semantic-indirection benign and
exfiltration-intent requests through the gateway, and passes only when CIFT is
active rather than degraded:

```bash
uv run aegis-proxy-cift-smoke \
  --url http://127.0.0.1:8000 \
  --sidecar-url http://127.0.0.1:9000 \
  --gateway-model mock-model \
  --report-id qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_gateway_smoke_mps_receipt_recheck_v1 \
  --timeout 120 \
  --detector-name cift_runtime \
  --sidecar-feature-key selected_choice_window_layer_21 \
  --expected-gateway-feature-source self_hosted_activation_extractor \
  --expected-extractor-id trusted-activation-sidecar \
  --expected-sidecar-model-id Qwen/Qwen3-4B \
  --expected-sidecar-revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --expected-sidecar-device mps \
  --expected-sidecar-hidden-size 2560 \
  --expected-sidecar-layer-count 36 \
  --expected-sidecar-tokenizer-fingerprint-sha256 41e00eccf531cffc2e562d38bdd879d41e5044ea279af5b73c6a32aabcc8fe04 \
  --expected-sidecar-special-tokens-map-sha256 edcb2fc2acbbe77f858a9c4fe51295ffdb84711efba5703ec5906b3d67282569 \
  --expected-sidecar-chat-template-sha256 a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8 \
  --selected-choice-readout-token-count 4 \
  --sidecar-api-key-env-var AEGIS_CIFT_EXTRACTOR_API_KEY \
  --output introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_gateway_smoke_mps_receipt_recheck_v1.json
```

The smoke report's device evidence comes from the live model loader, so a
configuration that requested MPS but fell back to CPU will not satisfy an
MPS-certified runtime.

Proxy-owned request errors use a stable envelope:

```json
{
  "error": {
    "schema_version": "aegis.proxy_error/v1",
    "code": "invalid_request",
    "message": "...",
    "details": {}
  }
}
```

Provider transport failures use the same envelope with status `502` and
`code=provider_error`. Aegis policy blocks still return HTTP `200`; the
enforcement decision lives in `aegis.policy_decision.final_action`. When policy
returns `sanitize`, `block`, or `escalate`, the proxy does not return raw model
output in `choices`; it returns sanitized output when available or the fixed
`[aegis output withheld]` message.

Run the proxy smoke test from another terminal after the development proxy is
listening:

```bash
uv run aegis-proxy-smoke \
  --url http://127.0.0.1:8000 \
  --timeout 5 \
  --output introspection/data/reports/aegis_default_mock_provider_smoke_ambiguous_protected_v1.json
```

The default mock-provider smoke command checks health, readiness, capability
discovery, a benign chat turn with an explicit protected credential slot, an
ambiguous protected workflow fail-closed error with structured credential-slot
details, an optional CIFT pre-generation exfiltration-intent block, a raw
credential-shaped tool-payload egress block, an encoded DP-HONEY leak turn, a
metadata-declared credential-slot leak turn, deterministic NIMBUS partial-leak
accounting, audit readback, and `/audit/explain` stage reconstruction.
Readiness validation includes provider name, mock-control state, and CIFT
capability mode. The canary/NIMBUS leak probes use
CIFT-compatible safe selected-choice wording and then force the mock provider
to leak, so a strict CIFT profile can still exercise post-generation leakage
detection. It exits nonzero if the gateway contract is broken.

When the gateway is running with the certified self-hosted CIFT profile, require
the integrated CIFT block check as part of the same sentinel smoke:

```bash
uv run aegis-proxy-smoke \
  --url http://127.0.0.1:8000 \
  --timeout 120 \
  --require-cift-pre-generation-block \
  --output introspection/data/reports/aegis_self_hosted_cift_smoke_v1.json
```

Send one local chat request:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mock-model",
    "messages": [{"role": "user", "content": "hello"}],
    "metadata": {"trace_id": "trace-local-1", "session_id": "session-local-1"}
  }'
```

Redteam scenarios should target the HTTP surface first. For pre-server tests,
use the mock proxy only inside Aegis unit tests. The separate redteam repo
should treat Aegis as a black-box HTTP target and inspect the assistant response
plus the `aegis` detector/policy/trace block.

The development proxy supports deterministic mock response controls through
request metadata:

```json
{
  "metadata": {
    "mock_response_mode": "leak_first_honeytoken"
  }
}
```

Supported modes are:

- `default` - return the default benign mock response.
- `echo_last_user` - return the last user message.
- `leak_first_honeytoken` - return the first fake honeytoken visible in the prompt.
- `base64_first_honeytoken` - return the first fake honeytoken as base64.
- `partial_first_honeytoken` - return a prefix of the first fake honeytoken.

Leak modes require a planted honeytoken visible to the current mock turn. Use a
credential placeholder in the scenario turn, declare an explicit credential slot
in request metadata, or use the mock-only `POST /test/seed-canary` route for
session-scoped dev setup.

To exercise the integrated DP-HONEY/canary path through the proxy, declare
explicit credential slots in request metadata:

```json
{
  "metadata": {
    "protected_workflow": true,
    "credential_slots": [
      {"slot_name": "repo_pat", "credential_type": "github_pat"}
    ]
  }
}
```

`protected_workflow=true` fails closed unless at least one concrete credential
slot is declared or deterministically inferred. Ambiguous protected workflows
return `credential_slot_status=ambiguous_protected_workflow` in the versioned
error details, with `fail_closed=true`. The proxy plants a registered DP-HONEY
canary for each resolved slot, adds audit-safe credential-slot evidence, and
never requires or persists a real credential value for this path.

You can also include credential placeholders in chat messages or top-level
`tool_calls` string arguments:

```text
{{CREDENTIAL:repo_pat:github_pat}}
```

Credential-need detection v1 is deterministic and bounded. In addition to
explicit metadata and placeholders, the proxy can infer credential need from:

- OpenAI-style `tools[].function.parameters.properties` entries with
  secret-like names such as `github_token`, `api_key`, `access_key`, `secret`,
  `password`, or explicit `x-aegis-credential*` schema annotations.
- env/config-shaped message references such as `OPENAI_API_KEY`.
- top-level `tool_calls` arguments with obvious secret-like field names and
  safe placeholder values such as `credential_handle`, `credential_ref`, or a
  slot-like identifier.

These inferred paths plant DP-HONEY canaries through the same ledger as explicit
slots. Raw credential-shaped values are still treated as real secrets and are
blocked by the provider egress guard before provider completion. Client-supplied
`metadata.secret_context_handle` is reserved; HTTP callers should declare
credential slots instead of activating NIMBUS provenance directly.

Credential-slot detection reports one of the bounded v1 statuses:
`no_credential_path`, `credential_needed`, `honeytoken_substituted`,
`real_secret_present`, or `ambiguous_protected_workflow`. The ambiguous status
appears only on the fail-closed error path.

The development proxy replaces placeholders with DP-HONEY canaries, attaches
`SensitiveSpan` metadata, blocks real credential-shaped tool/message egress
before provider completion, runs canary detectors when canaries exist, feeds the
same planted-canary registry to the deterministic beta NIMBUS critic, and
returns the detector outputs in the `aegis` block.

Generate DP-HONEY scanner evidence with separate false negative and false
positive rates:

```bash
uv run dp-honey eval-scanner \
  --positive-per-format 25 \
  --target-alpha 0.1 \
  --seed 11 \
  --output introspection/data/reports/dp_honey_scanner_eval_v1.json
```

This registry-shaped scanner evaluation includes a split-conformal confidence
threshold over benign calibration scores. It calibrates the scanner confidence
gate, not generator indistinguishability.

Generate DP-HONEY generation-realism evidence with aggregate generated-vs-
reference metrics and no raw token serialization:

```bash
uv run dp-honey eval-realism \
  --count-per-format 25 \
  --seed 11 \
  --output introspection/data/reports/dp_honey_generation_realism_eval_v2.json
```

This bounded realism report covers all registered formats with format validity,
duplicate-rate, character-entropy, and model-likelihood metrics. It is still not
the paper's full statistical-distinguisher suite.

Generate DP-HONEY statistical-distinguisher evidence with the paper-named test
families and no raw token serialization:

```bash
uv run dp-honey eval-statistical-distinguishers \
  --train-count-per-format 25 \
  --test-count-per-format 25 \
  --alpha 0.1 \
  --seed 11 \
  --output introspection/data/reports/dp_honey_statistical_distinguisher_eval_v2.json
```

This suite evaluates character entropy, bigram likelihood, numeric-substring
features, and a discriminator MLP. The current promoted artifact passes all four
families under the bounded same-format synthetic holdout, so
`paper_faithful_statistical_distinguisher=true`. This is still not a
production-secret indistinguishability proof.

Generate the DP-HONEY paper-faithfulness checklist from the scanner,
generation-realism, statistical-distinguisher, gateway smoke, and audit evidence:

```bash
uv run aegis-dp-honey-paper-evidence \
  --scanner-eval introspection/data/reports/dp_honey_scanner_eval_v1.json \
  --generation-realism-eval introspection/data/reports/dp_honey_generation_realism_eval_v2.json \
  --statistical-distinguisher-eval introspection/data/reports/dp_honey_statistical_distinguisher_eval_v2.json \
  --smoke introspection/data/reports/aegis_default_mock_provider_smoke_dp_honey_segment_v2.json \
  --audit-jsonl introspection/data/reports/aegis_default_mock_provider_smoke_dp_honey_segment_audit_v2.jsonl \
  --output introspection/data/reports/dp_honey_paper_evidence_v4.json
```

The checklist is deliberately stricter than the scanner metric. It fails closed
if runtime audit DP-HONEY generator metadata does not match the generation and
statistical evaluation parameters. The current v4 report sets
`paper_faithful_plus=true` and `promotion_eligible=true` for the local synthetic
DP-HONEY registry. That claim does not cover provider-valid credentials,
training on real secret corpora, or external secret-manager registration.

To seed a session canary without putting a placeholder in the user turn, use the
mock-only test route:

```bash
curl -s http://127.0.0.1:8000/test/seed-canary \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "session-local-1",
    "slot_name": "repo_pat",
    "credential_type": "github_pat",
    "turn_index": 0
  }'
```

The seed response uses schema version `aegis.test_seed_canary/v1` and returns
only safe identifiers, credential type, source, and SHA-256. It never returns
the generated canary value. Repeating the same `session_id` and `slot_name` is
idempotent; using the same slot with a different credential type is rejected.
`turn_index` is optional and defaults to `0`.
Seeded canaries are a redteam/dev fixture, not DP-HONEY injection. Seeded-only
turns activate canary/NIMBUS detection while the runtime trace still reports the
DP-HONEY stage as not configured.

Use `POST /test/reset` between redteam runs. An empty JSON object clears all
development proxy audit and session state:

```bash
curl -s http://127.0.0.1:8000/test/reset \
  -H 'Content-Type: application/json' \
  -d '{}'
```

To clear only one NIMBUS session and its audit events, pass a session id:

```bash
curl -s http://127.0.0.1:8000/test/reset \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "session-local-1"}'
```

This route is a local development/testing affordance, not a production API.

Every chat response includes an additive `aegis.runtime_trace` object with
schema version `aegis.runtime_trace/v1`. The trace summarizes the ordered stages:
normalize, DP-HONEY, CIFT, provider egress guard, provider, canary, NIMBUS,
policy, and audit. It contains detector names, statuses, provider/model
identifiers, counts, and actions only; it does not include raw secrets, canary
values, feature vectors, or model output.

Fetch recent audit events:

```bash
curl -s http://127.0.0.1:8000/audit/recent
```

Filter recent audit events by session and limit:

```bash
curl -s 'http://127.0.0.1:8000/audit/recent?session_id=session-local-1&limit=5'
```

Enable durable local audit JSONL by setting `AEGIS_AUDIT_JSONL_PATH` before
starting the proxy:

```bash
AEGIS_AUDIT_JSONL_PATH=/tmp/aegis-audit.jsonl \
uv run aegis-proxy --host 127.0.0.1 --port 8000
```

The JSONL sink writes the same redacted audit representation returned by
`/audit/recent`. It preserves trace id, session id, detector results, policy
decision, CIFT certification/runtime hashes when present, provider
skipped/completed metadata, and latency. It must not contain raw credential
values or raw canary values.

Every audit record also carries an additive `runtime_evidence` object with
schema version `aegis.audit_runtime_evidence/v1`. It captures policy mode,
final action, provider skipped/completed state, credential-slot status,
detector versions, detector latencies, whitelisted artifact hashes, CIFT
certification/runtime summary fields, fail-closed events, and total latency.
The block is meant for release evidence and operator debugging; it is still
redacted and must not contain raw prompts, raw secrets, raw canaries, activation
vectors, or model output.

Explain one trace from the audit store:

```bash
curl -s 'http://127.0.0.1:8000/audit/explain?trace_id=trace-local-1'
```

`GET /audit/explain` returns schema version `aegis.audit_explain/v1` with a
stage timeline for normalize, DP-HONEY, CIFT, provider egress guard, provider,
canary, NIMBUS, policy, and audit. It summarizes detector actions and safe
evidence fields only.

Explain one trace from a saved durable audit JSONL after the proxy has stopped:

```bash
uv run aegis-audit-explain \
  --input introspection/data/reports/aegis_default_mock_provider_smoke_ambiguous_protected_audit_v1.jsonl \
  --trace-id smoke-egress-guard-trace \
  --output introspection/data/reports/aegis_default_mock_provider_smoke_ambiguous_protected_audit_explain_v1.json
```

The offline explainer emits the same `aegis.audit_explain/v1` shape as
`GET /audit/explain` and fails if no matching `trace_id` or `session_id` is
present in the JSONL.

Summarize NIMBUS behavior from external redteam JSONL results:

```bash
uv run aegis-nimbus-report --input ../watchman-redteam/results/aegis-local.jsonl
```

Evaluate labeled NIMBUS sessions with separate false negative and false
positive rates:

```bash
uv run aegis-nimbus-eval \
  --input ../watchman-redteam/results/aegis-local.jsonl \
  --labels ../watchman-redteam/results/aegis-local-labels.json \
  --output introspection/data/reports/aegis_nimbus_eval_v1.json
```

Generate the bootstrap learned-NIMBUS training corpus contract and manifest:

```bash
uv run aegis-nimbus-training-corpus \
  --output introspection/data/reports/aegis_nimbus_training_corpus_v0.jsonl \
  --manifest-output introspection/data/reports/aegis_nimbus_training_corpus_manifest_v0.json

uv run aegis-nimbus-training-corpus \
  --profile sealed_holdout \
  --output introspection/data/reports/aegis_nimbus_sealed_holdout_corpus_v0.jsonl \
  --manifest-output introspection/data/reports/aegis_nimbus_sealed_holdout_corpus_manifest_v0.json
```

This writes `nimbus-training-turn/v0` records with explicit session split-group
keys; benign, exact canary, encoded, partial/multi-turn, paraphrased,
tool-output, and delayed leakage scenarios; 16 InfoNCE-style negative contexts;
and manifests that mark the artifacts as `not_promotable_training_contract_only`.
The `sealed_holdout` profile uses distinct synthetic secret contexts and session
groups so the learned scaffold can be evaluated without `--allow-training-eval`.
See [docs/nimbus-training-corpus.md](docs/nimbus-training-corpus.md).

Train and evaluate the offline lexical InfoNCE scaffold against that corpus:

```bash
uv run aegis-nimbus-train-infonce \
  --input introspection/data/reports/aegis_nimbus_training_corpus_v0.jsonl \
  --output introspection/data/reports/aegis_nimbus_infonce_model_v0.json

uv run aegis-nimbus-eval-infonce \
  --input introspection/data/reports/aegis_nimbus_training_corpus_v0.jsonl \
  --model introspection/data/reports/aegis_nimbus_infonce_model_v0.json \
  --output introspection/data/reports/aegis_nimbus_infonce_eval_v0.json \
  --allow-training-eval \
  --grouped-cv-output introspection/data/reports/aegis_nimbus_infonce_grouped_cv_v0.json

uv run aegis-nimbus-eval-infonce \
  --input introspection/data/reports/aegis_nimbus_sealed_holdout_corpus_v0.jsonl \
  --model introspection/data/reports/aegis_nimbus_infonce_model_v0.json \
  --output introspection/data/reports/aegis_nimbus_infonce_sealed_holdout_eval_v0.json
```

Evaluate the learned InfoNCE scaffold through the in-process runtime adapter
against the sealed holdout. This is beta evidence only; it uses registered
canary contexts rather than a production secret-context candidate store and
does not count as live gateway evidence:

```bash
uv run aegis-nimbus-runtime-beta-eval \
  --input introspection/data/reports/aegis_nimbus_sealed_holdout_corpus_v0.jsonl \
  --model introspection/data/reports/aegis_nimbus_infonce_model_v0.json \
  --output introspection/data/reports/aegis_nimbus_runtime_beta_eval_v0.json
```

Bind the deterministic runtime baseline, learned scaffold evals, sealed holdout,
runtime beta eval, and gateway smoke into a single promotion decision:

```bash
uv run aegis-nimbus-promotion-evidence \
  --deterministic-eval introspection/data/reports/aegis_nimbus_deterministic_beta_eval_v1.json \
  --calibration-manifest introspection/data/reports/aegis_nimbus_training_corpus_manifest_v0.json \
  --sealed-manifest introspection/data/reports/aegis_nimbus_sealed_holdout_corpus_manifest_v0.json \
  --infonce-model introspection/data/reports/aegis_nimbus_infonce_model_v0.json \
  --grouped-cv introspection/data/reports/aegis_nimbus_infonce_grouped_cv_v0.json \
  --sealed-holdout introspection/data/reports/aegis_nimbus_infonce_sealed_holdout_eval_v0.json \
  --gateway-smoke introspection/data/reports/aegis_default_mock_provider_smoke_nimbus_dp_honey_refresh_v2.json \
  --runtime-beta-eval introspection/data/reports/aegis_nimbus_runtime_beta_eval_v0.json \
  --output introspection/data/reports/aegis_nimbus_promotion_evidence_v0.json
```

The v0 InfoNCE artifact is an offline lexical scaffold. It reports
`promotion_status=not_promotable_offline_scaffold` and
`paper_faithful_learned_critic=false`. It can be wired into runtime policy only
with the explicit `learned_infonce_beta` configuration below, and remains
non-promotable.
The `--allow-training-eval` flag labels the main report as a training
diagnostic. Current grouped-CV and sealed-holdout evidence both report turn-level
FP/FN and session-level FP/FN separately: turn FPR `0.0`, turn FNR `0.214286`,
session FPR `0.0`, and session FNR `0.0`. The runtime beta eval reports turn
FP=0, turn FN=5, turn FPR `0.0`, turn FNR `0.357143`, session FP=0, session
FN=1, session FPR `0.0`, and session FNR `0.125`. The scaffold remains
non-promotable because it is a tiny lexical model with runtime false negatives,
no live learned gateway FN/FP evidence, and no promotion manifest. The promotion
evidence report records `promote_learned_runtime=false` and recommends keeping
deterministic canary NIMBUS as the active runtime critic.

Generate a local in-process NIMBUS fixture JSONL when the external redteam
runner is not available:

```bash
uv run aegis-nimbus-fixtures --output /tmp/aegis-nimbus-fixtures.jsonl
uv run aegis-nimbus-report --input /tmp/aegis-nimbus-fixtures.jsonl
uv run aegis-nimbus-eval \
  --input tests/aegis/fixtures/nimbus_redteam/external_runner_sanitized_v1.jsonl \
  --labels tests/aegis/fixtures/nimbus_redteam/eval_labels_v1.json \
  --output introspection/data/reports/aegis_nimbus_deterministic_beta_eval_v1.json
```

The report reads detector and policy metadata only. It distinguishes immediate
public canary detectors from NIMBUS critic evidence, so partial leakage can be
understood even when `encoded_canary` does not trigger on a single turn. See
[docs/nimbus-redteam-eval.md](docs/nimbus-redteam-eval.md) for the full
black-box redteam evaluation loop.

NIMBUS runtime calibration is environment-driven. Defaults preserve the current
local profile: exact and encoded matches count as `1.0` leakage bit, partial
matches count as `0.8 * fragment_ratio`, and cumulative budget fractions map to
`warn` at `0.3`, `sanitize` at `0.6`, and `block` at `0.9`.
The active runtime critic is deterministic canary-based beta
(`critic_kind=canary`, `paper_faithful_learned_critic=false`). A paper-faithful
learned NIMBUS release now has a bootstrap corpus contract and offline lexical
InfoNCE scaffold with turn-level and session-level grouped-CV plus small sealed
holdout evidence plus an opt-in in-process runtime beta adapter, but still
needs a larger session-leakage corpus, a production secret-context candidate
store, live gateway FN/FP metrics, and a promotion manifest.
`aegis-nimbus-eval` is the deterministic-beta labeled evaluation wrapper; it
is not a learned NIMBUS promotion path.

```bash
AEGIS_NIMBUS_EXACT_MATCH_LEAKAGE_BITS=1.0
AEGIS_NIMBUS_ENCODED_MATCH_LEAKAGE_BITS=1.0
AEGIS_NIMBUS_PARTIAL_MATCH_LEAKAGE_BITS=0.8
AEGIS_NIMBUS_PARTIAL_MATCH_THRESHOLD=0.4
AEGIS_NIMBUS_CONFIDENCE=0.8
AEGIS_NIMBUS_BUDGET_BITS=1.0
AEGIS_NIMBUS_WARN_THRESHOLD=0.3
AEGIS_NIMBUS_SANITIZE_THRESHOLD=0.6
AEGIS_NIMBUS_BLOCK_THRESHOLD=0.9
AEGIS_NIMBUS_MAX_TURNS=20
AEGIS_NIMBUS_CRITIC_VERSION=canary-v0
```

To exercise the learned runtime beta path explicitly, provide the model path
and set the critic kind. This mode fails closed if the model artifact cannot be
loaded and surfaces `promotion_status=learned_runtime_beta_not_promotable` in
readiness/capability evidence:

```bash
AEGIS_NIMBUS_CRITIC_KIND=learned_infonce_beta
AEGIS_NIMBUS_INFONCE_MODEL_PATH=introspection/data/reports/aegis_nimbus_infonce_model_v0.json
AEGIS_NIMBUS_CRITIC_VERSION=nimbus-infonce-lexical-v0
```

For a stricter local profile where one roughly half-token partial fragment can
block, lower the action thresholds intentionally rather than changing detector
code.

The proxy smoke command checks the default profile unless told otherwise:

```bash
uv run aegis-proxy-smoke \
  --url http://127.0.0.1:8000 \
  --timeout 5 \
  --output introspection/data/reports/aegis_default_mock_provider_smoke_ambiguous_protected_v1.json
```

For a strict self-hosted CIFT gateway, add the integrated pre-generation block
requirement:

```bash
uv run aegis-proxy-smoke \
  --url http://127.0.0.1:8000 \
  --timeout 120 \
  --require-cift-pre-generation-block \
  --output introspection/data/reports/aegis_self_hosted_cift_smoke_v1.json
```

To demo aggressive partial-leak blocking, start the proxy with a stricter
NIMBUS profile:

```bash
AEGIS_NIMBUS_SANITIZE_THRESHOLD=0.35 \
AEGIS_NIMBUS_BLOCK_THRESHOLD=0.36 \
uv run aegis-proxy --host 127.0.0.1 --port 8000
```

Then verify that a single partial seeded-canary leak blocks:

```bash
uv run aegis-proxy-smoke \
  --url http://127.0.0.1:8000 \
  --timeout 5 \
  --nimbus-profile strict-partial-block \
  --output introspection/data/reports/aegis_strict_nimbus_smoke_v1.json
```

Generate controlled trace-collection assignments for human operators:

```bash
uv run aegis-trace-assignments \
  --participant alice \
  --participant bob \
  --output data/trace_collection/assignments.jsonl
```

Build normalized records from completed collection inputs:

```bash
uv run aegis-trace-seed-inputs \
  --assignments data/trace_collection/assignments.jsonl \
  --variants-per-label 20 \
  --output data/trace_collection/collection_inputs.generated.jsonl

uv run aegis-trace-build-records \
  --assignments data/trace_collection/assignments.jsonl \
  --inputs data/trace_collection/collection_inputs.generated.jsonl \
  --output data/trace_collection/records.generated.jsonl \
  --model-provider mock \
  --model-id mock-model \
  --capability-mode offline_eval
```

The trace-collection harness emits proxy-shaped `NormalizedTurn` records with
DP-HONEY canaries, labels, families, `SensitiveSpan` metadata, and pending CIFT
tokenization markers. It is for controlled fake-secret data collection, not for
recording production credentials. See
[docs/trace-collection-harness.md](docs/trace-collection-harness.md) for the
input schema and workflow.

Exercise the mock proxy in Python:

```python
from aegis.proxy.mock_app import create_default_proxy

proxy = create_default_proxy()
status, payload = proxy.handle(
    method="POST",
    path="/v1/chat/completions",
    body={
        "model": "mock-model",
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": {"trace_id": "trace-1", "session_id": "session-1"},
        "tool_calls": [
            {"name": "send_slack_message", "arguments": {"channel": "#ir", "text": "status only"}}
        ],
    },
)

print(status)
print(payload["choices"][0]["message"]["content"])
print(payload["aegis"]["schema_version"])
print(payload["aegis"]["trace_id"])
print(payload["aegis"]["session_id"])
print(payload["aegis"]["policy_decision"])
```

`/v1/chat/completions` accepts only JSON-object request bodies and returns a
stable `payload["aegis"]` envelope containing `schema_version`, `trace_id`,
`session_id`, `turn_index`, `capability_mode`, `detector_count`, detector
results, and the final policy decision. Optional `tool_calls` are normalized into
runtime `ToolCall` contracts for detector use. `/audit/recent` returns a safe
audit projection: trace/session/turn handles, a turn summary, detector results,
policy decision, and whitelisted span metadata. It does not echo raw message
content, arbitrary metadata values, tool-call arguments, or raw request bodies.
When the final policy action is `sanitize`, `block`, or `escalate`, the proxy
must not deliver raw model output in `choices`; it returns sanitized output when
available or a fixed withheld-output message.
For local redteam harnesses, `POST /test/seed-canary` registers a supplied
canary value without returning it, then the mock proxy scans exact and encoded
model output plus normalized tool-call arguments. `POST /test/reset` clears audit
history and seeded canaries.

## Quality Gates

The repository treats the runtime spine as an enforced contract. Pull requests
to `main` must pass CI on Python 3.11 and Python 3.12.

The local gate is:

```bash
make quality
```

It runs:

```bash
uv run --extra dev ruff check src/aegis src/detect tests/aegis tests/dp_honey scripts
uv run --extra dev ruff format --check src/aegis src/detect tests/aegis tests/dp_honey scripts
uv run --extra dev mypy src/aegis src/detect scripts
uv run python scripts/check_import_boundaries.py
uv run python scripts/check_artifact_boundaries.py
uv run --extra dev pytest
```

Coverage is enforced at 90 percent for the runtime package.

## Repository Layout

```text
src/aegis/core/        Runtime contracts and orchestrator
src/aegis/canaries/    Honeytoken registration and injection helpers
src/aegis/demo/        Built-in runtime demo scenarios
src/aegis/detectors/   Detector stage implementations
src/aegis/policy/      Policy decision logic
src/aegis/audit/       Audit sinks
src/aegis/providers/   Model provider adapters
src/aegis/proxy/       Proxy adapters and mock proxy surface
src/aegis/replay/      Offline replay harnesses for fixtures and demos
src/aegis/sdk/         SDK entrypoint for embedding the runtime
tests/aegis/           Runtime spine tests
scripts/               Repository quality and architecture checks
introspection/         Research notebooks, activation experiments, and CIFT reports
docs/                  Project and setup documentation
```

## Contribution Rules

New runtime work must preserve the spine boundaries:

- Adapters create `NormalizedTurn`; detectors consume normalized data.
- Detectors return `DetectorResult`; they do not emit `PolicyDecision`.
- Policy is the only layer that emits `PolicyDecision`.
- Audit records normalized input summary, detector outputs, policy decision,
  and latency.
- CIFT must emit either activation risk or explicit unavailable evidence.
- Runtime CIFT detectors consume promoted JSON artifacts and feature-vector
  metadata. Feature-vector annotators may attach derived activation features
  before detectors run, but training pickles and research modules remain in
  `introspection/`.
- Selected-choice CIFT metadata is the primary runtime route. Broader
  payload/query readout is degraded research fallback coverage and must be
  labeled as such in detector evidence. Production self-hosted CIFT with block
  activation-failure policy fails closed when selected-choice geometry is absent
  instead of silently routing to fallback.
- DP-HONEY injection/registration and canary detection remain separate stages.
- Tool-call sensitive spans should carry `metadata.tool_call_name` and
  `metadata.argument_path` using the `arguments.payload.items[1]` path dialect.
  The provider egress guard may report those handles as evidence, but never raw
  argument values.
- NIMBUS-style detectors emit cumulative session risk as ordinary
  `DetectorResult` values; policy still owns the final action.
- The canonical NIMBUS runtime path is `NimbusDetector` plus a `NimbusCritic`.
  `CanaryNimbusCritic` is the deterministic local critic for proxy and redteam
  positive controls; paper-faithful critics should satisfy the same interface.
- Real credentials cross runtime boundaries as handles, spans, hashes, or
  evidence, not raw production secrets.
- Generated trace data, local worktrees, OCR assets, pycache files, and raw
  model or activation artifacts should not be committed into runtime paths.

See [CONTRIBUTING.md](CONTRIBUTING.md) before adding detector, policy, proxy, or
adapter code.

## Related Docs

- [Runtime spine](docs/aegis-runtime-spine.md)
- [Trace collection harness](docs/trace-collection-harness.md)
- [Contributing](CONTRIBUTING.md)
- [Asana MCP setup](docs/ASANA_MCP_SETUP.md)
