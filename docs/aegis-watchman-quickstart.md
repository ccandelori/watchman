# Aegis Watchman Quickstart

This guide gets a local agent running through Aegis Watchman with the current
reference setup:

- Ollama serving `qwen3:4b` on `http://127.0.0.1:11434`
- Aegis Watchman running the protection gateway on `http://127.0.0.1:8000/v1`
- Hermes Agent, or another OpenAI-compatible client, pointed at Aegis instead
  of directly at Ollama

The important routing rule is simple:

```text
Hermes Agent -> Aegis Watchman on 8000/v1 -> Ollama on 11434 -> qwen3:4b
```

Do not point the agent at Ollama directly when testing Aegis. Direct Ollama
traffic bypasses DP-HONEY, CIFT, provider egress guard, canary detection,
NIMBUS, policy, and audit.

## 1. Check prerequisites

You need:

- macOS with Apple Silicon for the current Qwen3-4B MPS reference path.
- Ollama installed.
- The Aegis repository checked out locally.
- Project dependencies installed with `uv sync --extra dev`.
- `qwen3:4b` available in Ollama, or enough time for Watchman/Ollama to pull it.

The current certified CIFT claim is model-specific. It applies to the promoted
`Qwen/Qwen3-4B` MPS reference path only. Other local models can still be routed
through the gateway, but they are not CIFT-certified until they pass their own
calibration and release gate.

## 2. Open Watchman

From the repository root:

```bash
scripts/run-aegis-desktop-launcher.sh
```

For the packaged local app:

```bash
cd desktop/aegis-launcher
npm run pack:mac
open "dist/mac-arm64/Aegis Watchman.app"
```

The app should open to the Launch tab. Use the app as the normal control
surface. The browser launcher and direct terminal commands are fallback/debug
paths.

## 3. Confirm the provider

In the Provider card:

- Ollama should show as running on `11434`.
- The selected model should be `qwen3:4b`.
- If multiple Ollama models are installed, choose `qwen3:4b` from the picker.
- If `qwen3:4b` is missing, use the app's model action to pull it, or run
  `ollama pull qwen3:4b`.

If Ollama is not running, click **Start Ollama** in the app. Ollama remains the
model provider; Watchman does not replace the model.

## 4. Start protection

Click **Start Protection**.

The app runs the local protection sequence:

```text
Ollama server
Local model
Machine checks
GPU
CIFT
Gateway
Console
```

Expected result:

- Protection status changes to running.
- GPU reports `MPS` on the current Mac reference path.
- CIFT reports running or adopted.
- Gateway reports running or adopted.
- Agent settings show `http://127.0.0.1:8000/v1`.

If protection is already running, the primary action changes to a stop action.
Use **Refresh** if the app was opened after services were already started.

## 5. Point Hermes at Aegis

In Hermes Agent, choose **Local / custom endpoint** and use:

```text
Base URL: http://127.0.0.1:8000/v1
API key:  aegis-local-dev-key
Model:    qwen3:4b
```

The API key only needs to be non-empty for local client compatibility. Provider
credentials stay server-side in Aegis configuration.

If Hermes says the endpoint advertises no models, check that the Watchman gateway
is running and that the selected provider model is available in Ollama.

## 6. Verify protection in the app

Click **Run Verification**.

The app should produce evidence for:

- A benign request that reaches the model.
- A CIFT block before provider/model completion.
- A provider-egress block before a raw credential-shaped payload can leave the
  gateway.

Then open:

- **Activity** for recent allow/block decisions.
- **Latency** for gateway and detector timing.
- **Detectors** for the detectors behind the decision.
- **Trace** for the selected request path and redacted runtime evidence.

For a good demo, the important proof is the CIFT block trace: the provider/model
call is skipped because Aegis stopped the turn before generation.

## 7. Stop or reset

Use **Stop Protection** to stop the Aegis gateway and CIFT sidecar. If old
statuses are confusing after a failed or interrupted run, use the app's clear
action before starting again.

Ollama may continue running after Watchman stops. Stop it from the app when the
provider card offers that action, or quit Ollama normally from macOS.

## Troubleshooting

### Hermes works with Ollama but not Aegis

Hermes is probably still pointed at `http://127.0.0.1:11434/v1`. Change the base
URL to `http://127.0.0.1:8000/v1`.

### CIFT is unavailable

Check the Launch tab first. CIFT requires the Qwen3-4B sidecar and the promoted
artifact binding to match model id, revision, tokenizer hash, chat-template
hash, selected layer/window, device, and dtype. Do not present CIFT enforcement
as active if the app reports it as unavailable or degraded.

### GPU is not MPS

The current reference certification is MPS-bound. If the device check does not
detect MPS on the Mac reference path, fix that before claiming certified CIFT
evidence.

### Ollama is already running

That is fine. Watchman can adopt an already-running Ollama server. The app should
show that Ollama is responding on `127.0.0.1:11434`.

### Need the command-level demo harness

The app is the preferred user experience. Command-level demo details are kept in
[`docs/aegis-watchman-demo-scenario.md`](aegis-watchman-demo-scenario.md) for
repeatable evidence generation and archival reference.
