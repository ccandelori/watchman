"use strict";

const state = {
  backend: null,
  launcher: null,
  ollama: null,
  observability: null,
  activeTab: "launch",
  selectedTraceId: null,
  selectedTracePinned: false,
  run: {
    active: false,
    kind: null,
    currentStep: null,
    events: {},
  },
  pollTimer: null,
  coreRefreshInFlight: null,
  observabilityRefreshInFlight: null,
  observedFinishedAt: {},
};

const protectionSteps = [
  { id: "ollama", title: "Ollama server", summary: "Start or detect the local provider." },
  { id: "model", title: "Local model", summary: "Use the selected model in Ollama." },
  { id: "preflight", title: "Machine checks", summary: "Check paths, ports, and local tools." },
  { id: "mps_preflight", title: "GPU", summary: "Choose the best available compute device." },
  { id: "cift_sidecar", title: "CIFT", summary: "Start model introspection." },
  { id: "gateway", title: "Gateway", summary: "Put Aegis in front of the model." },
  { id: "console", title: "Console", summary: "Start the operator evidence view." },
];

const verificationSteps = [
  { id: "cift_smoke", title: "CIFT block check", summary: "Prove exfiltration intent is blocked before response." },
  { id: "real_provider_smoke", title: "Model allow check", summary: "Prove benign traffic reaches the local model." },
];

const timeouts = {
  ollama: 60,
  model: 900,
  preflight: 45,
  mps_preflight: 90,
  cift_sidecar: 180,
  gateway: 120,
  console: 120,
  cift_smoke: 300,
  real_provider_smoke: 300,
};

const detectorDisplayNames = new Map([
  ["cift_runtime", "CIFT intent check"],
  ["encoded_canary", "Encoded canary check"],
  ["nimbus", "Session leak check"],
  ["nimbus_tool_egress", "Tool-output leak check"],
  ["noop_canary", "Canary check"],
  ["provider_egress_guard", "Provider safety check"],
  ["text_canary", "Text canary check"],
  ["tool_call_canary", "Tool-call leak check"],
]);

const stageDisplayNames = new Map([
  ["normalize", "Prepare request"],
  ["credential_slot", "Credential handling"],
  ["dp_honey", "Honeytoken setup"],
  ["cift", "CIFT intent check"],
  ["provider_egress_guard", "Provider safety check"],
  ["provider", "Local model"],
  ["canary", "Canary checks"],
  ["nimbus", "Session leak check"],
  ["policy", "Policy decision"],
  ["audit", "Audit log"],
]);

const statusDisplayNames = new Map([
  ["active", "Active"],
  ["allow", "Allow"],
  ["block", "Block"],
  ["blocked", "Blocked"],
  ["completed", "Answered"],
  ["done", "Done"],
  ["failed", "Failed"],
  ["idle", "Idle"],
  ["not_configured", "Not configured"],
  ["not_run", "Not run"],
  ["optional", "Optional"],
  ["passed", "Passed"],
  ["running", "Running"],
  ["skipped", "Not called"],
  ["stopped", "Stopped"],
  ["unavailable", "Not used"],
  ["unknown", "Unknown"],
  ["warn", "Warn"],
  ["warned", "Needs review"],
  ["waiting", "Waiting"],
  ["written", "Written"],
]);

const fieldDisplayNames = new Map([
  ["canary_count", "Canaries"],
  ["credential_needed_count", "Credential requests"],
  ["final_action", "Decision"],
  ["honeytoken_substituted_count", "Honeytokens placed"],
  ["model_id", "Model"],
  ["provider", "Provider"],
  ["reason", "Reason"],
  ["real_secret_present_count", "Real secrets"],
]);

const valueDisplayNames = new Map([
  ["blocked_sensitive_value_before_provider_egress", "Sensitive value was headed to the model"],
  ["cift_pre_generation_policy_block", "CIFT stopped the turn before generation"],
  ["deterministic_beta", "Deterministic beta"],
  ["honeytoken_substituted", "Honeytoken substituted"],
  ["learned_infonce_beta", "Learned leakage critic beta"],
  ["learned_runtime_beta", "Learned beta"],
  ["learned_runtime_beta_not_promotable", "Learned beta, not promotable"],
  ["not_configured", "Not configured"],
  ["pre_generation_policy_block", "Blocked before the model was called"],
  ["real_secret_present", "A real secret was present"],
  ["selected block from highest-severity detector recommendation.", "Blocked by the strongest detector result"],
  ["self_hosted_introspection", "Self-hosted model introspection"],
  ["severity", "Strongest detector wins"],
]);

const displayTokenOverrides = new Map([
  ["api", "API"],
  ["cift", "CIFT"],
  ["cpu", "CPU"],
  ["cuda", "CUDA"],
  ["dp", "DP"],
  ["honey", "HONEY"],
  ["id", "ID"],
  ["mps", "MPS"],
  ["nimbus", "NIMBUS"],
  ["noop", "No-op"],
  ["openai", "OpenAI"],
  ["url", "URL"],
]);

const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([key, value]) => {
    if (key === "class") node.className = String(value);
    else if (key === "text") node.textContent = value === null || value === undefined ? "" : String(value);
    else if (value === true) node.setAttribute(key, "");
    else if (value !== false && value !== null && value !== undefined) node.setAttribute(key, String(value));
  });
  children.forEach((child) => {
    if (typeof child === "string") node.appendChild(document.createTextNode(child));
    else if (child) node.appendChild(child);
  });
  return node;
};

async function loadAll(options = {}) {
  const includeObservability = options.includeObservability !== false;
  await loadCore();
  if (state.observability === null) {
    state.observability = defaultObservabilityStatus("Observability has not connected to the gateway yet.");
  }
  render();
  if (!includeObservability) return;
  state.observability = await loadObservability();
  reconcileSelectedTrace();
  render();
}

async function loadCore() {
  if (state.coreRefreshInFlight !== null) return state.coreRefreshInFlight;
  state.coreRefreshInFlight = loadCoreSnapshot().finally(() => {
    state.coreRefreshInFlight = null;
  });
  return state.coreRefreshInFlight;
}

async function loadCoreSnapshot() {
  const [backend, launcher] = await Promise.all([
    window.aegisDesktop.backendStatus(),
    window.aegisDesktop.launcherApi({ method: "GET", path: "/api/state", body: null }),
  ]);
  state.backend = backend;
  state.launcher = launcher;
  if (state.ollama === null) {
    state.ollama = defaultOllamaStatus("Checking Ollama on 127.0.0.1:11434.");
  }
  render();
  state.ollama = await window.aegisDesktop
    .ollamaStatus()
    .catch((error) => defaultOllamaStatus(error instanceof Error ? error.message : String(error)));
}

async function loadObservability() {
  if (state.observabilityRefreshInFlight !== null) return state.observabilityRefreshInFlight;
  state.observabilityRefreshInFlight = loadObservabilitySnapshot().finally(() => {
    state.observabilityRefreshInFlight = null;
  });
  return state.observabilityRefreshInFlight;
}

async function loadObservabilitySnapshot() {
  try {
    return await window.aegisDesktop.launcherApi({ method: "GET", path: "/api/observability?limit=30", body: null });
  } catch (error) {
    return defaultObservabilityStatus(error instanceof Error ? error.message : String(error));
  }
}

function defaultObservabilityStatus(error) {
  return {
    schema_version: "aegis.launcher_observability/unavailable",
    gateway_base_url: state.launcher?.agent_settings?.base_url || "",
    overview: null,
    events: { events: [], detector_activity: {}, source: "unavailable" },
    latency: null,
    error,
  };
}

function render() {
  if (!state.launcher || !state.ollama) return;
  const errors = [
    renderSection("tabs", renderTabs),
    renderSection("summary", renderSummary),
    renderSection("run panel", renderRunPanel),
    renderSection("timeline", renderTimeline),
    renderSection("provider", renderProvider),
    renderSection("agent settings", renderAgentSettings),
    renderSection("diagnostics", renderDiagnostics),
    renderSection("logs", renderLogs),
    renderSection("observability status", renderObservabilityStatus),
    renderSection("activity", renderActivity),
    renderSection("latency", renderLatency),
    renderSection("detectors", renderDetectors),
    renderSection("trace", renderTrace),
  ].filter(Boolean);
  if (errors.length > 0) {
    showMessage(errors.join(" "));
  }
  configurePolling();
}

function renderSection(name, callback) {
  try {
    callback();
    return null;
  } catch (error) {
    return `${name} failed to render: ${error instanceof Error ? error.message : String(error)}.`;
  }
}

function renderTabs() {
  document.querySelectorAll(".tab-button").forEach((button) => {
    const tab = button.getAttribute("data-tab");
    const active = tab === state.activeTab;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
    button.setAttribute("tabindex", active ? "0" : "-1");
  });
  document.querySelectorAll(".tab-view").forEach((view) => {
    view.hidden = view.id !== `${state.activeTab}-view`;
  });
}

function reconcileSelectedTrace() {
  const events = observabilityEvents();
  if (events.length === 0) {
    if (!state.selectedTracePinned) state.selectedTraceId = null;
    return;
  }
  const selected = events.some((event) => event.trace_id === state.selectedTraceId);
  if (!selected && !state.selectedTracePinned) {
    state.selectedTraceId = events[0].trace_id || null;
  }
}

function observabilityUnavailable() {
  const observability = state.observability;
  return !observability || Boolean(observability.error);
}

function observabilityUnavailableState() {
  const error = state.observability?.error || "Gateway observability is unavailable.";
  const gateway = state.observability?.gateway_base_url || state.launcher?.agent_settings?.base_url || "gateway unknown";
  return emptyState("Observability unavailable", `${error} ${gateway}`);
}

function renderObservabilityStatus() {
  const status = document.getElementById("observability-status");
  if (!status) {
    return;
  }
  const observability = state.observability;
  if (!observability || observability.error) {
    status.replaceChildren(badge("unavailable"));
    return;
  }
  const protection = observability.overview?.protection || {};
  status.replaceChildren(
    el("span", { class: "status-copy", text: protection.state || "Gateway status unknown" }),
    badge(protection.severity || "waiting"),
  );
}

function renderActivity() {
  const list = document.getElementById("activity-list");
  if (observabilityUnavailable()) {
    list.replaceChildren(observabilityUnavailableState());
    return;
  }
  const events = observabilityEvents();
  if (events.length === 0) {
    list.replaceChildren(emptyState("No audited requests yet", "Send traffic through Aegis, then refresh this view."));
    return;
  }
  list.replaceChildren(...events.map((event) => eventCard(event)));
  list.querySelectorAll(".event-card").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedTraceId = button.getAttribute("data-trace-id");
      state.selectedTracePinned = true;
      state.activeTab = "trace";
      render();
    });
  });
}

function renderLatency() {
  const container = document.getElementById("latency-summary");
  if (observabilityUnavailable()) {
    container.replaceChildren(observabilityUnavailableState());
    return;
  }
  const latency = state.observability?.latency;
  if (!latency) {
    container.replaceChildren(emptyState("Latency unavailable", "Start the gateway and send a request through Aegis."));
    return;
  }
  const requestStats = latency.request_latency_ms || {};
  const detectorStats = latency.detector_latency_ms || {};
  const detectorByName = latency.detector_latency_by_name_ms || {};
  const detectorRows = Object.entries(detectorByName)
    .sort(([left], [right]) => displayDetectorName(left).localeCompare(displayDetectorName(right)))
    .map(([name, stats]) => latencyRow(name, stats));
  container.replaceChildren(
    metricCard("Median Aegis request", formatMs(requestStats.p50), `${requestStats.count || 0} recent audited requests`),
    metricCard("Slow Aegis request", formatMs(requestStats.p95), latencyRangeText(requestStats)),
    metricCard("Median detector time", formatMs(detectorStats.p50), `${detectorStats.count || 0} detector timings`),
    metricCard("Direct model baseline", "Not measured", modelBaselineDetail(latency.direct_provider_baseline)),
    el(
      "section",
      { class: "latency-table" },
      el("h3", { text: "Detector latency" }),
      detectorRows.length > 0 ? el("div", { class: "latency-rows" }, ...detectorRows) : emptyState("No detector timings", "Run protected traffic to populate detector latency."),
    ),
  );
}

function renderDetectors() {
  const container = document.getElementById("detector-dashboard");
  if (observabilityUnavailable()) {
    container.replaceChildren(observabilityUnavailableState());
    return;
  }
  const events = state.observability?.events || {};
  const activity = events.detector_activity || {};
  const overview = state.observability?.overview || {};
  const cift = overview.cift || {};
  const nimbus = overview.nimbus || {};
  container.replaceChildren(
    detectorCard("DP-HONEY", `${activity.dp_honey_substitutions || 0}`, "honeytoken substitutions", "credential handling evidence"),
    detectorCard("CIFT", `${activity.cift_pre_generation_blocks || 0}`, "blocks before response", "certified model intent check"),
    detectorCard("Canaries", `${activity.canary_detections || 0}`, "canary detections", "response and tool-output leak checks"),
    detectorCard("NIMBUS", `${activity.nimbus_warnings || 0}`, "warnings or stronger", nimbus.label ? displayValue(nimbus.label, "session leak check") : "session leak check"),
    detectorCard("Provider safety", `${activity.provider_egress_blocks || 0}`, "blocks before model call", "sensitive-value scanning"),
    detectorCard("Fail closed", `${activity.fail_closed_events || 0}`, "fail closed events", "refusal evidence"),
  );
}

function renderTrace() {
  const container = document.getElementById("trace-detail");
  if (observabilityUnavailable()) {
    container.replaceChildren(observabilityUnavailableState());
    return;
  }
  const event = selectedEvent();
  if (!event) {
    const message = state.selectedTracePinned
      ? "The selected trace is no longer in the recent results. Refresh after loading more audit history."
      : "Select a recent decision from Activity.";
    container.replaceChildren(emptyState("No trace selected", message));
    return;
  }
  container.replaceChildren(
    el(
      "section",
      { class: "trace-summary" },
      metricCard("Decision", displayStatus(event.final_action, "Unknown"), traceDecisionDetail(event)),
      metricCard("Model call", modelCallValue(event.provider_status), modelCallDetail(event)),
      metricCard("Elapsed time", formatMs(event.latency_ms), eventActivityContext(event)),
    ),
    requestPreview(event),
    el("section", { class: "stage-rail" }, ...stageTimeline(event.stage_timeline || [], event)),
    el(
      "section",
      { class: "detector-table" },
      el("h3", { text: "Detector results" }),
      detectorRows(event.detector_results || []),
    ),
    runtimeEvidencePanel(event.runtime_evidence || {}),
  );
}

function eventCard(event) {
  const selected = event.trace_id === state.selectedTraceId;
  const detectors = eventDetectorIds(event);
  return el(
    "button",
    {
      class: `event-card ${selected ? "selected" : ""}`,
      type: "button",
      "data-trace-id": event.trace_id || "",
    },
    el(
      "div",
      { class: "event-main" },
      el("span", { class: "event-trace", text: eventActivityContext(event) }),
      el("strong", { text: eventActivityTitle(event) }),
      el("p", { text: eventActivityDetail(event) }),
      requestPreview(event),
    ),
    el(
      "div",
      { class: "event-meta" },
      el("span", { text: formatEventTimestamp(event.created_at) }),
      el("span", { class: "mono", text: formatMs(event.latency_ms) }),
      detectors.length > 0 ? el("span", { text: detectors.map(displayDetectorName).join(", ") }) : el("span", { text: "No intervention" }),
    ),
  );
}

function requestPreview(event) {
  const request = event && typeof event.request === "object" && event.request !== null ? event.request : {};
  const preview = typeof request.preview === "string" && request.preview.trim() !== "" ? request.preview : "No request text recorded.";
  return el(
    "div",
    { class: "request-preview" },
    el("span", { text: "Request" }),
    el("p", { text: preview }),
    el("small", { text: requestFacts(request) }),
  );
}

function requestFacts(request) {
  const messages = numberLabel(request.message_count, "message", "messages");
  const tools = numberLabel(request.tool_call_count, "tool call", "tool calls");
  const sensitive = numberLabel(request.sensitive_span_count, "sensitive span", "sensitive spans");
  return `${messages} | ${tools} | ${sensitive}`;
}

function eventDetectorIds(event) {
  const fired = Array.isArray(event.detectors_fired) ? event.detectors_fired.filter((name) => typeof name === "string") : [];
  if (fired.length > 0) return fired;
  const results = Array.isArray(event.detector_results) ? event.detector_results : [];
  return results
    .filter((detector) => detector && detector.recommended_action && detector.recommended_action !== "allow")
    .map((detector) => detector.detector_name)
    .filter((name) => typeof name === "string");
}

function eventHasDetector(event, detectorName) {
  return eventDetectorIds(event).includes(detectorName);
}

function eventActivityTitle(event) {
  const action = typeof event.final_action === "string" ? event.final_action : "";
  const providerStatus = typeof event.provider_status === "string" ? event.provider_status : "";
  if (action === "block" && eventHasDetector(event, "cift_runtime")) {
    return "CIFT blocked exfiltration intent";
  }
  if (action === "block" && eventHasDetector(event, "provider_egress_guard")) {
    return "Sensitive data stopped before model";
  }
  if (action === "block") {
    return "Request blocked";
  }
  if (action === "allow" && providerStatus === "completed") {
    return "Request reached the local model";
  }
  if (action === "allow") {
    return "Request allowed";
  }
  return displayIdentifier(action, "Decision recorded");
}

function eventActivityContext(event) {
  const action = typeof event.final_action === "string" ? event.final_action : "";
  const providerStatus = typeof event.provider_status === "string" ? event.provider_status : "";
  if (action === "block" && providerStatus === "skipped" && eventHasDetector(event, "cift_runtime")) {
    return "Blocked before response";
  }
  if (action === "block" && providerStatus === "skipped" && eventHasDetector(event, "provider_egress_guard")) {
    return "Blocked before model";
  }
  if (action === "block" && providerStatus === "skipped") {
    return "Blocked before model";
  }
  if (action === "allow" && providerStatus === "completed") {
    return "Model answered";
  }
  return displayStatus(providerStatus || action, "Audited decision");
}

function eventActivityDetail(event) {
  const action = typeof event.final_action === "string" ? event.final_action : "";
  const providerStatus = typeof event.provider_status === "string" ? event.provider_status : "";
  if (action === "block" && providerStatus === "skipped" && eventHasDetector(event, "cift_runtime")) {
    return "CIFT saw unsafe intent, so Aegis stopped the turn before the model generated a response.";
  }
  if (action === "block" && providerStatus === "skipped" && eventHasDetector(event, "provider_egress_guard")) {
    return "Sensitive content was headed toward the model, so Aegis stopped the turn before the model ran.";
  }
  if (action === "block" && providerStatus === "skipped") {
    return "A detector required a block, so the model was not called.";
  }
  if (action === "allow" && providerStatus === "completed") {
    return "No detector asked to intervene, so the local model completed the request.";
  }
  if (action === "allow") {
    return "No detector asked to intervene.";
  }
  return displayValue(event.reason, "Aegis recorded this decision in the audit log.");
}

function traceDecisionDetail(event) {
  const action = typeof event.final_action === "string" ? event.final_action : "";
  const providerStatus = typeof event.provider_status === "string" ? event.provider_status : "";
  if (action === "block" && providerStatus === "skipped" && eventHasDetector(event, "cift_runtime")) {
    return "CIFT stopped the turn before the model generated a response.";
  }
  if (action === "block" && providerStatus === "skipped" && eventHasDetector(event, "provider_egress_guard")) {
    return "Sensitive content was stopped before it reached the model.";
  }
  if (action === "allow" && providerStatus === "completed") {
    return "No detector asked to intervene.";
  }
  return displayValue(event.reason, "Decision recorded in the audit log.");
}

function metricCard(label, value, detail) {
  return el(
    "article",
    { class: "metric-card" },
    el("span", { text: label }),
    el("strong", { class: "mono", text: value }),
    el("p", { text: detail }),
  );
}

function detectorCard(label, value, metric, detail) {
  return el(
    "article",
    { class: "detector-card" },
    el("span", { class: "detector-label", text: label }),
    el("strong", { class: "mono", text: value }),
    el("p", { text: metric }),
    el("small", { text: detail }),
  );
}

function latencyRow(name, stats) {
  return el(
    "div",
    { class: "latency-row" },
    el("strong", { text: displayDetectorName(name) }),
    el("span", { text: `median ${formatMs(stats.p50)}` }),
    el("span", { text: `slow end ${formatMs(stats.p95)}` }),
    el("span", { text: `${stats.count || 0} samples` }),
  );
}

function modelCallValue(providerStatus) {
  const status = typeof providerStatus === "string" ? providerStatus : "";
  if (status === "skipped") return "Not called";
  if (status === "completed") return "Answered";
  return displayStatus(status, "Unknown");
}

function modelCallDetail(event) {
  const providerStatus = typeof event.provider_status === "string" ? event.provider_status : "";
  if (providerStatus === "skipped") {
    return "Aegis made the decision before forwarding anything to the local model.";
  }
  if (providerStatus === "completed") {
    return "The local model received the sanitized request and returned a response.";
  }
  return displayValue(event.provider, "Model-call evidence unavailable.");
}

function stageTimeline(stages, event) {
  if (!Array.isArray(stages) || stages.length === 0) {
    return [emptyState("No timeline", "This trace has no stage timeline.")];
  }
  return stages.map((stage) =>
    el(
      "article",
      { class: `stage-item ${stage.status || "unknown"}` },
      stageBadge(stage),
      el("strong", { text: displayStageName(stage.stage) }),
      el("p", { text: stageDetail(stage, event) }),
    ),
  );
}

function detectorRows(detectors) {
  if (!Array.isArray(detectors) || detectors.length === 0) {
    return emptyState("No detector results", "This request did not record detector output.");
  }
  return el(
    "div",
    { class: "detector-rows" },
    ...detectors.map((detector) =>
      el(
        "div",
        { class: "detector-row" },
        el("strong", { text: displayDetectorName(detector.detector_name) }),
        badge(detector.recommended_action || "allow"),
        el("span", { text: displayComponentName(detector.component) }),
        el("span", { class: "mono", text: formatScore(detector.score) }),
        el("span", { class: "mono", text: formatMs(detector.latency_ms) }),
      ),
    ),
  );
}

function runtimeEvidencePanel(runtimeEvidence) {
  const providerState = runtimeEvidence.provider_state || {};
  const failClosed = Array.isArray(runtimeEvidence.fail_closed_events) ? runtimeEvidence.fail_closed_events.length : 0;
  return el(
    "section",
    { class: "runtime-evidence" },
    el("h3", { text: "Runtime evidence" }),
    summaryRow("Decision policy", displayValue(runtimeEvidence.policy_mode, "Unknown")),
    summaryRow("Credential handling", displayValue(runtimeEvidence.credential_slot_status, "Unknown")),
    summaryRow("Model call", modelCallValue(providerState.status)),
    summaryRow("Fail-closed events", String(failClosed)),
  );
}

function emptyState(title, detail) {
  return el("div", { class: "empty-state" }, el("strong", { text: title }), el("p", { text: detail }));
}

function observabilityEvents() {
  const events = state.observability?.events?.events;
  return Array.isArray(events) ? events : [];
}

function selectedEvent() {
  const events = observabilityEvents();
  return events.find((event) => event.trace_id === state.selectedTraceId) || events[0] || null;
}

function stageDetail(stage, event) {
  const customDetail = stageSpecificDetail(stage, event);
  if (customDetail !== null) return customDetail;
  const keys = ["reason", "provider", "model_id", "final_action", "credential_needed_count", "honeytoken_substituted_count", "canary_count"];
  const values = keys
    .map((key) => {
      const value = stage[key];
      const displayedValue = displayStageFieldValue(key, value);
      return value === null || value === undefined ? null : `${displayFieldName(key)}: ${displayedValue}`;
    })
    .filter(Boolean);
  if (values.length > 0) return values.join(" | ");
  return displayValue(stage.detail || stage.original_status, "No additional detail.");
}

function stageSpecificDetail(stage, event) {
  const stageName = typeof stage.stage === "string" ? stage.stage : "";
  const status = typeof stage.status === "string" ? stage.status : "";
  if (stageName === "normalize") return "Request prepared for Aegis checks.";
  if (stageName === "credential_slot") return credentialStageDetail(stage);
  if (stageName === "dp_honey") return honeytokenStageDetail(stage);
  if (stageName === "cift") return ciftStageDetail(stage);
  if (stageName === "provider_egress_guard") return providerSafetyStageDetail(stage);
  if (stageName === "provider") return status === "skipped" ? "The local model was not called." : "The local model answered the request.";
  if (stageName === "canary") return postResponseStageDetail(stage, event, "Canary checks");
  if (stageName === "nimbus") return postResponseStageDetail(stage, event, "The session leak check");
  if (stageName === "policy") return policyStageDetail(stage);
  if (stageName === "audit") return "This decision was written to the audit log.";
  return null;
}

function credentialStageDetail(stage) {
  const neededCount = Number(stage.credential_needed_count);
  const honeytokenCount = Number(stage.honeytoken_substituted_count);
  const realSecretCount = Number(stage.real_secret_present_count);
  if (realSecretCount > 0 || stage.original_status === "real_secret_present") return "A real secret was present in the request context.";
  if (honeytokenCount > 0) return `${numberLabel(honeytokenCount, "honeytoken was", "honeytokens were")} placed before detector checks.`;
  if (neededCount > 0) return `${numberLabel(neededCount, "credential slot was", "credential slots were")} identified.`;
  return "No credential slot needed changes.";
}

function honeytokenStageDetail(stage) {
  const canaryCount = Number(stage.canary_count);
  if (Number.isFinite(canaryCount) && canaryCount > 0) return `${numberLabel(canaryCount, "canary was", "canaries were")} available for leak detection.`;
  if (stage.status === "unavailable") return "Honeytoken setup was not used for this turn.";
  return "No honeytokens were added for this turn.";
}

function ciftStageDetail(stage) {
  if (stage.status === "blocked") return "Hidden-state intent looked unsafe, so CIFT stopped generation.";
  if (stage.status === "active" || stage.status === "passed") return "Hidden-state intent check allowed this turn to continue.";
  if (stage.status === "unavailable") return "Hidden-state intent check was not available for this turn.";
  return displayValue(stage.original_status, "Hidden-state intent evidence recorded.");
}

function providerSafetyStageDetail(stage) {
  if (stage.status === "blocked") return "Sensitive content would have reached the model, so Aegis stopped it.";
  if (stage.status === "active" || stage.status === "passed") return "Sensitive content check allowed this turn to continue.";
  return displayValue(stage.original_status, "Provider safety evidence recorded.");
}

function postResponseStageDetail(stage, event, detectorName) {
  const providerStatus = typeof event.provider_status === "string" ? event.provider_status : "";
  if (providerStatus === "skipped") return `${detectorName} did not run because the model was not called.`;
  if (stage.status === "unavailable") return `${detectorName} was not configured for this turn.`;
  if (stage.status === "passed" || stage.status === "active") return `${detectorName} found no leak evidence.`;
  return displayValue(stage.original_status, `${detectorName} evidence recorded.`);
}

function policyStageDetail(stage) {
  if (stage.final_action === "block" || stage.status === "blocked") return "Policy selected block from the strongest detector result.";
  if (stage.final_action === "allow" || stage.status === "passed") return "Policy allowed the request.";
  return displayValue(stage.reason, "Policy decision recorded.");
}

function latencyRangeText(stats) {
  if (!stats || stats.count === 0) return "No recent latency samples";
  return `Fastest ${formatMs(stats.min)}, slowest ${formatMs(stats.max)}`;
}

function modelBaselineDetail(baseline) {
  if (!baseline || baseline.status === "not_measured") {
    return "Run a direct-model benchmark to measure Aegis overhead.";
  }
  return displayValue(baseline.detail, "Direct-model benchmark evidence recorded.");
}

function formatMs(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "n/a";
  if (number >= 1000) return `${(number / 1000).toFixed(2)}s`;
  return `${number.toFixed(number >= 10 ? 0 : 1)}ms`;
}

function formatEventTimestamp(value) {
  if (typeof value !== "string" || value.trim() === "") {
    return "Time unknown";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Time unknown";
  }
  return new Intl.DateTimeFormat([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function numberLabel(value, singular, plural) {
  const number = Number(value);
  const count = Number.isFinite(number) ? number : 0;
  return `${count} ${count === 1 ? singular : plural}`;
}

function formatScore(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "score n/a";
  return `score ${number.toFixed(3)}`;
}

function defaultOllamaStatus(lastError) {
  return {
    running: false,
    baseUrl: "http://127.0.0.1:11434",
    apiTagsUrl: "http://127.0.0.1:11434/api/tags",
    installedModels: [],
    server: emptyProcessSnapshot("idle"),
    pull: emptyProcessSnapshot("idle"),
    lastError,
  };
}

function emptyProcessSnapshot(status) {
  return {
    status,
    pid: null,
    exitCode: null,
    signalCode: null,
    logPath: "",
    logExcerpt: "",
  };
}

function renderSummary() {
  const profile = state.launcher.profile;
  const running = protectionIsRunning();
  const providerStatus = state.ollama.running ? "running" : "stopped";
  const ciftStatus = processStatus("cift_sidecar");
  const gatewayStatus = processStatus("gateway");
  const summary = document.getElementById("summary");
  summary.replaceChildren(
    el(
      "div",
      { class: "summary-head" },
      el("strong", { text: running ? "Protection running" : "Protection stopped" }),
      badge(running ? "running" : "stopped"),
    ),
    el(
      "div",
      { class: "summary-rows" },
      summaryRow("Provider", `${profile.provider_model} (${providerStatus})`),
      summaryRow("Aegis URL", state.launcher.agent_settings.base_url),
      summaryRow("CIFT", `${profile.cift_binding.model_id} on ${runtimeDeviceLabel()}`),
      summaryRow("Services", `CIFT ${ciftStatus}, gateway ${gatewayStatus}`),
    ),
  );
}

function renderRunPanel() {
  const running = state.run.active;
  const ready = protectionIsRunning();
  const title = document.getElementById("run-title");
  const primary = document.getElementById("primary-run");
  const verify = document.getElementById("verify-run");
  const activeProcess = document.getElementById("active-process");
  title.textContent = running ? runTitle() : ready ? "Protection is running" : "Protection is stopped";
  primary.textContent = running ? "Running" : ready ? "Stop Protection" : "Start Protection";
  primary.className = ready && !running ? "button danger" : "button primary";
  primary.disabled = running;
  verify.disabled = running || !ready;
  const process = state.run.currentStep ? activeProcessForStep(state.run.currentStep) : null;
  if (process) {
    activeProcess.hidden = false;
    activeProcess.replaceChildren(processPanel(process));
  } else {
    activeProcess.hidden = true;
    activeProcess.replaceChildren();
  }
}

function renderTimeline() {
  const timeline = document.getElementById("timeline");
  const rows = [...protectionSteps, ...verificationSteps].map((step, index) => timelineItem(step, index + 1));
  timeline.replaceChildren(...rows);
}

function renderProvider() {
  const profile = state.launcher.profile;
  const installed = state.ollama.installedModels || [];
  const providerCard = document.getElementById("provider-card");
  const providerAction = document.getElementById("start-ollama");
  const selectedInstalled = installed.includes(profile.provider_model);
  providerAction.textContent = state.ollama.running ? "Stop Ollama" : "Start Ollama";
  providerAction.className = state.ollama.running ? "button danger small" : "button small";
  providerAction.disabled = state.run.active;
  const modelControl =
    installed.length > 0
      ? el(
          "select",
          { id: "model-picker", "aria-label": "Provider model" },
          ...Array.from(new Set([...installed, profile.provider_model])).map((model) =>
            el("option", { value: model, selected: model === profile.provider_model, text: model }),
          ),
        )
      : el("input", { id: "model-input", type: "text", value: profile.provider_model, "aria-label": "Provider model" });
  const rows = [
    summaryRow("Ollama", state.ollama.running ? "running on 11434" : "not running"),
    summaryRow("Model", selectedInstalled ? profile.provider_model : `${profile.provider_model} missing`),
  ];
  providerCard.replaceChildren(
    ...rows,
    el(
      "div",
      { class: "provider-controls" },
      el("div", { class: "model-control" }, el("label", { for: modelControl.id, text: "Selected model" }), modelControl),
      el(
        "button",
        {
          id: "pull-model",
          class: "button small",
          type: "button",
          disabled: state.ollama.running && !selectedInstalled ? false : true,
          text: selectedInstalled ? "Model Ready" : "Pull Model",
        },
      ),
    ),
  );
  const picker = document.getElementById("model-picker");
  if (picker instanceof HTMLSelectElement) {
    picker.addEventListener("change", () => updateModel(picker.value));
  }
  const input = document.getElementById("model-input");
  if (input instanceof HTMLInputElement) {
    input.addEventListener("change", () => updateModel(input.value));
  }
  document.getElementById("pull-model").addEventListener("click", () => runSingleStep("model"));
}

function renderAgentSettings() {
  const settings = state.launcher.agent_settings;
  const list = document.getElementById("agent-settings");
  list.replaceChildren(
    summaryRow("Base URL", settings.base_url),
    summaryRow("API key", settings.api_key),
    summaryRow("Model", settings.model),
  );
}

function renderDiagnostics() {
  const diagnostics = document.getElementById("diagnostics");
  const checks = state.launcher.preflight.checks || [];
  const failed = checks.filter((check) => check.status === "failed");
  const warned = checks.filter((check) => check.status === "warn");
  diagnostics.replaceChildren(
    summaryRow("Checks", `${failed.length} failed, ${warned.length} warning`),
    summaryRow("Backend", state.backend?.baseUrl || "not available"),
    summaryRow("Profile", state.launcher.profile.name),
  );
}

function renderLogs() {
  const logs = document.getElementById("logs");
  const launcherProcesses = Object.values(state.launcher.processes || {});
  const ollamaLogs = [
    { label: "Ollama server", status: state.ollama.server.status, logExcerpt: state.ollama.server.logExcerpt },
    { label: "Ollama pull", status: state.ollama.pull.status, logExcerpt: state.ollama.pull.logExcerpt },
  ];
  const processLogs = launcherProcesses.map((process) => ({
    label: process.label,
    status: process.status,
    logExcerpt: process.log_excerpt,
  }));
  const entries = [...ollamaLogs, ...processLogs].filter((entry) => entry.logExcerpt);
  if (entries.length === 0) {
    logs.replaceChildren(el("div", { class: "log-item" }, el("h3", { text: "No command output yet" })));
    return;
  }
  logs.replaceChildren(...entries.map((entry) => logItem(entry)));
}

function timelineItem(step, number) {
  const status = stepStatus(step.id);
  const event = state.run.events[step.id];
  const detail = event?.detail || stepDetail(step.id, step.summary, status);
  return el(
    "article",
    { class: `timeline-item ${status}` },
    el("div", { class: "timeline-marker", text: String(number) }),
    el(
      "div",
      { class: "timeline-main" },
      el("h3", { text: step.title }),
      el("p", { text: detail }),
      timelineMeta(step.id, status) ? el("span", { class: "timeline-meta", text: timelineMeta(step.id, status) }) : null,
    ),
    badge(status),
  );
}

function stepStatus(stepId) {
  const event = state.run.events[stepId];
  const observed = observedStepStatus(stepId);
  if (observed !== "waiting") return observed;
  if (event?.status === "running" || event?.status === "failed" || event?.status === "done") return event.status;
  return observed;
}

function observedStepStatus(stepId) {
  if (stepId === "ollama") {
    if (state.ollama.running) return "done";
    if (state.ollama.server.status === "exited" && Number(state.ollama.server.exitCode || 0) !== 0) return "failed";
    return "waiting";
  }
  if (stepId === "model") {
    if (modelInstalled()) return "done";
    if (state.ollama.pull.status === "exited" && Number(state.ollama.pull.exitCode || 0) !== 0) return "failed";
    return "waiting";
  }
  if (stepId === "preflight") {
    const status = state.launcher.preflight.overall_status;
    if (status === "passed") return "done";
    if (status === "failed") return "failed";
    if (status === "warn") return "warn";
    return "waiting";
  }
  if (stepId === "mps_preflight" && protectionIsRunning()) return "done";
  if (stepId === "console") {
    const process = processById(stepId);
    if (!process && protectionIsRunning()) return "optional";
  }
  if ((stepId === "cift_smoke" || stepId === "real_provider_smoke") && !processById(stepId)) {
    return "not_run";
  }
  const process = processById(stepId);
  if (!process) return "waiting";
  if (process.status === "running") return processKind(stepId) === "long-running" ? "done" : "running";
  if (Number(process.exit_code || 0) !== 0) return "failed";
  if (processKind(stepId) === "long-running") return "stopped";
  return "done";
}

function stepDetail(stepId, fallback, status) {
  if (stepId === "ollama") {
    return state.ollama.running ? "Ollama is responding on 127.0.0.1:11434." : "Start the local provider server.";
  }
  if (stepId === "model") {
    return modelInstalled() ? `${state.launcher.profile.provider_model} is available.` : "Pull the selected model before starting Aegis.";
  }
  if (stepId === "preflight") {
    return `Readiness is ${state.launcher.preflight.overall_status}.`;
  }
  if (stepId === "mps_preflight" && status === "done" && protectionIsRunning()) {
    return `CIFT is using ${runtimeDeviceLabel()}.`;
  }
  if (stepId === "console" && status === "optional") {
    return "The desktop app is the operator view.";
  }
  if (stepId === "cift_smoke" && status === "not_run") {
    return "Run verification to produce fresh block evidence.";
  }
  if (stepId === "real_provider_smoke" && status === "not_run") {
    return "Run verification to prove benign traffic reaches the model.";
  }
  const process = processById(stepId);
  if (process) return processSummary(process, status);
  return fallback;
}

function timelineMeta(stepId, status) {
  if (stepId === "ollama") return "provider port 11434";
  if (stepId === "model") return state.launcher.profile.provider_model;
  if (stepId === "mps_preflight") return runtimeDeviceLabel();
  const process = processById(stepId);
  if (!process) return null;
  return status === "done" ? processElapsedText(process) : null;
}

function activeProcessForStep(stepId) {
  if (stepId === "ollama") return ollamaProcessPanel("Ollama server", state.ollama.server);
  if (stepId === "model") return ollamaProcessPanel("Ollama pull", state.ollama.pull);
  const process = processById(stepId);
  return process ? launcherProcessPanel(process) : null;
}

function processPanel(panel) {
  return el(
    "section",
    {},
    el("div", { class: "process-head" }, el("strong", { text: panel.label }), badge(panel.status)),
    el("div", { class: "process-facts" }, processFact("Elapsed", panel.elapsed), processFact("Log", panel.logPath || "not available")),
    el("pre", { class: "process-output", text: panel.output || "No output yet." }),
  );
}

function launcherProcessPanel(process) {
  return {
    label: process.label || process.action_id,
    status: process.status === "running" ? "running" : Number(process.exit_code || 0) === 0 ? "done" : "failed",
    elapsed: processElapsedText(process),
    logPath: process.log_path,
    output: process.log_excerpt,
  };
}

function ollamaProcessPanel(label, process) {
  return {
    label,
    status: process.status,
    elapsed: process.status,
    logPath: process.logPath,
    output: process.logExcerpt,
  };
}

async function startProtection() {
  if (protectionIsRunning()) {
    await stopProtection();
    return;
  }
  await runSequence("start", protectionSteps.map((step) => step.id));
}

async function runVerification() {
  await runSequence("verify", verificationSteps.map((step) => step.id));
}

async function runSingleStep(stepId) {
  await runSequence("single", [stepId]);
}

async function toggleOllama() {
  if (state.ollama.running) {
    await stopOllama();
    return;
  }
  await runSingleStep("ollama");
}

async function stopOllama() {
  await withMessage(async () => {
    const status = await window.aegisDesktop.stopOllama();
    state.ollama = status;
    await loadAll();
    if (state.ollama.running) {
      throw new Error(state.ollama.lastError || "Ollama is still responding on 127.0.0.1:11434.");
    }
  });
}

async function runSequence(kind, stepIds) {
  await withMessage(async () => {
    state.run.active = true;
    state.run.kind = kind;
    state.run.currentStep = null;
    stepIds.forEach((stepId) => delete state.run.events[stepId]);
    render();
    try {
      for (const stepId of stepIds) {
        await runStep(stepId);
      }
    } finally {
      state.run.active = false;
      state.run.kind = null;
      state.run.currentStep = null;
      await loadAll();
    }
  });
}

async function runStep(stepId) {
  const startedAt = performance.now();
  state.run.currentStep = stepId;
  setEvent(stepId, "running", progressDetail(stepId, 0));
  await loadAll({ includeObservability: false });
  if (stepComplete(stepId)) {
    setEvent(stepId, "done", completionDetail(stepId, elapsedSeconds(startedAt), true));
    render();
    return;
  }
  if (stepId === "ollama") {
    await window.aegisDesktop.startOllama();
  } else if (stepId === "model") {
    await window.aegisDesktop.pullOllamaModel(state.launcher.profile.provider_model);
  } else if (stepId === "preflight") {
    await window.aegisDesktop.launcherApi({ method: "POST", path: "/api/preflight", body: null });
  } else {
    await window.aegisDesktop.launcherApi({ method: "POST", path: `/api/actions/${stepId}/start`, body: null });
  }
  await waitForStep(stepId, startedAt);
  setEvent(stepId, "done", completionDetail(stepId, elapsedSeconds(startedAt), false));
  render();
}

async function waitForStep(stepId, startedAt) {
  const timeout = timeouts[stepId];
  while (elapsedSeconds(startedAt) <= timeout) {
    await delay(1000);
    await loadAll({ includeObservability: false });
    const status = stepStatus(stepId);
    setEvent(stepId, "running", progressDetail(stepId, elapsedSeconds(startedAt)));
    if (status === "failed") {
      const detail = failureDetail(stepId);
      setEvent(stepId, "failed", detail);
      throw new Error(detail);
    }
    if (stepComplete(stepId)) return;
  }
  const detail = `${stepTitle(stepId)} did not finish within ${durationText(timeout)}.`;
  setEvent(stepId, "failed", detail);
  throw new Error(detail);
}

function stepComplete(stepId) {
  if (stepId === "ollama") return state.ollama.running;
  if (stepId === "model") return modelInstalled();
  if (stepId === "preflight") return ["passed", "warn"].includes(state.launcher.preflight.overall_status);
  if (stepId === "mps_preflight" && protectionIsRunning()) return true;
  if (stepId === "console" && protectionIsRunning()) return true;
  const process = processById(stepId);
  if (!process) return false;
  if (processKind(stepId) === "long-running") return process.status === "running";
  return process.status === "exited" && Number(process.exit_code || 0) === 0;
}

async function stopProtection() {
  await withMessage(async () => {
    await window.aegisDesktop.launcherApi({ method: "POST", path: "/api/actions/stop-all", body: null });
    await loadAll();
  });
}

async function clearStatuses() {
  await withMessage(async () => {
    await window.aegisDesktop.launcherApi({ method: "POST", path: "/api/actions/clear-statuses", body: null });
    state.run.events = {};
    state.observedFinishedAt = {};
    state.selectedTraceId = null;
    state.selectedTracePinned = false;
    await loadAll();
  });
}

async function updateModel(model) {
  const value = String(model || "").trim();
  if (value === "") return;
  await withMessage(async () => {
    await window.aegisDesktop.launcherApi({
      method: "PUT",
      path: "/api/profile",
      body: { provider_model: value },
    });
    await loadAll();
  });
}

function setEvent(stepId, status, detail) {
  state.run.events[stepId] = { status, detail };
}

function progressDetail(stepId, seconds) {
  const suffix = seconds > 0 ? ` ${durationText(seconds)} elapsed.` : "";
  if (stepId === "ollama") return `Starting or checking Ollama.${suffix}`;
  if (stepId === "model") return `Checking ${state.launcher.profile.provider_model}.${suffix}`;
  if (stepId === "preflight") return `Running readiness checks.${suffix}`;
  if (stepId === "mps_preflight") return `Checking ${runtimeDeviceLabel()}.${suffix}`;
  if (stepId === "cift_sidecar") return `Starting CIFT.${suffix}`;
  if (stepId === "gateway") return `Starting gateway.${suffix}`;
  if (stepId === "console") return `Starting console.${suffix}`;
  if (stepId === "cift_smoke") return `Running CIFT block check.${suffix}`;
  if (stepId === "real_provider_smoke") return `Running model allow check.${suffix}`;
  return `Running ${stepTitle(stepId)}.${suffix}`;
}

function completionDetail(stepId, seconds, alreadyComplete) {
  const duration = durationText(seconds);
  if (stepId === "ollama") return alreadyComplete ? "Ollama is already running." : `Ollama started in ${duration}.`;
  if (stepId === "model") return alreadyComplete ? `${state.launcher.profile.provider_model} is already installed.` : `${state.launcher.profile.provider_model} installed in ${duration}.`;
  if (stepId === "preflight") return `Checks finished in ${duration}.`;
  if (stepId === "mps_preflight") return `${runtimeDeviceLabel()} detected in ${duration}.`;
  if (stepId === "cift_sidecar") return alreadyComplete ? "CIFT is already running." : `CIFT started in ${duration}.`;
  if (stepId === "gateway") return alreadyComplete ? "Gateway is already running." : `Gateway started in ${duration}.`;
  if (stepId === "console") return alreadyComplete ? "Console is already running." : `Console started in ${duration}.`;
  if (stepId === "cift_smoke") return `CIFT block check passed in ${duration}.`;
  if (stepId === "real_provider_smoke") return `Model allow check passed in ${duration}.`;
  return `${stepTitle(stepId)} finished in ${duration}.`;
}

function failureDetail(stepId) {
  const process = processById(stepId);
  if (process) return `${stepTitle(stepId)} failed with exit ${exitCodeText(process)}.`;
  if (stepId === "model" && state.ollama.pull.status === "exited" && Number(state.ollama.pull.exitCode || 0) !== 0) {
    return `Model pull failed with exit ${state.ollama.pull.exitCode}.`;
  }
  return `${stepTitle(stepId)} failed.`;
}

function runTitle() {
  if (state.run.kind === "verify") return "Verification is running";
  if (state.run.kind === "single") return "Action is running";
  return "Starting protection";
}

function processById(actionId) {
  return (state.launcher.processes || {})[actionId];
}

function processKind(actionId) {
  const action = (state.launcher.actions || []).find((item) => item.action_id === actionId);
  return action?.kind || "one-shot";
}

function processStatus(actionId) {
  const process = processById(actionId);
  if (!process) return "stopped";
  if (process.status === "running") return "running";
  if (Number(process.exit_code || 0) === 0) return "stopped";
  return "failed";
}

function protectionIsRunning() {
  return processStatus("cift_sidecar") === "running" && processStatus("gateway") === "running";
}

function modelInstalled() {
  return (state.ollama.installedModels || []).includes(state.launcher.profile.provider_model);
}

function stepTitle(stepId) {
  const step = [...protectionSteps, ...verificationSteps].find((item) => item.id === stepId);
  return step ? step.title : stepId;
}

function runtimeDeviceLabel() {
  const device = state.launcher.profile.cift_binding.device;
  if (device === "mps") return "MPS";
  if (device === "cuda") return "CUDA";
  if (device === "cpu") return "CPU";
  return String(device).toUpperCase();
}

function displayDetectorName(value) {
  if (typeof value === "string") {
    const displayName = detectorDisplayNames.get(value);
    if (displayName) return displayName;
  }
  return displayIdentifier(value, "Unknown detector");
}

function displayStageName(value) {
  if (typeof value === "string") {
    const displayName = stageDisplayNames.get(value);
    if (displayName) return displayName;
  }
  return displayIdentifier(value, "Unknown stage");
}

function displayComponentName(value) {
  if (typeof value === "string") {
    const displayName = detectorDisplayNames.get(value);
    if (displayName) return displayName;
  }
  return displayIdentifier(value, "Component unknown");
}

function displayFieldName(value) {
  if (typeof value === "string") {
    const displayName = fieldDisplayNames.get(value);
    if (displayName) return displayName;
  }
  return displayIdentifier(value, "Detail");
}

function displayStatus(value, fallback) {
  if (typeof value === "string") {
    const displayName = statusDisplayNames.get(value);
    if (displayName) return displayName;
  }
  return displayValue(value, fallback);
}

function displayValue(value, fallback) {
  if (value === null || value === undefined) return fallback;
  if (typeof value !== "string") return String(value);
  const trimmed = value.trim();
  if (trimmed === "") return fallback;
  const displayName = valueDisplayNames.get(trimmed.toLowerCase()) || valueDisplayNames.get(trimmed);
  if (displayName) return displayName;
  return displayIdentifier(trimmed, fallback);
}

function displayStageFieldValue(key, value) {
  if (value === null || value === undefined) return "";
  if (key === "model_id") return String(value);
  if (key === "provider") return displayValue(value, String(value));
  if (key === "final_action" || key === "reason") return displayValue(value, String(value));
  return displayValue(value, String(value));
}

function displayIdentifier(value, fallback) {
  if (typeof value !== "string") return fallback;
  const trimmed = value.trim();
  if (trimmed === "") return fallback;
  return trimmed
    .split(/[_-]+/)
    .filter(Boolean)
    .map(displayToken)
    .join(" ");
}

function displayToken(value) {
  const lower = value.toLowerCase();
  const override = displayTokenOverrides.get(lower);
  if (override) return override;
  return `${lower.slice(0, 1).toUpperCase()}${lower.slice(1)}`;
}

function summaryRow(label, value) {
  return el("div", { class: "summary-row" }, el("span", { text: label }), el("strong", { class: "mono", text: value || "" }));
}

function processFact(label, value) {
  return el("div", { class: "process-fact" }, el("span", { text: label }), el("strong", { class: "mono", text: value }));
}

function badge(status) {
  return statusBadge(status, displayStatus(String(status), "Unknown"));
}

function stageBadge(stage) {
  const status = typeof stage.status === "string" ? stage.status : "unknown";
  if (stage.stage === "provider" && status === "skipped") return statusBadge(status, "Not called");
  if ((stage.stage === "canary" || stage.stage === "nimbus" || stage.stage === "dp_honey") && status === "unavailable") {
    return statusBadge(status, "Not used");
  }
  return statusBadge(status, displayStatus(status, "Unknown"));
}

function statusBadge(status, text) {
  return el("span", { class: `badge ${String(status).replaceAll("_", "-")}`, text });
}

function logItem(entry) {
  return el(
    "article",
    { class: "log-item" },
    el("h3", {}, el("span", { text: entry.label }), badge(entry.status)),
    el("pre", { class: "log-output", text: entry.logExcerpt }),
  );
}

function processSummary(process, status) {
  if (process.source === "external" && status === "done") return process.log_excerpt || "Already running.";
  if (status === "running") return `Running ${processElapsedText(process)}.`;
  if (status === "failed") return `Failed after ${processElapsedText(process)}.`;
  if (status === "done") return `Passed in ${processElapsedText(process)}.`;
  return process.status;
}

function exitCodeText(process) {
  if (process.exit_code === null || process.exit_code === undefined) return "unknown";
  return String(process.exit_code);
}

function processElapsedText(process) {
  const startedAt = Number(process.started_at);
  if (!Number.isFinite(startedAt)) return "unknown";
  if (process.status !== "running") {
    const storedSeconds = Number(process.runtime_seconds);
    if (Number.isFinite(storedSeconds)) return durationText(storedSeconds);
    const finishedAt = Number(process.finished_at);
    if (Number.isFinite(finishedAt)) return durationText(finishedAt - startedAt);
    return durationText(observedFinishedTime(process) - startedAt);
  }
  return durationText(Date.now() / 1000 - startedAt);
}

function observedFinishedTime(process) {
  const key = [process.action_id || "unknown", process.pid || "nopid", process.started_at || "nostart"].join(":");
  const current = state.observedFinishedAt[key];
  if (Number.isFinite(Number(current))) return Number(current);
  const observedAt = Number(process.observed_at);
  state.observedFinishedAt[key] = Number.isFinite(observedAt) ? observedAt : Date.now() / 1000;
  return state.observedFinishedAt[key];
}

function elapsedSeconds(startedAt) {
  return Math.max(0, (performance.now() - startedAt) / 1000);
}

function durationText(totalSeconds) {
  const seconds = Math.max(0, Math.floor(Number(totalSeconds)));
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  if (minutes > 0) return `${minutes}m ${String(remainingSeconds).padStart(2, "0")}s`;
  return `${remainingSeconds}s`;
}

function configurePolling() {
  if (state.pollTimer !== null) return;
  state.pollTimer = window.setTimeout(async () => {
    state.pollTimer = null;
    if (state.run.active) {
      configurePolling();
      return;
    }
    try {
      await loadAll();
    } catch (error) {
      showMessage(error instanceof Error ? error.message : String(error));
    }
    configurePolling();
  }, pollingDelay());
}

function pollingDelay() {
  if (state.run.active || state.observability?.error) return 5000;
  return 2500;
}

async function withMessage(action) {
  hideMessage();
  try {
    await action();
  } catch (error) {
    showMessage(error instanceof Error ? error.message : String(error));
  }
}

function showMessage(message) {
  const node = document.getElementById("message");
  node.textContent = message;
  node.hidden = false;
}

function hideMessage() {
  const node = document.getElementById("message");
  node.textContent = "";
  node.hidden = true;
}

function delay(milliseconds) {
  return new Promise((resolve) => {
    setTimeout(resolve, milliseconds);
  });
}

document.getElementById("primary-run").addEventListener("click", startProtection);
document.getElementById("verify-run").addEventListener("click", runVerification);
document.getElementById("refresh").addEventListener("click", () => withMessage(loadAll));
document.getElementById("open-web").addEventListener("click", () => window.aegisDesktop.openBackendInBrowser());
document.getElementById("start-ollama").addEventListener("click", toggleOllama);
document.getElementById("clear-statuses").addEventListener("click", clearStatuses);
document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => {
    const tab = button.getAttribute("data-tab");
    if (tab) {
      state.activeTab = tab;
      render();
    }
  });
});
document.getElementById("copy-agent").addEventListener("click", async () => {
  await navigator.clipboard.writeText(state.launcher.agent_settings.text);
});

loadAll().catch((error) => {
  showMessage(error instanceof Error ? error.message : String(error));
});
