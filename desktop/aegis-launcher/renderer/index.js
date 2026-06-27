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
  { id: "model", title: "Provider model", summary: "Use the selected model in Ollama." },
  { id: "preflight", title: "Machine checks", summary: "Check paths, ports, and local tools." },
  { id: "mps_preflight", title: "GPU", summary: "Verify the certified runtime device." },
  { id: "cift_sidecar", title: "CIFT", summary: "Start hidden-state extraction." },
  { id: "gateway", title: "Gateway", summary: "Put Aegis in front of the model." },
  { id: "console", title: "Console", summary: "Start the operator evidence view." },
];

const verificationSteps = [
  { id: "cift_smoke", title: "CIFT smoke", summary: "Prove exfiltration intent skips provider completion." },
  { id: "real_provider_smoke", title: "Provider smoke", summary: "Prove benign traffic reaches the local model." },
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
  const detectorRows = Object.entries(detectorByName).map(([name, stats]) => latencyRow(name, stats));
  container.replaceChildren(
    metricCard("Aegis request p50", formatMs(requestStats.p50), `${requestStats.count || 0} recent audited requests`),
    metricCard("Aegis request p95", formatMs(requestStats.p95), latencyRangeText(requestStats)),
    metricCard("Detector p50", formatMs(detectorStats.p50), `${detectorStats.count || 0} detector timings`),
    metricCard("Direct provider baseline", "Not measured", latency.direct_provider_baseline?.detail || "No paired benchmark."),
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
    detectorCard("DP-HONEY", `${activity.dp_honey_substitutions || 0}`, "honeytoken substitutions", "credential-slot evidence"),
    detectorCard("CIFT", `${activity.cift_pre_generation_blocks || 0}`, "pre-generation blocks", cift.support_scope || "model-specific hidden-state check"),
    detectorCard("Canaries", `${activity.canary_detections || 0}`, "canary detections", "response and tool-output leak checks"),
    detectorCard("NIMBUS", `${activity.nimbus_warnings || 0}`, "warnings or stronger", nimbus.label || "session leakage critic"),
    detectorCard("Provider guard", `${activity.provider_egress_blocks || 0}`, "provider egress blocks", "pre-provider value scanning"),
    detectorCard("Fail closed", `${activity.fail_closed_events || 0}`, "fail-closed events", "runtime refusal evidence"),
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
      metricCard("Final action", event.final_action || "unknown", event.reason || "no reason available"),
      metricCard("Provider", event.provider_status || "unknown", event.provider || "provider evidence unavailable"),
      metricCard("Latency", formatMs(event.latency_ms), event.trace_id || "unknown trace"),
    ),
    el("section", { class: "stage-rail" }, ...stageTimeline(event.stage_timeline || [])),
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
  const detectors = Array.isArray(event.detectors_fired) ? event.detectors_fired : [];
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
      el("span", { class: "mono event-trace", text: event.trace_id || "unknown trace" }),
      el("strong", { text: event.final_action || "unknown action" }),
      el("p", { text: event.reason || "No policy reason recorded." }),
    ),
    el(
      "div",
      { class: "event-meta" },
      badge(event.provider_status || "unknown"),
      el("span", { class: "mono", text: formatMs(event.latency_ms) }),
      detectors.length > 0 ? el("span", { text: detectors.join(", ") }) : el("span", { text: "no detector fired" }),
    ),
  );
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
    el("strong", { text: name }),
    el("span", { text: `p50 ${formatMs(stats.p50)}` }),
    el("span", { text: `p95 ${formatMs(stats.p95)}` }),
    el("span", { text: `${stats.count || 0} samples` }),
  );
}

function stageTimeline(stages) {
  if (!Array.isArray(stages) || stages.length === 0) {
    return [emptyState("No timeline", "This trace has no stage timeline.")];
  }
  return stages.map((stage) =>
    el(
      "article",
      { class: `stage-item ${stage.status || "unknown"}` },
      badge(stage.status || "unknown"),
      el("strong", { text: stage.stage || "unknown stage" }),
      el("p", { text: stageDetail(stage) }),
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
        el("strong", { text: detector.detector_name || "unknown" }),
        badge(detector.recommended_action || "allow"),
        el("span", { text: detector.component || "component unknown" }),
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
    summaryRow("Policy mode", runtimeEvidence.policy_mode || "unknown"),
    summaryRow("Credential slot", runtimeEvidence.credential_slot_status || "unknown"),
    summaryRow("Provider state", providerState.status || "unknown"),
    summaryRow("Fail closed", String(failClosed)),
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

function stageDetail(stage) {
  const keys = ["reason", "provider", "model_id", "final_action", "credential_needed_count", "honeytoken_substituted_count", "canary_count"];
  const values = keys
    .map((key) => {
      const value = stage[key];
      return value === null || value === undefined ? null : `${key}: ${value}`;
    })
    .filter(Boolean);
  if (values.length > 0) return values.join(" | ");
  return stage.detail || stage.original_status || "No additional detail.";
}

function latencyRangeText(stats) {
  if (!stats || stats.count === 0) return "No recent latency samples";
  return `range ${formatMs(stats.min)} to ${formatMs(stats.max)}`;
}

function formatMs(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "n/a";
  if (number >= 1000) return `${(number / 1000).toFixed(2)}s`;
  return `${number.toFixed(number >= 10 ? 0 : 1)}ms`;
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
  if (stepId === "cift_smoke") return `Running CIFT smoke.${suffix}`;
  if (stepId === "real_provider_smoke") return `Running provider smoke.${suffix}`;
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
  if (stepId === "cift_smoke") return `CIFT smoke passed in ${duration}.`;
  if (stepId === "real_provider_smoke") return `Provider smoke passed in ${duration}.`;
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

function summaryRow(label, value) {
  return el("div", { class: "summary-row" }, el("span", { text: label }), el("strong", { class: "mono", text: value || "" }));
}

function processFact(label, value) {
  return el("div", { class: "process-fact" }, el("span", { text: label }), el("strong", { class: "mono", text: value }));
}

function badge(status) {
  return el("span", { class: `badge ${String(status).replaceAll("_", "-")}`, text: String(status) });
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
