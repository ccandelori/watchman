const state = {
  payload: null,
  pollTimer: null,
  observedFinishedAt: {},
  mode: window.localStorage.getItem("aegis-launcher-mode") === "advanced" ? "advanced" : "normal",
  openPopoverId: null,
  orchestrator: {
    running: false,
    cancelRequested: false,
    currentActionId: null,
    runKind: null,
    events: {},
  },
};

const editableFields = [
  ["name", "Profile name", "text", "Profile"],
  ["agent_kind", "Agent app", "text", "Agent"],
  ["provider_kind", "Provider kind", "text", "Provider"],
  ["provider_base_url", "Provider base URL", "text", "Provider"],
  ["provider_model", "Provider model", "text", "Provider"],
  ["provider_api_key", "Provider API key", "text", "Provider"],
  ["gateway_host", "Gateway host", "text", "Gateway"],
  ["gateway_port", "Gateway port", "number", "Gateway"],
  ["sidecar_host", "Sidecar host", "text", "CIFT"],
  ["sidecar_port", "Sidecar port", "number", "CIFT"],
  ["console_host", "Console host", "text", "Console"],
  ["console_port", "Console port", "number", "Console"],
  ["audit_jsonl_path", "Audit JSONL path", "text", "Audit"],
  ["cift_api_key", "Sidecar API key", "text", "CIFT"],
  ["mps_python_path", "CIFT Python path", "text", "CIFT"],
];

const workflowSteps = [
  {
    id: "open_ollama",
    title: "Ollama running",
    summary: "Use the local provider server.",
    evidence: "Ollama responds on the provider port.",
  },
  {
    id: "pull_model",
    title: "Model ready",
    summary: "Use the configured provider model.",
    evidence: "The model appears in Ollama tags.",
  },
  {
    id: "mps_preflight",
    title: "Check GPU",
    summary: "Verify the certified CIFT runtime device.",
    evidence: "The selected runtime device can allocate tensors.",
  },
  {
    id: "cift_sidecar",
    title: "Start CIFT",
    summary: "Run the hidden-state extractor.",
    evidence: "The sidecar attests the certified binding.",
  },
  {
    id: "gateway",
    title: "Start Gateway",
    summary: "Put Aegis in front of the provider.",
    evidence: "Readiness reports runtime-enforceable CIFT.",
  },
  {
    id: "console",
    title: "Open Console",
    summary: "Start the operator evidence view.",
    evidence: "The console can inspect gateway state.",
  },
  {
    id: "cift_smoke",
    title: "CIFT Smoke",
    summary: "Prove pre-generation block behavior.",
    evidence: "Exfiltration intent skips provider completion.",
  },
  {
    id: "real_provider_smoke",
    title: "Provider Smoke",
    summary: "Prove benign traffic reaches the model.",
    evidence: "Benign allow and exfil block both pass.",
  },
];

const normalStartSequence = ["preflight", "open_ollama", "pull_model", "mps_preflight", "cift_sidecar", "gateway", "console"];
const verificationSequence = ["cift_smoke", "real_provider_smoke"];

const sequenceTimeoutSeconds = {
  preflight: 45,
  open_ollama: 90,
  pull_model: 900,
  mps_preflight: 90,
  cift_sidecar: 180,
  gateway: 120,
  console: 120,
  cift_smoke: 300,
  real_provider_smoke: 300,
};

const setupSteps = [
  {
    id: "preflight",
    kind: "preflight",
    number: "1",
    title: "Run checks",
    summary: "Verify local tools, paths, ports, and model availability.",
    evidence: "The launcher has enough machine readiness to continue.",
  },
  ...workflowSteps.map((step, index) => ({
    ...step,
    kind: "action",
    number: String(index + 2),
  })),
];

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

function replaceNodeChildren(node, children) {
  node.replaceChildren(...children.filter((child) => child !== null && child !== undefined && child !== false));
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed with ${response.status}`);
  }
  return payload;
}

async function loadState() {
  state.payload = await api("/api/state");
  render();
}

function render() {
  if (!state.payload) return;
  renderStatus();
  renderPrerequisites();
  renderModeControls();
  if (state.mode === "advanced") {
    renderNextStep();
    renderRuntime();
  } else {
    renderNormalRun();
    renderNormalTimeline();
  }
  renderRunControls();
  renderAgentSettings();
  renderProfile();
  renderBinding();
  renderLogs();
  configureAutoRefresh();
}

function renderModeControls() {
  const normalButton = document.getElementById("mode-normal");
  const advancedButton = document.getElementById("mode-advanced");
  normalButton.classList.toggle("selected", state.mode === "normal");
  advancedButton.classList.toggle("selected", state.mode === "advanced");
  normalButton.setAttribute("aria-selected", state.mode === "normal" ? "true" : "false");
  advancedButton.setAttribute("aria-selected", state.mode === "advanced" ? "true" : "false");
  normalButton.disabled = state.orchestrator.running;
  advancedButton.disabled = state.orchestrator.running;
}

function renderStatus() {
  const profile = state.payload.profile;
  const processes = state.payload.processes || {};
  const running = Object.values(processes).filter((item) => item.status === "running").length;
  const status = document.getElementById("status");
  status.replaceChildren(
    el(
      "div",
      { class: "status-head" },
      el("strong", { text: protectionLabel() }),
      badge(state.payload.preflight.overall_status),
    ),
    el(
      "div",
      { class: "status-rows" },
      statusRow("Provider", `${profile.provider_kind} ${profile.provider_model}`),
      statusRow("Gateway", `${profile.gateway_host}:${profile.gateway_port}`),
      statusRow("CIFT", `${profile.cift_binding.model_id} on ${profile.cift_binding.device}`),
      statusRow("Running", String(running)),
    ),
  );
}

function renderNextStep() {
  const next = nextStep();
  const action = next.actionId ? actionById(next.actionId) : null;
  const process = next.actionId ? processById(next.actionId) : null;
  const isRunning = process && process.status === "running";
  const buttonAttrs = {
    id: "primary-action",
    class: "button primary active-action",
    type: "button",
    text: isRunning ? "Running" : next.button,
  };
  if (isRunning) buttonAttrs.disabled = true;
  replaceNodeChildren(document.getElementById("next-step"), [
    el(
      "div",
      { class: "active-kicker" },
      el("span", { class: "step-number large", text: next.number }),
      el("span", { class: "eyebrow", text: next.kind === "complete" ? "Ready" : "Current action" }),
      badge(next.status),
    ),
    el("h2", { text: next.title }),
    el("p", { class: "next-copy", text: next.detail }),
    process ? activeProcessPanel(process, next.status) : null,
    action
      ? el(
          "details",
          { class: "command-details active-command" },
          el("summary", { text: "Show command" }),
          el("pre", { class: "argv", text: action.argv.join(" ") }),
        )
      : null,
    el(
      "div",
      { class: "next-actions" },
      el("button", buttonAttrs),
      action && action.kind === "long-running" && isRunning
        ? el("button", { class: "button", type: "button", "data-stop-active": next.actionId, text: "Stop" })
        : null,
    ),
  ]);
  document.getElementById("primary-action").addEventListener("click", () => runNextStep(next));
  document.querySelectorAll("[data-stop-active]").forEach((button) => {
    button.addEventListener("click", () => stopAction(button.getAttribute("data-stop-active")));
  });
}

function renderRuntime() {
  const container = document.getElementById("runtime-list");
  const active = nextStep();
  container.replaceChildren(...runtimeSteps().map((step) => runtimeItem(step, active.stepId)));
}

function renderRunControls() {
  const processes = state.payload.processes || {};
  const running = hasRunningProcess();
  const hasStatuses = Object.keys(processes).length > 0;
  const clearButton = document.getElementById("clear-statuses");
  const stopButton = document.getElementById("stop-all");
  clearButton.disabled = running || !hasStatuses;
  clearButton.title = running ? "Stop running services before clearing statuses." : "Clear prior launcher statuses.";
  stopButton.disabled = !running;
}

function renderNormalRun() {
  const ready = protectionIsRunning();
  const running = state.orchestrator.running;
  const currentProcess = state.orchestrator.currentActionId ? processById(state.orchestrator.currentActionId) : null;
  const title = running ? normalRunTitle() : ready ? "Protection is running" : "Start protection";
  const copy = running
    ? "Keep this window open while the launcher works through the local checks and services."
    : ready
      ? "CIFT and the gateway are running. Run verification when you want fresh allow and block evidence."
      : "Checks the local provider, model, GPU, CIFT, gateway, and console in one run.";
  const startButtonAttrs = {
    id: "start-protection",
    class: ready ? "button danger active-action" : "button primary active-action",
    type: "button",
    text: running ? "Running" : ready ? "Stop Protection" : "Start Protection",
  };
  const verifyButtonAttrs = {
    id: "verify-protection",
    class: "button active-action",
    type: "button",
    text: "Run Verification",
  };
  if (running) startButtonAttrs.disabled = true;
  if (running || !ready) verifyButtonAttrs.disabled = true;
  replaceNodeChildren(document.getElementById("next-step"), [
    el(
      "div",
      { class: "active-kicker" },
      el("span", { class: "eyebrow", text: running ? "Normal mode" : ready ? "Ready" : "Normal mode" }),
      badge(running ? "running" : ready ? "done" : "waiting"),
    ),
    el("h2", { text: title }),
    el("p", { class: "next-copy", text: copy }),
    currentProcess ? activeProcessPanel(currentProcess, timelineStatus(state.orchestrator.currentActionId)) : null,
    el(
      "div",
      { class: "next-actions" },
      el("button", startButtonAttrs),
      el("button", verifyButtonAttrs),
    ),
  ]);
  document.getElementById("start-protection").addEventListener("click", ready ? stopAll : runProtectionSequence);
  document.getElementById("verify-protection").addEventListener("click", runVerificationSequence);
}

function renderNormalTimeline() {
  const container = document.getElementById("runtime-list");
  const rows = [
    el("div", { class: "timeline-group", text: "Start" }),
    ...normalStartSequence.map((actionId) => timelineItem(actionId)),
    el("div", { class: "timeline-group", text: "Verify" }),
    ...verificationSequence.map((actionId) => timelineItem(actionId)),
  ];
  container.replaceChildren(...rows);
}

function timelineItem(actionId) {
  const status = timelineStatus(actionId);
  const event = orchestratorEvent(actionId);
  const content = timelineStepContent(actionId, status);
  return el(
    "article",
    { class: `timeline-item ${status}` },
    el("div", { class: "timeline-marker", text: timelineMarkerText(actionId) }),
    el(
      "div",
      { class: "timeline-main" },
      el("h3", { text: content.title }),
      el("p", { text: event?.detail || content.summary }),
      content.meta ? el("span", { class: "runtime-meta", text: content.meta }) : null,
    ),
    badge(status),
  );
}

function runtimeItem(step, activeStepId) {
  const status = setupStepStatus(step);
  const active = step.id === activeStepId ? "active" : "";
  const content = runtimeStepContent(step, status);
  const process = processById(step.id);
  return el(
    "article",
    { class: `runtime-item ${status} ${active}` },
    el("div", { class: "step-number", text: step.number }),
    el(
      "div",
      { class: "runtime-main" },
      el("div", { class: "runtime-title" }, el("h3", { text: content.title })),
      el("p", { text: content.summary }),
      process ? el("span", { class: "runtime-meta", text: runtimeProcessSummary(process, status) }) : null,
    ),
    badge(status),
  );
}

function renderAgentSettings() {
  const settings = state.payload.agent_settings || {};
  const card = document.getElementById("agent-card");
  card.replaceChildren(
    settingRow("Base URL", settings.base_url),
    settingRow("API key", settings.api_key),
    settingRow("Model", settings.model),
  );
}

function renderPrerequisites() {
  const container = document.getElementById("prerequisite-list");
  const profile = state.payload.profile;
  const providerModels = state.payload.provider_models || {};
  const installedModels = Array.isArray(providerModels.installed) ? providerModels.installed : [];
  const readiness = readinessSummary();
  const serverCheck = checkByLabel("Ollama server") || { status: "waiting", detail: "Ollama has not been checked." };
  const modelCheck = checkByLabel("Provider model") || { status: "waiting", detail: "The model has not been checked." };
  container.replaceChildren(
    prerequisiteCard({
      id: "preflight",
      eyebrow: "Checks",
      title: "Machine ready",
      status: readiness.status,
      detail: readiness.status === "warn" ? "" : readiness.detail,
      rows: machineReadinessRows(readiness),
      actionLabel: "Run Checks",
    }),
    prerequisiteCard({
      id: "open_ollama",
      eyebrow: "Ollama",
      title: serverCheck.status === "passed" ? "Provider running" : "Provider unavailable",
      status: statusForCheck(serverCheck.status),
      detail: serverCheck.status === "passed" ? "" : "Start Ollama, then refresh the check.",
      rows: [
        settingRow("Host", hostFromUrl(profile.provider_base_url)),
        settingRow("API path", pathFromUrl(profile.provider_base_url)),
        settingRow("Health", "/api/tags"),
      ],
      actionLabel: "Open",
    }),
    prerequisiteCard({
      id: "pull_model",
      eyebrow: "Model",
      title: modelCheck.status === "passed" ? "Model installed" : "Model missing",
      status: statusForCheck(modelCheck.status),
      detail: modelCheck.status === "passed" ? "" : modelCheck.detail,
      rows: modelRows(profile.provider_model, installedModels),
      actionLabel: "Pull",
    }),
  );
  document.querySelectorAll("[data-prereq-action]").forEach((button) => {
    button.addEventListener("click", () => runPrerequisiteAction(button.getAttribute("data-prereq-action")));
  });
  const modelPicker = document.getElementById("model-picker");
  if (modelPicker instanceof HTMLSelectElement) {
    modelPicker.addEventListener("change", () => updateProviderModel(modelPicker.value));
  }
  document.querySelectorAll("[data-check-popover-trigger]").forEach((button) => {
    button.addEventListener("click", () => toggleCheckPopover(button.getAttribute("data-check-popover-trigger")));
  });
  document.querySelectorAll("[data-check-popover-close]").forEach((button) => {
    button.addEventListener("click", () => closeCheckPopover(button.getAttribute("data-check-popover-close")));
  });
}

function prerequisiteCard(config) {
  const showAction = config.status !== "done";
  return el(
    "article",
    { class: `prerequisite-card ${config.status}` },
    el(
      "div",
      { class: "prereq-card-head" },
      el("span", { class: "eyebrow", text: config.eyebrow }),
      badge(config.status),
    ),
    el("h3", { text: config.title }),
    config.detail ? el("p", { class: "prereq-detail", text: config.detail }) : null,
    el("div", { class: "prereq-rows" }, ...config.rows),
    showAction
      ? el("button", { class: "button primary prereq-action", type: "button", "data-prereq-action": config.id, text: config.actionLabel })
      : null,
  );
}

function machineReadinessRows(readiness) {
  return [
    settingRow("Failed", String(readiness.failed)),
    settingRow("Warnings", String(readiness.warned)),
    readiness.issues.length > 0 ? checkPopover(readiness.issues) : null,
  ];
}

function checkPopover(checks) {
  const failed = checks.filter((check) => check.status === "failed").length;
  const warned = checks.filter((check) => check.status === "warn").length;
  const title = failed > 0 ? `${failed} failed check${failed === 1 ? "" : "s"}` : `${warned} warning${warned === 1 ? "" : "s"}`;
  const popoverId = "machine-check-popover";
  const isOpen = state.openPopoverId === popoverId;
  return el(
    "div",
    { class: "check-popover-wrap" },
    el("span", { text: title }),
    el("button", {
      class: "button primary small check-popover-trigger",
      type: "button",
      "aria-expanded": isOpen ? "true" : "false",
      "aria-controls": popoverId,
      "data-check-popover-trigger": popoverId,
      text: "Show",
    }),
    el(
      "section",
      { id: popoverId, class: "check-popover", hidden: !isOpen },
      el(
        "div",
        { class: "check-popover-head" },
        el("strong", { text: "Machine checks" }),
        el("button", { class: "button small", type: "button", "data-check-popover-close": popoverId, text: "Close" }),
      ),
      el("div", { class: "check-detail-list" }, ...checks.map((check) => checkDetailRow(check))),
    ),
  );
}

function checkDetailRow(check) {
  return el(
    "div",
    { class: `check-detail-row ${statusForCheck(check.status)}` },
    el("span", { text: check.label }),
    el("strong", { text: check.detail }),
  );
}

function toggleCheckPopover(popoverId) {
  if (!popoverId) return;
  const popover = document.getElementById(popoverId);
  const trigger = document.querySelector(`[data-check-popover-trigger="${popoverId}"]`);
  if (!(popover instanceof HTMLElement) || !(trigger instanceof HTMLElement)) return;
  const shouldOpen = state.openPopoverId !== popoverId;
  closeCheckPopovers();
  state.openPopoverId = shouldOpen ? popoverId : null;
  popover.hidden = !shouldOpen;
  trigger.setAttribute("aria-expanded", shouldOpen ? "true" : "false");
}

function closeCheckPopover(popoverId) {
  if (!popoverId) return;
  const popover = document.getElementById(popoverId);
  const trigger = document.querySelector(`[data-check-popover-trigger="${popoverId}"]`);
  if (state.openPopoverId === popoverId) state.openPopoverId = null;
  if (popover instanceof HTMLElement) popover.hidden = true;
  if (trigger instanceof HTMLElement) trigger.setAttribute("aria-expanded", "false");
}

function closeCheckPopovers() {
  state.openPopoverId = null;
  document.querySelectorAll(".check-popover").forEach((popover) => {
    if (popover instanceof HTMLElement) popover.hidden = true;
  });
  document.querySelectorAll("[data-check-popover-trigger]").forEach((trigger) => {
    if (trigger instanceof HTMLElement) trigger.setAttribute("aria-expanded", "false");
  });
}

function modelRows(selectedModel, installedModels) {
  if (installedModels.length > 1) {
    return [
      controlRow(
        "Selected",
        el(
          "select",
          { id: "model-picker", class: "model-picker", "aria-label": "Provider model" },
          ...installedModels.map((model) => el("option", { value: model, selected: model === selectedModel, text: model })),
        ),
      ),
      settingRow("Installed", String(installedModels.length)),
    ];
  }
  if (installedModels.length === 1) {
    return [settingRow("Model", installedModels[0])];
  }
  return [settingRow("Model", selectedModel)];
}

function controlRow(label, control) {
  return el("div", { class: "setting-row control-row" }, el("span", { text: label }), control);
}

function renderPreflightSummary() {
  const preflight = state.payload.preflight;
  const checks = preflight.checks || [];
  const failed = checks.filter((check) => check.status === "failed");
  const warned = checks.filter((check) => check.status === "warn");
  const visibleChecks = [...failed, ...warned].slice(0, 5);
  const summary = document.getElementById("preflight-summary");
  if (visibleChecks.length === 0) {
    summary.replaceChildren(
      el("div", { class: "preflight-hero passed" }, el("strong", { text: "Checks passed" }), el("span", { text: "Ready to start services." })),
    );
    return;
  }
  summary.replaceChildren(
    el(
      "div",
      { class: `preflight-hero ${preflight.overall_status}` },
      el("strong", { text: `${failed.length} failed, ${warned.length} warning` }),
      el("span", { text: "Resolve failed checks before claiming protection." }),
    ),
    ...visibleChecks.map((check) =>
      el(
        "div",
        { class: "check-item compact-check" },
        el("div", {}, el("span", { text: check.label }), el("strong", { text: check.detail })),
        badge(check.status),
      ),
    ),
  );
}

function readinessSummary() {
  const preflight = state.payload.preflight;
  const checks = preflight.checks || [];
  const machineChecks = checks.filter((check) => !["Ollama server", "Provider model"].includes(check.label));
  const failed = machineChecks.filter((check) => check.status === "failed");
  const warned = machineChecks.filter((check) => check.status === "warn");
  const issues = [...failed, ...warned];
  if (failed.length > 0) {
    return {
      status: "failed",
      failed: failed.length,
      warned: warned.length,
      issues,
      detail: `${failed.length} check failed.`,
    };
  }
  if (warned.length > 0) {
    return {
      status: "warn",
      failed: 0,
      warned: warned.length,
      issues,
      detail: `${warned.length} check needs attention.`,
    };
  }
  return {
    status: "done",
    failed: 0,
    warned: 0,
    issues,
    detail: "Required local tools and paths are available.",
  };
}

function statusForCheck(status) {
  if (status === "passed") return "done";
  if (status === "failed") return "failed";
  if (status === "warn") return "warn";
  return "waiting";
}

function renderProfile() {
  const form = document.getElementById("profile-form");
  const profile = state.payload.profile;
  form.replaceChildren(...editableFields.map(([name, label, type, group]) => fieldNode(profile, name, label, type, group)));
}

function renderBinding() {
  const binding = state.payload.profile.cift_binding;
  const priorityKeys = [
    "model_id",
    "revision",
    "device",
    "feature_key",
    "freeform_feature_key",
    "selected_choice_readout_token_count",
    "hidden_size",
    "layer_count",
    "tokenizer_fingerprint_sha256",
    "special_tokens_map_sha256",
    "chat_template_sha256",
    "strict_deployment_env_path",
    "freeform_strict_deployment_env_path",
  ];
  document.getElementById("cift-binding").replaceChildren(
    ...priorityKeys.map((key) =>
      el(
        "div",
        { class: "binding-item" },
        el("span", { text: labelize(key) }),
        el("strong", { class: "mono", text: String(binding[key]) }),
      ),
    ),
  );
}

function renderLogs() {
  const list = document.getElementById("log-list");
  const entries = Object.values(state.payload.processes || {});
  if (entries.length === 0) {
    list.replaceChildren(el("p", { class: "muted", text: "No launcher-started process output yet." }));
    return;
  }
  list.replaceChildren(
    ...entries.map((process) =>
      el(
        "div",
        { class: "log-item" },
        el("h3", { text: `${process.label} (${process.status})` }),
        el("pre", { class: "log-output", text: process.log_excerpt || "No output yet." }),
      ),
    ),
  );
}

function fieldNode(profile, name, label, type, group) {
  return el(
    "div",
    { class: "field" },
    el("span", { class: "field-group", text: group }),
    el("label", { for: `field-${name}`, text: label }),
    el("input", {
      id: `field-${name}`,
      name,
      type,
      value: profile[name],
    }),
  );
}

function settingRow(label, value) {
  return el("div", { class: "setting-row" }, el("span", { text: label }), el("strong", { class: "mono", text: value || "" }));
}

function statusRow(label, value) {
  return el("div", { class: "status-row" }, el("span", { text: label }), el("strong", { class: "mono", text: value }));
}

function checkByLabel(label) {
  const checks = state.payload.preflight.checks || [];
  return checks.find((check) => check.label === label);
}

function originFromUrl(rawUrl) {
  try {
    return new URL(rawUrl).origin;
  } catch (_error) {
    return rawUrl;
  }
}

function hostFromUrl(rawUrl) {
  try {
    return new URL(rawUrl).host;
  } catch (_error) {
    return rawUrl;
  }
}

function pathFromUrl(rawUrl) {
  try {
    return new URL(rawUrl).pathname || "/";
  } catch (_error) {
    return rawUrl;
  }
}

function nextStep() {
  const preflightStatus = state.payload.preflight.overall_status;
  if (preflightStatus === "failed") {
    return {
      kind: "preflight",
      stepId: "preflight",
      number: "1",
      title: "Fix readiness checks",
      detail: "Resolve the failed local tool, path, port, or model checks.",
      button: "Run Checks",
      status: "failed",
    };
  }
  if (preflightStatus !== "passed" && preflightStatus !== "warn") {
    return {
      kind: "preflight",
      stepId: "preflight",
      number: "1",
      title: "Check this machine",
      detail: "Verify local tools, ports, paths, and the configured Ollama model.",
      button: "Run Checks",
      status: "waiting",
    };
  }
  const nextAction = workflowSteps.find((step) => !stepIsComplete(step));
  if (!nextAction) {
    return {
      kind: "complete",
      stepId: "complete",
      number: "✓",
      title: "Protection is ready",
      detail: "Copy the agent settings and keep the console open while you test.",
      button: "Copy Settings",
      status: "done",
    };
  }
  const setupStep = setupSteps.find((step) => step.id === nextAction.id);
  return {
    kind: "action",
    actionId: nextAction.id,
    stepId: nextAction.id,
    number: setupStep?.number || "",
    title: activeStepTitle(nextAction),
    detail: activeStepDetail(nextAction),
    button: actionLabel(nextAction.id),
    status: stepStatus(nextAction),
  };
}

async function runNextStep(next) {
  if (next.kind === "preflight") {
    await runPreflight();
    return;
  }
  if (next.kind === "complete") {
    await copyAgentSettings();
    return;
  }
  await startAction(next.actionId);
}

async function runPrerequisiteAction(actionId) {
  if (actionId === "preflight") {
    await runPreflight();
    return;
  }
  await startAction(actionId);
}

function processById(actionId) {
  return (state.payload.processes || {})[actionId];
}

function stepIsComplete(step) {
  const status = stepStatus(step);
  return status === "done" || (status === "running" && actionById(step.id)?.kind === "long-running");
}

function setupStepStatus(step) {
  if (step.kind === "preflight") {
    const status = state.payload.preflight.overall_status;
    if (status === "passed") return "done";
    if (status === "warn") return "warn";
    if (status === "failed") return "failed";
    return "waiting";
  }
  return stepStatus(step);
}

function stepStatus(step) {
  const liveStatus = liveActionStatus(step.id);
  if (liveStatus !== null) return liveStatus;
  const process = processById(step.id);
  if (!process) return "waiting";
  if (process.status === "running") return "running";
  if (Number(process.exit_code || 0) !== 0) return "failed";
  if (requiresLiveProof(step.id)) return "waiting";
  if (actionById(step.id)?.kind === "long-running") return "stopped";
  return "done";
}

function actionById(actionId) {
  return Object.fromEntries((state.payload.actions || []).map((action) => [action.action_id, action]))[actionId];
}

function liveActionStatus(actionId) {
  if (actionId === "open_ollama" && checkByLabel("Ollama server")?.status === "passed") return "done";
  if (actionId === "pull_model" && checkByLabel("Provider model")?.status === "passed") return "done";
  return null;
}

function requiresLiveProof(actionId) {
  return actionId === "open_ollama" || actionId === "pull_model";
}

function actionLabel(actionId) {
  if (actionId === "open_ollama") return "Open";
  if (actionId === "pull_model") return "Pull";
  if (actionId.endsWith("_smoke")) return "Run";
  if (actionId === "mps_preflight") return "Check";
  return "Start";
}

function activeStepTitle(step) {
  if (stepStatus(step) === "failed") return `${step.title} failed`;
  if (step.id === "open_ollama") return "Start Ollama";
  if (step.id === "pull_model") return `Pull ${state.payload.profile.provider_model}`;
  if (step.id === "mps_preflight" && stepStatus(step) === "done") return `${runtimeDeviceLabel()} verified`;
  return step.title;
}

function activeStepDetail(step) {
  const profile = state.payload.profile;
  const process = processById(step.id);
  if (stepStatus(step) === "failed" && process) return `${step.title} exited with code ${exitCodeText(process)}. Review the output below.`;
  if (process?.status === "running") return `${step.summary} Recent output appears below.`;
  if (step.id === "open_ollama") return `Ollama should respond at ${originFromUrl(profile.provider_base_url)}.`;
  if (step.id === "pull_model") return `${profile.provider_model} is not installed in Ollama.`;
  if (step.id === "mps_preflight") return gpuStepSummary(stepStatus(step));
  return step.summary;
}

function runtimeStepContent(step, status) {
  if (step.id !== "mps_preflight") {
    return { title: step.title, summary: step.summary };
  }
  if (status === "done") {
    return { title: `${runtimeDeviceLabel()} verified`, summary: "Certified runtime device passed the tensor check." };
  }
  if (status === "failed") {
    return { title: "GPU check failed", summary: `${runtimeDeviceLabel()} did not pass the runtime check.` };
  }
  return { title: "Check GPU", summary: gpuStepSummary(status) };
}

function timelineStepContent(actionId, status) {
  const process = processById(actionId);
  const step = setupSteps.find((item) => item.id === actionId) || workflowSteps.find((item) => item.id === actionId);
  if (actionId === "preflight") {
    return {
      title: "Machine checks",
      summary: readinessSummary().detail,
      meta: status === "done" ? "paths and ports ready" : null,
    };
  }
  if (actionId === "open_ollama") {
    return {
      title: "Ollama server",
      summary: checkByLabel("Ollama server")?.detail || "Waiting for Ollama.",
      meta: hostFromUrl(state.payload.profile.provider_base_url),
    };
  }
  if (actionId === "pull_model") {
    return {
      title: "Provider model",
      summary: checkByLabel("Provider model")?.detail || `${state.payload.profile.provider_model} has not been checked.`,
      meta: state.payload.profile.provider_model,
    };
  }
  if (actionId === "mps_preflight") {
    return {
      title: "GPU",
      summary: gpuTimelineSummary(status),
      meta: runtimeDeviceLabel(),
    };
  }
  if (process) {
    return {
      title: step ? step.title : actionId,
      summary: runtimeProcessSummary(process, status),
      meta: process.status === "running" ? "live" : null,
    };
  }
  return {
    title: step ? step.title : actionId,
    summary: step ? step.summary : "Waiting.",
    meta: null,
  };
}

function timelineMarkerText(actionId) {
  if (actionId === "preflight") return "1";
  const setupStep = setupSteps.find((step) => step.id === actionId);
  return setupStep?.number || "";
}

function timelineStatus(actionId) {
  const event = orchestratorEvent(actionId);
  if (event?.status === "running" || event?.status === "failed") return event.status;
  return sequenceStepStatus(actionId);
}

function sequenceStepStatus(actionId) {
  if (actionId === "preflight") return setupStepStatus(setupSteps[0]);
  const step = workflowSteps.find((item) => item.id === actionId);
  if (!step) return "waiting";
  return stepStatus(step);
}

function sequenceStepIsComplete(actionId) {
  const status = sequenceStepStatus(actionId);
  const action = actionById(actionId);
  if (actionId === "preflight") return status === "done" || status === "warn";
  return status === "done" || (status === "running" && action?.kind === "long-running");
}

function orchestratorEvent(actionId) {
  return state.orchestrator.events[actionId] || null;
}

function gpuTimelineSummary(status) {
  const device = runtimeDeviceLabel();
  if (status === "done") return `${device} detected and usable for this certificate.`;
  if (status === "failed") return `${device} check failed.`;
  return `Checking for ${device}.`;
}

function gpuStepSummary(status) {
  const device = runtimeDeviceLabel();
  if (status === "done") return `${device} passed the certified runtime check.`;
  if (status === "failed") return `${device} did not pass the certified runtime check.`;
  return `Verify ${device} for this CIFT certificate.`;
}

function runtimeDeviceLabel() {
  const device = state.payload.profile.cift_binding.device;
  if (device === "mps") return "MPS";
  if (device === "cuda") return "CUDA";
  if (device === "cpu") return "CPU";
  return String(device).toUpperCase();
}

function shortStepTitle(title) {
  if (title === "Run checks") return "Checks";
  if (title === "Ollama running") return "Ollama";
  if (title === "Model ready") return "Model";
  if (title === "Check GPU") return "GPU";
  if (title === "Start CIFT") return "CIFT";
  if (title === "Start Gateway") return "Gateway";
  if (title === "Open Console") return "Console";
  if (title === "CIFT Smoke") return "CIFT test";
  if (title === "Provider Smoke") return "Provider test";
  return title;
}

function runtimeSteps() {
  return setupSteps.filter((step) => !["preflight", "open_ollama", "pull_model"].includes(step.id));
}

function protectionLabel() {
  if (protectionIsRunning()) return "Protection running";
  const processes = state.payload.processes || {};
  const gateway = processes.gateway?.status === "running";
  if (gateway) return "Gateway running";
  return "Setup pending";
}

function protectionIsRunning() {
  const processes = state.payload.processes || {};
  return processes.cift_sidecar?.status === "running" && processes.gateway?.status === "running";
}

function normalRunTitle() {
  if (state.orchestrator.runKind === "verify") return "Verifying protection";
  return "Starting protection";
}

function badge(status) {
  return el("span", { class: `badge ${String(status).replaceAll("_", "-")}`, text: String(status) });
}

function activeProcessPanel(process, status) {
  return el(
    "section",
    { class: `process-panel ${status}` },
    el(
      "div",
      { class: "process-head" },
      el("strong", { text: process.label || process.action_id }),
      badge(processBadgeStatus(process, status)),
    ),
    el(
      "div",
      { class: "process-facts" },
      processFact(process.status === "running" ? "Elapsed" : "Duration", processElapsedText(process)),
      process.status === "running" ? processFact("PID", String(process.pid || "")) : processFact("Exit", exitCodeText(process)),
    ),
    el("pre", { class: "process-output", text: process.log_excerpt || "No output yet." }),
    process.log_path ? el("div", { class: "process-path mono", text: process.log_path }) : null,
  );
}

function processBadgeStatus(process, status) {
  if (status === "failed") return "failed";
  if (process.status === "running") return "running";
  if (Number(process.exit_code || 0) === 0) return "done";
  return "failed";
}

function runtimeProcessSummary(process, status) {
  if (status === "running") return `running ${processElapsedText(process)}`;
  if (status === "failed") return `failed after ${processElapsedText(process)}`;
  if (status === "done" && process.status === "exited") return `passed in ${processElapsedText(process)}`;
  if (status === "stopped") return `stopped after ${processElapsedText(process)}`;
  return process.status;
}

function processFact(label, value) {
  return el("div", { class: "process-fact" }, el("span", { text: label }), el("strong", { class: "mono", text: value }));
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
    const observedFinishedAt = observedFinishedTime(process);
    return durationText(observedFinishedAt - startedAt);
  }
  return durationText(Date.now() / 1000 - startedAt);
}

function observedFinishedTime(process) {
  const key = processIdentity(process);
  const current = state.observedFinishedAt[key];
  if (Number.isFinite(Number(current))) return Number(current);
  const observedAt = Number(process.observed_at);
  state.observedFinishedAt[key] = Number.isFinite(observedAt) ? observedAt : Date.now() / 1000;
  return state.observedFinishedAt[key];
}

function processIdentity(process) {
  return [process.action_id || "unknown", process.pid || "nopid", process.started_at || "nostart"].join(":");
}

function durationText(totalSeconds) {
  const seconds = Math.max(0, Math.floor(Number(totalSeconds)));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainingSeconds = seconds % 60;
  if (hours > 0) return `${hours}h ${String(minutes).padStart(2, "0")}m ${String(remainingSeconds).padStart(2, "0")}s`;
  if (minutes > 0) return `${minutes}m ${String(remainingSeconds).padStart(2, "0")}s`;
  return `${remainingSeconds}s`;
}

function configureAutoRefresh() {
  if (!state.payload) return;
  if (hasRunningProcess() && state.pollTimer === null) {
    state.pollTimer = window.setInterval(refreshFromPoll, 2000);
    return;
  }
  if (!hasRunningProcess() && state.pollTimer !== null) {
    stopAutoRefresh();
  }
}

function hasRunningProcess() {
  return Object.values(state.payload.processes || {}).some((process) => process.status === "running");
}

function stopAutoRefresh() {
  if (state.pollTimer !== null) {
    window.clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

async function refreshFromPoll() {
  try {
    state.payload = await api("/api/state");
    render();
  } catch (error) {
    stopAutoRefresh();
    showMessage(error.message || String(error));
  }
}

function labelize(key) {
  return key.replaceAll("_", " ");
}

function collectProfileForm() {
  const data = {};
  const form = document.getElementById("profile-form");
  editableFields.forEach(([name, _label, type]) => {
    const input = form.elements[name];
    data[name] = type === "number" ? Number(input.value) : input.value;
  });
  return data;
}

async function saveProfile() {
  await withMessageHandling(async () => {
    state.payload = await api("/api/profile", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectProfileForm()),
    });
    render();
  });
}

async function updateProviderModel(providerModel) {
  await withMessageHandling(async () => {
    state.payload = await api("/api/profile", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider_model: providerModel }),
    });
    render();
  });
}

async function runPreflight() {
  await withMessageHandling(async () => {
    state.payload.preflight = await api("/api/preflight", { method: "POST" });
    await loadState();
  });
}

async function startAction(actionId) {
  if (!actionId) return;
  await withMessageHandling(async () => {
    await api(`/api/actions/${actionId}/start`, { method: "POST" });
    await loadState();
  });
}

async function stopAction(actionId) {
  if (!actionId) return;
  await withMessageHandling(async () => {
    await api(`/api/actions/${actionId}/stop`, { method: "POST" });
    await loadState();
  });
}

async function stopAll() {
  await withMessageHandling(async () => {
    state.orchestrator.cancelRequested = true;
    await api("/api/actions/stop-all", { method: "POST" });
    await loadState();
  });
}

async function clearStatuses() {
  await withMessageHandling(async () => {
    await api("/api/actions/clear-statuses", { method: "POST" });
    state.observedFinishedAt = {};
    state.orchestrator.events = {};
    await loadState();
  });
}

function setLauncherMode(mode) {
  state.mode = mode;
  window.localStorage.setItem("aegis-launcher-mode", mode);
  render();
}

async function runProtectionSequence() {
  await runOrchestratedSequence("start", normalStartSequence);
}

async function runVerificationSequence() {
  await runOrchestratedSequence("verify", verificationSequence);
}

async function runOrchestratedSequence(runKind, actionIds) {
  await withMessageHandling(async () => {
    state.orchestrator.running = true;
    state.orchestrator.cancelRequested = false;
    state.orchestrator.currentActionId = null;
    state.orchestrator.runKind = runKind;
    actionIds.forEach((actionId) => clearOrchestratorEvent(actionId));
    render();
    try {
      for (const actionId of actionIds) {
        await runOrchestratedStep(actionId);
      }
    } finally {
      state.orchestrator.running = false;
      state.orchestrator.currentActionId = null;
      state.orchestrator.runKind = null;
      await loadState();
    }
  });
}

async function runOrchestratedStep(actionId) {
  const startedAt = performance.now();
  state.orchestrator.currentActionId = actionId;
  setOrchestratorEvent(actionId, {
    status: "running",
    detail: sequenceProgressDetail(actionId),
    startedAt,
    durationSeconds: null,
  });
  try {
    await loadState();
    assertSequenceNotCancelled();
    if (sequenceStepIsComplete(actionId)) {
      setOrchestratorEvent(actionId, {
        status: "done",
        detail: sequenceCompletionDetail(actionId, elapsedSeconds(startedAt), true),
        durationSeconds: elapsedSeconds(startedAt),
      });
      render();
      return;
    }
    if (actionId === "preflight") {
      state.payload.preflight = await api("/api/preflight", { method: "POST" });
    } else {
      await api(`/api/actions/${actionId}/start`, { method: "POST" });
    }
    await waitForSequenceStep(actionId, startedAt);
    setOrchestratorEvent(actionId, {
      status: "done",
      detail: sequenceCompletionDetail(actionId, elapsedSeconds(startedAt), false),
      durationSeconds: elapsedSeconds(startedAt),
    });
    render();
  } catch (error) {
    const event = orchestratorEvent(actionId);
    if (event?.status !== "failed") {
      setOrchestratorEvent(actionId, {
        status: "failed",
        detail: error.message || String(error),
        durationSeconds: elapsedSeconds(startedAt),
      });
      render();
    }
    throw error;
  }
}

async function waitForSequenceStep(actionId, startedAt) {
  const timeoutSeconds = sequenceTimeoutSeconds[actionId];
  while (elapsedSeconds(startedAt) <= timeoutSeconds) {
    await delay(1000);
    await loadState();
    assertSequenceNotCancelled();
    const status = sequenceStepStatus(actionId);
    if (status === "failed") {
      setOrchestratorEvent(actionId, {
        status: "failed",
        detail: sequenceFailureDetail(actionId),
        durationSeconds: elapsedSeconds(startedAt),
      });
      render();
      throw new Error(sequenceFailureDetail(actionId));
    }
    if (status === "stopped") {
      const detail = `${sequenceTitle(actionId)} stopped after ${durationText(elapsedSeconds(startedAt))}.`;
      setOrchestratorEvent(actionId, {
        status: "failed",
        detail,
        durationSeconds: elapsedSeconds(startedAt),
      });
      render();
      throw new Error(detail);
    }
    if (sequenceStepIsComplete(actionId)) return;
  }
  const detail = `${sequenceTitle(actionId)} did not finish within ${durationText(timeoutSeconds)}.`;
  setOrchestratorEvent(actionId, {
    status: "failed",
    detail,
    durationSeconds: elapsedSeconds(startedAt),
  });
  render();
  throw new Error(detail);
}

function assertSequenceNotCancelled() {
  if (!state.orchestrator.cancelRequested) return;
  throw new Error("Launch stopped.");
}

function setOrchestratorEvent(actionId, event) {
  state.orchestrator.events[actionId] = event;
}

function clearOrchestratorEvent(actionId) {
  delete state.orchestrator.events[actionId];
}

function sequenceProgressDetail(actionId) {
  if (actionId === "preflight") return "Checking local tools, paths, and ports.";
  if (actionId === "open_ollama") return `Checking Ollama at ${originFromUrl(state.payload.profile.provider_base_url)}.`;
  if (actionId === "pull_model") return `Checking ${state.payload.profile.provider_model}.`;
  if (actionId === "mps_preflight") return `Checking ${runtimeDeviceLabel()}.`;
  if (actionId === "cift_sidecar") return "Starting CIFT.";
  if (actionId === "gateway") return "Starting the local gateway.";
  if (actionId === "console") return "Starting the console.";
  if (actionId === "cift_smoke") return "Running the CIFT block check.";
  if (actionId === "real_provider_smoke") return "Running the provider allow check.";
  return `Running ${sequenceTitle(actionId)}.`;
}

function sequenceCompletionDetail(actionId, seconds, alreadyComplete) {
  const duration = durationText(seconds);
  if (actionId === "preflight" && sequenceStepStatus(actionId) === "warn") return `Checks completed with warnings in ${duration}.`;
  if (actionId === "preflight") return `Checks passed in ${duration}.`;
  if (actionId === "open_ollama") return alreadyComplete ? "Ollama is already running." : `Ollama responded in ${duration}.`;
  if (actionId === "pull_model") return alreadyComplete ? `${state.payload.profile.provider_model} is already installed.` : `${state.payload.profile.provider_model} installed in ${duration}.`;
  if (actionId === "mps_preflight") return `${runtimeDeviceLabel()} detected in ${duration}.`;
  if (actionId === "cift_sidecar") return alreadyComplete ? "CIFT is already running." : `CIFT started in ${duration}.`;
  if (actionId === "gateway") return alreadyComplete ? "Gateway is already running." : `Gateway started in ${duration}.`;
  if (actionId === "console") return alreadyComplete ? "Console is already running." : `Console started in ${duration}.`;
  if (actionId === "cift_smoke") return `CIFT smoke passed in ${duration}.`;
  if (actionId === "real_provider_smoke") return `Provider smoke passed in ${duration}.`;
  return `${sequenceTitle(actionId)} finished in ${duration}.`;
}

function sequenceFailureDetail(actionId) {
  const process = processById(actionId);
  if (process) return `${sequenceTitle(actionId)} failed after ${processElapsedText(process)}.`;
  return `${sequenceTitle(actionId)} failed.`;
}

function sequenceTitle(actionId) {
  if (actionId === "preflight") return "Machine checks";
  const step = setupSteps.find((item) => item.id === actionId) || workflowSteps.find((item) => item.id === actionId);
  return step ? step.title : actionId;
}

function elapsedSeconds(startedAt) {
  return (performance.now() - startedAt) / 1000;
}

function delay(milliseconds) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, milliseconds);
  });
}

async function copyAgentSettings() {
  const settings = state.payload.agent_settings || {};
  await navigator.clipboard.writeText(settings.text || "");
  const button = document.getElementById("copy-agent");
  button.textContent = "Copied";
  window.setTimeout(() => {
    button.textContent = "Copy";
  }, 1200);
}

async function withMessageHandling(operation) {
  clearMessage();
  try {
    await operation();
  } catch (error) {
    showMessage(error.message || String(error));
  }
}

function showMessage(message) {
  const node = document.getElementById("message");
  node.hidden = false;
  node.textContent = message;
}

function clearMessage() {
  const node = document.getElementById("message");
  node.hidden = true;
  node.textContent = "";
}

document.getElementById("refresh").addEventListener("click", loadState);
document.getElementById("save-profile").addEventListener("click", saveProfile);
document.getElementById("stop-all").addEventListener("click", stopAll);
document.getElementById("clear-statuses").addEventListener("click", clearStatuses);
document.getElementById("copy-agent").addEventListener("click", copyAgentSettings);
document.getElementById("mode-normal").addEventListener("click", () => setLauncherMode("normal"));
document.getElementById("mode-advanced").addEventListener("click", () => setLauncherMode("advanced"));
document.addEventListener("click", (event) => {
  if (!(event.target instanceof Element)) return;
  if (event.target.closest(".check-popover-wrap")) return;
  closeCheckPopovers();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeCheckPopovers();
});

loadState().catch((error) => {
  document.getElementById("status").replaceChildren(
    el("div", { class: "status-head" }, el("strong", { text: "Launcher failed" }), badge("failed")),
    el("div", { class: "status-row" }, el("span", { text: "Error" }), el("strong", { text: error.message })),
  );
});
