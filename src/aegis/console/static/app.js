"use strict";

const state = {
  tab: "overview",
  overview: null,
  events: null,
  setup: null,
  expandedTraceId: null,
  error: null,
};

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([key, value]) => {
    if (key === "class") node.className = value;
    else if (key === "dataset") {
      Object.entries(value).forEach(([dataKey, dataValue]) => {
        node.dataset[dataKey] = dataValue;
      });
    }
    else node.setAttribute(key, value);
  });
  children.forEach((child) => {
    if (child === null || child === undefined) return;
    if (typeof child === "string") node.append(document.createTextNode(child));
    else node.append(child);
  });
  return node;
}

async function api(path) {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

async function refresh() {
  state.error = null;
  try {
    const [overview, events, setup] = await Promise.all([
      api("/api/overview"),
      api("/api/events?limit=50"),
      api("/api/setup"),
    ]);
    state.overview = overview;
    state.events = events;
    state.setup = setup;
  } catch (error) {
    state.error = error.message;
  }
  render();
}

function render() {
  document.querySelectorAll(".tabs button").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === state.tab);
  });
  if (state.error) {
    document.getElementById("status").replaceChildren();
    document.getElementById("view").replaceChildren(el("div", { class: "error" }, state.error));
    return;
  }
  renderStatus();
  if (state.tab === "overview") renderOverview();
  if (state.tab === "requests") renderRequests();
  if (state.tab === "model") renderModel();
  if (state.tab === "detectors") renderDetectors();
  if (state.tab === "setup") renderSetup();
}

function renderStatus() {
  const overview = state.overview || {};
  const protection = overview.protection || {};
  const model = overview.model || {};
  const nimbus = overview.nimbus || {};
  document.getElementById("gateway-line").textContent = `Gateway: ${overview.gateway?.base_url || "unknown"}`;
  document.getElementById("status").replaceChildren(
    statusTile("Protection", protection.state || "Checking", protection.severity || "degraded"),
    statusTile("Profile", protection.active_profile || "unknown", "info"),
    statusTile("Model", model.model_id || "unknown", "info"),
    statusTile("NIMBUS", nimbus.label || "unknown", "degraded"),
  );
}

function statusTile(label, value, tone) {
  return el(
    "div",
    { class: "status-tile" },
    el("div", { class: "status-label" }, label),
    el("div", { class: `status-value ${tone}` }, String(value)),
  );
}

function renderOverview() {
  const overview = state.overview || {};
  const view = document.getElementById("view");
  view.replaceChildren(
    el(
      "div",
      { class: "grid" },
      panel("Protection Checklist", tableChecklist(overview.checklist || []), "span-8"),
      panel("Last Request", lastRequestView(overview.last_request), "span-4"),
      panel("CIFT", keyValueTable(ciftRows(overview.cift || {})), "span-6"),
      panel("Audit & Smoke", keyValueTable(auditRows(overview)), "span-6"),
    ),
  );
}

function renderRequests() {
  const eventsPayload = state.events || {};
  const events = eventsPayload.events || [];
  const source = eventsPayload.source || "unknown";
  const view = document.getElementById("view");
  const body = events.length === 0 ? el("p", { class: "muted" }, "No audit events are available.") : eventsTable(events);
  view.replaceChildren(
    el(
      "div",
      { class: "grid" },
      panel(`Recent Decisions (${source})`, body, "span-12"),
    ),
  );
}

function renderModel() {
  const overview = state.overview || {};
  const model = overview.model || {};
  const cift = overview.cift || {};
  const smoke = overview.last_smoke || {};
  const rows = [
    ["Model id", model.model_id],
    ["Revision", model.revision],
    ["Device", model.device || cift.source_selected_device],
    ["Hidden size", model.hidden_size || cift.source_hidden_size],
    ["Layer count", model.layer_count || cift.source_layer_count],
    ["Tokenizer hash", model.tokenizer_fingerprint_sha256 || cift.tokenizer_fingerprint_sha256],
    ["Chat-template hash", model.chat_template_sha256 || cift.chat_template_sha256],
    ["CIFT support tier", cift.support_tier],
    ["CIFT support scope", cift.support_scope],
    ["CIFT support reason", cift.support_reason],
    ["CIFT certificate", cift.certificate_status],
    ["Certification id", cift.certification_id],
    ["Certification mode", cift.certification_mode],
    ["Runtime SHA-256", cift.runtime_model_sha256],
    ["Release gate SHA-256", cift.release_gate_report_sha256],
    ["Runtime bundle", cift.runtime_model_bundle_id || smoke.gateway_readiness?.model_bundle_id],
    ["Selected layer/window/readout", cift.feature_key],
    ["Readout token count", cift.selected_choice_readout_token_count],
    ["Winning probe", winningProbe(cift.runtime_model_bundle_id || smoke.gateway_readiness?.model_bundle_id)],
    ["Grouped CV / sealed / live metrics", liveSmokeMetrics(smoke)],
    ["FN / FP", fnFpSummary(smoke.confusion_metrics)],
    ["Certification report", cift.certification_id || smoke.gateway_readiness?.certification_id],
  ];
  document.getElementById("view").replaceChildren(panel("Model & CIFT Certification", keyValueTable(rows), "span-12"));
}

function renderDetectors() {
  const activity = state.events?.detector_activity || {};
  const metrics = [
    ["CIFT blocks", activity.cift_pre_generation_blocks || 0],
    ["DP-HONEY substitutions", activity.dp_honey_substitutions || 0],
    ["Canary detections", activity.canary_detections || 0],
    ["Provider egress blocks", activity.provider_egress_blocks || 0],
    ["NIMBUS warnings", activity.nimbus_warnings || 0],
    ["Fail-closed events", activity.fail_closed_events || 0],
  ];
  const metricNodes = metrics.map(([label, value]) => el("div", { class: "metric" }, el("strong", {}, String(value)), el("span", {}, label)));
  document.getElementById("view").replaceChildren(
    el(
      "div",
      { class: "grid" },
      panel("Detector Activity", el("div", { class: "activity-grid" }, ...metricNodes), "span-12"),
      panel("NIMBUS Status", nimbusBetaView(state.overview?.nimbus || {}), "span-12"),
    ),
  );
}

function renderSetup() {
  const setup = state.setup || {};
  const overview = state.overview || {};
  const gateway = overview.gateway || {};
  const readiness = gateway.readiness || {};
  const readyPayload = readiness.payload || {};
  const capabilitiesPayload = gateway.capabilities?.payload || {};
  const providerReady = readyPayload.provider || {};
  const providerCapabilities = capabilitiesPayload.provider || {};
  const ciftReady = readyPayload.cift || {};
  const ciftSidecar = setup.cift_sidecar || {};
  const commands = setup.commands || {};
  const rows = [
    ["Gateway base URL", setup.gateway_base_url],
    ["Agent app base URL", setup.agent_gateway_base_url],
    ["OpenAI-compatible endpoint", setup.openai_compatible_endpoint],
    ["Example agent API key", setup.agent_settings?.api_key],
    ["Example agent model", setup.agent_settings?.model],
    ["Active provider mode", providerReady.name || providerCapabilities.name],
    ["Provider target URL", providerReady.target_url || providerCapabilities.target_url],
    ["Mock controls enabled", providerReady.mock_controls_enabled ?? providerCapabilities.mock_controls_enabled],
    ["Current /ready fetch", readiness.status],
    ["Current /ready status", readyPayload.status],
    ["Current /ready boolean", readyPayload.ready],
    ["Protection state", overview.protection?.state],
    ["Default CIFT sidecar URL", ciftSidecar.default_base_url],
    ["CIFT sidecar status source", ciftSidecar.status_source],
    ["CIFT support tier", ciftReady.support_tier],
    ["CIFT support scope", ciftReady.support_scope],
    ["CIFT /ready status", ciftReady.status],
    ["CIFT device", ciftReady.source_selected_device || ciftReady.extractor?.selected_device],
  ];
  const commandBlocks = Object.entries(commands).map(([name, command]) =>
    el("div", { class: "panel" }, el("h3", {}, commandLabel(name)), el("pre", {}, String(command))),
  );
  document.getElementById("view").replaceChildren(
    el(
      "div",
      { class: "grid" },
      panel("Connection", keyValueTable(rows), "span-12"),
      panel("Agent Settings", keyValueTable(Object.entries(setup.agent_settings || {}).map(([key, value]) => [commandLabel(key), value])), "span-6"),
      panel("Request Path", setupArchitecture(setup.architecture || []), "span-6"),
      panel("Provider Examples", providerExamples(setup.provider_examples || []), "span-12"),
      panel("Smoke Evidence", setupList(setup.smoke_expectations || []), "span-12"),
      el("section", { class: "span-12 grid" }, ...commandBlocks),
      panel("Common Degraded States", degradedStates(setup.common_degraded_states || []), "span-12"),
    ),
  );
}

function panel(title, content, spanClass) {
  return el("section", { class: `panel ${spanClass}` }, el("h2", {}, title), content);
}

function tableChecklist(items) {
  const rows = items.map((item) =>
    el(
      "tr",
      {},
      el("td", {}, el("span", { class: `badge ${item.status}` }, item.status || "unknown")),
      el("td", {}, item.label || ""),
      el("td", { class: "muted" }, item.detail || ""),
    ),
  );
  return el("table", { class: "checklist" }, el("tbody", {}, ...rows));
}

function keyValueTable(rows) {
  const tableRows = rows.map(([key, value]) =>
    el(
      "tr",
      {},
      el("td", {}, key),
      el("td", { class: "mono" }, value === null || value === undefined || value === "" ? "not available" : String(value)),
    ),
  );
  return el("table", { class: "kv" }, el("tbody", {}, ...tableRows));
}

function ciftRows(cift) {
  return [
    ["Support tier", cift.support_tier],
    ["Support scope", cift.support_scope],
    ["Support reason", cift.support_reason],
    ["Status", cift.certificate_status],
    ["Mode", cift.capability_mode],
    ["Certification", cift.certification_id],
    ["Feature key", cift.feature_key],
    ["Device", cift.source_selected_device || cift.extractor?.selected_device],
    ["Runtime SHA-256", cift.runtime_model_sha256],
  ];
}

function winningProbe(bundleId) {
  const text = String(bundleId || "").toLowerCase();
  if (text.includes("paper_mlp") || text.includes("paper-mlp")) return "paper MLP";
  if (text.includes("linear")) return "linear";
  return "not available";
}

function liveSmokeMetrics(smoke) {
  const live = smoke.cift_live_smoke || {};
  const benignAction = live.benign_cift_action || live.benign_final_action;
  const exfilAction = live.exfiltration_cift_action || live.exfiltration_final_action;
  if (!benignAction && !exfilAction) return "not available";
  return `live smoke benign=${benignAction || "unknown"} exfiltration=${exfilAction || "unknown"} provider=${live.exfiltration_provider_status || "unknown"} device=${live.hidden_state_device_observed || "unknown"}`;
}

function fnFpSummary(metrics) {
  if (!metrics) return "not available";
  const fnCount = metrics.false_negative_count;
  const fnRate = metrics.false_negative_rate;
  const fpCount = metrics.false_positive_count;
  const fpRate = metrics.false_positive_rate;
  if (fnCount === undefined && fpCount === undefined) return "not available";
  return `FN ${fnCount ?? "unknown"} (${formatRate(fnRate)}), FP ${fpCount ?? "unknown"} (${formatRate(fpRate)})`;
}

function formatRate(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "unknown";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function auditRows(overview) {
  const gateway = overview.gateway || {};
  const capabilities = gateway.capabilities?.payload || {};
  const audit = capabilities.audit || {};
  const smoke = overview.last_smoke || {};
  return [
    ["Durable audit", audit.durable_jsonl_enabled ? audit.durable_jsonl_path || "enabled" : "memory only"],
    ["Explain route", audit.explain_route],
    ["Last smoke", smoke.status],
    ["Smoke path", smoke.path],
  ];
}

function lastRequestView(event) {
  if (!event) return el("p", { class: "muted" }, "No recent request.");
  return keyValueTable([
    ["Trace", event.trace_id],
    ["Session", event.session_id],
    ["Action", event.final_action],
    ["Provider", event.provider_status],
    ["Reason", event.reason],
  ]);
}

function eventsTable(events) {
  const rows = [];
  events.forEach((event) => {
    const action = String(event.final_action || "unknown");
    rows.push(
      el(
        "tr",
        {},
        el("td", {}, event.created_at || ""),
        el("td", { class: "mono" }, shortId(event.session_id), " / ", shortId(event.trace_id)),
        el("td", {}, el("span", { class: `badge ${action}` }, action)),
        el("td", {}, event.provider_status || "unknown"),
        el("td", {}, (event.detectors_fired || []).join(", ") || "none"),
        el("td", {}, event.reason || ""),
        el("td", {}, traceButton(event.trace_id)),
      ),
    );
    if (state.expandedTraceId === event.trace_id) {
      rows.push(
        el(
          "tr",
          {},
          el(
            "td",
            { colspan: "7" },
            el(
              "div",
              { class: "row-detail" },
              timeline(event.stage_timeline || []),
              detectorEvidenceTable(event.detector_results || []),
            ),
          ),
        ),
      );
    }
  });
  return el(
    "table",
    { class: "event-table" },
    el(
      "thead",
      {},
      el("tr", {}, ...["Timestamp", "Session / Trace", "Action", "Provider", "Detectors", "Reason", ""].map((h) => el("th", {}, h))),
    ),
    el("tbody", {}, ...rows),
  );
}

function traceButton(traceId) {
  const button = el("button", { class: "trace-button", type: "button" }, state.expandedTraceId === traceId ? "Hide" : "Stages");
  button.onclick = () => {
    state.expandedTraceId = state.expandedTraceId === traceId ? null : traceId;
    renderRequests();
  };
  return button;
}

function timeline(stages) {
  const nodes = stages.map((stage) => {
    const status = String(stage.status || "unknown");
    const detail = stage.detail || stage.reason || stage.original_status || "";
    return el(
      "div",
      { class: `stage ${status}` },
      el("div", { class: "stage-name" }, String(stage.stage || "")),
      el("div", { class: "stage-status" }, status),
      detail ? el("div", { class: "stage-detail" }, String(detail)) : null,
    );
  });
  return el("div", { class: "timeline" }, ...nodes);
}

function detectorEvidenceTable(detectors) {
  if (detectors.length === 0) return el("p", { class: "muted detail-heading" }, "No detector evidence.");
  const rows = detectors.map((detector) =>
    el(
      "tr",
      {},
      el("td", {}, detector.detector_name || "unknown"),
      el("td", {}, detector.recommended_action || "unknown"),
      el("td", { class: "mono" }, detector.score === null || detector.score === undefined ? "not available" : String(detector.score)),
      el("td", { class: "mono" }, detector.confidence === null || detector.confidence === undefined ? "not available" : String(detector.confidence)),
      el("td", {}, detector.capability_status || "unknown"),
    ),
  );
  return el(
    "div",
    { class: "detector-summary" },
    el("h3", {}, "Detector Evidence"),
    el(
      "table",
      { class: "event-table" },
      el("thead", {}, el("tr", {}, ...["Detector", "Action", "Score", "Confidence", "Capability"].map((h) => el("th", {}, h)))),
      el("tbody", {}, ...rows),
    ),
  );
}

function nimbusBetaView(nimbus) {
  return keyValueTable([
    ["Runtime label", nimbus.label],
    ["Status", nimbus.status],
    ["Critic kind", nimbus.critic_kind],
    ["Critic version", nimbus.critic_version],
    ["Promotion status", nimbus.promotion_status],
    ["Paper-faithful learned critic", nimbus.paper_faithful_learned_critic],
    ["InfoNCE model", nimbus.infonce_model_path],
    ["Thresholds", JSON.stringify(nimbus.thresholds || {})],
  ]);
}

function degradedStates(states) {
  const rows = states.map((item) => el("tr", {}, el("td", { class: "mono" }, item.state || ""), el("td", {}, item.fix || "")));
  return el("table", { class: "kv" }, el("tbody", {}, ...rows));
}

function setupArchitecture(items) {
  const rows = items.map((item) =>
    el(
      "tr",
      {},
      el("td", { class: "mono" }, item.component || ""),
      el("td", {}, item.connection || ""),
      el("td", { class: "mono" }, item.endpoint || ""),
    ),
  );
  return el(
    "table",
    { class: "event-table" },
    el("thead", {}, el("tr", {}, ...["Component", "Connection", "Endpoint"].map((heading) => el("th", {}, heading)))),
    el("tbody", {}, ...rows),
  );
}

function providerExamples(items) {
  const rows = items.map((item) =>
    el(
      "tr",
      {},
      el("td", {}, item.name || ""),
      el("td", { class: "mono" }, item.provider_base_url || ""),
      el("td", { class: "mono" }, item.agent_base_url || ""),
      el("td", { class: "mono" }, item.provider_env || ""),
    ),
  );
  return el(
    "table",
    { class: "event-table" },
    el("thead", {}, el("tr", {}, ...["Provider", "Provider URL", "Agent URL", "Gateway env"].map((heading) => el("th", {}, heading)))),
    el("tbody", {}, ...rows),
  );
}

function setupList(items) {
  return el("ul", { class: "setup-list" }, ...items.map((item) => el("li", {}, String(item))));
}

function commandLabel(name) {
  return name.replaceAll("_", " ");
}

function shortId(value) {
  const text = String(value || "");
  if (text.length <= 18) return text;
  return `${text.slice(0, 8)}...${text.slice(-6)}`;
}

document.querySelectorAll(".tabs button").forEach((button) => {
  button.onclick = () => {
    state.tab = button.dataset.tab || "overview";
    render();
  };
});

document.getElementById("refresh").onclick = refresh;
document.getElementById("view").replaceChildren(el("div", { class: "panel span-12" }, el("p", { class: "muted" }, "Loading console state...")));
refresh();
