"use strict";

const COMMANDS = [
  { id: "list-formats", label: "Formats" },
  { id: "preview-corpus", label: "Preview corpus" },
  { id: "generate", label: "Generate" },
  { id: "report", label: "Report" },
  { id: "scan", label: "Scan & auto-decoy" },
  { id: "train", label: "Train" },
  { id: "inspect", label: "Inspect model" },
  { id: "validate", label: "Validate" },
];

let FORMATS = [];
let MODELS = [];

async function api(path, body) {
  const opts = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const resp = await fetch(path, opts);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
  return data;
}

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => (k === "class" ? (e.className = v) : e.setAttribute(k, v)));
  kids.forEach((k) => e.append(k));
  return e;
}

function formatOptions() {
  return FORMATS.map((f) => el("option", { value: f.slug }, `${f.slug} — ${f.name}`));
}
function modelOptions() {
  return MODELS.map((m) => el("option", { value: m.name }, `${m.name} (${m.source}${m.slug ? ", " + m.slug : ""})`));
}
function field(labelText, control) {
  const l = el("label", {}, labelText);
  l.append(control);
  return l;
}
function numInput(name, value) {
  return el("input", { name, type: "number", value: String(value) });
}

function sourcePicker() {
  const sel = el("select", { name: "source" }, el("option", { value: "format" }, "train on the fly"), el("option", { value: "model" }, "saved model"));
  return sel;
}

function showError(msg) {
  const e = document.getElementById("error");
  e.textContent = msg;
  e.hidden = false;
}
function clearError() {
  document.getElementById("error").hidden = true;
}
function setOutput(node) {
  const o = document.getElementById("output");
  o.replaceChildren(node);
}

function tokenList(tokens) {
  const wrap = el("div", { class: "tokens" });
  tokens.forEach((t) => wrap.append(el("div", {}, t, el("span", { class: "tag" }, "synthetic"))));
  const copy = el("button", { class: "copy" }, "copy all");
  copy.onclick = () => navigator.clipboard.writeText(tokens.join("\n"));
  const head = el("div", {}, `${tokens.length} synthetic, shape-only tokens`, copy);
  return el("div", {}, head, wrap);
}

function jsonBlock(obj) {
  return el("pre", {}, JSON.stringify(obj, null, 2));
}

function buildForm(cmd) {
  const form = document.getElementById("form");
  form.replaceChildren();
  document.getElementById("cmd-title").textContent = COMMANDS.find((c) => c.id === cmd).label;
  clearError();
  document.getElementById("output").replaceChildren();

  const needsSource = cmd === "generate" || cmd === "report";
  const needsModelOnly = cmd === "inspect" || cmd === "validate";

  if (cmd === "list-formats") {
    // no inputs
  } else if (cmd === "preview-corpus") {
    form.append(field("format", el("select", { name: "format" }, ...formatOptions())));
    form.append(field("count", numInput("count", 5)));
    form.append(field("seed", numInput("seed", 0)));
  } else if (needsSource) {
    const src = sourcePicker();
    form.append(field("source", src));
    const fmt = field("format", el("select", { name: "format" }, ...formatOptions()));
    const mdl = field("model", el("select", { name: "model" }, ...modelOptions()));
    form.append(fmt, mdl);
    const sync = () => { fmt.hidden = src.value !== "format"; mdl.hidden = src.value !== "model"; };
    src.onchange = sync; sync();
    form.append(field("count", numInput("count", cmd === "report" ? 100 : 10)));
    form.append(field("seed", numInput("seed", 1)));
    form.append(field("epsilon", numInput("epsilon", 1)));
    form.append(field("clip", numInput("clip", 1)));
    form.append(field("corpus_size", numInput("corpus_size", 200)));
    form.append(field("train_seed", numInput("train_seed", 0)));
  } else if (cmd === "train") {
    form.append(field("format", el("select", { name: "format" }, ...formatOptions())));
    form.append(field("out_name", el("input", { name: "out_name", value: "my-model" })));
    form.append(field("epsilon", numInput("epsilon", 1)));
    form.append(field("clip", numInput("clip", 1)));
    form.append(field("corpus_size", numInput("corpus_size", 200)));
    form.append(field("seed", numInput("seed", 0)));
    const force = el("input", { name: "force", type: "checkbox" });
    form.append(field("overwrite", force));
  } else if (cmd === "scan") {
    const ta = el("textarea", { name: "text", rows: "8", style: "width:100%; font-family:ui-monospace,monospace;" });
    const wrap = el("label", {}, "paste text to scan");
    wrap.append(ta);
    form.append(wrap);
    form.append(field("seed", numInput("seed", 1)));
  } else if (needsModelOnly) {
    form.append(field("model", el("select", { name: "model" }, ...modelOptions())));
  }

  form.append(el("button", { class: "run", type: "submit" }, "Run"));
  form.onsubmit = (ev) => { ev.preventDefault(); runCommand(cmd, form); };
}

function readForm(form) {
  const data = {};
  new FormData(form).forEach((v, k) => (data[k] = v));
  form.querySelectorAll("input[type=checkbox]").forEach((c) => (data[c.name] = c.checked));
  ["count", "seed", "epsilon", "clip", "corpus_size", "train_seed"].forEach((k) => {
    if (data[k] !== undefined && data[k] !== "") data[k] = Number(data[k]);
  });
  return data;
}

async function runCommand(cmd, form) {
  clearError();
  try {
    const body = readForm(form);
    if (cmd === "list-formats") {
      const formats = await api("/api/formats");
      setOutput(jsonBlock(formats));
    } else if (cmd === "preview-corpus") {
      const { examples } = await api("/api/preview-corpus", body);
      setOutput(tokenList(examples));
    } else if (cmd === "generate") {
      const { tokens } = await api("/api/generate", body);
      setOutput(tokenList(tokens));
    } else if (cmd === "report") {
      setOutput(jsonBlock(await api("/api/report", body)));
    } else if (cmd === "scan") {
      const result = await api("/api/auto-decoy", body);
      const lines = result.findings.map((f) => `${f.format}  [${f.start}-${f.end}]  ${f.confidence}`);
      const out = el("div", {});
      out.append(el("div", {}, `${result.findings.length} finding(s) — synthetic decoys below`));
      out.append(el("pre", {}, lines.join("\n") || "no secrets detected"));
      out.append(el("pre", {}, result.decoys.join("\n")));
      out.append(el("div", {}, "swapped text"));
      out.append(el("pre", {}, result.swapped_text));
      setOutput(out);
    } else if (cmd === "train") {
      const res = await api("/api/train", body);
      await refreshModels();
      setOutput(jsonBlock(res));
    } else if (cmd === "inspect") {
      setOutput(jsonBlock(await api("/api/inspect", body)));
    } else if (cmd === "validate") {
      const res = await api("/api/validate", body);
      const cls = res.valid ? "badge-ok" : "badge-bad";
      setOutput(el("div", { class: cls }, res.valid ? "VALID" : `INVALID: ${res.error}`));
    }
  } catch (err) {
    showError(err.message);
  }
}

async function refreshModels() {
  MODELS = await api("/api/models");
}

function buildNav(active) {
  const nav = document.getElementById("nav");
  nav.replaceChildren();
  COMMANDS.forEach((c) => {
    const b = el("button", c.id === active ? { class: "active" } : {}, c.label);
    b.onclick = () => { buildNav(c.id); buildForm(c.id); };
    nav.append(b);
  });
}

async function init() {
  try {
    FORMATS = await api("/api/formats");
    await refreshModels();
  } catch (err) {
    showError(err.message);
  }
  buildNav("generate");
  buildForm("generate");
}

init();
