"use strict";

const facts = document.getElementById("facts");
const logOutput = document.getElementById("log-output");
const selectRepoButton = document.getElementById("select-repo");
const retryButton = document.getElementById("retry");
const openBrowserButton = document.getElementById("open-browser");

function fact(label, value) {
  const node = document.createElement("div");
  node.className = "fact";
  const labelNode = document.createElement("span");
  labelNode.textContent = label;
  const valueNode = document.createElement("strong");
  valueNode.textContent = value || "unknown";
  node.append(labelNode, valueNode);
  return node;
}

async function renderStatus() {
  const status = await window.aegisDesktop.backendStatus();
  facts.replaceChildren(
    fact("Mode", status?.mode),
    fact("Backend URL", status?.baseUrl),
    fact("Repo", status?.repoRoot),
    fact("Log", status?.logPath),
  );
  logOutput.textContent = status?.logExcerpt || status?.lastError || "No backend output has been captured.";
}

selectRepoButton.addEventListener("click", async () => {
  selectRepoButton.disabled = true;
  try {
    await window.aegisDesktop.selectRepoRoot();
  } finally {
    selectRepoButton.disabled = false;
    await renderStatus();
  }
});

retryButton.addEventListener("click", async () => {
  retryButton.disabled = true;
  try {
    await window.aegisDesktop.retryBackend();
  } finally {
    retryButton.disabled = false;
    await renderStatus();
  }
});

openBrowserButton.addEventListener("click", async () => {
  await window.aegisDesktop.openBackendInBrowser();
});

renderStatus().catch((error) => {
  logOutput.textContent = error instanceof Error ? error.message : String(error);
});
