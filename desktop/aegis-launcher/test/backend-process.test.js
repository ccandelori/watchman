"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const http = require("node:http");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");

const {
  isAegisRepoRoot,
  launcherBaseUrl,
  preferredPortFromEnv,
  repoRootFromConfig,
  repoRootFromAppPath,
  requestJson,
  saveRepoRootConfig,
} = require("../src/backend-process");

test("repoRootFromAppPath uses explicit repo root when provided", () => {
  const repo = makeRepoFixture();
  const repoRoot = repoRootFromAppPath(path.join(repo, "desktop/aegis-launcher"), {
    AEGIS_DESKTOP_REPO_ROOT: repo,
  });

  assert.equal(repoRoot, repo);
});

test("repoRootFromAppPath defaults from desktop app directory", () => {
  const repo = makeRepoFixture();
  const repoRoot = repoRootFromAppPath(path.join(repo, "desktop/aegis-launcher"), {});

  assert.equal(repoRoot, repo);
});

test("repoRootFromAppPath returns null when no repo is found", () => {
  const empty = fs.mkdtempSync(path.join(os.tmpdir(), "aegis-empty-"));

  assert.equal(repoRootFromAppPath(path.join(empty, "desktop/aegis-launcher"), {}), null);
});

test("repoRootFromConfig saves and reloads a selected repo", () => {
  const repo = makeRepoFixture();
  const configPath = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "aegis-config-")), "config.json");

  assert.equal(saveRepoRootConfig(configPath, repo), repo);
  assert.equal(repoRootFromConfig(configPath), repo);
  assert.equal(isAegisRepoRoot(repo), true);
});

test("preferredPortFromEnv validates the backend port", () => {
  assert.equal(preferredPortFromEnv({}), 8790);
  assert.equal(preferredPortFromEnv({ AEGIS_DESKTOP_BACKEND_PORT: "8801" }), 8801);
  assert.throws(
    () => preferredPortFromEnv({ AEGIS_DESKTOP_BACKEND_PORT: "nope" }),
    /AEGIS_DESKTOP_BACKEND_PORT/,
  );
});

test("requestJson sends JSON and decodes launcher responses", async () => {
  const server = http.createServer((request, response) => {
    const chunks = [];
    request.on("data", (chunk) => chunks.push(chunk));
    request.on("end", () => {
      const body = Buffer.concat(chunks).toString("utf8");
      response.setHeader("Content-Type", "application/json");
      response.end(JSON.stringify({ method: request.method, url: request.url, body: body === "" ? null : JSON.parse(body) }));
    });
  });

  await new Promise((resolve) => {
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  assert.equal(typeof address, "object");
  const baseUrl = launcherBaseUrl("127.0.0.1", address.port);

  try {
    const payload = await requestJson({
      baseUrl,
      method: "PUT",
      apiPath: "/api/profile",
      body: { provider_model: "qwen3:4b" },
      timeoutMs: 1000,
    });

    assert.deepEqual(payload, {
      method: "PUT",
      url: "/api/profile",
      body: { provider_model: "qwen3:4b" },
    });
  } finally {
    await new Promise((resolve) => {
      server.close(resolve);
    });
  }
});

function makeRepoFixture() {
  const repo = fs.mkdtempSync(path.join(os.tmpdir(), "aegis-repo-"));
  fs.mkdirSync(path.join(repo, "src", "aegis", "launcher"), { recursive: true });
  fs.mkdirSync(path.join(repo, "introspection"), { recursive: true });
  fs.writeFileSync(path.join(repo, "pyproject.toml"), "[project]\nname = \"aegis\"\n", "utf8");
  fs.writeFileSync(path.join(repo, "src", "aegis", "launcher", "app.py"), "", "utf8");
  return repo;
}
