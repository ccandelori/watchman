"use strict";

const { spawn } = require("node:child_process");
const fs = require("node:fs");
const http = require("node:http");
const net = require("node:net");
const path = require("node:path");

const DEFAULT_HOST = "127.0.0.1";
const DEFAULT_PORT = 8790;
const DEFAULT_STARTUP_TIMEOUT_MS = 45000;
const DEFAULT_REQUEST_TIMEOUT_MS = 8000;
const PROBE_PORT_COUNT = 20;

class BackendHttpError extends Error {
  constructor(message, statusCode, payload) {
    super(message);
    this.name = "BackendHttpError";
    this.statusCode = statusCode;
    this.payload = payload;
  }
}

class BackendController {
  constructor(options) {
    this.repoRoot = options.repoRoot;
    this.host = options.host;
    this.preferredPort = options.preferredPort;
    this.env = options.env;
    this.process = null;
    this.mode = "stopped";
    this.baseUrl = null;
    this.logPath = options.logPath;
    this.lastError = null;
  }

  async start() {
    const target = await resolveBackendTarget({
      host: this.host,
      preferredPort: this.preferredPort,
      repoRoot: this.repoRoot,
    });
    this.baseUrl = launcherBaseUrl(this.host, target.port);
    this.lastError = null;
    if (target.mode === "connected") {
      this.mode = "connected";
      return this.status();
    }
    this.process = spawnLauncherBackend({
      repoRoot: this.repoRoot,
      host: this.host,
      port: target.port,
      env: this.env,
      logPath: this.logPath,
    });
    this.mode = "spawned";
    try {
      await waitForLauncherReady({
        baseUrl: this.baseUrl,
        process: this.process,
        timeoutMs: DEFAULT_STARTUP_TIMEOUT_MS,
        repoRoot: this.repoRoot,
      });
    } catch (error) {
      this.lastError = error instanceof Error ? error.message : String(error);
      throw error;
    }
    return this.status();
  }

  async request(method, apiPath, body) {
    if (this.baseUrl === null) {
      throw new Error("Aegis launcher backend has not been started.");
    }
    return requestJson({
      baseUrl: this.baseUrl,
      method,
      apiPath,
      body,
      timeoutMs: DEFAULT_REQUEST_TIMEOUT_MS,
    });
  }

  async stop() {
    if (this.process !== null && this.process.exitCode === null && this.process.signalCode === null) {
      this.process.kill("SIGTERM");
    }
    this.process = null;
    this.mode = "stopped";
    return this.status();
  }

  status() {
    return {
      mode: this.mode,
      baseUrl: this.baseUrl,
      repoRoot: this.repoRoot,
      logPath: this.logPath,
      logExcerpt: readTextTail(this.logPath, 16000),
      processPid: this.process === null ? null : this.process.pid,
      processExitCode: this.process === null ? null : this.process.exitCode,
      lastError: this.lastError,
    };
  }
}

function repoRootFromAppPath(appPath, env) {
  const explicitRoot = env.AEGIS_DESKTOP_REPO_ROOT;
  if (typeof explicitRoot === "string" && explicitRoot.trim() !== "" && isAegisRepoRoot(explicitRoot)) {
    return path.resolve(explicitRoot);
  }
  const candidates = repoRootCandidates(appPath, env);
  for (const candidate of candidates) {
    if (isAegisRepoRoot(candidate)) {
      return path.resolve(candidate);
    }
  }
  return null;
}

function repoRootCandidates(appPath, env) {
  const candidates = [
    env.AEGIS_DESKTOP_REPO_ROOT,
    process.cwd(),
    path.resolve(appPath, "../.."),
    path.resolve(appPath, "../../.."),
    path.resolve(appPath, "../../../.."),
    path.resolve(appPath, "../../../../.."),
    path.resolve(appPath, "../../../../../.."),
    path.resolve(appPath, "../../../../../../.."),
    path.resolve(path.dirname(process.execPath), "../../../../../../.."),
  ];
  return candidates.filter((candidate) => typeof candidate === "string" && candidate.trim() !== "");
}

function isAegisRepoRoot(candidate) {
  const root = path.resolve(candidate);
  return (
    fs.existsSync(path.join(root, "pyproject.toml")) &&
    fs.existsSync(path.join(root, "src", "aegis", "launcher", "app.py")) &&
    fs.existsSync(path.join(root, "introspection"))
  );
}

function readDesktopConfig(configPath) {
  if (!fs.existsSync(configPath)) {
    return {};
  }
  const payload = JSON.parse(fs.readFileSync(configPath, "utf8"));
  if (payload === null || typeof payload !== "object" || Array.isArray(payload)) {
    return {};
  }
  return payload;
}

function repoRootFromConfig(configPath) {
  const payload = readDesktopConfig(configPath);
  if (typeof payload.repoRoot === "string" && isAegisRepoRoot(payload.repoRoot)) {
    return path.resolve(payload.repoRoot);
  }
  return null;
}

function writeDesktopConfig(configPath, payload) {
  fs.mkdirSync(path.dirname(configPath), { recursive: true });
  fs.writeFileSync(configPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
}

function saveRepoRootConfig(configPath, repoRoot) {
  const root = path.resolve(repoRoot);
  if (!isAegisRepoRoot(root)) {
    throw new Error(`Selected folder is not an Aegis repository: ${root}`);
  }
  writeDesktopConfig(configPath, { ...readDesktopConfig(configPath), repoRoot: root });
  return root;
}

function preferredPortFromEnv(env) {
  const rawPort = env.AEGIS_DESKTOP_BACKEND_PORT;
  if (typeof rawPort !== "string" || rawPort.trim() === "") {
    return DEFAULT_PORT;
  }
  const port = Number.parseInt(rawPort, 10);
  if (!Number.isInteger(port) || port < 1024 || port > 65535) {
    throw new Error(`AEGIS_DESKTOP_BACKEND_PORT must be a TCP port from 1024 to 65535. Received '${rawPort}'.`);
  }
  return port;
}

async function resolveBackendTarget(options) {
  for (let offset = 0; offset < PROBE_PORT_COUNT; offset += 1) {
    const port = options.preferredPort + offset;
    const baseUrl = launcherBaseUrl(options.host, port);
    const probe = await probeLauncher({ baseUrl, repoRoot: options.repoRoot });
    if (probe.status === "launcher") {
      return { mode: "connected", port };
    }
    if (probe.status === "occupied") {
      continue;
    }
    const occupied = await portIsOccupied({ host: options.host, port });
    if (!occupied) {
      return { mode: "spawn", port };
    }
  }
  throw new Error(`No available Aegis launcher backend port found from ${options.preferredPort}.`);
}

async function probeLauncher(options) {
  try {
    const payload = await requestJson({
      baseUrl: options.baseUrl,
      method: "GET",
      apiPath: "/api/state",
      body: null,
      timeoutMs: 1000,
    });
    if (payload.schema_version !== "aegis.launcher_state/v1") {
      return { status: "occupied" };
    }
    if (path.resolve(String(payload.repo_root || "")) !== path.resolve(options.repoRoot)) {
      return { status: "occupied" };
    }
    return { status: "launcher" };
  } catch (_error) {
    return { status: "unknown" };
  }
}

function spawnLauncherBackend(options) {
  fs.mkdirSync(path.dirname(options.logPath), { recursive: true });
  const logStream = fs.createWriteStream(options.logPath, { flags: "a" });
  const child = spawn(
    "uv",
    [
      "run",
      "aegis-launcher",
      "--host",
      options.host,
      "--port",
      String(options.port),
      "--repo-root",
      options.repoRoot,
    ],
    {
      cwd: options.repoRoot,
      env: options.env,
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
  child.stdout.pipe(logStream);
  child.stderr.pipe(logStream);
  child.on("close", () => {
    logStream.end();
  });
  child.on("error", (error) => {
    logStream.write(`${error.message}\n`);
  });
  return child;
}

async function waitForLauncherReady(options) {
  const deadline = Date.now() + options.timeoutMs;
  let lastError = null;
  while (Date.now() <= deadline) {
    if (options.process.exitCode !== null) {
      throw new Error(`Aegis launcher backend exited with code ${options.process.exitCode}.`);
    }
    try {
      const probe = await probeLauncher({ baseUrl: options.baseUrl, repoRoot: options.repoRoot });
      if (probe.status === "launcher") {
        return;
      }
    } catch (error) {
      lastError = error;
    }
    await delay(500);
  }
  const suffix = lastError instanceof Error ? ` Last error: ${lastError.message}` : "";
  throw new Error(`Aegis launcher backend did not become ready within ${options.timeoutMs}ms.${suffix}`);
}

async function requestJson(options) {
  const url = new URL(options.apiPath, options.baseUrl);
  const payload = options.body === null || options.body === undefined ? null : JSON.stringify(options.body);
  const response = await new Promise((resolve, reject) => {
    const request = http.request(
      url,
      {
        method: options.method,
        timeout: options.timeoutMs,
        headers:
          payload === null
            ? { Accept: "application/json" }
            : {
                Accept: "application/json",
                "Content-Type": "application/json",
                "Content-Length": Buffer.byteLength(payload),
              },
      },
      (incoming) => {
        const chunks = [];
        incoming.on("data", (chunk) => chunks.push(chunk));
        incoming.on("end", () => {
          const rawBody = Buffer.concat(chunks).toString("utf8");
          resolve({ statusCode: incoming.statusCode || 0, rawBody });
        });
      },
    );
    request.on("timeout", () => {
      request.destroy(new Error(`Request to ${url.toString()} timed out after ${options.timeoutMs}ms.`));
    });
    request.on("error", reject);
    if (payload !== null) {
      request.write(payload);
    }
    request.end();
  });
  const decoded = response.rawBody === "" ? {} : JSON.parse(response.rawBody);
  if (response.statusCode < 200 || response.statusCode >= 300) {
    throw new BackendHttpError(`Aegis launcher API returned HTTP ${response.statusCode}.`, response.statusCode, decoded);
  }
  return decoded;
}

function launcherBaseUrl(host, port) {
  return `http://${host}:${port}`;
}

function portIsOccupied(options) {
  return new Promise((resolve) => {
    const socket = net.createConnection({ host: options.host, port: options.port });
    socket.setTimeout(500);
    socket.on("connect", () => {
      socket.destroy();
      resolve(true);
    });
    socket.on("timeout", () => {
      socket.destroy();
      resolve(false);
    });
    socket.on("error", () => {
      resolve(false);
    });
  });
}

function readTextTail(filePath, maxBytes) {
  if (!fs.existsSync(filePath)) {
    return "";
  }
  const stat = fs.statSync(filePath);
  const start = Math.max(0, stat.size - maxBytes);
  const length = stat.size - start;
  const fd = fs.openSync(filePath, "r");
  try {
    const buffer = Buffer.alloc(length);
    fs.readSync(fd, buffer, 0, length, start);
    return buffer.toString("utf8");
  } finally {
    fs.closeSync(fd);
  }
}

function delay(milliseconds) {
  return new Promise((resolve) => {
    setTimeout(resolve, milliseconds);
  });
}

module.exports = {
  BackendController,
  BackendHttpError,
  DEFAULT_HOST,
  DEFAULT_PORT,
  isAegisRepoRoot,
  launcherBaseUrl,
  preferredPortFromEnv,
  probeLauncher,
  repoRootFromConfig,
  repoRootFromAppPath,
  requestJson,
  resolveBackendTarget,
  saveRepoRootConfig,
};
