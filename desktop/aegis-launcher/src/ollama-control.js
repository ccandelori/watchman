"use strict";

const { spawn } = require("node:child_process");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");

const OLLAMA_BASE_URL = "http://127.0.0.1:11434";
const OLLAMA_TAGS_PATH = "/api/tags";
const OLLAMA_START_TIMEOUT_MS = 30000;
const OLLAMA_STOP_TIMEOUT_MS = 10000;
const OLLAMA_REQUEST_TIMEOUT_MS = 3000;

class OllamaController {
  constructor(options) {
    this.repoRoot = options.repoRoot;
    this.env = options.env;
    this.logDir = options.logDir;
    this.serverProcess = null;
    this.pullProcess = null;
    this.lastError = null;
  }

  async status() {
    const running = await ollamaIsRunning();
    const installedModels = running ? await ollamaModelNames() : [];
    return {
      running,
      baseUrl: OLLAMA_BASE_URL,
      apiTagsUrl: `${OLLAMA_BASE_URL}${OLLAMA_TAGS_PATH}`,
      installedModels,
      server: processSnapshot(this.serverProcess, this.serverLogPath()),
      pull: processSnapshot(this.pullProcess, this.pullLogPath()),
      lastError: this.lastError,
    };
  }

  async start() {
    this.lastError = null;
    if (await ollamaIsRunning()) {
      return this.status();
    }
    const appOpened = await tryOpenOllamaApp();
    if (appOpened) {
      try {
        await waitForOllama(OLLAMA_START_TIMEOUT_MS);
        return this.status();
      } catch (error) {
        this.lastError = error instanceof Error ? error.message : String(error);
      }
    }
    this.serverProcess = spawnLoggedProcess({
      argv: ["ollama", "serve"],
      cwd: this.repoRoot,
      env: this.env,
      logPath: this.serverLogPath(),
    });
    try {
      await waitForOllama(OLLAMA_START_TIMEOUT_MS);
    } catch (error) {
      this.lastError = error instanceof Error ? error.message : String(error);
      throw error;
    }
    return this.status();
  }

  async pullModel(modelName) {
    const model = requiredModelName(modelName);
    this.lastError = null;
    const status = await this.status();
    if (!status.running) {
      throw new Error("Ollama is not running. Start Ollama before pulling a model.");
    }
    if (status.installedModels.includes(model)) {
      return this.status();
    }
    if (this.pullProcess !== null && this.pullProcess.exitCode === null && this.pullProcess.signalCode === null) {
      throw new Error("A model pull is already running.");
    }
    this.pullProcess = spawnLoggedProcess({
      argv: ["ollama", "pull", model],
      cwd: this.repoRoot,
      env: this.env,
      logPath: this.pullLogPath(),
    });
    return this.status();
  }

  async stop() {
    this.lastError = null;
    const wasRunning = await ollamaIsRunning();
    stopProcess(this.pullProcess);
    stopProcess(this.serverProcess);
    this.pullProcess = null;
    this.serverProcess = null;
    if (!wasRunning) {
      return this.status();
    }
    const appQuitRequested = await tryQuitOllamaApp();
    try {
      await waitForOllamaStopped(OLLAMA_STOP_TIMEOUT_MS);
    } catch (_error) {
      this.lastError = appQuitRequested
        ? "Ollama is still responding on 127.0.0.1:11434 after quitting the app. It may be managed by another process."
        : "Ollama is still responding on 127.0.0.1:11434. It may be running from a terminal, service, or another app.";
    }
    return this.status();
  }

  async stopOwnedProcesses() {
    stopProcess(this.pullProcess);
    stopProcess(this.serverProcess);
    this.pullProcess = null;
    this.serverProcess = null;
    return this.status();
  }

  serverLogPath() {
    return path.join(this.logDir, "ollama-server.log");
  }

  pullLogPath() {
    return path.join(this.logDir, "ollama-pull.log");
  }
}

function requiredModelName(modelName) {
  if (typeof modelName !== "string" || modelName.trim() === "") {
    throw new Error("Model name is required.");
  }
  return modelName.trim();
}

async function ollamaIsRunning() {
  try {
    await requestOllamaJson(OLLAMA_TAGS_PATH, OLLAMA_REQUEST_TIMEOUT_MS);
    return true;
  } catch (_error) {
    return false;
  }
}

async function ollamaModelNames() {
  const payload = await requestOllamaJson(OLLAMA_TAGS_PATH, OLLAMA_REQUEST_TIMEOUT_MS);
  return modelNamesFromTagsPayload(payload);
}

function modelNamesFromTagsPayload(payload) {
  if (!Array.isArray(payload.models)) {
    return [];
  }
  const names = [];
  for (const item of payload.models) {
    if (item && typeof item.name === "string" && item.name.trim() !== "") {
      names.push(item.name);
    }
  }
  return Array.from(new Set(names)).sort();
}

function requestOllamaJson(apiPath, timeoutMs) {
  const url = new URL(apiPath, OLLAMA_BASE_URL);
  return new Promise((resolve, reject) => {
    const request = http.request(
      url,
      {
        method: "GET",
        timeout: timeoutMs,
        headers: { Accept: "application/json" },
      },
      (response) => {
        const chunks = [];
        response.on("data", (chunk) => chunks.push(chunk));
        response.on("end", () => {
          const body = Buffer.concat(chunks).toString("utf8");
          if ((response.statusCode || 0) < 200 || (response.statusCode || 0) >= 300) {
            reject(new Error(`Ollama returned HTTP ${response.statusCode}.`));
            return;
          }
          try {
            resolve(body === "" ? {} : JSON.parse(body));
          } catch (error) {
            reject(error);
          }
        });
      },
    );
    request.on("timeout", () => {
      request.destroy(new Error(`Ollama request timed out after ${timeoutMs}ms.`));
    });
    request.on("error", reject);
    request.end();
  });
}

function tryOpenOllamaApp() {
  if (process.platform !== "darwin") {
    return Promise.resolve(false);
  }
  return new Promise((resolve) => {
    const child = spawn("open", ["-a", "Ollama"], { stdio: "ignore" });
    child.on("error", () => resolve(false));
    child.on("close", (code) => resolve(code === 0));
  });
}

async function tryQuitOllamaApp() {
  if (process.platform !== "darwin") {
    return false;
  }
  if (!(await ollamaAppIsRunning())) {
    return false;
  }
  return (await commandExitCode("osascript", ["-e", 'tell application "Ollama" to quit'])) === 0;
}

async function ollamaAppIsRunning() {
  if (process.platform !== "darwin") {
    return false;
  }
  const result = await commandOutput("osascript", ["-e", 'application "Ollama" is running']);
  return ollamaAppRunningFromStdout(result.stdout);
}

function ollamaAppRunningFromStdout(stdout) {
  return stdout.trim().toLowerCase() === "true";
}

async function waitForOllama(timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() <= deadline) {
    if (await ollamaIsRunning()) {
      return;
    }
    await delay(500);
  }
  throw new Error(`Ollama did not respond within ${timeoutMs}ms.`);
}

async function waitForOllamaStopped(timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() <= deadline) {
    if (!(await ollamaIsRunning())) {
      return;
    }
    await delay(500);
  }
  throw new Error(`Ollama still responded after ${timeoutMs}ms.`);
}

function commandExitCode(command, args) {
  return new Promise((resolve) => {
    const child = spawn(command, args, { stdio: "ignore" });
    child.on("error", () => resolve(1));
    child.on("close", (code) => resolve(code === null ? 1 : code));
  });
}

function commandOutput(command, args) {
  return new Promise((resolve) => {
    const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
    const stdout = [];
    const stderr = [];
    child.stdout.on("data", (chunk) => stdout.push(chunk));
    child.stderr.on("data", (chunk) => stderr.push(chunk));
    child.on("error", (error) => resolve({ exitCode: 1, stdout: "", stderr: error.message }));
    child.on("close", (code) => {
      resolve({
        exitCode: code === null ? 1 : code,
        stdout: Buffer.concat(stdout).toString("utf8"),
        stderr: Buffer.concat(stderr).toString("utf8"),
      });
    });
  });
}

function spawnLoggedProcess(options) {
  fs.mkdirSync(path.dirname(options.logPath), { recursive: true });
  const logStream = fs.createWriteStream(options.logPath, { flags: "w" });
  const child = spawn(options.argv[0], options.argv.slice(1), {
    cwd: options.cwd,
    env: options.env,
    stdio: ["ignore", "pipe", "pipe"],
  });
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

function processSnapshot(processValue, logPath) {
  if (processValue === null) {
    return {
      status: "idle",
      pid: null,
      exitCode: null,
      signalCode: null,
      logPath,
      logExcerpt: readTextTail(logPath, 12000),
    };
  }
  return {
    status: processValue.exitCode === null && processValue.signalCode === null ? "running" : "exited",
    pid: processValue.pid,
    exitCode: processValue.exitCode,
    signalCode: processValue.signalCode,
    logPath,
    logExcerpt: readTextTail(logPath, 12000),
  };
}

function stopProcess(processValue) {
  if (processValue !== null && processValue.exitCode === null && processValue.signalCode === null) {
    processValue.kill("SIGTERM");
  }
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
  OLLAMA_BASE_URL,
  OllamaController,
  modelNamesFromTagsPayload,
  ollamaAppRunningFromStdout,
  ollamaModelNames,
  requestOllamaJson,
  requiredModelName,
};
