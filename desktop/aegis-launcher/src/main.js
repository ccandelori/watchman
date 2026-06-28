"use strict";

const { app, BrowserWindow, Menu, dialog, ipcMain, shell } = require("electron");
const path = require("node:path");

const {
  BackendController,
  DEFAULT_HOST,
  preferredPortFromEnv,
  repoRootFromConfig,
  repoRootFromAppPath,
  saveRepoRootConfig,
} = require("./backend-process");
const {
  isAllowedRendererDocumentUrl,
  isRendererAssetUrl,
} = require("./navigation");
const { OllamaController } = require("./ollama-control");

let mainWindow = null;
let backend = null;
let ollama = null;
let quitInProgress = false;
let configPath = null;

function backendLogPath(repoRoot) {
  return path.join(repoRoot, ".aegis", "electron-launcher", "backend.log");
}

function desktopConfigPath() {
  return path.join(app.getPath("userData"), "desktop-config.json");
}

function resolveRepoRoot() {
  if (configPath !== null) {
    const configuredRoot = repoRootFromConfig(configPath);
    if (configuredRoot !== null) {
      return configuredRoot;
    }
  }
  return repoRootFromAppPath(app.getAppPath(), process.env);
}

function createControllers(repoRoot) {
  const logDir = path.join(repoRoot, ".aegis", "electron-launcher");
  backend = new BackendController({
    repoRoot,
    host: DEFAULT_HOST,
    preferredPort: preferredPortFromEnv(process.env),
    env: process.env,
    logPath: backendLogPath(repoRoot),
  });
  ollama = new OllamaController({
    repoRoot,
    env: process.env,
    logDir,
  });
}

function rendererFilePath(fileName) {
  return path.join(__dirname, "../renderer", fileName);
}

async function createWindow() {
  configPath = desktopConfigPath();
  const repoRoot = resolveRepoRoot();

  mainWindow = new BrowserWindow({
    width: 1260,
    height: 860,
    minWidth: 980,
    minHeight: 700,
    title: "Aegis Watchman",
    backgroundColor: "#eef2f8",
    titleBarStyle: "hiddenInset",
    ...(process.platform === "darwin" ? { trafficLightPosition: { x: 16, y: 16 } } : {}),
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, "preload.js"),
      sandbox: false,
    },
  });

  installNavigationGuards(mainWindow);

  if (repoRoot === null) {
    await mainWindow.loadFile(rendererFilePath("backend-error.html"));
    return;
  }
  createControllers(repoRoot);
  await loadLauncherWindow();
}

async function loadLauncherWindow() {
  if (mainWindow === null || backend === null) {
    return;
  }
  try {
    await backend.start();
    await mainWindow.loadFile(rendererFilePath("index.html"));
  } catch (_error) {
    await mainWindow.loadFile(rendererFilePath("backend-error.html"));
  }
}

async function recoverLauncherWindow() {
  if (mainWindow === null) {
    return;
  }
  if (backend === null) {
    await mainWindow.loadFile(rendererFilePath("backend-error.html"));
    return;
  }
  await loadLauncherWindow();
}

function recoverLauncherWindowFromNavigationError(source, rawUrl) {
  recoverLauncherWindow().catch((error) => {
    console.warn("Aegis launcher failed to recover from renderer navigation.", { source, rawUrl, error });
  });
}

function installNavigationGuards(window) {
  window.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url).catch((error) => {
      console.warn("Aegis launcher failed to open external URL.", { url, error });
    });
    return { action: "deny" };
  });

  window.webContents.on("will-navigate", (event, rawUrl) => {
    if (isAllowedRendererDocumentUrl(rawUrl)) {
      return;
    }
    event.preventDefault();
    if (isRendererAssetUrl(rawUrl)) {
      recoverLauncherWindowFromNavigationError("will-navigate", rawUrl);
      return;
    }
    shell.openExternal(rawUrl).catch((error) => {
      console.warn("Aegis launcher failed to open top-level navigation externally.", { rawUrl, error });
    });
  });

  window.webContents.on("did-finish-load", () => {
    const rawUrl = window.webContents.getURL();
    if (isRendererAssetUrl(rawUrl)) {
      recoverLauncherWindowFromNavigationError("did-finish-load", rawUrl);
    }
  });
}

function installIpcHandlers() {
  ipcMain.handle("aegis:backend-status", () => {
    return backend === null ? null : backend.status();
  });
  ipcMain.handle("aegis:retry-backend", async () => {
    await loadLauncherWindow();
    return backend === null ? null : backend.status();
  });
  ipcMain.handle("aegis:open-backend-in-browser", async () => {
    if (backend !== null && backend.baseUrl !== null) {
      await shell.openExternal(backend.baseUrl);
    }
    return backend === null ? null : backend.status();
  });
  ipcMain.handle("aegis:select-repo-root", async () => {
    if (mainWindow === null || configPath === null) {
      throw new Error("Aegis desktop window is not initialized.");
    }
    const result = await dialog.showOpenDialog(mainWindow, {
      title: "Select Aegis repository",
      properties: ["openDirectory"],
    });
    if (result.canceled || result.filePaths.length === 0) {
      return { selected: false };
    }
    const repoRoot = saveRepoRootConfig(configPath, result.filePaths[0]);
    createControllers(repoRoot);
    await loadLauncherWindow();
    return { selected: true, repoRoot };
  });
  ipcMain.handle("aegis:launcher-api", async (_event, request) => {
    if (backend === null) {
      throw new Error("Aegis launcher backend is not initialized.");
    }
    return backend.request(request.method, request.path, request.body);
  });
  ipcMain.handle("aegis:ollama-status", async () => {
    if (ollama === null) {
      throw new Error("Ollama controller is not initialized.");
    }
    return ollama.status();
  });
  ipcMain.handle("aegis:ollama-start", async () => {
    if (ollama === null) {
      throw new Error("Ollama controller is not initialized.");
    }
    return ollama.start();
  });
  ipcMain.handle("aegis:ollama-stop", async () => {
    if (ollama === null) {
      throw new Error("Ollama controller is not initialized.");
    }
    return ollama.stop();
  });
  ipcMain.handle("aegis:ollama-pull", async (_event, modelName) => {
    if (ollama === null) {
      throw new Error("Ollama controller is not initialized.");
    }
    return ollama.pullModel(modelName);
  });
}

function installMenu() {
  const template = [
    {
      label: "Aegis Watchman",
      submenu: [
        { role: "about" },
        { type: "separator" },
        { role: "quit" },
      ],
    },
    {
      label: "View",
      submenu: [
        {
          label: "Reload Launcher",
          accelerator: "CommandOrControl+R",
          click: async () => {
            await recoverLauncherWindow();
          },
        },
        { role: "toggleDevTools" },
      ],
    },
    {
      label: "Launcher",
      submenu: [
        {
          label: "Retry Backend",
          click: async () => {
            await loadLauncherWindow();
          },
        },
        {
          label: "Open Web Fallback",
          click: async () => {
            if (backend !== null && backend.baseUrl !== null) {
              await shell.openExternal(backend.baseUrl);
            }
          },
        },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

app.whenReady().then(async () => {
  installIpcHandlers();
  installMenu();
  await createWindow();
});

app.on("activate", async () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    await createWindow();
  }
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", (event) => {
  if (quitInProgress) {
    return;
  }
  event.preventDefault();
  shutdownAndQuit(null);
});

process.on("SIGINT", () => {
  shutdownAndQuit(130);
});

process.on("SIGTERM", () => {
  shutdownAndQuit(143);
});

function shutdownAndQuit(exitCode) {
  if (quitInProgress) {
    return;
  }
  quitInProgress = true;
  shutdownBackend()
    .catch((error) => {
      console.warn("Aegis launcher shutdown reported an error.", error);
    })
    .finally(() => {
      if (exitCode === null) {
        app.quit();
        return;
      }
      app.exit(exitCode);
    });
}

async function shutdownBackend() {
  if (backend === null) {
    return;
  }
  const status = backend.status();
  try {
    if (status.mode === "spawned" && status.baseUrl !== null) {
      await backend.request("POST", "/api/actions/stop-all", null);
    }
  } finally {
    await backend.stop();
  }
  if (ollama !== null) {
    await ollama.stopOwnedProcesses();
  }
}
