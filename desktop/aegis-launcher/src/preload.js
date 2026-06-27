"use strict";

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("aegisDesktop", {
  backendStatus: () => ipcRenderer.invoke("aegis:backend-status"),
  retryBackend: () => ipcRenderer.invoke("aegis:retry-backend"),
  openBackendInBrowser: () => ipcRenderer.invoke("aegis:open-backend-in-browser"),
  selectRepoRoot: () => ipcRenderer.invoke("aegis:select-repo-root"),
  launcherApi: (request) => ipcRenderer.invoke("aegis:launcher-api", request),
  ollamaStatus: () => ipcRenderer.invoke("aegis:ollama-status"),
  startOllama: () => ipcRenderer.invoke("aegis:ollama-start"),
  stopOllama: () => ipcRenderer.invoke("aegis:ollama-stop"),
  pullOllamaModel: (modelName) => ipcRenderer.invoke("aegis:ollama-pull", modelName),
});
