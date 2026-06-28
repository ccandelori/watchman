"use strict";

const path = require("node:path");

const ALLOWED_RENDERER_HTML_FILES = new Set(["index.html", "backend-error.html"]);
const RENDERER_ASSET_EXTENSIONS = new Set([".css", ".js", ".map", ".png", ".jpg", ".jpeg", ".svg", ".webp", ".ico"]);

function filePathFromUrl(rawUrl) {
  try {
    const parsedUrl = new URL(rawUrl);
    if (parsedUrl.protocol !== "file:") {
      return null;
    }
    return decodeURIComponent(parsedUrl.pathname);
  } catch (_error) {
    return null;
  }
}

function rendererFileNameFromUrl(rawUrl) {
  const filePath = filePathFromUrl(rawUrl);
  if (filePath === null) {
    return null;
  }
  const normalizedPath = filePath.replace(/\\/g, "/");
  if (!normalizedPath.includes("/renderer/")) {
    return null;
  }
  return path.posix.basename(normalizedPath);
}

function isAllowedRendererDocumentUrl(rawUrl) {
  const fileName = rendererFileNameFromUrl(rawUrl);
  return fileName !== null && ALLOWED_RENDERER_HTML_FILES.has(fileName);
}

function isRendererAssetUrl(rawUrl) {
  const fileName = rendererFileNameFromUrl(rawUrl);
  if (fileName === null) {
    return false;
  }
  if (ALLOWED_RENDERER_HTML_FILES.has(fileName)) {
    return false;
  }
  return RENDERER_ASSET_EXTENSIONS.has(path.posix.extname(fileName).toLowerCase());
}

module.exports = {
  isAllowedRendererDocumentUrl,
  isRendererAssetUrl,
  rendererFileNameFromUrl,
};
