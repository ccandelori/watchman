"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const {
  isAllowedRendererDocumentUrl,
  isRendererAssetUrl,
  rendererFileNameFromUrl,
} = require("../src/navigation");

test("rendererFileNameFromUrl extracts files under the renderer directory", () => {
  const fileName = rendererFileNameFromUrl("file:///Applications/Aegis%20Watchman.app/Contents/Resources/app.asar/renderer/index.css");

  assert.equal(fileName, "index.css");
});

test("isAllowedRendererDocumentUrl allows launcher documents", () => {
  assert.equal(isAllowedRendererDocumentUrl("file:///repo/desktop/aegis-launcher/renderer/index.html"), true);
  assert.equal(isAllowedRendererDocumentUrl("file:///repo/desktop/aegis-launcher/renderer/backend-error.html"), true);
});

test("isAllowedRendererDocumentUrl rejects renderer assets", () => {
  assert.equal(isAllowedRendererDocumentUrl("file:///repo/desktop/aegis-launcher/renderer/index.css"), false);
  assert.equal(isAllowedRendererDocumentUrl("file:///repo/desktop/aegis-launcher/renderer/index.js"), false);
});

test("isRendererAssetUrl identifies renderer assets that should not be top-level pages", () => {
  assert.equal(isRendererAssetUrl("file:///repo/desktop/aegis-launcher/renderer/index.css"), true);
  assert.equal(isRendererAssetUrl("file:///repo/desktop/aegis-launcher/renderer/index.js"), true);
  assert.equal(isRendererAssetUrl("file:///repo/desktop/aegis-launcher/renderer/backend-error.css"), true);
});

test("isRendererAssetUrl ignores external urls and allowed html files", () => {
  assert.equal(isRendererAssetUrl("http://127.0.0.1:8790/static/styles.css"), false);
  assert.equal(isRendererAssetUrl("file:///repo/desktop/aegis-launcher/renderer/index.html"), false);
});
