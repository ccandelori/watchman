"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const {
  modelNamesFromTagsPayload,
  ollamaAppRunningFromStdout,
  requiredModelName,
} = require("../src/ollama-control");

test("modelNamesFromTagsPayload deduplicates and sorts Ollama tags", () => {
  const names = modelNamesFromTagsPayload({
    models: [
      { name: "qwen3:4b" },
      { name: "llama3.2:latest" },
      { name: "qwen3:4b" },
      { digest: "missing-name" },
      { name: "" },
    ],
  });

  assert.deepEqual(names, ["llama3.2:latest", "qwen3:4b"]);
});

test("requiredModelName trims and rejects empty model names", () => {
  assert.equal(requiredModelName(" qwen3:4b "), "qwen3:4b");
  assert.throws(() => requiredModelName(" "), /Model name is required/);
});

test("ollamaAppRunningFromStdout parses macOS application state", () => {
  assert.equal(ollamaAppRunningFromStdout("true\n"), true);
  assert.equal(ollamaAppRunningFromStdout("false\n"), false);
  assert.equal(ollamaAppRunningFromStdout("TRUE"), true);
});
