const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const read = (file) => fs.readFileSync(path.join(__dirname, "..", "..", file), "utf8");

test("MLP example joins Python lines with actual newline characters", () => {
  const source = read("docs/javascripts/playground.js");
  assert.ok(source.includes('.join("\\n")'));
  assert.ok(!source.includes('.join("\\\\n")'));
});

test("playground indentation handles cursors and multi-line selections", () => {
  const source = read("docs/javascripts/playground.js");
  const start = source.indexOf("  function indentSelection(outdent) {");
  const end = source.indexOf("\n  var examples =", start + 1);
  assert.ok(start >= 0 && end > start, "indentSelection implementation must exist");
  const implementation = source.slice(start, end);
  const makeEditor = new Function("editor", implementation + "; return indentSelection;");
  const run = (value, selectionStart, selectionEnd, outdent) => {
    const editor = { value, selectionStart, selectionEnd };
    makeEditor(editor)(outdent);
    return editor;
  };

  assert.deepEqual(run("a\nb", 2, 2, false), {
    value: "a\n    b",
    selectionStart: 6,
    selectionEnd: 6,
  });
  assert.deepEqual(run("a\nb\nc", 0, 3, false), {
    value: "    a\n    b\nc",
    selectionStart: 4,
    selectionEnd: 11,
  });
  assert.deepEqual(run("    a\n    b\nc", 0, 11, true), {
    value: "a\nb\nc",
    selectionStart: 0,
    selectionEnd: 3,
  });
  assert.deepEqual(run("a\n    b", 6, 6, true), {
    value: "a\nb",
    selectionStart: 2,
    selectionEnd: 2,
  });
  assert.deepEqual(run("a\nb\nc", 0, 2, false), {
    value: "    a\nb\nc",
    selectionStart: 4,
    selectionEnd: 6,
  });
});

test("playground package URL is resolved from the active page", () => {
  const source = read("docs/javascripts/playground.js");
  assert.ok(source.includes("new URL("));
  assert.ok(source.includes("window.location.href"));
  assert.ok(source.includes("os.path.commonpath"));
});

test("code-block runner resolves and validates the package archive", () => {
  const source = read("docs/javascripts/run-snippet.js");
  assert.ok(source.includes('new URL("assets/sneppx_py.zip", siteRoot)'));
  assert.ok(source.includes("Package download failed"));
  assert.ok(source.includes("os.path.commonpath"));
});

test("keyboard controls target the actual MkDocs Material inputs", () => {
  const source = read("docs/javascripts/shortcuts.js");
  assert.ok(source.includes('label[for="__toc"]'));
  assert.ok(source.includes('label[for="__drawer"]'));
  assert.ok(source.includes("reduceMotion"));
});

test("command palette exposes accessible page and site-control actions", () => {
  const source = read("docs/javascripts/command-palette.js");
  for (const action of [
    "Toggle table of contents",
    "Toggle primary navigation",
    "Open keyboard shortcuts",
    "Copy current page URL",
    "Print current page",
    "Open discussions",
  ]) {
    assert.ok(source.includes(action), "missing command: " + action);
  }
  assert.ok(source.includes('role", "dialog"'));
  assert.ok(source.includes("returnFocus.focus()"));
});
