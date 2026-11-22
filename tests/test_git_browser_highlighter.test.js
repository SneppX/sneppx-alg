const test = require("node:test");
const assert = require("node:assert/strict");
const h = require("../docs/javascripts/git-browser-highlighter.js");
test("maps common source extensions", () => {
  assert.equal(h.language("a.py"), "python");
  assert.equal(h.language("a.cpp"), "cpp");
  assert.equal(h.language("a.tsx"), "typescript");
  assert.equal(h.language("a.toml"), "toml");
  assert.equal(h.language("a.unknown"), "text");
});
test("highlights language keywords, strings, comments and numbers", () => {
  const t = h.tokenizeLine('def forward(x): return "ok" # note', "python");
  assert.ok(t.some((x) => x.type === "keyword" && x.value === "def"));
  assert.ok(t.some((x) => x.type === "string" && x.value === '"ok"'));
  assert.ok(t.some((x) => x.type === "comment"));
  assert.ok(h.tokenizeLine("const count = 42;", "javascript").some((x) => x.type === "number" && x.value === "42"));
});
test("recognizes CSS hex, functional and named colors", () => {
  const t = h.tokenizeLine("color: #12aBef; background: rgb(1, 2, 3); border-color: rebeccapurple;", "css");
  assert.deepEqual(
    t.filter((x) => x.type === "color").map((x) => x.value),
    ["#12aBef", "rgb(1, 2, 3)", "rebeccapurple"],
  );
});
test("does not mistake CSS keywords or custom properties for colors", () => {
  const t = h.tokenizeLine("color: var(--brand); display: block; color: transparent;", "css");
  assert.equal(t.filter((x) => x.type === "color").length, 0);
});
