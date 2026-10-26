/* Minimal DOM stub + smoke test for the visualizer scripts (node). */
"use strict";

function makeEl(tag) {
    const el = {
        tagName: (tag || "div").toUpperCase(),
        children: [],
        _attrs: {},
        _cls: new Set(),
        _text: "",
        style: {},
        dataset: {},
        disabled: false,
        value: "",
        get className() { return Array.from(this._cls).join(" "); },
        set className(v) { this._cls = new Set(String(v).split(/\s+/).filter(Boolean)); },
        classList: null,
        set innerHTML(v) { this._html = v; if (v === "") this.children = []; },
        get innerHTML() { return this._html || ""; },
        set textContent(v) { this._text = String(v); },
        get textContent() { return this._text; },
        setAttribute(k, v) { this._attrs[k] = String(v); },
        getAttribute(k) { return k in this._attrs ? this._attrs[k] : null; },
        appendChild(c) { this.children.push(c); return c; },
        insertBefore(c) { this.children.push(c); return c; },
        addEventListener() {},
        removeEventListener() {},
        removeChild(c) { this.children = this.children.filter((x) => x !== c); },
        querySelector() { return null; },
        querySelectorAll() { return []; },
        closest() { return null; },
        focus() {},
    };
    el.classList = {
        add: (c) => el._cls.add(c),
        remove: (c) => el._cls.delete(c),
        toggle: (c, on) => (on ? el._cls.add(c) : el._cls.delete(c)),
        contains: (c) => el._cls.has(c),
    };
    return el;
}

global.window = global;
global.document = {
    readyState: "complete",
    documentElement: { scrollHeight: 1000, clientHeight: 800 },
    body: makeEl("body"),
    head: makeEl("head"),
    createElement: (t) => makeEl(t),
    createElementNS: (ns, t) => makeEl(t),
    createTextNode: (t) => ({ nodeValue: t }),
    createDocumentFragment: () => makeEl("frag"),
    getElementById: () => null,
    querySelector: () => null,
    querySelectorAll: () => [],
    addEventListener: () => {},
};
global.matchMedia = () => ({ matches: false });
global.location = { hash: "", href: "http://x/", pathname: "/" };
global.scrollY = 0;
global.innerHeight = 800;
global.addEventListener = () => {};
global.localStorage = { getItem: () => null, setItem: () => {} };
global.setTimeout = (fn) => { try { fn(); } catch (e) {} return 0; };
global.requestAnimationFrame = (fn) => fn(0);
try { Object.defineProperty(global, "navigator", { value: { clipboard: null }, configurable: true }); } catch (e) {}

require("../../docs/javascripts/autograd-core.js");
require("../../docs/javascripts/matmul-viz.js");
require("../../docs/javascripts/conv-viz.js");
require("../../docs/javascripts/optimizer-race.js");
require("../../docs/javascripts/loss-gallery.js");
require("../../docs/javascripts/progress.js");
require("../../docs/javascripts/autograd-graph.js");
require("../../docs/javascripts/run-snippet.js");
require("../../docs/javascripts/command-palette.js");
require("../../docs/javascripts/shortcuts.js");

// With no host element present every module should be a harmless no-op.
console.log("visualizer smoke: 9 scripts loaded, no host -> no crash");
process.exit(0);