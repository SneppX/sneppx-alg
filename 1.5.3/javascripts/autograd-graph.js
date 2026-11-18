/* SNEPPX-Alg — interactive autograd graph visualizer.
 * Uses window.SxAgvCore for parsing + gradients; this file only renders.
 */
(function () {
    "use strict";

    var host = document.getElementById("sx-agv");
    if (!host) return;
    var core = window.SxAgvCore;
    if (!core) return;

    var VAR_DEFAULTS = { a: 2, b: 3, c: 1, x: 1.5, w: -0.5 };
    var NS = "http://www.w3.org/2000/svg";
    var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    var svg = host.querySelector(".sx-agv-svg");
    var errEl = host.querySelector(".sx-agv-error");
    var outEl = host.querySelector(".sx-agv-out");
    var exprInput = host.querySelector(".sx-agv-expr");
    var varBox = host.querySelector(".sx-agv-vars");
    var runBtn = host.querySelector(".sx-agv-run");
    var stepBtn = host.querySelector(".sx-agv-step");
    var resetBtn = host.querySelector(".sx-agv-reset");

    var env = {};
    Object.keys(VAR_DEFAULTS).forEach(function (k) {
        var row = document.createElement("label");
        row.className = "sx-agv-var";
        var name = document.createElement("span");
        name.textContent = k + " =";
        var inp = document.createElement("input");
        inp.type = "number";
        inp.step = "0.1";
        inp.value = VAR_DEFAULTS[k];
        inp.addEventListener("input", function () { env[k] = parseFloat(inp.value) || 0; });
        env[k] = VAR_DEFAULTS[k];
        row.appendChild(name);
        row.appendChild(inp);
        varBox.appendChild(row);
    });

    var state = null;

    function fmt(x) {
        if (!isFinite(x)) return String(x);
        return (Math.round(x * 1e6) / 1e6).toString();
    }
    function clearSvg() { while (svg.firstChild) svg.removeChild(svg.firstChild); }

    function build() {
        errEl.textContent = "";
        var src = (exprInput.value || "").trim();
        if (!src) return;
        var ast;
        try {
            ast = core.parse(src, env);
        } catch (e) {
            errEl.textContent = "Parse error: " + e.message;
            state = null;
            clearSvg();
            outEl.textContent = "";
            return;
        }
        var order = core.topo(ast.out);
        var depth = {};
        order.forEach(function (n) {
            var d = 0;
            n.inputs.forEach(function (p) { d = Math.max(d, (depth[p.id] || 0) + 1); });
            depth[n.id] = d;
        });
        var byDepth = {};
        order.forEach(function (n) { (byDepth[depth[n.id]] = byDepth[depth[n.id]] || []).push(n); });
        state = { ast: ast, order: order, byDepth: byDepth, step: -1 };
        render();
        outEl.textContent =
            "f = " + fmt(ast.out.value) + "   ·   " +
            Object.keys(env).map(function (k) { return k + "=" + fmt(env[k]); }).join(", ");
        stepBtn.disabled = false;
        resetBtn.disabled = false;
    }

    function render() {
        clearSvg();
        if (!state) return;
        var depths = Object.keys(state.byDepth).map(Number);
        var maxDepth = Math.max.apply(null, depths);
        var colGap = 130, rowGap = 64, padX = 46, padY = 40;
        var maxRows = 1;
        depths.forEach(function (d) { maxRows = Math.max(maxRows, state.byDepth[d].length); });
        var w = padX * 2 + maxDepth * colGap;
        var h = padY * 2 + (maxRows - 1) * rowGap;
        svg.setAttribute("viewBox", "0 0 " + w + " " + h);

        var pos = {};
        depths.forEach(function (d) {
            var list = state.byDepth[d];
            var x = padX + d * colGap;
            var total = (list.length - 1) * rowGap;
            var top = h / 2 - total / 2;
            list.forEach(function (n, i) { pos[n.id] = { x: x, y: top + i * rowGap }; });
        });

        state.order.forEach(function (n) {
            n.inputs.forEach(function (p) {
                var a = pos[p.id], b = pos[n.id];
                var line = document.createElementNS(NS, "line");
                line.setAttribute("x1", a.x + 24);
                line.setAttribute("y1", a.y);
                line.setAttribute("x2", b.x - 24);
                line.setAttribute("y2", b.y);
                line.setAttribute("class", "sx-agv-edge");
                svg.appendChild(line);
            });
        });

        state.order.forEach(function (n) {
            var p = pos[n.id];
            var g = document.createElementNS(NS, "g");
            g.setAttribute("class", "sx-agv-node");
            g.setAttribute("data-id", n.id);
            var rect = document.createElementNS(NS, "rect");
            rect.setAttribute("x", p.x - 24);
            rect.setAttribute("y", p.y - 19);
            rect.setAttribute("width", 48);
            rect.setAttribute("height", 38);
            rect.setAttribute("rx", 7);
            g.appendChild(rect);
            var t1 = document.createElementNS(NS, "text");
            t1.setAttribute("x", p.x);
            t1.setAttribute("y", p.y - 4);
            t1.setAttribute("text-anchor", "middle");
            t1.setAttribute("class", "sx-agv-op");
            t1.textContent = n.label.length > 6 ? n.label.slice(0, 6) : n.label;
            g.appendChild(t1);
            var t2 = document.createElementNS(NS, "text");
            t2.setAttribute("x", p.x);
            t2.setAttribute("y", p.y + 11);
            t2.setAttribute("text-anchor", "middle");
            t2.setAttribute("class", "sx-agv-val");
            t2.textContent = fmt(n.value);
            g.appendChild(t2);
            var t3 = document.createElementNS(NS, "text");
            t3.setAttribute("x", p.x);
            t3.setAttribute("y", p.y - 25);
            t3.setAttribute("text-anchor", "middle");
            t3.setAttribute("class", "sx-agv-grad");
            g.appendChild(t3);
            svg.appendChild(g);
        });
    }

    function revealGrads(step) {
        var order = state.order;
        order.forEach(function (n) { n.grad = 0; });
        order[order.length - 1].grad = 1;
        var threshold = order.length - 1 - step;
        for (var i = order.length - 1; i >= 0; i--) {
            if (i < threshold) break;
            var n = order[i];
            if (!n.inputs.length || !n.grad) continue;
            var gs = core.localGrads(n);
            n.inputs.forEach(function (p, k) { p.grad += n.grad * gs[k]; });
        }
    }

    function updateGradLabels() {
        state.order.forEach(function (n) {
            var g = svg.querySelector('.sx-agv-node[data-id="' + n.id + '"] .sx-agv-grad');
            if (g) g.textContent = n.grad ? "∂=" + fmt(n.grad) : "";
        });
    }

    function animate() {
        if (!state) return;
        state.step = -1;
        var total = state.order.length;
        (function tick() {
            if (!state) return;
            if (state.step >= total - 1) return;
            state.step++;
            revealGrads(state.step);
            updateGradLabels();
            if (state.step < total - 1) {
                setTimeout(tick, reduced ? 0 : 300);
            } else {
                outEl.textContent = outEl.textContent.replace(/ — backward complete$/, "") + " — backward complete";
            }
        })();
    }

    function stepBackward() {
        if (!state) build();
        if (!state) return;
        if (state.step >= state.order.length - 1) return;
        state.step++;
        revealGrads(state.step);
        updateGradLabels();
    }

    runBtn.addEventListener("click", function () { build(); if (state) animate(); });
    stepBtn.addEventListener("click", stepBackward);
    resetBtn.addEventListener("click", function () {
        state = null;
        clearSvg();
        outEl.textContent = "";
        errEl.textContent = "";
        stepBtn.disabled = true;
        resetBtn.disabled = true;
    });
    exprInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter") { e.preventDefault(); build(); if (state) animate(); }
    });
    host.querySelectorAll(".sx-agv-preset").forEach(function (b) {
        b.addEventListener("click", function () {
            exprInput.value = b.getAttribute("data-expr");
            build();
            if (state) animate();
        });
    });

    stepBtn.disabled = true;
    resetBtn.disabled = true;
    build();
    if (state) animate();
})();