/* SNEPPX-Alg — loss gallery: plot L(z) and dL/dz for a family of losses.
 * Canvas 2D, no dependencies. Derivative is computed numerically
 * (central difference) so every curve matches the loss it is drawn from.
 */
(function () {
    "use strict";

    var root = document.getElementById("sx-loss");
    if (!root) return;

    var canvas = root.querySelector(".sx-loss-canvas");
    var sel = root.querySelector(".sx-loss-select");
    var paramBox = root.querySelector(".sx-loss-params");
    var legend = root.querySelector(".sx-loss-legend");
    var readout = root.querySelector(".sx-loss-readout");
    var ctx = canvas.getContext("2d");

    var W = 660, H = 300;
    canvas.width = W;
    canvas.height = H;

    /* Each entry: value(z, p) and analytic help text. */
    var LOSSES = {
        mse: {
            name: "MSELoss (L2)",
            z: "residual y − t",
            domain: [-3, 3],
            formula: "z²",
            deriv: "2z",
            value: function (z) { return z * z; },
            params: [],
        },
        mae: {
            name: "L1Loss / MAE",
            z: "residual y − t",
            domain: [-3, 3],
            formula: "|z|",
            deriv: "sign(z)",
            value: function (z) { return Math.abs(z); },
            params: [],
        },
        huber: {
            name: "HuberLoss",
            z: "residual y − t",
            domain: [-3, 3],
            formula: "½z² if |z|≤δ else δ(|z|−½δ)",
            deriv: "z if |z|≤δ else δ·sign(z)",
            value: function (z, p) {
                var d = p.delta;
                return Math.abs(z) <= d ? 0.5 * z * z : d * (Math.abs(z) - 0.5 * d);
            },
            params: [{ key: "delta", label: "δ", min: 0.1, max: 2, step: 0.1, value: 1 }],
        },
        smoothl1: {
            name: "SmoothL1Loss",
            z: "residual y − t",
            domain: [-3, 3],
            formula: "0.5z²/β if |z|<β else |z|−0.5β",
            deriv: "z/β if |z|<β else sign(z)",
            value: function (z, p) {
                var b = p.beta;
                return Math.abs(z) < b ? (0.5 * z * z) / b : Math.abs(z) - 0.5 * b;
            },
            params: [{ key: "beta", label: "β", min: 0.1, max: 2, step: 0.1, value: 1 }],
        },
        logcosh: {
            name: "log-cosh",
            z: "residual y − t",
            domain: [-3, 3],
            formula: "log(cosh z)",
            deriv: "tanh z",
            value: function (z) { return Math.log(Math.cosh(z)); },
            params: [],
        },
        hinge: {
            name: "HingeEmbeddingLoss",
            z: "margin y",
            domain: [-2, 3],
            formula: "max(0, m − z)",
            deriv: "−1 if z < m else 0",
            value: function (z, p) { return Math.max(0, p.margin - z); },
            params: [{ key: "margin", label: "m", min: 0, max: 2, step: 0.1, value: 1 }],
        },
        sqhinge: {
            name: "squared hinge",
            z: "margin y",
            domain: [-2, 3],
            formula: "max(0, m − z)²",
            deriv: "−2·max(0, m − z)",
            value: function (z, p) { var t = Math.max(0, p.margin - z); return t * t; },
            params: [{ key: "margin", label: "m", min: 0, max: 2, step: 0.1, value: 1 }],
        },
        triplet: {
            name: "TripletMarginLoss",
            z: "d(a,p) − d(a,n)",
            domain: [-2, 3],
            formula: "max(0, z + m)",
            deriv: "1 if z > −m else 0",
            value: function (z, p) { return Math.max(0, z + p.margin); },
            params: [{ key: "margin", label: "m", min: 0, max: 2, step: 0.1, value: 1 }],
        },
        softmargin: {
            name: "SoftMarginLoss",
            z: "logit · target",
            domain: [-3, 3],
            formula: "log(1 + e^(−z))",
            deriv: "−σ(−z)",
            value: function (z) { return Math.log(1 + Math.exp(-z)); },
            params: [],
        },
        exp: {
            name: "exponential (AdaBoost)",
            z: "logit · target",
            domain: [-2, 3],
            formula: "e^(−z)",
            deriv: "−e^(−z)",
            value: function (z) { return Math.exp(-z); },
            params: [],
        },
        bce: {
            name: "BCEWithLogitsLoss",
            z: "logit (target 1)",
            domain: [-4, 4],
            formula: "log(1 + e^(−z))",
            deriv: "−σ(−z)",
            value: function (z) { return Math.log(1 + Math.exp(-z)); },
            params: [],
        },
        focal: {
            name: "focal loss",
            z: "logit (target 1)",
            domain: [-4, 4],
            formula: "−(1−p)^γ · log p,  p = σ(z)",
            deriv: "(numeric)",
            value: function (z, p) {
                var s = 1 / (1 + Math.exp(-z));
                return -Math.pow(1 - s, p.gamma) * Math.log(s + 1e-12);
            },
            params: [{ key: "gamma", label: "γ", min: 0, max: 5, step: 0.5, value: 2 }],
        },
        poisson: {
            name: "PoissonNLLLoss",
            z: "log(prediction)",
            domain: [-2, 3],
            formula: "e^z − t·z  (t = 1)",
            deriv: "e^z − 1",
            value: function (z) { return Math.exp(z) - z; },
            params: [],
        },
        quantile: {
            name: "quantile loss (pinball)",
            z: "residual y − t",
            domain: [-3, 3],
            formula: "τz if z ≥ 0 else (τ−1)z",
            deriv: "τ or τ−1",
            value: function (z, p) {
                var q = p.tau;
                return z >= 0 ? q * z : (q - 1) * z;
            },
            params: [{ key: "tau", label: "τ", min: 0.1, max: 0.9, step: 0.1, value: 0.5 }],
        },
        cosine: {
            name: "CosineEmbeddingLoss",
            z: "cos(x₁, x₂)",
            domain: [-1, 1],
            formula: "1 − z  (target 1)",
            deriv: "−1",
            value: function (z) { return 1 - z; },
            params: [],
        },
        softplus: {
            name: "Softplus / binary KL",
            z: "log-odds",
            domain: [-4, 4],
            formula: "log(1 + e^z) − ½z",
            deriv: "σ(z) − ½",
            value: function (z) { return Math.log(1 + Math.exp(z)) - 0.5 * z; },
            params: [],
        },
        cauchy: {
            name: "Cauchy / Lorentzian",
            z: "residual y − t",
            domain: [-3, 3],
            formula: "c²·log(1 + (z/c)²)",
            deriv: "2z / (1 + (z/c)²)",
            value: function (z, p) {
                var c = p.c;
                return c * c * Math.log(1 + (z / c) * (z / c));
            },
            params: [{ key: "c", label: "c", min: 0.2, max: 2, step: 0.1, value: 1 }],
        },
        geman: {
            name: "Geman-McClure",
            z: "residual y − t",
            domain: [-3, 3],
            formula: "z² / (1 + z²)",
            deriv: "2z / (1 + z²)²",
            value: function (z) { return (z * z) / (1 + z * z); },
            params: [],
        },
    };

    var current = null;
    var params = {};

    function numDeriv(spec, z) {
        var h = 1e-6;
        return (spec.value(z + h, params) - spec.value(z - h, params)) / (2 * h);
    }

    function sample(spec) {
        var d = spec.domain, n = 320;
        var pts = [], dpts = [];
        for (var i = 0; i <= n; i++) {
            var z = d[0] + ((d[1] - d[0]) * i) / n;
            var v = spec.value(z, params);
            var dv = numDeriv(spec, z);
            if (isFinite(v)) pts.push([z, v]);
            if (isFinite(dv)) dpts.push([z, dv]);
        }
        return { pts: pts, dpts: dpts };
    }

    function range(list) {
        var lo = Infinity, hi = -Infinity;
        list.forEach(function (p) {
            if (p[1] < lo) lo = p[1];
            if (p[1] > hi) hi = p[1];
        });
        if (!isFinite(lo) || !isFinite(hi)) { lo = 0; hi = 1; }
        if (hi - lo < 1e-9) { hi = lo + 1; }
        return [lo, hi];
    }

    function draw() {
        if (!current) return;
        var spec = LOSSES[current];
        var s = sample(spec);
        var padL = 46, padR = 46, padT = 18, padB = 34;
        var plotW = W - padL - padR, plotH = H - padT - padB;
        var d = spec.domain;
        var lr = range(s.pts), dr = range(s.dpts);
        // include zero in both ranges so the axes make sense
        lr = [Math.min(0, lr[0]), Math.max(0, lr[1])];
        dr = [Math.min(0, dr[0]), Math.max(0, dr[1])];
        if (lr[1] - lr[0] < 1e-9) lr[1] = lr[0] + 1;
        if (dr[1] - dr[0] < 1e-9) dr[1] = dr[0] + 1;

        function X(z) { return padL + ((z - d[0]) / (d[1] - d[0])) * plotW; }
        function Y(v, r) { return padT + plotH - ((v - r[0]) / (r[1] - r[0])) * plotH; }

        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = "rgba(127,127,127,0.06)";
        ctx.fillRect(padL, padT, plotW, plotH);

        // grid
        ctx.strokeStyle = "rgba(127,127,127,0.22)";
        ctx.lineWidth = 1;
        for (var g = 0; g <= 4; g++) {
            var y = padT + (plotH * g) / 4;
            ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();
            var x = padL + (plotW * g) / 4;
            ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
        }

        // zero lines
        ctx.strokeStyle = "rgba(127,127,127,0.65)";
        ctx.beginPath(); ctx.moveTo(padL, Y(0, lr)); ctx.lineTo(padL + plotW, Y(0, lr)); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(X(0), padT); ctx.lineTo(X(0), padT + plotH); ctx.stroke();

        function curve(pts, r, color, dash) {
            ctx.beginPath();
            ctx.setLineDash(dash || []);
            var started = false;
            pts.forEach(function (p) {
                var px = X(p[0]), py = Y(p[1], r);
                if (!started) { ctx.moveTo(px, py); started = true; }
                else ctx.lineTo(px, py);
            });
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.2;
            ctx.stroke();
            ctx.setLineDash([]);
        }

        curve(s.dpts, dr, "#39c07a", [6, 4]);
        curve(s.pts, lr, "#a06bff", []);

        // x ticks
        ctx.fillStyle = "currentColor";
        ctx.globalAlpha = 0.7;
        ctx.font = "10px system-ui, sans-serif";
        ctx.textAlign = "center";
        for (var t = 0; t <= 4; t++) {
            var zz = d[0] + ((d[1] - d[0]) * t) / 4;
            ctx.fillText(fmt(zz), X(zz), H - 12);
        }
        // y labels
        ctx.textAlign = "right";
        ctx.fillStyle = "#a06bff";
        ctx.fillText("L " + fmt(lr[1]), padL - 6, padT + 10);
        ctx.fillText("L " + fmt(lr[0]), padL - 6, padT + plotH);
        ctx.textAlign = "left";
        ctx.fillStyle = "#39c07a";
        ctx.fillText("dL " + fmt(dr[1]), padL + plotW + 6, padT + 10);
        ctx.fillText("dL " + fmt(dr[0]), padL + plotW + 6, padT + plotH);
        ctx.globalAlpha = 1;

        legend.innerHTML =
            '<span class="sx-loss-key"><i style="background:#a06bff"></i>L(z)</span>' +
            '<span class="sx-loss-key"><i style="background:#39c07a"></i>dL/dz (numeric)</span>';

        readout.textContent =
            spec.name + " · " + spec.formula +
            (Object.keys(params).length
                ? " · " + Object.keys(params).map(function (k) { return k + "=" + params[k]; }).join(", ")
                : "");
    }

    function fmt(x) {
        if (!isFinite(x)) return "∞";
        var a = Math.abs(x);
        if (a >= 100) return Math.round(x).toString();
        if (a >= 10) return (Math.round(x * 10) / 10).toString();
        return (Math.round(x * 100) / 100).toString();
    }

    function buildParams(spec) {
        paramBox.innerHTML = "";
        params = {};
        spec.params.forEach(function (p) {
            params[p.key] = p.value;
            var label = document.createElement("label");
            label.className = "sx-loss-param";
            var span = document.createElement("span");
            span.textContent = p.label + " = " + p.value;
            var inp = document.createElement("input");
            inp.type = "range";
            inp.min = p.min; inp.max = p.max; inp.step = p.step; inp.value = p.value;
            inp.addEventListener("input", function () {
                params[p.key] = parseFloat(inp.value);
                span.textContent = p.label + " = " + params[p.key];
                draw();
            });
            label.appendChild(span);
            label.appendChild(inp);
            paramBox.appendChild(label);
        });
    }

    function select(id) {
        current = id;
        buildParams(LOSSES[id]);
        draw();
    }

    Object.keys(LOSSES).forEach(function (id) {
        var o = document.createElement("option");
        o.value = id;
        o.textContent = LOSSES[id].name;
        sel.appendChild(o);
    });
    sel.addEventListener("change", function () { select(sel.value); });

    select(sel.value || "mse");
})();
