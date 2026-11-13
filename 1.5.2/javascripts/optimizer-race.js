/* SNEPPX-Alg — optimizer race: watch SGD / Adam / AdamW / NAdam descend
 * the same loss surface. Canvas 2D, no dependencies. */
(function () {
    "use strict";

    var root = document.getElementById("sx-race");
    if (!root) return;

    var canvas = root.querySelector(".sx-race-canvas");
    var ctx = canvas.getContext("2d");
    var startBtn = root.querySelector(".sx-race-start");
    var resetBtn = root.querySelector(".sx-race-reset");
    var surfaceSel = root.querySelector(".sx-race-surface");
    var legend = root.querySelector(".sx-race-legend");
    var readout = root.querySelector(".sx-race-readout");

    var W = 720, H = 420;
    canvas.width = W;
    canvas.height = H;

    var surfaces = {
        bowl: {
            name: "elliptic bowl",
            domain: [-2.5, 2.5, -2.5, 2.5],
            start: [2.2, 2.0],
            f: function (x, y) { return 0.5 * (x * x + 12 * y * y); },
            g: function (x, y) { return [x, 12 * y]; },
        },
        rosenbrock: {
            name: "Rosenbrock",
            domain: [-2, 2, -1, 3],
            start: [-1.6, 2.4],
            f: function (x, y) { var a = 1 - x, b = y - x * x; return a * a + 100 * b * b; },
            g: function (x, y) { var b = y - x * x; return [-2 * (1 - x) - 400 * x * b, 200 * b]; },
        },
        valley: {
            name: "curved valley",
            domain: [-2.5, 2.5, -2.5, 2.5],
            start: [2.0, 1.8],
            f: function (x, y) { return 0.5 * (x * x + 6 * y * y + 3 * x * y); },
            g: function (x, y) { return [x + 1.5 * y, 6 * y + 1.5 * x]; },
        },
    };

    var COLORS = { SGD: "#5aa9ff", Adam: "#f5a623", AdamW: "#a06bff", NAdam: "#39c07a" };

    var surf = surfaces.bowl;
    var racers = [];
    var running = false;
    var raf = null;
    var maxSteps = 260;
    var step = 0;

    function makeRacers() {
        var lr = surf === surfaces.rosenbrock ? 0.0016 : 0.035;
        racers = ["SGD", "Adam", "AdamW", "NAdam"].map(function (name) {
            return {
                name: name,
                pos: surf.start.slice(),
                traj: [surf.start.slice()],
                m: [0, 0], v: [0, 0], t: 0,
                lr: name === "SGD" ? lr * 0.6 : lr,
                done: false,
            };
        });
        step = 0;
    }

    function clip(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

    function stepRacer(r) {
        var g = surf.g(r.pos[0], r.pos[1]);
        var lr = r.lr;
        var b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 1e-3;
        if (r.name === "SGD") {
            var mom = 0.9;
            r.m[0] = mom * r.m[0] + g[0];
            r.m[1] = mom * r.m[1] + g[1];
            r.pos[0] -= lr * r.m[0];
            r.pos[1] -= lr * r.m[1];
        } else {
            r.t += 1;
            r.m[0] = b1 * r.m[0] + (1 - b1) * g[0];
            r.m[1] = b1 * r.m[1] + (1 - b1) * g[1];
            r.v[0] = b2 * r.v[0] + (1 - b2) * g[0] * g[0];
            r.v[1] = b2 * r.v[1] + (1 - b2) * g[1] * g[1];
            var mh0 = r.m[0] / (1 - Math.pow(b1, r.t));
            var mh1 = r.m[1] / (1 - Math.pow(b1, r.t));
            var vh0 = r.v[0] / (1 - Math.pow(b2, r.t));
            var vh1 = r.v[1] / (1 - Math.pow(b2, r.t));
            if (r.name === "NAdam") {
                var mu = 0.9, muNext = 0.9 * (1 - 0.5 * Math.pow(0.96, r.t * 0.01));
                mh0 = muNext * mh0 + (1 - mu) * g[0] / (1 - Math.pow(b1, r.t));
                mh1 = muNext * mh1 + (1 - mu) * g[1] / (1 - Math.pow(b1, r.t));
            }
            if (r.name === "AdamW") {
                r.pos[0] -= lr * (mh0 / (Math.sqrt(vh0) + eps) + wd * r.pos[0]);
                r.pos[1] -= lr * (mh1 / (Math.sqrt(vh1) + eps) + wd * r.pos[1]);
            } else {
                r.pos[0] -= lr * mh0 / (Math.sqrt(vh0) + eps);
                r.pos[1] -= lr * mh1 / (Math.sqrt(vh1) + eps);
            }
        }
        var d = surf.domain;
        r.pos[0] = clip(r.pos[0], d[0], d[1]);
        r.pos[1] = clip(r.pos[1], d[2], d[3]);
        r.traj.push(r.pos.slice());
        if (r.traj.length > 600) r.traj.shift();
        if (surf.f(r.pos[0], r.pos[1]) < 1e-6) r.done = true;
    }

    function toPx(x, y) {
        var d = surf.domain;
        return [
            (x - d[0]) / (d[1] - d[0]) * W,
            H - (y - d[2]) / (d[3] - d[2]) * H,
        ];
    }

    var contourCache = null, contourKey = "";

    function drawContour() {
        var d = surf.domain;
        var key = surf.name;
        if (contourKey === key && contourCache) { ctx.drawImage(contourCache, 0, 0); return; }
        var off = document.createElement("canvas");
        off.width = W; off.height = H;
        var octx = off.getContext("2d");
        var img = octx.createImageData(W, H);
        var vals = [], minV = Infinity, maxV = -Infinity;
        for (var py = 0; py < H; py += 2) {
            for (var px = 0; px < W; px += 2) {
                var x = d[0] + (px / W) * (d[1] - d[0]);
                var y = d[2] + ((H - py) / H) * (d[3] - d[2]);
                var v = surf.f(x, y);
                if (isFinite(v)) { minV = Math.min(minV, v); maxV = Math.max(maxV, v); }
            }
        }
        for (var y2 = 0; y2 < H; y2 += 2) {
            for (var x2 = 0; x2 < W; x2 += 2) {
                var xx = d[0] + (x2 / W) * (d[1] - d[0]);
                var yy = d[2] + ((H - y2) / H) * (d[3] - d[2]);
                var fv = surf.f(xx, yy);
                var t = Math.log1p(fv - Math.min(0, minV)) / Math.log1p(maxV - Math.min(0, minV) + 1e-9);
                t = Math.max(0, Math.min(1, t));
                // dark blue (low loss) -> warm (high loss)
                var rr = Math.round(30 + 200 * t);
                var gg = Math.round(40 + 120 * (1 - Math.abs(t - 0.45)));
                var bb = Math.round(140 * (1 - t) + 40);
                for (var dy = 0; dy < 2; dy++) {
                    for (var dx = 0; dx < 2; dx++) {
                        var idx = ((y2 + dy) * W + (x2 + dx)) * 4;
                        img.data[idx] = rr;
                        img.data[idx + 1] = gg;
                        img.data[idx + 2] = bb;
                        img.data[idx + 3] = 255;
                    }
                }
            }
        }
        octx.putImageData(img, 0, 0);
        contourCache = off; contourKey = key;
        ctx.drawImage(off, 0, 0);
    }

    function draw() {
        ctx.clearRect(0, 0, W, H);
        drawContour();
        // optimum marker
        var opt = surf === surfaces.rosenbrock ? toPx(1, 1) : toPx(0, 0);
        ctx.beginPath();
        ctx.arc(opt[0], opt[1], 5, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(255,255,255,0.9)";
        ctx.lineWidth = 1.5;
        ctx.stroke();

        racers.forEach(function (r) {
            ctx.beginPath();
            r.traj.forEach(function (p, i) {
                var q = toPx(p[0], p[1]);
                if (i === 0) ctx.moveTo(q[0], q[1]); else ctx.lineTo(q[0], q[1]);
            });
            ctx.strokeStyle = COLORS[r.name];
            ctx.lineWidth = 2;
            ctx.stroke();
            var last = toPx(r.pos[0], r.pos[1]);
            ctx.beginPath();
            ctx.arc(last[0], last[1], 4.5, 0, Math.PI * 2);
            ctx.fillStyle = COLORS[r.name];
            ctx.fill();
        });

        var txt = racers.map(function (r) {
            return r.name + " " + r.pos.map(function (v) { return (Math.round(v * 1000) / 1000).toFixed(3); }).join(", ");
        }).join("   ");
        readout.textContent = txt;
    }

    function tick() {
        if (!running) return;
        for (var i = 0; i < 2; i++) {
            if (step >= maxSteps) { running = false; break; }
            racers.forEach(function (r) { if (!r.done) stepRacer(r); });
            step++;
        }
        draw();
        if (running) raf = requestAnimationFrame(tick);
        else startBtn.textContent = "▶ Restart";
    }

    function buildLegend() {
        legend.innerHTML = "";
        racers.forEach(function (r) {
            var s = document.createElement("span");
            s.className = "sx-race-key";
            s.innerHTML = '<i style="background:' + COLORS[r.name] + '"></i>' + r.name;
            legend.appendChild(s);
        });
    }

    function reset(newSurface) {
        if (newSurface) { surf = surfaces[surfaceSel.value]; contourCache = null; }
        running = false;
        if (raf) cancelAnimationFrame(raf);
        makeRacers();
        buildLegend();
        draw();
        startBtn.textContent = "▶ Start";
    }

    startBtn.addEventListener("click", function () {
        if (running) { running = false; startBtn.textContent = "▶ Resume"; return; }
        if (startBtn.textContent.indexOf("Restart") >= 0) { reset(false); }
        running = true;
        startBtn.textContent = "⏸ Pause";
        tick();
    });
    resetBtn.addEventListener("click", function () { reset(true); });
    surfaceSel.addEventListener("change", function () { reset(true); });

    reset(true);
})();