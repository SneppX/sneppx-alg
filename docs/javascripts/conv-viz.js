/* SNEPPX-Alg — Conv2d visualizer: sliding window, stride, padding.
 * Hover an output cell to highlight its receptive field. No dependencies. */
(function () {
    "use strict";

    var host = document.getElementById("sx-cv");
    if (!host) return;

    var H = 7, W = 7, KD = 3;
    var input = [], kernel = [];
    var stride = 1, pad = 0;

    function rnd() { return Math.round((Math.random() * 2 - 1) * 10) / 10; }
    function fmt(x) { return (Math.round(x * 100) / 100).toString(); }

    function randomize() {
        input = []; kernel = [];
        for (var i = 0; i < H; i++) { var r = []; for (var j = 0; j < W; j++) r.push(rnd()); input.push(r); }
        for (var a = 0; a < KD; a++) { var k = []; for (var b = 0; b < KD; b++) k.push(rnd()); kernel.push(k); }
    }

    var inGrid = host.querySelector(".sx-cv-in");
    var kGrid = host.querySelector(".sx-cv-k");
    var outGrid = host.querySelector(".sx-cv-out");
    var explain = host.querySelector(".sx-cv-explain");
    var strideIn = host.querySelector(".sx-cv-stride");
    var padIn = host.querySelector(".sx-cv-pad");
    var strideVal = host.querySelector(".sx-cv-stride-val");
    var padVal = host.querySelector(".sx-cv-pad-val");
    var shapeEl = host.querySelector(".sx-cv-shape");

    function outSize() {
        return Math.floor((H + 2 * pad - KD) / stride) + 1;
    }

    function cell(text, cls, attrs) {
        var d = document.createElement("div");
        d.className = "sx-cv-cell " + (cls || "");
        d.textContent = text;
        if (attrs) Object.keys(attrs).forEach(function (k) { d.setAttribute(k, attrs[k]); });
        return d;
    }

    function render() {
        inGrid.innerHTML = ""; kGrid.innerHTML = ""; outGrid.innerHTML = "";
        var OS = outSize();
        var PW = W + 2 * pad, PH = H + 2 * pad;
        inGrid.style.gridTemplateColumns = "repeat(" + PW + ", 1fr)";
        kGrid.style.gridTemplateColumns = "repeat(" + KD + ", 1fr)";
        outGrid.style.gridTemplateColumns = "repeat(" + OS + ", 1fr)";

        for (var y = 0; y < PH; y++) {
            for (var x = 0; x < PW; x++) {
                var iy = y - pad, ix = x - pad;
                var inside = iy >= 0 && iy < H && ix >= 0 && ix < W;
                inGrid.appendChild(cell(inside ? fmt(input[iy][ix]) : "0",
                    inside ? "sx-cv-in-cell" : "sx-cv-in-cell sx-cv-pad-cell",
                    { "data-y": y, "data-x": x }));
            }
        }
        for (var a = 0; a < KD; a++)
            for (var b = 0; b < KD; b++)
                kGrid.appendChild(cell(fmt(kernel[a][b]), "sx-cv-k-cell", { "data-a": a, "data-b": b }));
        for (var oy = 0; oy < OS; oy++)
            for (var ox = 0; ox < OS; ox++)
                outGrid.appendChild(cell("·", "sx-cv-out-cell", { "data-oy": oy, "data-ox": ox }));

        shapeEl.textContent = "input " + H + "×" + W + " + pad " + pad + " → conv " + KD + "×" + KD +
            ", stride " + stride + " → output " + OS + "×" + OS;
        highlight(null);
    }

    function clearHL() {
        host.querySelectorAll(".sx-cv-in-cell").forEach(function (c) { c.classList.remove("sx-cv-hl"); });
        host.querySelectorAll(".sx-cv-out-cell").forEach(function (c) { c.classList.remove("sx-cv-hl-out"); });
        host.querySelectorAll(".sx-cv-k-cell").forEach(function (c) { c.classList.remove("sx-cv-hl-k"); });
    }

    function compute(oy, ox) {
        var sum = 0, terms = [];
        for (var a = 0; a < KD; a++) {
            for (var b = 0; b < KD; b++) {
                var y = oy * stride + a, x = ox * stride + b;
                var v = 0;
                if (y >= pad && y < pad + H && x >= pad && x < pad + W) v = input[y - pad][x - pad];
                sum += v * kernel[a][b];
                terms.push(fmt(v) + "×" + fmt(kernel[a][b]));
            }
        }
        return { sum: sum, terms: terms };
    }

    function highlight(outCell) {
        clearHL();
        host.querySelectorAll(".sx-cv-k-cell").forEach(function (c) { c.classList.add("sx-cv-hl-k"); });
        if (!outCell) { explain.textContent = "Hover an output cell to see its receptive field."; return; }
        var oy = Number(outCell.getAttribute("data-oy")), ox = Number(outCell.getAttribute("data-ox"));
        for (var a = 0; a < KD; a++) {
            for (var b = 0; b < KD; b++) {
                var y = oy * stride + a, x = ox * stride + b;
                var c = host.querySelector('.sx-cv-in-cell[data-y="' + y + '"][data-x="' + x + '"]');
                if (c) c.classList.add("sx-cv-hl");
            }
        }
        outCell.classList.add("sx-cv-hl-out");
        outCell.textContent = fmt(compute(oy, ox).sum);
        var r = compute(oy, ox);
        explain.textContent = "out[" + oy + "][" + ox + "] = " + r.terms.join(" + ") + " = " + fmt(r.sum);
    }

    outGrid.addEventListener("mouseover", function (e) {
        var c = e.target.closest(".sx-cv-out-cell");
        if (c) highlight(c);
    });
    outGrid.addEventListener("mouseleave", function () { render(); });
    if (strideIn) strideIn.addEventListener("input", function () {
        stride = Number(strideIn.value); strideVal.textContent = stride; render();
    });
    if (padIn) padIn.addEventListener("input", function () {
        pad = Number(padIn.value); padVal.textContent = pad; render();
    });
    var randBtn = host.querySelector(".sx-cv-rand");
    if (randBtn) randBtn.addEventListener("click", function () { randomize(); render(); });

    randomize();
    render();
})();