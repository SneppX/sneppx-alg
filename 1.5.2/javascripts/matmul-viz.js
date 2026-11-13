/* SNEPPX-Alg — matmul visualizer: hover an output cell to see the
 * row · column dot product that produced it. No dependencies. */
(function () {
    "use strict";

    var host = document.getElementById("sx-mm");
    if (!host) return;

    var M = 3, K = 4, N = 3;
    var A = [], B = [];

    function rnd() { return Math.round((Math.random() * 4 - 2) * 10) / 10; }
    function randomize() {
        A = []; B = [];
        for (var i = 0; i < M; i++) { var r = []; for (var k = 0; k < K; k++) r.push(rnd()); A.push(r); }
        for (var k2 = 0; k2 < K; k2++) { var r2 = []; for (var j = 0; j < N; j++) r2.push(rnd()); B.push(r2); }
    }
    function dot(i, j) {
        var s = 0;
        for (var k = 0; k < K; k++) s += A[i][k] * B[k][j];
        return s;
    }
    function fmt(x) { return (Math.round(x * 100) / 100).toString(); }

    var gridA = host.querySelector(".sx-mm-a");
    var gridB = host.querySelector(".sx-mm-b");
    var gridC = host.querySelector(".sx-mm-c");
    var explain = host.querySelector(".sx-mm-explain");

    function cell(text, cls, attrs) {
        var d = document.createElement("div");
        d.className = "sx-mm-cell " + (cls || "");
        d.textContent = text;
        if (attrs) Object.keys(attrs).forEach(function (k) { d.setAttribute(k, attrs[k]); });
        return d;
    }

    function render() {
        gridA.innerHTML = ""; gridB.innerHTML = ""; gridC.innerHTML = "";
        gridA.style.gridTemplateColumns = "repeat(" + K + ", 1fr)";
        gridB.style.gridTemplateColumns = "repeat(" + N + ", 1fr)";
        gridC.style.gridTemplateColumns = "repeat(" + N + ", 1fr)";

        for (var i = 0; i < M; i++)
            for (var k = 0; k < K; k++)
                gridA.appendChild(cell(fmt(A[i][k]), "sx-mm-a-cell", { "data-i": i, "data-k": k }));
        // corner filler to keep alignment simple
        for (var k2 = 0; k2 < K; k2++)
            for (var j2 = 0; j2 < N; j2++)
                gridB.appendChild(cell(fmt(B[k2][j2]), "sx-mm-b-cell", { "data-k": k2, "data-j": j2 }));
        for (var i2 = 0; i2 < M; i2++)
            for (var j3 = 0; j3 < N; j3++)
                gridC.appendChild(cell(fmt(dot(i2, j3)), "sx-mm-c-cell", { "data-i": i2, "data-j": j3 }));

        highlight(null);
    }

    function clearHL() {
        host.querySelectorAll(".sx-mm-cell").forEach(function (c) {
            c.classList.remove("sx-mm-hl-a", "sx-mm-hl-b", "sx-mm-hl-c");
        });
    }

    function highlight(cellEl) {
        clearHL();
        if (!cellEl) { explain.textContent = "Hover an output cell to see its row · column product."; return; }
        var i = cellEl.getAttribute("data-i"), j = cellEl.getAttribute("data-j");
        var terms = [], sum = 0;
        for (var k = 0; k < K; k++) {
            var a = A[i][k], b = B[k][j];
            sum += a * b;
            terms.push(fmt(a) + "×" + fmt(b));
        }
        host.querySelectorAll('.sx-mm-a-cell[data-i="' + i + '"]').forEach(function (c) { c.classList.add("sx-mm-hl-a"); });
        host.querySelectorAll('.sx-mm-b-cell[data-j="' + j + '"]').forEach(function (c) { c.classList.add("sx-mm-hl-b"); });
        cellEl.classList.add("sx-mm-hl-c");
        explain.textContent = "C[" + i + "][" + j + "] = " + terms.join(" + ") + " = " + fmt(sum);
    }

    gridC.addEventListener("mouseover", function (e) {
        var c = e.target.closest(".sx-mm-c-cell");
        if (c) highlight(c);
    });
    gridC.addEventListener("mouseleave", function () { highlight(null); });
    gridC.addEventListener("click", function (e) {
        var c = e.target.closest(".sx-mm-c-cell");
        if (c) highlight(c);
    });

    var sweepBtn = host.querySelector(".sx-mm-sweep");
    var randBtn = host.querySelector(".sx-mm-rand");
    if (randBtn) randBtn.addEventListener("click", function () { randomize(); render(); });
    if (sweepBtn) sweepBtn.addEventListener("click", function () {
        var cells = Array.prototype.slice.call(host.querySelectorAll(".sx-mm-c-cell"));
        var idx = 0;
        sweepBtn.disabled = true;
        (function tick() {
            if (idx >= cells.length) { sweepBtn.disabled = false; return; }
            highlight(cells[idx++]);
            setTimeout(tick, 380);
        })();
    });

    randomize();
    render();
})();