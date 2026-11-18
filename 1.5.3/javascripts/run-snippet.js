/* SNEPPX-Alg — "Run this snippet" buttons on code blocks.
 * Injects a ▶ Run button into every Python code block and executes it
 * in the shared Pyodide kernel (loaded on first click). Output is shown
 * in a collapsible panel below the block. Falls back to "open in
 * playground" if Pyodide cannot be loaded.
 */
(function () {
    "use strict";

    var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function loadScript(src) {
        return new Promise(function (resolve, reject) {
            var s = document.createElement("script");
            s.src = src;
            s.onload = resolve;
            s.onerror = function () { reject(new Error("failed to load " + src)); };
            document.head.appendChild(s);
        });
    }

    async function boot() {
        if (window.__sxPyodide) return window.__sxPyodide;
        if (window.__sxPyodideLoading) return window.__sxPyodideLoading;
        window.__sxPyodideLoading = (async function () {
            await loadScript("https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.js");
            var py = await window.loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.2/full/" });
            await py.loadPackage("numpy");
            // Reuse the playground's zip-fetch if available; otherwise fetch package.
            var zipUrl = (document.getElementById("sx-playground") || {}).getAttribute
                ? document.getElementById("sx-playground").getAttribute("data-zip")
                : null;
            zipUrl = zipUrl || "../assets/sneppx_py.zip";
            await py.runPythonAsync(
                [
                    "import sys, io, zipfile, os",
                    "from pyodide.http import pyfetch",
                    "resp = await pyfetch(" + JSON.stringify(zipUrl) + ")",
                    "data = await resp.bytes()",
                    "os.makedirs('/sneppx', exist_ok=True)",
                    "with zipfile.ZipFile(io.BytesIO(data)) as z:",
                    "    z.extractall('/sneppx')",
                    "if '/sneppx' not in sys.path:",
                    "    sys.path.insert(0, '/sneppx')",
                    "import SneppX_ALG",
                    "'ok'",
                ].join("\n")
            );
            window.__sxPyodide = py;
            return py;
        })();
        try {
            return await window.__sxPyodideLoading;
        } catch (e) {
            window.__sxPyodideLoading = null;
            throw e;
        }
    }

    function getCode(pre) {
        var codeEl = pre.querySelector("code");
        if (!codeEl) return "";
        return (codeEl.textContent || "").replace(/\u00A0/g, " ");
    }

    function runBlock(pre, btn, out) {
        btn.disabled = true;
        btn.textContent = "⏳ Running…";
        out.textContent = "";
        out.classList.add("sx-run-out--open");
        boot()
            .then(function (py) {
                py.setStdout({ batched: function (s) { out.textContent += s + "\n"; } });
                py.setStderr({ batched: function (s) { out.textContent += s + "\n"; } });
                return py.runPythonAsync(getCode(pre));
            })
            .then(function () {
                if (!out.textContent.trim()) out.textContent = "(no output)";
            })
            .catch(function (e) {
                out.textContent += String(e);
            })
            .then(function () {
                btn.disabled = false;
                btn.textContent = "▶ Run";
                out.scrollTop = out.scrollHeight;
            });
    }

    function buildBlock(pre) {
        if (pre.dataset.sxRunBound) return;
        var code = getCode(pre);
        if (!code.trim() || code.length > 4000) return; // skip huge blocks
        // only Python blocks
        var cls = (pre.querySelector("code") || {}).className || "";
        if (cls.indexOf("python") < 0 && cls.indexOf("py") < 0) return;
        pre.dataset.sxRunBound = "1";

        var wrap = document.createElement("div");
        wrap.className = "sx-run-wrap";

        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "sx-run-btn";
        btn.textContent = "▶ Run";
        btn.title = "Run this snippet in the browser (Pyodide)";

        var out = document.createElement("pre");
        out.className = "sx-run-out";

        btn.addEventListener("click", function () {
            runBlock(pre, btn, out);
        });

        pre.parentNode.insertBefore(wrap, pre);
        wrap.appendChild(pre);
        wrap.appendChild(btn);
        wrap.appendChild(out);
    }

    function init() {
        var pres = document.querySelectorAll(".md-typeset pre");
        pres.forEach(buildBlock);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();