/* SNEPPX-Alg — in-browser Pyodide playground.
 * Only activates on pages containing #sx-playground.
 * Pyodide + NumPy run fully client-side; the SneppX-Alg pure-Python package
 * is fetched as a zip from ../assets/sneppx_py.zip and added to sys.path.
 */
(function () {
    "use strict";

    var root = document.getElementById("sx-playground");
    if (!root) return;

    var statusEl = root.querySelector(".sx-pg-status");
    var editor = root.querySelector(".sx-pg-editor");
    var output = root.querySelector(".sx-pg-output");
    var zipUrl = root.getAttribute("data-zip") || "../assets/sneppx_py.zip";

    function setStatus(msg, cls) {
        if (!statusEl) return;
        statusEl.textContent = msg;
        statusEl.className = "sx-pg-status" + (cls ? " " + cls : "");
    }

    function loadScript(src) {
        return new Promise(function (resolve, reject) {
            var s = document.createElement("script");
            s.src = src;
            s.onload = resolve;
            s.onerror = function () {
                reject(new Error("failed to load " + src));
            };
            document.head.appendChild(s);
        });
    }

    async function boot() {
        if (window.__sxPyodide) return window.__sxPyodide;
        if (window.__sxPyodideLoading) return window.__sxPyodideLoading;
        window.__sxPyodideLoading = (async function () {
            setStatus("Loading Pyodide runtime…");
            await loadScript(
                "https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.js"
            );
            var py = await window.loadPyodide({
                indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.2/full/",
            });
            setStatus("Loading NumPy…");
            await py.loadPackage("numpy");
            setStatus("Fetching SneppX-Alg package…");
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
            setStatus("Ready — SneppX_Alg loaded. Press Run.", "sx-pg-ready");
            return py;
        })();
        try {
            return await window.__sxPyodideLoading;
        } catch (e) {
            window.__sxPyodideLoading = null;
            setStatus("Failed to start: " + e.message, "sx-pg-error");
            throw e;
        }
    }

    async function run() {
        var py;
        try {
            py = await boot();
        } catch (e) {
            return;
        }
        var output_el = output;
        output_el.textContent = "";
        py.setStdout({
            batched: function (s) {
                output_el.textContent += s + "\n";
            },
        });
        py.setStderr({
            batched: function (s) {
                output_el.textContent += s + "\n";
            },
        });
        try {
            await py.runPythonAsync(editor.value);
            if (!output_el.textContent) output_el.textContent = "(no output)";
        } catch (e) {
            output_el.textContent += String(e);
        }
        output_el.scrollTop = output_el.scrollHeight;
    }

    document
        .querySelectorAll(".sx-pg-run")
        .forEach(function (btn) {
            btn.addEventListener("click", run);
        });
    setStatus("Click Run to load the Python runtime (~10s first time).");
})();
