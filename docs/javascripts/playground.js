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
    /* ---- Restore from URL hash (shareable permalinks) --------------------- */
    try {
        var hash = location.hash || "";
        var m = hash.match(/[#&]code=([^&]+)/);
        if (m && editor) {
            try { editor.value = decodeURIComponent(escape(atob(m[1]))); } catch (e) {}
        }
    } catch (e) {}
    /* ---- Save / restore code via localStorage ----------------------------- */
    var storageKey = "sx-pg-code";
    try {
        var saved = localStorage.getItem(storageKey);
        if (saved && editor && !(location.hash || "").match(/[#&]code=/)) editor.value = saved;
    } catch (e) {}
    if (editor) {
        editor.addEventListener("input", function () {
            try {
                localStorage.setItem(storageKey, editor.value);
            } catch (e) {}
        });
    }

    /* ---- Example snippet buttons ----------------------------------------- */
    var examples = {
        mlp: "import numpy as np\\nfrom SneppX_ALG import Tensor\\nfrom SneppX_ALG.interface_bindings import nn, optim\\n\\nnp.random.seed(0)\\n\\nclass MLP(nn.Module):\\n    def __init__(self):\\n        super().__init__()\\n        self.fc1 = nn.Linear(4, 8)\\n        self.fc2 = nn.Linear(8, 2)\\n    def forward(self, x):\\n        return self.fc2(nn.functional.relu(self.fc1(x)))\\n\\nmodel = MLP()\\nopt = optim.Adam(model.parameters(), lr=1e-2)\\nfor step in range(50):\\n    x = Tensor(np.random.randn(8, 4).astype(np.float32))\\n    y = Tensor((2.0 * x.data[:, :2]).astype(np.float32))\\n    loss = nn.functional.mse_loss(model(x), y)\\n    loss.backward()\\n    opt.step()\\n    opt.zero_grad()\\nprint('loss:', round(float(np.asarray(loss.data).reshape(-1)[0]), 4))\\n",
        autograd: "import numpy as np\\nfrom SneppX_ALG import Tensor\\n\\nx = Tensor(np.array([2.0, 3.0]), requires_grad=True)\\ny = (x * x).sum()\\ny.backward()\\nprint('grad:', x.grad.data())\\n",
        sparse: "import numpy as np\\nfrom SneppX_ALG.interface_bindings.sparse import sparse_coo_tensor\\n\\nindices = np.array([[0, 1, 2], [0, 1, 2]])\\nvalues = np.array([1.0, 2.0, 3.0])\\nsp = sparse_coo_tensor(indices, values, (3, 3))\\nprint('dense:')\\nprint(sp.to_dense().data())\\n",
    };
    document.querySelectorAll(".sx-pg-example").forEach(function (btn) {
        btn.addEventListener("click", function () {
            var key = btn.getAttribute("data-example");
            if (examples[key] && editor) {
                editor.value = examples[key];
                try { localStorage.setItem(storageKey, editor.value); } catch (e) {}
            }
        });
    });
    /* ---- Share permalink button ------------------------------------------- */
    var shareBtn = document.createElement("button");
    shareBtn.type = "button";
    shareBtn.className = "sx-pg-share";
    shareBtn.textContent = "🔗 Copy link";
    shareBtn.title = "Copy a shareable link to this snippet";
    var pgControls = root.querySelector(".sx-pg-controls") || root;
    pgControls.appendChild(shareBtn);
    shareBtn.addEventListener("click", function () {
        try {
            var b64 = btoa(unescape(encodeURIComponent(editor.value)));
            var url = location.href.split("#")[0] + "#code=" + b64;
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(url).then(function () {
                    shareBtn.textContent = "✓ Copied";
                    setTimeout(function () { shareBtn.textContent = "🔗 Copy link"; }, 1500);
                });
            } else {
                prompt("Copy this link:", url);
            }
        } catch (e) {}
    });

    setStatus("Click Run to load the Python runtime (~10s first time).");
})();
