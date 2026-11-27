/* SneppX-ALG local Python playground powered by Pyodide. */
// prettier-ignore
(function () {
  "use strict";

  var root = document.getElementById("sx-playground");
  if (!root) return;

  var status = root.querySelector(".sx-pg-status");
  var editor = root.querySelector(".sx-pg-editor");
  var output = root.querySelector(".sx-pg-output");
  var runButton = root.querySelector(".sx-pg-run");
  var zipUrl = root.getAttribute("data-zip") || "../assets/sneppx_py.zip";
  var storageKey = "sx-pg-code-v2";
  var running = false;
  var starterCode = editor ? editor.value : "";

  function setStatus(message, kind) {
    if (!status) return;

    status.textContent = message;
    status.className = "sx-pg-status";
    if (kind === "ready") status.classList.add("sx-pg-ready");
    if (kind === "error") status.classList.add("sx-pg-error");
  }

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var script = document.createElement("script");
      script.src = src;
      script.onload = resolve;
      script.onerror = function () {
        reject(new Error("Could not load the Pyodide runtime."));
      };
      document.head.appendChild(script);
    });
  }

  async function boot() {
    if (window.__sxPyodide) return window.__sxPyodide;
    if (window.__sxPyodideLoading) return window.__sxPyodideLoading;

    window.__sxPyodideLoading = (async function () {
      setStatus("Downloading Python runtime…");
      await loadScript("https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.js");

      var py = await window.loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.2/full/",
      });

      setStatus("Installing NumPy…");
      await py.loadPackage("numpy");

      setStatus("Loading SneppX-ALG package…");
      var setup = [
        "import sys, io, zipfile, os",
        "from pyodide.http import pyfetch",
        "response = await pyfetch(" + JSON.stringify(zipUrl) + ")",
        "if not response.ok: raise RuntimeError('Package download failed: HTTP ' + str(response.status))",
        "archive = await response.bytes()",
        "os.makedirs('/sneppx', exist_ok=True)",
        "with zipfile.ZipFile(io.BytesIO(archive)) as package_zip:",
        "    package_zip.extractall('/sneppx')",
        "if '/sneppx' not in sys.path: sys.path.insert(0, '/sneppx')",
        "import SneppX_ALG",
        "'SneppX-ALG ready'",
      ].join("\n");

      await py.runPythonAsync(setup);
      window.__sxPyodide = py;
      setStatus("Ready — edit code and press Run.", "ready");
      return py;
    })();

    try {
      return await window.__sxPyodideLoading;
    } catch (error) {
      window.__sxPyodideLoading = null;
      setStatus("Runtime failed: " + error.message, "error");
      throw error;
    }
  }

  function setBusy(value) {
    running = value;
    if (runButton) {
      runButton.disabled = value;
      runButton.textContent = value ? "Running…" : "▶ Run";
    }

    root
      .querySelectorAll(".sx-pg-example, .sx-pg-clear")
      .forEach(function (button) {
        button.disabled = value;
      });
  }

  async function run() {
    if (running || !editor || !output) return;

    setBusy(true);
    output.textContent = "";
    output.classList.add("sx-pg-output--visible");
    setStatus("Starting Python…");

    try {
      var py = await boot();
      py.setStdout({
        batched: function (value) {
          output.textContent += value + "\n";
        },
      });
      py.setStderr({
        batched: function (value) {
          output.textContent += value + "\n";
        },
      });

      setStatus("Executing snippet…");
      await py.runPythonAsync(editor.value);
      if (!output.textContent.trim()) output.textContent = "(no output)";
      setStatus("Execution completed.", "ready");
    } catch (error) {
      output.textContent += String(error);
      setStatus("Execution failed. Review the output.", "error");
    } finally {
      setBusy(false);
      output.scrollTop = output.scrollHeight;
    }
  }

  function save() {
    try {
      localStorage.setItem(storageKey, editor.value);
      setStatus("Snippet saved in this browser.", "ready");
    } catch (error) {
      setStatus("Could not save locally: " + error.message, "error");
    }
  }

  function share() {
    try {
      var encoded = btoa(unescape(encodeURIComponent(editor.value)));
      var url = window.location.href.split("#")[0] + "#code=" + encoded;

      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard
          .writeText(url)
          .then(function () {
            setStatus("Share link copied.", "ready");
          })
          .catch(function () {
            window.prompt("Copy this share link:", url);
          });
      } else {
        window.prompt("Copy this share link:", url);
      }
    } catch (error) {
      setStatus("Could not create a share link: " + error.message, "error");
    }
  }

  function clearOutput() {
    output.textContent = "(output cleared)";
    output.classList.add("sx-pg-output--visible");
    setStatus("Output cleared.");
  }

  function indentSelection(outdent) {
    var start = editor.selectionStart;
    var end = editor.selectionEnd;
    var value = editor.value;

    if (!outdent) {
      editor.value = value.slice(0, start) + "    " + value.slice(end);
      editor.selectionStart = editor.selectionEnd = start + 4;
      return;
    }

    var lineStart = value.lastIndexOf("\n", start - 1) + 1;
    var lines = value.slice(lineStart, end).split("\n");
    var removed = 0;

    lines = lines.map(function (line) {
      if (line.slice(0, 4) === "    ") {
        removed += 4;
        return line.slice(4);
      }
      if (line.charAt(0) === "\t") {
        removed += 1;
        return line.slice(1);
      }
      return line;
    });

    editor.value =
      value.slice(0, lineStart) + lines.join("\n") + value.slice(end);
    editor.selectionStart = Math.max(
      lineStart,
      start - Math.min(4, start - lineStart),
    );
    editor.selectionEnd = Math.max(editor.selectionStart, end - removed);
  }

  var examples = {
    starter: starterCode,
    autograd: [
      "import numpy as np",
      "from SneppX_ALG import Tensor",
      "",
      "x = Tensor(np.array([2.0, 3.0]), requires_grad=True)",
      "y = (x * x).sum()",
      "y.backward()",
      "print('value:', y.data)",
      "print('gradient:', x.grad.data)",
    ].join("\n"),
    linear: [
      "from SneppX_ALG import Tensor",
      "from SneppX_ALG.interface_bindings import nn",
      "",
      "layer = nn.Linear(2, 3)",
      "x = Tensor([[1.0, 0.0]])",
      "y = layer(x)",
      "print('output shape:', y.data.shape)",
      "print('output:', y.data)",
    ].join("\n"),
    sparse: [
      "import numpy as np",
      "from SneppX_ALG.interface_bindings.sparse import sparse_coo_tensor",
      "",
      "indices = np.array([[0, 1, 2], [0, 1, 2]])",
      "values = np.array([1.0, 2.0, 3.0])",
      "sp = sparse_coo_tensor(indices, values, (3, 3))",
      "print('dense:')",
      "print(sp.to_dense().data)",
    ].join("\n"),
  };

  root.querySelectorAll(".sx-pg-run").forEach(function (button) {
    button.addEventListener("click", run);
  });

  root.querySelectorAll(".sx-pg-example").forEach(function (button) {
    button.addEventListener("click", function () {
      var key = button.getAttribute("data-example");
      if (!examples[key]) return;

      editor.value = examples[key];
      save();
      editor.focus();
    });
  });

  var clearButton = root.querySelector(".sx-pg-clear");
  var saveButton = root.querySelector(".sx-pg-save");
  var shareButton = root.querySelector(".sx-pg-share");

  if (clearButton) clearButton.addEventListener("click", clearOutput);
  if (saveButton) saveButton.addEventListener("click", save);
  if (shareButton) shareButton.addEventListener("click", share);

  try {
    var match = (window.location.hash || "").match(/[#&]code=([^&]+)/);
    if (match) {
      editor.value = decodeURIComponent(escape(atob(match[1])));
    } else {
      var saved = localStorage.getItem(storageKey);
      if (saved) editor.value = saved;
    }
  } catch (error) {
    setStatus("Could not restore saved code: " + error.message, "error");
  }

  if (editor) {
    editor.addEventListener("input", function () {
      try {
        localStorage.setItem(storageKey, editor.value);
      } catch (error) {
        // Local storage can be disabled by the browser.
      }
    });

    editor.addEventListener("keydown", function (event) {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        run();
      } else if (
        (event.ctrlKey || event.metaKey) &&
        event.shiftKey &&
        event.key.toLowerCase() === "s"
      ) {
        event.preventDefault();
        share();
      } else if (
        (event.ctrlKey || event.metaKey) &&
        event.key.toLowerCase() === "s"
      ) {
        event.preventDefault();
        save();
      } else if (
        (event.ctrlKey || event.metaKey) &&
        event.shiftKey &&
        event.key === "Backspace"
      ) {
        event.preventDefault();
        clearOutput();
      } else if (
        (event.ctrlKey || event.metaKey) &&
        event.shiftKey &&
        event.key.toLowerCase() === "r"
      ) {
        event.preventDefault();
        editor.value = examples.starter;
        save();
        setStatus("Starter example restored.");
      } else if (event.key === "Tab") {
        event.preventDefault();
        indentSelection(event.shiftKey);
      }
    });
  }

  setStatus("Ready. First run downloads the Python runtime.");
})();
