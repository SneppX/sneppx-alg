---
description: "Run SneppX-ALG Python tensor and autograd examples in the browser using Pyodide and WebAssembly."
---
# In-browser Python playground

Run SneppX-ALG Python code locally in your browser with Pyodide. The first run downloads the Python runtime and NumPy, then loads the pure-Python package archive.

<div id="sx-playground" class="sx-playground" data-zip="../assets/sneppx_py.zip">
<div class="sx-pg-toolbar">
<button class="sx-pg-run" type="button">▶ Run</button>
<button class="sx-pg-save" type="button">Save</button>
<button class="sx-pg-share" type="button">Copy share link</button>
<button class="sx-pg-clear" type="button">Clear output</button>
<span class="sx-pg-status" role="status" aria-live="polite">Ready.</span>
</div>
<div class="sx-pg-controls">
<span>Examples:</span>
<button class="sx-pg-example" type="button" data-example="starter">Starter</button>
<button class="sx-pg-example" type="button" data-example="autograd">Autograd</button>
<button class="sx-pg-example" type="button" data-example="linear">Linear layer</button>
<button class="sx-pg-example" type="button" data-example="sparse">Sparse tensor</button>
</div>
<label for="sx-pg-editor">Python editor</label>
<textarea id="sx-pg-editor" class="sx-pg-editor" spellcheck="false" rows="18">
import numpy as np
from SneppX_ALG import Tensor
a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
loss = (a @ b).sum()
loss.backward()
print("loss =", loss.data.tolist())
print("grad a =", a.grad.data.tolist())
print("grad b =", b.grad.data.tolist())
from SneppX_ALG.interface_bindings import nn
layer = nn.Linear(2, 3)
x = Tensor([[1.0, 0.0]])
y = layer(x)
print("linear output shape:", y.data.shape)
</textarea>
<pre class="sx-pg-output" aria-label="Python output" aria-live="polite">(output appears here)</pre>
<p>Ctrl/⌘+Enter run · Tab indent · Shift+Tab outdent · Ctrl/⌘+S save · ? help</p>
</div>

## Runtime limits

| Available | Not available |
|---|---|
| Pure-Python tensor operations and autograd | CUDA / GPU kernels |
| Included neural-network modules | Native _SNEPPX_c extension |
| Supported optimizers, initialization and sparse utilities | Large model downloads |
| NumPy and bundled pure-Python dependencies | SciPy, ONNX, Hugging Face and external native libraries |

Choose an example, edit code and press **Run**. Save stores code in this browser. Share links encode source in the URL fragment; never include credentials or private data. See the [keyboard guide](keyboard-shortcuts.md) for all shortcuts.