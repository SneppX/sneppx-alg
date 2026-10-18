# In-browser Playground

Run **real SneppX-Alg Python** directly in your browser — powered by
[Pyodide](https://pyodide.org) (CPython compiled to WebAssembly). No install,
no server; everything executes on your machine.

> First run downloads the Python runtime (~10&nbsp;s). The NumPy-backed
> `SneppX_ALG` package is fetched from this site and imported inside the WASM
> sandbox. Heavy optional dependencies (CUDA, ONNX, HF) are **not** available
> in the browser — pure-Python tensor / autograd / `nn` code is.

<div id="sx-playground" class="sx-playground" data-zip="../assets/sneppx_py.zip">
<div class="sx-pg-toolbar">
<button class="sx-pg-run" type="button">&#9654;&nbsp;Run</button>
<span class="sx-pg-status">Initialising&hellip;</span>
</div>
<textarea class="sx-pg-editor" spellcheck="false" rows="14">
import SneppX_ALG as sx
from SneppX_ALG import Tensor

# Two tensors that track gradients
a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

# Forward: matrix product, then reduce to a scalar
loss = (a @ b).sum()

# Reverse-mode autodiff
loss.backward()

print("loss   =", loss.data.tolist())
print("grad a =", a.grad.data.tolist())
print("grad b =", b.grad.data.tolist())

# A tiny neural-net layer built from the nn namespace
from SneppX_ALG.interface_bindings import nn
layer = nn.Linear(2, 3)
x = Tensor([[1.0, 0.0]])
y = layer(x)
print("linear out shape:", y.data.shape)
</textarea>
<pre class="sx-pg-output">(output appears here)</pre>
</div>

## What runs in the sandbox?

| Available | Not available |
|-----------|---------------|
| `Tensor`, autograd, `backward()` | CUDA / GPU kernels |
| `nn` modules & functional `F.*` | native `_SNEPPX_c` extension |
| optimizers, init, sparse tensors | model weight downloads |
| the 14 extra Python packages | anything needing SciPy/ONNX/HF |

Edit the code above and press **Run** again — the runtime stays warm between
runs.
