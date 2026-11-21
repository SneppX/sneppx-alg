"""JIT / trace-to-executable-graph pipeline (JAX-like transforms).

Traces a pure Python function operating on :class:`Tracer` proxies into a
symbolic ``GraphNode`` DAG, then provides JAX-style composable transforms:

- ``jit(fn)`` — trace once, execute the compiled graph on numpy arrays.
- ``grad(fn, argnums)`` — exact reverse-mode symbolic differentiation.
- ``value_and_grad`` — forward value + gradient.
- ``jacobian`` / ``hessian`` — full derivatives (arbitrary order via nested
  ``grad`` on the *graph*, not by re-running Python).
- ``vmap(fn)`` — vectorizing transform (batch over a leading axis).

Everything is pure NumPy; no LLM, no CUDA. Traceable ops are the element-wise
family (add/sub/mul/div/neg/abs/exp/log/relu/sigmoid/tanh/gelu/silu/clip,
plus the gradient-only step/sign/erf) and ``matmul`` / ``transpose``.

Typical usage::

    @jit
    def model(x, w, b):
        h = relu(x @ w + b)
        return h @ w.T

    y = model(x, w, b)
    df = grad(lambda x: model(x, w, b).sum(), 0)(x)
"""

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Tracer",
    "Trace",
    "jit",
    "grad",
    "value_and_grad",
    "jacobian",
    "hessian",
    "vmap",
    "trace_function",
]


# --------------------------------------------------------------------------
# op table (self-contained so graph_compiler stays untouched)
# --------------------------------------------------------------------------

def _erf(x):
    return np.vectorize(math.erf)(x)


_IMPL: Dict[str, Callable] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "neg": lambda a: -a,
    "abs": lambda a: np.abs(a),
    "exp": lambda a: np.exp(a),
    "log": lambda a: np.log(a),
    "relu": lambda a: np.maximum(a, 0.0),
    "sigmoid": lambda a: 1.0 / (1.0 + np.exp(-a)),
    "tanh": lambda a: np.tanh(a),
    "gelu": lambda a: 0.5 * a * (1.0 + _erf(a / math.sqrt(2.0))),
    "silu": lambda a: a / (1.0 + np.exp(-a)),
    "step": lambda a: (a > 0.0).astype(np.float32),
    "sign": lambda a: np.sign(a),
    "clip": lambda a, lo, hi: np.clip(a, lo, hi),
    "transpose": lambda a, axes: np.transpose(a, axes) if axes else np.transpose(a),
    "matmul": lambda a, b: np.matmul(a, b),
    "erf": _erf,
    "reduce": lambda a, axis, keepdims: np.sum(a, axis=axis, keepdims=bool(keepdims)),
    "mean": lambda a, axis, keepdims: np.mean(a, axis=axis, keepdims=bool(keepdims)),
    "reshape": lambda a, shape: np.reshape(a, shape),
    "stack": lambda *args: np.stack(args),
    "tuple": lambda *args: args,
}


def _matmul_grad_a(g, b):
    """Gradient w.r.t. the left operand of ``out = a @ b`` (runtime shapes)."""
    g = np.asarray(g)
    b = np.asarray(b)
    if b.ndim <= 1:
        if g.ndim == 0:
            return g * b
        return np.multiply.outer(g, b)
    return g @ np.transpose(b)


def _matmul_grad_b(a, g):
    """Gradient w.r.t. the right operand of ``out = a @ b`` (runtime shapes)."""
    a = np.asarray(a)
    g = np.asarray(g)
    if a.ndim <= 1:
        if g.ndim == 0:
            return g * a
        return np.multiply.outer(a, g)
    return np.transpose(a) @ g


_IMPL["matmul_grad_a"] = _matmul_grad_a
_IMPL["matmul_grad_b"] = _matmul_grad_b


def _apply(node, values: Dict[int, np.ndarray]) -> np.ndarray:
    op = node.op
    if op == "input":
        return values[id(node)]
    if op == "const" or op == "param":
        return node.params["value"]
    if op == "clip":
        return np.clip(
            values[id(node.inputs[0])],
            node.params["min"], node.params["max"],
        )
    if op == "transpose":
        return np.transpose(values[id(node.inputs[0])], node.params.get("axes"))
    if op == "transpose_inv":
        axes = node.params.get("axes")
        return np.transpose(values[id(node.inputs[0])], np.argsort(axes) if axes is not None else None)
    if op == "broadcast_to_shape":
        return np.broadcast_to(values[id(node.inputs[0])], values[id(node.inputs[1])].shape).copy()
    if op == "mean_adjoint":
        a = values[id(node.inputs[1])]
        g = values[id(node.inputs[0])]
        axis = node.params.get("axis")
        keepdims = bool(node.params.get("keepdims"))
        if axis is None:
            return np.full_like(g, np.sum(g) / a.size)
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        n = 1
        for ax in axes:
            n *= a.shape[ax]
        return np.sum(g, axis=axis, keepdims=keepdims) / n
    if op == "matmul":
        return values[id(node.inputs[0])] @ values[id(node.inputs[1])]
    if op == "reduce":
        return np.sum(values[id(node.inputs[0])], axis=node.params.get("axis"), keepdims=bool(node.params.get("keepdims")))
    if op == "mean":
        return np.mean(values[id(node.inputs[0])], axis=node.params.get("axis"), keepdims=bool(node.params.get("keepdims")))
    if op == "reshape":
        return np.reshape(values[id(node.inputs[0])], node.params["shape"])
    if op == "getitem":
        return values[id(node.inputs[0])][node.params["index"]]
    if op == "scatter_like":
        out = np.zeros_like(values[id(node.inputs[1])])
        out[node.params["index"]] = values[id(node.inputs[0])]
        return out
    if op == "reshape_grad":
        a = values[id(node.inputs[1])]
        return np.reshape(values[id(node.inputs[0])], a.shape)
    if op == "reduce_grad":
        a = values[id(node.inputs[1])]
        g = values[id(node.inputs[0])]
        axis = node.params.get("axis")
        keepdims = bool(node.params.get("keepdims"))
        if axis is None:
            return np.full_like(a, g)
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        b = g if keepdims else g
        if not keepdims:
            for ax in axes:
                b = np.expand_dims(b, ax)
        return np.broadcast_to(b, a.shape).copy()
    if op == "mean_grad":
        a = values[id(node.inputs[1])]
        g = values[id(node.inputs[0])]
        axis = node.params.get("axis")
        keepdims = bool(node.params.get("keepdims"))
        if axis is None:
            return np.full_like(a, np.sum(g) / a.size)
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        n = 1
        for ax in axes:
            n *= a.shape[ax]
        b = g
        if not keepdims:
            for ax in axes:
                b = np.expand_dims(b, ax)
        return np.broadcast_to(b, a.shape).copy() / n
    if op == "sum_to_shape":
        g = values[id(node.inputs[0])]
        target = tuple(values[id(node.inputs[1])].shape)
        extra = g.ndim - len(target)
        if extra > 0:
            g = g.sum(axis=tuple(range(extra)))
        if g.ndim < len(target):
            g = g.reshape((1,) * (len(target) - g.ndim) + g.shape)
        axes = tuple(i for i in range(len(target)) if g.shape[i] != target[i])
        if axes:
            g = g.sum(axis=axes, keepdims=True)
        if g.shape != target:
            g = np.broadcast_to(g, target).copy()
        return g
    fn = _IMPL[op]
    args = [values[id(i)] for i in node.inputs]
    return fn(*args)


# --------------------------------------------------------------------------
# Tracer — proxy that builds the GraphNode DAG
# --------------------------------------------------------------------------

class Tracer:
    """Symbolic proxy that records ops into a GraphNode DAG."""

    def __init__(self, node=None, name: str = "arg"):
        self.node = node or _node_input(name)

    def _bin(self, other, op: str):
        return Tracer(_node_op(op, [self.node, _tracer_node(other)]))

    def __add__(self, other): return self._bin(other, "add")
    def __radd__(self, other): return self._bin(other, "add")
    def __sub__(self, other): return self._bin(other, "sub")
    def __rsub__(self, other): return Tracer(_node_op("sub", [_tracer_node(other), self.node]))
    def __mul__(self, other): return self._bin(other, "mul")
    def __rmul__(self, other): return self._bin(other, "mul")
    def __truediv__(self, other): return self._bin(other, "div")
    def __rtruediv__(self, other): return Tracer(_node_op("div", [_tracer_node(other), self.node]))
    def __neg__(self): return Tracer(_node_op("neg", [self.node]))
    def __abs__(self): return Tracer(_node_op("abs", [self.node]))
    def __matmul__(self, other): return Tracer(_node_op("matmul", [self.node, _tracer_node(other)]))
    def __getitem__(self, idx): return Tracer(_node_op("getitem", [self.node], {"index": idx}))

    def relu(self): return Tracer(_node_op("relu", [self.node]))
    def sigmoid(self): return Tracer(_node_op("sigmoid", [self.node]))
    def tanh(self): return Tracer(_node_op("tanh", [self.node]))
    def gelu(self): return Tracer(_node_op("gelu", [self.node]))
    def silu(self): return Tracer(_node_op("silu", [self.node]))
    def exp(self): return Tracer(_node_op("exp", [self.node]))
    def log(self): return Tracer(_node_op("log", [self.node]))
    def clip(self, lo, hi): return Tracer(_node_op("clip", [self.node], {"min": float(lo), "max": float(hi)}))
    def transpose(self, axes=None): return Tracer(_node_op("transpose", [self.node], {"axes": axes}))
    def sum(self, axis=None, keepdims=False): return Tracer(_node_op("reduce", [self.node], {"axis": axis, "keepdims": bool(keepdims)}))
    def mean(self, axis=None, keepdims=False): return Tracer(_node_op("mean", [self.node], {"axis": axis, "keepdims": bool(keepdims)}))
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tracer(_node_op("reshape", [self.node], {"shape": tuple(shape)}))

    def __repr__(self):
        return f"Tracer({self.node!r})"


def _node_input(name: str):
    from .graph_compiler import GraphNode
    return GraphNode("input", name=name)


def _node_op(op: str, inputs, params=None):
    from .graph_compiler import GraphNode
    return GraphNode(op, inputs, params=params)


def _tracer_node(other) -> Any:
    if isinstance(other, Tracer):
        return other.node
    return _node_op("const", [], {"value": np.asarray(other, dtype=np.float32)})


# --------------------------------------------------------------------------
# DAG helpers
# --------------------------------------------------------------------------

def _all_nodes(entry) -> List[Any]:
    nodes: List[Any] = []
    visited = set()

    def visit(n):
        if id(n) in visited:
            return
        visited.add(id(n))
        for i in n.inputs:
            visit(i)
        nodes.append(n)

    visit(entry)
    return nodes


def _topo_inputs(nodes: List[Any]) -> List[Any]:
    return [n for n in nodes if n.op == "input"]


def _build_children_map(nodes: List[Any]) -> Dict[int, List[Any]]:
    children: Dict[int, List[Any]] = {}
    for n in nodes:
        children.setdefault(id(n), [])
        for inp in n.inputs:
            children.setdefault(id(inp), []).append(n)
    return children


def _eval_nodes(nodes: List[Any], entry, bindings: Dict[int, np.ndarray]) -> np.ndarray:
    values: Dict[int, np.ndarray] = dict(bindings)
    for n in nodes:
        values[id(n)] = _apply(n, values)
    return values[id(entry)]


# --------------------------------------------------------------------------
# Reverse-mode symbolic differentiation over the DAG
# --------------------------------------------------------------------------

def _vjp(node, g: Any) -> List[Any]:
    """Return per-input gradient GraphNodes for one node."""
    op = node.op
    a = node.inputs[0] if len(node.inputs) > 0 else None
    b = node.inputs[1] if len(node.inputs) > 1 else None

    def to_shape(ig, operand):
        return _node_op("sum_to_shape", [ig, operand])

    if op == "add":
        return [to_shape(g, a), to_shape(g, b)]
    if op == "sub":
        return [to_shape(g, a), to_shape(_node_op("neg", [g]), b)]
    if op == "mul":
        return [to_shape(_node_op("mul", [g, b]), a), to_shape(_node_op("mul", [g, a]), b)]
    if op == "div":
        inv_b = _node_op("div", [_const(1.0), b])
        ga = to_shape(_node_op("mul", [g, inv_b]), a)
        neg_num = _node_op("mul", [_node_op("neg", [g]), a])
        gb = to_shape(_node_op("div", [neg_num, _node_op("mul", [b, b])]), b)
        return [ga, gb]
    if op == "neg":
        return [_node_op("neg", [g])]
    if op == "abs":
        return [_node_op("mul", [g, _node_op("sign", [a])])]
    if op == "exp":
        return [_node_op("mul", [g, node])]
    if op == "log":
        return [_node_op("div", [g, a])]
    if op == "relu":
        return [_node_op("mul", [g, _node_op("step", [a])])]
    if op == "sigmoid":
        one = _node_op("sub", [_const(1.0), node])
        return [_node_op("mul", [g, _node_op("mul", [node, one])])]
    if op == "tanh":
        sq = _node_op("mul", [node, node])
        return [_node_op("mul", [g, _node_op("sub", [_const(1.0), sq])])]
    if op == "gelu":
        s = a / _const(math.sqrt(2.0))
        cdf = _node_op("mul", [_const(0.5), _node_op("add", [_const(1.0), _node_op("erf", [s])])])
        pdf = _node_op("div", [_node_op("exp", [_node_op("neg", [_node_op("mul", [_node_op("mul", [s, s]), _const(0.5)])])]), _const(math.sqrt(2.0 * math.pi))])
        return [_node_op("mul", [g, _node_op("add", [cdf, pdf])])]
    if op == "silu":
        sig = _node_op("sigmoid", [a])
        one_m = _node_op("sub", [_const(1.0), sig])
        d = _node_op("add", [sig, _node_op("mul", [a, _node_op("mul", [sig, one_m])])])
        return [_node_op("mul", [g, d])]
    if op == "step" or op == "sign":
        return [_const(0.0)]
    if op == "erf":
        two_over_sqrt_pi = _const(2.0 / math.sqrt(math.pi))
        exp_arg = _node_op("neg", [_node_op("mul", [a, a])])
        return [_node_op("mul", [g, _node_op("mul", [two_over_sqrt_pi, _node_op("exp", [exp_arg])])])]
    if op == "clip":
        inside = _node_op("step", [a - _const(node.params["min"])]) * _node_op("step", [_const(node.params["max"]) - a])
        return [_node_op("mul", [g, inside])]
    if op == "transpose":
        return [_node_op("transpose_inv", [g], {"axes": node.params.get("axes")})]
    if op == "reshape":
        return [_node_op("reshape_grad", [g, a])]
    if op == "getitem":
        return [_node_op("scatter_like", [g, a], {"index": node.params["index"]})]
    if op == "reduce":
        return [_node_op("reduce_grad", [g, a], {"axis": node.params.get("axis"), "keepdims": node.params.get("keepdims")})]
    if op == "mean":
        return [_node_op("mean_grad", [g, a], {"axis": node.params.get("axis"), "keepdims": node.params.get("keepdims")})]
    if op == "sum_to_shape":
        return [_node_op("broadcast_to_shape", [g, node.inputs[0]]), _const(0.0)]
    if op == "reduce_grad":
        return [_node_op("reduce", [g], {"axis": node.params.get("axis"), "keepdims": node.params.get("keepdims")}), _const(0.0)]
    if op == "mean_grad":
        return [_node_op("mean_adjoint", [g, a], {"axis": node.params.get("axis"), "keepdims": node.params.get("keepdims")}), _const(0.0)]
    if op == "reshape_grad":
        return [_node_op("reshape_grad", [g, a]), _const(0.0)]
    if op == "scatter_like":
        return [_node_op("getitem", [g], {"index": node.params["index"]}), _const(0.0)]
    if op == "matmul":
        ga = _node_op("matmul_grad_a", [g, b])
        gb = _node_op("matmul_grad_b", [a, g])
        return [ga, gb]
    raise ValueError(f"no gradient rule for op {op!r}")


def _const(value):
    return _node_op("const", [], {"value": np.asarray(value, dtype=np.float32)})


def _reverse_grads(nodes: List[Any], entry, wrt_ids: List[int],
                   seed: Any) -> Dict[int, Any]:
    children = _build_children_map(nodes)
    grads: Dict[int, Any] = {id(entry): seed}

    for n in reversed(nodes):
        g = grads.get(id(n))
        if g is None:
            continue
        if n.op in ("input", "const", "param"):
            continue
        local = _vjp(n, g)
        for inp, ig in zip(n.inputs, local):
            if inp.op in ("const", "param"):
                continue
            acc = grads.get(id(inp))
            grads[id(inp)] = ig if acc is None else _node_op("add", [acc, ig])

    out = {}
    for wid in wrt_ids:
        if wid in grads:
            out[wid] = grads[wid]
    return out


# --------------------------------------------------------------------------
# Trace — one differentiated DAG
# --------------------------------------------------------------------------

class Trace:
    """A traced DAG: ``nodes`` + ``entry`` + ordered input leaves."""

    def __init__(self, nodes: List[Any], entry: Any,
                 inputs: Optional[List[Any]] = None):
        self.nodes = nodes
        self.entry = entry
        self.inputs = inputs if inputs is not None else _topo_inputs(nodes)

    def execute(self, *args) -> Any:
        if len(args) != len(self.inputs):
            raise ValueError(
                f"expected {len(self.inputs)} inputs, got {len(args)}")
        bind = {id(n): np.asarray(a) for n, a in zip(self.inputs, args)}
        return _eval_nodes(self.nodes, self.entry, bind)

    def __call__(self, *args) -> Any:
        return self.execute(*args)

    def differentiate(self, argnums) -> "Trace":
        argnum_list = _argnum_list(argnums, len(self.inputs))
        wrt_ids = [id(self.inputs[i]) for i in argnum_list]
        grad_map = _reverse_grads(self.nodes, self.entry, wrt_ids, _const(1.0))
        grad_nodes = [grad_map[w] for w in wrt_ids]
        if len(grad_nodes) == 1:
            entry = grad_nodes[0]
        else:
            entry = _node_op("tuple", grad_nodes)
        nodes = _all_nodes(entry)
        inputs = _topo_inputs(nodes)
        if len(inputs) == len(self.inputs):
            inputs = self.inputs
        return Trace(nodes, entry, inputs)


def _argnum_list(argnums, n_inputs) -> List[int]:
    argnum_list = argnums if isinstance(argnums, (list, tuple)) else [argnums]
    for a in argnum_list:
        if not isinstance(a, int) or a < 0 or a >= n_inputs:
            raise ValueError(f"invalid argnum {a} for {n_inputs} inputs")
    return list(argnum_list)


# --------------------------------------------------------------------------
# public transforms
# --------------------------------------------------------------------------

def trace_function(fn: Callable, *args) -> Trace:
    """Trace ``fn`` over symbolic inputs matching ``args`` shapes."""
    tracers = [Tracer(name=f"arg_{i}") for i in range(len(args))]
    out = fn(*tracers)
    if isinstance(out, Tracer):
        entry = out.node
    else:
        entry = _node_op("const", [], {"value": np.asarray(out, dtype=np.float32)})
    return Trace(_all_nodes(entry), entry)


def jit(fn: Callable) -> Callable:
    """Trace ``fn`` once per input shape and execute the graph on arrays."""
    cache: Dict[tuple, Trace] = {}

    def wrapper(*args):
        key = tuple((a.shape, str(np.asarray(a).dtype)) for a in args)
        trace = cache.get(key)
        if trace is None:
            trace = trace_function(fn, *args)
            cache[key] = trace
        return trace.execute(*args)

    wrapper.__name__ = getattr(fn, "__name__", "jit_fn")
    return wrapper


def grad(fn: Callable, argnums: Any = 0) -> Callable:
    """Reverse-mode gradient of ``fn`` (JAX-style, scalar output required)."""
    cache: Dict[tuple, Trace] = {}

    def wrapper(*args):
        key = tuple((a.shape, str(np.asarray(a).dtype)) for a in args)
        gtrace = cache.get(key)
        if gtrace is None:
            argnum_list = _argnum_list(argnums, len(args))
            gtrace = trace_function(fn, *args).differentiate(argnum_list)
            cache[key] = gtrace
        return gtrace.execute(*args)

    wrapper.__name__ = f"grad_{getattr(fn, '__name__', 'fn')}"
    return wrapper


def value_and_grad(fn: Callable, argnums: Any = 0) -> Callable:
    """Return ``(value, grad)`` for scalar-output ``fn``."""
    cache: Dict[tuple, Any] = {}

    def wrapper(*args):
        key = tuple((a.shape, str(np.asarray(a).dtype)) for a in args)
        entry = cache.get(key)
        if entry is None:
            argnum_list = _argnum_list(argnums, len(args))
            trace = trace_function(fn, *args)
            gtrace = trace.differentiate(argnum_list)
            entry = (trace, gtrace)
            cache[key] = entry
        trace, gtrace = entry
        return trace.execute(*args), gtrace.execute(*args)

    wrapper.__name__ = f"value_and_grad_{getattr(fn, '__name__', 'fn')}"
    return wrapper


def jacobian(fn: Callable, argnums: Any = 0) -> Callable:
    """Full Jacobian of ``fn`` w.r.t. ``argnums`` (any output shape)."""
    cache: Dict[tuple, Any] = {}

    def wrapper(*args):
        key = tuple((a.shape, str(np.asarray(a).dtype)) for a in args)
        entry = cache.get(key)
        if entry is None:
            trace = trace_function(fn, *args)
            y = trace.execute(*args)
            argnum_list = _argnum_list(argnums, len(args))
            wrt_ids = [id(trace.inputs[i]) for i in argnum_list]
            y_flat = np.asarray(y).reshape(-1)
            n_out = y_flat.size
            rows = []
            for i in range(n_out):
                seed = np.zeros_like(np.asarray(y), dtype=np.float32)
                seed.reshape(-1)[i] = 1.0
                grad_map = _reverse_grads(trace.nodes, trace.entry, wrt_ids,
                                          _const(seed))
                bind = {id(n): np.asarray(a) for n, a in zip(trace.inputs, args)}
                grads = [
                    _eval_nodes(_all_nodes(grad_map[w]), grad_map[w], bind)
                    for w in wrt_ids
                ]
                rows.append(np.concatenate([np.asarray(g).reshape(-1) for g in grads]))
            J = np.stack(rows)
            entry = J, tuple(argnum_list), trace
            cache[key] = entry
        J, argnum_list, trace = entry
        out_shape = trace.execute(*args).shape
        in_shapes = [tuple(np.asarray(args[i]).shape) for i in argnum_list]
        total_in = sum(int(np.prod(s)) for s in in_shapes)
        J = J[:, :total_in]
        if len(argnum_list) == 1:
            return J.reshape(out_shape + in_shapes[0])
        return [J[:, start:start + int(np.prod(s))].reshape(out_shape + s)
                for s, start in zip(in_shapes, np.cumsum([0] + [int(np.prod(s)) for s in in_shapes[:-1]]))]

    wrapper.__name__ = f"jacobian_{getattr(fn, '__name__', 'fn')}"
    return wrapper


def hessian(fn: Callable, argnums: Any = 0) -> Callable:
    """Hessian of scalar-output ``fn`` (jacobian of the gradient DAG)."""
    cache: Dict[tuple, Any] = {}

    def wrapper(*args):
        key = tuple((a.shape, str(np.asarray(a).dtype)) for a in args)
        entry = cache.get(key)
        if entry is None:
            argnum_list = _argnum_list(argnums, len(args))
            g_trace = trace_function(fn, *args).differentiate(argnum_list)
            wrt_ids = [id(g_trace.inputs[i]) for i in argnum_list]
            g = np.asarray(g_trace.execute(*args))
            g_flat = g.reshape(-1)
            n_out = g_flat.size
            bind = {id(n): np.asarray(a) for n, a in zip(g_trace.inputs, args)}
            rows = []
            for i in range(n_out):
                seed = np.zeros_like(g, dtype=np.float32)
                seed.reshape(-1)[i] = 1.0
                grad_map = _reverse_grads(g_trace.nodes, g_trace.entry, wrt_ids, _const(seed))
                grads = [
                    _eval_nodes(_all_nodes(grad_map[w]), grad_map[w], bind)
                    for w in wrt_ids
                ]
                rows.append(np.concatenate([np.asarray(x).reshape(-1) for x in grads]))
            H = np.stack(rows)
            entry = (H, tuple(argnum_list), g_trace)
            cache[key] = entry
        H, argnum_list, g_trace = entry
        g_shape = np.asarray(g_trace.execute(*args)).shape
        in_shapes = [tuple(np.asarray(args[i]).shape) for i in argnum_list]
        total_in = sum(int(np.prod(s)) for s in in_shapes)
        H = H[:, :total_in]
        if len(argnum_list) == 1:
            return H.reshape(g_shape + in_shapes[0])
        return [H[:, start:start + int(np.prod(s))].reshape(g_shape + s)
                for s, start in zip(in_shapes, np.cumsum([0] + [int(np.prod(s)) for s in in_shapes[:-1]]))]

    wrapper.__name__ = f"hessian_{getattr(fn, '__name__', 'fn')}"
    return wrapper


def vmap(fn: Callable, in_axes: Any = 0) -> Callable:
    """Vectorizing transform: map the traced graph over a leading batch axis."""
    cache: Dict[tuple, Trace] = {}

    def wrapper(*args):
        single = []
        batch_info = []
        for i, a in enumerate(args):
            axis = in_axes[i] if isinstance(in_axes, (list, tuple)) else in_axes
            arr = np.asarray(a)
            if axis == 0:
                batch_info.append(i)
                single.append(arr[0])
            else:
                single.append(arr)
        key = tuple((a.shape, str(np.asarray(a).dtype)) for a in single)
        trace = cache.get(key)
        if trace is None:
            trace = trace_function(fn, *single)
            cache[key] = trace
        if not batch_info:
            return trace.execute(*args)
        bsize = int(np.asarray(args[batch_info[0]]).shape[0])
        out = [
            trace.execute(*[np.asarray(args[j])[k] if j in batch_info else np.asarray(args[j])
                            for j in range(len(args))])
            for k in range(bsize)
        ]
        return np.stack(out)

    wrapper.__name__ = f"vmap_{getattr(fn, '__name__', 'fn')}"
    return wrapper
