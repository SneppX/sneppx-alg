"""Graph compiler — element-wise op fusion, memory tiling, and C codegen.

Builds a compute DAG of ``GraphNode`` ops, fuses maximal chains of
element-wise ops into single fused kernels (avoiding intermediate
allocations), optionally tiles large element-wise kernels, and generates C
source for the fused kernels.

Typical usage::

    a = GraphNode.input_("a")
    b = GraphNode.input_("b")
    c = (a * b) + b
    out = c.relu()
    compiler = GraphCompiler()
    compiled = compiler.compile(out)
    y = compiled.forward({"a": ..., "b": ...})
    code = compiler.generate_c(compiled)
"""

import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "GraphNode",
    "GraphCompiler",
    "CompiledGraph",
    "FusedNode",
    "FUSABLE_OPS",
    "ELEMENTWISE_IMPL",
]

FUSABLE_OPS = {
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "abs",
    "exp",
    "log",
    "relu",
    "sigmoid",
    "tanh",
    "gelu",
    "silu",
    "clip",
}


def _relu(x):
    return np.maximum(x, 0.0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _gelu(x):
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


ELEMENTWISE_IMPL: Dict[str, Callable] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "neg": lambda x: -x,
    "abs": lambda x: np.abs(x),
    "exp": lambda x: np.exp(x),
    "log": lambda x: np.log(x),
    "relu": _relu,
    "sigmoid": _sigmoid,
    "tanh": lambda x: np.tanh(x),
    "gelu": _gelu,
    "silu": _silu,
    "clip": lambda x, lo, hi: np.clip(x, lo, hi),
}


class GraphNode:
    """A node in a compute DAG."""

    def __init__(
        self,
        op: str,
        inputs: Sequence["GraphNode"] = (),
        params: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ):
        self.op = op
        self.inputs = list(inputs)
        self.params = dict(params or {})
        self.name = name or f"{op}_{id(self) % 100000}"

    # ---- factory / fluent helpers ----------------------------------------
    @staticmethod
    def input_(name: str = "input") -> "GraphNode":
        return GraphNode("input", name=name)

    @staticmethod
    def constant(value: Any, name: Optional[str] = None) -> "GraphNode":
        return GraphNode("const", params={"value": np.asarray(value)}, name=name)

    @staticmethod
    def parameter(value: Any, name: Optional[str] = None) -> "GraphNode":
        return GraphNode("param", params={"value": np.asarray(value)}, name=name)

    @staticmethod
    def clip(x: "GraphNode", lo: float, hi: float) -> "GraphNode":
        return GraphNode("clip", [x], params={"min": float(lo), "max": float(hi)})

    # ---- element-wise sugar ----------------------------------------------
    def __add__(self, other):
        return GraphNode("add", [self, _node(other)])

    def __sub__(self, other):
        return GraphNode("sub", [self, _node(other)])

    def __mul__(self, other):
        return GraphNode("mul", [self, _node(other)])

    def __truediv__(self, other):
        return GraphNode("div", [self, _node(other)])

    def relu(self) -> "GraphNode":
        return GraphNode("relu", [self])

    def sigmoid(self) -> "GraphNode":
        return GraphNode("sigmoid", [self])

    def tanh(self) -> "GraphNode":
        return GraphNode("tanh", [self])

    def gelu(self) -> "GraphNode":
        return GraphNode("gelu", [self])

    def silu(self) -> "GraphNode":
        return GraphNode("silu", [self])

    def matmul(self, other: "GraphNode") -> "GraphNode":
        return GraphNode("matmul", [self, _node(other)])

    # ---- evaluation ------------------------------------------------------
    def evaluate(self, bindings: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        bindings = bindings or {}
        if self.op == "input":
            return np.asarray(bindings.get(self.name, bindings.get(id(self))))
        if self.op == "const":
            return self.params["value"]
        if self.op == "param":
            return self.params["value"]
        if self.op == "matmul":
            return self.inputs[0].evaluate(bindings) @ self.inputs[1].evaluate(bindings)
        if self.op == "clip":
            return np.clip(
                self.inputs[0].evaluate(bindings),
                self.params["min"],
                self.params["max"],
            )
        fn = ELEMENTWISE_IMPL.get(self.op)
        if fn is None:
            raise ValueError(f"unknown op: {self.op}")
        vals = [i.evaluate(bindings) for i in self.inputs]
        if self.op == "clip":
            return fn(*vals, self.params["min"], self.params["max"])
        return fn(*vals)

    def __repr__(self) -> str:
        return f"GraphNode({self.op}, name={self.name!r})"


def _node(other) -> GraphNode:
    if isinstance(other, GraphNode):
        return other
    return GraphNode.constant(other)


def _external_inputs(
    members: List[GraphNode], by_id: Dict[int, GraphNode]
) -> List[GraphNode]:
    """Inputs to the cluster that come from outside it, in first-use order."""
    member_ids = {id(m) for m in members}
    external = []
    seen = set()
    for m in members:
        for inp in m.inputs:
            if id(inp) in member_ids or id(inp) in seen:
                continue
            seen.add(id(inp))
            external.append(inp)
    return external


def _make_fused_fn(members: List[GraphNode], external: List[GraphNode]) -> Callable:
    """Compose the member element-wise ops into a single callable."""
    ext_index = {id(e): i for i, e in enumerate(external)}

    def lookup(vals: Dict[int, np.ndarray], node: GraphNode) -> np.ndarray:
        if id(node) in ext_index:
            return vals[ext_index[id(node)]]
        return vals[id(node)]

    def fn(*args):
        vals: Dict[int, np.ndarray] = {}
        for i, a in enumerate(args):
            vals[i] = a
        for m in members:
            if m.op == "clip":
                x = lookup(vals, m.inputs[0])
                vals[id(m)] = np.clip(x, m.params["min"], m.params["max"])
                continue
            impl = ELEMENTWISE_IMPL[m.op]
            if len(m.inputs) == 1:
                vals[id(m)] = impl(lookup(vals, m.inputs[0]))
            else:
                a = lookup(vals, m.inputs[0])
                b = lookup(vals, m.inputs[1])
                vals[id(m)] = impl(a, b)
        return vals[id(members[-1])]

    return fn


class FusedNode:
    """A maximal element-wise cluster collapsed into one kernel."""

    def __init__(self, members: List[GraphNode], name: str = "fused"):
        self.op = "fused"
        self.name = name
        self.members = members
        self.external_inputs: List[GraphNode] = []
        self.fn: Optional[Callable] = None

    def __repr__(self) -> str:
        return f"FusedNode({self.name}, {len(self.members)} ops)"


class CompiledGraph:
    """Topologically-ordered executable graph of (possibly fused) nodes."""

    def __init__(
        self, nodes: List[Any], entry: Any, replaced: Optional[Dict[int, Any]] = None
    ):
        self.nodes = nodes
        self.entry = entry
        self.replaced = replaced or {}

    def _resolve(self, node: Any) -> Any:
        if isinstance(node, FusedNode):
            return node
        return self.replaced.get(id(node), node)

    def forward(
        self, inputs: Dict[str, np.ndarray], tile_size: Optional[int] = None
    ) -> np.ndarray:
        values: Dict[int, np.ndarray] = {}
        bindings = {k: np.asarray(v) for k, v in inputs.items()}

        for node in self.nodes:
            if node.op == "input":
                values[id(node)] = bindings.get(node.name, bindings.get(id(node)))
            elif node.op == "const":
                values[id(node)] = node.params["value"]
            elif node.op == "param":
                values[id(node)] = node.params["value"]
            elif node.op == "matmul":
                a = self._resolve(node.inputs[0])
                b = self._resolve(node.inputs[1])
                values[id(node)] = values[id(a)] @ values[id(b)]
            elif node.op == "fused":
                values[id(node)] = _eval_fused(node, values, tile_size)
            elif node.op == "clip":
                x = values[id(self._resolve(node.inputs[0]))]
                values[id(node)] = np.clip(x, node.params["min"], node.params["max"])
            else:
                fn = ELEMENTWISE_IMPL.get(node.op)
                if fn is None:
                    raise ValueError(f"unknown op: {node.op}")
                args = [values[id(self._resolve(i))] for i in node.inputs]
                values[id(node)] = fn(*args)

        return values[id(self.entry)]


def _eval_fused(
    node: FusedNode, values: Dict[int, np.ndarray], tile_size: Optional[int] = None
) -> np.ndarray:
    inputs = [values[id(i)] for i in node.external_inputs]
    if tile_size is None:
        return node.fn(*inputs)
    out_shape = np.broadcast_shapes(*[a.shape for a in inputs])
    out = np.empty(out_shape, dtype=inputs[0].dtype)
    n = out_shape[0]
    for start in range(0, n, tile_size):
        end = min(start + tile_size, n)
        tile_inputs = []
        for a in inputs:
            if a.ndim and a.shape[0] == out_shape[0]:
                tile_inputs.append(a[start:end])
            else:
                tile_inputs.append(a)
        out[start:end] = node.fn(*tile_inputs)
    return out


class GraphCompiler:
    """Fusion, execution planning, tiling and C codegen for compute DAGs."""

    def __init__(self, fusable_ops: Optional[set] = None):
        self.fusable_ops = set(fusable_ops) if fusable_ops else set(FUSABLE_OPS)

    # ---- graph utilities -------------------------------------------------
    def _all_nodes(self, entry: GraphNode) -> List[GraphNode]:
        nodes: List[GraphNode] = []
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

    def _is_fusable(self, node: GraphNode) -> bool:
        return node.op in self.fusable_ops

    # ---- fusion ----------------------------------------------------------
    def compile(self, entry: GraphNode) -> CompiledGraph:
        """Fuse, plan and return an executable CompiledGraph."""
        return self.fuse(entry)

    def fuse(self, entry: GraphNode) -> CompiledGraph:
        nodes = self._all_nodes(entry)
        by_id = {id(n): n for n in nodes}

        parent = {id(n): id(n) for n in nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for node in nodes:
            if not self._is_fusable(node):
                continue
            for inp in node.inputs:
                if inp.op != "input" and self._is_fusable(inp):
                    union(id(node), id(inp))

        clusters: Dict[int, List[GraphNode]] = {}
        for n in nodes:
            if self._is_fusable(n):
                clusters.setdefault(find(id(n)), []).append(n)

        # Build a FusedNode per cluster and a replacement map.
        fused_by_cluster: Dict[int, FusedNode] = {}
        replaced: Dict[int, FusedNode] = {}
        for root, members in clusters.items():
            topo = sorted(members, key=lambda m: nodes.index(m))
            fused = FusedNode(topo, name=f"fused_{root % 100000}")
            fused.external_inputs = _external_inputs(members, by_id)
            fused.fn = _make_fused_fn(members, fused.external_inputs)
            fused_by_cluster[root] = fused
            for m in members:
                replaced[id(m)] = fused

        # New graph nodes: non-fusable nodes plus any fused node still consumed.
        new_nodes: List[Any] = []
        consumed = set()
        for n in nodes:
            if self._is_fusable(n):
                continue
            new_nodes.append(n)
            for inp in n.inputs:
                if id(inp) in replaced:
                    consumed.add(id(replaced[id(inp)]))
        for fused in fused_by_cluster.values():
            if id(fused) in consumed:
                new_nodes.append(fused)

        entry_out = replaced.get(id(entry), entry)
        if entry_out not in new_nodes:
            new_nodes.append(entry_out)

        # Topological order over the rewritten graph.
        def effective_inputs(n: Any) -> List[Any]:
            if isinstance(n, FusedNode):
                return list(n.external_inputs)
            outs = []
            for inp in n.inputs:
                mapped = replaced.get(id(inp), inp)
                if all(mapped is not o for o in outs):
                    outs.append(mapped)
            return outs

        order: List[Any] = []
        visited = set()

        def visit(n: Any):
            if id(n) in visited:
                return
            visited.add(id(n))
            for i in effective_inputs(n):
                visit(i)
            order.append(n)

        for n in new_nodes:
            visit(n)

        return CompiledGraph(order, entry_out, replaced=dict(replaced))

    # ---- C codegen -------------------------------------------------------
    def generate_c(self, compiled: CompiledGraph) -> str:
        """Generate C source for the compiled graph's fused kernels.

        Emits one ``static`` kernel per FusedNode (flat ``n``-element loop),
        a row-major ``sneppx_matmul_f32`` helper when the graph contains a
        matmul, and a ``sneppx_graph_forward`` driver that wires inputs and
        scratch buffers in topological order.
        """
        lines = [
            "/* Generated by SneppX GraphCompiler. */",
            "#include <math.h>",
            "#include <stddef.h>",
            "",
            "#define SNEPPX_MAX(a, b) ((a) > (b) ? (a) : (b))",
            "#define SNEPPX_MIN(a, b) ((a) < (b) ? (a) : (b))",
            "",
            "static float sneppx_sigmoid_f32(float x) { return 1.0f / (1.0f + expf(-x)); }",
            "static float sneppx_gelu_f32(float x) { return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f)); }",
            "static float sneppx_silu_f32(float x) { return x / (1.0f + expf(-x)); }",
            "",
        ]
        has_matmul = any(n.op == "matmul" for n in compiled.nodes)
        for node in compiled.nodes:
            if isinstance(node, FusedNode):
                lines.extend(self._gen_fused_kernel(node))
        if has_matmul:
            lines.extend(self._gen_matmul_helper())
        lines.extend(self._gen_driver(compiled, has_matmul))
        return "\n".join(lines)

    def _gen_fused_kernel(self, node: FusedNode) -> List[str]:
        name, params = _fused_kernel_spec(node)
        base: Dict[int, str] = {}
        for i, ident in params:
            base[i] = f"{ident}[i]"
        for i, e in enumerate(node.external_inputs):
            if e.op in ("const", "param") and np.asarray(e.params["value"]).size == 1:
                base[i] = _c_literal(
                    float(np.asarray(e.params["value"]).reshape(-1)[0])
                )

        temps: Dict[int, str] = {}
        exprs: List[str] = []

        def src(n: GraphNode) -> str:
            if id(n) in temps:
                return temps[id(n)]
            for i, e in enumerate(node.external_inputs):
                if e is n:
                    return base[i]
            raise ValueError(f"fused member input {n!r} is neither member nor external")

        for j, m in enumerate(node.members):
            if m.op == "clip":
                x = src(m.inputs[0])
                expr = ("SNEPPX_MIN(SNEPPX_MAX({x}, {lo}), {hi})").format(
                    x=x,
                    lo=_c_literal(m.params["min"]),
                    hi=_c_literal(m.params["max"]),
                )
            elif len(m.inputs) == 1:
                expr = _C_UNARY[m.op].format(x=src(m.inputs[0]))
            else:
                expr = _C_BINARY[m.op].format(a=src(m.inputs[0]), b=src(m.inputs[1]))
            temps[id(m)] = f"t{j}"
            exprs.append(expr)

        sig = ", ".join(
            [f"const float* {ident}" for _, ident in params if ident]
            + ["float* out", "size_t n"]
        )
        ops = ",".join(m.op for m in node.members)
        lines = [
            f"/* ---- fused kernel: {ops} ---- */",
            f"static void {name}({sig}) {{",
            "    for (size_t i = 0; i < n; ++i) {",
        ]
        for j, expr in enumerate(exprs):
            if j == len(exprs) - 1:
                lines.append(f"        out[i] = {expr};")
            else:
                lines.append(f"        float t{j} = {expr};")
        lines += ["    }", "}", ""]
        return lines

    @staticmethod
    def _gen_matmul_helper() -> List[str]:
        return [
            "/* ---- matmul helper (row-major) ---- */",
            "static void sneppx_matmul_f32(const float* A, const float* B, float* C,",
            "                              size_t M, size_t K, size_t N) {",
            "    for (size_t m = 0; m < M; ++m) {",
            "        for (size_t nn = 0; nn < N; ++nn) {",
            "            float acc = 0.0f;",
            "            for (size_t k = 0; k < K; ++k) {",
            "                acc += A[m * K + k] * B[k * N + nn];",
            "            }",
            "            C[m * N + nn] = acc;",
            "        }",
            "    }",
            "}",
            "",
        ]

    def _gen_driver(self, compiled: CompiledGraph, has_matmul: bool) -> List[str]:
        input_index = {
            id(n): k for k, n in enumerate(compiled.nodes) if n.op == "input"
        }
        non_inputs = [n for n in compiled.nodes if n.op != "input"]
        slot = {id(n): k for k, n in enumerate(non_inputs)}
        entry_id = id(compiled.entry)
        entry_is_input = compiled.entry.op == "input"

        def ref(n: Any) -> str:
            if n.op == "input":
                return f"inputs[{input_index[id(n)]}]"
            if id(n) == entry_id:
                return "out"
            return f"scratch[{slot[id(n)]}]"

        extra = ", size_t m, size_t k, size_t nn" if has_matmul else ""
        lines = [
            "/* ---- forward driver ---- */",
            "static void sneppx_graph_forward(",
            "    const float* const inputs[], float* scratch[], float* out,",
            f"    size_t n{extra}) {{",
        ]
        for node in compiled.nodes:
            if node.op == "input":
                continue
            if isinstance(node, FusedNode):
                name, params = _fused_kernel_spec(node)
                args = [ref(node.external_inputs[i]) for i, ident in params if ident]
                args += [ref(node), "n"]
                lines.append(f"    {name}({', '.join(args)});")
            elif node.op == "matmul":
                a = ref(compiled._resolve(node.inputs[0]))
                b = ref(compiled._resolve(node.inputs[1]))
                lines.append(f"    sneppx_matmul_f32({a}, {b}, {ref(node)}, m, k, nn);")
            elif node.op in ("const", "param"):
                continue
            elif node.op == "clip":
                _a = ref(compiled._resolve(node.inputs[0]))
                _o = ref(node)
                _lo = node.params["min"]
                _hi = node.params["max"]
                lines.append("    for (size_t i = 0; i < n; ++i) {")
                lines.append("        float _v = %s[i];" % _a)
                lines.append("        if (_v < %sf) _v = %sf;" % (_lo, _lo))
                lines.append("        if (_v > %sf) _v = %sf;" % (_hi, _hi))
                lines.append("        %s[i] = _v;" % _o)
                lines.append("    }")
            else:
                raise NotImplementedError(f"codegen for op {node.op!r}")
        if entry_is_input:
            lines.append("    /* graph output is a passthrough input */")
            lines.append(
                "    for (size_t i = 0; i < n; ++i) { out[i] = inputs["
                + str(input_index[entry_id])
                + "][i]; }"
            )
        lines.append("}")
        lines.append("")
        return lines


_C_UNARY = {
    "neg": "(-({x}))",
    "abs": "fabsf({x})",
    "exp": "expf({x})",
    "log": "logf({x})",
    "relu": "(({x}) > 0.0f ? ({x}) : 0.0f)",
    "sigmoid": "sneppx_sigmoid_f32({x})",
    "tanh": "tanhf({x})",
    "gelu": "sneppx_gelu_f32({x})",
    "silu": "sneppx_silu_f32({x})",
}

_C_BINARY = {
    "add": "(({a}) + ({b}))",
    "sub": "(({a}) - ({b}))",
    "mul": "(({a}) * ({b}))",
    "div": "(({a}) / ({b}))",
}


def _c_ident(name: str, index: int) -> str:
    ident = re.sub(r"[^A-Za-z0-9_]", "_", name or "")
    if not ident or ident[0].isdigit():
        ident = f"in_{index}"
    return ident


def _c_literal(value: float) -> str:
    return f"{value:.9g}f"


def _fused_kernel_spec(node: FusedNode) -> Tuple[str, List[Tuple[int, Optional[str]]]]:
    """Return (kernel_name, [(external_index, param_ident_or_None), ...]).

    ``None`` marks a scalar const/param external input that is inlined as a
    literal instead of being passed as a runtime array parameter.
    """
    name = _c_ident(node.name, 0)
    params: List[Tuple[int, Optional[str]]] = []
    for i, e in enumerate(node.external_inputs):
        if e.op in ("const", "param"):
            val = np.asarray(e.params["value"])
            if val.size == 1:
                params.append((i, None))
                continue
        params.append((i, _c_ident(e.name, i)))
    return name, params
