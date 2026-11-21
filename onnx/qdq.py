"""ONNX QDQ (QuantizeLinear/DequantizeLinear) support (numpy-only).

Inserts QDQ node pairs around float initializers so downstream ops keep
consuming float tensors while the graph is QDQ-marked for INT8 fusion by
runtimes. Includes a pure-numpy reference for the QDQ pair.
"""

from typing import Dict, List, Optional

import numpy as np

from .model import Graph, Model, Node, Tensor

__all__ = ["Quantizer", "quantize_model", "qdq_round_trip", "symmetric_scale"]


def symmetric_scale(data: np.ndarray, num_bits: int = 8) -> float:
    amax = float(np.max(np.abs(data))) if data.size else 1.0
    if amax == 0.0:
        return 1.0
    qmax = float(2 ** (num_bits - 1) - 1)
    return amax / qmax


def asymmetric_scale_zp(data: np.ndarray, num_bits: int = 8):
    dtype = np.int8 if num_bits <= 8 else np.int16
    dmin = float(np.min(data)) if data.size else 0.0
    dmax = float(np.max(data)) if data.size else 0.0
    if dmin == dmax:
        return 1.0, np.array(0, dtype=dtype)
    qmin = float(np.iinfo(dtype).min)
    qmax = float(np.iinfo(dtype).max)
    scale = (dmax - dmin) / (qmax - qmin)
    zp = qmin - dmin / scale
    zp = np.clip(np.round(zp), qmin, qmax).astype(dtype)
    return scale, zp


def qdq_round_trip(data, scale, zero_point, num_bits=8):
    """Reference (symmetric) quantize + dequantize for validation."""
    dtype = np.int8 if num_bits <= 8 else np.int16
    qmax = float(2 ** (num_bits - 1) - 1)
    q = np.clip(np.round(data / scale), -qmax, qmax)
    if zero_point is not None and np.any(np.asarray(zero_point)):
        q = q - zero_point
    q = q.astype(dtype)
    return (q.astype(np.float32) + zero_point) * scale


def _insert_qdq(
    graph: Graph,
    init: Tensor,
    counter: Dict[str, int],
    num_bits: int,
    per_channel: bool,
) -> List[Tensor]:
    """Insert a Q->DQ pair around ``init`` and rewire consumers to ``dq_out``.

    Returns the scale + zero-point initializers to append to the graph.
    """
    name = init.name
    if per_channel:
        c = init.data.shape[0] if init.data.ndim >= 2 else 1
        scales = np.array(
            [symmetric_scale(init.data[i], num_bits) for i in range(c)],
            dtype=np.float32,
        )
        zero_point = np.zeros(c, dtype=np.int8 if num_bits <= 8 else np.int16)
    else:
        scales = np.array(symmetric_scale(init.data, num_bits), dtype=np.float32)
        zero_point = np.zeros((), dtype=np.int8 if num_bits <= 8 else np.int16)

    def nxt(prefix):
        counter[prefix] = counter.get(prefix, 0) + 1
        return f"{prefix}_{counter[prefix]}"

    scale_name = nxt("q_scale")
    zp_name = nxt("q_zero")
    q_out = nxt("q_out")
    dq_out = nxt("dq_out")

    zp_dtype = "int8" if num_bits <= 8 else "int16"
    scale_t = Tensor(scale_name, "float32", list(scales.shape), scales)
    zp_t = Tensor(zp_name, zp_dtype, list(zero_point.shape), zero_point)

    q = Node("QuantizeLinear", [name, scale_name, zp_name], [q_out], name=q_out)
    dq = Node("DequantizeLinear", [q_out, scale_name, zp_name], [dq_out], name=dq_out)
    graph.nodes.insert(0, dq)
    graph.nodes.insert(0, q)

    # rewrite consumers of `name` to consume dq_out (skip the Q/DQ pair)
    for node in graph.nodes:
        if node is q or node is dq:
            continue
        node.inputs = [dq_out if i == name else i for i in node.inputs]
    for out in graph.outputs:
        if out.name == name:
            out.name = dq_out

    return [scale_t, zp_t]


class Quantizer:
    """Insert QDQ pairs around float initializers."""

    def __init__(self, num_bits: int = 8, per_channel: bool = False,
                 skip_patterns: Optional[List[str]] = None):
        self.num_bits = num_bits
        self.per_channel = per_channel
        self.skip_patterns = skip_patterns or ["bias"]

    def quantize(self, model: Model) -> Model:
        graph = model.graph
        counter: Dict[str, int] = {}
        kept = []
        new_inits: List[Tensor] = []
        for init in list(graph.initializers):
            if (
                init.data is not None
                and init.dtype in ("float32", "float16")
                and not any(p in init.name for p in self.skip_patterns)
            ):
                kept.append(init)
                new_inits.extend(
                    _insert_qdq(graph, init, counter, self.num_bits, self.per_channel)
                )
            else:
                kept.append(init)
        graph.initializers = kept + new_inits
        return model


def quantize_model(
    model: Model,
    num_bits: int = 8,
    per_channel: bool = False,
    skip_patterns: Optional[List[str]] = None,
) -> Model:
    """Convenience wrapper: quantize a model in place and return it."""
    return Quantizer(num_bits=num_bits, per_channel=per_channel,
                     skip_patterns=skip_patterns).quantize(model)
