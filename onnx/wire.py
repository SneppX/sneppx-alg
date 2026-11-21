"""Canonical ONNX protobuf wire-format primitives (pure Python, numpy-only).

Implements the length-delimited protobuf encoding/decoding that the ONNX IR
uses for ModelProto/GraphProto/NodeProto/TensorProto/AttrProto. Field numbers
are canonical (onnx/onnx.in.proto) and interoperable with the C exporter in
``fs/format/onnx_format.c`` (SneppX_onnx_save_graph / SneppX_onnx_validate)
and with standard runtimes.

Wire layout summary (documented once here, mirrored by parser/serializer):

  ModelProto:   ir_version=1, producer_name=2, producer_version=3,
                model_version=5, domain=4, doc_string=6, graph=7,
                opset_import=8, metadata_props=14
  GraphProto:   node=1, name=2, initializer=5, sparse_initializer=15,
                doc_string=10, input=11, output=12, value_info=13
  NodeProto:    input=1, output=2, name=3, op_type=4, attribute=5,
                domain=7, doc_string=6
  AttrProto:    name=1, f=2, i=3, s=4, t=5, g=6, floats=7, ints=8,
                strings=9, tensors=10, graphs=11, sparse_tensor=22,
                tp=14, type=20 (AttributeType)
  ValueInfo:    name=1, type=2, doc_string=3
  TypeProto:    tensor_type=1, sequence_type=4, map_type=5
  Tensor:       elem_type=1, shape=2, denotation=6
  ShapeProto:   dim=1
  Dimension:    dim_value=1, dim_param=2, denotation=3
  TensorProto:  dims=1 (packed int64), data_type=2, float_data=4,
                int32_data=5, string_data=6, int64_data=7, name=8,
                raw_data=9, double_data=10, uint64_data=11,
                data_location=14, external_data=13
  OpsetId:      domain=1, version=2
  StringStringEntryProto: key=1, value=2
"""

import struct
from typing import Any, Iterator, List, Optional, Tuple

__all__ = [
    "_pb_varint",
    "_pb_tag",
    "_pb_varint_field",
    "_pb_bytes_field",
    "_pb_string_field",
    "_pb_float_field",
    "_pb_sub",
    "_pb_read_varint",
    "_pb_signed",
    "_iter_fields",
    "VARINT_FIELD",
    "LEN_FIELD",
    "FIXED32_FIELD",
    "FIXED64_FIELD",
]

# Wire types
VARINT_FIELD = 0
FIXED64_FIELD = 1
LEN_FIELD = 2
FIXED32_FIELD = 5


def _pb_varint(value: int) -> bytes:
    value &= 0xFFFFFFFFFFFFFFFF
    out = bytearray()
    while True:
        b = value & 0x7F
        value >>= 7
        if value:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _pb_tag(field: int, wire: int) -> bytes:
    return _pb_varint((field << 3) | wire)


def _pb_varint_field(field: int, value: int) -> bytes:
    return _pb_tag(field, VARINT_FIELD) + _pb_varint(value)


def _pb_bytes_field(field: int, payload: bytes) -> bytes:
    return _pb_tag(field, LEN_FIELD) + _pb_varint(len(payload)) + payload


def _pb_string_field(field: int, value: str) -> bytes:
    return _pb_bytes_field(field, value.encode("utf-8"))


def _pb_float_field(field: int, value: float) -> bytes:
    return _pb_tag(field, FIXED32_FIELD) + struct.pack("<f", float(value))


def _pb_sub(field: int, payload: bytes) -> bytes:
    return _pb_bytes_field(field, payload)


def _pb_read_varint(data: bytes, pos: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def _pb_signed(value: int) -> int:
    if value > 0x7FFFFFFFFFFFFFFF:
        value -= 1 << 64
    return value


def _iter_fields(
    data: bytes, start: int = 0, end: Optional[int] = None
) -> Iterator[Tuple[int, int, Any]]:
    """Iterate protobuf fields yielding ``(field, wire_type, value)``.

    Wire 0 (varint) and wire 2 (length-delimited) values are decoded; wire 1
    (fixed64) and wire 5 (fixed32) yield the raw bytes.
    """
    pos = start
    limit = len(data) if end is None else end
    while pos < limit:
        key, pos = _pb_read_varint(data, pos)
        field, wire = key >> 3, key & 7
        if wire == VARINT_FIELD:
            val, pos = _pb_read_varint(data, pos)
            yield field, wire, val
        elif wire == FIXED64_FIELD:
            yield field, wire, data[pos : pos + 8]
            pos += 8
        elif wire == LEN_FIELD:
            ln, pos = _pb_read_varint(data, pos)
            yield field, wire, data[pos : pos + ln]
            pos += ln
        elif wire == FIXED32_FIELD:
            yield field, wire, data[pos : pos + 4]
            pos += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
