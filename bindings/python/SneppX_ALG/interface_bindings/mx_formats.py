"""MX (micro-scaling) deployment formats — MXFP4/MXFP8/MXFP6/NVFP4.

Implements the OCP micro-scaling encode/decode path used by next-generation
deployment formats (vLLM: FP8/MXFP8/MXFP4/NVFP4; NVIDIA Blackwell). A single
shared scale per group of ``block_size`` elements is stored as an 8-bit
power-of-two exponent, and each element is stored in a low-precision float
mantissa format.

Supported element formats:

- ``MXFP8_E4M3`` / ``MXFP8_E5M2``: 8-bit floats, shared scale per 32.
- ``MXFP6_E3M2`` / ``MXFP6_E2M3``: 6-bit floats, shared scale per 32.
- ``MXFP4_E2M1``: 4-bit floats, shared scale per 32.
- ``NVFP4_E2M1``: 4-bit floats, per-tensor scale (Blackwell style).

Typical usage::

    qarr, scales = quantize_mx(weights, format="mxfp4")
    recon = dequantize_mx(qarr, scales, shape, format="mxfp4")
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "MXFormat",
    "MXFormatSpec",
    "MX_FORMATS",
    "quantize_mx",
    "dequantize_mx",
    "mx_error",
]


class MXFormat:
    """Enum-like constants for supported micro-scaling formats."""

    MXFP8_E4M3 = "mxfp8_e4m3"
    MXFP8_E5M2 = "mxfp8_e5m2"
    MXFP6_E3M2 = "mxfp6_e3m2"
    MXFP6_E2M3 = "mxfp6_e2m3"
    MXFP4_E2M1 = "mxfp4_e2m1"
    NVFP4_E2M1 = "nvfp4_e2m1"


@dataclass(frozen=True)
class MXFormatSpec:
    """Encoding parameters for one micro-scaling format."""

    name: str
    bits: int          # bits per element
    exp_bits: int      # element exponent bits
    mant_bits: int     # element mantissa bits
    bias: int          # exponent bias
    block_size: int    # elements sharing one scale (0 = per-tensor)
    max_exp: float     # max finite value with scale=1


MX_FORMATS: dict = {
    MXFormat.MXFP8_E4M3: MXFormatSpec(
        "mxfp8_e4m3", 8, 4, 3, 7, 32, 448.0),
    MXFormat.MXFP8_E5M2: MXFormatSpec(
        "mxfp8_e5m2", 8, 5, 2, 15, 32, 57344.0),
    MXFormat.MXFP6_E3M2: MXFormatSpec(
        "mxfp6_e3m2", 6, 3, 2, 3, 32, 28.0),
    MXFormat.MXFP6_E2M3: MXFormatSpec(
        "mxfp6_e2m3", 6, 2, 3, 1, 32, 14.0),
    MXFormat.MXFP4_E2M1: MXFormatSpec(
        "mxfp4_e2m1", 4, 2, 1, 1, 32, 6.0),
    MXFormat.NVFP4_E2M1: MXFormatSpec(
        "nvfp4_e2m1", 4, 2, 1, 1, 0, 6.0),
}


def _decode_elements(raw: np.ndarray, spec: MXFormatSpec) -> np.ndarray:
    """Decode a packed uint8 array into float32 elements (low-bits-first)."""
    bits = spec.bits
    total = raw.size * 8
    n = total // bits
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    flat = np.unpackbits(raw, bitorder="little").astype(np.int32)[: n * bits]
    codes = np.zeros(n, dtype=np.int32)
    for b in range(bits):
        codes |= (flat[b::bits] << b)
    sign = (codes >> (bits - 1)) & 1
    exp = (codes >> spec.mant_bits) & ((1 << spec.exp_bits) - 1)
    mant = codes & ((1 << spec.mant_bits) - 1)
    out = np.where(
        exp == 0,
        (mant / (1 << spec.mant_bits)) * (2.0 ** (1 - spec.bias)),
        (1.0 + mant / (1 << spec.mant_bits)) * (2.0 ** (exp - spec.bias)),
    )
    return np.where(sign == 1, -out, out)


def _encode_elements(arr: np.ndarray, spec: MXFormatSpec) -> np.ndarray:
    """Encode float32 elements into a packed uint8 array (low-bits-first)."""
    bits = spec.bits
    n = arr.size
    sign = np.where(arr < 0, 1, 0).astype(np.int32)
    mag = np.abs(arr).astype(np.float64)
    mag = np.where(mag == 0, 1e-30, mag)
    exp = np.floor(np.log2(mag)).astype(np.int64)
    exp = np.clip(exp, -spec.bias, (1 << spec.exp_bits) - 1 - spec.bias)
    subnormal = exp <= -spec.bias
    exp_field = np.where(
        subnormal, 0, np.clip(exp + spec.bias, 1, (1 << spec.exp_bits) - 1))
    mant = np.where(
        subnormal,
        np.clip(np.round(mag * 2.0 ** (spec.mant_bits + spec.bias - 1)),
                0, (1 << spec.mant_bits) - 1),
        np.clip(np.round((mag / (2.0 ** exp) - 1.0) * (1 << spec.mant_bits)),
                0, (1 << spec.mant_bits) - 1),
    ).astype(np.int64)
    codes = (sign << (bits - 1)) | (exp_field << spec.mant_bits) | mant

    pad = (-(n * bits)) % 8
    codes_padded = np.concatenate([codes, np.zeros(pad, dtype=np.int32)])
    n_bits = codes_padded.size * bits
    bitplane = np.zeros(n_bits, dtype=np.uint8)
    for b in range(bits):
        bitplane[b::bits] = ((codes_padded >> b) & 1).astype(np.uint8)
    return np.packbits(bitplane, bitorder="little")


def _max_element(spec: MXFormatSpec) -> float:
    """Largest finite value one element can hold (all exponent fields used)."""
    exp_max = (1 << spec.exp_bits) - 1
    return (2.0 - 2.0 ** (-spec.mant_bits)) * (2.0 ** (exp_max - spec.bias))


def _scale_exp(peak: float, spec: MXFormatSpec) -> int:
    """Shared power-of-two exponent scaling ``peak`` into the format range.

    Picks the smallest integer scale so that ``peak * 2**-scale <= max_val``,
    i.e. the smallest scaling down (largest scaling up) that still fits, which
    keeps as many elements as possible in the normal (full-precision) range.
    """
    if peak <= 0:
        return 0
    return int(np.ceil(np.log2(peak) - np.log2(_max_element(spec))))


def quantize_mx(arr, format: str = MXFormat.MXFP4_E2M1,
                block_size: Optional[int] = None,
                block_axis: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize a tensor into a micro-scaling format.

    Args:
        arr: float32 tensor to quantize.
        format: one of :class:`MXFormat`.
        block_size: override the format's default block size.
        block_axis: axis to group blocks along (default last).

    Returns:
        ``(packed_q, scales)`` where ``packed_q`` is a uint8 array of packed
        codes and ``scales`` is a power-of-two exponent array (int8) with one
        entry per block.
    """
    arr = np.asarray(arr, dtype=np.float32)
    spec = MX_FORMATS[format]
    bs = block_size or spec.block_size

    if bs is None or bs <= 0:
        peak = float(np.abs(arr).max()) if arr.size else 1.0
        peak = peak if peak > 0 else 1.0
        scale_exp = _scale_exp(peak, spec)
        scaled = arr / (2.0 ** scale_exp)
        packed = _encode_elements(scaled.reshape(-1), spec)
        return packed, np.array([scale_exp], dtype=np.int8)

    moved = np.moveaxis(arr, block_axis, -1)
    orig_last = moved.shape[-1]
    pad = (-orig_last) % bs
    if pad:
        moved = np.concatenate(
            [moved, np.zeros(moved.shape[:-1] + (pad,), dtype=np.float32)],
            axis=-1,
        )
    nblocks = moved.shape[-1] // bs
    blocked = moved.reshape(moved.shape[:-1] + (nblocks, bs))
    peak = np.abs(blocked).max(axis=-1)
    peak = np.where(peak <= 0, 1.0, peak)
    scale_exp = np.ceil(
        np.log2(peak) - np.log2(_max_element(spec))).astype(np.int64)
    scaled = blocked / (2.0 ** scale_exp[..., None])
    packed = _encode_elements(scaled.reshape(-1), spec)

    scales = scale_exp.astype(np.int8)
    scales = np.moveaxis(scales, -1, block_axis)
    return packed, scales


def dequantize_mx(packed: np.ndarray, scales: np.ndarray,
                  shape, format: str = MXFormat.MXFP4_E2M1,
                  block_size: Optional[int] = None,
                  block_axis: int = -1) -> np.ndarray:
    """Dequantize a packed micro-scaling tensor back to float32."""
    spec = MX_FORMATS[format]
    bs = block_size or spec.block_size
    n = int(np.prod(shape))
    shape = tuple(int(s) for s in shape)

    if bs is None or bs <= 0:
        codes = _decode_elements(packed, spec)
        scale_exp = np.asarray(scales, dtype=np.float32).reshape(-1)
        s = 2.0 ** (scale_exp[0] if scale_exp.size else 0)
        return (codes * s).reshape(shape)

    codes = _decode_elements(packed, spec)
    orig_last = shape[-1]
    nblocks = (orig_last + bs - 1) // bs
    blocked = codes.reshape(shape[:-1] + (nblocks, bs))
    scale_flat = np.asarray(scales, dtype=np.float32)
    if scale_flat.size == 0:
        scale_flat = np.ones(shape[:-1] + (nblocks,), dtype=np.float32)
    recon = blocked * (2.0 ** scale_flat[..., None])
    flat_last = recon.reshape(shape[:-1] + (nblocks * bs,))
    return flat_last[..., :orig_last]


def mx_error(original, reconstructed, metric: str = "mse") -> float:
    orig = np.asarray(original, dtype=np.float32).reshape(-1)
    recon = np.asarray(reconstructed, dtype=np.float32).reshape(-1)
    if metric == "mse":
        return float(np.mean((orig - recon) ** 2))
    elif metric == "mae":
        return float(np.mean(np.abs(orig - recon)))
    elif metric == "snr":
        signal = float(np.mean(orig ** 2))
        noise = float(np.mean((orig - recon) ** 2))
        return float(10 * np.log10(signal / noise)) if noise > 0 else float("inf")
    return 0.0
