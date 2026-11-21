"""Shared ctypes type definitions used across ctypes-based binding modules."""

import ctypes
from typing import List, Optional


# --- Fixed-width integer typedefs matching C <stdint.h> ---
int8_t = ctypes.c_int8
uint8_t = ctypes.c_uint8
int16_t = ctypes.c_int16
uint16_t = ctypes.c_uint16
int32_t = ctypes.c_int32
uint32_t = ctypes.c_uint32
int64_t = ctypes.c_int64
uint64_t = ctypes.c_uint64

size_t = ctypes.c_size_t
ssize_t = ctypes.c_ssize_t

# --- Floating-point typedefs ---
float_t = ctypes.c_float
double_t = ctypes.c_double

# --- Pointer typedefs ---
const_char_p = ctypes.c_char_p
void_p = ctypes.c_void_p
const_void_p = ctypes.c_void_p


# --- Utility to create struct types from dict ---
def make_struct(name: str, fields: List[tuple]) -> type:
    """Create a :class:`ctypes.Structure` subclass dynamically.

    Parameters
    ----------
    name:
        The ``_fields_`` name of the struct .
    fields:
        List of ``(field_name, ctypes_type)`` tuples.

    Returns
    -------
    A :class:`ctypes.Structure` subclass usable as a ctypes argument or
    return value.
    """
    cls = type(name, (ctypes.Structure,), {"_fields_": fields})
    return cls


# --- Pre-defined data buffer descriptor (common pattern in C API) ---
class DataBuffer(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
    ]


# --- Callback type for progress / logging ---
ProgressCallback = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p)


# --- Utility: null-terminate a string for C ---
def c_string(s: str) -> bytes:
    return s.encode("utf-8")


# --- Utility: Python buffer from DataBuffer ---
def buffer_from_databuf(db: DataBuffer) -> Optional[memoryview]:
    if db.data and db.size:
        return (ctypes.c_uint8 * db.size).from_address(db.data)
    return None
