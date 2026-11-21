"""Tests for nn.Module serialization robustness:

- nested modules, buffers (incl. non-persistent), register_buffer(get|register)
- state_dict -> load_state_dict roundtrip
- strict mode: missing / unexpected keys, shape mismatch
- lenient (strict=False) partial loads preserving matched keys
- IncompatibleKeys return value, ParameterList/ModuleDict traversal
"""

import sys

sys.path.insert(0, "bindings/python")

import numpy as np

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings import nn


def test_state_dict_roundtrip_basic():
    m = nn.Linear(4, 3)
    x = Tensor.randn((2, 4))
    y1 = m(x)
    sd = m.state_dict()
    assert set(sd.keys()) == {"weight", "bias"}
    w0 = np.asarray(m.weight.data).copy()
    sd["weight"][0, 0] = 123.0  # must not mutate the model (copy semantics)
    assert np.asarray(m.weight.data)[0, 0] == w0[0, 0]

    m2 = nn.Linear(4, 3)
    m2.load_state_dict(m.state_dict())
    y2 = m2(x)
    assert np.allclose(np.asarray(y1.data), np.asarray(y2.data))


def test_state_dict_nested_and_buffers():
    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = Tensor.randn((3, 2))
            self.register_buffer("running_mean", Tensor.zeros((2,)))

        def forward(self, x):
            return x @ self.weight.T + self.running_mean

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = nn.ModuleList([Inner(), Inner()])

        def forward(self, x):
            return self.seq(x)

    o = Outer()
    sd = o.state_dict()
    assert set(sd.keys()) == {
        "seq.0.weight",
        "seq.0.running_mean",
        "seq.1.weight",
        "seq.1.running_mean",
    }
    o2 = Outer()
    res = o2.load_state_dict(sd)
    assert res.missing_keys == [] and res.unexpected_keys == []
    assert np.allclose(
        np.asarray(o2.seq[0].weight.data), np.asarray(o.seq[0].weight.data)
    )
    assert np.allclose(
        np.asarray(o2.seq[1].running_mean.data), np.asarray(o.seq[1].running_mean.data)
    )


def test_non_persistent_buffer_excluded():
    class Foo(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = Tensor.randn((2, 2))
            self.register_buffer("persist", Tensor.ones((2,)))
            self.register_buffer("temp", Tensor.zeros((2,)), persistent=False)

        def forward(self, x):
            return x @ self.weight.T

    f = Foo()
    sd = f.state_dict()
    assert set(sd.keys()) == {"weight", "persist"}
    assert "temp" not in sd
    # named_buffers default still reports the non-persistent one
    assert {n for n, _ in f.named_buffers()} == {"persist", "temp"}
    assert {n for n, _ in f.named_buffers(persistent=True)} == {"persist"}
    assert {n for n, _ in f.named_buffers(persistent=False)} == {"temp"}


def test_get_parameter_get_buffer_register():
    f = nn.Linear(3, 2)
    assert f.get_parameter("weight") is f.weight
    f.register_buffer("buf", Tensor.ones((3,)))
    assert f.get_buffer("buf") is not None
    try:
        f.get_buffer("nope")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_strict_missing_key_raises():
    m = nn.Linear(4, 3)
    try:
        m.load_state_dict({"weight": m.weight.data.copy()})
        assert False, "expected RuntimeError for missing bias"
    except RuntimeError as e:
        assert "bias" in str(e)


def test_strict_partial_load_returns_missing():
    m = nn.Linear(4, 3)
    res = m.load_state_dict({"weight": m.weight.data.copy()}, strict=False)
    assert res.missing_keys == ["bias"]
    assert res.unexpected_keys == []


def test_unexpected_key_raises_in_strict():
    m = nn.Linear(4, 3)
    try:
        m.load_state_dict(
            {"weight": m.weight.data.copy(), "bias": m.bias.data.copy(), "extra": 1}
        )
        assert False, "expected RuntimeError for unexpected key"
    except RuntimeError as e:
        assert "extra" in str(e)


def test_shape_mismatch_raises():
    m = nn.Linear(4, 3)
    try:
        m.load_state_dict({"weight": np.zeros((5, 4)), "bias": m.bias.data.copy()})
        assert False, "expected RuntimeError for size mismatch"
    except RuntimeError as e:
        assert "size mismatch" in str(e)


def test_state_dict_accepts_tensor_values():
    m = nn.Linear(4, 3)
    sd = {"weight": Tensor.randn((3, 4)), "bias": Tensor.randn((3,))}
    res = m.load_state_dict(sd, strict=False)
    assert res.unexpected_keys == []
    assert np.allclose(np.asarray(m.weight.data), np.asarray(sd["weight"].data))


def test_modulelist_moduledict_roundtrip():
    fld = nn.Sequential(nn.ModuleList([nn.Linear(4, 3), nn.Linear(3, 2)]))
    md = nn.ModuleDict({"a": nn.Linear(2, 2), "b": nn.Linear(2, 4)})
    x = Tensor.randn((1, 4))
    o1a = fld(x)
    sda = fld.state_dict()
    assert "0.0.weight" in sda and "0.1.bias" in sda

    fld2 = nn.Sequential(nn.ModuleList([nn.Linear(4, 3), nn.Linear(3, 2)]))
    fld2.load_state_dict(sda)
    assert np.allclose(np.asarray(fld2(x).data), np.asarray(o1a.data))

    sdb = md.state_dict()
    assert "a.weight" in sdb and "b.bias" in sdb
    md2 = nn.ModuleDict({"a": nn.Linear(2, 2), "b": nn.Linear(2, 4)})
    md2.load_state_dict(sdb)
    assert np.allclose(
        np.asarray(md2["a"].weight.data), np.asarray(md["a"].weight.data)
    )