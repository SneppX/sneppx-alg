"""Tests for sneppx_serve.obfuscation_bridge (sneppx-alg interface_bindings.obfuscation_bridge)."""

from SneppX_ALG.interface_bindings.obfuscation_bridge import (
    ObfuscationConfig,
    CfgFlattening,
    InstructionSubstitution,
    OpaquePredicate,
    StringEncryption,
    ObfuscationPipeline,
)


def test_config_defaults():
    cfg = ObfuscationConfig()
    assert cfg is not None


def test_cfg_flattening():
    cf = CfgFlattening()
    out = cf.flatten("def f(): pass")
    assert isinstance(out, list)
    assert len(out) > 0


def test_instruction_substitution():
    ss = InstructionSubstitution()
    out = ss.substitute("x", ("1",))
    assert isinstance(out, list)


def test_opaque_predicate_true():
    op = OpaquePredicate()
    t, code = op.generate_true_predicate({"x": 1})
    assert isinstance(t, str)
    assert isinstance(code, str)


def test_opaque_predicate_false():
    op = OpaquePredicate()
    f, code = op.generate_false_predicate({"x": 1})
    assert isinstance(f, str)


def test_opaque_predicate_wrap():
    op = OpaquePredicate()
    out = op.wrap_block(["x = 1", "y = 2"], {"x": 1})
    assert isinstance(out, list)


def test_string_encryption_roundtrip():
    se = StringEncryption()
    enc, key = se.encrypt("hello world")
    assert isinstance(enc, (bytes, bytearray))
    dec = se.decrypt(enc, key)
    assert dec == "hello world"


def test_obfuscation_pipeline():
    cfg = ObfuscationConfig()
    pipe = ObfuscationPipeline(cfg)
    out = pipe.run(["def f():", "    return 1"])
    assert isinstance(out, list)
