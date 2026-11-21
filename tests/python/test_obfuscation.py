"""Tests for obfuscation_bridge.py — S2 C++ obfuscation."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings import (
    ObfuscationConfig, CfgFlattening, InstructionSubstitution,
    OpaquePredicate, StringEncryption, ObfuscationPipeline,
)


def test_cfg_flatten():
    cf = CfgFlattening()
    code = ["block_a:", "mov eax, 1", "block_b:", "mov ebx, 2"]
    result = cf.flatten(code)
    assert len(result) > len(code)
    assert all(isinstance(l, str) for l in result)


def test_cfg_preserves_blocks():
    cf = CfgFlattening()
    code = ["block_start:", "nop", "block_end:", "ret"]
    result = cf.flatten(code)
    block_count = sum(1 for l in result if l.startswith("if (__switch"))
    assert block_count > 0


def test_inst_sub_add():
    result = InstructionSubstitution.substitute("add", ("a", "b", "c"))
    assert len(result) >= 2


def test_inst_sub_sub():
    result = InstructionSubstitution.substitute("sub", ("a", "b", "c"))
    assert len(result) >= 1


def test_inst_sub_xor():
    result = InstructionSubstitution.substitute("xor", ("a", "b", "c"))
    assert len(result) >= 1


def test_opaque_pred_true():
    pred, label = OpaquePredicate.generate_true_predicate({"x": 7})
    assert "true_branch" in label


def test_opaque_pred_false():
    pred, label = OpaquePredicate.generate_false_predicate({"y": 8})
    assert "false_branch" in label


def test_opaque_wrap():
    code = ["mov eax, 1", "ret"]
    wrapped = OpaquePredicate.wrap_block(code, {"x": 7, "y": 8})
    assert len(wrapped) > len(code)
    assert "if (" in wrapped[0]


def test_string_encrypt_decrypt():
    se = StringEncryption()
    enc, key = se.encrypt("hello world")
    dec = se.decrypt(enc, key)
    assert dec == "hello world"


def test_string_encrypt_empty():
    se = StringEncryption()
    enc, key = se.encrypt("")
    dec = se.decrypt(enc, key)
    assert dec == ""


def test_string_encrypt_unicode():
    se = StringEncryption()
    enc, key = se.encrypt("héllo")
    dec = se.decrypt(enc, key)
    assert dec == "héllo"


def test_pipeline_default():
    pipe = ObfuscationPipeline()
    code = ["block_a:", "nop", "block_b:", "ret"]
    result = pipe.run(code)
    assert len(result) > 0


def test_pipeline_disabled():
    config = ObfuscationConfig()
    config.cfg_flatten = False
    pipe = ObfuscationPipeline(config)
    code = ["mov eax, 1", "ret"]
    result = pipe.run(code)
    assert result == code


if __name__ == "__main__":
    test_cfg_flatten()
    test_cfg_preserves_blocks()
    test_inst_sub_add()
    test_inst_sub_sub()
    test_inst_sub_xor()
    test_opaque_pred_true()
    test_opaque_pred_false()
    test_opaque_wrap()
    test_string_encrypt_decrypt()
    test_string_encrypt_empty()
    test_string_encrypt_unicode()
    test_pipeline_default()
    test_pipeline_disabled()
    print("All obfuscation tests passed.")
