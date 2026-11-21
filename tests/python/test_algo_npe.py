"""Tests for algo_npe.py — NPE algorithm bindings."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

import numpy as np
from SneppX_ALG.interface_bindings import NPEInstruction, NPEProgram, NPECompiler, NPEVM, NPEVerify


def test_instruction_create():
    instr = NPEInstruction("add", ["x", "y"], "z")
    assert instr.opcode == "add"
    assert instr.dest == "z"
    assert "z = add(x, y)" in repr(instr)


def test_program_add():
    prog = NPEProgram("test")
    prog.add(NPEInstruction("nop", []))
    assert len(prog) == 1


def test_program_index():
    prog = NPEProgram()
    instr = NPEInstruction("copy", ["x"], "y")
    prog.add(instr)
    assert prog[0] is instr


def test_compiler_compile():
    comp = NPECompiler()
    ops = [
        {"type": "matmul", "args": ["a", "b"], "dest": "c"},
        {"type": "relu", "args": ["c"], "dest": "d"},
    ]
    prog = comp.compile(ops)
    assert len(prog) == 2
    assert prog[0].opcode == "matmul"


def test_vm_execute_add():
    vm = NPEVM()
    prog = NPEProgram()
    prog.add(NPEInstruction("add", ["x", "y"], "z"))
    inputs = {"x": np.array(3.0), "y": np.array(4.0)}
    outputs = vm.execute(prog, inputs)
    assert "z" in outputs
    assert np.allclose(outputs["z"], 7.0)


def test_vm_execute_mul():
    vm = NPEVM()
    prog = NPEProgram()
    prog.add(NPEInstruction("mul", ["a", "b"], "c"))
    inputs = {"a": np.array([2.0, 3.0]), "b": np.array([4.0, 5.0])}
    outputs = vm.execute(prog, inputs)
    assert np.allclose(outputs["c"], [8.0, 15.0])


def test_vm_execute_matmul():
    vm = NPEVM()
    prog = NPEProgram()
    prog.add(NPEInstruction("matmul", ["a", "b"], "c"))
    a = np.random.randn(2, 3).astype(np.float32)
    b = np.random.randn(3, 4).astype(np.float32)
    outputs = vm.execute(prog, {"a": a, "b": b})
    assert np.allclose(outputs["c"], a @ b)


def test_vm_execute_relu():
    vm = NPEVM()
    prog = NPEProgram()
    prog.add(NPEInstruction("relu", ["x"], "y"))
    x = np.array([-1.0, 0.0, 2.0])
    outputs = vm.execute(prog, {"x": x})
    assert np.allclose(outputs["y"], [0.0, 0.0, 2.0])


def test_vm_reset():
    vm = NPEVM()
    prog = NPEProgram()
    prog.add(NPEInstruction("copy", ["x"], "y"))
    vm.execute(prog, {"x": np.array(1.0)})
    vm.reset()
    assert True


def test_verify_types():
    v = NPEVerify()
    prog = NPEProgram()
    prog.add(NPEInstruction("add", ["a", "b"], "c"))
    errors = v.check_types(prog, {"a": "float", "b": "float"})
    assert len(errors) == 0


def test_verify_types_missing():
    v = NPEVerify()
    prog = NPEProgram()
    prog.add(NPEInstruction("add", ["undefined_var"], "c"))
    errors = v.check_types(prog, {})
    assert len(errors) > 0


def test_verify_bounds():
    v = NPEVerify()
    prog = NPEProgram()
    for _ in range(5):
        prog.add(NPEInstruction("nop", []))
    assert v.check_bounds(prog, max_ops=10)
    assert not v.check_bounds(prog, max_ops=3)


if __name__ == "__main__":
    test_instruction_create()
    test_program_add()
    test_program_index()
    test_compiler_compile()
    test_vm_execute_add()
    test_vm_execute_mul()
    test_vm_execute_matmul()
    test_vm_execute_relu()
    test_vm_reset()
    test_verify_types()
    test_verify_types_missing()
    test_verify_bounds()
    print("All algo_npe tests passed.")
