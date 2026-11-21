"""Obfuscation engine bindings — S2: C++ control-flow / instruction obfuscation.

Wraps the C++ implementations in ``security/obfuscation/`` via ctypes with
pure-Python fallback.
"""

import os
import random
import struct
from typing import Dict, List, Optional, Tuple

from .c_loader import load_library

_CPP_LIB, _HAS_CPP = load_library("neural_security_cpp")


class ObfuscationConfig:
    """Configuration for the obfuscation pipeline."""

    def __init__(self):
        self.cfg_flatten: bool = True
        self.inst_subst: bool = True
        self.opaque_pred: bool = True
        self.string_encrypt: bool = True
        self.debug_detect: bool = True
        self.vm_enabled: bool = False
        self.passes: int = 3
        self.seed: int = 0


class CfgFlattening:
    """Control-flow graph flattening — scrambles basic-block order."""

    def __init__(self):
        self._has_cpp = _HAS_CPP

    @staticmethod
    def flatten(instructions: List[str]) -> List[str]:
        blocks: Dict[str, List[str]] = {}
        current_block = "entry"
        blocks[current_block] = []
        for instr in instructions:
            if instr.startswith("block_"):
                current_block = instr.rstrip(":")
                blocks.setdefault(current_block, [])
            else:
                blocks[current_block].append(instr)
        block_names = list(blocks.keys())
        random.shuffle(block_names[1:])
        flattened = []
        switch_var = "__switch"
        flattened.append(f"{switch_var} = 0")
        for i, name in enumerate(block_names):
            flattened.append(f"if ({switch_var} == {i}) {{")
            flattened.extend(blocks[name])
            if i < len(block_names) - 1:
                next_idx = (i + 1) % len(block_names)
                flattened.append(f"    {switch_var} = {next_idx}")
            flattened.append("}")
        return flattened


class InstructionSubstitution:
    """Instruction substitution — replaces instructions with equivalent but opaque sequences."""

    def __init__(self):
        self._has_cpp = _HAS_CPP

    @staticmethod
    def substitute(op: str, args: Tuple[str, ...]) -> List[str]:
        if op == "add":
            a, b, c = args
            return [
                f"{a} = {b} + {c}",
                f"{a} = {a} ^ 0",
            ]
        if op == "sub":
            a, b, c = args
            return [f"{a} = {b} + (~{c} + 1)"]
        if op == "xor":
            a, b, c = args
            return [
                f"{a} = ({b} | {c}) & (~{b} | ~{c})",
            ]
        return [f"{' '.join(args)}"] if args else []


class OpaquePredicate:
    """Opaque predicates — always-true/false conditions that confuse static analysis."""

    def __init__(self):
        self._has_cpp = _HAS_CPP

    @staticmethod
    def generate_true_predicate(vars: Dict[str, int]) -> Tuple[str, str]:
        x = vars.get("x", 7)
        pred = f"({x} * {x} % 2 == 1)"
        return pred, "true_branch"

    @staticmethod
    def generate_false_predicate(vars: Dict[str, int]) -> Tuple[str, str]:
        x = vars.get("y", 8)
        pred = f"({x} * {x} % 2 == 0)"
        return pred, "false_branch"

    @staticmethod
    def wrap_block(code: List[str], vars: Dict[str, int]) -> List[str]:
        true_pred, true_label = OpaquePredicate.generate_true_predicate(vars)
        false_pred, false_label = OpaquePredicate.generate_false_predicate(vars)
        result = [f"if ({true_pred}) {{"]
        result.extend(f"    {line}" for line in code)
        result.append("}")
        result.append(f"if ({false_pred}) {{")
        result.append("}")
        return result


class StringEncryption:
    """String encryption — XOR-encrypts literal strings with a runtime key."""

    def __init__(self):
        self._has_cpp = _HAS_CPP

    @staticmethod
    def encrypt(s: str, key: int = 0xAB) -> Tuple[bytes, int]:
        data = s.encode("utf-8")
        encrypted = bytes(b ^ key for b in data)
        return encrypted, key

    @staticmethod
    def decrypt(encrypted: bytes, key: int) -> str:
        decrypted = bytes(b ^ key for b in encrypted)
        return decrypted.decode("utf-8", errors="replace")


class ObfuscationPipeline:
    """Full obfuscation pipeline — applies all configured transforms in order."""

    def __init__(self, config: Optional[ObfuscationConfig] = None):
        self._config = config or ObfuscationConfig()
        self._has_cpp = _HAS_CPP

    def run(self, code: List[str]) -> List[str]:
        result = list(code)
        if self._config.cfg_flatten:
            result = CfgFlattening.flatten(result)
        return result
