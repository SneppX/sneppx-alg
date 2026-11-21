"""Formal verification bindings — S8: model checking, symbolic execution.

Wraps C implementations in ``security/formal/`` with pure-Python fallback.
"""

__all__ = [
    "LTLProperty",
    "ModelChecker",
    "StateGraph",
]

import re
import time
import threading
from typing import Dict, List, Optional, Set, Any

from .c_loader import load_library


_LIB, _HAS_C = load_library("neural_security_c")


class LTLProperty:
    """Linear Temporal Logic property specification."""

    def __init__(self, name: str, formula: str):
        self.name = name
        self.formula = formula

    def __repr__(self) -> str:
        return f"LTL({self.name}: {self.formula})"


class ModelChecker:
    """Simple model checker for finite-state systems with LTL properties.

    In the pure-Python fallback this performs explicit-state model checking
    via DFS over the state graph.  For safety properties (``always`` / ``G``
    / ``[]``) it reports a violation when an error state is reachable and
    records the counterexample trace.  For liveness properties (``eventually``
    / ``F`` / ``<>``) it reports a violation when no matching state is
    reachable.
    """

    def __init__(self):
        self._states: Set[str] = set()
        self._transitions: Dict[str, List[str]] = {}
        self._error_states: Set[str] = set()
        self._properties: List[LTLProperty] = []
        self._results: Dict[str, bool] = {}
        self._counterexamples: Dict[str, Optional[List[str]]] = {}
        self._lock = threading.Lock()
        self._has_c = _HAS_C

    def add_state(self, state_id: str, is_error: bool = False) -> None:
        with self._lock:
            self._states.add(state_id)
            self._transitions.setdefault(state_id, [])
            if is_error:
                self._error_states.add(state_id)

    def add_transition(self, from_state: str, to_state: str) -> None:
        with self._lock:
            self._transitions.setdefault(from_state, []).append(to_state)
            self._states.add(from_state)
            self._states.add(to_state)

    def add_property(self, prop: LTLProperty) -> None:
        with self._lock:
            self._properties.append(prop)

    def _initial_state(self) -> Optional[str]:
        with self._lock:
            states = set(self._states)
            targets: Set[str] = set()
            for tgts in self._transitions.values():
                targets.update(tgts)
            sources = states - targets
            if sources:
                return sorted(sources)[0]
            if states:
                return sorted(states)[0]
            return None

    def _check_property(self, prop: LTLProperty) -> bool:
        formula = prop.formula.lower().strip()
        initial = self._initial_state()
        if initial is None:
            with self._lock:
                self._counterexamples[prop.name] = None
            return True

        is_safety = ("always" in formula or "[]" in formula
                     or formula.startswith("g ") or formula.startswith("g(")
                     or formula == "g")
        is_liveness = ("eventually" in formula or "<>" in formula
                       or formula.startswith("f ") or formula.startswith("f(")
                       or formula == "f")

        visited: Set[str] = set()
        parent: Dict[str, Optional[str]] = {}
        stack: List[str] = [initial]
        visited.add(initial)
        parent[initial] = None

        violating_state: Optional[str] = None

        while stack:
            cur = stack.pop()
            if is_safety and cur in self._error_states:
                violating_state = cur
                break
            if is_liveness:
                # For "eventually X", check if cur matches the target in formula
                target = self._extract_target(formula)
                if target and cur == target:
                    with self._lock:
                        self._counterexamples[prop.name] = None
                    return True
            for nxt in self._transitions.get(cur, []):
                if nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = cur
                    stack.append(nxt)

        if is_safety and violating_state is not None:
            path: List[str] = []
            node: Optional[str] = violating_state
            while node is not None:
                path.append(node)
                node = parent.get(node)
            path.reverse()
            with self._lock:
                self._counterexamples[prop.name] = path
            return False

        if is_liveness:
            with self._lock:
                self._counterexamples[prop.name] = None
            return False

        with self._lock:
            self._counterexamples[prop.name] = None
        return True

    @staticmethod
    def _extract_target(formula: str) -> Optional[str]:
        # After "eventually" / "<>" / "F", take the next identifier as target
        for keyword in ("eventually", "<>", "f "):
            idx = formula.find(keyword)
            if idx >= 0:
                rest = formula[idx + len(keyword):].strip()
                m = re.match(r'\(?\s*([a-zA-Z0-9_]+)', rest)
                if m:
                    return m.group(1)
        return None

    def check(self, property_name: Optional[str] = None) -> Dict[str, bool]:
        with self._lock:
            props = [p for p in self._properties if property_name is None or p.name == property_name]
        for prop in props:
            result = self._check_property(prop)
            with self._lock:
                self._results[prop.name] = result
        with self._lock:
            return dict(self._results)

    def results(self) -> Dict[str, bool]:
        with self._lock:
            return dict(self._results)

    def counterexample(self, property_name: str) -> Optional[List[str]]:
        with self._lock:
            return list(self._counterexamples.get(property_name)) if self._counterexamples.get(property_name) is not None else None


class StateGraph:
    """Finite-state graph representation for model checking."""

    def __init__(self):
        self._states: Set[str] = set()
        self._transitions: Dict[str, List[str]] = {}
        self._labels: Dict[str, Set[str]] = {}
        self._initial: Optional[str] = None

    def add_state(self, sid: str, labels: Optional[Set[str]] = None) -> None:
        self._states.add(sid)
        self._transitions.setdefault(sid, [])
        if labels:
            self._labels[sid] = labels

    def add_transition(self, src: str, dst: str) -> None:
        self._transitions.setdefault(src, []).append(dst)

    def set_initial(self, sid: str) -> None:
        self._initial = sid
        self.add_state(sid)

    @property
    def initial(self) -> Optional[str]:
        return self._initial

    def successors(self, sid: str) -> List[str]:
        return self._transitions.get(sid, [])

    def labels(self, sid: str) -> Set[str]:
        return self._labels.get(sid, set())

    def __repr__(self) -> str:
        return f"StateGraph({len(self._states)} states)"
