"""Tests for formal_verify.py — S8 formal verification."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings import LTLProperty, ModelChecker, StateGraph


def test_ltl_create():
    p = LTLProperty("safety", "always safe")
    assert p.name == "safety"
    assert p.formula == "always safe"


def test_model_checker_add_states():
    mc = ModelChecker()
    mc.add_state("s1")
    mc.add_state("s2")
    mc.add_transition("s1", "s2")
    mc.add_property(LTLProperty("safe", "always safe"))
    res = mc.check()
    assert "safe" in res
    assert res["safe"]


def test_model_checker_multiple_props():
    mc = ModelChecker()
    mc.add_state("s1")
    mc.add_property(LTLProperty("p1", "always x"))
    mc.add_property(LTLProperty("p2", "eventually y"))
    results = mc.check()
    assert "p1" in results
    assert "p2" in results


def test_model_checker_single_prop():
    mc = ModelChecker()
    mc.add_state("s")
    mc.add_property(LTLProperty("only_one", "formula"))
    results = mc.check("only_one")
    assert len(results) == 1


def test_model_checker_no_props():
    mc = ModelChecker()
    assert mc.check() == {}


def test_model_checker_counterexample():
    mc = ModelChecker()
    assert mc.counterexample("any") is None


def test_state_graph():
    g = StateGraph()
    g.add_state("a", {"init"})
    g.add_state("b", {"accept"})
    g.add_transition("a", "b")
    g.set_initial("a")
    assert g.initial == "a"
    assert "b" in g.successors("a")
    assert "init" in g.labels("a")
    assert "accept" in g.labels("b")


def test_state_graph_empty():
    g = StateGraph()
    assert g.initial is None
    assert g.successors("x") == []


def test_state_graph_labels_default():
    g = StateGraph()
    g.add_state("s")
    assert g.labels("s") == set()


def test_model_checker_always_violated_with_trace():
    mc = ModelChecker()
    mc.add_state("s0", is_error=False)
    mc.add_state("s1", is_error=True)
    mc.add_transition("s0", "s1")
    mc.add_property(LTLProperty("safe", "always ok"))
    res = mc.check()
    assert res["safe"] is False
    ce = mc.counterexample("safe")
    assert ce is not None
    assert ce == ["s0", "s1"]


def test_model_checker_always_satisfied_no_trace():
    mc = ModelChecker()
    mc.add_state("s0", is_error=False)
    mc.add_state("s1", is_error=False)
    mc.add_transition("s0", "s1")
    mc.add_property(LTLProperty("safe", "always ok"))
    res = mc.check()
    assert res["safe"] is True
    assert mc.counterexample("safe") is None


def test_model_checker_eventually_violated():
    mc = ModelChecker()
    mc.add_state("s0")
    mc.add_state("s1")
    mc.add_transition("s0", "s1")
    mc.add_property(LTLProperty("live", "eventually goal"))
    res = mc.check()
    assert res["live"] is False


def test_model_checker_eventually_satisfied():
    mc = ModelChecker()
    mc.add_state("s0")
    mc.add_state("goal")
    mc.add_transition("s0", "goal")
    mc.add_property(LTLProperty("live", "eventually goal"))
    res = mc.check()
    assert res["live"] is True


def test_model_checker_counterexample_none_before_check():
    mc = ModelChecker()
    assert mc.counterexample("any") is None


def test_model_checker_branch_counterexample():
    mc = ModelChecker()
    mc.add_state("s0", is_error=False)
    mc.add_state("s1", is_error=False)
    mc.add_state("s2", is_error=True)
    mc.add_transition("s0", "s1")
    mc.add_transition("s0", "s2")
    mc.add_property(LTLProperty("safe", "always ok"))
    res = mc.check()
    assert res["safe"] is False
    ce = mc.counterexample("safe")
    assert ce is not None
    assert ce[0] == "s0"
    assert ce[-1] == "s2"


if __name__ == "__main__":
    test_ltl_create()
    test_model_checker_add_states()
    test_model_checker_multiple_props()
    test_model_checker_single_prop()
    test_model_checker_no_props()
    test_model_checker_counterexample()
    test_model_checker_always_violated_with_trace()
    test_model_checker_always_satisfied_no_trace()
    test_model_checker_eventually_violated()
    test_model_checker_eventually_satisfied()
    test_model_checker_counterexample_none_before_check()
    test_model_checker_branch_counterexample()
    test_state_graph()
    test_state_graph_empty()
    test_state_graph_labels_default()
    print("All formal_verify tests passed.")
