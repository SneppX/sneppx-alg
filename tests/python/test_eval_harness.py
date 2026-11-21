"""Tests for LM Evaluation Harness."""

import json
import tempfile
from pathlib import Path

from SneppX_ALG.interface_bindings.eval_harness import (
    EvalHarness,
    EvalConfig,
    ExactMatchTask,
    MultipleChoiceTask,
    CodingTask,
    extract_answer_gsm8k,
    extract_answer_math,
    extract_answer_mmlu,
    estimate_pass_at_k,
    extract_code,
    MMLU_CATEGORIES,
)


def test_extract_answer_gsm8k_basic():
    text = "The answer is 42.#### 42"
    assert extract_answer_gsm8k(text) == "42"


def test_extract_answer_gsm8k_fallback():
    text = "The final number is 7"
    assert extract_answer_gsm8k(text) == "7"


def test_extract_answer_gsm8k_empty():
    assert extract_answer_gsm8k("") is None


def test_extract_answer_math_boxed():
    text = "We find \\boxed{42}."
    assert extract_answer_math(text) == "42"


def test_extract_answer_mmlu_direct():
    assert extract_answer_mmlu("A", ["A", "B", "C"]) == "A"


def test_extract_answer_mmlu_in_text():
    assert extract_answer_mmlu("I think B is correct", ["A", "B", "C"]) == "B"


def test_extract_answer_mmlu_none():
    assert extract_answer_mmlu("No letter here", ["A", "B", "C"]) is None


def test_extract_code_triple_backtick():
    text = "```python\ndef foo(): pass\n```"
    assert extract_code(text) == "def foo(): pass"


def test_extract_code_plain():
    text = "def foo(): pass"
    assert extract_code(text) == "def foo(): pass"


def test_extract_code_empty():
    assert extract_code("") == ""


def test_pass_at_k_perfect():
    assert estimate_pass_at_k(10, 10, 1) == 1.0


def test_pass_at_k_zero():
    assert estimate_pass_at_k(10, 0, 1) == 0.0


def test_pass_at_k_no_samples():
    assert estimate_pass_at_k(0, 0, 1) == 0.0


def test_harness_register():
    harness = EvalHarness()
    assert len(harness.tasks) == 0
    harness.register_defaults()
    assert "mmlu" in harness.tasks
    assert "gsm8k" in harness.tasks
    assert "humaneval" in harness.tasks
    assert "math" in harness.tasks


def test_harness_run_single():
    harness = EvalHarness(EvalConfig(tasks=["gsm8k"]))
    harness.register_task(ExactMatchTask("gsm8k", extract_answer_gsm8k))

    def dummy_model(prompt, ctx):
        return "The answer is 42.#### 42"

    harness.run(dummy_model)
    assert "gsm8k" in harness.results
    # Falls back to synthetic data when no file exists
    assert harness.results["gsm8k"].num_samples > 0


def test_exact_match_task_evaluate():
    task = ExactMatchTask("test", extract_answer_gsm8k)
    result = task.evaluate(
        ["Answer: 42.#### 42", "Answer: 10.#### 10", "Answer: x"],
        ["42", "10", "5"],
    )
    assert result.metric == "exact_match"
    assert result.value == 2.0 / 3.0 * 100


def test_multiple_choice_task_evaluate():
    task = MultipleChoiceTask("test_mmlu", fewshot=0)
    result = task.evaluate(
        ["A", "B", "C", "D"],
        ["A", "B", "C", "D"],
    )
    assert result.metric == "accuracy"
    assert result.value == 100.0
    assert result.num_samples == 4


def test_multiple_choice_half_correct():
    task = MultipleChoiceTask("test_mmlu", fewshot=0)
    result = task.evaluate(
        ["A", "B", "A", "B"],
        ["A", "B", "C", "D"],
    )
    assert result.value == 50.0


def test_coding_task_evaluate():
    task = CodingTask("test_code", fewshot=0)
    result = task.evaluate(
        ["```python\ndef foo(): return 1\n```", "bad code"],
        ["def foo(): return 1", "def bar(): pass"],
    )
    assert result.metric == "pass@1"
    assert result.num_samples == 2


def test_extract_code_triple_backtick_with_lang():
    text = "```python\ndef foo():\n    pass\n```"
    assert extract_code(text) == "def foo():\n    pass"


def test_extract_code_multi_backtick():
    text = "Here is the code:\n```\ndef bar(x): return x+1\n```"
    assert extract_code(text) == "def bar(x): return x+1"


def test_distribution_tail():
    """Test that pass@k handles edge cases near zero samples."""
    for k in [1, 5, 10]:
        assert 0.0 <= estimate_pass_at_k(0, 0, k) <= 1.0
        assert 0.0 <= estimate_pass_at_k(1, 0, k) <= 1.0


def test_harness_summary():
    harness = EvalHarness()
    harness.register_defaults()
    harness.results["test"] = type("R", (), {"name": "test", "metric": "acc", "value": 85.0, "num_samples": 100})()
    s = harness.summary()
    assert "test" in s
    assert "85.00" in s


def test_export_json():
    harness = EvalHarness()
    harness.results["test"] = type("R", (), {"name": "test", "metric": "acc", "value": 85.0, "num_samples": 100})()
    j = harness.export_json()
    data = json.loads(j)
    assert data["test"]["value"] == 85.0


def test_export_json_with_path():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    try:
        harness = EvalHarness(EvalConfig(output_path=path))
        harness.results["test"] = type("R", (), {"name": "test", "metric": "acc", "value": 85.0, "num_samples": 100})()
        harness.export_json()
        with open(path) as f2:
            data = json.load(f2)
        assert data["test"]["value"] == 85.0
    finally:
        Path(path).unlink(missing_ok=True)


def test_mmlu_categories_count():
    assert len(MMLU_CATEGORIES) == 57


if __name__ == "__main__":
    import sys
    locals_ = locals().copy()
    passed = 0
    failed = 0
    for name, fn in sorted(locals_.items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"  PASS {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL {name}: {e}")
                failed += 1
    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    sys.exit(failed)
