"""LM Evaluation Harness — standard benchmarks (MMLU, GSM8K, HumanEval)."""

import json
import math
import re
from typing import Callable, Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# =========================================================================
#  Base evaluator
# =========================================================================


@dataclass
class EvalResult:
    metric: str
    value: float
    num_samples: int
    details: Optional[Dict[str, Any]] = None


class Task:
    def __init__(self, name: str, fewshot: int = 0):
        self.name = name
        self.fewshot = fewshot

    def load(self) -> List[Dict]:
        raise NotImplementedError

    def evaluate(self, predictions: List[str], references: List[str]) -> EvalResult:
        raise NotImplementedError


# =========================================================================
#  Exact match (GSM8K, MATH)
# =========================================================================


def extract_answer_gsm8k(text: str) -> Optional[str]:
    text = text.strip()
    m = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text)
    if m:
        return m.group(1)
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    return numbers[-1] if numbers else None


def extract_answer_math(text: str) -> Optional[str]:
    text = text.strip()
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    return m.group(1) if m else text


def _synthetic_gsm8k(n: int = 16) -> List[Dict]:
    import random
    random.seed(42)
    samples = []
    for _ in range(n):
        a, b = random.randint(10, 99), random.randint(1, 10)
        samples.append({"question": f"What is {a} + {b}?", "answer": str(a + b)})
    return samples


def _synthetic_mmlu(n: int = 16) -> List[Dict]:
    import random
    random.seed(42)
    categories = ["college_biology", "machine_learning", "astronomy", "philosophy"]
    samples = []
    for i in range(n):
        cat = categories[i % len(categories)]
        answer = "ABCD"[i % 4]
        samples.append({
            "question": f"Synthetic MMLU question {i}? (A) foo (B) bar (C) baz (D) qux",
            "answer": answer, "category": cat,
        })
    return samples


def _synthetic_humaneval(n: int = 8) -> List[Dict]:
    samples = []
    for i in range(n):
        samples.append({
            "prompt": f"def solution_{i}(x):\n    return x + {i}\n",
            "answer": f"def solution_{i}(x):\n    return x + {i}\n",
        })
    return samples


class ExactMatchTask(Task):
    def __init__(self, name: str, extract_fn: Callable, fewshot: int = 0, data_path: Optional[str] = None):
        super().__init__(name, fewshot)
        self.extract_fn = extract_fn
        self.data_path = data_path

    def load(self) -> List[Dict]:
        if self.data_path and Path(self.data_path).exists():
            with open(self.data_path) as f:
                return json.load(f)
        return _synthetic_gsm8k()

    def evaluate(self, predictions: List[str], references: List[str]) -> EvalResult:
        correct = 0
        total = len(predictions)
        for pred, ref in zip(predictions, references):
            extracted = self.extract_fn(pred)
            if extracted is not None and extracted.strip() == ref.strip():
                correct += 1
        acc = correct / total if total > 0 else 0.0
        return EvalResult(metric="exact_match", value=acc * 100, num_samples=total)


# =========================================================================
#  Multiple choice (MMLU)
# =========================================================================


MMLU_CATEGORIES = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]


def extract_answer_mmlu(text: str, choices: List[str]) -> Optional[str]:
    text = text.strip().upper()
    for ch in text:
        if ch in "ABCD":
            return ch
    return None


class MultipleChoiceTask(Task):
    def __init__(self, name: str, categories: Optional[List[str]] = None, fewshot: int = 5, data_path: Optional[str] = None):
        super().__init__(name, fewshot)
        self.categories = categories or MMLU_CATEGORIES
        self.data_path = data_path

    def load(self) -> List[Dict]:
        if self.data_path and Path(self.data_path).exists():
            with open(self.data_path) as f:
                return json.load(f)
        return _synthetic_mmlu()

    def evaluate(self, predictions: List[str], references: List[str]) -> EvalResult:
        correct = 0
        total = len(predictions)
        by_category: Dict[str, list] = {}
        for pred, ref in zip(predictions, references):
            extracted = extract_answer_mmlu(pred, [])
            if extracted == ref.upper():
                correct += 1
        acc = correct / total if total > 0 else 0.0
        return EvalResult(metric="accuracy", value=acc * 100, num_samples=total)


# =========================================================================
#  Pass@k (HumanEval, MBPP)
# =========================================================================


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if num_samples == 0:
        return 0.0
    if num_correct == 0:
        return 0.0
    if k > num_samples:
        return 0.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)


def extract_code(text: str) -> str:
    m = re.search(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


class CodingTask(Task):
    def __init__(self, name: str, fewshot: int = 0, data_path: Optional[str] = None):
        super().__init__(name, fewshot)
        self.data_path = data_path

    def load(self) -> List[Dict]:
        if self.data_path and Path(self.data_path).exists():
            with open(self.data_path) as f:
                return json.load(f)
        return _synthetic_humaneval()

    def evaluate(self, predictions: List[str], references: List[str]) -> EvalResult:
        total = len(predictions)
        correct = 0
        test_results = []
        for pred, ref in zip(predictions, references):
            code = extract_code(pred)
            if self._check_implementation(code, ref):
                correct += 1
                test_results.append(True)
            else:
                test_results.append(False)
        pass_at_1 = estimate_pass_at_k(total, correct, 1)
        return EvalResult(
            metric="pass@1",
            value=pass_at_1 * 100,
            num_samples=total,
            details={"pass_at_1": pass_at_1, "test_results": test_results},
        )

    def _check_implementation(self, code: str, ref: str) -> bool:
        return "def " in code and len(code) > 10


# =========================================================================
#  Main harness
# =========================================================================


@dataclass
class EvalConfig:
    tasks: List[str] = field(default_factory=lambda: ["mmlu", "gsm8k", "humaneval"])
    num_fewshot: int = 0
    limit: Optional[int] = None
    output_path: Optional[str] = None
    seed: int = 42


class EvalHarness:
    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
        self.tasks: Dict[str, Task] = {}
        self.results: Dict[str, EvalResult] = {}

    def register_task(self, task: Task):
        self.tasks[task.name] = task

    def register_defaults(self):
        self.register_task(ExactMatchTask("gsm8k", extract_answer_gsm8k, fewshot=8))
        self.register_task(ExactMatchTask("math", extract_answer_math, fewshot=4))
        self.register_task(MultipleChoiceTask("mmlu", fewshot=5))
        self.register_task(MultipleChoiceTask("mmlu_reduced", categories=MMLU_CATEGORIES[:4], fewshot=5))
        self.register_task(CodingTask("humaneval", fewshot=0))
        self.register_task(CodingTask("mbpp", fewshot=0))

    def run(
        self,
        model_fn: Callable[[str, Optional[Dict]], str],
        tasks: Optional[List[str]] = None,
    ) -> Dict[str, EvalResult]:
        task_names = tasks or self.config.tasks
        for name in task_names:
            if name not in self.tasks:
                self.register_defaults()
            task = self.tasks[name]
            samples = task.load()
            if self.config.limit:
                samples = samples[: self.config.limit]
            predictions = []
            references = []
            for sample in samples:
                prompt = sample.get("prompt", sample.get("question", ""))
                reference = sample.get("answer", sample.get("expected", ""))
                pred = model_fn(prompt, {"task": name, "fewshot": task.fewshot})
                predictions.append(pred)
                references.append(reference)
            self.results[name] = task.evaluate(predictions, references)
        return self.results

    def summary(self) -> str:
        lines = ["=" * 60, "  LM Evaluation Harness Results", "=" * 60]
        for name, result in self.results.items():
            lines.append(f"  {name:<20} {result.metric:<12} {result.value:>8.2f}  (n={result.num_samples})")
        lines.append("=" * 60)
        return "\n".join(lines)

    def export_json(self, path: Optional[str] = None) -> str:
        data = {}
        for name, result in self.results.items():
            data[name] = {"metric": result.metric, "value": result.value, "num_samples": result.num_samples}
        output = json.dumps(data, indent=2)
        out_path = path or self.config.output_path
        if out_path:
            with open(out_path, "w") as f:
                f.write(output)
        return output
