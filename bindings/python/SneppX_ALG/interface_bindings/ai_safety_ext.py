"""Extended AI safety bindings — S5: RLHF safety, prompt filter, output verifier, data poisoning defense.

Extends ``s5_safety.py`` with additional safety modules.

Wraps C implementations in ``security/ai/`` with pure-Python fallback.
"""

import os
import re
import math
import hashlib
import threading
from typing import List, Optional, Set, Dict, Tuple

import numpy as np

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_security_c")


# ---- RLHF Safety extensions ---------------------------------------------


class RLHFSafety:
    """RLHF-based safety — factuality, bias, semantic injection, membership inference."""

    def __init__(self):
        self._factuality_threshold: float = 0.7
        self._bias_scores: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._has_c = _HAS_C

    @staticmethod
    def _demographic_terms() -> Dict[str, List[str]]:
        return {
            "gender": ["he", "she", "him", "her", "man", "woman", "male", "female"],
            "race": ["black", "white", "asian", "hispanic"],
            "age": ["old", "young", "elderly", "youth"],
        }

    def check_factuality(self, statement: str, context: str) -> float:
        """Advisory lexical-support signal (NOT a factuality oracle).

        Returns the fraction of meaningful statement tokens that are present
        in ``context``, halved when the context contains a negation. This is a
        cheap triage signal; it cannot establish truth and must not be used as
        a security control.
        """
        stop_words = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "to",
            "of",
            "and",
            "or",
            "in",
            "on",
            "at",
            "it",
            "its",
        }
        stmt_tokens = {
            w
            for w in re.findall(r"[a-z0-9]+", (statement or "").lower())
            if len(w) > 1 and w not in stop_words
        }
        if not stmt_tokens:
            return 0.0
        context_lower = (context or "").lower()
        ctx_tokens = set(re.findall(r"[a-z0-9]+", context_lower))
        if not ctx_tokens:
            return 0.0
        support = len(stmt_tokens & ctx_tokens) / len(stmt_tokens)
        if re.search(r"\b(not|no|never|isn't|isnt|doesn't|doesnt)\b", context_lower):
            support *= 0.5
        return float(max(0.0, min(1.0, support)))

    def check_bias(self, text: str) -> Dict[str, float]:
        scores = {}
        text_lower = text.lower()
        for category, terms in self._demographic_terms().items():
            matches = sum(1 for t in terms if t in text_lower)
            scores[category] = min(1.0, matches / max(len(terms), 1))
        with self._lock:
            self._bias_scores = scores
        return scores

    def detect_semantic_injection(self, text: str, base_embedding: List[float]) -> float:
        """Advisory cosine-distance signal between ``text`` and a baseline.

        A stable bag-of-tokens projection of ``text`` is compared against the
        caller-supplied ``base_embedding``. ``0.0`` means the text matches the
        baseline; larger values indicate drift worth inspecting. Advisory
        heuristic only — it cannot prove an injection occurred.
        """
        if not base_embedding:
            return 0.0
        dim = len(base_embedding)
        vec = [0.0] * dim
        for token in re.findall(r"[a-z0-9]+", (text or "").lower()):
            slot = int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little") % dim
            vec[slot] += 1.0
        norm_vec = math.sqrt(sum(x * x for x in vec))
        norm_base = math.sqrt(sum(x * x for x in base_embedding))
        if norm_vec == 0.0 or norm_base == 0.0:
            return 0.0
        dot = sum(a * b for a, b in zip(vec, base_embedding))
        cosine = max(-1.0, min(1.0, dot / (norm_vec * norm_base)))
        return float(1.0 - cosine)

    def detect_membership_inference(self, sample: bytes, model_output: bytes) -> float:
        """Advisory confidence-proxy signal based on output entropy.

        Returns ``1 - normalised_shannon_entropy(model_output)``: a near-constant
        output (low entropy) is a weak indicator of memorisation. Advisory
        heuristic only — not a certified membership-inference test.
        """
        if not sample or not model_output:
            return 0.0
        counts: Dict[int, int] = {}
        for byte in model_output:
            counts[byte] = counts.get(byte, 0) + 1
        total = len(model_output)
        entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
        max_entropy = math.log2(min(total, 256)) or 1.0
        return float(max(0.0, min(1.0, 1.0 - entropy / max_entropy)))

    def detect_model_inversion(self, input_gradients: np.ndarray) -> float:
        return float(np.max(np.abs(input_gradients)))

    @property
    def factuality_threshold(self) -> float:
        return self._factuality_threshold

    @factuality_threshold.setter
    def factuality_threshold(self, v: float) -> None:
        self._factuality_threshold = max(0.0, min(1.0, v))


# ---- Prompt Filter extensions -------------------------------------------


class PromptFilter:
    """Extended prompt filtering — jailbreak, encoded attack, policy enforcement.

    Complement to ``s5_safety.S5PromptFilter`` with additional detection
    methods.
    """

    _JAILBREAK_PATTERNS = [
        r"ignor(e|es|ed|ing)\s+(above|previous|all|prior)",
        r"dan|jailbreak|override\s+(mode|protocol|instructions)",
        r"act\s+as\s+(if|though)",
        r"you\s+(are|must|will)\s+(now|act)",
        r"character\s+mode",
        r"do\s+(not|n't)\s+(follow|obey)",
    ]

    def __init__(self):
        self._custom_patterns: List[str] = []
        self._lock = threading.Lock()
        self._has_c = _HAS_C

    def detect_jailbreak(self, text: str) -> Tuple[bool, float]:
        text_lower = text.lower()
        all_patterns = self._JAILBREAK_PATTERNS + self._custom_patterns
        matches = 0
        for p in all_patterns:
            if re.search(p, text_lower):
                matches += 1
        score = min(1.0, matches / 3.0)
        return matches > 0, score

    def detect_encoded_attack(self, text: str) -> Tuple[bool, str]:
        if self._is_base64(text) or self._is_hex(text):
            return True, "encoded_content"
        return False, ""

    @staticmethod
    def _is_base64(s: str) -> bool:
        if len(s) < 8:
            return False
        return bool(re.match(r"^[A-Za-z0-9+/]*={0,2}$", s))

    @staticmethod
    def _is_hex(s: str) -> bool:
        if len(s) < 8:
            return False
        return bool(re.match(r"^[0-9a-fA-F]+$", s))

    def add_pattern(self, pattern: str) -> None:
        with self._lock:
            self._custom_patterns.append(pattern)

    def clear_patterns(self) -> None:
        with self._lock:
            self._custom_patterns.clear()


# ---- Output Verifier extensions -----------------------------------------


class OutputVerifier:
    """Extended output verification — PII/secret leak prevention.

    Complement to ``s5_safety.S5OutputVerifier``.
    """

    _PII_PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "ip": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
    }

    def __init__(self):
        self._custom_patterns: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._has_c = _HAS_C

    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        findings = {}
        all_patterns = {**self._PII_PATTERNS, **self._custom_patterns}
        for pii_type, pattern in all_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                findings[pii_type] = matches
        return findings

    def detect_secrets(self, text: str) -> List[Tuple[str, str]]:
        secrets = []
        api_key = r'(?:api[_-]?key|secret|token)\s*[:=]\s*["\']?([A-Za-z0-9_\-]{16,})["\']?'
        for match in re.finditer(api_key, text, re.IGNORECASE):
            secrets.append(("api_key", match.group(1)))
        return secrets

    def redact(self, text: str, pii_types: Optional[List[str]] = None) -> str:
        types = pii_types or list(self._PII_PATTERNS.keys())
        result = text
        for t in types:
            pattern = self._PII_PATTERNS.get(t)
            if pattern:
                result = re.sub(pattern, "[REDACTED]", result)
        return result

    def add_pattern(self, name: str, pattern: str) -> None:
        with self._lock:
            self._custom_patterns[name] = pattern


# ---- Data Poisoning Defense ---------------------------------------------


class DataPoisoningDefense:
    """Detection and mitigation of data poisoning attacks."""

    def __init__(self):
        self._has_c = _HAS_C

    @staticmethod
    def detect_outliers(embeddings: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        mean = np.mean(embeddings, axis=0, keepdims=True)
        std = np.std(embeddings, axis=0, keepdims=True)
        z_scores = np.abs((embeddings - mean) / (std + 1e-8))
        return np.any(z_scores > threshold, axis=1)

    @staticmethod
    def detect_label_flipping(
        labels: np.ndarray, predictions: np.ndarray, window: int = 100
    ) -> np.ndarray:
        if len(labels) < window:
            return np.zeros(len(labels), dtype=bool)
        disagreement = (labels != predictions).astype(np.float32)
        cumulative = np.convolve(disagreement, np.ones(window) / window, mode="same")
        return cumulative > 0.5

    @staticmethod
    def sanitize_batch(
        data: np.ndarray, labels: np.ndarray, max_deviation: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        outliers = DataPoisoningDefense.detect_outliers(data, max_deviation)
        return data[~outliers], labels[~outliers]

    @staticmethod
    def gradient_norm_detection(gradients: List[np.ndarray], threshold: float = 10.0) -> List[bool]:
        return [float(np.linalg.norm(g)) > threshold for g in gradients]
