"""Python bindings for the C S5 AI safety layer (security/ai/s5_extensions.c).

The module exposes three high-level classes:

* :class:`S5PromptFilter`   — jailbreak / encoded-attack / policy enforcement.
* :class:`S5OutputVerifier` — sensitive-data (PII/secret) leakage prevention.
* :class:`S5RLHFSafety`     — factuality, bias, semantic-injection, token
  anomaly, membership-inference & model-inversion defenses, watermarking.

When the compiled ``neural_security_c`` shared library is available it is loaded
via :mod:`ctypes`; otherwise a faithful pure-Python implementation (matching
the C semantics) is used so the API is always available.
"""

import os
import ctypes
import math
import hashlib
from typing import List, Optional, Tuple, Dict, Any

import numpy as np


# ===========================================================================
#  Backend selection: ctypes if compiled lib present, else pure Python
# ===========================================================================


def _find_library() -> Optional[ctypes.CDLL]:
    """Locate the compiled S5 shared library if it was built."""
    candidates = [
        "neural_security_c",
        "libneural_security_c",
        "neural_security_c.dll",
        "libneural_security_c.so",
        "libneural_security_c.dylib",
    ]
    search_dirs = [
        os.environ.get("SNEPPX_SECURITY_LIB", ""),
        os.path.join(os.path.dirname(__file__), "..", "..", "build"),
        os.path.join(os.path.dirname(__file__), "..", "..", "build", "Release"),
        os.path.join(os.path.dirname(__file__), "..", "..", "build", "Debug"),
        ".",
    ]
    for d in search_dirs:
        if not d:
            continue
        for name in candidates:
            path = os.path.join(d, name) if not name.endswith((".dll", ".so", ".dylib")) else name
            try:
                return ctypes.CDLL(path)
            except OSError:
                try:
                    return ctypes.CDLL(name)
                except OSError:
                    continue
    return None


try:
    _LIB = _find_library()
except Exception:
    _LIB = None

_HAS_C_S5 = _LIB is not None


# ===========================================================================
#  Pure-Python re-implementation (mirrors s5_extensions.c)
# ===========================================================================


def _lower(text: str) -> str:
    return text.lower()


def _ml_jailbreak_patterns() -> List[str]:
    return ["ignora", "ignorer", "忽略", "무시", "ignore", "jailbreak", "dan", "override"]


def _py_ml_jailbreak_detect(text: str) -> int:
    low = _lower(text)
    for p in _ml_jailbreak_patterns():
        if p in low:
            return 1
    return 0


_B64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


def _b64_rev() -> Dict[int, int]:
    rev = {ord(c): i for i, c in enumerate(_B64_ALPHABET)}
    return rev


def _is_base64(s: str) -> bool:
    if not s:
        return False
    for c in s:
        if c == "=" or c.isalnum() or c in "+/":
            continue
        return False
    return True


def _is_hex(s: str) -> bool:
    if len(s) < 2:
        return False
    for c in s:
        if c not in "0123456789abcdefABCDEF":
            return False
    return True


def _b64_decode(inp: str) -> bytes:
    rev = _b64_rev()
    out = bytearray()
    val = 0
    valb = -8
    for c in inp:
        if c == "=":
            break
        d = rev.get(ord(c), -1)
        if d == -1:
            continue
        val = (val << 6) | d
        valb += 6
        if valb >= 0:
            out.append((val >> valb) & 0xFF)
            valb -= 8
    return bytes(out)


def _hex_decode(inp: str) -> bytes:
    out = bytearray()
    for i in range(0, len(inp) - 1, 2):
        v1 = int(inp[i], 16)
        v2 = int(inp[i + 1], 16)
        out.append((v1 << 4) | v2)
    return bytes(out)


def _rot13_decode(inp: str) -> str:
    out = []
    for c in inp:
        if "a" <= c <= "z":
            out.append(chr((ord(c) - ord("a") + 13) % 26 + ord("a")))
        elif "A" <= c <= "Z":
            out.append(chr((ord(c) - ord("A") + 13) % 26 + ord("A")))
        else:
            out.append(c)
    return "".join(out)


def _encoded_attack_decode(text: str, in_len: Optional[int] = None) -> str:
    n = in_len if in_len is not None else len(text)
    if n >= 2 and text[0] == "0" and text[1] in "xX":
        return _hex_decode(text[2:n]).decode("latin-1")
    if n % 4 == 0 and _is_base64(text[:n]):
        return _b64_decode(text[:n]).decode("latin-1")
    return _rot13_decode(text[:n])


_ENCODED_PATTERNS = [
    "drop table", "select * from", "delete from", "insert into",
    "exec ", "eval(", "os.system", "subprocess",
    "<?php", "<script>", "javascript:",
    "ignore all instructions", "ignore previous",
    "jailbreak", "override filter",
]


def _encoded_attack_scan(text: str) -> int:
    decoded = _encoded_attack_decode(text)
    low = _lower(decoded)
    for p in _ENCODED_PATTERNS:
        if p in low:
            return 1
    return 0


def _py_token_anomaly_score(probs: List[float]) -> float:
    if not probs:
        return 0.0
    total = 0.0
    for p in probs:
        p = max(1e-10, min(1.0, float(p)))
        total += -math.log(p)
    return total / len(probs)


def _py_model_inversion_apply(gradients: np.ndarray, noise_scale: float, clip_norm: float) -> np.ndarray:
    g = gradients.astype(np.float64).copy()
    norm = float(np.sqrt(np.sum(g * g)))
    if clip_norm > 0 and norm > clip_norm:
        g = g / norm * clip_norm
    rng = np.random.RandomState(0)
    g = g + (rng.rand(*g.shape) - 0.5) * noise_scale
    return g


def _py_membership_inference_defense(logits: np.ndarray, epsilon: float) -> np.ndarray:
    out = logits.astype(np.float64).copy()
    out = np.clip(out, -epsilon, epsilon)
    rng = np.random.RandomState(0)
    scale = 1.0 / epsilon if epsilon > 0 else 1.0
    u = rng.rand(*out.shape)
    lap = np.where(u < 0.5, scale * np.log(2 * u), -scale * np.log(2 * (1 - u)))
    return out + lap


def _py_data_extraction_prevent(text: str) -> int:
    n = len(text)
    i = 0
    # SSN-like patterns: dddd-dddd or ddd-dd-dddd
    while i + 13 < n:
        if (text[i:i + 4].isdigit() and text[i + 4] == "-" and text[i + 5:i + 9].isdigit()
                and text[i + 9] == "-" and text[i + 10:i + 14].isdigit()):
            return 1
        i += 1
    i = 0
    while i + 10 < n:
        if (text[i:i + 3].isdigit() and text[i + 3] == "-" and text[i + 4:i + 6].isdigit()
                and text[i + 6] == "-" and text[i + 7:i + 11].isdigit()):
            return 1
        i += 1
    low = _lower(text)
    for p in ["sk-", "pk-", "api_key", "apikey", "api-key", "akia", "asia",
               "ghp_", "gho_", "ghu_", "ghs_"]:
        if p in low:
            return 1
    return 0


def _py_training_sanitize(text: str) -> str:
    out = []
    i = 0
    n = len(text)
    redacted = "[REDACTED]"
    while i < n:
        if i + 3 < n and text[i:i + 4].lower() == "http":
            end = i
            while end < n and text[end] not in " \t\n\r,":
                end += 1
            out.append(redacted)
            i = end
            continue
        if text[i] == "@":
            start = i
            while start > 0 and text[start - 1] not in " \t\n\r,":
                start -= 1
            end = i
            while end < n and text[end] not in " \t\n\r,":
                end += 1
            out.append(redacted)
            i = end
            continue
        if i + 6 < n and text[i:i + 3].isdigit() and text[i + 3] == "." and text[i + 4:i + 7].isdigit():
            end = i
            while end < n and (text[end].isdigit() or text[end] == "."):
                end += 1
            out.append(redacted)
            i = end
            continue
        if i + 2 < n and text[i:i + 2].isdigit() and text[i + 2] == "-":
            end = i
            while end < n and (text[end].isdigit() or text[end] == "-"):
                end += 1
            out.append(redacted)
            i = end
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _py_factuality_score(statement: str, reference: str) -> float:
    if not statement or not reference or len(statement) < 2 or len(reference) < 2:
        return 0.5
    s_bi = set(statement[i:i + 2].lower() for i in range(len(statement) - 1))
    r_bi = set(reference[i:i + 2].lower() for i in range(len(reference) - 1))
    if not s_bi or not r_bi:
        return 0.5
    intersect = len(s_bi & r_bi)
    return (2.0 * intersect) / (len(s_bi) + len(r_bi))


def _py_bias_measure(predictions: List[float], sensitive: List[int]) -> Dict[str, float]:
    sum0 = sum1 = 0.0
    cnt0 = cnt1 = 0
    for p, s in zip(predictions, sensitive):
        if s == 0:
            sum0 += p
            cnt0 += 1
        else:
            sum1 += p
            cnt1 += 1
    mean0 = sum0 / cnt0 if cnt0 else 0.0
    mean1 = sum1 / cnt1 if cnt1 else 0.0
    return {
        "demographic_parity": mean1 - mean0,
        "equalized_odds": (mean1 - mean0) / (mean0 + 1e-10) if (mean0 > 0 and mean1 > 0) else 0.0,
    }


def _text_to_embedding(text: str) -> np.ndarray:
    h = np.zeros(8, dtype=np.float64)
    n = len(text)
    for i, c in enumerate(text):
        ci = ord(c) if ord(c) < 256 else 0
        h[i % 8] += ci
        h[(i + 1) % 8] += math.sin(ci) * (i + 1)
    h /= (n + 1)
    norm = float(np.sqrt(np.sum(h * h)))
    if norm > 1e-10:
        h /= norm
    return h


def _py_semantic_injection_score(detector_attacks: List[np.ndarray], embedding: np.ndarray, threshold: float) -> Tuple[int, float]:
    best = 0.0
    for atk in detector_attacks:
        dot = float(np.dot(embedding, atk))
        n1 = float(np.sqrt(np.sum(embedding * embedding)))
        n2 = float(np.sqrt(np.sum(atk * atk)))
        sim = dot / (n1 * n2 + 1e-10)
        best = max(best, sim)
    return (1 if best > threshold else 0, best)


def _py_watermark_key_from_bytes(key: bytes) -> np.ndarray:
    k = np.frombuffer(key[:32].ljust(32, b"\0"), dtype=np.uint8).astype(np.float64)
    return k


def _py_watermark_expected(weights: np.ndarray, key: np.ndarray) -> np.ndarray:
    i = np.arange(len(weights))
    return 0.001 * np.sin(key[i % len(key)] * i)


def _py_watermark_detect(weights: np.ndarray, key: np.ndarray) -> int:
    expected = _py_watermark_expected(weights, key)
    residual = weights - expected
    dot = float(np.dot(expected, residual))
    n1 = float(np.sqrt(np.sum(expected * expected)))
    n2 = float(np.sqrt(np.sum(residual * residual)))
    if n1 * n2 < 1e-10:
        return 0
    return 1 if (dot / (n1 * n2)) > 0.7 else 0


# ===========================================================================
#  ctypes wrapper (only populated when _LIB is available)
# ===========================================================================


def _ctypes_ml_jailbreak_detect(text: str) -> int:
    b = text.encode("utf-8")
    _LIB.SNEPPX_ml_jailbreak_detect.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    _LIB.SNEPPX_ml_jailbreak_detect.restype = ctypes.c_int
    return _LIB.SNEPPX_ml_jailbreak_detect(b, len(b))


def _ctypes_encoded_attack_scan(text: str) -> int:
    b = text.encode("utf-8")
    _LIB.SNEPPX_encoded_attack_scan.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    _LIB.SNEPPX_encoded_attack_scan.restype = ctypes.c_int
    return _LIB.SNEPPX_encoded_attack_scan(b, len(b))


def _ctypes_data_extraction_prevent(text: str) -> int:
    b = text.encode("utf-8")
    out = ctypes.c_int()
    _LIB.SNEPPX_data_extraction_prevent.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]
    _LIB.SNEPPX_data_extraction_prevent.restype = ctypes.c_int
    rc = _LIB.SNEPPX_data_extraction_prevent(b, len(b), ctypes.byref(out))
    return out.value if rc == 0 else 0


# ===========================================================================
#  Public bindings
# ===========================================================================


class S5PromptFilter:
    """Prompt-side safety: jailbreak, encoded-attack, and policy enforcement.

    Thin Python binding over the C ``SNEPPX_ml_jailbreak_*``,
    ``SNEPPX_encoded_attack_*`` and ``SNEPPX_prompt_policy_*`` functions.
    """

    def __init__(self):
        self.custom_patterns: List[str] = []

    def scan(self, text: str) -> str:
        """Return ``"clean"`` or ``"blocked"`` for a prompt."""
        if self._ml_jailbreak(text):
            return "blocked"
        if self._encoded_attack(text):
            return "blocked"
        low = _lower(text)
        for p in self.custom_patterns:
            if _lower(p) in low:
                return "blocked"
        return "clean"

    def add_pattern(self, pattern: str) -> None:
        if pattern not in self.custom_patterns:
            self.custom_patterns.append(pattern)

    def remove_pattern(self, pattern: str) -> None:
        if pattern in self.custom_patterns:
            self.custom_patterns.remove(pattern)

    def sanitize(self, text: str) -> str:
        """Prompt sanitization is detection-focused; the middleware regex
        layer performs redaction.  Return text unchanged here."""
        return text

    def pattern_count(self) -> int:
        return len(self.custom_patterns)

    def _ml_jailbreak(self, text: str) -> int:
        if _HAS_C_S5:
            return _ctypes_ml_jailbreak_detect(text)
        return _py_ml_jailbreak_detect(text)

    def _encoded_attack(self, text: str) -> int:
        if _HAS_C_S5:
            return _ctypes_encoded_attack_scan(text)
        return _encoded_attack_scan(text)


class S5OutputVerifier:
    """Output-side safety: PII / secret leakage prevention.

    Binding over ``SNEPPX_data_extraction_prevent`` and the training
    sanitization helpers.
    """

    def __init__(self):
        self.custom_rules: List[str] = []

    def check(self, text: str) -> str:
        """Return ``"clean"`` or ``"blocked"`` for generated output."""
        if _HAS_C_S5:
            if _ctypes_data_extraction_prevent(text):
                return "blocked"
        else:
            if _py_data_extraction_prevent(text):
                return "blocked"
        low = _lower(text)
        for r in self.custom_rules:
            if _lower(r) in low:
                return "blocked"
        return "clean"

    def sanitize(self, text: str) -> str:
        if _HAS_C_S5:
            return self._sanitize_ctypes(text)
        return _py_training_sanitize(text)

    def add_rule(self, rule: str) -> None:
        if rule not in self.custom_rules:
            self.custom_rules.append(rule)

    def remove_rule(self, rule: str) -> None:
        if rule in self.custom_rules:
            self.custom_rules.remove(rule)

    def rule_count(self) -> int:
        return len(self.custom_rules)

    def _sanitize_ctypes(self, text: str) -> str:
        b = text.encode("utf-8")
        buf = ctypes.create_string_buffer(len(b) + 1024)
        out_len = ctypes.c_size_t(len(buf))
        _LIB.SNEPPX_training_sanitize.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)
        ]
        _LIB.SNEPPX_training_sanitize.restype = ctypes.c_int
        _LIB.SNEPPX_training_sanitize(b, len(b), buf, ctypes.byref(out_len))
        return buf.value.decode("utf-8", "replace")


class S5RLHFSafety:
    """RLHF / safety metrics and defenses (factuality, bias, semantic
    injection, token anomaly, membership-inference & model-inversion
    defenses, watermarking).

    All functions are pure-Python bindings mirroring the C API; the ctypes
    path is used automatically when ``neural_security_c`` is built.
    """

    def __init__(self, watermark_key: Optional[bytes] = None):
        self._semantic_attacks: List[np.ndarray] = []
        self._semantic_threshold = 0.85
        self._token_anomaly_threshold = 3.0
        self._membership_epsilon = 1.0
        self._inv_noise = 0.01
        self._inv_clip = 1.0
        self._watermark_key = (
            np.frombuffer((watermark_key or b"sneppx-s5").ljust(32, b"\0")[:32], dtype=np.uint8).astype(np.float64)
        )

    # --- Factuality -------------------------------------------------------
    def factuality_score(self, statement: str, reference: str) -> float:
        if _HAS_C_S5:
            _LIB.SNEPPX_factuality_score.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            _LIB.SNEPPX_factuality_score.restype = ctypes.c_double
            return _LIB.SNEPPX_factuality_score(
                statement.encode("utf-8"), reference.encode("utf-8")
            )
        return _py_factuality_score(statement, reference)

    # --- Bias -------------------------------------------------------------
    def bias_measure(self, predictions: List[float], sensitive: List[int]) -> Dict[str, float]:
        return _py_bias_measure(predictions, sensitive)

    def bias_report(self, predictions: List[float], sensitive: List[int]) -> str:
        m = self.bias_measure(predictions, sensitive)
        return f"DP={m['demographic_parity']:.4f} EO={m['equalized_odds']:.4f}"

    # --- Semantic injection ------------------------------------------------
    def add_attack_text(self, text: str) -> None:
        self._semantic_attacks.append(_text_to_embedding(text))

    def clear_attacks(self) -> None:
        self._semantic_attacks = []

    def attack_count(self) -> int:
        return len(self._semantic_attacks)

    def semantic_injection_score(self, text: str) -> Tuple[int, float]:
        emb = _text_to_embedding(text)
        return _py_semantic_injection_score(self._semantic_attacks, emb, self._semantic_threshold)

    # --- Token anomaly ----------------------------------------------------
    def token_anomaly_score(self, probs: List[float]) -> float:
        return _py_token_anomaly_score(probs)

    def set_token_anomaly_threshold(self, t: float) -> None:
        self._token_anomaly_threshold = t

    def token_anomaly_check(self, probs: List[float]) -> int:
        return 1 if self.token_anomaly_score(probs) > self._token_anomaly_threshold else 0

    # --- Membership inference defense -------------------------------------
    def membership_inference_defense(self, logits: np.ndarray, epsilon: Optional[float] = None) -> np.ndarray:
        eps = self._membership_epsilon if epsilon is None else epsilon
        if _HAS_C_S5:
            arr = np.asarray(logits, dtype=np.float64).reshape(-1)
            out = np.empty_like(arr)
            _LIB.SNEPPX_membership_inference_defense_apply.argtypes = [
                ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_double,
                ctypes.POINTER(ctypes.c_double),
            ]
            _LIB.SNEPPX_membership_inference_defense_apply.restype = ctypes.c_int
            _LIB.SNEPPX_membership_inference_defense_apply(
                arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                arr.size, eps,
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
            return out.reshape(np.asarray(logits).shape)
        return _py_membership_inference_defense(np.asarray(logits, dtype=np.float64), eps)

    # --- Model inversion defense ------------------------------------------
    def model_inversion_defense(self, gradients: np.ndarray, noise: Optional[float] = None,
                                clip: Optional[float] = None) -> np.ndarray:
        g = np.asarray(gradients, dtype=np.float64)
        noise = self._inv_noise if noise is None else noise
        clip = self._inv_clip if clip is None else clip
        if _HAS_C_S5:
            out = np.empty_like(g)
            _LIB.SNEPPX_model_inversion_defend_gradients.argtypes = [
                ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_double, ctypes.c_double
            ]
            _LIB.SNEPPX_model_inversion_defend_gradients.restype = ctypes.c_int
            _LIB.SNEPPX_model_inversion_defend_gradients(
                g.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                g.size, noise, clip,
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
            return out
        return _py_model_inversion_apply(g, noise, clip)

    # --- Watermarking -----------------------------------------------------
    def watermark_embed(self, weights: np.ndarray) -> np.ndarray:
        w = np.asarray(weights, dtype=np.float64).copy()
        i = np.arange(w.size)
        w += 0.001 * np.sin(self._watermark_key[i % self._watermark_key.size] * i)
        return w

    def watermark_detect(self, weights: np.ndarray) -> int:
        return _py_watermark_detect(np.asarray(weights, dtype=np.float64), self._watermark_key)

    # --- Adversarial smoothing --------------------------------------------
    def adversarial_smooth(self, inp: np.ndarray, epsilon: float) -> np.ndarray:
        x = np.asarray(inp, dtype=np.float64).copy()
        rng = np.random.RandomState(0)
        x += (rng.rand(*x.shape) - 0.5) * 2.0 * epsilon
        return x
