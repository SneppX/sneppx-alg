"""Sequence-to-sequence / text metrics (pure NumPy).

BLEU, ROUGE, edit distance, word/character error rate, chrF and exact match.
All operate on plain token lists or strings.
"""

import math
from collections import Counter
from typing import List, Sequence

import numpy as np


def _tokens(x) -> List[str]:
    if isinstance(x, str):
        return x.split()
    return list(x)


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein distance between two sequences."""
    a, b = list(a), list(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def sentence_bleu(
    reference: Sequence[str], hypothesis: Sequence[str], max_n: int = 4
) -> float:
    """Sentence (smoothed) BLEU-``max_n`` with brevity penalty."""
    ref = _tokens(reference)
    hyp = _tokens(hypothesis)
    precisions = [1.0] * max_n
    for n in range(1, max_n + 1):
        h = _ngrams(hyp, n)
        if not h:
            precisions[n - 1] = 1.0
            continue
        r = _ngrams(ref, n)
        matches = sum((h & r).values())
        precisions[n - 1] = (matches + 1e-9) / (sum(h.values()) + 1e-9)
    if any(p == 0 for p in precisions):
        return 0.0
    bp = 1.0
    if len(hyp) < len(_tokens(reference)):
        bp = math.exp(1.0 - len(_tokens(reference)) / max(len(hyp), 1))
    return float(bp * math.exp(np.mean(np.log(precisions))))


def corpus_bleu(
    references: Sequence[Sequence[str]],
    hypotheses: Sequence[Sequence[str]],
    max_n: int = 4,
) -> float:
    """Corpus-level BLEU-``max_n`` over parallel reference/hypothesis lists."""
    matches = np.zeros(max_n, dtype=np.float64)
    totals = np.zeros(max_n, dtype=np.float64)
    hyp_len = 0
    ref_len = 0
    for ref, hyp in zip(references, hypotheses):
        r = _tokens(ref)
        h = _tokens(hyp)
        hyp_len += len(h)
        ref_len += len(r)
        for n in range(1, max_n + 1):
            hn = _ngrams(h, n)
            rn = _ngrams(r, n)
            totals[n - 1] += sum(hn.values())
            matches[n - 1] += sum((hn & rn).values())
    if hyp_len == 0:
        return 0.0
    precisions = np.where(totals > 0, matches / np.maximum(totals, 1), 1.0)
    if np.any(precisions == 0):
        return 0.0
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / hyp_len)
    return float(bp * math.exp(np.mean(np.log(precisions))))


def rouge_n(reference: Sequence[str], hypothesis: Sequence[str], n: int = 1) -> float:
    """ROUGE-N recall of reference n-grams covered by the hypothesis."""
    ref = _ngrams(_tokens(reference), n)
    hyp = _ngrams(_tokens(hypothesis), n)
    total = sum(ref.values())
    if total == 0:
        return 0.0
    return float(sum((ref & hyp).values()) / total)


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    dp = [0] * (len(b) + 1)
    for ca in a:
        prev = 0
        for j, cb in enumerate(b, 1):
            tmp = dp[j]
            dp[j] = prev + 1 if ca == cb else max(dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]


def rouge_l(reference: Sequence[str], hypothesis: Sequence[str], beta: float = 1.2):
    """ROUGE-L F-measure based on the longest common subsequence."""
    r = _tokens(reference)
    h = _tokens(hypothesis)
    if not r or not h:
        return {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
    lcs = _lcs(r, h)
    prec = lcs / len(h)
    rec = lcs / len(r)
    if prec + rec == 0:
        f = 0.0
    else:
        f = (1 + beta**2) * prec * rec / (rec + beta**2 * prec)
    return {"precision": float(prec), "recall": float(rec), "fmeasure": float(f)}


def word_error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    ref = _tokens(reference)
    hyp = _tokens(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(edit_distance(ref, hyp) / len(ref))


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref = list(reference)
    hyp = list(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(edit_distance(ref, hyp) / len(ref))


def chrf(reference: str, hypothesis: str, n: int = 6, beta: float = 2.0) -> float:
    """chrF: character n-gram F-score averaged over ``n = 1..n``."""
    scores = []
    for k in range(1, n + 1):
        ref = _ngrams(list(reference), k)
        hyp = _ngrams(list(hypothesis), k)
        total_ref = sum(ref.values())
        total_hyp = sum(hyp.values())
        if total_ref == 0 and total_hyp == 0:
            continue
        if total_ref == 0 or total_hyp == 0:
            scores.append(0.0)
            continue
        match = sum((ref & hyp).values())
        prec = match / total_hyp
        rec = match / total_ref
        if prec + rec == 0:
            scores.append(0.0)
        else:
            scores.append((1 + beta**2) * prec * rec / (rec + beta**2 * prec))
    return float(np.mean(scores)) if scores else 0.0


def exact_match(reference, hypothesis) -> float:
    return 1.0 if _tokens(reference) == _tokens(hypothesis) else 0.0
