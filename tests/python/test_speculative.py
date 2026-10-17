from SneppX_ALG import speculative as sp

import numpy as np


def test_ngram_drafter():
    corpus = [
        [1, 2, 3, 4, 1, 2, 3, 5],
        [1, 2, 3, 6],
    ]
    drafter = sp.NGramDrafter(n=3, max_draft=4).fit(corpus)
    assert drafter.draft([1, 2, 3]) in ([4], [4, 1], [4, 1, 2], [4, 1, 2, 3])
    assert sp.NGramDrafter(n=1).draft([10]) == []


def test_prompt_lookup():
    p = [1, 2, 3, 1, 2, 4]
    d = sp.PromptLookupDrafter(n=2, max_draft=3)
    assert d.draft(p, k=2) == []


def test_acceptance_and_residual():
    pd = np.array([0.8, 0.2])
    pt = np.array([0.4, 0.6])
    assert abs(sp.acceptance_probability(pd, pt, 0) - 0.5) < 1e-12
    assert sp.acceptance_probability(pd, pt, 1) == 1.0
    resid = sp.residual_distribution(pd, pt)
    assert abs(resid.sum() - 1.0) < 1e-12


def test_verify_greedy():
    assert sp.verify_greedy([1, 2, 3], [1, 2, 4]) == 2
    assert sp.verify_greedy([1, 2], [1, 2, 3]) == 2


def test_speculative_accept_stops():
    rng = np.random.default_rng(0)
    draft = [1, 0]
    pd = [np.array([0.1, 0.9]), np.array([0.9, 0.1])]
    pt = [np.array([0.2, 0.8]), np.array([0.1, 0.9])]
    acc = sp.speculative_accept(draft, pd, pt, rng=rng)
    assert 0 <= len(acc) <= 2


class DummyTarget:
    def sample(self, prefix, num=1):
        return list(range(num))

    def log_probs(self, prefix, tokens):
        return [np.array([1.0, 0.0])] * len(tokens)


def test_speculative_decoder():
    d = sp.NGramDrafter(n=1, max_draft=2).fit([[5, 6, 7]])
    dec = sp.SpeculativeDecoder(drafter=d, target=DummyTarget(), max_draft=2)
    res = dec.decode([5], max_new_tokens=3)
    assert res.new_tokens
    assert len(res.new_tokens) <= 3
