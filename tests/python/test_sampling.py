import numpy as np

from SneppX_ALG import sampling as sp
from SneppX_ALG.sampling import logits as L


def test_temperature():
    x = np.array([1.0, 2.0])
    assert np.allclose(L.temperature(x, 0.5), [2.0, 4.0])


def test_top_k_filter():
    x = np.array([0.1, 0.9, 0.2, 0.8])
    out = L.top_k_filter(x, 2)
    finite = np.isfinite(out)
    assert finite.sum() == 2
    assert set(np.flatnonzero(finite).tolist()) == {1, 3}


def test_top_p_filter():
    x = np.array([10.0, 0.0, 0.0])
    out = L.top_p_filter(x, 0.5)
    assert np.isfinite(out[0])
    assert np.isneginf(out[1]) and np.isneginf(out[2])


def test_min_p_filter():
    x = np.array([10.0, 0.0, 0.0])
    out = L.min_p_filter(x, 0.5)
    assert np.isfinite(out[0])
    assert np.isneginf(out[1])


def test_repetition_penalty():
    x = np.array([2.0, -2.0, 1.0])
    out = L.repetition_penalty(x, [0, 1], 2.0)
    assert abs(out[0] - 1.0) < 1e-9
    assert abs(out[1] - (-4.0)) < 1e-9
    assert abs(out[2] - 1.0) < 1e-9


def test_filter_logits_and_softmax():
    x = np.array([3.0, 2.0, 1.0, 0.0])
    out = L.filter_logits(x, temp=0.5, k=2)
    assert np.isfinite(out[:2]).all()
    assert np.isneginf(out[2:]).all()
    p = L.softmax(x)
    assert abs(p.sum() - 1.0) < 1e-12


def test_greedy_and_categorical():
    assert L.softmax([0, 100]).argmax() == 1
    assert sp.greedy_sample([0.1, 5.0, 0.2]) == 1
    s = sp.sample_categorical([0.0, 10.0, 0.0], num_samples=3, seed=0)
    assert list(s) == [1, 1, 1]


def test_gumbel_softmax():
    x = np.array([0.1, 0.2, 5.0])
    hard = sp.gumbel_softmax(x, tau=0.5, hard=True, seed=0)
    assert np.array_equal(hard, np.array([0.0, 0.0, 1.0]))
    soft = sp.gumbel_softmax(x, tau=1.0, hard=False, seed=0)
    assert abs(soft.sum() - 1.0) < 1e-9
    assert sp.gumbel_max_sample(np.array([0.0, 100.0]), seed=0) == 1


def test_sampler_config():
    s = sp.Sampler(greedy=True)
    assert s.sample([0.1, 0.9, 0.2]) == 1
    s2 = sp.Sampler(top_k=1, seed=0)
    assert s2.sample([0.1, 0.9, 0.2]) == 1
    cfg = sp.SamplerConfig(temperature=0.8, top_k=5, top_p=0.9, seed=0)
    probs = sp.Sampler(cfg).probs([1.0, 2.0, 3.0])
    assert abs(probs.sum() - 1.0) < 1e-9


def test_beam_search():
    target = [1, 2, 3]

    def step(prefix):
        lp = np.full(5, -10.0)
        i = len(prefix) - 1
        if i < len(target):
            lp[target[i]] = 0.0
        else:
            lp[0] = 0.0
        return lp

    tokens, score = sp.beam_search(step, bos=0, beam_width=2, max_len=3)
    assert tokens == [1, 2, 3]
