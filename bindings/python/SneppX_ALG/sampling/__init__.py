"""SNEPPX sampling — decoding-time logit processors and samplers.

Pure-NumPy helpers for temperature, top-k/top-p/min-p filtering, repetition
penalty, Gumbel sampling, categorical sampling and beam search::

    from SneppX_ALG import sampling

    logits = sampling.filter_logits(raw, temp=0.7, k=40, p=0.9)
    tok = sampling.greedy_sample(logits)
"""

from . import beam, gumbel, logits, samplers
from .beam import beam_search
from .gumbel import (
    gumbel_max_sample,
    gumbel_noise,
    gumbel_softmax,
    straight_through,
)
from .logits import (
    filter_logits,
    min_p_filter,
    repetition_penalty,
    softmax,
    temperature,
    top_k_filter,
    top_p_filter,
)
from .samplers import (
    Sampler,
    SamplerConfig,
    greedy_sample,
    sample_categorical,
    sample_tokens,
)

__all__ = [
    "logits",
    "samplers",
    "gumbel",
    "beam",
    "temperature",
    "top_k_filter",
    "top_p_filter",
    "min_p_filter",
    "repetition_penalty",
    "filter_logits",
    "softmax",
    "gumbel_noise",
    "gumbel_softmax",
    "gumbel_max_sample",
    "straight_through",
    "greedy_sample",
    "sample_categorical",
    "sample_tokens",
    "SamplerConfig",
    "Sampler",
    "beam_search",
]
