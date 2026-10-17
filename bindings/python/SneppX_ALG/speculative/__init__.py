"""Speculative decoding utilities: draft generators, verification, decoding loop."""

from . import decoding, draft, verify
from .decoding import (
    CategoricalSampler,
    SpeculativeDecoder,
    SpeculativeResult,
    TokenSampler,
)
from .draft import NGramDrafter, PromptLookupDrafter
from .verify import (
    acceptance_probability,
    residual_distribution,
    speculative_accept,
    verify_greedy,
)

__all__ = [
    "draft",
    "verify",
    "decoding",
    "NGramDrafter",
    "PromptLookupDrafter",
    "acceptance_probability",
    "residual_distribution",
    "speculative_accept",
    "verify_greedy",
    "CategoricalSampler",
    "SpeculativeResult",
    "SpeculativeDecoder",
    "TokenSampler",
]
