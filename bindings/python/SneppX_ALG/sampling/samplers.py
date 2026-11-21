"""Categorical samplers and a configurable decoding sampler (pure NumPy)."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .logits import filter_logits, repetition_penalty, softmax


def greedy_sample(logits) -> int:
    """Return the index of the maximum logit (single vector)."""
    x = np.asarray(logits, dtype=np.float64)
    return int(np.argmax(x))


def sample_categorical(logits, num_samples: int = 1, seed: Optional[int] = None):
    """Draw ``num_samples`` categorical samples from ``logits`` (softmax)."""
    probs = softmax(logits)
    rng = np.random.default_rng(seed)
    return rng.choice(probs.shape[-1], size=num_samples, p=probs)


@dataclass
class SamplerConfig:
    """Decoding hyperparameters."""

    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    greedy: bool = False
    seed: Optional[int] = None


class Sampler:
    """Reusable sampler applying temperature/top-k/top-p/min-p then drawing."""

    def __init__(self, config: Optional[SamplerConfig] = None, **kwargs):
        if config is None:
            config = SamplerConfig(**kwargs)
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._step = 0

    def probs(self, logits, prev_ids=None) -> np.ndarray:
        c = self.config
        x = filter_logits(
            logits, temp=c.temperature, k=c.top_k, p=c.top_p, min_p=c.min_p
        )
        if c.repetition_penalty and prev_ids is not None:
            x = repetition_penalty(x, prev_ids, c.repetition_penalty)
        return softmax(x)

    def sample(self, logits, prev_ids=None) -> int:
        if self.config.greedy:
            return greedy_sample(logits)
        probs = self.probs(logits, prev_ids)
        self._step += 1
        return int(self._rng.choice(probs.shape[-1], p=probs))

    def sample_many(self, logits, num_samples: int = 1, prev_ids=None):
        probs = self.probs(logits, prev_ids)
        return self._rng.choice(probs.shape[-1], size=num_samples, p=probs)


def sample_tokens(logits, temperature: float = 1.0, seed: Optional[int] = None):
    """Convenience: temperature-scaled categorical sample (int)."""
    x = np.asarray(logits, dtype=np.float64) / max(temperature, 1e-10)
    return int(sample_categorical(x, 1, seed=seed)[0])
