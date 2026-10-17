"""A tiny speculative decoding loop skeleton (pure Python)."""

from dataclasses import dataclass
from typing import List, Optional, Protocol

import numpy as np

from .verify import residual_distribution, speculative_accept


class Drafter(Protocol):
    def draft(self, prefix: List[int], k: int) -> List[int]: ...


class Target(Protocol):
    def sample(self, prefix: List[int], num: int) -> List[int]: ...

    def log_probs(self, prefix: List[int], tokens: List[int]) -> List[np.ndarray]: ...


class TokenSampler(Protocol):
    def sample(self, logits_or_probs: np.ndarray) -> int: ...


class CategoricalSampler:
    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()

    def sample(self, probs: np.ndarray) -> int:
        p = np.asarray(probs, dtype=np.float64)
        if p.sum() <= 0:
            p = np.ones_like(p) / p.size
        else:
            p = p / p.sum()
        return int(self.rng.choice(len(p), p=p))


@dataclass
class SpeculativeResult:
    accepted: List[int]
    generated: List[int]
    new_tokens: List[int]
    steps: int


class SpeculativeDecoder:
    """Greedy-approximate skeleton; good for API shape and unit tests."""

    def __init__(
        self,
        drafter: Drafter,
        target: Target,
        sampler: Optional[TokenSampler] = None,
        max_draft: int = 4,
    ):
        self.drafter = drafter
        self.target = target
        self.sampler = sampler or CategoricalSampler()
        self.max_draft = max_draft

    def decode(self, prefix: List[int], max_new_tokens: int) -> SpeculativeResult:
        generated = list(prefix)
        accepted_total: List[int] = []
        steps = 0
        while len(generated) < len(prefix) + max_new_tokens:
            steps += 1
            draft = self.drafter.draft(generated, k=self.max_draft)
            if not draft:
                tgt = self.target.sample(generated, num=1)
                generated.extend(tgt)
                accepted_total.extend(tgt)
                break
            target_greedy = self.target.sample(generated, num=len(draft))
            m = 0
            for d, t in zip(draft, target_greedy):
                if d != t:
                    break
                m += 1
            if m < len(draft):
                generated.extend(target_greedy[: m + 1])
                accepted_total.extend(target_greedy[: m + 1])
            else:
                generated.extend(target_greedy)
                accepted_total.extend(target_greedy)
                generated.append(target_greedy[-1])
                accepted_total.append(target_greedy[-1])
                break
            if len(accepted_total) >= max_new_tokens:
                break
        new_tokens = generated[len(prefix) :]
        if len(new_tokens) > max_new_tokens:
            new_tokens = new_tokens[:max_new_tokens]
        return SpeculativeResult(
            accepted=list(accepted_total),
            generated=list(generated)[: len(prefix) + len(new_tokens)],
            new_tokens=new_tokens,
            steps=steps,
        )
