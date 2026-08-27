"""Deterministic seeding utilities for reproducible SNEPPX-Alg runs."""
import os
import random

try:
    import numpy as np
except Exception:
    np = None

__all__ = ["set_global_seed", "make_rng"]

def set_global_seed(seed: int) -> None:
    """Seed Python's global RNG and (if available) NumPy for reproducibility."""
    if not isinstance(seed, int):
        raise TypeError("seed must be an int")
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    os.environ["SNEAPX_SEED"] = str(seed)

def make_rng(seed: int):
    """Return a fresh, independent ``random.Random`` instance."""
    return random.Random(seed)