"""Model Watermarking & Fingerprinting.

Three complementary ownership / provenance mechanisms:

* :class:`ModelHasher` — deterministic content hash (optionally HMAC-signed)
  over all model weights for tamper-evident fingerprints.
* :class:`WeightWatermark` — embeds a secret key-derived pattern into a
  small fraction of model weights; detectable later without retraining.
* :class:`BackdoorWatermark` — trigger-based watermark trained into the
  model (Adi et al., 2018 style): key inputs mapped to a fixed target.
"""

import hashlib
import hmac
import numpy as np
from typing import Optional, List, Tuple, Callable, Any

from .tensor import Tensor
from .nn import Module
from .optim import Optimizer


# ===========================================================================
#  Deterministic key derivation
# ===========================================================================


def _key_seed(key: str) -> int:
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little") % (2**32)


def _seeded_rng(key: str, salt: str = "") -> np.random.RandomState:
    seed = _key_seed(key + "::" + salt)
    return np.random.RandomState(seed)


# ===========================================================================
#  Model hash / fingerprint
# ===========================================================================


class ModelHasher:
    """Content fingerprint over all model parameters.

    The fingerprint is a deterministic hash of the model's weights when
    ``secret`` is ``None``.  With a ``secret``, an HMAC-SHA256 signature is
    produced instead, allowing verification that a fingerprint was issued by a
    holder of the secret.
    """

    def __init__(self, secret: Optional[str] = None):
        self.secret = secret

    def fingerprint(self, model: Module) -> str:
        parts = []
        for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
            arr = np.asarray(p.data, dtype=np.float32).ravel()
            parts.append(name.encode("utf-8"))
            parts.append(arr.tobytes())
        blob = b"".join(parts)
        if self.secret is None:
            return hashlib.sha256(blob).hexdigest()
        return hmac.new(
            self.secret.encode("utf-8"), blob, hashlib.sha256
        ).hexdigest()

    def verify(self, model: Module, signature: str) -> bool:
        if self.secret is None:
            raise ValueError("Cannot verify without a secret key.")
        return hmac.compare_digest(self.fingerprint(model), signature)


# ===========================================================================
#  Weight watermark (post-training, no retrain)
# ===========================================================================


class WeightWatermark:
    """Embed a key-derived pattern in a fraction of the model's weights.

    The pattern is selected deterministically from ``key`` so it can be
    reproduced at detection time.  Embedding perturbs ``ratio`` of entries in
    each targeted parameter toward ``strength * pattern`` where ``pattern`` is
    a ±1 signal derived from the key.  Detection recomputes the expected
    pattern at the same positions and measures agreement.
    """

    def __init__(self, key: str, strength: float = 0.05, ratio: float = 0.05):
        if not (0.0 < ratio <= 1.0):
            raise ValueError("ratio must be in (0, 1]")
        if strength <= 0:
            raise ValueError("strength must be > 0")
        self.key = key
        self.strength = strength
        self.ratio = ratio

    def _select(self, param: Tensor, salt: str) -> np.ndarray:
        arr = np.asarray(param.data, dtype=np.float64)
        rng = _seeded_rng(self.key, salt=salt + f"::{arr.shape}")
        n = arr.size
        k = max(1, int(round(n * self.ratio)))
        idx = rng.choice(n, size=k, replace=False)
        pattern = rng.choice([-1.0, 1.0], size=k)
        return idx, pattern

    def embed(self, model: Module) -> dict:
        info = {}
        for name, p in model.named_parameters():
            arr = np.array(p.data, dtype=np.float64)
            idx, pattern = self._select(p, salt=name)
            arr.flat[idx] = self.strength * pattern
            p.data = arr.astype(np.float32)
            info[name] = {"selected": int(idx.size), "total": int(arr.size)}
        return info

    def detect(self, model: Module, threshold: float = 0.5) -> Tuple[bool, float]:
        """Return ``(is_watermarked, score)``.

        ``score`` is the mean Pearson correlation between the key pattern and
        the observed weight values at the key-selected positions.  A
        watermarked model scores ~1.0; an unwatermarked (random) model
        scores ~0.0, giving a clean separation.
        """
        scores = []
        for name, p in model.named_parameters():
            arr = np.asarray(p.data, dtype=np.float64)
            idx, pattern = self._select(p, salt=name)
            if idx.size < 2:
                # Too few points for a meaningful statistic; skip.
                continue
            observed = arr.flat[idx]
            if observed.std() < 1e-12:
                scores.append(0.0)
                continue
            r = np.corrcoef(observed, pattern)[0, 1]
            scores.append(0.0 if np.isnan(r) else max(0.0, r))
        score = float(np.mean(scores)) if scores else 0.0
        return (score >= threshold, score)


# ===========================================================================
#  Backdoor / trigger watermark (trained)
# ===========================================================================


class BackdoorWatermark:
    """Train a trigger-based watermark into a model.

    A set of ``num_keys`` random "key" inputs are generated from ``key`` and
    mapped to a single ``target_label``.  Ownership is verified by querying
    the model with the regenerated keys and checking the fraction that is
    classified as ``target_label`` exceeds ``threshold``.
    """

    def __init__(
        self,
        key: str,
        input_shape: tuple,
        target_label: int,
        num_keys: int = 100,
        threshold: float = 0.9,
    ):
        if num_keys <= 0:
            raise ValueError("num_keys must be > 0")
        self.key = key
        self.input_shape = tuple(input_shape)
        self.target_label = int(target_label)
        self.num_keys = num_keys
        self.threshold = threshold
        self._keys = self._make_keys()

    def _make_keys(self) -> List[Tensor]:
        rng = _seeded_rng(self.key, salt="keys")
        keys = []
        for _ in range(self.num_keys):
            arr = rng.randn(*self.input_shape).astype(np.float32)
            keys.append(Tensor(arr))
        return keys

    def train(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_fn: Callable[[Any, Any], Tensor],
        epochs: int = 5,
    ) -> List[float]:
        """Train the model to map keys -> target_label. Returns loss history."""
        history: List[float] = []
        labels = Tensor(
            np.full(self.num_keys, self.target_label, dtype=np.int64)
        )
        for _ in range(epochs):
            optimizer.zero_grad()
            for p in model.parameters():
                if p.grad is not None:
                    p.zero_grad_()
            # Forward all keys in one batch (stack to (num_keys, *shape)).
            batch = self._stack_keys()
            out = model(batch)
            loss = loss_fn(out, labels)
            if not isinstance(loss, Tensor):
                loss = Tensor(np.asarray(loss, dtype=np.float32))
            loss.backward()
            optimizer.step()
            history.append(
                float(loss.data.item())
                if hasattr(loss.data, "item")
                else float(loss.data)
            )
        return history

    def _stack_keys(self) -> Tensor:
        arrs = [np.asarray(k.data) for k in self._keys]
        return Tensor(np.stack(arrs, axis=0))

    def verify(self, model: Module) -> Tuple[bool, float]:
        """Return ``(is_watermarked, match_fraction)``."""
        was_training = getattr(model, "training", True)
        if hasattr(model, "eval"):
            model.eval()
        try:
            logits = model(self._stack_keys())
            if isinstance(logits, dict):
                logits = logits.get("logits", logits)
            arr = np.asarray(logits.data)
            if arr.ndim == 3:
                arr = arr[:, -1, :]
            preds = np.argmax(arr, axis=-1)
            frac = float(np.mean(preds == self.target_label))
        finally:
            if hasattr(model, "train") and was_training:
                model.train()
        return (frac >= self.threshold, frac)
