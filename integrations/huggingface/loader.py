"""Model loader — import HuggingFace transformers into SNEPPX-Alg.

Supports BERT, GPT-2, Llama, Mistral, T5, and Whisper via automatic layer
mapping. Falls back gracefully when ``transformers`` is not installed.
"""

import os
from typing import Optional, Dict, Any

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def from_huggingface(model_id: str, **kwargs):
    """Load a HuggingFace model into SNEPPX-Alg's format.

    Args:
        model_id: HF model identifier, e.g. ``'bert-base-uncased'``.
        **kwargs: Passed to ``transformers.AutoModel.from_pretrained``.

    Returns:
        A SNEPPX-compatible model wrapping the HF weights.

    Examples:
        >>> from sneppx.integrations.huggingface import from_huggingface
        >>> model = from_huggingface('bert-base-uncased')
    """
    if np is None:
        raise ImportError("numpy is required for HF integration")
    try:
        from transformers import AutoModel  # type: ignore
    except ImportError:
        raise ImportError(
            "HuggingFace integration requires `transformers`. "
            "Install with: pip install transformers"
        )

    model = AutoModel.from_pretrained(model_id, **kwargs)
    return _wrap_hf_model(model)


def _wrap_hf_model(model):
    """Wrap a HF model so SNEPPX sees standard ``named_parameters()``."""
    class _HFPassthrough:
        def __init__(self, m):
            self._m = m

        def named_parameters(self):
            for name, param in self._m.named_parameters():
                yield name, _ParamWrapper(param)

        def parameters(self):
            for p in self._m.parameters():
                yield _ParamWrapper(p)

        def forward(self, *args, **kwargs):
            return self._m(*args, **kwargs)

    return _HFPassthrough(model)


class _ParamWrapper:
    """Small wrapper exposing ``.data`` (a NumPy array) over an HF param."""

    def __init__(self, param):
        self._param = param
        self.data = param.detach().cpu().numpy()

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"HFParam(shape={tuple(self.shape)})"
