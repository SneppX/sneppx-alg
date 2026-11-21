"""Pipeline API mirror — `sneppx.pipeline` mirroring HuggingFace transformers.pipeline."""

from typing import Optional, List, Dict, Any

_SUPPORTED = ["text-generation", "text-classification", "fill-mask", "feature-extraction", "summarization"]


def pipeline(task: str, model=None, tokenizer=None, **kwargs):
    """Create a pipeline mirroring ``transformers.pipeline``.

    Args:
        task: Task name. One of: text-generation, text-classification,
            fill-mask, feature-extraction, summarization.
        model: Optional model. If ``None``, a lightweight default model is
            attempted via ``from_huggingface``.
        tokenizer: Optional tokenizer instance.
        **kwargs: Extra pipeline options (device, batch_size, max_length).

    Returns:
        A Pipeline callable.
    """
    if task not in _SUPPORTED:
        raise ValueError(
            f"Unsupported task {task!r}. Supported: {_SUPPORTED}"
        )

    if model is None:
        model = _default_model(task)

    return Pipeline(task=task, model=model, tokenizer=tokenizer, **kwargs)


def _default_model(task: str):
    default_ids = {
        "text-generation": "gpt2",
        "text-classification": "distilbert-base-uncased-finetuned-sst-2-english",
        "fill-mask": "bert-base-uncased",
        "feature-extraction": "bert-base-uncased",
        "summarization": "t5-small",
    }
    model_id = default_ids[task]
    try:
        from .loader import from_huggingface
        return from_huggingface(model_id)
    except ImportError:
        return None


class Pipeline:
    """Callable pipeline object mirroring ``transformers.Pipeline``."""

    def __init__(self, task: str, model=None, tokenizer=None, **kwargs):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        self.device = kwargs.get("device", -1)
        self.batch_size = kwargs.get("batch_size", None)
        self.max_length = kwargs.get("max_length", None)

    def __call__(self, inputs, **kwargs):
        if isinstance(inputs, str):
            inputs = [inputs]
        return [self._run_one(i, **kwargs) for i in inputs]

    def _run_one(self, text: str, **kwargs):
        max_length = kwargs.get("max_length", self.max_length)
        if self.task == "text-generation":
            return self._generate(text, max_length)
        if self.task == "feature-extraction":
            return self._extract(text)
        if self.task in ("text-classification", "fill-mask", "summarization"):
            raise NotImplementedError(
                f"{self.task!r} requires a full model forward pass; "
                "see docs/migration-guide.md for wiring."
            )
        return {"input": text}

    def _generate(self, text: str, max_length: Optional[int]):
        # Minimal greedy demo path when a HF model with generate() is wrapped.
        gen = getattr(self.model, "_m", None)
        if gen is not None and hasattr(gen, "generate"):
            out = gen.generate(
                gen_tokenize(gen, text), max_length=max_length or 20
            )
            return {"generated_text": _decode(gen, out)}
        return {"generated_text": f"{text} (generation stub)"}

    def _extract(self, text: str):
        backend = getattr(self.model, "_m", None)
        if backend is not None and hasattr(backend, "forward"):
            import numpy as np
            tokens = gen_tokenize(backend, text)
            with _torch_inference_mode():
                last_hidden = backend(input_ids=tokens).last_hidden_state
            return {"features": np.asarray(last_hidden.detach().cpu().numpy())}
        return {"features": None}

    def __repr__(self):
        return f"Pipeline(task={self.task!r})"


# --- tiny helpers that avoid hard `torch` import when not needed -------------

def gen_tokenize(model, text: str):
    try:
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(model.config._name_or_path)
        return tok(text, return_tensors="pt").input_ids
    except Exception:
        raise RuntimeError("A tokenizer is required for pipeline tasks; pass tokenizer=...")


def _decode(model, token_ids):
    try:
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(model.config._name_or_path)
        return tok.decode(token_ids[0], skip_special_tokens=True)
    except Exception:
        return str(list(token_ids))


def _torch_inference_mode():
    try:
        import torch
        return torch.inference_mode()
    except ImportError:
        class _NoOp:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
        return _NoOp()
