"""Generation — autoregressive text generation with sampling, beam search, and streaming."""

from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict, Any, Iterator, Union
import numpy as np

from .tensor import Tensor
from .nn import Module

# ===========================================================================
#  Generation Config
# ===========================================================================


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    num_beams: int = 1
    pad_token_id: int = 0
    eos_token_id: int = 2
    bos_token_id: int = 1
    stop_strings: Optional[List[str]] = None
    early_stopping: bool = False
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0


# ===========================================================================
#  Logits Processing
# ===========================================================================


def top_k_top_p_filtering(
    logits: np.ndarray,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> np.ndarray:
    """Filter logits using top-k and/or top-p (nucleus) filtering.

    Args:
        logits: shape (batch, vocab_size) logits array.
        top_k: keep only top-k tokens (0 = disabled).
        top_p: keep tokens with cumulative probability >= top_p (1.0 = disabled).
        filter_value: value to assign to filtered-out positions.
        min_tokens_to_keep: minimum number of tokens to keep (even if below threshold).
    Returns:
        Filtered logits of the same shape.
    """
    logits = logits.copy()
    if top_k > 0:
        top_k = max(top_k, min_tokens_to_keep)
        indices_to_remove = (
            logits < np.partition(logits, -top_k, axis=-1)[:, -top_k : -top_k + 1]
        )
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_indices = np.argsort(logits, axis=-1)[:, ::-1]
        sorted_logits = np.take_along_axis(logits, sorted_indices, axis=-1)
        sorted_probs = np.exp(sorted_logits - sorted_logits.max(axis=-1, keepdims=True))
        sorted_probs = sorted_probs / sorted_probs.sum(axis=-1, keepdims=True)
        cumsum = np.cumsum(sorted_probs, axis=-1)

        remove_mask = cumsum - sorted_probs > top_p
        sorted_logits[remove_mask] = filter_value
        reverse_indices = np.argsort(sorted_indices, axis=-1)
        logits = np.take_along_axis(sorted_logits, reverse_indices, axis=-1)

    return logits


def sample(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> np.ndarray:
    """Sample from logits with temperature, top-k, and top-p.

    Args:
        logits: shape (batch, vocab_size).
        temperature: scaling factor (>0). Lower = more deterministic.
        top_k: top-k filtering.
        top_p: nucleus filtering.
    Returns:
        Sampled token IDs, shape (batch, 1).
    """
    logits = logits / max(temperature, 1e-8)
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    tokens = np.array(
        [
            np.random.choice(vocab_size, p=probs[i])
            for i, vocab_size in enumerate(probs.shape[0] * [probs.shape[1]])
        ]
    )
    return tokens.reshape(-1, 1)


def apply_repetition_penalty(
    logits: np.ndarray,
    input_ids: np.ndarray,
    penalty: float = 1.0,
) -> np.ndarray:
    """Apply repetition penalty to logits.

    Args:
        logits: shape (batch, vocab_size).
        input_ids: shape (batch, seq_len) of already-generated token IDs.
        penalty: >1 penalizes repetition, <1 encourages repetition.
    Returns:
        Modified logits.
    """
    if penalty == 1.0:
        return logits
    logits = logits.copy()
    for batch_idx in range(logits.shape[0]):
        for token_id in set(input_ids[batch_idx].flat):
            if logits[batch_idx, token_id] < 0:
                logits[batch_idx, token_id] *= penalty
            else:
                logits[batch_idx, token_id] /= penalty
    return logits


# ===========================================================================
#  Streamers
# ===========================================================================


class Streamer:
    """Base class for token streaming."""

    def put(self, token_id: int):
        pass

    def put_text(self, text: str):
        pass

    def end(self):
        pass


class TextStreamer(Streamer):
    """Streams decoded text, printing token-by-token."""

    def __init__(self, tokenizer, skip_prompt: bool = True, print_fn: Callable = print):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.print_fn = print_fn
        self._buffer = ""
        self._seen_tokens = 0

    def put(self, token_id: int):
        self._seen_tokens += 1
        if self.skip_prompt:
            text = (
                self.tokenizer.decode([token_id]) if self.tokenizer else str(token_id)
            )
        else:
            text = (
                self.tokenizer.decode([token_id]) if self.tokenizer else str(token_id)
            )
        self._buffer += text
        self.print_fn(text, end="", flush=True)

    def end(self):
        if self._buffer:
            self.print_fn()


class TokenStreamer(Streamer):
    """Streams raw token IDs as a list."""

    def __init__(self):
        self.tokens: List[int] = []

    def put(self, token_id: int):
        self.tokens.append(token_id)

    def end(self):
        pass

    def get_tokens(self) -> List[int]:
        return self.tokens


# ===========================================================================
#  Core Generation
# ===========================================================================


def _prepare_input_ids(input_ids: Union[List[int], np.ndarray, Tensor]) -> np.ndarray:
    """Normalize input_ids to a 2D numpy int64 array."""
    if isinstance(input_ids, Tensor):
        input_ids = input_ids.data
    if isinstance(input_ids, list):
        input_ids = np.array([input_ids], dtype=np.int64)
    if input_ids.ndim == 1:
        input_ids = input_ids[np.newaxis, :]
    return input_ids.astype(np.int64)


def _get_logits(
    model, input_ids, past_key_values, attention_mask=None, position_ids=None
):
    """Run model forward and return logits for the last position."""
    input_tensor = Tensor(
        input_ids.astype(np.float32) if input_ids.dtype.kind == "i" else input_ids
    )
    # Actually, input_ids should be int for embedding lookup. The Embedding module
    # currently does idx = indices.data.astype(np.int64), so passing float is fine.
    # But cleaner to just pass as float32 since Tensor stores float32.
    input_tensor = Tensor(input_ids.astype(np.float32))

    mask_tensor = None
    if attention_mask is not None:
        mask_tensor = Tensor(attention_mask.astype(np.float32))

    pos_tensor = None
    if position_ids is not None:
        pos_tensor = position_ids

    outputs = model.forward(
        input_ids=input_tensor,
        attention_mask=mask_tensor,
        position_ids=pos_tensor,
        past_key_values=past_key_values,
        use_cache=True,
    )
    logits = outputs["logits"].data
    new_cache = outputs.get("past_key_values")
    return logits, new_cache


def greedy_search(
    model,
    input_ids: np.ndarray,
    config: GenerationConfig,
    streamer: Optional[Streamer] = None,
    past_key_values=None,
    attention_mask=None,
    position_ids=None,
) -> Dict[str, Any]:
    """Greedy decoding (deterministic)."""
    batch_size = input_ids.shape[0]
    cur_len = input_ids.shape[1]
    max_len = cur_len + config.max_new_tokens
    output_ids = input_ids.copy()

    unfinished = np.ones(batch_size, dtype=bool)

    while cur_len < max_len:
        logits, past_key_values = _get_logits(
            model, input_ids, past_key_values, attention_mask, position_ids
        )
        next_token_logits = logits[:, -1, :]

        next_token_logits = apply_repetition_penalty(
            next_token_logits, output_ids[:, :cur_len], config.repetition_penalty
        )

        next_tokens = np.argmax(next_token_logits, axis=-1)
        next_tokens = next_tokens * unfinished + config.pad_token_id * (1 - unfinished)
        next_tokens = next_tokens.reshape(-1, 1)

        output_ids = np.concatenate([output_ids, next_tokens], axis=1)

        for i in range(batch_size):
            if unfinished[i] and next_tokens[i, 0] == config.eos_token_id:
                unfinished[i] = False

        if streamer is not None:
            for i in range(batch_size):
                streamer.put(int(next_tokens[i, 0]))

        input_ids = next_tokens
        attention_mask = None
        position_ids = None
        cur_len += 1

        if not unfinished.any():
            break

    if streamer is not None:
        streamer.end()

    return {"output_ids": output_ids, "past_key_values": past_key_values}


def sample_generation(
    model,
    input_ids: np.ndarray,
    config: GenerationConfig,
    streamer: Optional[Streamer] = None,
    past_key_values=None,
    attention_mask=None,
    position_ids=None,
) -> Dict[str, Any]:
    """Sampling-based decoding (stochastic)."""
    batch_size = input_ids.shape[0]
    cur_len = input_ids.shape[1]
    max_len = cur_len + config.max_new_tokens
    output_ids = input_ids.copy()

    unfinished = np.ones(batch_size, dtype=bool)

    while cur_len < max_len:
        logits, past_key_values = _get_logits(
            model, input_ids, past_key_values, attention_mask, position_ids
        )
        next_token_logits = logits[:, -1, :]

        next_token_logits = apply_repetition_penalty(
            next_token_logits, output_ids[:, :cur_len], config.repetition_penalty
        )

        next_token_logits = top_k_top_p_filtering(
            next_token_logits,
            top_k=config.top_k,
            top_p=config.top_p,
        )

        scaled_logits = next_token_logits / max(config.temperature, 1e-8)

        if config.temperature < 1e-6:
            next_tokens = np.argmax(scaled_logits, axis=-1)
        else:
            probs = np.exp(scaled_logits - scaled_logits.max(axis=-1, keepdims=True))
            probs = probs / probs.sum(axis=-1, keepdims=True)
            vocab_size = probs.shape[-1]
            next_tokens = np.array(
                [np.random.choice(vocab_size, p=probs[i]) for i in range(batch_size)]
            )

        next_tokens = next_tokens.reshape(-1, 1)
        next_tokens = next_tokens * unfinished[:, None] + config.pad_token_id * (
            1 - unfinished[:, None]
        )

        output_ids = np.concatenate([output_ids, next_tokens], axis=1)

        for i in range(batch_size):
            if unfinished[i] and next_tokens[i, 0] == config.eos_token_id:
                unfinished[i] = False

        if streamer is not None:
            for i in range(batch_size):
                streamer.put(int(next_tokens[i, 0]))

        input_ids = next_tokens
        attention_mask = None
        position_ids = None
        cur_len += 1

        if not unfinished.any():
            break

    if streamer is not None:
        streamer.end()

    return {"output_ids": output_ids, "past_key_values": past_key_values}


def beam_search(
    model,
    input_ids: np.ndarray,
    config: GenerationConfig,
    past_key_values=None,
    attention_mask=None,
    position_ids=None,
) -> Dict[str, Any]:
    """Beam search decoding.

    Simplified implementation that supports batch_size=1.
    """
    batch_size = input_ids.shape[0]
    num_beams = config.num_beams
    cur_len = input_ids.shape[1]
    max_len = cur_len + config.max_new_tokens

    beam_scores = np.zeros((batch_size, num_beams), dtype=np.float64)
    beam_scores[:, 1:] = -1e9

    input_ids = np.repeat(input_ids, num_beams, axis=0)
    if past_key_values is not None:
        past_key_values = [
            (k, v) for layer_kv in past_key_values for k, v in [layer_kv]  # untested
        ]

    done_beams = [[] for _ in range(batch_size)]
    done_scores = [[] for _ in range(batch_size)]

    while cur_len < max_len:
        logits, past_key_values = _get_logits(
            model, input_ids, past_key_values, attention_mask, position_ids
        )
        next_logits = logits[:, -1, :]

        next_logits = apply_repetition_penalty(
            next_logits, input_ids[:, :cur_len], config.repetition_penalty
        )

        vocab_size = next_logits.shape[-1]
        log_probs = (
            next_logits
            - next_logits.max(axis=-1, keepdims=True)
            - np.log(
                np.exp(next_logits - next_logits.max(axis=-1, keepdims=True)).sum(
                    axis=-1, keepdims=True
                )
            )
        )

        next_scores = beam_scores.reshape(-1, 1) + log_probs
        next_scores = next_scores.reshape(batch_size, num_beams * vocab_size)

        topk_scores = np.partition(next_scores, -num_beams, axis=-1)[:, -num_beams:]
        topk_indices = np.argpartition(next_scores, -num_beams, axis=-1)[:, -num_beams:]

        beam_indices = topk_indices // vocab_size
        token_indices = topk_indices % vocab_size

        new_input_ids = []
        new_past = [[] for _ in range(batch_size)]
        new_scores = np.zeros_like(beam_scores)

        for batch_idx in range(batch_size):
            for beam_idx in range(num_beams):
                src_beam = int(beam_indices[batch_idx, beam_idx])
                token = int(token_indices[batch_idx, beam_idx])
                score = float(topk_scores[batch_idx, beam_idx])

                if token == config.eos_token_id and config.early_stopping:
                    done_scores[batch_idx].append(score)
                    done_beams[batch_idx].append(
                        np.concatenate(
                            [
                                input_ids[batch_idx * num_beams + src_beam, :cur_len],
                                np.array([token]),
                            ]
                        )
                    )
                    new_scores[batch_idx, beam_idx] = -1e9
                    new_input_ids.append(np.array([config.pad_token_id]))
                else:
                    new_scores[batch_idx, beam_idx] = score
                    new_input_ids.append(
                        np.concatenate(
                            [
                                input_ids[batch_idx * num_beams + src_beam, :cur_len],
                                np.array([token]),
                            ]
                        )
                    )

        input_ids = np.stack(new_input_ids)
        beam_scores = new_scores
        cur_len += 1

        if all(len(d) >= num_beams for d in done_beams):
            break

    output_ids = []
    for batch_idx in range(batch_size):
        all_candidates = list(zip(done_scores[batch_idx], done_beams[batch_idx]))
        if not all_candidates:
            for beam_idx in range(num_beams):
                all_candidates.append(
                    (
                        beam_scores[batch_idx, beam_idx],
                        input_ids[batch_idx * num_beams + beam_idx],
                    )
                )
        all_candidates.sort(
            key=lambda x: x[0] / (len(x[1]) ** config.length_penalty), reverse=True
        )
        output_ids.append(all_candidates[0][1])

    return {"output_ids": np.array(output_ids), "past_key_values": past_key_values}


# ===========================================================================
#  Main Entry Point
# ===========================================================================


def generate(
    model: Module,
    input_ids: Union[List[int], np.ndarray, Tensor],
    generation_config: Optional[GenerationConfig] = None,
    streamer: Optional[Streamer] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Generate text autoregressively.

    Args:
        model: A causal LM model with ``.forward(input_ids, attention_mask, position_ids, past_key_values, use_cache)``
               returning a dict with ``logits`` and ``past_key_values``.
        input_ids: Input token IDs (1D list/array or 2D batch).
        generation_config: Generation parameters (defaults used if None).
        streamer: Optional streamer for token-by-token output.
        **kwargs: Override generation config fields.

    Returns:
        Dict with keys ``output_ids`` (np.ndarray), ``past_key_values`` (optional).
    """
    config = generation_config or GenerationConfig()
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)

    input_ids = _prepare_input_ids(input_ids)

    if config.num_beams > 1:
        return beam_search(model, input_ids, config, streamer=streamer)

    if config.do_sample and config.temperature > 1e-6:
        return sample_generation(model, input_ids, config, streamer=streamer)
    else:
        return greedy_search(model, input_ids, config, streamer=streamer)


# ===========================================================================
#  Batch Inference
# ===========================================================================


def _pad_sequences(
    sequences: List[np.ndarray],
    pad_token_id: int,
    max_length: Optional[int] = None,
) -> np.ndarray:
    """Pad a list of 1D token arrays to a 2D batch."""
    max_len = max_length or max(len(s) for s in sequences)
    batch = np.full((len(sequences), max_len), pad_token_id, dtype=np.int64)
    for i, seq in enumerate(sequences):
        batch[i, : len(seq)] = seq
    return batch


def _create_attention_mask(
    input_ids: np.ndarray,
    pad_token_id: int,
) -> np.ndarray:
    """Create attention mask (1 for real, 0 for padding)."""
    return (input_ids != pad_token_id).astype(np.float32)


def batch_generate(
    model: Module,
    prompts: List[Union[List[int], np.ndarray]],
    generation_config: Optional[GenerationConfig] = None,
    streamer: Optional[Streamer] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Generate for a batch of prompts with padding.

    Args:
        model: Causal LM model.
        prompts: List of 1D token ID arrays (or lists).
        generation_config: Generation parameters.
        streamer: Optional streamer.
        **kwargs: Config overrides.

    Returns:
        Dict with ``output_ids`` (np.ndarray) and per-sequence results.
    """
    config = generation_config or GenerationConfig()

    padded = _pad_sequences(
        [np.array(p, dtype=np.int64) if isinstance(p, list) else p for p in prompts],
        config.pad_token_id,
    )
    attention_mask = _create_attention_mask(padded, config.pad_token_id)

    return generate(model, padded, config, streamer=streamer)


__all__ = [
    "GenerationConfig",
    "top_k_top_p_filtering",
    "sample",
    "apply_repetition_penalty",
    "Streamer",
    "TextStreamer",
    "TokenStreamer",
    "greedy_search",
    "sample_generation",
    "beam_search",
    "generate",
    "batch_generate",
]
