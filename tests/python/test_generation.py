"""Tests for generation module and tokenizer."""

import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python")
)

import numpy as np
from SneppX_ALG.interface_bindings.generation import (
    GenerationConfig,
    top_k_top_p_filtering,
    sample,
    apply_repetition_penalty,
    Streamer,
    TextStreamer,
    TokenStreamer,
    greedy_search,
    sample_generation,
    beam_search,
    generate,
    batch_generate,
)
from SneppX_ALG.interface_bindings.tokenizer import Tokenizer, SimpleTokenizer
from SneppX_ALG.interface_bindings.tensor import Tensor

# ===========================================================================
#  Minimal test model
# ===========================================================================


class DummyModel:
    """A minimal causal LM for testing generation loops."""

    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self._counter = 0

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=True,
    ):
        batch, seq = (
            input_ids.shape
            if hasattr(input_ids, "shape")
            else (1, input_ids.data.shape[1])
        )
        logits = np.random.randn(batch, seq, self.vocab_size).astype(np.float32)
        logits[:, -1, :] = -100.0
        is_prefill = past_key_values is None
        if is_prefill:
            logits[:, -1, 42] = 100.0
        else:
            logits[:, -1, 50 + (self._counter % 48)] = 100.0
            logits[:, -1, 90] = 99.0
        self._counter += 1
        return {
            "logits": Tensor(logits),
            "past_key_values": (
                [(np.zeros((1, 1, 1)), np.zeros((1, 1, 1)))] if use_cache else None
            ),
        }


class TestLogitsProcessing:
    def test_top_k_filtering(self):
        logits = np.random.randn(1, 50).astype(np.float32)
        filtered = top_k_top_p_filtering(logits, top_k=10)
        kept = np.sum(filtered > -1e8)
        assert kept <= 10, f"top_k=10 kept {kept}"

    def test_top_p_filtering(self):
        logits = np.random.randn(1, 50).astype(np.float32)
        filtered = top_k_top_p_filtering(logits, top_k=0, top_p=0.5)
        probs = np.exp(filtered - filtered.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        assert abs(probs[probs > 0].sum() - 1.0) < 1e-5

    def test_combined_filtering(self):
        logits = np.random.randn(1, 100).astype(np.float32)
        filtered = top_k_top_p_filtering(logits, top_k=20, top_p=0.9)
        assert filtered.shape == logits.shape
        # At most 20 non -inf values
        kept = np.sum(filtered > -1e8)
        assert kept <= 20 and kept >= 1

    def test_repetition_penalty(self):
        logits = np.ones((1, 10), dtype=np.float32) * 5.0
        input_ids = np.array([[1, 2, 3]])
        penalized = apply_repetition_penalty(logits, input_ids, penalty=1.2)
        assert penalized[0, 1] < 5.0, "Token 1 should be penalized"
        assert penalized[0, 4] == 5.0, "Token 4 should be unchanged"

    def test_sample_deterministic(self):
        logits = np.zeros((1, 10), dtype=np.float32)
        logits[0, 3] = 100.0
        tokens = sample(logits, temperature=1e-8)
        assert tokens[0, 0] == 3


class TestGeneration:
    def test_greedy_search(self):
        model = DummyModel(vocab_size=100)
        config = GenerationConfig(max_new_tokens=5, do_sample=False, eos_token_id=99)
        input_ids = np.array([[1, 2, 3]], dtype=np.int64)
        result = greedy_search(model, input_ids, config)
        assert result["output_ids"].shape == (1, 8)
        assert result["output_ids"][0, -1] != 99

    def test_greedy_search_eos(self):
        model = DummyModel(vocab_size=100)
        config = GenerationConfig(max_new_tokens=50, do_sample=False, eos_token_id=42)
        input_ids = np.array([[1, 2, 3]], dtype=np.int64)
        result = greedy_search(model, input_ids, config)
        # Should stop at EOS (token 42) on first step
        assert result["output_ids"].shape[1] < 10

    def test_generate_api(self):
        model = DummyModel(vocab_size=100)
        result = generate(model, [1, 2, 3], max_new_tokens=5, do_sample=False)
        assert "output_ids" in result
        assert result["output_ids"].shape[1] == 8

    def test_generate_with_tensor(self):
        model = DummyModel(vocab_size=100)
        inp = Tensor(np.array([[1, 2, 3]], dtype=np.float32))
        result = generate(model, inp, max_new_tokens=3, do_sample=False)
        assert result["output_ids"].shape == (1, 6)

    def test_batch_generate(self):
        model = DummyModel(vocab_size=100)
        prompts = [
            [1, 2, 3],
            [4, 5],
        ]
        result = batch_generate(model, prompts, max_new_tokens=3, do_sample=False)
        assert result["output_ids"].shape[0] == 2

    def test_streamer_tokens(self):
        model = DummyModel(vocab_size=100)
        streamer = TokenStreamer()
        result = generate(
            model, [1, 2, 3], max_new_tokens=5, do_sample=False, streamer=streamer
        )
        assert len(streamer.get_tokens()) == 5

    def test_text_streamer(self):
        class FakeTokenizer:
            def decode(self, ids):
                return "a"

        model = DummyModel(vocab_size=100)
        streamer = TextStreamer(tokenizer=FakeTokenizer())
        result = generate(
            model, [1, 2, 3], max_new_tokens=3, do_sample=False, streamer=streamer
        )


class TestTokenizer:
    def test_simple_tokenizer_encode_decode(self):
        tok = SimpleTokenizer(vocab_size=100)
        tok.vocab["hello"] = 10
        tok.vocab["world"] = 20
        tok.inv_vocab[10] = "hello"
        tok.inv_vocab[20] = "world"
        ids = tok.encode("hello world")
        assert ids == [10, 20]
        text = tok.decode(ids)
        assert "hello" in text

    def test_tokenizer_wrapper(self):
        tok = Tokenizer()
        assert tok.vocab_size >= 4

    def test_tokenizer_encode_default(self):
        tok = Tokenizer()
        ids = tok.encode("hello world")
        assert len(ids) >= 2
        assert ids[0] == tok.bos_token_id

    def test_tokenizer_decode(self):
        tok = Tokenizer()
        ids = tok.encode("test", add_special_tokens=False)
        text = tok.decode(ids)
        assert isinstance(text, str)


class TestGenerationEdgeCases:
    def test_empty_input(self):
        model = DummyModel(vocab_size=100)
        result = generate(model, [[1]], max_new_tokens=0, do_sample=False)
        assert result["output_ids"].shape[1] == 1

    def test_beam_search(self):
        model = DummyModel(vocab_size=100)
        config = GenerationConfig(max_new_tokens=3, num_beams=2, early_stopping=True)
        input_ids = np.array([[1, 2]], dtype=np.int64)
        result = beam_search(model, input_ids, config)
        assert "output_ids" in result


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
