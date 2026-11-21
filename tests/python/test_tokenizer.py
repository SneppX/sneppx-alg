"""Tests for Tokenizer."""

import json
import tempfile
from pathlib import Path
from SneppX_ALG.interface_bindings.tokenizer import Tokenizer


def test_tokenizer_default_creation():
    tok = Tokenizer()
    assert tok.pad_token_id == 0
    assert tok.unk_token_id == 1
    assert tok.bos_token_id == 2
    assert tok.eos_token_id == 3


def test_tokenizer_encode_decode():
    tok = Tokenizer()
    text = "hello world"
    ids = tok.encode(text)
    assert isinstance(ids, list)
    assert len(ids) > 1
    assert ids[0] == tok.bos_token_id
    assert ids[-1] == tok.eos_token_id
    assert all(isinstance(i, int) for i in ids)


def test_tokenizer_roundtrip():
    tok = Tokenizer()
    text = "test message"
    ids = tok.encode(text)
    decoded = tok.decode(ids, skip_special_tokens=False)
    assert isinstance(decoded, str)
    assert len(decoded) > 0


def test_tokenizer_encode_empty():
    tok = Tokenizer()
    ids = tok.encode("")
    assert isinstance(ids, list)
    assert len(ids) == 2  # bos + eos
    assert ids[0] == tok.bos_token_id
    assert ids[-1] == tok.eos_token_id


def test_tokenizer_special_tokens():
    tok = Tokenizer()
    assert tok.bos_token_id != tok.eos_token_id
    assert tok.pad_token_id != tok.unk_token_id


def test_tokenizer_vocab_size_property():
    tok = Tokenizer()
    vs = tok.vocab_size
    assert isinstance(vs, int)
    assert vs > 0


def test_tokenizer_no_special_tokens():
    tok = Tokenizer()
    ids = tok.encode("hello", add_special_tokens=False)
    assert ids[0] != tok.bos_token_id
    assert ids[-1] != tok.eos_token_id


def test_tokenizer_encode_batch():
    tok = Tokenizer()
    texts = ["hello", "world"]
    batch = tok.encode_batch(texts)
    assert len(batch) == 2
    assert all(isinstance(ids, list) for ids in batch)


def test_tokenizer_decode_batch():
    tok = Tokenizer()
    texts = ["hello", "world"]
    batch = tok.encode_batch(texts)
    decoded = tok.decode_batch(batch)
    assert len(decoded) == 2
    assert all(isinstance(d, str) for d in decoded)


def test_tokenizer_apply_chat_template():
    tok = Tokenizer()
    messages = [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "Hi!"},
    ]
    result = tok.apply_chat_template(messages)
    assert "<|system|>" in result
    assert "<|user|>" in result
    assert "<|assistant|>" in result


if __name__ == "__main__":
    import sys
    locals_ = locals().copy()
    passed = 0
    failed = 0
    for name, fn in sorted(locals_.items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"  PASS {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL {name}: {e}")
                failed += 1
    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    sys.exit(failed)
