"""Tests for data.py — datasets, loaders, tokenizers."""

import numpy as np
import tempfile
import os
from SneppX_ALG.interface_bindings import (
    Tensor,
    Dataset,
    TensorDataset,
    TextDataset,
    SimpleTokenizer,
    BatchCollator,
    Preprocessor,
)


def test_dataset():
    data = [1, 2, 3, 4, 5]
    targets = [0, 1, 0, 1, 0]
    ds = Dataset(data, targets=targets)
    assert len(ds) == 5
    x, y = ds[0]
    assert x == 1
    assert y == 0
    print("  test_dataset PASS")


def test_dataset_no_targets():
    data = [1, 2, 3]
    ds = Dataset(data)
    assert ds[0] == 1
    print("  test_dataset_no_targets PASS")


def test_tensor_dataset():
    a = Tensor.ones((3, 4))
    b = Tensor.zeros((3,))
    ds = TensorDataset(a, b)
    assert len(ds) == 3
    x, y = ds[0]
    assert isinstance(x, Tensor)
    assert isinstance(y, Tensor)
    print("  test_tensor_dataset PASS")


def test_text_dataset():
    tokenizer = SimpleTokenizer(vocab_size=100)
    tokenizer.train(["hello world", "test data"])
    texts = ["hello world"]
    ds = TextDataset(texts, tokenizer, max_length=10)
    item = ds[0]
    assert isinstance(item, Tensor)
    assert len(item.data) == 10
    print("  test_text_dataset PASS")


def test_simple_tokenizer():
    tokenizer = SimpleTokenizer(vocab_size=100)
    tokenizer.train(["hello world this is a test", "another sentence here"])
    encoded = tokenizer.encode("hello world")
    assert encoded[0] == 2  # <s>
    assert encoded[-1] == 3  # </s>
    decoded = tokenizer.decode(encoded)
    assert isinstance(decoded, str)
    assert len(decoded) > 0
    print("  test_simple_tokenizer PASS")


def test_batch_collator():
    a = Tensor.ones((3,))
    b = Tensor.ones((5,))
    collator = BatchCollator(padding_value=0)
    batch = collator([a, b])
    assert batch.shape == (2, 5), f"Expected (2,5), got {batch.shape}"
    assert batch.data[0, 3] == 0  # padded
    print("  test_batch_collator PASS")


def test_batch_collator_tuples():
    x1 = Tensor.ones((3,))
    x2 = Tensor.ones((5,))
    y1 = Tensor.zeros((1,))
    y2 = Tensor.zeros((1,))
    collator = BatchCollator()
    xs, ys = collator([(x1, y1), (x2, y2)])
    assert xs.shape[0] == 2
    assert ys.shape[0] == 2
    print("  test_batch_collator_tuples PASS")


def test_preprocessor():
    p = Preprocessor()
    called = []

    def add_one(x):
        called.append(True)
        return x + 1

    p.add(add_one)
    p.add(add_one)
    result = p(5)
    assert result == 7
    assert len(called) == 2
    print("  test_preprocessor PASS")


if __name__ == "__main__":
    test_dataset()
    test_dataset_no_targets()
    test_tensor_dataset()
    test_text_dataset()
    test_simple_tokenizer()
    test_batch_collator()
    test_batch_collator_tuples()
    test_preprocessor()
    print("\nAll data tests passed!")
