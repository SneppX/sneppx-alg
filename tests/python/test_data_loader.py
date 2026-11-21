"""Tests for Data Loader."""

import tempfile
from pathlib import Path
import numpy as np
from SneppX_ALG.interface_bindings.data_loader import (
    Dataset, TensorDataset, DataLoader, DistributedSampler, default_collate,
)


def test_dataset_basic():
    ds = Dataset(data=[1, 2, 3], targets=[4, 5, 6])
    assert len(ds) == 3
    assert ds[0] == (1, 4)
    assert ds[2] == (3, 6)


def test_dataset_no_targets():
    ds = Dataset(data=[10, 20])
    assert ds[0] == 10
    assert ds[1] == 20


def test_tensor_dataset():
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    target = np.array([0, 1, 0], dtype=np.int64)
    ds = TensorDataset(data, target)
    assert len(ds) == 3
    x1, y1 = ds[0]
    assert np.allclose(x1, data[0])
    assert y1 == target[0]


def test_tensor_dataset_no_target():
    data = np.array([1, 2, 3], dtype=np.float32)
    ds = TensorDataset(data)
    assert len(ds) == 3
    assert ds[1] == data[1]


def test_default_collate_ndarray():
    batch = [np.array([1, 2]), np.array([3, 4])]
    result = default_collate(batch)
    assert result.shape == (2, 2)
    assert np.allclose(result, [[1, 2], [3, 4]])


def test_default_collate_list():
    batch = [([1, 2], 0), ([3, 4], 1)]
    result = default_collate(batch)
    assert len(result) == 2


def test_data_loader_basic():
    ds = Dataset(data=list(range(100)))
    loader = DataLoader(ds, batch_size=10, shuffle=False)
    batches = list(loader)
    assert len(batches) == 10
    assert len(batches[0]) == 10
    assert batches[0][0] == 0
    assert batches[-1][0] == 90


def test_data_loader_drop_last():
    ds = Dataset(data=list(range(15)))
    loader = DataLoader(ds, batch_size=4, drop_last=True)
    batches = list(loader)
    assert len(batches) == 3
    assert len(batches[0]) == 4


def test_data_loader_len():
    ds = Dataset(data=list(range(100)))
    loader = DataLoader(ds, batch_size=32)
    assert len(loader) == 4


def test_distributed_sampler():
    ds = Dataset(data=list(range(100)))
    sampler = DistributedSampler(ds, num_replicas=4, rank=0, shuffle=False)
    indices = list(sampler)
    assert len(indices) == 25
    assert indices[0] == 0
    assert indices[-1] == 96


def test_distributed_sampler_shuffle():
    ds = Dataset(data=list(range(100)))
    sampler = DistributedSampler(ds, num_replicas=2, rank=0, shuffle=True, seed=42)
    indices = list(sampler)
    assert len(indices) == 50
    assert indices != list(range(0, 100, 2))


def test_distributed_sampler_set_epoch():
    ds = Dataset(data=list(range(100)))
    sampler = DistributedSampler(ds, num_replicas=2, rank=0, shuffle=True, seed=42)
    indices_epoch0 = list(sampler)
    sampler.set_epoch(1)
    indices_epoch1 = list(sampler)
    assert indices_epoch0 != indices_epoch1


def test_data_loader_with_sampler():
    ds = Dataset(data=list(range(100)))
    sampler = DistributedSampler(ds, num_replicas=2, rank=0, shuffle=False)
    loader = DataLoader(ds, batch_size=10, sampler=sampler)
    batches = list(loader)
    assert len(batches) == 5


def test_data_loader_pin_memory():
    ds = Dataset(data=[np.array([1.0, 2.0, 3.0]) for _ in range(5)])
    loader = DataLoader(ds, batch_size=2, pin_memory=True)
    for batch in loader:
        assert isinstance(batch, np.ndarray)


def test_data_loader_set_epoch():
    ds = Dataset(data=list(range(100)))
    loader = DataLoader(ds, batch_size=10, shuffle=True)
    loader.set_epoch(1)
    batches1 = list(loader)
    loader.set_epoch(2)
    batches2 = list(loader)
    idx1 = [item for batch in batches1 for item in batch]
    idx2 = [item for batch in batches2 for item in batch]
    assert idx1 != idx2


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
