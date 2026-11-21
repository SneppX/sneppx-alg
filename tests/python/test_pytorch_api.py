"""Tests for sneppx_serve.pytorch_api (sneppx-alg interface_bindings.pytorch_api)."""

import numpy as np

from SneppX_ALG.interface_bindings.pytorch_api import Tensor, Linear, Module, Dataset, DataLoader, TensorDataset, Subset, ConcatDataset, random_split


def test_tensor_shape():
    t = Tensor(np.array([1.0, 2, 3]))
    assert t.shape == (3,)


def test_tensor_dtype():
    t = Tensor(np.array([1.0, 2, 3]))
    assert t.dtype is not None


def test_linear_forward():
    m = Linear(3, 2)
    t = Tensor(np.array([1.0, 2, 3], dtype=np.float64))
    out = m(t)
    assert out.shape == (2,)


def test_module_zero_grad():
    m = Linear(3, 2)
    assert hasattr(m, "zero_grad")


def test_accuracy():
    from SneppX_ALG.interface_bindings.pytorch_api import accuracy

    a = accuracy(np.array([0, 1, 2]), np.array([0, 1, 3]))
    assert 0 <= a <= 1



class _ToyDataset(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        return np.array([index], dtype=np.float32), index


def test_dataloader_batching_drop_last_and_threaded_fetch():
    dataset = _ToyDataset()
    loader = DataLoader(dataset, batch_size=2, num_workers=2, drop_last=True)
    assert len(loader) == 2
    batches = list(loader)
    assert len(batches) == 2
    assert batches[0][0].shape == (2, 1)
    assert batches[0][1].shape == (2,)
    assert np.array_equal(batches[0][1].data, np.array([0, 1]))
    assert np.array_equal(batches[1][1].data, np.array([2, 3]))


def test_dataloader_seeded_shuffle_and_custom_collate():
    dataset = _ToyDataset()
    first = DataLoader(dataset, batch_size=5, shuffle=True, seed=123)
    second = DataLoader(dataset, batch_size=5, shuffle=True, seed=123)
    first_order = list(first)[0][1].data
    second_order = list(second)[0][1].data
    assert np.array_equal(first_order, second_order)

    custom = DataLoader(
        dataset, batch_size=2, collate_fn=lambda samples: [sample[1] for sample in samples]
    )
    assert list(custom)[0] == [0, 1]


def test_dataloader_rejects_invalid_configuration():
    dataset = _ToyDataset()
    for kwargs in (
        {"batch_size": 0},
        {"batch_size": -1},
        {"batch_size": 1.5},
        {"num_workers": -1},
        {"num_workers": 1.5},
    ):
        try:
            DataLoader(dataset, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid DataLoader configuration accepted: {kwargs!r}")



def test_tensor_dataset_subset_and_concat_dataset():
    features = np.arange(10, dtype=np.float32).reshape(5, 2)
    labels = np.arange(5, dtype=np.int64)
    dataset = TensorDataset(features, labels)
    assert len(dataset) == 5
    assert np.array_equal(dataset[2][0], features[2])
    assert dataset[2][1] == 2

    subset = Subset(dataset, [4, 1])
    assert len(subset) == 2
    assert subset[0][1] == 4
    assert subset[1][1] == 1

    combined = ConcatDataset([dataset, subset])
    assert len(combined) == 7
    assert combined[5][1] == 4
    assert combined[-1][1] == 1
    try:
        combined[7]
    except IndexError:
        pass
    else:
        raise AssertionError("ConcatDataset must reject out-of-range indices")

    batches = list(DataLoader(dataset, batch_size=3))
    assert batches[0][0].shape == (3, 2)
    assert batches[1][1].shape == (2,)


def test_tensor_dataset_rejects_invalid_inputs():
    for args in ((), (np.array(1),), (np.arange(3), np.arange(4))):
        try:
            TensorDataset(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid TensorDataset inputs accepted: {args!r}")



def test_random_split_integer_fractional_and_seeded():
    dataset = TensorDataset(np.arange(10, dtype=np.int64))
    train_a, val_a = random_split(dataset, [8, 2], generator=123)
    train_b, val_b = random_split(dataset, [8, 2], generator=123)
    assert len(train_a) == 8 and len(val_a) == 2
    assert [train_a[i][0].item() for i in range(len(train_a))] == [
        train_b[i][0].item() for i in range(len(train_b))
    ]
    assert [val_a[i][0].item() for i in range(len(val_a))] == [
        val_b[i][0].item() for i in range(len(val_b))
    ]

    fractional = random_split(dataset, [0.7, 0.2, 0.1], generator=7)
    assert [len(part) for part in fractional] == [7, 2, 1]

    try:
        random_split(dataset, [3, 3])
    except ValueError:
        pass
    else:
        raise AssertionError("random_split must reject lengths with wrong total")
