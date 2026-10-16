import numpy as np

from SneppX_ALG import datasets as ds


def test_train_test_split_sizes_and_repro():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    Xtr, Xte, ytr, yte = ds.train_test_split(X, y, test_size=0.3, seed=0)
    assert Xtr.shape[0] == 7 and Xte.shape[0] == 3
    assert ytr.shape[0] == 7 and yte.shape[0] == 3
    Xtr2, Xte2, _, _ = ds.train_test_split(X, y, test_size=0.3, seed=0)
    assert np.array_equal(Xtr, Xtr2)


def test_batch_indices():
    batches = ds.batch_indices(10, 4)
    assert [len(b) for b in batches] == [4, 4, 2]
    assert [len(b) for b in ds.batch_indices(10, 4, drop_last=True)] == [4, 4]
    covered = np.concatenate(ds.batch_indices(10, 3, shuffle=True, seed=1))
    assert sorted(covered.tolist()) == list(range(10))


def test_k_fold_indices():
    folds = ds.k_fold_indices(10, 5, seed=0)
    assert len(folds) == 5
    for train, val in folds:
        assert len(val) == 2
        assert len(train) == 8
        assert set(train).isdisjoint(set(val))
    all_val = np.concatenate([v for _, v in folds])
    assert sorted(all_val.tolist()) == list(range(10))


def test_one_hot_roundtrip():
    oh = ds.one_hot([0, 2, 1], num_classes=3)
    assert oh.shape == (3, 3)
    assert np.array_equal(ds.one_hot_decode(oh), [0, 2, 1])
    assert oh.sum() == 3.0


def test_normalize():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    xn, mean, std = ds.normalize(x, axis=0)
    assert np.allclose(xn.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(xn.std(axis=0), 1.0, atol=1e-9)


def test_class_counts_and_stratified_split():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert ds.class_counts(y) == {0: 4, 1: 4}
    X = np.arange(16).reshape(8, 2)
    Xtr, Xte, ytr, yte = ds.stratified_split(X, y, test_size=0.25, seed=0)
    assert ds.class_counts(ytr) == {0: 3, 1: 3}
    assert ds.class_counts(yte) == {0: 1, 1: 1}


def test_vocabulary():
    v = ds.Vocabulary.from_corpus(["the cat sat", "the cat ran"])
    assert "the" in v
    ids = v.encode(["the", "cat"])
    assert v.decode(ids) == ["the", "cat"]
    assert v.token_to_id("nope") == v.unk_id
    assert len(v.most_common(1)) == 1
    assert v.most_common(1)[0][0] == "the"


def test_char_vocabulary():
    v = ds.CharVocabulary.from_corpus(["abc", "abd"])
    ids = v.encode("ab")
    assert v.decode(ids) == ["a", "b"]


def test_tabular_dataset():
    X = np.arange(12).reshape(4, 3)
    y = np.array([0, 1, 0, 1])
    d = ds.TabularDataset(X, y, feature_names=["a", "b", "c"])
    assert len(d) == 4 and d.n_features == 3
    xb, yb = d.head(2)
    assert xb.shape == (2, 3) and yb.tolist() == [0, 1]
    d.standardize()
    assert np.allclose(d.X.mean(axis=0), 0.0, atol=1e-9)


def test_transforms():
    t = ds.Compose(ds.ToFloat32(), ds.Rescale(0.5))
    out = t(np.ones((2, 2)))
    assert out.dtype == np.float32
    assert np.allclose(out, 0.5)

    n = ds.Normalize([1.0, 2.0], [1.0, 1.0])
    xn = n(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert np.allclose(xn, [[0.0, 0.0], [2.0, 2.0]])

    flip = ds.RandomHorizontalFlip(p=1.0, seed=0)
    a = np.array([[1, 2], [3, 4]])
    assert np.array_equal(flip(a), np.flip(a, axis=-1))


def test_array_dataloader():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    loader = ds.ArrayDataLoader(X, y, batch_size=4, shuffle=False)
    assert len(loader) == 3
    batches = list(loader)
    assert [b[0].shape[0] for b in batches] == [4, 4, 2]
    assert np.array_equal(np.concatenate([b[1] for b in batches]), y)

    ld2 = ds.ArrayDataLoader(X, batch_size=4, drop_last=True)
    assert len(ld2) == 2
    assert all(b.shape[0] == 4 for b in ld2)
