"""SNEPPX datasets — dataset construction, splitting, batching and transforms.

Complements the core ``SneppX_ALG.data`` module with tabular/text helpers,
cross-validation utilities and pure-NumPy transforms::

    from SneppX_ALG import datasets

    Xtr, Xte, ytr, yte = datasets.train_test_split(X, y, test_size=0.2, seed=0)
    loader = datasets.ArrayDataLoader(Xtr, ytr, batch_size=32, shuffle=True)
"""

from . import loader, tabular, text, transforms, utils
from .loader import ArrayDataLoader
from .tabular import TabularDataset, class_counts, stratified_split
from .text import CharVocabulary, Vocabulary
from .transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    Rescale,
    ToFloat32,
)
from .utils import (
    batch_indices,
    k_fold_indices,
    normalize,
    one_hot,
    one_hot_decode,
    train_test_split,
)

__all__ = [
    "utils",
    "tabular",
    "text",
    "transforms",
    "loader",
    "train_test_split",
    "k_fold_indices",
    "batch_indices",
    "one_hot",
    "one_hot_decode",
    "normalize",
    "TabularDataset",
    "class_counts",
    "stratified_split",
    "Vocabulary",
    "CharVocabulary",
    "Compose",
    "ToFloat32",
    "Rescale",
    "Normalize",
    "RandomHorizontalFlip",
    "RandomCrop",
    "ArrayDataLoader",
]
