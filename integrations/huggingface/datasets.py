"""Dataset integration — load HuggingFace datasets directly into SNEPPX DataLoader."""

from typing import Optional, Callable, Any


def load_hf_dataset(dataset_id: str, split: Optional[str] = None,
                    preprocess: Optional[Callable] = None):
    """Load a HuggingFace dataset and wrap it as a SNEPPX-compatible Dataset.

    Args:
        dataset_id: HF dataset name, e.g. ``'imdb'``, ``'wikitext'``.
        split: Dataset split to load (e.g. ``'train'``, ``'test'``).
        preprocess: Optional callable ``f(example) -> (x, y)`` applied to each
            example to produce (features, label) pairs.

    Returns:
        A ``HFIterableDataset`` usable with
        ``SneppX_ALG.data.DataLoader``.

    Examples:
        >>> from sneppx.integrations.huggingface import load_hf_dataset
        >>> ds = load_hf_dataset('imdb', split='train')
        >>> from SneppX_ALG import DataLoader
        >>> loader = DataLoader(ds, batch_size=8, shuffle=True)
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError(
            "Dataset integration requires the `datasets` package. "
            "Install with: pip install datasets"
        )

    hf_ds = load_dataset(dataset_id, split=split)
    return HFIterableDataset(hf_ds, preprocess=preprocess)


class HFIterableDataset:
    """Adapter mapping an HF dataset to SNEPPX's Dataset protocol."""

    def __init__(self, hf_dataset, preprocess: Optional[Callable] = None):
        self._ds = hf_dataset
        self._preprocess = preprocess

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx):
        example = self._ds[idx]
        if self._preprocess is not None:
            return self._preprocess(example)
        return example

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        return f"HFIterableDataset({getattr(self._ds, 'info', None) and self._ds.info.dataset_name or 'dataset'}, len={len(self)})"
