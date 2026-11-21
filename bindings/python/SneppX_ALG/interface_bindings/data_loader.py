"""Industrial DataLoader — prefetch workers, mmap, pinning, collate.

A torch.utils.data.DataLoader-style pipeline with:
  * multi-worker prefetch (background threads)
  * memory-mapped dataset support for huge corpora
  * optional collate function to batch samples
  * drop-last / shuffle with a seeded RNG
  * distributed-sampler awareness (so DDP/ZeRO jobs shard data)
"""

import os
import mmap
import threading
import struct
import random
import numpy as np
from typing import List, Optional, Callable, Iterable, Iterator, Any


class Dataset:
    """Base dataset interface."""

    def __init__(self, data=None, targets=None):
        if data is not None:
            self._data = data
            self._targets = targets

    def __len__(self) -> int:
        if hasattr(self, "_data"):
            return len(self._data)
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Any:
        if hasattr(self, "_data"):
            if self._targets is not None:
                return self._data[idx], self._targets[idx]
            return self._data[idx]
        raise NotImplementedError


class TensorDataset(Dataset):
    """In-memory dataset of (data, target) tensor pairs."""

    def __init__(self, data, target=None):
        self.data = data
        self.target = target

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        if self.target is not None:
            return self.data[idx], self.target[idx]
        return self.data[idx]


class MemoryMappedTextDataset(Dataset):
    """Memory-mapped corpus: each sample is ``seq_len`` tokens packed as int32.

    The backing file is opened with mmap so multi-GB datasets never fully
    load into RAM. Tokenization is assumed precomputed into the file.
    """

    def __init__(self, path: str, seq_len: int = 2048, dtype: str = "int32"):
        self.path = path
        self.seq_len = seq_len
        self.itemsize = 4 if dtype == "int32" else 2
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self._f = open(path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        nbytes = os.path.getsize(path)
        self.num_tokens = nbytes // self.itemsize
        self.num_samples = max(0, self.num_tokens - seq_len)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(idx)
        start = idx * self.itemsize
        buf = self._mm[start : start + self.seq_len * self.itemsize]
        arr = np.frombuffer(buf, dtype=np.int32)
        # input/target shift handled by the training loop
        return arr

    def close(self):
        self._mm.close()
        self._f.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class DistributedSampler:
    """Deterministic sharder for distributed training.

    Each rank sees ``len(dataset) // world_size`` samples, with an optional
    epoch-based seed for shuffling so each epoch visits a different order.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.num_replicas = max(1, num_replicas)
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.num_samples = int(len(dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)
        # trim to multiple of world_size
        indices = indices[: self.total_size]
        # shard
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples


class _PrefetchWorker:
    """Background fetcher that keeps ``prefetch_factor`` batches ready."""

    def __init__(
        self,
        dataset: Dataset,
        indices: List[int],
        collate_fn: Optional[Callable],
        prefetch: int,
    ):
        self.dataset = dataset
        self.indices = indices
        self.collate_fn = collate_fn
        self.prefetch = prefetch
        self.queue: "queue.Queue" = __import__("queue").Queue(maxsize=prefetch)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.pos = 0
        self._dead = False

    def _run(self):
        while self.pos < len(self.indices) and not self._dead:
            idx = self.indices[self.pos]
            self.pos += 1
            sample = self.dataset[idx]
            if self.collate_fn:
                sample = self.collate_fn([sample])
            self.queue.put(sample)
        self.queue.put(None)  # sentinel

    def start(self):
        self.thread.start()

    def get(self):
        return self.queue.get()

    def stop(self):
        self._dead = True


def default_collate(batch: List[Any]) -> Any:
    """Stack a list of samples into a batched array (best-effort)."""
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch, axis=0)
    if isinstance(batch[0], (list, tuple)):
        return [default_collate([b[i] for b in batch]) for i in range(len(batch[0]))]
    return batch


class DataLoader:
    """Configurable data loader with prefetch and sharding."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        drop_last: bool = False,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        sampler: Optional[Any] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn or default_collate
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.sampler = sampler
        self._seed = 0

    def _indices(self) -> List[int]:
        if self.sampler is not None:
            return list(self.sampler)
        n = len(self.dataset)
        idx = list(range(n))
        if self.shuffle:
            rng = random.Random(self._seed)
            rng.shuffle(idx)
        return idx

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Any]:
        indices = self._indices()
        if self.num_workers > 0:
            yield from self._iter_workers(indices)
        else:
            yield from self._iter_single(indices)

    def _iter_single(self, indices: List[int]):
        for i in range(0, len(indices), self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                continue
            samples = [self.dataset[j] for j in batch_idx]
            batch = self.collate_fn(samples)
            if self.pin_memory:
                batch = self._pin(batch)
            yield batch

    def _iter_workers(self, indices: List[int]):
        # Split indices into per-worker chunks
        chunks = [indices[p :: self.num_workers] for p in range(self.num_workers)]
        workers = []
        for c in chunks:
            w = _PrefetchWorker(self.dataset, c, self.collate_fn, self.prefetch_factor)
            w.start()
            workers.append(w)
        # Round-robin merge of worker outputs
        active = list(workers)
        buffers = [w.get() for w in workers]
        while any(b is not None for b in buffers):
            for wi, b in enumerate(buffers):
                if b is None:
                    continue
                yield b
                buffers[wi] = workers[wi].get()
        for w in workers:
            w.stop()

    def _pin(self, batch: Any) -> Any:
        if isinstance(batch, np.ndarray):
            return np.ascontiguousarray(batch)
        return batch

    def set_epoch(self, epoch: int):
        """Advance the RNG seed (used with distributed samplers)."""
        self._seed = epoch
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


class StreamingTokenDataset(Dataset):
    """Stream a text file token-by-token in sliding ``seq_len`` windows.

    Suitable for online pretraining where the corpus exceeds RAM; only one
    window is materialized per access.
    """

    def __init__(self, path: str, seq_len: int = 2048, tokenizer=None):
        self.path = path
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        with open(path, "r", encoding="utf-8") as f:
            self.text = f.read()
        if tokenizer is not None:
            self.tokens = tokenizer.encode(self.text)
        else:
            self.tokens = [ord(c) for c in self.text]
        self.num_samples = max(0, len(self.tokens) - seq_len)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> np.ndarray:
        return np.array(self.tokens[idx : idx + self.seq_len], dtype=np.int32)
