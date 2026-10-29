import time
import struct
import json
import base64
import threading
import os
import tempfile
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List

from .crypto_sign import Ed25519
from .crypto_hash import Blake3

# Map between tensor dtype codes (matching bindings/python tensor Dtype) and numpy dtypes
_DTYPE_CODE_TO_NP = {
    0: np.float32,
    1: np.float64,
    2: np.float16,
    4: np.int32,
    5: np.int64,
    8: np.uint8,
}
_NP_TO_DTYPE_CODE = {v: k for k, v in _DTYPE_CODE_TO_NP.items()}

SNEPPX_CKPT_MAGIC = 0x41524958
SNEPPX_CKPT_MAGIC_HI = 0x434B5054
SNEPPX_CKPT_VERSION = 1


@dataclass
class CheckpointHeader:
    magic_lo: int = SNEPPX_CKPT_MAGIC
    magic_hi: int = SNEPPX_CKPT_MAGIC_HI
    version: int = SNEPPX_CKPT_VERSION
    num_tensors: int = 0
    metadata_offset: int = 0
    metadata_size: int = 0
    total_size: int = 0


@dataclass
class TensorRecord:
    shape: tuple = ()
    ndim: int = 0
    dtype: int = 0
    data_offset: int = 0
    data_size: int = 0
    strides: tuple = ()


CHECKPOINT_HEADER_FMT = "<IIIIQQQ32x"
CHECKPOINT_HEADER_SIZE = struct.calcsize(CHECKPOINT_HEADER_FMT)
TENSOR_RECORD_FMT = "<8QIIQQ8Q"
TENSOR_RECORD_SIZE = struct.calcsize(TENSOR_RECORD_FMT)  # 152 bytes


def _pack_header(h: CheckpointHeader) -> bytes:
    return struct.pack(
        CHECKPOINT_HEADER_FMT,
        h.magic_lo,
        h.magic_hi,
        h.version,
        h.num_tensors,
        h.metadata_offset,
        h.metadata_size,
        h.total_size,
    )


def _unpack_header(data: bytes) -> CheckpointHeader:
    vals = struct.unpack(CHECKPOINT_HEADER_FMT, data)
    return CheckpointHeader(
        magic_lo=vals[0],
        magic_hi=vals[1],
        version=vals[2],
        num_tensors=vals[3],
        metadata_offset=vals[4],
        metadata_size=vals[5],
        total_size=vals[6],
    )


def _pack_tensor_record(r: TensorRecord) -> bytes:
    shape_padded = [int(v) for v in r.shape] + [0] * (8 - len(r.shape))
    stride_padded = [int(v) for v in r.strides] + [0] * (8 - len(r.strides))
    return struct.pack(
        TENSOR_RECORD_FMT,
        *shape_padded,
        r.ndim,
        r.dtype,
        r.data_offset,
        r.data_size,
        *stride_padded,
    )


def _unpack_tensor_record(data: bytes, offset: int = 0) -> TensorRecord:
    vals = struct.unpack(TENSOR_RECORD_FMT, data[offset : offset + TENSOR_RECORD_SIZE])
    shape = tuple(v for v in vals[:8] if v != 0)
    strides = tuple(v for v in vals[12:] if v != 0)
    return TensorRecord(
        shape=shape,
        ndim=vals[8],
        dtype=vals[9],
        data_offset=vals[10],
        data_size=vals[11],
        strides=strides,
    )


class CheckpointWriter:
    def __init__(self, path: str, signing_key: Optional[bytes] = None):
        self.path = path
        self.file = open(path, "wb")
        self.header = CheckpointHeader()
        self.records: List[TensorRecord] = []
        self._data_parts: List[bytes] = []
        self._meta = {}
        self._signing_key = signing_key

    def write_tensor(
        self, data: bytes, shape: tuple = (), dtype: int = 0, strides: tuple = ()
    ):
        rec = TensorRecord(
            shape=shape,
            ndim=len(shape),
            dtype=dtype,
            data_offset=0,
            data_size=len(data),
            strides=strides,
        )
        self.records.append(rec)
        self._data_parts.append(data)

    def write_metadata(self, meta: dict):
        self._meta.update(meta)

    def close(self):
        # Layout: header | records | tensor_data | metadata [| signature]
        records_offset = CHECKPOINT_HEADER_SIZE
        data_offset = records_offset + len(self.records) * TENSOR_RECORD_SIZE

        # Calculate data offsets
        current_data_offset = data_offset
        for i, rec in enumerate(self.records):
            self.records[i].data_offset = current_data_offset
            current_data_offset += rec.data_size

        # Write header placeholder
        self.file.seek(0)
        self.file.write(_pack_header(self.header))

        # Write records + tensor data, capturing body bytes for hashing
        body = bytearray()
        for rec in self.records:
            rec_bytes = _pack_tensor_record(rec)
            self.file.write(rec_bytes)
            body.extend(rec_bytes)
        for part in self._data_parts:
            self.file.write(part)
            body.extend(part)

        # Integrity: BLAKE3(body) + Ed25519 signature over the digest
        body_digest = Blake3().hash(bytes(body))
        signature = None
        if self._signing_key is not None:
            signature = Ed25519.sign(self._signing_key, body_digest)

        # Write metadata (integrity folded into the JSON so the region is a
        # single valid document)
        if signature is not None:
            self._meta["_integrity"] = {
                "hash_alg": "blake3",
                "hash": body_digest.hex(),
                "sig_alg": "ed25519",
                "signature": base64.b64encode(signature).decode("ascii"),
            }
        meta_data = json.dumps(self._meta).encode("utf-8") if self._meta else b""
        meta_offset = self.file.tell()
        self.header.metadata_offset = meta_offset
        self.header.metadata_size = len(meta_data)
        if meta_data:
            self.file.write(meta_data)

        # Finalize header
        end_pos = self.file.tell()
        self.header.num_tensors = len(self.records)
        self.header.total_size = end_pos
        self.file.seek(0)
        self.file.write(_pack_header(self.header))
        self.file.close()


class CheckpointReader:
    def __init__(self, path: str, trusted_public_key: Optional[bytes] = None):
        self.path = path
        self.file = open(path, "rb")
        header_data = self.file.read(CHECKPOINT_HEADER_SIZE)
        self.header = _unpack_header(header_data)
        if (
            self.header.magic_lo != SNEPPX_CKPT_MAGIC
            or self.header.magic_hi != SNEPPX_CKPT_MAGIC_HI
        ):
            raise ValueError(f"Invalid checkpoint magic at {path}")
        if self.header.version > SNEPPX_CKPT_VERSION:
            raise ValueError(f"Unsupported version {self.header.version}")
        self.records: List[TensorRecord] = []
        for i in range(self.header.num_tensors):
            rec_data = self.file.read(TENSOR_RECORD_SIZE)
            if len(rec_data) < TENSOR_RECORD_SIZE:
                raise ValueError(f"Truncated tensor record {i}")
            rec = _unpack_tensor_record(rec_data, 0)
            self.records.append(rec)

        self.trusted_public_key = trusted_public_key
        self.verified = False
        self._verify_integrity()

    def _verify_integrity(self) -> None:
        """Recompute BLAKE3 over the body and verify the embedded Ed25519
        signature against ``trusted_public_key``. Fail-closed: a configured
        trusted key with a missing or invalid signature raises ValueError."""
        if self.trusted_public_key is None:
            return

        body = bytearray()
        for rec in self.records:
            body.extend(_pack_tensor_record(rec))
            self.file.seek(rec.data_offset)
            body.extend(self.file.read(rec.data_size))

        body_digest = Blake3().hash(bytes(body))

        meta = {}
        if self.header.metadata_size > 0:
            self.file.seek(self.header.metadata_offset)
            raw = self.file.read(self.header.metadata_size)
            try:
                meta = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                meta = {}

        integrity = meta.get("_integrity") if isinstance(meta, dict) else None
        signature_b64 = None
        stored_hash = None
        if isinstance(integrity, dict):
            signature_b64 = integrity.get("signature")
            stored_hash = integrity.get("hash")

        if signature_b64 is None or stored_hash is None:
            raise ValueError(
                "Checkpoint is signed but no integrity record found; refusing to load"
            )

        if stored_hash != body_digest.hex():
            raise ValueError("Checkpoint integrity hash mismatch; refusing to load")

        try:
            signature = base64.b64decode(signature_b64)
        except Exception:
            signature = b""
        if not Ed25519.verify(self.trusted_public_key, body_digest, signature):
            raise ValueError("Checkpoint signature verification failed; refusing to load")

        self.verified = True

    def read_tensor(self, idx: int) -> bytes:
        rec = self.records[idx]
        self.file.seek(rec.data_offset)
        return self.file.read(rec.data_size)

    def read_metadata(self) -> dict:
        if self.header.metadata_size == 0:
            return {}
        self.file.seek(self.header.metadata_offset)
        data = self.file.read(self.header.metadata_size)
        return json.loads(data.decode("utf-8"))

    def close(self):
        self.file.close()


def validate_checkpoint(path: str, trusted_public_key: Optional[bytes] = None) -> bool:
    try:
        r = CheckpointReader(path, trusted_public_key=trusted_public_key)
        r.close()
        return True
    except Exception:
        return False


class CheckpointCoordinator:
    def __init__(
        self,
        checkpoint_dir: str,
        world_size: int = 1,
        rank: int = 0,
        save_interval: int = 100,
        keep_last: int = 5,
        async_save: bool = True,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.world_size = world_size
        self.rank = rank
        self.save_interval = save_interval
        self.keep_last = keep_last
        self.async_save = async_save
        self.current_step = 0
        self.last_checkpoint_path: Optional[str] = None
        self._save_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _build_path(self, step: int) -> str:
        return os.path.join(
            self.checkpoint_dir, f"checkpoint_{step}_rank_{self.rank}.sneppx"
        )

    def _do_async_save(
        self,
        path: str,
        state_bytes: bytes,
        step: int,
        meta: dict,
        shape: tuple,
        dtype_code: int,
    ):
        w = CheckpointWriter(path)
        w.write_tensor(state_bytes, shape=shape, dtype=dtype_code)
        w.write_metadata(meta)
        w.close()
        if self.rank == 0:
            print(f"[SNEPPX Checkpoint] Saved to {path} (step {step})")

    def save(
        self,
        state,
        step: int,
        metadata: Optional[dict] = None,
        barrier_fn: Optional[Callable] = None,
    ):
        if step % self.save_interval != 0:
            return
        self.current_step = step
        path = self._build_path(step)
        if self._save_thread and self._save_thread.is_alive():
            self._save_thread.join()
        meta = {"step": step, "world_size": self.world_size, "rank": self.rank}
        if metadata:
            meta.update(metadata)

        if isinstance(state, np.ndarray):
            dtype_code = _NP_TO_DTYPE_CODE.get(np.dtype(state.dtype), 0)
            shape = tuple(state.shape)
            state_bytes = state.tobytes()
        else:
            state_bytes = bytes(state)
            shape = (len(state_bytes),)
            dtype_code = 8  # np.uint8 — round-trips raw bytes losslessly
            meta["_raw_bytes"] = True

        if self.async_save:
            self._save_thread = threading.Thread(
                target=self._do_async_save,
                args=(path, state_bytes, step, meta, shape, dtype_code),
            )
            self._save_thread.start()
        else:
            self._do_async_save(path, state_bytes, step, meta, shape, dtype_code)

        self.last_checkpoint_path = path
        self._cleanup_old()

    def _cleanup_old(self):
        import glob
        import re

        pattern = os.path.join(
            self.checkpoint_dir, f"checkpoint_*_rank_{self.rank}.sneppx"
        )
        files = glob.glob(pattern)

        def _step_from_path(p):
            m = re.search(r"checkpoint_(\d+)_rank_", os.path.basename(p))
            return int(m.group(1)) if m else 0

        files = sorted(files, key=_step_from_path)
        while len(files) > self.keep_last:
            os.remove(files.pop(0))

    def load(self) -> tuple:
        path = self.last_checkpoint_path or self._build_path(self.current_step)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
        r = CheckpointReader(path)
        rec = r.records[0]
        data_bytes = r.read_tensor(0)
        np_dtype = _DTYPE_CODE_TO_NP.get(rec.dtype, np.float32)
        data = np.frombuffer(data_bytes, dtype=np_dtype).reshape(rec.shape).copy()
        meta = r.read_metadata()
        r.close()
        self.current_step = meta.get("step", 0)
        if self.rank == 0:
            print(f"[SNEPPX Checkpoint] Loaded from {path} (step {self.current_step})")
        if meta.get("_raw_bytes"):
            return bytes(data), meta
        return data, meta

    def coordinated_save(
        self,
        state: bytes,
        step: int,
        metadata: Optional[dict] = None,
        barrier_fn: Optional[Callable] = None,
    ):
        if barrier_fn:
            barrier_fn()
        self.save(state, step, metadata)
        if barrier_fn:
            barrier_fn()

    def destroy(self):
        if self._save_thread and self._save_thread.is_alive():
            self._save_thread.join()


@dataclass
class HeartbeatStatus:
    ALIVE = 0
    SUSPECT = 1
    DEAD = 2
    JOIN = 3
    LEAVE = 4


class HeartbeatMonitor:
    def __init__(
        self,
        world_size: int,
        rank: int,
        interval_ms: int = 1000,
        timeout_ms: int = 5000,
    ):
        self.world_size = world_size
        self.rank = rank
        self.interval_ms = interval_ms
        self.timeout_ms = timeout_ms
        self._status = [HeartbeatStatus.ALIVE] * world_size
        self._last_seen = [time.monotonic_ns()] * world_size
        self._missed = [0] * world_size
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _monitor_loop(self):
        while self._running:
            self.check_alive()
            time.sleep(self.interval_ms / 1000.0)

    def report_alive(self, rank: int):
        with self._lock:
            self._last_seen[rank] = time.monotonic_ns()
            self._missed[rank] = 0
            self._status[rank] = HeartbeatStatus.ALIVE

    def check_alive(self) -> int:
        now = time.monotonic_ns()
        alive = 0
        with self._lock:
            for i in range(self.world_size):
                if i == self.rank:
                    alive += 1
                    continue
                elapsed_ms = (now - self._last_seen[i]) / 1_000_000
                if elapsed_ms > self.timeout_ms:
                    self._missed[i] += 1
                    if self._missed[i] >= 3:
                        self._status[i] = HeartbeatStatus.DEAD
                    else:
                        self._status[i] = HeartbeatStatus.SUSPECT
                else:
                    self._status[i] = HeartbeatStatus.ALIVE
                    alive += 1
        return alive

    def get_alive_ranks(self) -> List[int]:
        with self._lock:
            return [
                i
                for i in range(self.world_size)
                if self._status[i] == HeartbeatStatus.ALIVE
            ]

    def get_status(self) -> List[int]:
        with self._lock:
            return list(self._status)


class ElasticTrainer:
    def __init__(self, world_size: int, rank: int, max_restarts: int = 3):
        self.world_size = world_size
        self.rank = rank
        self.max_restarts = max_restarts
        self.restart_count = 0
        self.version = 1
        self._alive_ranks = [True] * world_size
        self._state = "OK"
        self.barrier_fn: Optional[Callable] = None
        self.checkpoint_restore_fn: Optional[Callable] = None

    def handle_join(self, new_rank: int):
        if new_rank >= self.world_size:
            old_size = self.world_size
            self.world_size = new_rank + 1
            self._alive_ranks.extend([False] * (self.world_size - old_size))
            self._alive_ranks[old_size:] = [False] * (self.world_size - old_size)
        self._alive_ranks[new_rank] = True
        self.version += 1

    def handle_leave(self, leaving_rank: int):
        self._alive_ranks[leaving_rank] = False
        self.version += 1
        if leaving_rank == self.rank:
            raise RuntimeError(f"This rank ({self.rank}) is leaving")
        alive = sum(self._alive_ranks)
        if alive < self.world_size // 2:
            raise RuntimeError(
                f"Less than half ranks alive ({alive}/{self.world_size})"
            )

    def handle_failure(self, failed_rank: int):
        self._alive_ranks[failed_rank] = False
        self.restart_count += 1
        if self.restart_count > self.max_restarts:
            raise RuntimeError(f"Max restarts ({self.max_restarts}) exceeded")

        if self.checkpoint_restore_fn:
            restore_ver = max(self.version - 1, 1)
            self.checkpoint_restore_fn(restore_ver)

        self.version += 1
        print(
            f"[SNEPPX Elastic] Rank {self.rank}: recovery attempt "
            f"{self.restart_count}/{self.max_restarts}, topology v{self.version}"
        )

    def reconfigure(self) -> int:
        if self.barrier_fn:
            self.barrier_fn()
        if self.checkpoint_restore_fn:
            self.checkpoint_restore_fn(self.version)
        alive = sum(self._alive_ranks)
        print(
            f"[SNEPPX Elastic] Reconfig complete: {alive}/{self.world_size} alive (v{self.version})"
        )
        return alive

    def get_new_topology(self) -> tuple:
        alive_count = 0
        my_new_rank = -1
        for i in range(self.world_size):
            if self._alive_ranks[i]:
                if i == self.rank:
                    my_new_rank = alive_count
                alive_count += 1
        new_world = max(alive_count, 1)
        new_rank = my_new_rank if my_new_rank >= 0 else self.rank
        return new_world, new_rank


class FaultToleranceManager:
    def __init__(
        self,
        world_size: int,
        rank: int,
        heartbeat_interval_ms: int = 1000,
        timeout_ms: int = 5000,
        max_restarts: int = 3,
    ):
        self.heartbeat = HeartbeatMonitor(
            world_size, rank, heartbeat_interval_ms, timeout_ms
        )
        self.elastic = ElasticTrainer(world_size, rank, max_restarts)
        self.world_size = world_size
        self.rank = rank

    def start(self):
        self.heartbeat.start()

    def stop(self):
        self.heartbeat.stop()

    def check_health(self) -> int:
        return self.heartbeat.check_alive()

    def handle_failure(self, failed_rank: int):
        print(f"[SNEPPX FT] Rank {self.rank} detected failure of rank {failed_rank}")
        self.elastic.handle_failure(failed_rank)
        new_world, new_rank = self.elastic.get_new_topology()
        self.elastic.reconfigure()
        print(
            f"[SNEPPX FT] Recovery: world={new_world} rank={new_rank} "
            f"(attempt {self.elastic.restart_count}/{self.elastic.max_restarts})"
        )
