"""Tests for storage backend and model serialization."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tempfile
import pytest

from model_hub.storage.backend import (
    StorageConfig, LocalStorageBackend, get_storage_backend,
)
from model_hub.converters import (
    save_sneppx_native, load_sneppx_native, detect_format,
    convert_pt_to_sneppx, convert_sneppx_to_hf,
)
from model_hub.utils import compute_sha256


class TestStorageBackend:
    def _get_backend(self):
        tmpdir = tempfile.mkdtemp()
        config = StorageConfig(base_path=tmpdir)
        return get_storage_backend(config), tmpdir

    async def _async_test(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_local_upload_download(self):
        import asyncio
        backend, tmpdir = self._get_backend()

        with tempfile.NamedTemporaryFile(delete=False, mode="wb", suffix=".bin") as f:
            f.write(b"test data for upload")
            local_path = f.name

        try:
            # Upload
            stored = asyncio.run(backend.upload_file(local_path, "test/model.bin"))
            assert stored.size > 0
            assert stored.is_lfs == False

            # Download
            dest = os.path.join(tmpdir, "downloaded.bin")
            asyncio.run(backend.download_file("test/model.bin", dest))
            with open(dest, "rb") as f:
                assert f.read() == b"test data for upload"

            # Check exists
            assert asyncio.run(backend.file_exists("test/model.bin"))

            # Get info
            info = asyncio.run(backend.get_file_info("test/model.bin"))
            assert info is not None
            assert info.size == len(b"test data for upload")

            # Delete
            assert asyncio.run(backend.delete_file("test/model.bin"))
            assert not asyncio.run(backend.file_exists("test/model.bin"))
        finally:
            os.unlink(local_path)

    def test_large_file_lfs_detection(self):
        config = StorageConfig(lfs_threshold=100)
        assert config.lfs_threshold == 100

    def test_s3_config(self):
        config = StorageConfig(
            backend="s3",
            bucket_name="my-models",
            endpoint_url="http://localhost:9000",
        )
        assert config.backend == "s3"
        assert config.bucket_name == "my-models"


class TestSNEPPXFormat:
    def test_save_and_load(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.sneppx.bin")

            # Create test tensors
            tensors = {
                "model.weight": np.random.randn(10, 10).astype(np.float32),
                "model.bias": np.random.randn(10).astype(np.float32),
            }

            # Save
            save_sneppx_native(path, tensors, {"test": True})

            # Verify it's detected
            assert detect_format(path) == "sneppx-native"

            # Load
            meta, loaded = load_sneppx_native(path)
            assert meta["test"] == True
            assert len(loaded) == 2
            assert "model.weight" in loaded

            # Verify data
            loaded_arr = np.frombuffer(loaded["model.weight"], dtype=np.float32).reshape(10, 10)
            np.testing.assert_array_equal(loaded_arr, tensors["model.weight"])


class TestConverters:
    def test_detect_formats(self):
        assert detect_format("model.pt") == "pytorch"
        assert detect_format("model.safetensors") == "safetensors"
        assert detect_format("model.bin") == "huggingface"
        assert detect_format("model.sneppx.bin") == "sneppx-native"
        assert detect_format("model.onnx") == "unknown"

    def test_detect_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                f.write("{}")
            with open(os.path.join(tmpdir, "pytorch_model.bin"), "wb") as f:
                f.write(b"test")
            assert detect_format(tmpdir) == "huggingface"
