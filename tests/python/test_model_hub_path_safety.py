"""Regression: model hub storage keys and card filenames must stay contained."""

import asyncio
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)

from model_hub.api.models import ModelFile
from model_hub.storage.backend import LocalStorageBackend, StorageConfig


def _backend(base):
    return LocalStorageBackend(StorageConfig(base_path=str(base)))


@pytest.mark.parametrize(
    "key",
    [
        "../outside.txt",
        "nested/../../outside.txt",
        "nested\\..\\..\\outside.txt",
        "/etc/passwd",
        "file:///etc/passwd",
        "C:\\Windows\\win.ini",
    ],
)
def test_full_path_rejects_escaping_keys(tmp_path, key):
    backend = _backend(tmp_path / "storage")
    with pytest.raises(ValueError):
        backend._full_path(key)


def test_internal_file_url_downloads_within_storage(tmp_path):
    base = tmp_path / "storage"
    backend = _backend(base)
    stored = asyncio.run(backend.upload_bytes(b"weights", "org/model/weights.bin"))
    dest = tmp_path / "downloaded.bin"

    asyncio.run(backend.download_file(stored.url, str(dest)))
    assert dest.read_bytes() == b"weights"


def test_file_url_outside_storage_is_rejected(tmp_path):
    base = tmp_path / "storage"
    backend = _backend(base)
    secret = tmp_path / "secret.txt"
    secret.write_text("sensitive", encoding="utf-8")
    dest = tmp_path / "stolen.txt"

    with pytest.raises(ValueError, match="storage root"):
        asyncio.run(backend.download_file(f"file://{secret}", str(dest)))
    assert not dest.exists()


@pytest.mark.parametrize(
    "filename", ["../secret.txt", "nested/file.bin", "C:\\secret.txt"]
)
def test_model_file_rejects_traversal_filename(filename):
    with pytest.raises(ValueError, match="plain basename"):
        ModelFile(filename=filename, sha256="0" * 64, size=1, url="weights.bin")


def test_model_file_accepts_plain_filename():
    f = ModelFile(
        filename="weights.bin", sha256="0" * 64, size=1, url="org/model/weights.bin"
    )
    assert f.filename == "weights.bin"
