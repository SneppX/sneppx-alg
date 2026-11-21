"""
SNEPPX Model Hub — Storage Backend Abstraction

Abstract interface for storage backends (local filesystem, S3, minio, etc.)
with LFS (Large File Support) awareness for files > 1GB.
"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StorageConfig:
    """Configuration for a storage backend."""
    backend: str = "local"  # "local", "s3", "minio", "gcs"
    base_path: str = ".sneppx/models"

    # S3 / minio specific
    endpoint_url: Optional[str] = None
    bucket_name: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    region: Optional[str] = None
    use_ssl: bool = True

    # LFS threshold (bytes) — files above this go through LFS-aware upload
    lfs_threshold: int = 1024 * 1024 * 1024  # 1 GB

    # Local cache for downloaded files
    cache_dir: Optional[str] = None

    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = os.path.join(self.base_path, ".cache")
        os.makedirs(self.cache_dir, exist_ok=True)


@dataclass
class StoredFile:
    """Metadata for a stored file."""
    key: str  # storage key / path
    size: int
    sha256: str
    is_lfs: bool = False
    url: Optional[str] = None
    storage_backend: str = "local"
    metadata: Dict[str, Any] = field(default_factory=dict)


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    async def upload_file(self, local_path: str, dest_key: str) -> StoredFile:
        """Upload a local file to storage."""
        ...

    @abstractmethod
    async def upload_bytes(self, data: bytes, dest_key: str) -> StoredFile:
        """Upload raw bytes to storage."""
        ...

    @abstractmethod
    async def download_file(self, key: str, dest_path: str) -> str:
        """Download a file from storage to a local path."""
        ...

    @abstractmethod
    async def download_bytes(self, key: str) -> bytes:
        """Download a file as bytes."""
        ...

    @abstractmethod
    async def delete_file(self, key: str) -> bool:
        """Delete a file from storage."""
        ...

    @abstractmethod
    async def get_file_info(self, key: str) -> Optional[StoredFile]:
        """Get metadata for a stored file."""
        ...

    @abstractmethod
    async def file_exists(self, key: str) -> bool:
        """Check if a file exists in storage."""
        ...

    @abstractmethod
    async def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Generate a presigned download URL."""
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        ...

    @property
    def lfs_threshold(self) -> int:
        return self._lfs_threshold

    def is_lfs_file(self, size: int) -> bool:
        """Check if a file should use LFS (>= threshold)."""
        return size >= self.lfs_threshold


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, config: StorageConfig):
        self._config = config
        self._lfs_threshold = config.lfs_threshold
        self._base = config.base_path
        os.makedirs(self._base, exist_ok=True)

    @property
    def backend_name(self) -> str:
        return "local"

    def _full_path(self, key: str) -> str:
        """Convert a storage key to a safe filesystem path."""
        # Prevent path traversal
        clean = os.path.normpath(key).lstrip("/")
        return os.path.join(self._base, clean)

    async def upload_file(self, local_path: str, dest_key: str) -> StoredFile:
        from ..utils import compute_sha256_async

        dest = self._full_path(dest_key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(local_path, dest)
        size = os.path.getsize(dest)
        sha = await compute_sha256_async(dest)

        return StoredFile(
            key=dest_key,
            size=size,
            sha256=sha,
            is_lfs=self.is_lfs_file(size),
            url=f"file://{dest}",
            storage_backend=self.backend_name,
        )

    async def upload_bytes(self, data: bytes, dest_key: str) -> StoredFile:
        import hashlib

        dest = self._full_path(dest_key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)

        return StoredFile(
            key=dest_key,
            size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            is_lfs=self.is_lfs_file(len(data)),
            url=f"file://{dest}",
            storage_backend=self.backend_name,
        )

    async def download_file(self, key: str, dest_path: str) -> str:
        src = self._full_path(key)
        if key.startswith("file://"):
            src = key[len("file://"):]
        if not os.path.exists(src):
            raise FileNotFoundError(f"File not found: {key}")
        shutil.copy2(src, dest_path)
        return dest_path

    async def download_bytes(self, key: str) -> bytes:
        src = self._full_path(key)
        if not os.path.exists(src):
            raise FileNotFoundError(f"File not found: {key}")
        with open(src, "rb") as f:
            return f.read()

    async def delete_file(self, key: str) -> bool:
        path = self._full_path(key)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    async def get_file_info(self, key: str) -> Optional[StoredFile]:
        from ..utils import compute_sha256_async

        path = self._full_path(key)
        if not os.path.exists(path):
            return None
        size = os.path.getsize(path)
        sha = await compute_sha256_async(path)
        return StoredFile(
            key=key,
            size=size,
            sha256=sha,
            is_lfs=self.is_lfs_file(size),
            url=f"file://{path}",
            storage_backend=self.backend_name,
        )

    async def file_exists(self, key: str) -> bool:
        return os.path.exists(self._full_path(key))

    async def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        path = self._full_path(key)
        return f"file://{path}"


class S3StorageBackend(StorageBackend):
    """S3 / MinIO compatible storage backend."""

    def __init__(self, config: StorageConfig):
        self._config = config
        self._lfs_threshold = config.lfs_threshold
        self._bucket = config.bucket_name or "sneppx-models"
        self._endpoint = config.endpoint_url or "https://s3.amazonaws.com"

        import boto3
        from botocore.client import Config as BotoConfig

        self._session = boto3.Session(
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            region_name=config.region or "us-east-1",
        )
        self._s3 = self._session.resource(
            "s3",
            endpoint_url=self._endpoint,
            config=BotoConfig(
                signature_version="s3v4",
                max_pool_connections=50,
                parameter_validation=False,
            ),
        )
        self._s3_client = self._s3.meta.client

    @property
    def backend_name(self) -> str:
        return "s3"

    async def upload_file(self, local_path: str, dest_key: str) -> StoredFile:
        from ..utils import compute_sha256_async

        size = os.path.getsize(local_path)
        is_lfs = self.is_lfs_file(size)

        if is_lfs:
            await self._multipart_upload(local_path, dest_key)
        else:
            self._s3_client.upload_file(local_path, self._bucket, dest_key)

        sha = await compute_sha256_async(local_path)
        url = await self.generate_presigned_url(dest_key, 86400)

        return StoredFile(
            key=dest_key,
            size=size,
            sha256=sha,
            is_lfs=is_lfs,
            url=url,
            storage_backend=self.backend_name,
        )

    async def upload_bytes(self, data: bytes, dest_key: str) -> StoredFile:
        import io
        import hashlib

        size = len(data)
        is_lfs = self.is_lfs_file(size)

        if is_lfs:
            await self._multipart_upload_bytes(data, dest_key)
        else:
            self._s3_client.upload_fileobj(io.BytesIO(data), self._bucket, dest_key)

        return StoredFile(
            key=dest_key,
            size=size,
            sha256=hashlib.sha256(data).hexdigest(),
            is_lfs=is_lfs,
            storage_backend=self.backend_name,
        )

    async def download_file(self, key: str, dest_path: str) -> str:
        self._s3_client.download_file(self._bucket, key, dest_path)
        return dest_path

    async def download_bytes(self, key: str) -> bytes:
        import io

        obj = self._s3_client.get_object(Bucket=self._bucket, Key=key)
        return obj["Body"].read()

    async def delete_file(self, key: str) -> bool:
        try:
            self._s3_client.delete_object(Bucket=self._bucket, Key=key)
            return True
        except Exception:
            return False

    async def get_file_info(self, key: str) -> Optional[StoredFile]:
        try:
            resp = self._s3_client.head_object(Bucket=self._bucket, Key=key)
            return StoredFile(
                key=key,
                size=resp["ContentLength"],
                sha256="",
                is_lfs=self.is_lfs_file(resp["ContentLength"]),
                url=await self.generate_presigned_url(key, 86400),
                storage_backend=self.backend_name,
                metadata={k: v for k, v in resp.get("Metadata", {}).items()},
            )
        except Exception:
            return None

    async def file_exists(self, key: str) -> bool:
        try:
            self._s3_client.head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception:
            return False

    async def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        return self._s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    async def _multipart_upload(self, local_path: str, dest_key: str, part_size: int = 128 * 1024 * 1024):
        """Upload using S3 multipart upload for large files."""
        import concurrent.futures

        file_size = os.path.getsize(local_path)
        upload_id = self._s3_client.create_multipart_upload(
            Bucket=self._bucket, Key=dest_key
        )["UploadId"]

        parts = []
        part_number = 1
        offset = 0

        with open(local_path, "rb") as f:
            while offset < file_size:
                end = min(offset + part_size, file_size)
                f.seek(offset)
                data = f.read(end - offset)
                parts.append({"PartNumber": part_number, "UploadPart": data})
                offset = end
                part_number += 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            etags = {}
            for part in parts:
                future = executor.submit(
                    self._s3_client.upload_part,
                    Bucket=self._bucket,
                    Key=dest_key,
                    PartNumber=part["PartNumber"],
                    UploadId=upload_id,
                    Body=part["UploadPart"],
                )
                futures.append((future, part["PartNumber"]))

            for future, part_num in futures:
                etags[part_num] = future.result()

        self._s3_client.complete_multipart_upload(
            Bucket=self._bucket,
            Key=dest_key,
            UploadId=upload_id,
            MultipartUpload={"Parts": [
                {"ETag": etags[i]["ETag"], "PartNumber": i} for i in sorted(etags.keys())
            ]},
        )

    async def _multipart_upload_bytes(self, data: bytes, dest_key: str, part_size: int = 128 * 1024 * 1024):
        """Upload bytes using S3 multipart upload."""
        import io
        import concurrent.futures

        upload_id = self._s3_client.create_multipart_upload(
            Bucket=self._bucket, Key=dest_key
        )["UploadId"]

        parts = []
        for i in range(0, len(data), part_size):
            parts.append(data[i:i + part_size])

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            etags = {}
            for i, part in enumerate(parts):
                future = executor.submit(
                    self._s3_client.upload_part,
                    Bucket=self._bucket,
                    Key=dest_key,
                    PartNumber=i + 1,
                    UploadId=upload_id,
                    Body=part,
                )
                futures.append((future, i + 1))

            for future, part_num in futures:
                etags[part_num] = future.result()

        self._s3_client.complete_multipart_upload(
            Bucket=self._bucket,
            Key=dest_key,
            UploadId=upload_id,
            MultipartUpload={"Parts": [
                {"ETag": etags[i]["ETag"], "PartNumber": i} for i in sorted(etags.keys())
            ]},
        )


class StorageBackendRegistry:
    """Factory for creating storage backends by name."""

    _BACKENDS: Dict[str, type] = {
        "local": LocalStorageBackend,
    }

    @classmethod
    def register(cls, name: str, backend_cls: type):
        cls._BACKENDS[name] = backend_cls

    @classmethod
    def create(cls, config: StorageConfig) -> StorageBackend:
        if config.backend in cls._BACKENDS:
            return cls._BACKENDS[config.backend](config)
        if config.backend in ("s3", "minio"):
            cls._BACKENDS[config.backend] = S3StorageBackend
            return S3StorageBackend(config)
        raise ValueError(f"Unknown storage backend: {config.backend}")


def get_storage_backend(config: StorageConfig) -> StorageBackend:
    """Get or create a storage backend instance."""
    return StorageBackendRegistry.create(config)
