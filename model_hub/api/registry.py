"""
SNEPPX Model Hub — Model Registry

High-level model registry that coordinates between the API client and
storage backend for model lifecycle management.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..api.models import (
    ModelCard,
    ModelFormat,
    ModelTask,
    ModelVersion,
    ModelVisibility,
    SemVer,
)
from ..api.models import BenchmarkResult, LeaderboardEntry
from ..storage.backend import StorageBackend, StorageConfig, StoredFile
from ..storage.backend import get_storage_backend
from ..utils import compute_sha256, ensure_dir, format_file_size


class ModelRegistry:
    """Centralized model registry with local + remote (API) support."""

    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        local_registry: Optional[str] = None,
        client=None,
    ):
        self._storage = storage or get_storage_backend(StorageConfig())
        self._local_registry = local_registry or os.path.join(
            os.path.expanduser("~"), ".sneppx", "hub", "registry"
        )
        self._client = client
        ensure_dir(self._local_registry)

    @property
    def storage(self) -> StorageBackend:
        return self._storage

    @property
    def client(self):
        return self._client

    def set_client(self, client):
        self._client = client

    def _model_dir(self, model_name: str, version: str) -> str:
        clean_name = model_name.replace("/", "_").replace("\\", "_")
        return os.path.join(self._local_registry, clean_name, version)

    # ---- Local operations ----

    def register_model(self, card: ModelCard, files_dir: str) -> ModelCard:
        """Register a model by uploading its files to local storage."""
        model_dir = self._model_dir(card.name, card.version)
        ensure_dir(model_dir)

        uploaded_files = []
        for filename in os.listdir(files_dir):
            src = os.path.join(files_dir, filename)
            if not os.path.isfile(src):
                continue

            from ..utils import compute_sha256 as _sha

            sha = _sha(src) if hasattr(_sha, "_coro") else _sha(src)
            dest_key = f"{card.name}/{card.version}/{filename}"

            import asyncio
            loop = asyncio.new_event_loop()
            try:
                stored = loop.run_until_complete(
                    self._storage.upload_file(src, dest_key)
                )
            finally:
                loop.close()

            uploaded_files.append({
                "filename": filename,
                "size": stored.size,
                "sha256": stored.sha256,
                "is_lfs": stored.is_lfs,
                "url": stored.url,
            })

        card.files = [
            ModelCard.model_fields["files"].__class__(**f) if False else _FileMeta(**f)
            for f in uploaded_files
        ]
        card.updated_at = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)

        # Save model card
        import json
        card_path = os.path.join(model_dir, "model_card.json")
        with open(card_path, "w") as f:
            json.dump(card.model_dump(mode="json"), f, indent=2)

        return card

    def get_model_local(self, name: str, version: str = "latest") -> Optional[ModelCard]:
        """Get a model from local registry."""
        if version == "latest":
            versions = self.list_versions_local(name)
            if not versions:
                return None
            version = versions[0].version

        model_dir = self._model_dir(name, version)
        card_path = os.path.join(model_dir, "model_card.json")
        if not os.path.exists(card_path):
            return None

        import json
        with open(card_path) as f:
            data = json.load(f)

        # Convert file dicts to ModelFile objects
        from ..api.models import ModelFile
        if "files" in data:
            data["files"] = [ModelFile(**f) for f in data["files"]]

        return ModelCard(**data)

    def list_versions_local(self, name: str) -> List[ModelVersion]:
        """List all versions of a model (local only)."""
        clean = name.replace("/", "_").replace("\\", "_")
        model_base = os.path.join(self._local_registry, clean)
        if not os.path.isdir(model_base):
            return []

        versions = []
        for ver in sorted(os.listdir(model_base), reverse=True):
            ver_dir = os.path.join(model_base, ver)
            if not os.path.isdir(ver_dir):
                continue
            card_path = os.path.join(ver_dir, "model_card.json")
            if os.path.exists(card_path):
                import json
                with open(card_path) as f:
                    data = json.load(f)
                versions.append(ModelVersion(**data))
        return versions

    def download_model(self, name: str, version: str = "latest", download_dir: Optional[str] = None) -> str:
        """Download a model's files to a local directory."""
        card = None
        # Prefer the remote card when a client is available (fresh files list)
        if self._client:
            try:
                card = self._client.get_model(name, version)
                self._save_remote_card(card)
            except Exception:
                card = None
        if card is None:
            card = self.get_model_local(name, version)
        if card is None and self._client:
            card = self._client.get_model(name, version)
            self._save_remote_card(card)

        if card is None:
            raise FileNotFoundError(f"Model not found: {name}@{version}")

        if download_dir is None:
            from ..utils import get_cache_dir
            download_dir = os.path.join(get_cache_dir(), "models", card.name.replace("/", "_"), card.version)

        ensure_dir(download_dir)

        import asyncio
        for f in card.files:
            dest = os.path.join(download_dir, f.filename)
            asyncio.run(self._storage.download_file(f.url or f.filename, dest))
            # Verify SHA256
            sha = compute_sha256(dest)
            if sha != f.sha256:
                raise ValueError(f"SHA256 mismatch for {f.filename}: expected {f.sha256}, got {sha}")

        # Save the model card
        import json
        card_path = os.path.join(download_dir, "model_card.json")
        with open(card_path, "w") as f:
            json.dump(card.model_dump(mode="json"), f, indent=2)

        return download_dir

    def _save_remote_card(self, card: ModelCard):
        """Save a model card received from the API to local registry."""
        model_dir = self._model_dir(card.name, card.version)
        ensure_dir(model_dir)
        import json
        card_path = os.path.join(model_dir, "model_card.json")
        with open(card_path, "w") as f:
            json.dump(card.model_dump(mode="json"), f, indent=2)

    # ---- Hybrid operations (remote + local) ----

    def get_model(self, name: str, version: str = "latest") -> Optional[ModelCard]:
        """Get a model, checking local first, then remote."""
        card = self.get_model_local(name, version)
        if card:
            return card
        if self._client:
            try:
                card = self._client.get_model(name, version)
                self._save_remote_card(card)
                return card
            except Exception:
                return None
        return None

    def list_models(
        self,
        query: Optional[str] = None,
        task: Optional[str] = None,
        tags: Optional[List[str]] = None,
        local_only: bool = False,
    ) -> List[ModelCard]:
        """List models, with optional filtering."""
        if not self._client or local_only:
            results = []
            for model_dir in os.listdir(self._local_registry):
                full = os.path.join(self._local_registry, model_dir)
                if not os.path.isdir(full):
                    continue
                for ver in os.listdir(full):
                    card = self.get_model_local(model_dir, ver)
                    if card:
                        results.append(card)
            return self._filter_models(results, query, task, tags)

        return self._filter_models(
            self._client.list_models(query=query, task=task, tag=tags[0] if tags else None, page_size=1000),
            query, task, tags
        )

    def _filter_models(
        self,
        models: List[ModelCard],
        query: Optional[str],
        task: Optional[str],
        tags: Optional[List[str]],
    ) -> List[ModelCard]:
        def _get(m, field, default=None):
            if isinstance(m, dict):
                v = m.get(field, default)
            else:
                v = getattr(m, field, default)
            if hasattr(v, "value"):
                v = v.value
            return v

        def _matches(m):
            if query:
                name = _get(m, "name", "") or (_get(m, "full_name", ""))
                desc = _get(m, "description", "")
                if query.lower() not in str(name).lower() and query.lower() not in str(desc).lower():
                    return False
            if task:
                m_task = _get(m, "task")
                if str(m_task).lower() != task.lower():
                    return False
            if tags:
                m_tags = _get(m, "tags", [])
                if not all(t in m_tags for t in tags):
                    return False
            return True

        return [m for m in models if _matches(m)]

    def list_versions(self, name: str) -> List[ModelVersion]:
        """List all versions of a model (checks local + remote)."""
        versions = self.list_versions_local(name)
        if self._client:
            try:
                remote_versions = self._client.get_model_versions(name)
                # Merge, avoiding duplicates
                existing = {v.version for v in versions}
                for rv in remote_versions:
                    if rv.version not in existing:
                        versions.append(rv)
            except Exception:
                pass
        # Sort by semver descending
        versions.sort(key=lambda v: SemVer(v.version), reverse=True)
        return versions

    def submit_benchmark(self, result: BenchmarkResult) -> bool:
        """Submit a benchmark result."""
        if self._client:
            return self._client.submit_benchmark(result)
        return False

    def get_leaderboard(self, **kwargs) -> List[LeaderboardEntry]:
        """Get leaderboard entries."""
        if self._client:
            lb = self._client.get_leaderboard(**kwargs)
            return lb.entries
        return []


# Helper class for file metadata when ModelFile isn't imported
class _FileMeta:
    """Lightweight file metadata."""

    def __init__(self, filename: str, size: int, sha256: str = "", is_lfs: bool = False, url: str = ""):
        self.filename = filename
        self.size = size
        self.sha256 = sha256
        self.is_lfs = is_lfs
        self.url = url
