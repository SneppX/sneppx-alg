"""Tests for hub API client and registry (mocked)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tempfile
import json

from model_hub.api.models import (
    ModelCard, ModelFormat, ModelTask, ModelVersion, SemVer,
)
from model_hub.api.registry import ModelRegistry
from model_hub.storage.backend import StorageConfig


class MockClient:
    """Mock API client for testing."""
    def __init__(self):
        self.calls = []
        self.models = [
            {"name": "sneppx/test-model", "version": "v1.0.0", "description": "A test model",
             "architecture": "Transformer", "format": "sneppx-native", "task": "text-generation",
             "license": "mit", "visibility": "public", "tags": ["test"], "total_size": 1024,
             "created_at": "2024-01-01T00:00:00Z"},
            {"name": "sneppx/test-model", "version": "v1.1.0", "description": "Updated test model",
             "architecture": "Transformer", "format": "sneppx-native", "task": "text-generation",
             "license": "mit", "visibility": "public", "tags": ["test"], "total_size": 2048,
             "created_at": "2024-06-01T00:00:00Z"},
        ]
        self.versions = ["v1.0.0", "v1.1.0"]

    def _request(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        if path == "/api/v1/models":
            return {"items": self.models, "total": len(self.models)}
        if path.startswith("/api/v1/models/"):
            # Parse path: /api/v1/models/{name}/{version}
            parts = path[len("/api/v1/models/"):].split("/")
            if len(parts) == 2:
                name, ver = parts
            elif len(parts) == 1:
                name = parts[0]
                ver = "latest"
            else:
                ver = "latest"

            if ver == "versions":
                return {"versions": [{"version": v} for v in self.versions]}

            model = None
            if ver == "latest":
                model = self.models[-1]
            else:
                for m in self.models:
                    if m["version"] == ver:
                        model = m
                        break
            if model:
                return model
        return {}

    def get_model(self, name, version="latest"):
        if name == "sneppx/test-model":
            if version == "latest" or not version:
                model = self.models[-1]
            else:
                model = next((m for m in self.models if m["version"] == version), None)
            if model:
                return ModelCard(**model)
        return None

    def get_model_versions(self, name):
        if name == "sneppx/test-model":
            return [ModelVersion(name=name, version=v, format=ModelFormat.SNEPPX_NATIVE,
                                 task=ModelTask.TEXT_GENERATION) for v in self.versions]
        return []

    def list_models(self, query=None, task=None, tag=None, page_size=1000):
        """List models with filtering."""
        result = self.models
        if query:
            result = [m for m in result if query.lower() in m.get("name", "").lower() or query.lower() in m.get("description", "").lower()]
        if task:
            result = [m for m in result if m.get("task") == task]
        if tag:
            result = [m for m in result if tag in (m.get("tags") or [])]
        # Return ModelCard objects like the real client does
        return [ModelCard(**m) if isinstance(m, dict) else m for m in result]


class TestModelRegistry:
    def _make_registry(self):
        tmpdir = tempfile.mkdtemp()
        config = StorageConfig(base_path=tmpdir)
        client = MockClient()
        reg = ModelRegistry(
            storage=None,  # Will use default local
            local_registry=tmpdir,
            client=client,
        )
        return reg, tmpdir, client

    def test_list_models(self):
        reg, tmpdir, client = self._make_registry()
        models = reg.list_models(query="test")
        assert len(models) == 2

    def test_get_model_from_cache(self):
        reg, tmpdir, client = self._make_registry()
        # First call should hit API and cache
        card = reg.get_model("sneppx/test-model", "v1.0.0")
        assert card is not None
        assert card.version == "v1.0.0"

        # Second call should use local cache
        card2 = reg.get_model_local("sneppx/test-model", "v1.0.0")
        assert card2 is not None
        assert card2.version == "v1.0.0"

    def test_list_versions(self):
        reg, tmpdir, client = self._make_registry()
        # Need to call get_model first to cache it
        reg.get_model("sneppx/test-model", "v1.0.0")
        versions = reg.list_versions("sneppx/test-model")
        # Should have remote versions
        versions_str = [v.version for v in versions]
        assert "v1.0.0" in versions_str
        assert "v1.1.0" in versions_str

    def test_filter_by_task(self):
        reg, tmpdir, client = self._make_registry()
        models = reg.list_models(task="text-generation")
        assert len(models) == 2

    def test_filter_by_tags(self):
        reg, tmpdir, client = self._make_registry()
        models = reg.list_models(tags=["test"])
        assert len(models) == 2
