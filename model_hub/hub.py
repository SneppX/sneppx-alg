"""
SNEPPX Model Hub — Python Bindings

Provides sneppx.hub.load() and high-level model management API.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

# Lazy imports - only load heavy deps when needed
try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = dict
    Field = None

from .api.client import HubAPIClient
from .api.models import (
    ModelCard,
    ModelFormat,
    ModelTask,
    ModelVersion,
    ModelVisibility,
    SemVer,
    BenchmarkResult,
)
from .api.registry import ModelRegistry
from .storage.backend import StorageConfig, get_storage_backend


def _get_default_url() -> str:
    return os.environ.get("SNEPPX_HUB_URL", "https://hub.sneppx.ai")


def _get_api_key() -> Optional[str]:
    return os.environ.get("SNEPPX_HUB_API_KEY")


# ---- Core API ----

class Hub:
    """Main hub client — manages models, auth, and storage."""

    _instance: Optional["Hub"] = None

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self._base_url = base_url or _get_default_url()
        self._api_key = api_key or _get_api_key()
        self._cache_dir = cache_dir or os.path.join(
            os.environ.get("SNEPPX_HUB_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "sneppx", "hub")),
            "models"
        )
        os.makedirs(self._cache_dir, exist_ok=True)

        self._client = HubAPIClient(base_url=self._base_url, api_key=self._api_key)
        self._registry = ModelRegistry(
            storage=get_storage_backend(StorageConfig(base_path=self._cache_dir)),
            local_registry=self._cache_dir,
            client=self._client,
        )

    @property
    def client(self) -> HubAPIClient:
        return self._client

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    @property
    def cache_dir(self) -> str:
        return self._cache_dir

    def list_models(self, query: Optional[str] = None, task: Optional[str] = None,
                    tag: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List available models with optional filtering."""
        params = {"page": 1, "page_size": limit, "sort_by": "downloads"}
        if query: params["q"] = query
        if task: params["task"] = task
        if tag: params["tag"] = tag
        result = self._client._request("GET", "/api/v1/models", params=params)
        return result.get("items", result)

    def search(self, query: str, task: Optional[str] = None,
               tags: Optional[List[str]] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Search for models by query string."""
        params = {"q": query, "limit": limit}
        if task: params["task"] = task
        if tags: params["tags"] = ",".join(tags)
        result = self._request("GET", "/api/v1/search", params=params)
        return result.get("results", [])

    def _request(self, method: str, path: str, **kwargs):
        return self._client._request(method, path, **kwargs)

    def get_model(self, name: str, version: str = "latest") -> ModelCard:
        """Get model card for a specific model."""
        return self._client.get_model(name, version)

    def get_versions(self, name: str) -> List[str]:
        """Get available versions for a model."""
        versions = self._client.get_model_versions(name)
        return [v.version for v in versions]

    def download(self, name: str, version: str = "latest", dest: Optional[str] = None) -> str:
        """Download model files to local storage."""
        return self._registry.download_model(name, version, dest)

    def upload(self, model_dir: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Upload a model directory to the hub."""
        model_dir = os.path.abspath(model_dir)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Directory not found: {model_dir}")

        card_path = os.path.join(model_dir, "model_card.json")
        if not os.path.exists(card_path):
            raise FileNotFoundError("model_card.json not found in model directory")

        with open(card_path) as f:
            card_data = json.load(f)

        # Upload metadata (server expects the card as a Form field)
        resp = self._client._request("POST", "/api/v1/models/upload",
                                     data={"card": json.dumps(card_data)})

        # Upload files
        org = card_data.get("organization")
        card_name = card_data.get("name")
        model_name = f"{org}/{card_name}" if org and org not in str(card_name) else card_name
        version = card_data.get("version")
        for filename in os.listdir(model_dir):
            if filename == "model_card.json" or filename.endswith(".py"):
                continue
            filepath = os.path.join(model_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, "rb") as f:
                    self._client._request(
                        "POST",
                        f"/api/v1/models/{self._client._model_segment(model_name)}/{version}/files/{filename}",
                        files={"file": (filename, f, "application/octet-stream")},
                    )

        return resp

    def submit_benchmark(self, model_name: str, version: str, task: str,
                         metric: str, score: float, dataset: str,
                         higher_is_better: bool = True) -> bool:
        """Submit a benchmark result to the leaderboard."""
        result = BenchmarkResult(
            model_name=model_name,
            version=version,
            task=ModelTask(task),
            metric=metric,
            score=score,
            dataset=dataset,
            higher_is_better=higher_is_better,
        )
        return self._client.submit_benchmark(result)

    def leaderboard(self, task: Optional[str] = None, metric: Optional[str] = None) -> List[Dict]:
        """Get leaderboard entries."""
        lb = self._client.get_leaderboard(task=task, metric=metric)
        return [e.model_dump(mode="json") for e in lb.entries]

    def list_organizations(self) -> List[Dict[str, Any]]:
        """List all organizations on the hub."""
        result = self._client._request("GET", "/api/v1/orgs")
        orgs = result.get("orgs", result)
        return orgs if isinstance(orgs, list) else []

    @classmethod
    def get_default(cls) -> "Hub":
        """Get or create the default hub client singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def load(model_name: str, version: str = "latest", task: Optional[str] = None,
         **kwargs) -> Any:
    """Load a model from the hub.

    Parameters
    ----------
    model_name : str
        Model identifier (e.g. "sneppx/llama-7b").
    version : str
        Semantic version string (e.g. "v1.0.0") or "latest".
    task : str, optional
        Task type to filter by.

    Returns
    -------
    A loaded model object or directory path with model files.
    """
    hub = Hub.get_default()
    card = hub.get_model(model_name, version)
    if card is None:
        raise FileNotFoundError(f"Model not found: {model_name}@{version}")

    download_dir = kwargs.get("download_dir") or kwargs.get("dest")
    result_dir = hub.download(model_name, version, download_dir)

    # If format is supported by the existing loader, use it
    fmt = card.format
    config_path = os.path.join(result_dir, "config.json")
    if fmt == ModelFormat.SNEPPX_NATIVE and os.path.exists(config_path):
        try:
            from SneppX_ALG.model_zoo import build_model_from_config
            with open(config_path) as f:
                config = json.load(f)
            return build_model_from_config(config)
        except ImportError:
            pass

    # Return the download directory for other formats
    return result_dir


def search(query: str, **kwargs) -> List[Dict[str, Any]]:
    """Search for models on the hub."""
    hub = Hub.get_default()
    return hub.search(query, **kwargs)


def list_models(**kwargs) -> List[Dict[str, Any]]:
    """List available models."""
    hub = Hub.get_default()
    return hub.list_models(**kwargs)


def get_model(model_name: str, version: str = "latest") -> ModelCard:
    """Get model card metadata."""
    hub = Hub.get_default()
    return hub.get_model(model_name, version)


def get_versions(model_name: str) -> List[str]:
    """List available versions for a model."""
    hub = Hub.get_default()
    return hub.get_versions(model_name)


def download(model_name: str, version: str = "latest", dest: Optional[str] = None) -> str:
    """Download model files."""
    hub = Hub.get_default()
    return hub.download(model_name, version, dest)


def upload(model_dir: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Upload a model directory to the hub."""
    hub = Hub.get_default()
    if api_key:
        hub._api_key = api_key
        hub._client = HubAPIClient(base_url=hub._base_url, api_key=api_key)
    return hub.upload(model_dir)


def leaderboard(task: Optional[str] = None, metric: Optional[str] = None) -> List[Dict]:
    """Get leaderboard entries."""
    hub = Hub.get_default()
    return hub.leaderboard(task, metric)


def list_organizations() -> List[Dict[str, Any]]:
    """List all organizations on the hub."""
    hub = Hub.get_default()
    return hub.list_organizations()


def submit_benchmark(model_name: str, version: str, **kwargs) -> bool:
    """Submit a benchmark result."""
    hub = Hub.get_default()
    return hub.submit_benchmark(model_name, version, **kwargs)


def login(api_key: str) -> bool:
    """Set the API key for authentication."""
    hub = Hub.get_default()
    hub._api_key = api_key
    hub._client = HubAPIClient(base_url=hub._base_url, api_key=api_key)
    return hub.client.verify_api_key() is not None
