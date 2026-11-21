"""
SNEPPX Model Hub — API Client

HTTP client for talking to the SNEPPX Model Hub backend.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlsplit

try:
    import requests
except ImportError:
    requests = None

from ..api.models import (
    APIKey,
    BenchmarkResult,
    Leaderboard,
    LeaderboardEntry,
    ModelCard,
    ModelFormat,
    ModelTask,
    ModelVersion,
    ModelVisibility,
    SemVer,
)


class HubAPIError(Exception):
    """Raised when the hub API returns an error."""
    pass


class HubAPIClient:
    """HTTP client for the SNEPPX Model Hub API."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        try:
            parsed_url = urlsplit(base_url)
            hostname = (parsed_url.hostname or "").lower()
        except ValueError as exc:
            raise ValueError("base_url must be a valid HTTP(S) URL") from exc
        if parsed_url.scheme not in {"http", "https"} or not hostname:
            raise ValueError("base_url must be a valid HTTP(S) URL")

        self._api_key = api_key or os.environ.get("SNEPPX_HUB_API_KEY")
        is_loopback = hostname in {"localhost", "127.0.0.1", "::1"}
        if self._api_key and parsed_url.scheme != "https" and not is_loopback:
            raise ValueError("refusing to send a Hub API key over non-local HTTP; use HTTPS")

        self._base = base_url.rstrip("/")
        self._session = requests.Session() if requests else None
        if self._api_key and self._session:
            self._session.headers.update({"Authorization": f"Bearer {self._api_key}"})

    def _request(self, method: str, path: str, **kwargs) -> Any:
        if self._session is None:
            raise ImportError("requests library is required: pip install requests")
        url = f"{self._base}{path}"
        kwargs.setdefault("timeout", 60)
        resp = self._session.request(method, url, **kwargs)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise HubAPIError(f"API error {resp.status_code}: {detail}")
        if resp.headers.get("content-type", "").startswith("application/json"):
            return resp.json()
        return resp.content

    @staticmethod
    def _model_segment(name: str) -> str:
        """Convert a model name for use in a URL path segment."""
        return name.replace("/", "_").replace("\\", "_")

    # ---- Model operations ----

    def list_models(
        self,
        query: Optional[str] = None,
        task: Optional[str] = None,
        tag: Optional[str] = None,
        organization: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "downloads",
    ) -> List[ModelCard]:
        params = {"page": page, "page_size": page_size, "sort_by": sort_by}
        if query:
            params["q"] = query
        if task:
            params["task"] = task
        if tag:
            params["tag"] = tag
        if organization:
            params["org"] = organization

        data = self._request("GET", "/api/v1/models", params=params)
        return [ModelCard(**item) for item in data.get("items", [])]

    def get_model(self, name: str, version: str = "latest") -> ModelCard:
        data = self._request("GET", f"/api/v1/models/{self._model_segment(name)}/{version}")
        return ModelCard(**data)

    def get_model_versions(self, name: str) -> List[ModelVersion]:
        data = self._request("GET", f"/api/v1/models/{self._model_segment(name)}/versions")
        items = []
        for item in data.get("versions", []):
            version = item.get("version") if isinstance(item, dict) else item
            try:
                items.append(ModelVersion(**item))
            except Exception:
                items.append(ModelVersion(
                    name=name,
                    version=version,
                    format=ModelFormat.SNEPPX_NATIVE,
                    task=ModelTask.TEXT_GENERATION,
                ))
        return items

    def get_model_card(self, name: str, version: str = "latest") -> ModelCard:
        return self.get_model(name, version)

    def search(
        self,
        query: str,
        task: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[ModelCard]:
        params = {"q": query, "limit": limit}
        if task:
            params["task"] = task
        if tags:
            params["tags"] = ",".join(tags)

        data = self._request("GET", "/api/v1/search", params=params)
        return [ModelCard(**item) for item in data.get("results", [])]

    # ---- Upload ----

    def upload_model(
        self,
        model_card: ModelCard,
        files: Dict[str, str],  # filename -> local path
        update: bool = False,
    ) -> ModelCard:
        """Upload a model with its metadata."""
        meta = model_card.to_public_dict()
        meta["files"] = []

        for filename, local_path in files.items():
            size = os.path.getsize(local_path)
            meta["files"].append({
                "filename": filename,
                "size": size,
                "sha256": "",  # Will be computed server-side
                "is_lfs": size >= 1024 * 1024 * 1024,  # 1GB threshold
            })

        resp = self._request("POST", "/api/v1/models/upload", json=meta)
        return ModelCard(**resp)

    def upload_file(self, model_name: str, version: str, filename: str, local_path: str) -> str:
        """Upload a single model file (for large files)."""
        import io

        size = os.path.getsize(local_path)
        is_lfs = size >= 1024 * 1024 * 1024

        with open(local_path, "rb") as f:
            # Use streaming upload
            files = {"file": (filename, f, "application/octet-stream")}
            data = self._request("POST", f"/api/v1/models/{self._model_segment(model_name)}/{version}/files/{filename}", files=files)
            if isinstance(data, dict):
                return data.get("url", "")
            return str(data)

    # ---- Auth / User ----

    def verify_api_key(self) -> Optional[str]:
        """Verify the current API key and return the username."""
        try:
            data = self._request("GET", "/api/v1/auth/verify")
            return data.get("username")
        except HubAPIError:
            return None

    def create_api_key(self, username: str, scopes: Optional[List[str]] = None) -> str:
        """Create a new API key (admin only)."""
        data = self._request("POST", "/api/v1/keys", json={
            "username": username,
            "scopes": scopes or ["read", "write", "models"],
        })
        return data["api_key"]

    # ---- Leaderboard ----

    def get_leaderboard(
        self,
        task: Optional[str] = None,
        metric: Optional[str] = None,
        dataset: Optional[str] = None,
        limit: int = 100,
    ) -> Leaderboard:
        params = {"limit": limit}
        if task:
            params["task"] = task
        if metric:
            params["metric"] = metric
        if dataset:
            params["dataset"] = dataset

        data = self._request("GET", "/api/v1/leaderboard", params=params)
        entries = [LeaderboardEntry(**e) for e in data.get("entries", [])]
        return Leaderboard(
            task=data.get("task", task or ModelTask.TEXT_GENERATION),
            metric=data.get("metric", "accuracy"),
            dataset=data.get("dataset", "unknown"),
            entries=entries,
            higher_is_better=data.get("higher_is_better", True),
        )

    def submit_benchmark(self, result: BenchmarkResult) -> bool:
        """Submit a benchmark result to the leaderboard."""
        data = result.model_dump(mode="json")
        self._request("POST", "/api/v1/leaderboard/submit", json=data)
        return True

    # ---- Organizations ----

    def list_organizations(self) -> List[Dict[str, Any]]:
        return self._request("GET", "/api/v1/orgs")

    def get_organization(self, org_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/api/v1/orgs/{org_id}")
