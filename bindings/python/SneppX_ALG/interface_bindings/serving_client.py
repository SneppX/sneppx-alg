"""Thin Python client SDK for the SneppX serving API.

Uses only the standard library (``urllib``) so it works with no extra deps.
Talks to either the embedded FastAPI inference server (``inference_server.py``
once the /metrics /readyz /versions endpoints are wired in) or the lightweight
stdlib :class:`ServingServer`.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["ServingClient", "ServingError"]


class ServingError(RuntimeError):
    """Raised on HTTP / API errors."""


class ServingClient:
    """Minimal HTTP client for the SneppX serving API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8300",
        api_key: Optional[str] = None,
        timeout: float = 10.0,
    ):
        try:
            parsed_url = urlsplit(base_url)
            hostname = (parsed_url.hostname or "").lower()
        except ValueError as exc:
            raise ValueError("base_url must be a valid HTTP(S) URL") from exc
        if parsed_url.scheme not in {"http", "https"} or not hostname:
            raise ValueError("base_url must be a valid HTTP(S) URL")
        is_loopback = hostname in {"localhost", "127.0.0.1", "::1"}
        if api_key and parsed_url.scheme != "https" and not is_loopback:
            raise ValueError(
                "refusing to send a serving API key over non-local HTTP; use HTTPS"
            )

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    # ---- internal ----
    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        h = {"Accept": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if extra:
            h.update(extra)
        return h

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, Any]:
        url = self.base_url + path
        data = None
        hdrs = self._headers(headers)
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            hdrs["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                status = resp.status
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8") if e.fp else ""
            status = e.code
        try:
            parsed = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            parsed = {"raw": raw}
        if status >= 400:
            raise ServingError(f"{method} {path} -> {status}: {parsed!r}")
        return status, parsed

    # ---- convenience wrappers ----
    def health(self) -> Dict[str, Any]:
        _, data = self._request("GET", "/v1/health")
        return data or {}

    def healthz(self) -> bool:
        _, data = self._request("GET", "/healthz")
        return bool(data.get("status") == "ok")

    def readyz(self) -> bool:
        try:
            self._request("GET", "/readyz")
            return True
        except ServingError:
            return False

    def metrics(self, as_json: bool = False) -> Any:
        if as_json:
            _, data = self._request(
                "GET", "/metrics", headers={"Accept": "application/json"}
            )
            return data
        _, data = self._request("GET", "/metrics", headers={"Accept": "text/plain"})
        # the raw text path may not parse as json -> return the response object?
        # urllib already reads; for plain text we return status + raw via data["raw"]
        return data

    def models(self) -> List[Dict[str, Any]]:
        _, data = self._request("GET", "/v1/models/versions")
        return (data or {}).get("models", [])

    def versions(self, model: str) -> List[Dict[str, Any]]:
        _, data = self._request(
            "GET", "/v1/models/versions?model=" + urllib.parse.quote(model)
        )
        return (data or {}).get("versions", []) if data else []

    def deploy(
        self,
        model: str,
        version: str,
        description: str = "",
        weight: int = 0,
        promote: bool = False,
    ) -> Dict[str, Any]:
        body = {
            "model": model,
            "version": version,
            "description": description,
            "weight": weight,
            "promote": promote,
        }
        _, data = self._request("POST", "/v1/deploy", body=body)
        return data or {}

    def set_traffic(self, model: str, split: str) -> Dict[str, Any]:
        """Traffic split as ``"v1:70,v2:30"``."""
        _, data = self._request(
            "POST", "/v1/traffic", body={"model": model, "split": split}
        )
        return data or {}

    def generate(
        self,
        model: str,
        inputs: Any,
        request_id: Optional[str] = None,
        version: Optional[str] = None,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        body = {
            "model": model,
            "inputs": inputs,
            "request_id": request_id or f"{time.time_ns()}",
            "version": version,
        }
        _, data = self._request("POST", "/v1/generate", body=body)
        version = (data or {}).get("version")
        # strip the echoed version key for the caller
        out = {k: v for k, v in (data or {}).items() if k != "version"}
        return version, out

    # ---- polling helpers ----
    def wait_ready(self, timeout: float = 30.0, interval: float = 0.2) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.readyz():
                return True
            time.sleep(interval)
        return False
