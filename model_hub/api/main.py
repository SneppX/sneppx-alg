"""
SNEPPX Model Hub — FastAPI Backend

Provides a REST API for model discovery, upload, download, and leaderboard
management. Supports S3/minio storage, API key authentication, and LFS
for large weight files.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Annotated
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Query, Header, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..api.models import (
    APIKey,
    BenchmarkResult,
    Leaderboard,
    LeaderboardEntry,
    ModelCard,
    ModelFile,
    ModelFormat,
    ModelTask,
    ModelVersion,
    ModelVisibility,
    Org,
    User,
)
from ..storage.backend import StorageConfig, get_storage_backend, StorageBackend
from ..utils import compute_sha256, ensure_dir, format_file_size


# ---- Database (file-based for now) ----

class HubDatabase:
    """File-based database for the model hub.

    Stores model metadata in a directory structure:
      data/
        models/
          <org>/
            <model>/
              <version>/
                model_card.json
        users.json
        keys.json
        leaderboard.json
        orgs.json
    """

    def __init__(self, data_dir: str):
        self._dir = data_dir
        self._models_dir = os.path.join(data_dir, "models")
        self._users_file = os.path.join(data_dir, "users.json")
        self._keys_file = os.path.join(data_dir, "keys.json")
        self._leaderboard_file = os.path.join(data_dir, "leaderboard.json")
        self._orgs_file = os.path.join(data_dir, "orgs.json")
        ensure_dir(self._dir)
        ensure_dir(self._models_dir)
        self._load_json_store()

    def _load_json_store(self):
        from ..utils import ensure_dir
        self._users: List[Dict] = self._load_json(self._users_file, default=[])
        self._keys: List[Dict] = self._load_json(self._keys_file, default=[])
        self._leaderboard: List[Dict] = self._load_json(self._leaderboard_file, default=[])
        self._orgs: List[Dict] = self._load_json(self._orgs_file, default=[])

    def _load_json(self, path: str, default: Any = None) -> Any:
        if not os.path.exists(path):
            return default
        import json
        with open(path) as f:
            return json.load(f)

    def _save_json(self, path: str, data: Any):
        import json
        ensure_dir(os.path.dirname(path))
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _model_card_path(self, name: str, version: str) -> str:
        clean_name = name.replace("/", "_").replace("\\", "_")
        return os.path.join(self._models_dir, clean_name, version, "model_card.json")

    # ---- Models ----

    def list_models(self, page: int = 1, page_size: int = 20, sort_by: str = "downloads",
                    query: Optional[str] = None, task: Optional[str] = None,
                    tag: Optional[str] = None, org: Optional[str] = None) -> Dict[str, Any]:
        all_models = self._scan_models()

        # Filter
        if query:
            all_models = [m for m in all_models if query.lower() in m.get("name", "").lower()]
        if task:
            all_models = [m for m in all_models if m.get("task") == task]
        if tag:
            all_models = [m for m in all_models if tag in (m.get("tags") or [])]
        if org:
            all_models = [m for m in all_models if m.get("organization") == org or m.get("name", "").startswith(org + "/")]

        total = len(all_models)

        # Sort
        if sort_by == "downloads":
            all_models.sort(key=lambda m: m.get("_downloads", 0), reverse=True)
        elif sort_by == "created_at":
            all_models.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        elif sort_by == "name":
            all_models.sort(key=lambda m: m.get("name", ""))

        # Paginate
        start = (page - 1) * page_size
        end = start + page_size
        items = all_models[start:end]

        return {"items": items, "total": total, "page": page, "page_size": page_size}

    def _scan_models(self) -> List[Dict]:
        """Scan the models directory for all model cards."""
        results = []
        if not os.path.isdir(self._models_dir):
            return results
        for model_name in os.listdir(self._models_dir):
            model_path = os.path.join(self._models_dir, model_name)
            if not os.path.isdir(model_path):
                continue
            for version in os.listdir(model_path):
                card_path = os.path.join(model_path, version, "model_card.json")
                if os.path.exists(card_path):
                    import json
                    with open(card_path) as f:
                        data = json.load(f)
                    data["name"] = model_name.replace("_", "/").replace("__", "/")  # restore org/name
                    # Normalize the name back properly
                    if "/" in data.get("name", ""):
                        pass  # Already has slash
                    else:
                        # Convert _ to / for org separation
                        parts = model_name.split("_", 1)
                        if len(parts) == 2:
                            data["name"] = f"{parts[0]}/{parts[1]}"
                        else:
                            data["name"] = model_name
                    data["_model_name"] = model_name
                    data["_version"] = version
                    data["_downloads"] = 0  # Would come from usage tracking
                    data["_size"] = sum(f.get("size", 0) for f in (data.get("files") or []))
                    results.append(data)
        return results

    def get_model_card(self, name: str, version: str = "latest") -> Optional[Dict]:
        if version == "latest":
            cards = self._scan_models()
            clean = name.replace("/", "_").replace("\\", "_")
            model_cards = [c for c in cards if c.get("_model_name") == clean or c.get("name") == name]
            if not model_cards:
                return None
            # Sort by version descending (latest first)
            from ..api.models import SemVer
            model_cards.sort(key=lambda c: SemVer(c["_version"]), reverse=True)
            version = model_cards[0]["_version"]

        clean_name = name.replace("/", "_").replace("\\", "_")
        path = self._model_card_path(name, version)
        card = self._load_json(path)
        if card:
            stored_name = card.get("name")
            org = card.get("organization")
            if org and stored_name and org not in str(stored_name):
                card["name"] = f"{org}/{stored_name}"
            elif stored_name and "/" in str(stored_name):
                card["name"] = stored_name
            else:
                card["name"] = name
            card["version"] = version
        # Compute total_size from recorded files
        card["total_size"] = sum(f.get("size", 0) for f in (card.get("files") or []))
        return card

    def get_model_versions(self, name: str) -> List[str]:
        from ..api.models import SemVer
        cards = self._scan_models()
        clean = name.replace("/", "_").replace("\\", "_")
        versions = []
        for c in cards:
            if c.get("_model_name") == clean or c.get("name") == name:
                versions.append(c["_version"])
        versions.sort(key=lambda v: SemVer(v), reverse=True)
        return versions

    def save_model_card(self, name: str, version: str, card: Dict) -> None:
        path = self._model_card_path(name, version)
        ensure_dir(os.path.dirname(path))
        self._save_json(path, card)

    def delete_model(self, name: str, version: Optional[str] = None) -> bool:
        """Delete a model or all versions of a model."""
        from pathlib import Path
        clean_name = name.replace("/", "_").replace("\\", "_")
        if version:
            path = Path(self._model_card_path(name, version)).parent
            if path.exists():
                shutil.rmtree(path)
                return True
            return False
        else:
            path = os.path.join(self._models_dir, clean_name)
            if os.path.exists(path):
                shutil.rmtree(path)
                return True
            return False

    # ---- Users / Auth ----

    def get_user(self, user_id: str) -> Optional[User]:
        for u in self._users:
            if u["id"] == user_id:
                return User(**u)
        return None

    def get_user_by_username(self, username: str) -> Optional[User]:
        for u in self._users:
            if u["username"] == username:
                return User(**u)
        return None

    def create_user(self, username: str, email: Optional[str] = None, is_admin: bool = False, org_id: Optional[str] = None) -> User:
        user = User(username=username, email=email, is_admin=is_admin, org_id=org_id)
        self._users.append(user.model_dump(mode="json"))
        self._save_json(self._users_file, self._users)
        return user

    # ---- API Keys ----

    def get_api_key(self, key_id: str) -> Optional[APIKey]:
        for k in self._keys:
            if k["key_id"] == key_id:
                return APIKey(**k)
        return None

    def create_api_key(self, user_id: str, org_id: Optional[str] = None, scopes: Optional[List[str]] = None) -> tuple[APIKey, str]:
        """Create an API key. Returns (key_obj, raw_key_string)."""
        key, _raw = APIKey.generate(user_id=user_id, org_id=org_id)
        key.scopes = scopes or ["read", "write", "models"]
        key_dict = key.model_dump(mode="json")
        # Store the raw key (in production, hash it)
        key_dict["raw_key"] = _raw
        self._keys.append(key_dict)
        self._save_json(self._keys_file, self._keys)
        return key, key_dict["raw_key"]

    def bootstrap_admin(self) -> Optional[str]:
        """Create the initial admin user + API key if none exist.

        Returns the raw API key if it was freshly created, else None.
        Uses SNEPPX_HUB_ADMIN_USER / SNEPPX_HUB_ADMIN_PASSWORD env vars
        (defaults: admin / admin).
        """
        import os
        from ..utils import hash_password

        if self._users:
            return None

        admin_user = os.environ.get("SNEPPX_HUB_ADMIN_USER", "admin")
        admin_pass = os.environ.get("SNEPPX_HUB_ADMIN_PASSWORD", "admin")
        user = self.create_user(username=admin_user, is_admin=True)
        user.hashed_api_key = hash_password(admin_pass)
        user_dict = user.model_dump(mode="json")
        for i, u in enumerate(self._users):
            if u["id"] == user.id:
                self._users[i] = user_dict
                break
        self._save_json(self._users_file, self._users)

        _, raw = self.create_api_key(user.id, None, ["read", "write", "models", "admin"])
        return raw

    def verify_api_key(self, api_key: str) -> Optional[User]:
        import hmac
        for k in self._keys:
            key_obj = APIKey(**k)
            if APIKey.verify(api_key, key_obj):
                # Update last_used
                k["last_used"] = datetime.now(timezone.utc).isoformat()
                self._save_json(self._keys_file, self._keys)
                return self.get_user(key_obj.user_id)
        return None

    # ---- Leaderboard ----

    def get_leaderboard(self, task: Optional[str] = None, metric: Optional[str] = None) -> List[LeaderboardEntry]:
        entries = []
        for e in self._leaderboard:
            if task and e.get("task") != task:
                continue
            if metric and e.get("metric") != metric:
                continue
            entries.append(LeaderboardEntry(**e))
        entries.sort(key=lambda e: e.score, reverse=True)
        for i, entry in enumerate(entries):
            entry.rank = i + 1
        return entries

    def add_leaderboard_entry(self, entry: BenchmarkResult) -> None:
        # Check if entry already exists
        for existing in self._leaderboard:
            if (existing["model_name"] == entry.model_name and
                existing["version"] == entry.version and
                existing["task"] == entry.task.value and
                existing["metric"] == entry.metric and
                existing["dataset"] == entry.dataset):
                # Update existing
                existing["score"] = entry.score
                existing["evaluated_at"] = entry.evaluated_at.isoformat()
                break
        else:
            self._leaderboard.append({
                "model_name": entry.model_name,
                "version": entry.version,
                "task": entry.task.value,
                "metric": entry.metric,
                "score": entry.score,
                "dataset": entry.dataset,
                "evaluated_at": entry.evaluated_at.isoformat(),
                "is_public": True,
            })
        self._save_json(self._leaderboard_file, self._leaderboard)


# ---- FastAPI app ----

def create_app(db_path: str, storage_config: Optional[StorageConfig] = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    storage_config = storage_config or StorageConfig()
    db = HubDatabase(db_path)
    storage = get_storage_backend(storage_config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        ensure_dir(db_path)
        ensure_dir(storage_config.base_path)
        fresh_key = db.bootstrap_admin()
        if fresh_key:
            print(f"[sneppx-hub] Created default admin user + API key: {fresh_key}")
        yield
        # Shutdown
        pass

    app = FastAPI(
        title="SNEPPX Model Hub",
        description="Centralized model registry for SNEPPX-Alg",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Auth dependency ----

    async def get_current_user(
        authorization: Annotated[Optional[str], Header()] = None,
        x_api_key: Annotated[Optional[str], Header()] = None,
    ) -> Optional[User]:
        api_key = None
        if authorization and authorization.startswith("Bearer "):
            api_key = authorization[7:]
        elif x_api_key:
            api_key = x_api_key

        if not api_key:
            return None  # Public endpoints allowed

        user = db.verify_api_key(api_key)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return user

    async def require_auth(user: Optional[User] = Depends(get_current_user)) -> User:
        if user is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        return user

    # ---- Models API ----

    @app.get("/api/v1/models")
    async def api_list_models(
        q: Optional[str] = Query(default=None),
        task: Optional[str] = Query(default=None),
        tag: Optional[str] = Query(default=None),
        org: Optional[str] = Query(default=None),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=100),
        sort_by: str = Query(default="downloads"),
    ):
        return db.list_models(
            page=page, page_size=page_size, sort_by=sort_by,
            query=q, task=task, tag=tag, org=org,
        )

    @app.get("/api/v1/models/{model_name}")
    async def api_get_model(model_name: str, version: str = Query(default="latest")):
        card = db.get_model_card(model_name, version)
        if not card:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}@{version}")
        return card

    @app.get("/api/v1/models/{model_name}/{version}")
    async def api_get_model_version(model_name: str, version: str):
        if version == "versions":
            versions = db.get_model_versions(model_name)
            return {"versions": [{"version": v} for v in versions]}
        card = db.get_model_card(model_name, version)
        if not card:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}@{version}")
        return card

    @app.post("/api/v1/models/upload", status_code=201)
    async def api_upload_model(
        card: str = Form(...),
        user: User = Depends(require_auth),
    ):
        import json
        card_data = json.loads(card)

        # Validate the model card
        try:
            model_card = ModelCard(**card_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid model card: {e}")

        # Check permissions for private models
        if model_card.visibility == ModelVisibility.PRIVATE:
            if not user.org_id or (model_card.organization and model_card.organization != user.org_id):
                raise HTTPException(status_code=403, detail="Cannot create private model without organization access")

        # Save metadata
        db.save_model_card(model_card.full_name, model_card.version, card_data)
        return {"status": "uploaded", "name": model_card.full_name, "version": model_card.version}

    @app.post("/api/v1/models/{model_name}/{version}/files/{filename}")
    async def api_upload_file(
        model_name: str,
        version: str,
        filename: str,
        file: UploadFile = File(...),
        user: User = Depends(require_auth),
    ):
        # Verify model exists and user has access
        card = db.get_model_card(model_name, version)
        if not card:
            raise HTTPException(status_code=404, detail="Model not found")

        # Check permissions
        if card.get("visibility") == ModelVisibility.PRIVATE:
            if not user.org_id or (card.get("organization") and card.get("organization") != user.org_id):
                raise HTTPException(status_code=403, detail="Access denied")

        # Save to storage
        with tempfile.NamedTemporaryFile(delete=False, suffix=filename) as tmp:
            shutil.copyfileobj(file.file, tmp.file)
            tmp_path = tmp.name

        try:
            dest_key = f"{model_name}/{version}/{filename}"
            stored = await storage.upload_file(tmp_path, dest_key)
        finally:
            os.unlink(tmp_path)

        # Record the file on the model card
        files = card.get("files") or []
        new_file = {
            "filename": filename,
            "size": stored.size,
            "sha256": stored.sha256,
            "is_lfs": stored.is_lfs,
            "url": stored.url,
        }
        files = [f for f in files if f.get("filename") != filename]
        files.append(new_file)
        card["files"] = files
        db.save_model_card(card.get("name", model_name), version, card)

        return {"url": stored.url, "sha256": stored.sha256, "size": stored.size}

    @app.delete("/api/v1/models/{model_name}")
    async def api_delete_model(model_name: str, version: Optional[str] = Query(default=None), user: User = Depends(require_auth)):
        card = db.get_model_card(model_name, version or "latest")
        if not card:
            raise HTTPException(status_code=404, detail="Model not found")

        if not user.is_admin and card.get("organization") and card.get("organization") != user.org_id:
            raise HTTPException(status_code=403, detail="Access denied")

        return {"deleted": db.delete_model(model_name, version)}

    # ---- Search ----

    @app.get("/api/v1/search")
    async def api_search(
        q: str = Query(...),
        task: Optional[str] = Query(default=None),
        tags: Optional[str] = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
    ):
        tag_list = tags.split(",") if tags else None
        all_models = db._scan_models()
        results = []
        for m in all_models:
            if q.lower() not in m.get("name", "").lower() and q.lower() not in m.get("description", "").lower():
                continue
            if task and m.get("task") != task:
                continue
            if tag_list and not all(t in (m.get("tags") or []) for t in tag_list):
                continue
            results.append(m)
            if len(results) >= limit:
                break
        return {"results": results, "query": q}

    # ---- Auth ----

    @app.post("/api/v1/auth/login")
    async def api_login(username: str = Form(...), password: str = Form(...)):
        from ..utils import verify_password

        user = db.get_user_by_username(username)
        if not user or not verify_password(password, user.hashed_api_key or ""):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Issue an API key for the user so the returned token is usable
        from ..api.models import APIKey as _APIKeyModel
        raw = None
        for k in db._keys:
            key_obj = _APIKeyModel(**k)
            if key_obj.user_id == user.id:
                raw = k.get("raw_key") or (key_obj.prefix + key_obj.key_id)
                break
        if not raw:
            _, raw = db.create_api_key(user.id, user.org_id, ["read", "write", "models"])

        return {"access_token": raw, "token_type": "bearer", "username": user.username}

    @app.get("/api/v1/auth/verify")
    async def api_verify(current_user: Optional[User] = Depends(get_current_user)):
        if not current_user:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return {"username": current_user.username, "is_admin": current_user.is_admin}

    @app.post("/api/v1/keys")
    async def api_create_key(
        username: str = Form(...),
        scopes: Optional[str] = Form(default="read,write,models"),
        admin_user: User = Depends(require_auth),
    ):
        if not admin_user.is_admin:
            raise HTTPException(status_code=403, detail="Admin access required")

        user = db.get_user_by_username(username)
        if not user:
            user = db.create_user(username=username)

        scope_list = scopes.split(",") if scopes else ["read", "write", "models"]
        key, raw = db.create_api_key(user.id, user.org_id, scope_list)
        return {"api_key": raw, "key_id": key.key_id, "scopes": key.scopes}

    @app.get("/api/v1/keys/me")
    async def api_list_keys(current_user: User = Depends(require_auth)):
        user_keys = [k for k in db._keys if k["user_id"] == current_user.id]
        return {"keys": [{"key_id": k["key_id"], "created_at": k["created_at"], "last_used": k.get("last_used")} for k in user_keys]}

    # ---- Leaderboard ----

    @app.get("/api/v1/leaderboard")
    async def api_get_leaderboard(
        task: Optional[str] = Query(default=None),
        metric: Optional[str] = Query(default=None),
    ):
        entries = db.get_leaderboard(task, metric)
        return {"entries": [e.model_dump(mode="json") for e in entries]}

    @app.post("/api/v1/leaderboard/submit")
    async def api_submit_benchmark(
        result: BenchmarkResult,
        current_user: User = Depends(require_auth),
    ):
        result.evaluated_by = current_user.username
        db.add_leaderboard_entry(result)
        return {"status": "submitted"}

    @app.get("/api/v1/leaderboard/{task}/{metric}")
    async def api_task_leaderboard(task: str, metric: str):
        entries = db.get_leaderboard(task=task, metric=metric)
        return {"entries": [e.model_dump(mode="json") for e in entries]}

    # ---- Organizations ----

    @app.get("/api/v1/orgs")
    async def api_list_orgs():
        return {"orgs": db._orgs}

    @app.post("/api/v1/orgs")
    async def api_create_org(
        name: str = Form(...),
        description: str = Form(default=""),
        is_public: bool = Form(default=True),
        admin_user: User = Depends(require_auth),
    ):
        if not admin_user.is_admin:
            raise HTTPException(status_code=403, detail="Admin access required")

        org = Org(name=name, description=description, is_public=is_public)
        org_dict = org.model_dump()
        db._orgs.append(org_dict)
        # Save
        import json
        with open(db._orgs_file, "w") as f:
            json.dump(db._orgs, f, indent=2)
        return {"org": org_dict}

    # ---- Stats ----

    @app.get("/api/v1/stats")
    async def api_stats():
        models = db._scan_models()
        total_size = sum(m.get("_size", 0) for m in models)
        return {
            "total_models": len(models),
            "total_versions": len(set((m.get("name"), m.get("version")) for m in models)),
            "total_size_human": format_file_size(total_size),
            "total_size_bytes": total_size,
            "total_users": len(db._users),
            "total_orgs": len(db._orgs),
            "leaderboard_entries": len(db._leaderboard),
        }

    return app


def create_app_factory(
    db_path: str = os.path.join(os.path.expanduser("~"), ".sneppx", "hub", "data"),
    storage_config: Optional[StorageConfig] = None,
):
    """App factory for UVicorn."""
    return create_app(db_path, storage_config)
