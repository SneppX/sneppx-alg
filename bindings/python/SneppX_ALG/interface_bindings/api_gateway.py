"""Public API gateway — user registration, login, self-service key management."""

import time
import logging
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException, Query, Header
from pydantic import BaseModel, Field, EmailStr

from .user_store import UserStore, get_user_store
from .api_key_manager import get_key_manager
from .api_key_store import ApiKeyStore
from .usage_tracker import get_usage_tracker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/auth", tags=["auth"])


# ===========================================================================
#  Pydantic Models
# ===========================================================================


class RegisterRequest(BaseModel):
    email: str = Field(..., description="Email address")
    password: str = Field(..., min_length=8, max_length=128, description="Password (min 8 chars)")


class RegisterResponse(BaseModel):
    user_id: str
    email: str
    tier: str
    api_key: str
    message: str = "Account created. Save your API key — it will not be shown again."


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    user_id: str
    email: str
    tier: str
    token: str


class KeyInfo(BaseModel):
    id: str
    key_prefix: str
    name: str
    tier: str
    is_active: bool
    created_at: float
    expires_at: Optional[float] = None


class KeyListResponse(BaseModel):
    keys: List[KeyInfo]


class KeyCreateRequest(BaseModel):
    name: str = Field(default="", max_length=64)
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=365)


class UsageResponse(BaseModel):
    requests: int
    tokens_in: int
    tokens_out: int
    avg_response_ms: float


# ===========================================================================
#  Auth helpers
# ===========================================================================


def _get_user_id_from_key(authorization: Optional[str]) -> Optional[str]:
    """Extract user_id from an API key using the key manager."""
    if not authorization:
        return None
    key = authorization.strip()
    scheme, separator, credential = key.partition(" ")
    if separator and scheme.casefold() == "bearer":
        key = credential.strip()
    km = get_key_manager()
    info = km.validate_key(key)
    if info and info.get("user_id"):
        return info["user_id"]
    session = get_user_store().validate_session_token(key)
    if session:
        return session["user_id"]
    return None


# ===========================================================================
#  Endpoints
# ===========================================================================


@router.post("/register", response_model=RegisterResponse)
async def register(req: RegisterRequest):
    us = get_user_store()
    user = us.create_user(email=req.email, password=req.password, tier="free")
    if user is None:
        raise HTTPException(409, "Email already registered")
    km = get_key_manager()
    key_result = km.create_key(
        name=f"Default key for {req.email}",
        tier="free",
        expires_in_days=None,
        user_id=user["id"],
    )
    return RegisterResponse(
        user_id=user["id"],
        email=user["email"],
        tier=user["tier"],
        api_key=key_result["key"],
    )


@router.post("/logout")
async def logout(authorization: Optional[str] = Header(default=None, alias="Authorization")):
    if not authorization:
        raise HTTPException(401, "Authentication required")
    token = authorization.strip()
    scheme, separator, credential = token.partition(" ")
    if separator and scheme.casefold() == "bearer":
        token = credential.strip()
    if not get_user_store().revoke_session_token(token):
        raise HTTPException(401, "Invalid or expired session")
    return {"status": "logged_out"}


@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    us = get_user_store()
    user = us.authenticate_user(email=req.email, password=req.password)
    if user is None:
        raise HTTPException(401, "Invalid email or password")
    token = us.create_session_token(user["id"], expires_in_seconds=86400)
    return LoginResponse(
        user_id=user["id"],
        email=user["email"],
        tier=user["tier"],
        token=token,
    )


# ===========================================================================
#  Self-service key management (mounted under /v1)
# ===========================================================================

self_service_router = APIRouter(tags=["keys"])


@self_service_router.get("/v1/keys", response_model=KeyListResponse)
async def list_my_keys(authorization: Optional[str] = Header(default=None, alias="Authorization")):
    user_id = _get_user_id_from_key(authorization)
    if not user_id:
        raise HTTPException(401, "Authentication required")
    km = get_key_manager()
    all_keys = km.list_keys()
    my_keys = [k for k in all_keys if k.get("user_id") == user_id]
    return KeyListResponse(keys=[KeyInfo(**k) for k in my_keys])


@self_service_router.post("/v1/keys", response_model=Dict[str, Any])
async def create_my_key(
    req: KeyCreateRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    user_id = _get_user_id_from_key(authorization)
    if not user_id:
        raise HTTPException(401, "Authentication required")
    km = get_key_manager()
    keys = km.list_keys()
    my_key_count = sum(1 for k in keys if k.get("user_id") == user_id and k.get("is_active"))
    tier = "free"
    for k in keys:
        if k.get("user_id") == user_id and k.get("is_active"):
            tier = k.get("tier", "free")
            break
    tier_cfg = km.get_tier_config(tier)
    max_keys = tier_cfg.get("max_keys", 2)
    if my_key_count >= max_keys:
        raise HTTPException(429, f"Key limit reached for tier '{tier}' (max {max_keys})")
    result = km.create_key(
        name=req.name or f"Key {my_key_count + 1}",
        tier=tier,
        expires_in_days=req.expires_in_days,
        user_id=user_id,
    )
    return result


@self_service_router.delete("/v1/keys/{key_id}")
async def revoke_my_key(
    key_id: str,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    user_id = _get_user_id_from_key(authorization)
    if not user_id:
        raise HTTPException(401, "Authentication required")
    km = get_key_manager()
    key = km.get_key(key_id)
    if not key or key.get("user_id") != user_id:
        raise HTTPException(404, "Key not found")
    km.revoke_key(key_id)
    return {"status": "revoked", "key_id": key_id}


@self_service_router.get("/v1/usage")
async def my_usage(
    since: Optional[float] = Query(default=None),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    user_id = _get_user_id_from_key(authorization)
    if not user_id:
        raise HTTPException(401, "Authentication required")
    km = get_key_manager()
    keys = km.list_keys()
    my_keys = [k for k in keys if k.get("user_id") == user_id]
    tracker = get_usage_tracker()
    total = {"requests": 0, "tokens_in": 0, "tokens_out": 0, "avg_response_ms": 0.0}
    for k in my_keys:
        stats = tracker.get_stats(k.get("key_hash", ""), since=since)
        if stats:
            total["requests"] += stats["requests"]
            total["tokens_in"] += stats["tokens_in"]
            total["tokens_out"] += stats["tokens_out"]
    if total["requests"] > 0:
        total["avg_response_ms"] = round(total.get("avg_response_ms", 0), 1)
    return total
