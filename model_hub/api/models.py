"""
SNEPPX Model Hub — Core Data Models

Defines the data models for the model hub: ModelCard, Version, Leaderboard,
User/Org auth, etc. Uses Pydantic v2 schemas for API validation and JSON
serialization.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
except ImportError:
    from pydantic.v1 import BaseModel, Field, validator as field_validator, root_validator as model_validator

from ..storage.backend import StorageBackend


class ModelVisibility(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    ORG_ONLY = "org_only"


class ModelTask(str, Enum):
    TEXT_GENERATION = "text-generation"
    TEXT_CLASSIFICATION = "text-classification"
    IMAGE_CLASSIFICATION = "image-classification"
    OBJECT_DETECTION = "object-detection"
    SEGMENTATION = "segmentation"
    EMBEDDING = "embedding"
    TOKEN_CLS = "token-classification"
    QUESTION_ANSWERING = "question-answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CODE_GENERATION = "code-generation"
    REINFORCEMENT_LEARNING = "reinforcement-learning"
    AUTOMATIC_SPEECH_RECOGNITION = "automatic-speech-recognition"
    AUDIO_CLASSIFICATION = "audio-classification"
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_TO_IMAGE = "image-to-image"
    FEATURE_EXTRACTION = "feature-extraction"


class ModelFormat(str, Enum):
    SNEPPX_NATIVE = "sneppx-native"
    PYTORCH = "pytorch"
    SAFETENSORS = "safetensors"
    ONNX = "onnx"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    HUGGINGFACE = "huggingface"


class License(str, Enum):
    MIT = "mit"
    APACHE_2_0 = "apache-2.0"
    BSD_3_CLAUSE = "bsd-3-clause"
    GPL_3_0 = "gpl-3.0"
    CREATIVE_COMMONS = "creative-commons"
    COMMERCIAL = "commercial"
    ARR = "all-rights-reserved"
    CUSTOM = "custom"


class SemVer:
    """Semantic version parsing and comparison (v1.0.0, v1.1.0, etc.)."""

    _RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.]+))?(?:\+([a-zA-Z0-9.]+))?$")

    def __init__(self, version: str):
        m = self._RE.match(version)
        if not m:
            raise ValueError(f"Invalid semantic version: {version}")
        self.major = int(m.group(1))
        self.minor = int(m.group(2))
        self.patch = int(m.group(3))
        self.prerelease = m.group(4)
        self.build = m.group(5)
        self.raw = version

    def __str__(self) -> str:
        s = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            s += f"-{self.prerelease}"
        if self.build:
            s += f"+{self.build}"
        return s

    def __lt__(self, other: "SemVer") -> bool:
        return self._tuple() < other._tuple()

    def __le__(self, other: "SemVer") -> bool:
        return self._tuple() <= other._tuple()

    def __gt__(self, other: "SemVer") -> bool:
        return self._tuple() > other._tuple()

    def __ge__(self, other: "SemVer") -> bool:
        return self._tuple() >= other._tuple()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemVer):
            return NotImplemented
        return self._tuple() == other._tuple()

    def _tuple(self):
        base = (self.major, self.minor, self.patch)
        if self.prerelease:
            return base + (0, self.prerelease)
        return base + (1, "")

    @property
    def is_prerelease(self) -> bool:
        return self.prerelease is not None


class ModelFile(BaseModel):
    """Represents a single file in a model version."""
    filename: str
    sha256: str
    size: int
    url: str
    content_type: str = "application/octet-stream"

    model_config = {"extra": "allow"}


class ModelRequirements(BaseModel):
    """System requirements for running a model."""
    min_ram: Optional[int] = None  # in bytes
    min_disk: Optional[int] = None  # in bytes
    gpu_memory: Optional[int] = None  # in bytes, 0 = CPU only
    python_version: Optional[str] = None
    cuda_version: Optional[str] = None
    extra_deps: List[str] = Field(default_factory=list)


class TrainingConfig(BaseModel):
    """Training configuration metadata."""
    model_config = {"extra": "allow"}
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    optimizer: Optional[str] = None
    warmup_steps: Optional[int] = None
    lr_scheduler: Optional[str] = None
    mixed_precision: Optional[str] = None
    gradient_accumulation_steps: Optional[int] = None
    max_grad_norm: Optional[float] = None


class BenchmarkResult(BaseModel):
    """A single benchmark result for a model version."""
    model_name: str
    version: str
    task: ModelTask
    metric: str  # e.g., "accuracy", "perplexity", "mAP"
    score: float
    higher_is_better: bool = True
    dataset: str
    eval_config: Dict[str, Any] = Field(default_factory=dict)
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    evaluated_by: Optional[str] = None


class ModelCard(BaseModel):
    """Full model card with metadata."""
    # Core identity
    name: str
    organization: Optional[str] = None
    description: str = ""
    version: str  # semantic version, e.g. "v1.0.0"
    visibility: ModelVisibility = ModelVisibility.PUBLIC

    # Architecture
    architecture: str
    format: ModelFormat = ModelFormat.SNEPPX_NATIVE
    task: ModelTask
    license: License = License.MIT

    # Files
    files: List[ModelFile] = Field(default_factory=list)
    requirements: ModelRequirements = Field(default_factory=ModelRequirements)

    # Metadata
    tags: List[str] = Field(default_factory=list)
    language: Optional[str] = None
    datasets: List[str] = Field(default_factory=list)
    training_config: Optional[TrainingConfig] = None

    # Provenance
    author: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Derived
    model_config = {"extra": "allow"}

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        if not v:
            return "v1.0.0"
        if not v.startswith("v"):
            v = "v" + v
        try:
            SemVer(v)
        except ValueError:
            raise ValueError(f"version must be a semantic version (e.g. 'v1.0.0'), got: {v}")
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_./-]+$", v):
            raise ValueError("model name can only contain alphanumeric, dash, underscore, dot, and slash")
        parts = v.split("/")
        if len(parts) > 2:
            raise ValueError("model name can have at most one '/' (organization/name)")
        return v

    @property
    def full_name(self) -> str:
        if self.organization:
            return f"{self.organization}/{self.name}"
        return self.name

    @property
    def total_size(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def semver(self) -> SemVer:
        return SemVer(self.version)

    def is_latest(self, all_versions: List["ModelCard"]) -> bool:
        return self.version == max(v.version for v in all_versions)

    def to_public_dict(self) -> Dict[str, Any]:
        """Return a dict suitable for public API responses."""
        return {
            "name": self.full_name,
            "description": self.description,
            "version": self.version,
            "architecture": self.architecture,
            "format": self.format,
            "task": self.task,
            "license": self.license,
            "tags": self.tags,
            "language": self.language,
            "datasets": self.datasets,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "total_size": self.total_size,
            "requirements": self.requirements.model_dump(),
        }


class ModelVersion(BaseModel):
    """A single version of a model (lightweight, for listing)."""
    name: str
    version: str
    description: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_size: int = 0
    format: ModelFormat
    task: ModelTask
    is_latest: bool = False
    tags: List[str] = Field(default_factory=list)
    model_config = {"extra": "allow"}


class LeaderboardEntry(BaseModel):
    """An entry in the public leaderboard."""
    model_name: str
    version: str
    task: ModelTask
    metric: str
    score: float
    rank: int = 0
    dataset: str
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_public: bool = True


class Leaderboard(BaseModel):
    """Leaderboard for a specific task + metric combination."""
    task: ModelTask
    metric: str
    dataset: str
    entries: List[LeaderboardEntry] = Field(default_factory=list)
    higher_is_better: bool = True
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Org(BaseModel):
    """Organization model."""
    id: str = Field(default_factory=lambda: secrets.token_hex(8))
    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_public: bool = True
    quota_gb: int = 10

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("org name can only contain alphanumeric, dash, and underscore")
        return v


class User(BaseModel):
    """User model."""
    id: str = Field(default_factory=lambda: secrets.token_hex(8))
    username: str
    email: Optional[str] = None
    hashed_api_key: Optional[str] = None
    api_key_prefix: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_admin: bool = False
    org_id: Optional[str] = None

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("username can only contain alphanumeric, dash, and underscore")
        return v


class APIKey(BaseModel):
    """API key for authentication."""
    key_id: str = Field(default_factory=lambda: secrets.token_hex(8))
    prefix: str
    hashed: str
    user_id: str
    org_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True
    scopes: List[str] = Field(default_factory=lambda: ["read", "write", "models"])

    @staticmethod
    def generate(user_id: str, org_id: Optional[str] = None, scopes: Optional[List[str]] = None) -> "tuple[APIKey, str]":
        """Generate a new API key. Returns (key_obj, raw_key_string)."""
        raw = secrets.token_urlsafe(32)
        prefix = raw[:8]
        hashed = hashlib.sha256(raw.encode()).hexdigest()
        key = APIKey(
            prefix=prefix,
            hashed=hashed,
            user_id=user_id,
            org_id=org_id,
            scopes=scopes or ["read", "write", "models"],
        )
        return key, raw

    @staticmethod
    def verify(api_key: str, stored: "APIKey") -> bool:
        """Verify a raw API key against a stored APIKey."""
        if not api_key.startswith(stored.prefix):
            return False
        hashed = hashlib.sha256(api_key.encode()).hexdigest()
        return secrets.compare_digest(hashed, stored.hashed)
