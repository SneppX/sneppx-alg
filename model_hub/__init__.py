"""
SNEPPX Model Hub — A centralized model registry system.

Provides:
  - sneppx.hub.load(model_name, version, task) API
  - Model cards with metadata (architecture, dataset, training config, benchmarks)
  - Semantic versioning (v1.0.0, v1.1.0, etc.)
  - LFS support for large weight files (>1GB)
  - Fine-tuning scripts downloadable with each model
  - Leaderboard for community model performance comparisons
  - Private model hosting with authentication
  - CLI tool: sneppx-hub

Usage:
    from sneppx import hub
    model = hub.load("sneppx/llama-7b", version="v1.0.0")
"""

from __future__ import annotations

__version__ = "0.1.0"

from .hub import (
    Hub,
    load,
    search,
    list_models,
    get_model,
    get_versions,
    download,
    leaderboard,
    submit_benchmark,
    login,
)

from .api.models import (
    ModelCard,
    ModelFormat,
    ModelTask,
    ModelVersion,
    ModelVisibility,
    License,
    SemVer,
    BenchmarkResult,
    Leaderboard,
    LeaderboardEntry,
    User,
    Org,
    APIKey,
    ModelFile,
    ModelRequirements,
    TrainingConfig,
)

from .converters import (
    detect_format,
    load_sneppx_native,
    save_sneppx_native,
    convert_pt_to_sneppx,
    convert_sneppx_to_pt,
    convert_hf_to_sneppx,
    convert_sneppx_to_hf,
    convert_to_hf,
    convert_from_hf,
    convert_to_sneppx,
    convert_from_torch,
)

__all__ = [
    "Hub",
    "load",
    "search",
    "list_models",
    "get_model",
    "get_versions",
    "download",
    "leaderboard",
    "submit_benchmark",
    "login",
    "ModelCard",
    "ModelFormat",
    "ModelTask",
    "ModelVersion",
    "ModelVisibility",
    "License",
    "SemVer",
    "BenchmarkResult",
    "Leaderboard",
    "LeaderboardEntry",
    "User",
    "Org",
    "APIKey",
    "ModelFile",
    "ModelRequirements",
    "TrainingConfig",
    "detect_format",
    "load_sneppx_native",
    "save_sneppx_native",
    "convert_pt_to_sneppx",
    "convert_sneppx_to_pt",
    "convert_hf_to_sneppx",
    "convert_sneppx_to_hf",
    "convert_to_hf",
    "convert_from_hf",
    "convert_to_sneppx",
    "convert_from_torch",
]
