"""Smoke tests for the import paths used by the curated Python API guide."""

from __future__ import annotations


def test_documented_top_level_api_imports() -> None:
    from SneppX_ALG import (
        AdamW,
        Linear,
        Model,
        Optimizer,
        Sequential,
        Tensor,
        Trainer,
    )

    assert all(
        value is not None
        for value in (AdamW, Linear, Model, Optimizer, Sequential, Tensor, Trainer)
    )


def test_documented_algorithm_and_utility_imports() -> None:
    from SneppX_ALG import agents, observability, rag, speculative, structured_output
    from SneppX_ALG.interface_bindings import HSSModel, SERModel
    from SneppX_ALG.interface_bindings.checkpoint import (
        CheckpointReader,
        CheckpointWriter,
    )
    from SneppX_ALG.interface_bindings.profiler import Profiler, timeit
    from SneppX_ALG.interface_bindings.quantization import QuantMode, QuantizedLinear

    assert all(
        value is not None
        for value in (
            agents,
            observability,
            rag,
            speculative,
            structured_output,
            HSSModel,
            SERModel,
            CheckpointReader,
            CheckpointWriter,
            Profiler,
            timeit,
            QuantMode,
            QuantizedLinear,
        )
    )


def test_source_install_instructions_use_the_repository_build_root() -> None:
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    index = (root / "docs" / "index.md").read_text(encoding="utf-8")
    api = (root / "docs" / "api" / "python.md").read_text(encoding="utf-8")

    where = (
        pyproject.get("tool", {})
        .get("setuptools", {})
        .get("packages", {})
        .get("find", {})
        .get("where", [])
    )
    assert "bindings/python" in where
    assert "python -m pip install ." in index
    assert "python -m pip install ." in api
    assert "python -m pip install ./bindings/python" not in index
