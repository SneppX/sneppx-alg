"""Tests for sneppx_serve.mlflow_like (sneppx-alg interface_bindings.mlflow_like)."""

import os
import tempfile

from SneppX_ALG.interface_bindings.mlflow_like import (
    ModelRegistry,
    JSONStore,
    ModelVersion,
)


def _registry():
    d = tempfile.mkdtemp()
    store = JSONStore(d)
    return ModelRegistry(store=store), d


def test_create_registered_model():
    reg, d = _registry()
    m = reg.create_registered_model("mymodel")
    assert m.name == "mymodel"
    assert reg.get_registered_model("mymodel") is not None
    assert "mymodel" in reg.list_registered_models()


def test_delete_registered_model():
    reg, d = _registry()
    reg.create_registered_model("mymodel")
    assert reg.delete_registered_model("mymodel") is True
    assert reg.delete_registered_model("mymodel") is False
    assert reg.get_registered_model("mymodel") is None


def test_register_model_version():
    reg, d = _registry()
    reg.create_registered_model("mymodel")
    v = reg.register_model(
        "mymodel",
        run_id="runs:/abc/weights",
        source="runs:/abc/weights",
    )
    assert v.version == 1
    assert v.run_id == "runs:/abc/weights"
    assert v.source == "runs:/abc/weights"


def test_get_model_version():
    reg, d = _registry()
    reg.create_registered_model("mymodel")
    reg.register_model("mymodel", run_id="src1")
    v = reg.get_model_version("mymodel", 1)
    assert v is not None
    assert v.version == 1


def test_latest_versions():
    reg, d = _registry()
    reg.create_registered_model("mymodel")
    reg.register_model("mymodel", run_id="src1")
    reg.register_model("mymodel", run_id="src2")
    vs = reg.get_latest_versions("mymodel")
    assert len(vs) >= 1


def test_transition_stage():
    reg, d = _registry()
    reg.create_registered_model("mymodel")
    reg.register_model("mymodel", run_id="src1")
    result = reg.transition_model_version_stage("mymodel", 1, "Production")
    assert result is not None
    v = reg.get_model_version("mymodel", 1)
    assert v.stage == "Production"


def test_alias():
    reg, d = _registry()
    reg.create_registered_model("mymodel")
    reg.register_model("mymodel", run_id="src1")
    reg.set_registered_model_alias("mymodel", "champion", 1)
    v = reg.get_model_version_by_alias("mymodel", "champion")
    assert v is not None
    assert v.version == 1


def test_update_model_version():
    reg, d = _registry()
    reg.create_registered_model("mymodel")
    reg.register_model("mymodel", run_id="src1")
    v = reg.get_model_version("mymodel", 1)
    v.description = "updated"
    reg.update_model_version(v)
    v2 = reg.get_model_version("mymodel", 1)
    assert v2.description == "updated"


def test_model_version_to_dict():
    v = ModelVersion(model_name="m", version=1, source="src")
    d = v.to_dict()
    assert d["model_name"] == "m" or d.get("name") == "m"
    assert d["version"] == 1
