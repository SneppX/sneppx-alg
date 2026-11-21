"""Tests for core data models."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model_hub.api.models import (
    ModelCard, ModelFormat, ModelTask, SemVer, User, APIKey,
    BenchmarkResult, ModelVisibility, License,
)
from model_hub.utils import hash_password, verify_password


class TestSemVer:
    def test_basic_parse(self):
        sv = SemVer("v1.0.0")
        assert sv.major == 1
        assert sv.minor == 0
        assert sv.patch == 0
        assert str(sv) == "1.0.0"

    def test_comparison(self):
        assert SemVer("1.0.0") < SemVer("1.1.0")
        assert SemVer("1.0.0") < SemVer("2.0.0")
        assert SemVer("1.0.0") == SemVer("1.0.0")
        assert SemVer("1.0.0") > SemVer("0.9.9")

    def test_prerelease(self):
        sv = SemVer("v2.1.0-beta")
        assert sv.prerelease == "beta"
        assert sv.is_prerelease

    def test_invalid(self):
        try:
            SemVer("not-a-version")
            assert False, "Should have raised"
        except ValueError:
            pass


class TestModelCard:
    def _make(self, **kw):
        d = dict(name="test-org/test-model", version="v1.0.0",
                 architecture="Transformer", format=ModelFormat.SNEPPX_NATIVE,
                 task=ModelTask.TEXT_GENERATION, description="Test model")
        d.update(kw)
        return ModelCard(**d)

    def test_create_basic(self):
        card = self._make()
        assert card.full_name == "test-org/test-model"
        assert card.semver.major == 1

    def test_full_name(self):
        card = self._make(name="llama-7b", organization="sneppx")
        assert card.full_name == "sneppx/llama-7b"

    def test_validation_version(self):
        card = self._make()
        assert card.version == "v1.0.0"

    def test_validation_invalid_name(self):
        try:
            self._make(name="test@model")
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_public_dict(self):
        card = self._make(tags=["test", "demo"])
        d = card.to_public_dict()
        assert d["name"] == "test-org/test-model"
        assert "test" in d["tags"]


class TestUser:
    def test_create(self):
        u = User(username="testuser", email="test@example.com")
        assert u.username == "testuser"

    def test_invalid_username(self):
        try:
            User(username="test@user")
            assert False, "Should have raised"
        except ValueError:
            pass


class TestAPIKey:
    def test_generate_and_verify(self):
        key, raw = APIKey.generate(user_id="user123")
        assert key.user_id == "user123"
        assert key.prefix in raw
        assert APIKey.verify(raw, key)
        assert not APIKey.verify("wrong-key", key)

    def test_scopes(self):
        key, _ = APIKey.generate(user_id="user123", scopes=["read", "models"])
        assert "read" in key.scopes
        assert "write" not in key.scopes


class TestPasswordHashing:
    def test_hash_and_verify(self):
        hashed = hash_password("mypassword")
        assert verify_password("mypassword", hashed)
        assert not verify_password("wrongpassword", hashed)

    def test_unique_hashes(self):
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2  # different salts
        assert verify_password("same", h1)
        assert verify_password("same", h2)
