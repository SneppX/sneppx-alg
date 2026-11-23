"""Container supply-chain pinning guards."""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_DOCKERFILE = os.path.join(_ROOT, "Dockerfile")
_COMPOSE = os.path.join(_ROOT, "docker-compose.yml")
_DEPENDABOT = os.path.join(_ROOT, ".github", "dependabot.yml")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_base_images_are_digest_pinned():
    for line in _read(_DOCKERFILE).splitlines():
        if not line.startswith("FROM "):
            continue
        image = line.split()[1]
        # Stage aliases (e.g. `FROM cuda-runtime AS default`) are local.
        if "/" in image:
            assert "@sha256:" in line, line


def test_uvicorn_installed_at_build_with_exact_version():
    dockerfile = _read(_DOCKERFILE)
    assert "pip3 install --no-cache-dir uvicorn==" in dockerfile
    assert "pip install uvicorn" not in _read(_COMPOSE)


def test_dependabot_tracks_images_and_workflows():
    config = _read(_DEPENDABOT)
    for ecosystem in ("docker", "github-actions", "pip"):
        assert f"package-ecosystem: {ecosystem}" in config
