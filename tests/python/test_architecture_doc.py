"""Guard the architecture document against overstated assurance claims."""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_ARCH = os.path.join(_ROOT, "docs", "ARCHITECTURE.md")


def test_architecture_doc_has_alpha_banner():
    with open(_ARCH, encoding="utf-8") as f:
        text = f.read()
    assert "Alpha software" in text.split("## Documentation Standards")[0]
    assert "independently audited or certified" in text


def test_security_layers_do_not_claim_completion():
    with open(_ARCH, encoding="utf-8") as f:
        text = f.read()
    layers = text.split("## Security Layers", 1)[1]
    assert "Status: Complete" not in layers
    assert "experimental / partial" in layers.lower()
