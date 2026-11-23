"""RFC 9116 security.txt integrity checks."""

import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_ROOT_TXT = os.path.join(_ROOT, "security.txt")
_WELL_KNOWN_TXT = os.path.join(_ROOT, ".well-known", "security.txt")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_security_txt_exists_at_both_locations():
    assert os.path.isfile(_ROOT_TXT)
    assert os.path.isfile(_WELL_KNOWN_TXT)
    assert _read(_ROOT_TXT) == _read(_WELL_KNOWN_TXT)


def test_plain_field_format_without_pem_markers():
    text = _read(_ROOT_TXT)
    assert "BEGIN SECURITY CONTACT" not in text
    assert "-----" not in text
    for line in text.splitlines():
        if line and not line.startswith("#"):
            assert re.match(r"^[A-Za-z-]+: .+$", line), line


def test_required_fields_present():
    fields = {}
    for line in _read(_ROOT_TXT).splitlines():
        if ":" in line and not line.startswith("#"):
            name, value = line.split(":", 1)
            fields.setdefault(name, []).append(value.strip())
    assert any(v.startswith("mailto:") for v in fields["Contact"])
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", fields["Expires"][0])
    assert fields["Preferred-Languages"] == ["en"]
    assert fields["Canonical"][0].endswith("/.well-known/security.txt")
