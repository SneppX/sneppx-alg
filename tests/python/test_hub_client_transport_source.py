"""Regression checks for credential-sensitive WinHTTP header parsing."""

from pathlib import Path


def test_winhttp_header_trimming_uses_real_tab_character_literal():
    source = Path("model_hub/src/hub_client.c").read_text(encoding="utf-8")
    assert "*name == '\\\\t'" not in source
    assert "name[name_len - 1] == '\\\\t'" not in source
    assert "*name == '\\t'" in source
    assert "name[name_len - 1] == '\\t'" in source
