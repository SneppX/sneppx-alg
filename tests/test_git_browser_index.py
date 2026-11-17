"""Tests for the static repository browser search index."""
import importlib.util
from pathlib import Path
MODULE_PATH = Path(__file__).parents[1] / "scripts" / "build_git_browser_index.py"
SPEC = importlib.util.spec_from_file_location("git_browser_index", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

def test_identifier_is_url_safe_and_deterministic():
    value = MODULE.identifier("feature/search-ui")
    assert value == MODULE.identifier("feature/search-ui")
    assert "/" not in value and "+" not in value and "=" not in value

def test_parse_search_lines_preserves_path_line_and_content():
    rows, truncated = MODULE.parse_search_lines("src/model.py:17:def forward(self):\nsrc/model.py:18:    return x + 1\n")
    assert rows == [{"path":"src/model.py","line":17,"text":"def forward(self):"},{"path":"src/model.py","line":18,"text":"    return x + 1"}]
    assert truncated is False

def test_parse_search_lines_enforces_line_limit():
    rows, truncated = MODULE.parse_search_lines("a.py:1:first\na.py:2:second\na.py:3:third\n", max_lines=2)
    assert len(rows) == 2 and truncated is True

def test_parse_search_lines_enforces_byte_limit():
    rows, truncated = MODULE.parse_search_lines("a.py:1:" + "x"*100 + "\n", max_bytes=20)
    assert rows == [] and truncated is True


def test_cache_source_blobs_deduplicates_and_skips_binary(tmp_path, monkeypatch):
    entries = [
        {"type": "blob", "size": 5, "sha": "a" * 40},
        {"type": "blob", "size": 5, "sha": "a" * 40},
        {"type": "blob", "size": 2, "sha": "b" * 40},
        {"type": "blob", "size": 2_000_000, "sha": "c" * 40},
    ]
    def fake_git(*args, binary=False):
        if args[-1] == "a" * 40:
            return b"hello"
        return b"\\xff\\xfe"
    monkeypatch.setattr(MODULE, "git", fake_git)
    stats = MODULE.cache_source_blobs(entries, tmp_path)
    assert stats == {"cached_blob_count": 1, "skipped_binary_count": 1}
    assert (tmp_path / ("a" * 40 + ".txt")).read_text(encoding="utf-8") == "hello"
    assert not (tmp_path / ("b" * 40 + ".txt")).exists()
