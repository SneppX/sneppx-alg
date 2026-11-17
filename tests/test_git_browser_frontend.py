"""Static integration checks for repository browser frontend."""
from pathlib import Path
ROOT = Path(__file__).parents[1]
BROWSER = (ROOT / "docs/javascripts/git-browser.js").read_text(encoding="utf-8")
PAGE = (ROOT / "docs/code-browser/index.md").read_text(encoding="utf-8")
THEME = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
def test_source_views_share_language_aware_highlighter():
    assert "window.SneppXGitHighlighter.language(path)" in BROWSER
    assert "highlightedLine(line||\\" \\",lang)" in BROWSER
    assert 'source.replace(/\\\\r\\\\n/g,"\\\\n").split("\\\\n")' in BROWSER
def test_highlighter_script_loads_before_browser():
    assert PAGE.index("git-browser-highlighter.js") < PAGE.index("git-browser.js")
def test_theme_is_dark_only():
    assert "scheme: slate" in THEME
    assert "prefers-color-scheme: light" not in THEME
    assert "Switch to light mode" not in THEME
    assert "scheme: default" not in THEME