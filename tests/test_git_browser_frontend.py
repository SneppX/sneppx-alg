"""Static integration checks for repository browser frontend."""

from pathlib import Path

ROOT = Path(__file__).parents[1]
BROWSER = (ROOT / "docs/javascripts/git-browser.js").read_text(encoding="utf-8")
PAGE = (ROOT / "docs/code-browser/index.md").read_text(encoding="utf-8")
MKDOCS = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
DEPLOY = (ROOT / ".github/workflows/docs-pages.yml").read_text(encoding="utf-8")
THEME = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")


def test_source_views_share_language_aware_highlighter():
    assert "window.SneppXGitHighlighter.language(path)" in BROWSER
    assert 'highlightedLine(line||" ",lang)' in BROWSER
    assert 'source.replace(/\\r\\n/g,"\\n").split("\\n")' in BROWSER


def test_highlighter_and_browser_are_registered_in_load_order():
    assert MKDOCS.index("javascripts/git-browser-highlighter.js") < MKDOCS.index(
        "javascripts/git-browser.js"
    )
    assert "git-browser-highlighter.js" not in PAGE


def test_root_404_deployment_includes_both_browser_assets():
    assert (
        'cp "$RUNNER_TEMP/git-browser-highlighter.js" javascripts/git-browser-highlighter.js'
        in DEPLOY
    )
    assert DEPLOY.index(
        'latest/javascripts/git-browser-highlighter.js" defer'
    ) < DEPLOY.index('javascripts/git-browser.js" defer')


def test_theme_is_dark_only():
    assert "scheme: slate" in THEME
    assert "prefers-color-scheme: light" not in THEME
    assert "Switch to light mode" not in THEME
    assert "scheme: default" not in THEME


def test_pages_workflow_copies_highlighter_from_checked_out_source():
    assert (
        'cp "$RUNNER_TEMP/git-browser-highlighter.js" javascripts/git-browser-highlighter.js'
        in DEPLOY
    )


def test_pages_workflow_preserves_highlighter_before_mike_switches_branch():
    assert (
        'cp docs/javascripts/git-browser-highlighter.js "$RUNNER_TEMP/git-browser-highlighter.js"'
        in DEPLOY
    )
    assert (
        'cp "$RUNNER_TEMP/git-browser-highlighter.js" javascripts/git-browser-highlighter.js'
        in DEPLOY
    )


def test_pages_workflow_stages_root_highlighter_asset():
    assert (
        "git add robots.txt sitemap.xml 404.html javascripts/git-browser-highlighter.js"
        in DEPLOY
    )


def test_root_fallback_uses_live_versioned_highlighter_asset():
    assert "/sneppx-alg/latest/javascripts/git-browser-highlighter.js" in DEPLOY
