from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_site_description_is_search_snippet_length():
    config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    description = " ".join(config["site_description"].split())
    assert 80 <= len(description) <= 160


def test_site_uses_canonical_project_pages_url():
    config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    assert config["site_url"] == "https://sneppx.github.io/sneppx-alg/"


def test_properdocs_owns_social_metadata_without_duplicate_tags():
    override = (ROOT / "overrides/main.html").read_text(encoding="utf-8")
    assert not re.search(r'<meta\s+property=["\'](?:og:|twitter:)', override)
    assert '"@type": "SoftwareSourceCode"' in override
    assert '"@type": "WebSite"' in override
    assert '"@type": "TechArticle"' in override
    assert "page.canonical_url | tojson" in override


def test_crawling_metadata_points_to_root_sitemap_and_assistant_index():
    robots = (ROOT / "docs/robots.txt").read_text(encoding="utf-8")
    llms = (ROOT / "docs/llms.txt").read_text(encoding="utf-8")
    assert "https://sneppx.github.io/sneppx-alg/sitemap.xml" in robots
    assert "Python playground" in llms
    assert "Keyboard-first guide" in llms
