# Documentation Site: Build, Validate, and Publish

The public documentation is a Material for MkDocs site configured by
[`mkdocs.yml`](../../mkdocs.yml). Its canonical URL is:

`https://SneppX.github.io/sneppx-alg/`

The source of truth is the `docs/` directory on the repository's `main`
branch. Do not edit generated site output as a substitute for source changes.

## Current automation

This repository contains two GitHub Actions workflows:

- `.github/workflows/ci.yml`: CPU-safe Python tests and an optional native build/test job.
- `.github/workflows/lint-docs.yml`: selected source parsing, README integrity,
  and MkDocs YAML parsing.

There is currently no GitHub Pages deployment workflow in this repository.
A green lint job validates YAML syntax, not the full MkDocs build, external
links, generated Doxygen output, or the live Pages deployment.

## Build prerequisites

Use the documentation extra and install the RSS plugin configured in
`mkdocs.yml`:

```powershell
python -m pip install -e ".[docs]"
python -m pip install "mkdocs-rss-plugin>=1.3"
```

The Material social-card feature may require platform image libraries. If a
build fails in the social plugin, follow the upstream Material for MkDocs
installation guidance for the host operating system.

## Validate locally

From the repository root:

```powershell
python -c "import yaml; yaml.safe_load(open('mkdocs.yml', encoding='utf-8'))"
mkdocs build --strict
```

The YAML parse check is intentionally separate so configuration errors are
reported before the full site build. Strict mode catches navigation and
Markdown reference problems. A full build can still require generated
Doxygen output referenced by `docs/api/index.md`.

Before publishing, verify:

1. The MkDocs build exits successfully with `--strict`.
2. All new pages are included in `nav:` where appropriate.
3. Internal links resolve in the generated site.
4. No private paths, tokens, credentials, or local machine details appear.
5. Security and capability claims match implementation and test evidence.
6. The built site has been reviewed in a browser at desktop and mobile widths.

## Publish to GitHub Pages

Publishing requires repository write access and the correct Pages source
configuration. The GitHub Pages settings are not inferred from this document;
confirm them in **Settings → Pages** before deploying.

If Pages is configured to publish from the `gh-pages` branch at its root,
the maintainer can publish with:

```powershell
mkdocs gh-deploy
```

Do not add `--force` as a routine workaround. Inspect the target branch and
deployment configuration first. Confirm the resulting Pages deployment in
**Actions → Deployments** (or the Pages environment) and open the canonical
URL after GitHub reports deployment success.

## Versioned documentation

The `mike` plugin is configured for optional versioned documentation. Version
publishing is a separate release operation; do not change the default version
until that version has been built and checked.

## External mirrors

A Vercel mirror has been referenced in older project notes, but this repository
does not contain its deployment configuration. Treat mirror ownership,
project settings, and current availability as unverified until checked in the
separate hosting project. GitHub Pages remains the canonical URL configured
by `site_url`.

## Operational policy

- Keep source documentation in `docs/`.
- Keep site configuration in `mkdocs.yml`.
- Do not commit generated `site/` output to `main`.
- Do not claim a deployment succeeded based only on a successful local build.
- Preserve the existing CI workflows; the old statement that all workflow files
  are forbidden is obsolete and contradicted by the repository itself.
