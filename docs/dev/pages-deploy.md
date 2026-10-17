# Documentation Publishing

> **Repo policy: no CI/CD workflow files.** Automated `gh-pages` workflows,
> `.github/workflows/*.yml`, `CODEOWNERS`, and CI configs are **intentionally
> forbidden** in `ammar49-cyber/sneppx-alg`. All verification is local.

The documentation is a static **Material for MkDocs** site. There is a **primary**
origin (GitHub Pages) and a **mirror** on the company Vercel project.

## Primary: `SneppX.github.io/sneppx-alg` (GitHub Pages)

The canonical docs site is published from the `ammar49-cyber/sneppx-alg`
repository's `gh-pages` branch. `mkdocs.yml` -> `site_url` points here:

```yaml
site_url: https://SneppX.github.io/sneppx-alg/
```

## Mirror: `sneppxalg.vercel.app/sneppxalg` (Ariz-Site)

The same built site is **mirrored, unchanged**, into the companion
`ammar49-cyber/Arix-Site` (Next.js, `output: "export"`) repository, under
`public/sneppxalg/`. Vercel serves `public/` verbatim, so the docs become
available at `https://sneppxalg.vercel.app/sneppxalg/`.

Why this is safe and lossless:
- `site_url` stays the **GitHub** canonical, so canonical/Sitemap/RSS tags
  always point at the primary origin (no SEO duplication surprises).
- MkDocs emits **depth-relative** asset references and a `.nojekyll`-style
  static layout, so the **same `site/` directory mounts verbatim** into a
  subfolder of any static host — no HTML rewriting or `base` tag is required.

## Prerequisites

```powershell
# In the hermes venv (or any maintainer venv)
python -m pip install "mkdocs-material>=9.5" mkdocs-git-revision-date-localized-plugin
```

The build uses **Material for MkDocs** + the `git-revision-date-localized`
plugin (for "last updated" timestamps). No other plugins are required.

## Build locally

```powershell
mkdocs build --strict          # -> site/  (open site\index.html)
```

`--strict` fails the build on broken internal links and missing references.
If it fails:

- Check every `nav:` entry in `mkdocs.yml` maps to an existing file.
- The Doxygen frame (`docs/api/index.md`) references generated HTML. Build it
  first (`doxygen Doxyfile` writes `docs/api/doxygen/html/index.html`) so the
  iframe resolves; otherwise links to it are plain HTML hrefs (not
  Markdown-relative) and are not link-checked by `--strict`.

## Deploy to GitHub Pages (primary)

```powershell
# One-time: configure Pages on the ammar49-cyber/sneppx-alg repo
# Settings -> Pages -> Source: "Deploy from a branch" -> gh-pages (root)

# To publish a release:
mkdocs gh-deploy --force
```

- `gh-deploy` writes the built site to the `gh-pages` branch and (with
  `--force`) replaces any prior deployment.
- Because there is **no CI**, every publish is a deliberate maintainer action.
  Tag the repo (`git tag v1.2.0 && git push --tags`) before deploying so the
  changelog and "last updated" metadata reflect the release.
- `docs/.nojekyll` is present, so GitHub Pages serves directories like `assets/`
  and the (optional) `doxygen/html/` without Jekyll filtering.

## Publish the Vercel mirror (`sneppxalg.vercel.app/sneppxalg`)

After a successful `mkdocs build --strict`:

```powershell
# 1. From the sneppx-alg repo root, build the static site
mkdocs build --strict          # -> site/

# 2. Mirror site/ into the Ariz-Site static folder (overwrite /sneppxalg)
robocopy site "..\Arix-Site\public\sneppxalg" /E /NFL /NDL /NJH /NJS /NC /R:1 /W:1

# 3. Commit + push in Arix-Site; Vercel auto-redeploys on main
cd ..\Arix-Site
git add -A
git commit -m "docs: mirror SNEPPX-Algo docs at /sneppxalg"
git push origin main
```

Vercel rebuilds the Arix-Site Next.js (`output: "export"`) app on every push to
`main`, and `public/sneppxalg/` is served verbatim at `/sneppxalg/`. No
`next.config.js` or `vercel.json` changes are required.

> If a Vercel deploy reports **"Deployment was blocked"**, it is a GitHub/Vercel
> deployment-protection gate on the `Arix-Site` project (not a build error). The
> docs are still correct on GitHub Pages (primary); the mirror needs an
> approve-and-redeploy from a Vercel project owner.

## Versioned docs (mike) — optional

`mkdocs.yml` configures `extra.version.provider: mike` for versioned URLs.
Maintain versioned builds manually only if you need per-version docs:

```powershell
pip install mike
mike deploy --push latest            # tip
mike deploy --push 1.1.x            # backport branch
mike set-default --push 1.1.x       # default version
```

This is **optional** — a single latest build (above) is the default flow.

## Intentional design choices

| Decision | Reason |
|----------|--------|
| No `.github/workflows/*.yml` | Repo policy: all verification is local |
| GitHub Pages = primary; Vercel = mirror | Primary origin is canonical; Vercel mirrors unchanged |
| Mount `site/` into `Arix-Site/public/sneppxalg/` verbatim | Depth-relative assets need no rewrite; no Next config risk |
| `site_url` = GitHub canonical | Avoids canonical mismatch across the two origins |
| `docs/.nojekyll` committed | Lets GitHub Pages serve `assets/` and `doxygen/` dirs |
| `mkdocs build --strict` | Catches broken links and nav drift in PRs |
| `git-revision-date-localized` plugin | Lightweight "last updated" metadata; no version branch required |
