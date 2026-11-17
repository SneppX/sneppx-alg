---
title: Repository
description: Browse every tracked branch, tag, directory, source file, and commit in SneppX-ALG.
---

<link rel="stylesheet" href="/sneppx-alg/stylesheets/git-browser.css">
<script src="/sneppx-alg/javascripts/git-browser-highlighter.js" defer></script>
<script src="/sneppx-alg/javascripts/git-browser.js" defer></script>

<div id="git-browser-root" data-git-browser></div>

The read-only repository browser is generated from Git references during the documentation build. GitHub remains the source of truth.

## Direct routes

| Route | Contents |
|---|---|
| [Main branch](/sneppx-alg/tree/main/) | Browse the default branch |
| [All branches](/sneppx-alg/branches/) | List every public branch |
| [All tags](/sneppx-alg/tags/) | Browse release and development tags |
| [Main commit history](/sneppx-alg/commits/main/) | Inspect the complete indexed history |
| [Repository search](/sneppx-alg/search/) | Search indexed file and directory paths |

## URL structure

- \`/tree/{branch}/{path}\` lists the contents of a directory.
- \`/blob/{branch}/{path}\` displays a tracked file with line numbers.
- \`/tag/{tag}/tree/{path}\` browses a tagged snapshot.
- \`/tag/{tag}/blob/{path}\` displays a file from a tagged snapshot.
- \`/commit/{sha}\` displays commit metadata and changed files.
- \`/commits/{branch}/\` lists the branch's indexed commit history.

Branch names containing slashes are supported. For example, \`/tree/ci-gated/self-audit-integrity/\`.

## Source and revision integrity

Directory listings, file paths, object IDs, branch heads, tags, and commit histories are generated from Git during deployment. Source files are retrieved using the immutable commit SHA associated with the selected branch or tag. GitHub links open the corresponding commit, file history, blame view, or branch comparison.

The browser is read-only. It does not create a second editable copy of the source tree.
