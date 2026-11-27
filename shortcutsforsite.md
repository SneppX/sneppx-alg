# SneppX-ALG Website Keyboard Shortcuts

This file is the repository-level reference for the keyboard-first documentation site. Press **?** for the live searchable overlay. Press **Ctrl+K** on Windows/Linux or **⌘+K** on macOS for the command palette.

## Global controls

- **?** — open searchable shortcut help.
- **Ctrl/Cmd + K** — open or close the command palette.
- **/** — focus documentation search.
- **Esc** — close the active shortcut overlay, palette or search.
- **Ctrl/Cmd + Shift + L** — copy the current page URL.
- **Ctrl/Cmd + Shift + F** — focus documentation search.
- **Up / Down** — move the selected command-palette result.
- **Enter** — open the selected page or run the selected command.

## Quick navigation sequences

Press **g**, then the destination key within approximately 1.1 seconds.

- **g, h** — documentation home.
- **g, g** — current page top.
- **g, p** — Python playground.
- **g, a** — Python API.
- **g, c** — cookbook.
- **g, t** — tutorials.
- **g, s** — security overview.
- **g, b** — engineering blog.
- **g, r** — roadmap.
- **g, m** — matrix multiplication visualizer.
- **g, v** — convolution visualizer.
- **g, o** — optimizer comparison.
- **g, l** — loss gallery.
- **g, i** — installation.
- **g, q** — quick start.
- **g, d** — distributed guide.
- **g, x** — repository browser.
- **g, k** — keyboard guide.
- **g, f** — FAQ.
- **g, n** — learning progress.
- **g, w** — code walkthrough.
- **g, u** — autograd guide.
- **g, e** — API explorer.

## Reading and page movement

- **j / k** — scroll down / up.
- **d / u** — scroll about half a viewport.
- **n / Shift + N** — next / previous heading.
- **[ / ]** — previous / next primary-navigation entry.
- **Home / End** — top / bottom of page.
- **Alt + Shift + Left / Right** — previous / next navigation page.
- **Alt + Shift + Up / Down** — previous / next heading.
- **Alt + Shift + 0 / 9** — top / bottom of page.

## Site navigation and layout

- **Alt + Shift + H** — documentation home.
- **Alt + Shift + P** — Python playground.
- **Alt + Shift + A** — Python API.
- **Alt + Shift + S** — focus search.
- **Alt + Shift + K** — open keyboard help.
- **Alt + Shift + T** — toggle table of contents.
- **Alt + Shift + N** — toggle primary navigation.

## Playground editor

- **Ctrl/Cmd + Enter** — execute the current Python snippet.
- **Tab** — insert four spaces at the cursor.
- **Shift + Tab** — remove one indentation level from selected lines.
- **Ctrl/Cmd + S** — save code in local browser storage.
- **Ctrl/Cmd + Shift + S** — copy a share link containing the code.
- **Ctrl/Cmd + Shift + Backspace** — clear the output panel.
- **Ctrl/Cmd + Shift + R** — restore the starter snippet.

## Command palette actions

Search every MkDocs navigation page and these commands:

- Open home, playground, Python API, API explorer, FAQ, installation, getting started, cookbook, tutorials, security overview, roadmap, blog, repository browser or autograd guide.
- Focus search, open keyboard help, toggle light/dark theme, table of contents or primary navigation.
- Scroll to top or bottom; jump to the next or previous heading.
- Print the current page or copy its URL.
- Open the GitHub repository, issue tracker, discussions or changelog RSS feed.
- Search and open every page listed in the documentation navigation.

## Accessibility and limits

Outside the editor, **Tab** moves forward; **Shift + Tab** moves backwards through interactive controls.
- **Enter** or **Space** activates the focused link or button.
- Shortcuts do not intercept typing in inputs, textareas, selects or content-editable regions.
- Browser-reserved shortcuts, zoom, tab switching, browser history and screen-reader commands are not overridden.
- Reduced-motion preferences are respected.
- The browser playground runs in WebAssembly locally. Share links encode source in the URL and must not contain credentials or private information.

Published guide: [Keyboard-first guide](https://sneppx.github.io/sneppx-alg/latest/keyboard-shortcuts/).

## Python code blocks

Every Python code block on the documentation site has a **Run** button. Activate it to execute that snippet in the shared browser-based Pyodide runtime. Output appears below the code block; the playground and code-block runner share the same loaded Python kernel.
