---
description: "Keyboard-first navigation, search, reading, playground controls and accessibility shortcuts for the SneppX-ALG documentation."
---

# Keyboard-first guide

Use **Ctrl/Cmd + K** to search all documentation pages and available commands. Press **?** to open the searchable shortcut reference.

## Global controls

- **?** — open shortcut help.
- **Ctrl/Cmd + K** — open the command palette.
- **/** — focus documentation search.
- **Esc** — close the active overlay, palette or search.
- **Ctrl/Cmd + Shift + L** — copy the current page URL.
- **Ctrl/Cmd + Shift + F** — focus documentation search.

Single-key navigation shortcuts pause while an input, textarea, select or content-editable element has focus. Ctrl/Cmd + K remains available to open the command palette.

## Quick navigation sequences

Press **g**, then the destination key within about one second.

- **g, h** — documentation home.
- **g, g** — top of the current page.
- **g, p** — Python playground.
- **g, a** — Python API.
- **g, c** — cookbook.
- **g, t** — tutorials.
- **g, s** — security overview.
- **g, b** — engineering blog.
- **g, r** — project roadmap.
- **g, m** — matrix multiplication visualizer.
- **g, v** — convolution visualizer.
- **g, o** — optimizer comparison.
- **g, l** — loss gallery.
- **g, i** — installation guide.
- **g, q** — source quick start.
- **g, d** — distributed support guide.
- **g, x** — repository browser.
- **g, k** — keyboard shortcut guide.
- **g, f** — FAQ.
- **g, n** — learning progress.
- **g, w** — code walkthrough.
- **g, u** — autograd guide.
- **g, e** — API explorer.

## Reading and page navigation

- **j / k** — scroll down / up.
- **d / u** — scroll down / up by about half a screen.
- **n / Shift + N** — jump to the next / previous heading.
- **[ / ]** — open the previous / next navigation entry.
- **Home / End** — go to the top / bottom of the page.
- **Alt + Shift + Left / Right** — open the previous / next navigation entry.
- **Alt + Shift + Up / Down** — jump to the previous / next heading.
- **Alt + Shift + 0 / 9** — go to the top / bottom of the page.

## Site controls

- **Alt + Shift + H** — documentation home.
- **Alt + Shift + P** — Python playground.
- **Alt + Shift + A** — Python API.
- **Alt + Shift + S** — focus documentation search.
- **Alt + Shift + K** — open keyboard help.
- **Alt + Shift + T** — toggle the table of contents.
- **Alt + Shift + N** — toggle primary navigation.

## Command palette

Press **Ctrl/Cmd + K**, type a page title or action, move with **Up / Down**, and press **Enter**. Press **Esc** to close.

The palette searches every page in the MkDocs navigation. Its quick actions include home, playground, Python API, API explorer, FAQ, installation, getting started, cookbook, tutorials, security overview, roadmap, blog, repository browser, autograd guide, keyboard help, search, theme, navigation and table-of-contents controls, page scrolling, heading navigation, printing, copying the current URL, the GitHub repository, issue tracker, discussions and changelog RSS feed.

## Playground editor

- **Ctrl/Cmd + Enter** — run the current Python snippet.
- **Tab** — indent the current line or selected lines with four spaces.
- **Shift + Tab** — outdent the current line or selected lines by one indentation level.
- **Ctrl/Cmd + S** — save the snippet in this browser.
- **Ctrl/Cmd + Shift + S** — copy a shareable link containing the code.
- **Ctrl/Cmd + Shift + Backspace** — clear the output panel.
- **Ctrl/Cmd + Shift + R** — restore the starter example.

Saved snippets stay in local browser storage. Share links encode code in the URL fragment; do not put secrets or private data in a shared snippet.

## Accessibility and boundaries

Outside the editor, **Tab** and **Shift + Tab** move through interactive controls.

- **Enter** or **Space** activates a focused link or button.
- Browser and operating-system shortcuts may take precedence. The site does not intentionally remap zoom, tab switching, browser history or screen-reader commands.
- Reduced-motion preferences disable smooth scrolling.
- The Python playground runs in browser WebAssembly. It does not provide CUDA, native extension modules or server-side execution.

See the repository-level [shortcutsforsite.md](https://github.com/SneppX/sneppx-alg/blob/main/shortcutsforsite.md) for the complete inventory.

## Python code blocks

Eligible Python code blocks (up to 4,000 characters) have a **Run** button. Activate it to execute that snippet in the shared browser-based Pyodide runtime. Output appears below the code block; the playground and code-block runner share the same loaded Python kernel.
