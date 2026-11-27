/* SNEPPX-Alg — Ctrl/Cmd+K command palette.
 * Fuzzy-ish search over nav pages + quick actions. No dependencies.
 */
// prettier-ignore
(function () {
  "use strict";

  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Collect nav pages once.
  function collectPages() {
    var out = [];
    var seen = {};
    document.querySelectorAll(".md-nav a, .md-sidebar a").forEach(function (a) {
      var href = a.getAttribute("href");
      var title = (a.textContent || "").trim();
      if (!href || !title || href.indexOf("http") === 0) return;
      if (seen[href]) return;
      seen[href] = 1;
      out.push({ title: title, url: href, kind: "page" });
    });
    return out;
  }

  function siteRoot() {
    var logo = document.querySelector('[data-md-component="logo"]');
    return new URL(logo ? logo.getAttribute("href") : "/", window.location.href);
  }

  function navigate(path) {
    window.location.href = new URL(path, siteRoot()).href;
  }

  function toggleControl(id) {
    var label = document.querySelector('label[for="' + id + '"]');
    if (label) label.click();
  }

  function focusSearch() {
    var toggle = document.querySelector('[data-md-toggle="search"]');
    if (toggle && !toggle.checked) {
      toggle.checked = true;
      toggle.dispatchEvent(new Event("change", { bubbles: true }));
    }
    var input = document.querySelector(".md-search__input");
    if (input) {
      input.focus();
      if (input.select) input.select();
    }
  }

  function moveHeading(direction) {
    var headings = Array.prototype.slice.call(
      document.querySelectorAll(".md-content__inner h1, .md-content__inner h2, .md-content__inner h3, .md-content__inner h4"),
    ).filter(function (heading) {
      return heading.getClientRects().length > 0;
    });
    if (!headings.length) return;
    var threshold = window.scrollY + 130;
    var target = null;
    if (direction > 0) {
      target = headings.find(function (heading) {
        return heading.getBoundingClientRect().top + window.scrollY > threshold;
      }) || headings[headings.length - 1];
    } else {
      for (var i = headings.length - 1; i >= 0; i--) {
        if (headings[i].getBoundingClientRect().top + window.scrollY < threshold - 20) {
          target = headings[i];
          break;
        }
      }
      target = target || headings[0];
    }
    target.setAttribute("tabindex", "-1");
    target.focus({ preventScroll: true });
    target.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
  }

  function copyPageUrl() {
    var url = window.location.href;
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(url).catch(function () {
        window.prompt("Copy this page URL:", url);
      });
    } else {
      window.prompt("Copy this page URL:", url);
    }
  }

  var actions = [
    {
      title: "Toggle dark / light theme",
      kind: "action",
      run: function () {
        var current = document.body.getAttribute("data-md-color-scheme");
        var nextScheme = current === "slate" ? "default" : "slate";
        var option = document.querySelector(
          'input[data-md-color-scheme="' + nextScheme + '"]',
        );
        if (option) toggleControl(option.id);
      },
    },
    {
      title: "Open playground",
      kind: "action",
      run: function () {
        navigate("playground/");
      },
    },
    {
      title: "Go to home",
      kind: "action",
      run: function () {
        navigate("");
      },
    },
    {
      title: "Focus search",
      kind: "action",
      run: function () {
        focusSearch();
      },
    },
    {
      title: "Scroll to top",
      kind: "action",
      run: function () {
        window.scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" });
      },
    },
    { title: "Scroll to bottom", kind: "action", run: function () {
      window.scrollTo({ top: document.documentElement.scrollHeight, behavior: reduced ? "auto" : "smooth" });
    } },
    { title: "Next heading", kind: "action", run: function () { moveHeading(1); } },
    { title: "Previous heading", kind: "action", run: function () { moveHeading(-1); } },
    { title: "Toggle table of contents", kind: "action", run: function () { toggleControl("__toc"); } },
    { title: "Toggle primary navigation", kind: "action", run: function () { toggleControl("__drawer"); } },
    { title: "Open keyboard shortcuts", kind: "action", run: function () {
      if (window.SxShortcuts) window.SxShortcuts.open();
    } },
    { title: "Open Python API", kind: "action", run: function () { navigate("api/python/"); } },
    { title: "Open API explorer", kind: "action", run: function () { navigate("api-explorer/"); } },
    { title: "Open FAQ", kind: "action", run: function () { navigate("FAQ/"); } },
    { title: "Open installation guide", kind: "action", run: function () { navigate("installation/"); } },
    { title: "Open getting started", kind: "action", run: function () { navigate("getting-started/"); } },
    { title: "Open cookbook", kind: "action", run: function () { navigate("cookbook/"); } },
    { title: "Open tutorials", kind: "action", run: function () { navigate("tutorials/"); } },
    { title: "Open security overview", kind: "action", run: function () { navigate("security/"); } },
    { title: "Open roadmap", kind: "action", run: function () { navigate("roadmap/tensorflow-pytorch-parity/"); } },
    { title: "Open engineering blog", kind: "action", run: function () { navigate("blog/"); } },
    { title: "Open repository browser", kind: "action", run: function () { navigate("code-browser/"); } },
    { title: "Open autograd guide", kind: "action", run: function () { navigate("autograd-guide/"); } },
    { title: "Copy current page URL", kind: "action", run: copyPageUrl },
    { title: "Print current page", kind: "action", run: function () { window.print(); } },
    { title: "Open GitHub repository", kind: "action", run: function () {
      window.location.href = "https://github.com/SneppX/sneppx-alg";
    } },
    { title: "Open issue tracker", kind: "action", run: function () {
      window.location.href = "https://github.com/SneppX/sneppx-alg/issues";
    } },
    { title: "Open discussions", kind: "action", run: function () {
      window.location.href = "https://github.com/SneppX/sneppx-alg/discussions";
    } },
    { title: "Open changelog RSS feed", kind: "action", run: function () {
      navigate("feed_rss_updated.xml");
    } },
  ];

  function score(q, item) {
    q = q.toLowerCase();
    var t = item.title.toLowerCase();
    if (t === q) return 100;
    if (t.indexOf(q) === 0) return 80;
    if (t.indexOf(q) >= 0) return 60;
    // token match
    var parts = q.split(/\s+/);
    var hit = 0;
    parts.forEach(function (p) {
      if (p && t.indexOf(p) >= 0) hit++;
    });
    return hit ? 30 + hit * 5 : 0;
  }

  var overlay,
    input,
    list,
    pages = [];
  var selected = 0;
  var current = [];
  var returnFocus = null;

  function build() {
    if (overlay) return;
    pages = collectPages();
    overlay = document.createElement("div");
    overlay.className = "sx-palette";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", "Command palette");
    overlay.innerHTML =
      '<div class="sx-palette__box">' +
      '  <input class="sx-palette__input" type="text" placeholder="Search pages, actions…" autocomplete="off" />' +
      '  <ul class="sx-palette__list"></ul>' +
      '  <div class="sx-palette__hint">↑↓ navigate · Enter open · Esc close</div>' +
      "</div>";
    document.body.appendChild(overlay);
    input = overlay.querySelector(".sx-palette__input");
    input.setAttribute("aria-label", "Search documentation pages and actions");
    input.setAttribute("aria-controls", "sx-palette-results");
    list = overlay.querySelector(".sx-palette__list");
    list.id = "sx-palette-results";
    list.setAttribute("role", "listbox");
    list.setAttribute("aria-label", "Command palette results");
    input.addEventListener("input", render);
    input.addEventListener("keydown", onKey);
    overlay.addEventListener("click", function (e) {
      if (e.target === overlay) close();
    });
    overlay.addEventListener("keydown", function (e) {
      if (e.key === "Tab") {
        e.preventDefault();
        input.focus();
      }
    });
    render();
  }

  function open() {
    returnFocus = document.activeElement;
    build();
    overlay.classList.add("sx-palette--open");
    input.value = "";
    selected = 0;
    render();
    setTimeout(function () {
      input.focus();
    }, 10);
  }
  function close() {
    if (overlay) overlay.classList.remove("sx-palette--open");
    if (returnFocus && returnFocus.focus) returnFocus.focus();
  }

  function render() {
    var q = (input.value || "").trim();
    var items = [];
    if (!q) {
      // show actions + first pages
      actions.forEach(function (a) {
        items.push(a);
      });
      pages.slice(0, 12).forEach(function (p) {
        items.push(p);
      });
    } else {
      var all = pages.concat(actions);
      all.forEach(function (it) {
        var s = score(q, it);
        if (s > 0) items.push({ _s: s, item: it });
      });
      items.sort(function (a, b) {
        return b._s - a._s;
      });
      items = items.slice(0, 24).map(function (x) {
        return x.item;
      });
    }
    current = items;
    if (selected >= current.length) selected = Math.max(0, current.length - 1);
    list.innerHTML = "";
    current.forEach(function (it, i) {
      var li = document.createElement("li");
      li.className = "sx-palette__item" + (i === selected ? " sx-palette__item--active" : "");
      li.setAttribute("role", "option");
      li.setAttribute("aria-selected", String(i === selected));
      li.innerHTML =
        '<span class="sx-palette__kind">' +
        it.kind +
        "</span>" +
        '<span class="sx-palette__title">' +
        escapeHtml(it.title) +
        "</span>";
      li.addEventListener("mouseenter", function () {
        selected = i;
        updateActive();
      });
      li.addEventListener("click", function () {
        activate(i);
      });
      list.appendChild(li);
    });
    if (!current.length) {
      var li = document.createElement("li");
      li.className = "sx-palette__empty";
      li.textContent = "No matches.";
      list.appendChild(li);
    }
  }

  function updateActive() {
    var lis = list.querySelectorAll(".sx-palette__item");
    lis.forEach(function (li, i) {
      li.classList.toggle("sx-palette__item--active", i === selected);
      li.setAttribute("aria-selected", String(i === selected));
    });
  }
  function activate(i) {
    var it = current[i];
    if (!it) return;
    close();
    if (it.run) it.run();
    else if (it.url) window.location.href = it.url;
  }
  function onKey(e) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      selected = Math.min(current.length - 1, selected + 1);
      updateActive();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      selected = Math.max(0, selected - 1);
      updateActive();
    } else if (e.key === "Enter") {
      e.preventDefault();
      activate(selected);
    } else if (e.key === "Escape") {
      e.preventDefault();
      close();
    }
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  document.addEventListener("keydown", function (e) {
    if ((e.ctrlKey || e.metaKey) && (e.key === "k" || e.key === "K")) {
      e.preventDefault();
      if (overlay && overlay.classList.contains("sx-palette--open")) close();
      else open();
    }
    if (e.key === "Escape" && overlay && overlay.classList.contains("sx-palette--open")) {
      close();
    }
  });

  // inject a small trigger button into header (optional)
  var btn = document.createElement("button");
  btn.type = "button";
  btn.className = "sx-palette-trigger";
  btn.textContent = "⌘K";
  btn.title = "Open command palette";
  btn.addEventListener("click", open);
  var inner = document.querySelector(".md-header__inner");
  if (inner) inner.appendChild(btn);
})();
