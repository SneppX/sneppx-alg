/* SNEPPX-Alg — Ctrl/Cmd+K command palette.
 * Fuzzy-ish search over nav pages + quick actions. No dependencies.
 */
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

    var actions = [
        { title: "Toggle dark / light theme", kind: "action", run: function () {
            var toggle = document.querySelector('[data-md-color-scheme]') ||
                document.querySelector(".md-header__button--theme");
            if (toggle) toggle.click();
        } },
        { title: "Open playground", kind: "action", run: function () {
            window.location.href = "playground/";
        } },
        { title: "Go to home", kind: "action", run: function () {
            window.location.href = (document.querySelector('[data-md-component="logo"]') || {}).getAttribute("href") || "/";
        } },
        { title: "Focus search", kind: "action", run: function () {
            var t = document.querySelector('[data-md-toggle="search"]');
            if (t && !t.checked) { t.checked = true; t.dispatchEvent(new Event("change")); }
            var i = document.querySelector(".md-search__input");
            if (i) { i.focus(); i.select && i.select(); }
        } },
        { title: "Scroll to top", kind: "action", run: function () { window.scrollTo({ top: 0, behavior: "smooth" }); } },
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
        parts.forEach(function (p) { if (p && t.indexOf(p) >= 0) hit++; });
        return hit ? 30 + hit * 5 : 0;
    }

    var overlay, input, list, pages = [];
    var selected = 0;
    var current = [];

    function build() {
        if (overlay) return;
        pages = collectPages();
        overlay = document.createElement("div");
        overlay.className = "sx-palette";
        overlay.innerHTML =
            '<div class="sx-palette__box">' +
            '  <input class="sx-palette__input" type="text" placeholder="Search pages, actions…" autocomplete="off" />' +
            '  <ul class="sx-palette__list"></ul>' +
            '  <div class="sx-palette__hint">↑↓ navigate · Enter open · Esc close</div>' +
            "</div>";
        document.body.appendChild(overlay);
        input = overlay.querySelector(".sx-palette__input");
        list = overlay.querySelector(".sx-palette__list");
        input.addEventListener("input", render);
        input.addEventListener("keydown", onKey);
        overlay.addEventListener("click", function (e) {
            if (e.target === overlay) close();
        });
        render();
    }

    function open() {
        build();
        overlay.classList.add("sx-palette--open");
        input.value = "";
        selected = 0;
        render();
        setTimeout(function () { input.focus(); }, 10);
    }
    function close() {
        if (overlay) overlay.classList.remove("sx-palette--open");
    }

    function render() {
        var q = (input.value || "").trim();
        var items = [];
        if (!q) {
            // show actions + first pages
            actions.forEach(function (a) { items.push(a); });
            pages.slice(0, 12).forEach(function (p) { items.push(p); });
        } else {
            var all = pages.concat(actions);
            all.forEach(function (it) {
                var s = score(q, it);
                if (s > 0) items.push({ _s: s, item: it });
            });
            items.sort(function (a, b) { return b._s - a._s; });
            items = items.slice(0, 24).map(function (x) { return x.item; });
        }
        current = items;
        if (selected >= current.length) selected = Math.max(0, current.length - 1);
        list.innerHTML = "";
        current.forEach(function (it, i) {
            var li = document.createElement("li");
            li.className = "sx-palette__item" + (i === selected ? " sx-palette__item--active" : "");
            li.innerHTML =
                '<span class="sx-palette__kind">' + it.kind + "</span>" +
                '<span class="sx-palette__title">' + escapeHtml(it.title) + "</span>";
            li.addEventListener("mouseenter", function () { selected = i; updateActive(); });
            li.addEventListener("click", function () { activate(i); });
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
        if (e.key === "ArrowDown") { e.preventDefault(); selected = Math.min(current.length - 1, selected + 1); updateActive(); }
        else if (e.key === "ArrowUp") { e.preventDefault(); selected = Math.max(0, selected - 1); updateActive(); }
        else if (e.key === "Enter") { e.preventDefault(); activate(selected); }
        else if (e.key === "Escape") { e.preventDefault(); close(); }
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