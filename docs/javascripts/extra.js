/* SNEPPX-Alg site scripts
 * - scroll reveal via IntersectionObserver
 * - word-by-word heading reveal on load
 * - stat counter count-up on scroll (elements with .sx-stat-value[data-count])
 * Respects prefers-reduced-motion; no external dependencies.
 */
(function () {
    "use strict";

    /* ---- 404 helpers (always run) ----------------------------------------- */
    (function () {
        var url = document.getElementById("sx-404-url");
        if (url) {
            url.textContent = location.pathname + location.search;
        }
        var searchBtn = document.querySelector(".sx-404-search");
        if (searchBtn) {
            searchBtn.addEventListener("click", function () {
                var toggle = document.querySelector(
                    '[data-md-toggle="search"]'
                );
                if (toggle && !toggle.checked) {
                    toggle.checked = true;
                    toggle.dispatchEvent(new Event("change"));
                }
                var input = document.querySelector(".md-search__input");
                if (input) {
                    input.focus();
                    input.select && input.select();
                }
            });
        }
    })();

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        // Reduced motion: show everything immediately, skip animations.
        document.querySelectorAll(".reveal").forEach(function (el) {
            el.classList.add("reveal-visible");
        });
        return;
    }

    /* ---- Scroll reveal ---------------------------------------------------- */
    var revealTargets = document.querySelectorAll(
        ".md-typeset h2, .md-typeset h3, .md-typeset table, " +
            ".md-typeset .admonition, .md-typeset blockquote, " +
            ".md-typeset .sx-card, .md-typeset .sx-stat"
    );
    revealTargets.forEach(function (el) {
        el.classList.add("reveal");
    });

    var revealObserver = new IntersectionObserver(
        function (entries, observer) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    entry.target.classList.add("reveal-visible");
                    observer.unobserve(entry.target);
                }
            });
        },
        { threshold: 0.1 }
    );
    revealTargets.forEach(function (el) {
        revealObserver.observe(el);
    });

    /* ---- Word-by-word heading reveal -------------------------------------- */
    function splitWords(el) {
        // Skip if already processed or contains non-text nodes (code, links ok).
        var walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null);
        var texts = [];
        var node;
        while ((node = walker.nextNode())) {
            if (node.nodeValue.trim()) {
                texts.push(node);
            }
        }
        var index = 0;
        texts.forEach(function (textNode) {
            var frag = document.createDocumentFragment();
            textNode.nodeValue.split(/(\s+)/).forEach(function (part) {
                if (/^\s+$/.test(part)) {
                    frag.appendChild(document.createTextNode(part));
                } else if (part.length > 0) {
                    var span = document.createElement("span");
                    span.className = "word";
                    span.style.setProperty("--word-index", index++);
                    span.textContent = part;
                    frag.appendChild(span);
                }
            });
            textNode.parentNode.replaceChild(frag, textNode);
        });
    }
    var headings = document.querySelectorAll(
        ".md-content h1, .md-content h2, .md-content h3"
    );
    headings.forEach(splitWords);

    /* ---- Version selector pill -------------------------------------------- */
    (function () {
        var logo = document.querySelector('[data-md-component="logo"]');
        if (!logo) return;
        var versionRoot;
        try {
            versionRoot = new URL(logo.getAttribute("href"), location);
        } catch (e) {
            return;
        }
        var versionsUrl = new URL("../versions.json", versionRoot);
        fetch(versionsUrl, { cache: "no-cache" })
            .then(function (r) {
                return r.ok ? r.json() : null;
            })
            .then(function (versions) {
                if (!versions || !versions.length) return;
                var current = versionRoot.pathname
                    .replace(/\/$/, "")
                    .split("/")
                    .pop();
                var entry = versions.filter(function (v) {
                    return v.version === current;
                })[0];
                var isLatest =
                    entry &&
                    entry.aliases &&
                    entry.aliases.indexOf("latest") >= 0;
                var pill = document.createElement("span");
                pill.className = "sx-version-pill";
                pill.textContent = current + (isLatest ? " · latest" : "");
                pill.title = "Documentation version";
                var inner = document.querySelector(".md-header__inner");
                if (inner) inner.appendChild(pill);
            })
            .catch(function () {});
    })();

    /* ---- Page feedback widget --------------------------------------------- */
    function buildFeedback() {
        var content = document.querySelector(".md-content__inner");
        if (!content || content.querySelector(".sx-feedback")) return;
        var key = "sx-feedback:" + location.pathname;
        var box = document.createElement("div");
        box.className = "sx-feedback";
        var saved = null;
        try {
            saved = localStorage.getItem(key);
        } catch (e) {}
        if (saved) {
            box.innerHTML =
                '<span class="sx-feedback-thanks">Thanks for the feedback!</span>';
            content.appendChild(box);
            return;
        }
        box.innerHTML = "<span>Was this page helpful?</span>";
        ["👍", "👎"].forEach(function (emoji) {
            var btn = document.createElement("button");
            btn.type = "button";
            btn.textContent = emoji;
            btn.setAttribute("aria-label", emoji === "👍" ? "Yes" : "No");
            btn.addEventListener("click", function () {
                try {
                    localStorage.setItem(key, emoji);
                } catch (e) {}
                box.innerHTML =
                    '<span class="sx-feedback-thanks">Thanks for the feedback!</span>';
            });
            box.appendChild(btn);
        });
        content.appendChild(box);
    }
    if (document.querySelector(".md-content__inner")) {
        buildFeedback();
    }

    /* ---- Scroll progress bar ---------------------------------------------- */
    var bar = document.createElement("div");
    bar.className = "sx-progress";
    document.body.appendChild(bar);
    window.addEventListener(
        "scroll",
        function () {
            var doc = document.documentElement;
            var max = doc.scrollHeight - doc.clientHeight;
            var pct = max > 0 ? (doc.scrollTop / max) * 100 : 0;
            bar.style.width = pct + "%";
        },
        { passive: true }
    );

    /* ---- Code block copy flash -------------------------------------------- */
    (function () {
        document.addEventListener("click", function (e) {
            var btn = e.target.closest(".md-clipboard");
            if (!btn) return;
            var pre = btn.closest("pre");
            if (!pre) return;
            pre.classList.add("copied");
            setTimeout(function () {
                pre.classList.remove("copied");
            }, 600);
        });
    })();

    /* ---- Smooth scroll for anchor links ----------------------------------- */
    document.addEventListener("click", function (e) {
        var a = e.target.closest('a[href^="#"]');
        if (!a) return;
        var id = a.getAttribute("href").slice(1);
        if (!id) return;
        var target = document.getElementById(id);
        if (target) {
            e.preventDefault();
            target.scrollIntoView({ behavior: "smooth", block: "start" });
            history.replaceState(null, "", "#" + id);
        }
    });

    /* ---- Keyboard shortcut: / to focus search ----------------------------- */
    document.addEventListener("keydown", function (e) {
        if (e.key === "/" && !e.ctrlKey && !e.metaKey && !e.altKey) {
            var active = document.activeElement;
            if (active && (active.tagName === "INPUT" || active.tagName === "TEXTAREA")) return;
            e.preventDefault();
            var toggle = document.querySelector('[data-md-toggle="search"]');
            if (toggle && !toggle.checked) {
                toggle.checked = true;
                toggle.dispatchEvent(new Event("change"));
            }
            var input = document.querySelector(".md-search__input");
            if (input) { input.focus(); input.select && input.select(); }
        }
        if (e.key === "Escape") {
            var toggle = document.querySelector('[data-md-toggle="search"]');
            if (toggle && toggle.checked) {
                toggle.checked = false;
                toggle.dispatchEvent(new Event("change"));
            }
        }
    });

    /* ---- Back-to-top button ----------------------------------------------- */
    (function () {
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "sx-back-to-top";
        btn.setAttribute("aria-label", "Back to top");
        btn.textContent = "↑";
        document.body.appendChild(btn);
        function update() {
            if (window.scrollY > 400) {
                btn.classList.add("sx-back-to-top--visible");
            } else {
                btn.classList.remove("sx-back-to-top--visible");
            }
        }
        window.addEventListener("scroll", update, { passive: true });
        update();
        btn.addEventListener("click", function () {
            window.scrollTo({ top: 0, behavior: "smooth" });
        });
    })();

    /* ---- Stat counter count-up -------------------------------------------- */
    function animateCount(el) {
        var target = parseInt(el.getAttribute("data-count"), 10);
        var suffix = el.getAttribute("data-suffix") || "";
        var duration = 1200;
        var start = null;
        function frame(ts) {
            if (!start) start = ts;
            var progress = Math.min((ts - start) / duration, 1);
            // ease-out cubic
            var eased = 1 - Math.pow(1 - progress, 3);
            el.textContent = Math.round(target * eased).toLocaleString() + suffix;
            if (progress < 1) {
                requestAnimationFrame(frame);
            }
        }
        requestAnimationFrame(frame);
    }
    var counters = document.querySelectorAll("[data-count]");
    if (counters.length) {
        var countObserver = new IntersectionObserver(
            function (entries, observer) {
                entries.forEach(function (entry) {
                    if (entry.isIntersecting) {
                        animateCount(entry.target);
                        observer.unobserve(entry.target);
                    }
                });
            },
            { threshold: 0.4 }
        );
        counters.forEach(function (el) {
            countObserver.observe(el);
        });
    }
})();
