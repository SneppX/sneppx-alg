/* SNEPPX-Alg site scripts
 * - scroll reveal via IntersectionObserver
 * - word-by-word heading reveal on load
 * - stat counter count-up on scroll (elements with .sx-stat-value[data-count])
 * Respects prefers-reduced-motion; no external dependencies.
 */
(function () {
    "use strict";

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
