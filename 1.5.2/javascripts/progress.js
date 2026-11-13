/* SNEPPX-Alg — local learning progress.
 *
 * Everything is stored in localStorage on the reader's own machine. Nothing is
 * sent anywhere: there is no server, no analytics and no leaderboard.
 *
 * Public surface (window.SxProgress):
 *   slug()            current page id
 *   isRead(slug)      has this page been read?
 *   markRead()        mark the current page as read
 *   complete(key)     mark an arbitrary achievement key (e.g. a quiz id)
 *   tracks()          curated learning tracks with completion
 *   renderRing(el)    draw the sidebar progress ring
 *   renderPanel(el)   draw the full progress panel
 */
(function () {
    "use strict";

    var STORE = "sx-progress-v1";

    var TRACKS = [
        {
            id: "start",
            name: "Start here",
            pages: [
                ["getting-started", "Getting started"],
                ["playground", "Playground"],
                ["cookbook", "Cookbook"],
                ["keyboard-shortcuts", "Keyboard shortcuts"],
            ],
        },
        {
            id: "autograd",
            name: "Autograd",
            pages: [
                ["autograd-guide", "Autograd guide"],
                ["autograd-visualizer", "Autograd visualizer"],
                ["higher-order-ad", "Higher-order AD"],
                ["loss-gallery", "Loss gallery"],
            ],
        },
        {
            id: "ops",
            name: "Tensors & ops",
            pages: [
                ["matmul-visualizer", "Matmul visualizer"],
                ["conv-visualizer", "Conv2d visualizer"],
                ["sparse-tensors", "Sparse tensors"],
                ["optimizer-race", "Optimizer race"],
            ],
        },
        {
            id: "scale",
            name: "Scaling out",
            pages: [
                ["distributed-guide", "Distributed guide"],
                ["architecture/distributed_training", "Distributed training"],
            ],
        },
        {
            id: "practice",
            name: "Practice",
            pages: [
                ["quiz", "Quiz"],
            ],
        },
    ];

    function load() {
        try {
            var raw = localStorage.getItem(STORE);
            return raw ? JSON.parse(raw) : {};
        } catch (e) {
            return {};
        }
    }

    function save(s) {
        try {
            localStorage.setItem(STORE, JSON.stringify(s));
        } catch (e) {}
    }

    function slug() {
        var p = location.pathname.replace(/\/index\.html$/, "/");
        var segs = p.split("/").filter(Boolean);
        // strip an org/repo prefix plus an optional mike version segment
        for (var i = 0; i < segs.length; i++) {
            if (segs[i] === "latest" || /^\d+\.\d+\.\d+$/.test(segs[i])) {
                segs = segs.slice(i + 1);
                break;
            }
        }
        if (!segs.length) return "index";
        var last = segs[segs.length - 1];
        if (last === "sneppx-alg") return "index";
        return segs.join("/");
    }

    function isRead(id) {
        return !!load()["read:" + id];
    }

    function markRead(id) {
        id = id || slug();
        var s = load();
        if (s["read:" + id]) return false;
        s["read:" + id] = Date.now();
        save(s);
        return true;
    }

    function complete(key) {
        var s = load();
        if (s["done:" + key]) return false;
        s["done:" + key] = Date.now();
        save(s);
        return true;
    }

    function isComplete(key) {
        return !!load()["done:" + key];
    }

    function trackState(t) {
        var done = 0;
        t.pages.forEach(function (p) {
            if (isRead(p[0]) || isComplete("track:" + t.id + ":" + p[0])) done++;
        });
        return { total: t.pages.length, done: done, pct: t.pages.length ? done / t.pages.length : 0 };
    }

    function tracks() {
        return TRACKS.map(function (t) {
            var st = trackState(t);
            return {
                id: t.id,
                name: t.name,
                pages: t.pages.map(function (p) {
                    return { slug: p[0], title: p[1], done: isRead(p[0]) || isComplete("track:" + t.id + ":" + p[0]) };
                }),
                done: st.done,
                total: st.total,
                pct: st.pct,
            };
        });
    }

    function overall() {
        var all = tracks();
        var done = 0, total = 0;
        all.forEach(function (t) { done += t.done; total += t.total; });
        return { done: done, total: total, pct: total ? done / total : 0 };
    }

    /* ---------------- rendering ---------------- */

    function ringSvg(pct, size) {
        var r = size / 2 - 4;
        var c = 2 * Math.PI * r;
        var off = c * (1 - pct);
        return (
            '<svg class="sx-ring-svg" viewBox="0 0 ' + size + " " + size + '">' +
            '<circle cx="' + size / 2 + '" cy="' + size / 2 + '" r="' + r +
            '" class="sx-ring-bg" fill="none" stroke-width="4"></circle>' +
            '<circle cx="' + size / 2 + '" cy="' + size / 2 + '" r="' + r +
            '" class="sx-ring-fg" fill="none" stroke-width="4" ' +
            'stroke-dasharray="' + c.toFixed(2) + '" stroke-dashoffset="' + off.toFixed(2) + '" ' +
            'transform="rotate(-90 ' + size / 2 + " " + size / 2 + ')"></circle>' +
            "</svg>"
        );
    }

    function renderRing(el) {
        if (!el) return;
        var o = overall();
        el.innerHTML =
            '<div class="sx-ring">' + ringSvg(o.pct, 56) +
            '<span class="sx-ring-label">' + Math.round(o.pct * 100) + "%</span></div>" +
            '<div class="sx-ring-caption">' + o.done + "/" + o.total + " pages</div>";
        el.classList.add("sx-ring-host");
    }

    function renderPanel(el) {
        if (!el) return;
        var o = overall();
        var html = '<div class="sx-progress-head">' + ringSvg(o.pct, 84) +
            '<div><div class="sx-progress-pct">' + Math.round(o.pct * 100) + "%</div>" +
            '<div class="sx-progress-sub">' + o.done + " of " + o.total + " pages</div></div></div>";

        tracks().forEach(function (t) {
            html += '<div class="sx-track"><div class="sx-track-head">' +
                '<span class="sx-track-name">' + t.name + "</span>" +
                '<span class="sx-track-count">' + t.done + "/" + t.total + "</span></div>" +
                '<div class="sx-track-bar"><i style="width:' + Math.round(t.pct * 100) + '%"></i></div>' +
                '<ul class="sx-track-pages">';
            t.pages.forEach(function (p) {
                html += '<li class="' + (p.done ? "is-done" : "") + '">' +
                    '<span class="sx-check">' + (p.done ? "✔" : "○") + "</span>" +
                    '<a href="' + linkFor(p.slug) + '">' + p.title + "</a></li>";
            });
            html += "</ul></div>";
        });

        el.innerHTML = html;
        el.classList.add("sx-progress-panel");
    }

    function linkFor(id) {
        return id === "index" ? "../" : "../" + id + "/";
    }

    /* ---------------- auto tracking ---------------- */

    function autoTrack() {
        var id = slug();
        if (isRead(id)) return;
        var marked = false;
        function tryMark() {
            if (marked) return;
            var doc = document.documentElement;
            var atBottom = window.scrollY + window.innerHeight >= doc.scrollHeight - 220;
            if (atBottom || doc.scrollHeight <= window.innerHeight + 40) {
                marked = markRead(id);
                if (marked) {
                    var el = document.getElementById("sx-progress-ring");
                    if (el) renderRing(el);
                }
            }
        }
        window.addEventListener("scroll", tryMark, { passive: true });
        setTimeout(tryMark, 1200);
    }

    window.SxProgress = {
        slug: slug,
        isRead: isRead,
        markRead: markRead,
        complete: complete,
        isComplete: isComplete,
        tracks: tracks,
        overall: overall,
        renderRing: renderRing,
        renderPanel: renderPanel,
        linkFor: linkFor,
    };

    function boot() {
        var ring = document.getElementById("sx-progress-ring");
        if (!ring) {
            var side = document.querySelector(".md-sidebar--secondary .md-sidebar__inner");
            if (side) {
                ring = document.createElement("div");
                ring.id = "sx-progress-ring";
                ring.className = "sx-ring-sidebar";
                side.insertBefore(ring, side.firstChild);
            }
        }
        if (ring) renderRing(ring);
        var panel = document.getElementById("sx-progress-panel");
        if (panel) renderPanel(panel);
        autoTrack();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot);
    } else {
        boot();
    }
})();
