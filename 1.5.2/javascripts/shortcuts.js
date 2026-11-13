/* SNEPPX-Alg — keyboard shortcuts overlay (press ?) */
(function () {
    "use strict";
    var shortcuts = [
        ["/", "Focus search"],
        ["Ctrl/Cmd + K", "Open command palette"],
        ["?", "Show this shortcuts overlay"],
        ["Esc", "Close search / palette / overlay"],
        ["↑ ↓", "Navigate palette results"],
        ["Enter", "Open selected result"],
    ];
    var overlay = null;
    function build() {
        if (overlay) return;
        overlay = document.createElement("div");
        overlay.className = "sx-shortcuts";
        var rows = shortcuts.map(function (s) {
            return '<tr><td><kbd>' + s[0] + '</kbd></td><td>' + s[1] + '</td></tr>';
        }).join("");
        overlay.innerHTML =
            '<div class="sx-shortcuts__box">' +
            '  <h3>Keyboard shortcuts</h3>' +
            '  <table>' + rows + '</table>' +
            '  <p class="sx-shortcuts__hint">Press ? or Esc to close.</p>' +
            '</div>';
        document.body.appendChild(overlay);
        overlay.addEventListener("click", function (e) { if (e.target === overlay) close(); });
    }
    function open() { build(); overlay.classList.add("sx-shortcuts--open"); }
    function close() { if (overlay) overlay.classList.remove("sx-shortcuts--open"); }
    document.addEventListener("keydown", function (e) {
        var active = document.activeElement;
        if (active && (active.tagName === "INPUT" || active.tagName === "TEXTAREA")) return;
        if (e.key === "?" ) { e.preventDefault(); open(); }
        if (e.key === "Escape") close();
    });
})();