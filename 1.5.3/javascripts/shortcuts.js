/* SneppX-ALG keyboard-first navigation and shortcut help. */
(function () {
  "use strict";

  var logo = document.querySelector('[data-md-component="logo"]');
  var baseUrl = logo && logo.getAttribute("href")
    ? new URL(logo.getAttribute("href"), location.href)
    : new URL("/", location.href);

  var routes = {
    h: "",
    p: "playground/",
    a: "api/python/",
    c: "cookbook/",
    t: "tutorials/",
    s: "security/",
    b: "blog/",
    r: "roadmap/tensorflow-pytorch-parity/",
    m: "matmul-visualizer/",
    v: "conv-visualizer/",
    o: "optimizer-race/",
    l: "loss-gallery/",
    i: "installation/",
    q: "guide/quickstart/",
    d: "distributed-guide/",
    x: "code-browser/",
    k: "keyboard-shortcuts/",
    f: "FAQ/",
    n: "progress/",
    w: "code-walkthrough/",
    u: "autograd-guide/",
    e: "api-explorer/",
  };

  var helpRows = [
    { keys: "?", action: "Open the searchable shortcut reference" },
    { keys: "Ctrl/Cmd + K", action: "Open the command palette" },
    { keys: "/", action: "Focus documentation search" },
    { keys: "Esc", action: "Close an overlay, palette or search" },
    { keys: "g, then h", action: "Open documentation home" },
    { keys: "g, then g", action: "Go to the top of this page" },
    { keys: "g, then p", action: "Open Python playground" },
    { keys: "g, then a", action: "Open Python API" },
    { keys: "g, then c", action: "Open cookbook" },
    { keys: "g, then t", action: "Open tutorials" },
    { keys: "g, then s", action: "Open security overview" },
    { keys: "g, then b", action: "Open engineering blog" },
    { keys: "g, then r", action: "Open roadmap" },
    { keys: "g, then m", action: "Open matrix multiplication visualizer" },
    { keys: "g, then v", action: "Open convolution visualizer" },
    { keys: "g, then o", action: "Open optimizer comparison" },
    { keys: "g, then l", action: "Open loss gallery" },
    { keys: "g, then i", action: "Open installation guide" },
    { keys: "g, then q", action: "Open quick start" },
    { keys: "g, then d", action: "Open distributed support guide" },
    { keys: "g, then x", action: "Open repository browser" },
    { keys: "g, then k", action: "Open keyboard guide" },
    { keys: "g, then f", action: "Open FAQ" },
    { keys: "g, then n", action: "Open learning progress" },
    { keys: "g, then w", action: "Open code walkthrough" },
    { keys: "g, then u", action: "Open autograd guide" },
    { keys: "g, then e", action: "Open API explorer" },
    { keys: "j / k", action: "Scroll down / up" },
    { keys: "d / u", action: "Scroll down / up by half a screen" },
    { keys: "n / Shift + N", action: "Jump to next / previous heading" },
    { keys: "[ / ]", action: "Open previous / next navigation page" },
    { keys: "Home / End", action: "Go to top / bottom of page" },
    { keys: "Alt + Shift + Left / Right", action: "Open previous / next navigation page" },
    { keys: "Alt + Shift + Up / Down", action: "Jump to previous / next heading" },
    { keys: "Alt + Shift + H", action: "Open documentation home" },
    { keys: "Alt + Shift + P", action: "Open Python playground" },
    { keys: "Alt + Shift + A", action: "Open Python API" },
    { keys: "Alt + Shift + S", action: "Focus documentation search" },
    { keys: "Alt + Shift + K", action: "Open shortcut help" },
    { keys: "Alt + Shift + T", action: "Toggle table of contents" },
    { keys: "Alt + Shift + N", action: "Toggle primary navigation" },
    { keys: "Ctrl/Cmd + Shift + L", action: "Copy current page URL" },
    { keys: "Ctrl/Cmd + Shift + F", action: "Focus documentation search" },
    { keys: "Alt + Shift + 0 / 9", action: "Go to top / bottom of page" },
    { keys: "Ctrl/Cmd + Enter", action: "Run playground snippet" },
    { keys: "Tab", action: "Indent four spaces in playground" },
    { keys: "Shift + Tab", action: "Outdent selected playground lines" },
    { keys: "Ctrl/Cmd + S", action: "Save playground code locally" },
    { keys: "Ctrl/Cmd + Shift + S", action: "Copy playground share link" },
    { keys: "Ctrl/Cmd + Shift + Backspace", action: "Clear playground output" },
    { keys: "Ctrl/Cmd + Shift + R", action: "Restore playground starter code" },
    { keys: "Up / Down", action: "Move command-palette selection" },
    { keys: "Enter", action: "Open selected command or page" },
    { keys: "Tab / Shift + Tab", action: "Move focus forward / backward" },
    { keys: "Enter / Space", action: "Activate focused link or button" },
  ];

  var sequence = "";
  var sequenceTimer = null;
  var overlay = null;
  var toast = null;
  var returnFocus = null;

  function isTyping(target) {
    if (!target) return false;
    var tag = (target.tagName || "").toLowerCase();
    return (
      target.isContentEditable ||
      tag === "input" ||
      tag === "textarea" ||
      tag === "select"
    );
  }

  function navigate(path) {
    window.location.href = new URL(path, baseUrl).href;
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

  function scrollByViewport(amount) {
    window.scrollBy({
      top: amount * Math.max(240, window.innerHeight * 0.42),
      behavior: "smooth",
    });
  }

  function moveHeading(direction) {
    var headings = Array.prototype.slice
      .call(
        document.querySelectorAll(
          ".md-content__inner h1, .md-content__inner h2, .md-content__inner h3, .md-content__inner h4",
        ),
      )
      .filter(function (heading) {
        return heading.getClientRects().length > 0;
      });

    if (!headings.length) return;

    var threshold = window.scrollY + 130;
    var target = null;

    if (direction > 0) {
      target =
        headings.find(function (heading) {
          return heading.getBoundingClientRect().top + window.scrollY > threshold;
        }) || headings[headings.length - 1];
    } else {
      for (var i = headings.length - 1; i >= 0; i--) {
        if (
          headings[i].getBoundingClientRect().top + window.scrollY <
          threshold - 20
        ) {
          target = headings[i];
          break;
        }
      }
      target = target || headings[0];
    }

    target.setAttribute("tabindex", "-1");
    target.focus({ preventScroll: true });
    target.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function moveNavigationPage(direction) {
    var links = Array.prototype.slice.call(
      document.querySelectorAll(".md-nav--primary a.md-nav__link"),
    );
    if (!links.length) return;

    var current = links.findIndex(function (link) {
      return link.classList.contains("md-nav__link--active");
    });
    if (current < 0) current = 0;

    var next = Math.max(0, Math.min(links.length - 1, current + direction));
    if (links[next] && links[next].href) {
      window.location.href = links[next].href;
    }
  }

  function announce(message) {
    if (!toast) {
      toast = document.createElement("div");
      toast.className = "sx-shortcuts-toast";
      toast.setAttribute("role", "status");
      toast.setAttribute("aria-live", "polite");
      document.body.appendChild(toast);
    }

    toast.textContent = message;
    toast.classList.add("is-visible");
    window.clearTimeout(toast._timer);
    toast._timer = window.setTimeout(function () {
      toast.classList.remove("is-visible");
    }, 1500);
  }

  function closeOverlay() {
    if (overlay) overlay.classList.remove("sx-shortcuts--open");
    if (returnFocus && returnFocus.focus) returnFocus.focus();
  }

  function renderRows(body, query) {
    body.replaceChildren();
    var normalized = query.toLowerCase().trim();
    var matches = helpRows.filter(function (row) {
      return (
        !normalized ||
        (row.keys + " " + row.action).toLowerCase().indexOf(normalized) !== -1
      );
    });

    matches.forEach(function (row) {
      var tr = document.createElement("tr");
      var keyCell = document.createElement("td");
      var actionCell = document.createElement("td");
      var key = document.createElement("kbd");

      key.textContent = row.keys;
      keyCell.appendChild(key);
      actionCell.textContent = row.action;
      tr.append(keyCell, actionCell);
      body.appendChild(tr);
    });

    if (!matches.length) {
      var emptyRow = document.createElement("tr");
      var emptyCell = document.createElement("td");
      emptyCell.colSpan = 2;
      emptyCell.textContent = "No matching shortcuts.";
      emptyRow.appendChild(emptyCell);
      body.appendChild(emptyRow);
    }
  }

  function buildOverlay() {
    if (overlay) return;

    overlay = document.createElement("div");
    overlay.className = "sx-shortcuts";

    var box = document.createElement("section");
    box.className = "sx-shortcuts__box";
    box.setAttribute("role", "dialog");
    box.setAttribute("aria-modal", "true");
    box.setAttribute("aria-label", "Keyboard shortcuts");

    var header = document.createElement("div");
    header.className = "sx-shortcuts__header";

    var title = document.createElement("h2");
    title.textContent = "Keyboard shortcuts";

    var closeButton = document.createElement("button");
    closeButton.type = "button";
    closeButton.className = "sx-shortcuts__close";
    closeButton.textContent = "Close";
    closeButton.addEventListener("click", closeOverlay);
    header.append(title, closeButton);

    var filter = document.createElement("input");
    filter.type = "search";
    filter.className = "sx-shortcuts__filter";
    filter.placeholder = "Filter shortcuts…";
    filter.setAttribute("aria-label", "Filter keyboard shortcuts");

    var table = document.createElement("table");
    table.className = "sx-shortcuts__table";
    var body = document.createElement("tbody");
    table.appendChild(body);

    var hint = document.createElement("p");
    hint.className = "sx-shortcuts__hint";
    hint.textContent = "Press ? or Esc to close. Shortcuts pause while typing.";

    box.append(header, filter, table, hint);
    overlay.appendChild(box);
    document.body.appendChild(overlay);

    filter.addEventListener("input", function () {
      renderRows(body, filter.value);
    });

    overlay.addEventListener("click", function (event) {
      if (event.target === overlay) closeOverlay();
    });

    overlay.addEventListener("keydown", function (event) {
      if (event.key !== "Tab") return;

      var focusable = box.querySelectorAll("button, input");
      var first = focusable[0];
      var last = focusable[focusable.length - 1];

      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    });

    overlay._filter = filter;
    renderRows(body, "");
  }

  function openOverlay() {
    returnFocus = document.activeElement;
    buildOverlay();
    overlay.classList.add("sx-shortcuts--open");
    overlay._filter.value = "";
    overlay._filter.dispatchEvent(new Event("input"));
    overlay._filter.focus();
  }

  function handleSequence(key) {
    if (!sequence) {
      if (key.toLowerCase() !== "g") return false;

      sequence = "g";
      window.clearTimeout(sequenceTimer);
      sequenceTimer = window.setTimeout(function () {
        sequence = "";
      }, 1100);
      announce("Go to…");
      return true;
    }

    window.clearTimeout(sequenceTimer);
    sequence = "";

    if (key.toLowerCase() === "g") {
      window.scrollTo({ top: 0, behavior: "smooth" });
      return true;
    }

    var route = routes[key.toLowerCase()];
    if (route !== undefined) {
      navigate(route);
      return true;
    }

    return false;
  }

  document.addEventListener(
    "keydown",
    function (event) {
      var target = event.target || document.activeElement;

      if (event.key === "Escape") {
        if (overlay && overlay.classList.contains("sx-shortcuts--open")) {
          closeOverlay();
          return;
        }

        var palette = document.querySelector(".sx-palette--open");
        if (palette) {
          palette.classList.remove("sx-palette--open");
          return;
        }

        var searchToggle = document.querySelector('[data-md-toggle="search"]');
        if (searchToggle && searchToggle.checked) {
          searchToggle.checked = false;
          searchToggle.dispatchEvent(new Event("change", { bubbles: true }));
          return;
        }
      }

      if (isTyping(target)) return;

      if (
        !event.ctrlKey &&
        !event.metaKey &&
        !event.altKey &&
        handleSequence(event.key)
      ) {
        event.preventDefault();
        return;
      }

      if (
        !event.ctrlKey &&
        !event.metaKey &&
        !event.altKey &&
        event.key === "?"
      ) {
        event.preventDefault();
        openOverlay();
        return;
      }

      if (
        (event.ctrlKey || event.metaKey) &&
        event.shiftKey &&
        event.key.toLowerCase() === "l"
      ) {
        event.preventDefault();
        if (navigator.clipboard) {
          navigator.clipboard.writeText(window.location.href).then(function () {
            announce("Page link copied.");
          });
        }
        return;
      }

      if (
        (event.ctrlKey || event.metaKey) &&
        event.shiftKey &&
        event.key.toLowerCase() === "f"
      ) {
        event.preventDefault();
        focusSearch();
        return;
      }

      if (event.altKey && event.shiftKey) {
        var key = event.key.toLowerCase();
        var altRoutes = {
          h: "",
          p: "playground/",
          a: "api/python/",
        };

        if (Object.prototype.hasOwnProperty.call(altRoutes, key)) {
          event.preventDefault();
          navigate(altRoutes[key]);
          return;
        }

        if (key === "s") {
          event.preventDefault();
          focusSearch();
          return;
        }
        if (key === "k") {
          event.preventDefault();
          openOverlay();
          return;
        }
        if (key === "t") {
          event.preventDefault();
          var toc = document.querySelector('[data-md-toggle="toc"]');
          if (toc) toc.click();
          return;
        }
        if (key === "n") {
          event.preventDefault();
          var nav = document.querySelector('[data-md-toggle="navigation"]');
          if (nav) nav.click();
          return;
        }
        if (key === "arrowleft") {
          event.preventDefault();
          moveNavigationPage(-1);
          return;
        }
        if (key === "arrowright") {
          event.preventDefault();
          moveNavigationPage(1);
          return;
        }
        if (key === "arrowup") {
          event.preventDefault();
          moveHeading(-1);
          return;
        }
        if (key === "arrowdown") {
          event.preventDefault();
          moveHeading(1);
          return;
        }
        if (key === "0") {
          event.preventDefault();
          window.scrollTo({ top: 0, behavior: "smooth" });
          return;
        }
        if (key === "9") {
          event.preventDefault();
          window.scrollTo({
            top: document.documentElement.scrollHeight,
            behavior: "smooth",
          });
          return;
        }
      }

      if (
        event.key === "/" &&
        !event.ctrlKey &&
        !event.metaKey &&
        !event.altKey
      ) {
        event.preventDefault();
        focusSearch();
        return;
      }
      if (event.key === "[") {
        event.preventDefault();
        moveNavigationPage(-1);
        return;
      }
      if (event.key === "]") {
        event.preventDefault();
        moveNavigationPage(1);
        return;
      }
      if (event.key === "j") {
        event.preventDefault();
        scrollByViewport(1);
        return;
      }
      if (event.key === "k") {
        event.preventDefault();
        scrollByViewport(-1);
        return;
      }
      if (event.key === "d") {
        event.preventDefault();
        scrollByViewport(1.5);
        return;
      }
      if (event.key === "u") {
        event.preventDefault();
        scrollByViewport(-1.5);
        return;
      }
      if (event.key === "n") {
        event.preventDefault();
        moveHeading(1);
        return;
      }
      if (event.key === "N") {
        event.preventDefault();
        moveHeading(-1);
        return;
      }
      if (event.key === "Home") {
        event.preventDefault();
        window.scrollTo({ top: 0, behavior: "smooth" });
        return;
      }
      if (event.key === "End") {
        event.preventDefault();
        window.scrollTo({
          top: document.documentElement.scrollHeight,
          behavior: "smooth",
        });
      }
    },
    true,
  );

  window.SxShortcuts = {
    open: openOverlay,
    close: closeOverlay,
    routes: routes,
    count: helpRows.length,
  };
})();