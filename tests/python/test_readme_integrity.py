"""Repository documentation integrity guards.

These are pure-text checks (stdlib only, no framework import) that catch the
classes of damage that have actually corrupted ``README.md`` in the past:

* stray C0 control characters introduced by shell/editor escaping (a form
  feed replaced the ``f`` of ``from``; a literal ```n`` replaced a newline
  and fused two table rows);
* unbalanced fenced-code delimiters, which break rendering for everything
  after them;
* Markdown table rows whose column count disagrees with the table header,
  which GitHub renders as plain text instead of a table.

Keeping these green means the landing page stays renderable.
"""

import io
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

_README = os.path.join(_ROOT, "README.md")


def _read(path):
    with io.open(path, encoding="utf-8") as f:
        return f.read()


def _readme_lines():
    return _read(_README).split("\n")


def test_readme_exists():
    assert os.path.isfile(_README), "README.md missing at %s" % _README


def test_no_stray_control_characters():
    """No C0 control chars other than the newline separator."""
    offenders = []
    for lineno, line in enumerate(_readme_lines(), 1):
        for ch in line:
            if ord(ch) < 32:
                offenders.append((lineno, ord(ch)))
    assert not offenders, "stray control chars (line, codepoint): %r" % offenders


def test_code_fences_are_balanced():
    """Every ``` opener must have a matching closer."""
    fences = [
        i
        for i, line in enumerate(_readme_lines(), 1)
        if line.lstrip().startswith("```")
    ]
    assert len(fences) % 2 == 0, (
        "unbalanced code fences (%d total); opener lines: %r" % (len(fences), fences)
    )


def test_table_rows_match_header_width():
    """Every row in a table must have the same pipe count as its header."""
    lines = _readme_lines()
    header_re = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
    problems = []
    i = 0
    while i < len(lines) - 1:
        if lines[i].lstrip().startswith("|") and header_re.match(lines[i + 1]):
            want = lines[i].count("|")
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                got = lines[j].count("|")
                if got != want:
                    problems.append((j + 1, want, got))
                j += 1
            i = j
        else:
            i += 1
    assert not problems, "table column mismatches (line, want, got): %r" % problems


def test_no_literal_newline_escapes_in_tables():
    """A mangled newline inside a table row leaves a literal ``|`n|`` seam.

    A bare backtick followed by ``n`` is NOT an error — inline code spans such
    as ``net/distributed/nccl.c`` produce it legitimately. The damage
    signature is a *pipe* immediately preceding that seam.
    """
    bad = [
        i
        for i, line in enumerate(_readme_lines(), 1)
        if line.lstrip().startswith("|") and "|`n" in line
    ]
    assert not bad, "literal `n newline artifacts inside table rows: %r" % bad


def test_trailing_whitespace_free():
    """Trailing spaces create unintended hard line breaks when rendered."""
    bad = [i for i, line in enumerate(_readme_lines(), 1) if line != line.rstrip()]
    assert not bad, "trailing whitespace on lines: %r" % bad
