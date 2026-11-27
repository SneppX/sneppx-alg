"""Static AST audit of SneppX_ALG python package."""
import ast, os, sys, json, collections

ROOT = r"C:\Users\PC\SneppX\sneppx-alg\bindings\python\SneppX_ALG"

def funcs(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node

def is_ellipsis(stmt):
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis

def is_docstring(stmt):
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)

def is_pass(stmt):
    return isinstance(stmt, ast.Pass)

def is_raise_ni(stmt):
    if not isinstance(stmt, ast.Raise) or stmt.exc is None:
        return False
    e = stmt.exc
    if isinstance(e, ast.Name):
        return e.id in ("NotImplementedError",)
    if isinstance(e, ast.Call):
        f = e.func
        return isinstance(f, ast.Name) and f.id == "NotImplementedError"
    return False

def is_log_only(stmt):
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return False
    c = stmt.value.func
    name = ""
    if isinstance(c, ast.Attribute):
        name = c.attr
    elif isinstance(c, ast.Name):
        name = c.id
    return name in ("warning", "warn", "debug", "info", "error", "print", "log")

def ret_const(stmt):
    """return <literal> -> literal value; return None explicit."""
    if not isinstance(stmt, ast.Return):
        return None
    v = stmt.value
    if v is None:
        return "bare_return"
    if isinstance(v, ast.Constant):
        if v.value is None:
            return "None"
        return repr(v.value)[:40]
    return None

rows = []
tot_lines = 0
per_file = {}
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d != "__pycache__"]
    for fn in sorted(filenames):
        if not fn.endswith(".py"):
            continue
        p = os.path.join(dirpath, fn)
        rel = os.path.relpath(p, ROOT)
        src = open(p, encoding="utf-8", errors="replace").read()
        nlines = src.count("\n") + 1
        tot_lines += nlines
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            rows.append(dict(file=rel, line=e.lineno, kind="SYNTAX_ERROR", name="", detail=str(e)))
            continue
        fl = list(funcs(tree))
        per_file[rel] = (nlines, len(fl))
        for f in fl:
            body = list(f.body)
            nb = [s for s in body if not is_docstring(s)]
            kind = None
            detail = ""
            if not nb:
                kind = "EMPTY_DOCSTRING_ONLY"
                detail = "body is only docstring"
            elif all(is_pass(s) for s in nb):
                kind = "PASS_ONLY"
                detail = "pass"
            elif all(is_ellipsis(s) for s in nb):
                kind = "ELLIPSIS_ONLY"
                detail = "..."
            elif len(nb) == 1 and is_raise_ni(nb[0]):
                kind = "RAISE_NIE"
                detail = "raise NotImplementedError"
            elif nb and all(is_pass(s) or is_log_only(s) for s in nb):
                kind = "LOG_OR_PASS_ONLY"
                detail = "; ".join(type(s).__name__ for s in nb)
            elif len(nb) == 1 and ret_const(nb[0]) is not None:
                kind = "CONST_RETURN"
                detail = "returns " + str(ret_const(nb[0]))
            if kind:
                rows.append(dict(file=rel, line=f.lineno, kind=kind,
                                 name=f.name, detail=detail,
                                 endline=getattr(f, "end_lineno", f.lineno)))

print("TOTAL_LINES", tot_lines)
print("TOTAL_FILES", len(per_file))
cnt = collections.Counter(r["kind"] for r in rows)
print("COUNTS", dict(cnt))
print()
for r in rows:
    print(f'{r["kind"]:22s} {r["file"]}:{r["line"]}  {r["name"]}  [{r["detail"]}]')

with open(r"C:\Users\PC\SneppX\sneppx-alg\.cache\ast_rows.json", "w", encoding="utf-8") as fh:
    json.dump(dict(rows=rows, per_file=per_file, total_lines=tot_lines), fh, indent=1)
