"""Check __all__ resolution + import health for every module."""
import ast, os, sys, io, json, traceback, importlib

ROOT = r"C:\Users\PC\SneppX\sneppx-alg\bindings\python\SneppX_ALG"
sys.path.insert(0, r"C:\Users\PC\SneppX\sneppx-alg\bindings\python")

mods = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d != "__pycache__"]
    for fn in sorted(filenames):
        if fn.endswith(".py"):
            rel = os.path.relpath(os.path.join(dirpath, fn), ROOT)
            mod = "SneppX_ALG." + rel[:-3].replace(os.sep, ".")
            if mod.endswith(".__init__"):
                mod = mod[: -len(".__init__")]
            mods.append((rel, mod))

# ---- static __all__ resolution
print("=========== STATIC __all__ RESOLUTION ===========")
bad_all = []
for rel, mod in mods:
    p = os.path.join(ROOT, rel)
    src = open(p, encoding="utf-8", errors="replace").read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    bound = set()
    allnodes = []
    star = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    bound.add(t.id)
                    if t.id == "__all__":
                        allnodes.append(node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
        elif isinstance(node, ast.Import):
            for a in node.names:
                bound.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "*":
                    star = True
                else:
                    bound.add(a.asname or a.name)
        elif isinstance(node, (ast.If, ast.Try)):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    bound.add(sub.name)
                elif isinstance(sub, ast.ImportFrom):
                    for a in sub.names:
                        if a.name != "*":
                            bound.add(a.asname or a.name)
                elif isinstance(sub, ast.Import):
                    for a in sub.names:
                        bound.add(a.asname or a.name.split(".")[0])
                elif isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if isinstance(t, ast.Name):
                            bound.add(t.id)
    for node in allnodes:
        try:
            val = ast.literal_eval(node.value)
        except Exception:
            continue
        if not isinstance(val, (list, tuple)):
            continue
        missing = [n for n in val if isinstance(n, str) and n not in bound]
        if star:
            # can't be sure; only report if clearly not bound anywhere
            pass
        if missing:
            bad_all.append((rel, node.lineno, sorted(missing)))
for rel, ln, missing in bad_all:
    print(f"  __all__-MISSING  {rel}:{ln}  ->  {missing}")
print(f"  total __all__ name mismatches: {len(bad_all)}")

# ---- import health
print()
print("=========== IMPORT HEALTH ===========")
buf = io.StringIO()
ok, fail = [], []
for rel, mod in mods:
    try:
        importlib.import_module(mod)
        ok.append(mod)
    except BaseException as e:
        fail.append((rel, mod, type(e).__name__, str(e)[:200]))
print(f"OK: {len(ok)}  FAIL: {len(fail)}")
for rel, mod, et, msg in fail:
    print(f"  IMPORT-FAIL {rel}  [{et}] {msg}")
json.dump(dict(ok=ok, fail=fail, mods=[m for _, m in mods]), open(r"C:\Users\PC\SneppX\sneppx-alg\.cache\imports.json", "w"), indent=1)
