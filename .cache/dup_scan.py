"""Duplicate-name, collision and orphan analysis."""
import ast, os, sys, json, collections, importlib

PYROOT = r"C:\Users\PC\SneppX\sneppx-alg\bindings\python"
ROOT = os.path.join(PYROOT, "SneppX_ALG")
sys.path.insert(0, PYROOT)

# ---- collect module-level defs + imports per file
files = {}
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d != "__pycache__"]
    for fn in sorted(filenames):
        if not fn.endswith(".py"):
            continue
        p = os.path.join(dirpath, fn)
        rel = os.path.relpath(p, ROOT)
        src = open(p, encoding="utf-8", errors="replace").read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        defs = {}
        imports = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defs.setdefault(node.name, node.lineno)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level and node.module:
                imports.add(node.module.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level and not node.module:
                pass
        files[rel] = dict(defs=defs, imports=imports, lines=src.count("\n") + 1)

# ---- name collisions among interface_bindings modules
print("======= CROSS-MODULE TOP-LEVEL NAME COLLISIONS (interface_bindings root) =======")
owner = collections.defaultdict(list)
for rel, d in files.items():
    if not rel.startswith("interface_bindings") or os.sep in rel[len("interface_bindings") + 1:]:
        pass
    if rel.startswith("interface_bindings") and rel.count(os.sep) == 1:
        for name, ln in d["defs"].items():
            owner[name].append((rel, ln))
dups = {k: v for k, v in owner.items() if len(v) > 1}
interesting = ["run_benchmarks", "BenchmarkConfig", "BenchmarkSuite", "BenchmarkResult",
               "BenchmarkTimer", "Trainer", "Experiment", "main", "Evaluator", "TrainerConfig",
               "evaluate", "train", "ModelConfig"]
print(f"  total colliding names: {len(dups)}")
for k in interesting:
    if k in dups:
        print(f"  COLLISION {k}:")
        for rel, ln in dups[k]:
            print(f"      {rel}:{ln}")
print("  --- all collisions (name: files) ---")
for k in sorted(dups):
    mods = ",".join(os.path.basename(r) for r, _ in dups[k])
    print(f"    {k}  <-  {mods}")

# ---- which module wins for a name in the package namespace
print()
print("======= RESOLVED SYMBOL ORIGIN IN SneppX_ALG.interface_bindings =======")
ib = importlib.import_module("SneppX_ALG.interface_bindings")
for k in interesting:
    if hasattr(ib, k):
        obj = getattr(ib, k)
        print(f"  {k} -> {getattr(obj, '__module__', '?')}")

# ---- orphan detection: modules never imported by any other module
print()
print("======= ORPHAN MODULES (never imported by any other file) =======")
imported = set()
for rel, d in files.items():
    for imp in d["imports"]:
        imported.add(imp)
    # also catch `from . import x`
for rel in sorted(files):
    base = os.path.basename(rel)[:-3]
    if base == "__init__":
        continue
    stem = base
    used = False
    for other, d in files.items():
        if other == rel:
            continue
        if stem in d["imports"]:
            used = True
            break
    if not used:
        print(f"  ORPHAN? {rel}  (defs={len(files[rel]['defs'])}, lines={files[rel]['lines']})")

json.dump({k: dict(v) for k, v in files.items()},
          open(r"C:\Users\PC\SneppX\sneppx-alg\.cache\files.json", "w"), indent=1)
