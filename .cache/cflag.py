import sys, importlib, os, re
sys.path.insert(0, r"C:\Users\PC\SneppX\sneppx-alg\bindings\python")

import SneppX_ALG.interface_bindings as ib
import pkgutil

print("=== c_loader.load_library source ===")
from SneppX_ALG.interface_bindings import c_loader
import inspect
print(inspect.getsource(c_loader.load_library)[:1200])

print()
print("=== _HAS_C / _C values across all modules ===")
import SneppX_ALG
pkg = SneppX_ALG.interface_bindings
found = []
for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
    try:
        mod = importlib.import_module(m.name)
    except Exception:
        continue
    for attr in ("_HAS_C", "_C", "HAS_C", "_LIB"):
        if hasattr(mod, attr):
            v = getattr(mod, attr)
            if attr == "_LIB":
                desc = "None" if v is None else "loaded"
            else:
                desc = repr(v) if not hasattr(v, "__name__") else "module:"+getattr(v,"__name__","?")
            found.append((m.name, attr, desc))
for n, a, d in sorted(found):
    print("  %-58s %-7s = %s" % (n.replace("SneppX_ALG.",""), a, d))
print("  total flags:", len(found))
