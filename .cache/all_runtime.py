"""Runtime __all__ resolution check."""
import importlib, json, sys, ast, os
sys.path.insert(0, r"C:\Users\PC\SneppX\sneppx-alg\bindings\python")
ROOT = r"C:\Users\PC\SneppX\sneppx-alg\bindings\python\SneppX_ALG"

targets = ["interface_bindings.distillation", "interface_bindings.vision_transformers",
           "interface_bindings.vizmon", "interface_bindings.dispatch",
           "SneppX_ALG", "interface_bindings"]
for t in targets:
    modname = t if t == "SneppX_ALG" else ("SneppX_ALG." + t if not t.startswith("SneppX_ALG") else t)
    try:
        m = importlib.import_module(modname)
    except Exception as e:
        print(f"{modname}: IMPORT FAIL {type(e).__name__}: {e}")
        continue
    all_ = getattr(m, "__all__", None)
    print(f"--- {modname}  __all__ len={len(all_) if all_ else None}")
    if not all_:
        continue
    missing = [n for n in all_ if not hasattr(m, n)]
    if missing:
        print(f"    RESOLVE FAIL ({len(missing)}): {missing}")
    else:
        print("    all names resolve OK")
