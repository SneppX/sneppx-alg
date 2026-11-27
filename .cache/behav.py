import sys
sys.path.insert(0, r"C:\Users\PC\SneppX\sneppx-alg\bindings\python")
import numpy as np

from SneppX_ALG.interface_bindings import graph_optimizer as go
print("=== graph_optimizer.trace() ===")
print("  trace returns:", repr(go.trace(lambda a, b: a * b, 1, 2)))

from SneppX_ALG.interface_bindings.s4_network import TransportSecurity
ts = TransportSecurity()
print("=== TransportSecurity.verify_certificate ===")
print("  bogus cert, wrong CN ->", ts.verify_certificate("", "totally-wrong.example"))
print("  garbage cert         ->", ts.verify_certificate("NOT A CERT", "x"))

from SneppX_ALG.interface_bindings.runtime_monitor import AdvancedMonitor
print("=== AdvancedMonitor.detect_anomaly ===")
print("  all-zero metrics ->", AdvancedMonitor.detect_anomaly({}))
print("  extreme metrics  ->", AdvancedMonitor.detect_anomaly({"cpu": 999.0, "err": 1e9}))

from SneppX_ALG.interface_bindings import secure_memory as sm
print("=== secure_memory ===")
for fn in ("w_xor_x", "seccomp_filter"):
    f = getattr(sm, fn)
    try:
        print("  %s ->" % fn, f())
    except Exception as e:
        print("  %s -> %s: %s" % (fn, type(e).__name__, e))

from SneppX_ALG.interface_bindings import model_implementations as mi
print("=== model_implementations.get_model_flops ===")
try:
    print("  ->", mi.get_model_flops(None))
except Exception as e:
    print("  ->", type(e).__name__, e)

from SneppX_ALG.interface_bindings import hw_backends as hb
print("=== hw_backends available flags ===")
import inspect
for nm, obj in vars(hb).items():
    if inspect.isclass(obj):
        if "available" in vars(obj):
            try:
                print("  %s.available ->" % nm, obj().available)
            except Exception as e:
                print("  %s.available -> %s" % (nm, type(e).__name__))
