import sys
sys.path.insert(0, r"C:\Users\PC\SneppX\sneppx-alg\bindings\python")
import SneppX_ALG as sx
import SneppX_ALG.interface_bindings as ib

names = ['zeros_', 'xavier_uniform', 'kaiming_normal_', 'calculate_gain', 'conv2d',
         'linear', 'batch_norm', 'relu', 'jvp', 'vjp', 'grad', 'Tensor', 'Linear',
         'Trainer', 'Experiment', 'run_benchmarks']
for n in names:
    o = getattr(ib, n, None)
    mod = getattr(o, '__module__', None)
    print("%-18s -> %s" % (n, mod))

print("--- functional module ---")
print("has attr functional:", hasattr(ib, 'functional'))
try:
    import SneppX_ALG.interface_bindings.functional as f
    print("functional import OK; names:", [d for d in dir(f) if not d.startswith('_')][:14])
except Exception as e:
    print("ERR", type(e).__name__, e)

print("--- root sx.functional ---")
try:
    print("sx.functional:", type(sx.functional))
except Exception as e:
    print("sx.functional ->", type(e).__name__, e)

print("--- init vs nn_init loaded ---")
print("init loaded:", 'SneppX_ALG.interface_bindings.init' in sys.modules)
print("nn_init loaded:", 'SneppX_ALG.interface_bindings.nn_init' in sys.modules)

print("--- bridge fallback ---")
b = sx._neural_engine_bridge
print("bridge:", b.__name__, "attrs:", len([d for d in dir(b) if not d.startswith('_')]))
try:
    print("SNEPPXDtype:", b.SNEPPXDtype)
except Exception as e:
    print("SNEPPXDtype ->", type(e).__name__, e)
