import sys, os, time, json

if "__file__" in globals():
    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../bindings/python"
        ),
    )
else:
    sys.path.insert(0, os.path.join(os.getcwd(), "tests/python/../../bindings/python"))

from SneppX_ALG.interface_bindings.profiler import (
    Profiler,
    ProfileEntry,
    Timer,
    MemoryTracker,
    TrainProfiler,
    timeit,
    get_profiler,
    _GLOBAL_PROFILER,
)

failed = []


def check(name, cond):
    if not cond:
        print(f"  FAIL {name}")
        failed.append(name)
    else:
        print(f"  PASS {name}")


def test_profiler_record():
    p = Profiler()
    p.record("op1", 0.5)
    p.record("op1", 0.3)
    p.record("op2", 1.0)
    e = p.get("op1")
    check("op1 calls", e and e.num_calls == 2)
    check("op1 total", e and abs(e.total_time_s - 0.8) < 1e-9)
    check("op1 avg", e and abs(e.avg_time_s - 0.4) < 1e-9)
    check("op2 calls", p.get("op2") and p.get("op2").num_calls == 1)
    check("op3 missing", p.get("op3") is None)


def test_profiler_disabled():
    p = Profiler(enabled=False)
    p.record("op", 1.0)
    check("disabled no record", p.get("op") is None)


def test_profiler_reset():
    p = Profiler()
    p.record("a", 1.0)
    p.reset()
    check("reset clears", p.get("a") is None)
    check("no entries", len(p._entries) == 0)


def test_timer_context():
    t = Timer("test_op")
    with t:
        time.sleep(0.01)
    check("timer elapsed > 10ms", t.elapsed_s >= 0.008)
    e = get_profiler().get("test_op")
    check("timer recorded in global", e is not None)


def test_timer_start_stop():
    t = Timer("manual")
    t.start()
    time.sleep(0.005)
    elapsed = t.stop()
    check("manual timer", elapsed >= 0.003)


def test_timeit_decorator():
    _GLOBAL_PROFILER.reset()

    @timeit("decorated_fn")
    def dummy():
        time.sleep(0.005)

    dummy()
    e = _GLOBAL_PROFILER.get("decorated_fn")
    check("decorator recorded", e and e.num_calls == 1)


_AUTO_FN_CALLED = 0


@timeit()
def _auto_name_fn():
    import time

    time.sleep(0.001)


def test_timeit_auto_name():
    global _AUTO_FN_CALLED
    _GLOBAL_PROFILER.reset()
    _auto_name_fn()
    e = _GLOBAL_PROFILER.get("_auto_name_fn")
    check("auto name", e is not None and e.num_calls == 1)


def test_profiler_print_summary():
    p = Profiler()
    p.record("a", 0.1)
    p.record("b", 0.2)
    try:
        p.print_summary()
        check("print does not crash", True)
    except Exception:
        check("print does not crash", False)


def test_profiler_to_json():
    p = Profiler()
    p.record("x", 0.5)
    js = p.to_json()
    data = json.loads(js)
    check("json has profiler key", "profiler" in data)
    check("json has entry", len(data["profiler"]) == 1)
    check("json entry name", data["profiler"][0]["name"] == "x")


def test_profiler_save_json():
    import tempfile

    p = Profiler()
    p.record("test", 0.123)
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        path = f.name
    try:
        p.save_json(path)
        with open(path) as f:
            data = json.load(f)
        check("saved json parses", "profiler" in data)
    finally:
        os.unlink(path)


def test_memory_tracker():
    mt = MemoryTracker()
    check("initial peak 0", mt.peak_mb == 0.0)
    mt.update(10 * 1024 * 1024)
    check("10MB alloc", abs(mt.peak_mb - 10.0) < 0.1)
    mt.update(5 * 1024 * 1024)
    check("peak stays 10", abs(mt.peak_mb - 10.0) < 0.1)
    mt.reset()
    check("reset", mt.peak_mb == 0.0)


def test_train_profiler():
    tp = TrainProfiler()
    tp.record_step(0.1)
    tp.record_step(0.2)
    tp.memory.update(100 * 1024 * 1024)
    check("train 2 steps", len(tp._step_times) == 2)
    check("steps/sec window=1", abs(tp.steps_per_sec(1) - 5.0) < 0.01)
    try:
        tp.summary()
        check("summary does not crash", True)
    except Exception:
        check("summary does not crash", False)


if __name__ == "__main__":
    test_profiler_record()
    test_profiler_disabled()
    test_profiler_reset()
    test_timer_context()
    test_timer_start_stop()
    test_timeit_decorator()
    test_timeit_auto_name()
    test_profiler_print_summary()
    test_profiler_to_json()
    test_profiler_save_json()
    test_memory_tracker()
    test_train_profiler()

    print(
        f"\n{'All profiler tests passed!' if not failed else f'{len(failed)} failures: {failed}'}"
    )
