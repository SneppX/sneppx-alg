from SneppX_ALG import observability as ob


def test_span_and_tracer():
    tracer = ob.Tracer("unit")
    with tracer.span("outer", mode="test") as outer:
        outer.set_attribute("k", 1)
        outer.add_event("checkpoint")
        with tracer.span("inner"):
            pass
    assert len(tracer.spans) == 2
    inner = tracer.spans[1]
    assert inner.parent == "outer"
    assert outer.attributes["k"] == 1
    assert outer.duration is not None
    assert tracer.to_dict()["spans"][0]["name"] == "outer"


def test_span_error_status():
    tracer = ob.Tracer()
    try:
        with tracer.span("boom"):
            raise ValueError("x")
    except ValueError:
        pass
    assert tracer.spans[0].status == "error"
    assert tracer.spans[0].end is not None


def test_timer_and_stopwatch():
    with ob.Timer() as t:
        pass
    assert t.elapsed >= 0.0
    sw = ob.Stopwatch()
    sw.start()
    sw.stop()
    assert sw.total >= 0.0
    sw.reset()
    assert sw.total == 0.0


def test_event_bus():
    bus = ob.EventBus()
    seen = []
    bus.on("tick", lambda p: seen.append(p))
    bus.subscribe("tick")(lambda p: seen.append(p * 2))
    out = bus.emit("tick", 3)
    assert out == [None, None]
    assert seen == [3, 6]
    bus.clear("tick")
    assert bus.emit("tick", 1) == []


def test_structured_logger():
    import io

    stream = io.StringIO()
    log = ob.get_logger("test", stream=stream, level="INFO")
    log.info("hello", user="a")
    log.debug("hidden")
    log.error("bad")
    assert len(log.records) == 3
    assert log.get_records("ERROR")[0]["message"] == "bad"
    lines = stream.getvalue().strip().splitlines()
    assert len(lines) == 2
    assert '"message": "hello"' in lines[0]


def test_metric_registry():
    reg = ob.MetricRegistry()
    reg.counter("requests")
    reg.counter("requests", 2)
    assert reg.counters["requests"] == 3
    reg.gauge("load", 0.5)
    assert reg.gauges["load"] == 0.5
    for v in [1, 2, 3, 4, 5]:
        reg.histogram("latency", v)
    summary = reg.summary()
    assert summary["histograms"]["latency"]["count"] == 5
    assert summary["histograms"]["latency"]["mean"] == 3.0
    assert reg.histogram_summary("missing") == {"count": 0}
