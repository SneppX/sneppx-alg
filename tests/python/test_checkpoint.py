import sys, os, time, json, struct, tempfile, threading, math

_base = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else "C:/Users/PC/SneppX/sneppx-alg/tests/python"
)
if not os.environ.get("PYTHONPATH"):
    sys.path.insert(0, os.path.join(os.path.dirname(_base), "../../bindings/python"))

from SneppX_ALG.interface_bindings.checkpoint import (
    CheckpointWriter,
    CheckpointReader,
    CheckpointCoordinator,
    HeartbeatMonitor,
    ElasticTrainer,
    FaultToleranceManager,
    validate_checkpoint,
    CheckpointHeader,
    TensorRecord,
    SNEPPX_CKPT_MAGIC,
    SNEPPX_CKPT_MAGIC_HI,
    SNEPPX_CKPT_VERSION,
    CHECKPOINT_HEADER_SIZE,
    TENSOR_RECORD_SIZE,
)

failed = []


def check(name, cond):
    if not cond:
        print(f"  FAIL {name}")
        failed.append(name)
    else:
        print(f"  PASS {name}")


def test_write_read_roundtrip():
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".sneppx", delete=False) as f:
        path = f.name
    try:
        w = CheckpointWriter(path)
        try:
            data = b"\x01\x02\x03\x04" * 256
            w.write_tensor(data, shape=(1024,), dtype=0)
            meta = {"step": 42, "loss": 0.5}
            w.write_metadata(meta)
        finally:
            w.close()

        r = CheckpointReader(path)
        check(
            "header magic",
            r.header.magic_lo == SNEPPX_CKPT_MAGIC
            and r.header.magic_hi == SNEPPX_CKPT_MAGIC_HI,
        )
        check("header version", r.header.version == SNEPPX_CKPT_VERSION)
        check("num tensors", r.header.num_tensors == 1)

        read_data = r.read_tensor(0)
        check("data roundtrip", read_data == data)

        read_meta = r.read_metadata()
        check("metadata step", read_meta.get("step") == 42)
        check("metadata loss", read_meta.get("loss") == 0.5)

        check("validate", validate_checkpoint(path))
        r.close()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_multiple_tensors():
    with tempfile.NamedTemporaryFile(suffix=".sneppx", delete=False) as f:
        path = f.name
    try:
        w = CheckpointWriter(path)
        t1 = b"\xaa" * 64
        t2 = b"\xbb" * 128
        t3 = b"\xcc" * 32
        w.write_tensor(t1, shape=(8, 8), dtype=0)
        w.write_tensor(t2, shape=(16, 8), dtype=0)
        w.write_tensor(t3, shape=(8, 4), dtype=0)
        w.write_metadata({"num": 3})
        w.close()

        r = CheckpointReader(path)
        check("3 tensors", r.header.num_tensors == 3)
        check("t1", r.read_tensor(0) == t1)
        check("t2", r.read_tensor(1) == t2)
        check("t3", r.read_tensor(2) == t3)
        r.close()
    finally:
        os.unlink(path)


def test_coordinator_sync_save():
    with tempfile.TemporaryDirectory() as tmpdir:
        coord = CheckpointCoordinator(
            tmpdir, world_size=2, rank=0, save_interval=1, keep_last=3, async_save=False
        )
        state = b"\x00\x01\x02\x03" * 1000
        coord.save(state, 100, metadata={"epoch": 1})
        time.sleep(0.1)

        path = coord.last_checkpoint_path
        check("path exists", path and os.path.exists(path))

        data, meta = coord.load()
        check("loaded data matches", data == state)
        check("meta epoch", meta.get("epoch") == 1)
        check("meta step", meta.get("step") == 100)


def test_coordinator_async_save():
    with tempfile.TemporaryDirectory() as tmpdir:
        coord = CheckpointCoordinator(
            tmpdir, world_size=1, rank=0, save_interval=1, keep_last=2, async_save=True
        )
        state = b"\xff" * 5000
        coord.save(state, 50)
        time.sleep(0.3)
        data, meta = coord.load()
        check("async data", data == state)
        coord.destroy()


def test_coordinator_keep_last():
    with tempfile.TemporaryDirectory() as tmpdir:
        coord = CheckpointCoordinator(
            tmpdir, world_size=1, rank=0, save_interval=1, keep_last=2, async_save=False
        )
        for step in range(1, 6):
            coord.save(b"\x00" * 100, step)
        import glob

        files = glob.glob(os.path.join(tmpdir, "*.sneppx"))
        check("keep_last max 2", len(files) <= 2)


def test_heartbeat_monitor():
    hb = HeartbeatMonitor(world_size=4, rank=0, interval_ms=100, timeout_ms=500)
    hb.start()
    time.sleep(0.3)
    alive = hb.check_alive()
    check("self always alive", alive >= 1)
    alive_ranks = hb.get_alive_ranks()
    check("rank 0 in alive", 0 in alive_ranks)
    hb.report_alive(1)
    time.sleep(0.1)
    hb.report_alive(2)
    alive = hb.check_alive()
    status = hb.get_status()
    check("reported ranks alive", status[1] == 0 and status[2] == 0)
    hb.stop()


def test_heartbeat_suspect_dead():
    hb = HeartbeatMonitor(world_size=3, rank=0, interval_ms=50, timeout_ms=80)
    hb.start()
    time.sleep(0.4)
    alive = hb.check_alive()
    status = hb.get_status()
    hb.stop()
    check("unreported ranks suspect or dead", status[1] != 0 or status[2] != 0)


def test_elastic_join():
    et = ElasticTrainer(world_size=4, rank=0)
    et.handle_join(5)
    check("world size grows", et.world_size == 6)
    check("version bumped", et.version >= 2)


def test_elastic_leave():
    et = ElasticTrainer(world_size=4, rank=0)
    et.handle_leave(2)
    check("rank 2 removed", not et._alive_ranks[2])
    check("version bumped", et.version >= 2)


def test_elastic_leave_self():
    et = ElasticTrainer(world_size=4, rank=0)
    try:
        et.handle_leave(0)
        check("leave self raises", False)
    except RuntimeError:
        check("leave self raises", True)


def test_elastic_failure_max_restarts():
    et = ElasticTrainer(world_size=4, rank=0, max_restarts=2)
    for i in range(2):
        et.handle_failure(1)
    check("restart count", et.restart_count == 2)
    try:
        et.handle_failure(2)
        check("exceed max raises", False)
    except RuntimeError:
        check("exceed max raises", True)


def test_elastic_get_topology():
    et = ElasticTrainer(world_size=4, rank=2)
    et.handle_leave(0)
    et.handle_leave(3)
    new_world, new_rank = et.get_new_topology()
    check("smaller world", new_world == 2)
    check("rank renumbered", new_rank == 1)


def test_fault_tolerance_manager():
    ft = FaultToleranceManager(
        world_size=4, rank=0, heartbeat_interval_ms=100, timeout_ms=500, max_restarts=2
    )
    ft.start()
    time.sleep(0.2)
    alive = ft.check_health()
    check("ft health check works", alive >= 1)
    ft.stop()


def test_invalid_checkpoint():
    with tempfile.NamedTemporaryFile(suffix=".sneppx", delete=False) as f:
        f.write(b"garbage data")
        path = f.name
    try:
        check("invalid rejected", not validate_checkpoint(path))
    finally:
        os.unlink(path)


def test_empty_metadata():
    with tempfile.NamedTemporaryFile(suffix=".sneppx", delete=False) as f:
        path = f.name
    try:
        w = CheckpointWriter(path)
        w.write_tensor(b"\x00" * 16, shape=(4, 4), dtype=0)
        w.close()
        r = CheckpointReader(path)
        meta = r.read_metadata()
        check("empty metadata", meta == {})
        r.close()
    finally:
        os.unlink(path)


def test_coordinated_save():
    barrier_called = [0]

    def barrier():
        barrier_called[0] += 1

    with tempfile.TemporaryDirectory() as tmpdir:
        coord = CheckpointCoordinator(
            tmpdir, world_size=2, rank=0, save_interval=1, keep_last=3, async_save=False
        )
        coord.coordinated_save(b"\x01" * 100, 200, barrier_fn=barrier)
        check("barrier called twice", barrier_called[0] == 2)


if __name__ == "__main__":
    test_write_read_roundtrip()
    test_multiple_tensors()
    test_coordinator_sync_save()
    test_coordinator_async_save()
    test_coordinator_keep_last()
    test_heartbeat_monitor()
    test_heartbeat_suspect_dead()
    test_elastic_join()
    test_elastic_leave()
    test_elastic_leave_self()
    test_elastic_failure_max_restarts()
    test_elastic_get_topology()
    test_fault_tolerance_manager()
    test_invalid_checkpoint()
    test_empty_metadata()
    test_coordinated_save()

    print(
        f"\n{'All checkpoint tests passed!' if not failed else f'{len(failed)} failures: {failed}'}"
    )
