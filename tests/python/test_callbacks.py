import os

import numpy as np

from SneppX_ALG import callbacks as cb


def test_callback_list_dispatch():
    calls = []

    class Recorder(cb.Callback):
        def on_epoch_end(self, epoch, logs=None):
            calls.append(epoch)

    cl = cb.CallbackList([Recorder(), Recorder()])
    assert len(cl) == 2
    cl.on_epoch_end(3, {"loss": 1.0})
    assert calls == [3, 3]


def test_history():
    h = cb.History()
    cl = cb.CallbackList([h])
    cl.on_epoch_end(0, {"loss": 1.0, "val_loss": 1.2})
    cl.on_epoch_end(1, {"loss": 0.8, "val_loss": 0.9})
    assert h.history["loss"] == [1.0, 0.8]
    assert h.history["val_loss"] == [1.2, 0.9]
    assert h.epochs == [0, 1]


def test_history_direct():
    h = cb.History()
    cb.CallbackList([h]).on_epoch_end(0, {"a": 1.0})
    cb.CallbackList([h]).on_epoch_end(1, {"a": 2.0})
    assert h.history["a"] == [1.0, 2.0]
    assert h.epochs == [0, 1]


def test_csv_logger(tmp_path):
    path = os.path.join(str(tmp_path), "log.csv")
    logger = cb.CSVLogger(path)
    logger.on_epoch_end(0, {"loss": 1.5, "acc": 0.5})
    logger.on_epoch_end(1, {"loss": 1.0, "acc": 0.7})
    with open(path) as f:
        lines = f.read().strip().splitlines()
    assert lines[0] == "epoch,loss,acc"
    assert lines[1] == "0,1.5,0.5"
    assert lines[2] == "1,1.0,0.7"


def test_lambda_callback():
    seen = []
    c = cb.LambdaCallback(on_epoch_end=lambda e, logs: seen.append(e))
    c.on_epoch_end(7)
    assert seen == [7]


def test_model_checkpoint():
    params = {"w": np.array([1.0, 2.0])}
    ckpt = cb.ModelCheckpoint(monitor="val_loss", mode="min", params_fn=lambda: params)
    ckpt.on_epoch_end(0, {"val_loss": 0.5})
    assert ckpt.best == 0.5 and ckpt.best_epoch == 0
    params["w"] = np.array([9.0, 9.0])
    ckpt.on_epoch_end(1, {"val_loss": 0.9})
    assert ckpt.best == 0.5
    assert np.allclose(ckpt.best_params["w"], [1.0, 2.0])


def test_reduce_lr_on_plateau():
    sched = cb.ReduceLROnPlateau(patience=2, factor=0.5, initial_lr=1.0)
    sched.on_epoch_end(0, {"val_loss": 1.0})
    sched.on_epoch_end(1, {"val_loss": 1.1})
    sched.on_epoch_end(2, {"val_loss": 1.2})
    assert sched.lr == 0.5
    assert sched.num_reduce_lr == 1


def test_warmup_lr():
    w = cb.WarmupLR(base_lr=0.0, warmup_epochs=4, target_lr=1.0)
    w.on_epoch_begin(0)
    assert abs(w.lr - 0.25) < 1e-9
    w.on_epoch_begin(3)
    assert abs(w.lr - 1.0) < 1e-9


def test_progress_bar():
    import io

    stream = io.StringIO()
    pb = cb.ProgressBar(total_epochs=2, stream=stream)
    pb.on_train_begin()
    pb.on_epoch_end(0, {"loss": 1.0})
    pb.on_epoch_end(1, {"loss": 0.5})
    pb.on_train_end()
    assert "2/2" in stream.getvalue()
