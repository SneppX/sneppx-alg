import numpy as np
from SneppX_ALG import Model


def test_forward():
    model = Model({"input_dim": 8, "output_dim": 8})
    x = np.random.randn(1, 4, 8).astype(np.float32)
    out = model.forward(x)
    assert out is not None, "Output is None"
    assert out.shape[-1] == 8, f"Expected last dim 8, got {out.shape}"
    print(f"  test_forward output shape: {out.shape} PASS")


def test_parameters():
    model = Model({"input_dim": 8, "output_dim": 8})
    params = model.parameters()
    assert isinstance(params, list), "Parameters should be a list"
    print(f"  test_parameters: {len(params)} params PASS")


if __name__ == "__main__":
    test_forward()
    test_parameters()
    print("All model tests passed.")
