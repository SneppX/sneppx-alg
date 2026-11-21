"""Tests for C Library Loader."""

from SneppX_ALG.interface_bindings.c_loader import load_library, find_load


def test_load_library_nonexistent():
    lib, has_c = load_library("nonexistent_library_xyz")
    assert has_c is False
    assert lib is None


def test_find_load_nonexistent():
    lib, has_c = find_load("nonexistent_library_xyz")
    assert has_c is False
    assert lib is None


if __name__ == "__main__":
    import sys
    for name, fn in sorted({k: v for k, v in locals().items() if k.startswith("test_")}.items()):
        try:
            fn()
            print(f"  PASS {name}")
        except Exception as e:
            print(f"  FAIL {name}: {e}")
    print(f"\n{'='*50}")
    print("  Done")
