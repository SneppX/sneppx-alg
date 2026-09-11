"""Unified ctypes library loader — finds and loads compiled C shared libraries.

Reusable across all ctypes-based binding modules.  Follows the established
:func:`s5_safety._find_library` search pattern but is library-agnostic.
"""

import os
import ctypes
import sys
from typing import Dict, List, Optional, Tuple


def load_library(
    lib_name: str,
    candidates: Optional[List[str]] = None,
    search_dirs: Optional[List[str]] = None,
    extra_env_var: str = "",
) -> Tuple[Optional[ctypes.CDLL], bool]:
    """Locate and load a shared library.

    Parameters
    ----------
    lib_name:
        Logical name (used only for diagnostics).
    candidates:
        Platform-specific library filenames to try (e.g. ``["mylib.dll",
        "libmylib.so", "libmylib.dylib"]``).  If *None* a best-effort
        default list is generated from *lib_name*.
    search_dirs:
        Directories to search.  If *None* a default list is used that
        includes ``SNEPPX_<LIB_NAME>_LIB``, common build directories, and
        ``.``.
    extra_env_var:
        Optional environment variable name to prepend to *search_dirs*.

    Returns
    -------
    (lib, found) where *lib* is the loaded :class:`ctypes.CDLL` (or
    *None*) and *found* is ``True`` iff loading succeeded.
    """
    if candidates is None:
        candidates = [
            lib_name,
            f"lib{lib_name}",
            f"{lib_name}.dll",
            f"lib{lib_name}.so",
            f"lib{lib_name}.dylib",
        ]

    if search_dirs is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Path: .../sneppx-alg/bindings/python/SneppX_ALG/interface_bindings/
        # Need to go up 4 levels to reach the sneppx-alg project root
        project_root = os.path.normpath(os.path.join(script_dir, "..", "..", "..", ".."))
        # Also check the SneppX root (one level up from project_root)
        sneppx_root = os.path.dirname(project_root)
        env_var_name = f"SNEPPX_{lib_name.upper()}_LIB"
        search_dirs = [
            os.environ.get(env_var_name, ""),
            os.environ.get(extra_env_var, ""),
            os.path.join(project_root, "build"),
            os.path.join(project_root, "build", "Release"),
            os.path.join(project_root, "build", "Debug"),
            os.path.join(sneppx_root, "build"),
            os.path.join(sneppx_root, "build", "Release"),
            os.path.join(sneppx_root, "build", "Debug"),
            ".",
        ]

    for d in search_dirs:
        if not d:
            continue
        for name in candidates:
            full = os.path.join(d, name) if not name.endswith((".dll", ".so", ".dylib")) else name
            try:
                lib = ctypes.CDLL(full)
                return lib, True
            except OSError:
                try:
                    lib = ctypes.CDLL(name)
                    return lib, True
                except OSError:
                    continue
    return None, False


def find_load(
    lib_name: str,
    func_signatures: Optional[Dict[str, Tuple[Optional[List[type]], Optional[type]]]] = None,
    **kwargs,
) -> Tuple[Optional[ctypes.CDLL], bool]:
    """Convenience: call :func:`load_library` and optionally set up function signatures.

    Parameters
    ----------
    lib_name:
        Passed verbatim to :func:`load_library`.
    func_signatures:
        Optional mapping ``{name: (argtypes, restype)}``.  *argtypes*
        may be *None* to skip, *restype* may be *None* for ``void``.
    **kwargs:
        Forwarded to :func:`load_library`.

    Returns
    -------
    (lib, found) — same as :func:`load_library`.
    """
    lib, found = load_library(lib_name, **kwargs)
    if found and func_signatures:
        for fn_name, (argtypes, restype) in func_signatures.items():
            fn = getattr(lib, fn_name, None)
            if fn is not None:
                if argtypes is not None:
                    fn.argtypes = argtypes
                if restype is not None:
                    fn.restype = restype
                else:
                    fn.restype = None
    return lib, found
