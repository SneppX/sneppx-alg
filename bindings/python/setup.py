"""Setup shim for SneppX-ALG Python package.

Canonical metadata lives in pyproject.toml. This file exists only to:
  1. Keep `pip install -e .` working (legacy editable installs)
  2. Force a correct platform wheel tag for the prebuilt C extension
     (`_SNEPPX_c.cp311-win_amd64.pyd`) so wheels are not published as
     `py3-none-any`.
"""

import os
import re
import sys

from setuptools import setup
from setuptools.dist import Distribution

pkg_dir = os.path.join(os.path.dirname(__file__), "SneppX_ALG")


def _find_pyd():
    if not os.path.isdir(pkg_dir):
        return None
    for f in os.listdir(pkg_dir):
        if f.startswith(("_SNEPPX_c", "_sneppx_c", "_arix_c")) and f.endswith(
            (".pyd", ".so")
        ):
            return f
    return None


_pyd = _find_pyd()
_has_c_ext = _pyd is not None


def _detect_tag(pyd_name):
    """Extract (python, abi, plat) from a prebuilt extension name.

    e.g. `_SNEPPX_c.cp311-win_amd64.pyd` -> ("cp311", "cp311", "win_amd64").
    """
    if pyd_name:
        m = re.search(r"\.(cp\d{2,3})-([^.]+)\.(?:pyd|so)$", pyd_name)
        if m:
            return m.group(1), m.group(1), m.group(2)
    import sysconfig

    impl = sys.implementation.name
    ver = "".join(map(str, sys.version_info[:2]))
    return (
        f"{impl}{ver}",
        f"{impl}{ver}",
        sysconfig.get_platform().replace("-", "_").replace(".", "_"),
    )


class BinaryDistribution(Distribution):
    """Force a platform-specific wheel when a prebuilt .pyd is bundled."""

    def has_ext_modules(self):
        return _has_c_ext


_cmdclass = {}
try:
    from wheel.bdist_wheel import bdist_wheel as _bw

    class BdistWheel(_bw):
        """Tag the wheel for the prebuilt C extension's CPython ABI."""

        def get_tag(self):
            if _has_c_ext:
                py, abi, plat = _detect_tag(_pyd)
                return py, abi, plat
            return super().get_tag()

    _cmdclass["bdist_wheel"] = BdistWheel
except ImportError:
    pass


_readme = os.path.join(os.path.dirname(__file__), "README.md")
if not os.path.exists(_readme):
    _readme = os.path.join(os.path.dirname(__file__), "..", "..", "README.md")

_long_description = ""
if os.path.exists(_readme):
    with open(_readme, encoding="utf-8") as readme_file:
        _long_description = readme_file.read()

setup(
    long_description=_long_description,
    long_description_content_type="text/markdown",
    distclass=BinaryDistribution,
    cmdclass=_cmdclass,
    zip_safe=False,
)
