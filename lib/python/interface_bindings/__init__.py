"""
SneppX-ALG Python Library

This path is deprecated. Use bindings/python/ instead:
    from SneppX_ALG import _SNEPPX_c as ax
"""

import warnings
warnings.warn(
    "lib/python/ is deprecated. Use 'from SneppX_ALG import _SNEPPX_c' from bindings/python/.",
    DeprecationWarning,
    stacklevel=2,
)
