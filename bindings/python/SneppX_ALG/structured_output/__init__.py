"""SNEPPX structured output — JSON schema validation, parsing, constrained decoding.

from SneppX_ALG import structured_output as so

errors = so.validate(obj, schema)
data = so.loads('```json\\n{"a": 1,}\\n```')
c = so.ChoiceConstrainer(["yes", "no"])
"""

from . import constrain, json_utils, schema
from .constrain import (
    NEG_INF,
    ChoiceConstrainer,
    Constrainer,
    mask_logits,
    top_token,
)
from .json_utils import extract_json, loads, repair_json, strip_code_fence
from .schema import is_valid, validate

__all__ = [
    "schema",
    "json_utils",
    "constrain",
    "validate",
    "is_valid",
    "extract_json",
    "repair_json",
    "strip_code_fence",
    "loads",
    "mask_logits",
    "top_token",
    "Constrainer",
    "ChoiceConstrainer",
    "NEG_INF",
]
