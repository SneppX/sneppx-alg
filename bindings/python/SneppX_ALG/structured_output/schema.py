"""A pragmatic JSON-Schema (draft subset) validator (pure Python).

Supports: type, enum, const, minimum/maximum (exclusive), minLength/maxLength,
pattern, minItems/maxItems, uniqueItems, items, properties, required,
additionalProperties, minProperties/maxProperties, anyOf/allOf/oneOf/not.
"""

import re
from typing import Any, Dict, List, Optional

_TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
}


def _type_of(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _type_matches(value: Any, expected: str) -> bool:
    actual = _type_of(value)
    if expected == "number":
        return actual in ("number", "integer")
    if expected == "integer":
        return actual == "integer"
    return actual == expected


def validate(instance: Any, schema: Dict[str, Any], path: str = "$") -> List[str]:
    """Return a list of human-readable validation errors (empty if valid)."""
    errors: List[str] = []

    if not isinstance(schema, dict):
        return [f"{path}: schema is not an object"]

    if "type" in schema:
        expected = schema["type"]
        types = expected if isinstance(expected, list) else [expected]
        if not any(_type_matches(instance, t) for t in types):
            errors.append(f"{path}: expected type {expected}, got {_type_of(instance)}")
            return errors

    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: value must equal {schema['const']!r}")

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: value not in enum {schema['enum']!r}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} > maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: value must be > {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
            errors.append(f"{path}: value must be < {schema['exclusiveMaximum']}")

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(
                f"{path}: string shorter than minLength {schema['minLength']}"
            )
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}: string longer than maxLength {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            errors.append(
                f"{path}: string does not match pattern {schema['pattern']!r}"
            )

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: too few items (< {schema['minItems']})")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: too many items (> {schema['maxItems']})")
        if schema.get("uniqueItems"):
            seen = []
            for item in instance:
                if item in seen:
                    errors.append(f"{path}: duplicate item {item!r}")
                seen.append(item)
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(instance):
                errors.extend(validate(item, item_schema, f"{path}[{i}]"))

    if isinstance(instance, dict):
        if "minProperties" in schema and len(instance) < schema["minProperties"]:
            errors.append(f"{path}: too few properties")
        if "maxProperties" in schema and len(instance) > schema["maxProperties"]:
            errors.append(f"{path}: too many properties")
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required property {key!r}")
        props = schema.get("properties", {})
        for key, subschema in props.items():
            if key in instance:
                errors.extend(validate(instance[key], subschema, f"{path}.{key}"))
        addl = schema.get("additionalProperties", True)
        if addl is False:
            for key in instance:
                if key not in props:
                    errors.append(f"{path}: unexpected property {key!r}")
        elif isinstance(addl, dict):
            for key in instance:
                if key not in props:
                    errors.extend(validate(instance[key], addl, f"{path}.{key}"))

    for key in ("allOf", "anyOf", "oneOf"):
        if key in schema:
            subschemas = schema[key]
            sub_errors = [validate(instance, s, path) for s in subschemas]
            if key == "allOf":
                for errs in sub_errors:
                    errors.extend(errs)
            elif key == "anyOf":
                if all(errs for errs in sub_errors):
                    errors.append(f"{path}: does not match anyOf")
            elif key == "oneOf":
                matched = sum(1 for errs in sub_errors if not errs)
                if matched != 1:
                    errors.append(
                        f"{path}: must match exactly one of oneOf (matched {matched})"
                    )

    if "not" in schema and not validate(instance, schema["not"], path):
        errors.append(f"{path}: value matches forbidden schema")

    return errors


def is_valid(instance: Any, schema: Dict[str, Any]) -> bool:
    """True if ``instance`` satisfies ``schema``."""
    return not validate(instance, schema)
