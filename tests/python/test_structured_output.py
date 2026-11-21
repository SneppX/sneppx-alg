from SneppX_ALG import structured_output as so

SCHEMA = {
    "type": "object",
    "required": ["name", "age"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}


def test_schema_valid():
    obj = {"name": "Ammar", "age": 30, "tags": ["a", "b"]}
    assert so.validate(obj, SCHEMA) == []
    assert so.is_valid(obj, SCHEMA)


def test_schema_errors():
    assert so.validate({"age": 30}, SCHEMA)
    assert so.validate({"name": "x", "age": -1}, SCHEMA)
    assert so.validate({"name": "x", "age": 30, "extra": 1}, SCHEMA)
    assert so.validate({"name": "", "age": 30}, SCHEMA)
    assert so.validate({"name": "x", "age": 30, "tags": [1]}, SCHEMA)


def test_schema_enum_and_logical():
    schema = {"enum": [1, 2, 3]}
    assert so.is_valid(2, schema)
    assert not so.is_valid(4, schema)
    schema2 = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert so.is_valid("x", schema2)
    assert so.is_valid(5, schema2)
    assert not so.is_valid(1.5, schema2)


def test_strip_code_fence():
    text = 'Here you go:\n```json\n{"a": 1}\n```\n'
    assert so.strip_code_fence(text) == '{"a": 1}'


def test_extract_json():
    text = 'prefix {"a": [1, 2, {"b": 3}]} suffix'
    assert so.extract_json(text) == '{"a": [1, 2, {"b": 3}]}'
    assert so.extract_json("no json here") is None
    assert so.extract_json("x [1, 2] y", prefer="array") == "[1, 2]"


def test_loads_with_repair():
    assert so.loads('```json\n{"a": 1, "b": 2,}\n```') == {"a": 1, "b": 2}
    assert so.loads("garbage text [1, 2, 3] trailing") == [1, 2, 3]


def test_mask_logits():
    import numpy as np

    logits = np.array([1.0, 5.0, 3.0])
    masked = so.mask_logits(logits, [0, 2])
    assert masked[1] <= so.NEG_INF
    assert so.top_token(logits, [0, 2]) == 2


def test_choice_constrainer():
    c = so.ChoiceConstrainer(["yes", "yeah", "no"])
    assert c.allowed() == {"y", "n"}
    c.advance("y")
    assert c.allowed() == {"e"}
    c.advance("e")
    assert c.allowed() == {"s", "a"}
    c.advance("s")
    assert c.is_complete()
