"""Tests for protocol/calbredge_protocol/canonical.py: canonical identifiers and RFC 8785 JSON.

Every expected serialization below was cross-checked against V8's
JSON.stringify (the ECMAScript algorithm RFC 8785 adopts), including 20,000
random doubles, when this module was written.
"""

import struct

import pytest
from calbredge_protocol.canonical import (
    canonical_identifier,
    canonical_json,
    is_canonical_identifier,
    runtime_family,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("MAXN SUPER", "maxn_super"),
        ("maxn_super", "maxn_super"),
        ("15W", "15w"),
        ("  --Foo__Bar--  ", "foo_bar"),
        # /proc/device-tree/model is NUL-terminated.
        (
            "NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super\x00",
            "nvidia_jetson_orin_nano_engineering_reference_developer_kit_super",
        ),
    ],
)
def test_canonical_identifier(raw, expected):
    assert canonical_identifier(raw) == expected
    assert is_canonical_identifier(expected)


@pytest.mark.parametrize("value", ["MAXN_SUPER", "maxn super", "", "_x", "a-b"])
def test_non_canonical_identifiers_are_detected(value):
    assert not is_canonical_identifier(value)


@pytest.mark.parametrize(
    ("label", "family"),
    [("tensorrt-fp16", "tensorrt"), ("onnxruntime-cpu", "onnxruntime"), ("rknn", "rknn")],
)
def test_runtime_family(label, family):
    assert runtime_family(label) == family


# RFC 8785 Appendix B: IEEE-754 bit pattern -> required serialization.
APPENDIX_B = [
    ("0000000000000000", "0"),
    ("8000000000000000", "0"),
    ("0000000000000001", "5e-324"),
    ("8000000000000001", "-5e-324"),
    ("7fefffffffffffff", "1.7976931348623157e+308"),
    ("ffefffffffffffff", "-1.7976931348623157e+308"),
    ("4340000000000000", "9007199254740992"),
    ("c340000000000000", "-9007199254740992"),
    ("4430000000000000", "295147905179352830000"),
    ("44b52d02c7e14af5", "9.999999999999997e+22"),
    ("44b52d02c7e14af6", "1e+23"),
    ("44b52d02c7e14af7", "1.0000000000000001e+23"),
    ("444b1ae4d6e2ef4e", "999999999999999700000"),
    ("444b1ae4d6e2ef4f", "999999999999999900000"),
    ("444b1ae4d6e2ef50", "1e+21"),
    ("3eb0c6f7a0b5ed8c", "9.999999999999997e-7"),
    ("3eb0c6f7a0b5ed8d", "0.000001"),
    ("41b3de4355555553", "333333333.3333332"),
    ("41b3de4355555554", "333333333.33333325"),
    ("41b3de4355555555", "333333333.3333333"),
    ("41b3de4355555556", "333333333.3333334"),
    ("41b3de4355555557", "333333333.33333343"),
    ("becbf647612f3696", "-0.0000033333333333333333"),
    ("43143ff3c1cb0959", "1424953923781206.2"),
]


@pytest.mark.parametrize(("bits", "expected"), APPENDIX_B)
def test_rfc8785_number_vectors(bits, expected):
    value = struct.unpack(">d", bytes.fromhex(bits))[0]
    assert canonical_json(value) == expected.encode()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (4.4542, "4.4542"),
        (100.0, "100"),
        (123.0, "123"),
        (0.1 + 0.2, "0.30000000000000004"),
        (7, "7"),
    ],
)
def test_numbers_we_actually_store(value, expected):
    assert canonical_json(value) == expected.encode()


def test_rfc8785_key_sorting_by_utf16_code_units():
    # RFC 8785 §3.2.3. U+1F600 sorts before U+FB33 in UTF-16 (a surrogate
    # pair starting 0xD83D), though after it by code point.
    obj = {
        "€": "Euro Sign",
        "\r": "Carriage Return",
        "דּ": "Hebrew Letter Dalet With Dagesh",
        "1": "One",
        "\U0001f600": "Emoji: Grinning Face",
        "\u0080": "Control",
        "ö": "Latin Small Letter O With Diaeresis",
    }
    keys_in_output_order = ["\r", "1", "\u0080", "ö", "€", "\U0001f600", "דּ"]
    expected = (
        "{"
        + ",".join(
            canonical_json(k).decode() + ":" + canonical_json(obj[k]).decode()
            for k in keys_in_output_order
        )
        + "}"
    )
    assert canonical_json(obj) == expected.encode("utf-8")


def test_rfc8785_example_document():
    # RFC 8785 §3.2.2 input and its canonical output.
    string = "".join(
        map(chr, [0x20AC, 0x24, 0x0F, 0x0A, 0x41, 0x27, 0x42, 0x22, 0x5C, 0x5C, 0x22, 0x2F])
    )
    doc = {
        "numbers": [333333333.33333329, 1e30, 4.50, 2e-3, 1e-27],
        "string": string,
        "literals": [None, True, False],
    }
    expected = (
        '{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],'
        '"string":"€$\\u000f\\nA\'B\\"\\\\\\\\\\"/"}'
    )
    assert canonical_json(doc) == expected.encode("utf-8")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_are_rejected(value):
    with pytest.raises(ValueError, match="no JSON representation"):
        canonical_json({"x": [value]})


def test_integers_beyond_double_precision_are_rejected():
    assert canonical_json(2**53) == b"9007199254740992"
    with pytest.raises(ValueError, match="not exactly representable"):
        canonical_json(2**53 + 1)


@pytest.mark.parametrize("value", [{1: "int key"}, {"s": {1, 2}}, {"b": b"bytes"}])
def test_non_json_data_is_rejected(value):
    with pytest.raises(TypeError):
        canonical_json(value)
