"""conversion_input in the result schema (2.3): how a converted model's inputs were converted,
recorded in ``candidate.engine_build`` by the RKNN adapter.
"""

import pytest
from calbredge_protocol.result_v2 import SCHEMA_VERSION, ConversionInput, EngineBuild
from calbredge_protocol.support_matrix import lookup, status_of
from pydantic import ValidationError

pytestmark = pytest.mark.unit

CONVERSION = {
    "feed_dtype": "uint8",
    "mean_values": [[0, 0, 0]],
    "std_values": [[255, 255, 255]],
    "inputs": [{"source_layout": "nchw", "converted_shape": [1, 3, 224, 224]}],
    "graph_edits": [],
}
ENGINE_BUILD = {
    "build_key": "c" * 64,
    "built_from_model_hash": "a" * 64,
    "tensorrt_version": None,
    "calibration_digest": None,
}


def test_the_schema_is_2_3():
    assert SCHEMA_VERSION == "2.3"


def test_an_engine_build_without_conversion_input_is_valid_as_tensorrt_writes_it():
    build = EngineBuild.model_validate({**ENGINE_BUILD, "tensorrt_version": "10.3.0"})
    assert build.conversion_input is None


def test_an_engine_build_with_conversion_input_round_trips():
    build = EngineBuild.model_validate({**ENGINE_BUILD, "conversion_input": CONVERSION})
    assert build.conversion_input is not None
    assert build.conversion_input.model_dump() == CONVERSION


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"graph_edits": ["b", "a"]}, "sorted and without duplicates"),
        ({"graph_edits": ["a", "a"]}, "sorted and without duplicates"),
        ({"inputs": [{"source_layout": "chw", "converted_shape": [1]}]}, "source_layout"),
        ({"inputs": [{"source_layout": "nchw", "converted_shape": [1, 0, 2]}]}, "positive"),
        ({"feed_dtype": None}, "feed_dtype"),
        ({"extra": 1}, "extra"),
    ],
)
def test_malformed_conversion_input_is_rejected(changes, message):
    with pytest.raises(ValidationError, match=message):
        ConversionInput.model_validate({**CONVERSION, **changes})


def test_layout_and_shape_may_be_unknown_but_not_missing():
    unknown = {**CONVERSION, "inputs": [{"source_layout": None, "converted_shape": None}]}
    assert ConversionInput.model_validate(unknown).inputs[0].source_layout is None
    with pytest.raises(ValidationError):
        ConversionInput.model_validate({**CONVERSION, "inputs": [{"source_layout": "nchw"}]})


def test_rknn_fp16_on_the_rk3588_is_supported_and_int8_is_results_only():
    fp16 = lookup("rknn", None, "linux", "rk3588", precision="fp16")
    assert status_of(fp16) == "supported"
    assert fp16 is not None
    assert "build record" in fp16.reason
    assert "informational for FP16" in fp16.reason
    int8 = lookup("rknn", None, "linux", "rk3588", precision="int8")
    assert status_of(int8) == "results_only"
    assert int8 is not None
    assert "recorded but not gated" in int8.reason
    # An RKNN result whose precision is unknown has no row, so it is unsupported.
    assert lookup("rknn", None, "linux", "rk3588", precision=None) is None
    assert status_of(lookup("rknn", None, "linux", "jetson", precision="fp16")) == "unsupported"
