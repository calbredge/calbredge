"""The support matrix: tiers by runtime, precision and workload."""

import pytest
from calbredge_protocol.support_matrix import lookup, status_of

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("precision", "status"),
    [("fp16", "supported"), ("fp32", "supported"), ("int8", "results_only"), (None, "unsupported")],
)
def test_tensorrt_on_jetson_by_precision(precision, status):
    row = lookup("tensorrt", None, "linux", "jetson", precision=precision)
    assert status_of(row) == status


def test_int8_row_explains_the_noise_calibration():
    row = lookup("tensorrt", None, "linux", "jetson", precision="int8")
    assert row is not None
    assert "random noise" in row.reason


def test_iterative_policy_is_unsupported_even_where_its_runtime_is_supported():
    row = lookup(
        "tensorrt",
        None,
        "linux",
        "jetson",
        precision="fp16",
        workload_type="iterative_action_policy",
    )
    assert status_of(row) == "unsupported"
    assert row is not None
    assert "iterative_action_policy" in row.reason


def test_single_forward_pass_fp16_on_jetson_is_supported():
    row = lookup(
        "tensorrt", None, "linux", "jetson", precision="fp16", workload_type="single_forward_pass"
    )
    assert status_of(row) == "supported"


def test_support_matrix_rows():
    for precision in ("fp16", "fp32"):
        row = lookup("tensorrt", None, "linux", "jetson", precision=precision)
        assert row is not None
        assert row.status == "supported"
    # A precision the matrix doesn't name, or doesn't know, is never supported.
    assert lookup("tensorrt", None, "linux", "jetson") is None
    assert lookup("onnxruntime", "cuda", "linux", "x86_64").status == "results_only"  # type: ignore[union-attr]
    assert lookup("tflite", None, "android", "phone").status == "unsupported"  # type: ignore[union-attr]
    assert lookup("tensorrt", None, "windows", "x86_64") is None
