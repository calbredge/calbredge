"""Tests for protocol/calbredge_protocol/result_v2.py (docs/RESULT_SCHEMA.md)."""

import copy
import json
import types
import typing
from pathlib import Path

import pytest
from calbredge_protocol.models import BenchmarkResult
from calbredge_protocol.result_v2 import (
    Candidate,
    Capabilities,
    Metrics,
    ProtocolInfo,
    Quality,
    ResultV2,
    Target,
    Workload,
)
from pydantic import ValidationError

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parent / "fixtures" / "v2"
GOLDEN = FIXTURES / "golden_result_v2_0.json"
ITERATIVE = FIXTURES / "result_iterative_action_policy.json"

SECTIONS = {
    "workload": Workload,
    "candidate": Candidate,
    "target": Target,
    "protocol": ProtocolInfo,
    "capabilities": Capabilities,
    "metrics": Metrics,
    "quality": Quality,
}


def _golden() -> dict:
    return json.loads(GOLDEN.read_text())


def _allows_none(annotation) -> bool:
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        return type(None) in typing.get_args(annotation)
    return annotation is type(None)


def _required_non_null():
    for section, model in SECTIONS.items():
        for name, field in model.model_fields.items():
            if not _allows_none(field.annotation):
                yield section, name


# --- golden fixtures ----------------------------------------------------------


def test_golden_v2_0_result_always_loads():
    """Never edit golden_result_v2_0.json: every future reader must accept it."""
    result = ResultV2.model_validate_json(GOLDEN.read_text())
    assert result.schema_version == "2.0"
    assert result.target.power_mode == "maxn_super"


def test_iterative_action_policy_result_loads():
    result = ResultV2.model_validate(json.loads(ITERATIVE.read_text()))
    assert result.workload.work_unit == "decision"
    assert result.workload_metrics["N_denoise"] == 16


def test_round_trip_is_lossless():
    result = ResultV2.model_validate(_golden())
    assert ResultV2.model_validate(result.model_dump(mode="json")) == result


# --- required fields, unknown fields, strictness -------------------------------


@pytest.mark.parametrize(("section", "field"), list(_required_non_null()))
def test_required_fields_reject_null(section, field):
    data = _golden()
    data[section][field] = None
    with pytest.raises(ValidationError):
        ResultV2.model_validate(data)


@pytest.mark.parametrize("field", ["result_id", "timestamp", "workload_metrics"])
def test_required_top_level_fields_reject_null(field):
    data = _golden()
    data[field] = None
    with pytest.raises(ValidationError):
        ResultV2.model_validate(data)


@pytest.mark.parametrize("section", [None, *SECTIONS])
def test_unknown_fields_are_rejected(section):
    data = _golden()
    (data if section is None else data[section])["surprise"] = 1
    with pytest.raises(ValidationError, match="surprise"):
        ResultV2.model_validate(data)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("protocol", "protocol_version", 1.0),  # unquoted YAML/JSON number
        ("protocol", "batch_size", "1"),
        ("target", "clocks_pinned", "true"),
        ("metrics", "latency_p50_ms", "4.45"),
        ("target", "power_mode", True),
    ],
)
def test_no_type_coercion(section, field, value):
    data = _golden()
    data[section][field] = value
    with pytest.raises(ValidationError):
        ResultV2.model_validate(data)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("target", "power_mode", "MAXN_SUPER"),
        ("target", "hardware_family", "NVIDIA Jetson Orin Nano"),
        ("candidate", "runtime", "TensorRT-FP16"),
        ("candidate", "runtime_version", "unknown"),
        ("candidate", "runtime_version", "n/a"),
        ("candidate", "runtime_version", "N/A"),
        ("candidate", "model_hash", "A" * 64),
        ("candidate", "git_commit", "0123456"),
        ("protocol", "preprocessing", "None"),
    ],
)
def test_non_canonical_values_are_rejected(section, field, value):
    data = _golden()
    data[section][field] = value
    with pytest.raises(ValidationError):
        ResultV2.model_validate(data)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_metrics_are_rejected(value):
    data = _golden()
    data["metrics"]["latency_p99_ms"] = value
    with pytest.raises(ValidationError, match="finite"):
        ResultV2.model_validate(data)


# --- cross-field rules ----------------------------------------------------------


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["candidate"].update(runtime_family="rknn"), "does not match runtime"),
        (lambda d: d["candidate"].update(precision="int8"), "contradicts runtime label"),
        (lambda d: d["candidate"].update(git_dirty=None), "both set"),
        (lambda d: d["workload"].update(work_unit="decision"), "must be 'inference'"),
        (
            lambda d: d["workload"].update(
                workload_type="iterative_action_policy", work_unit="decision"
            ),
            "requires workload_params",
        ),
        (lambda d: d["target"].update(cpu_affinity_tier="custom"), "requires cpu_affinity.applied"),
        (lambda d: d["capabilities"].update(telemetry=["power", "power"]), "unique"),
        (lambda d: d["protocol"].update(measurement_iterations=0), "must be > 0"),
    ],
)
def test_cross_field_rules(mutate, message):
    data = _golden()
    mutate(data)
    with pytest.raises(ValidationError, match=message):
        ResultV2.model_validate(data)


def test_precision_is_free_when_runtime_has_no_suffix():
    data = _golden()
    data["candidate"].update(
        runtime="onnxruntime-cpu", runtime_family="onnxruntime", precision="int8", engine_hash=None
    )
    assert ResultV2.model_validate(data).candidate.precision == "int8"


# --- schema evolution (RESULT_SCHEMA.md §1) --------------------------------------


@pytest.mark.parametrize(
    ("version", "message"),
    [
        ("3.0", "not supported"),
        ("1.0", "not supported"),
        ("2.4", "Upgrade Calbredge"),
        ("2", "MAJOR.MINOR"),
        ("02.0", "MAJOR.MINOR"),
        (2.0, "MAJOR.MINOR"),
        (None, "MAJOR.MINOR"),
    ],
)
def test_unreadable_schema_versions_are_rejected(version, message):
    data = _golden()
    data["schema_version"] = version
    with pytest.raises(ValidationError, match=message):
        ResultV2.model_validate(data)


def test_newer_minor_fails_on_version_not_on_its_new_fields():
    data = _golden()
    data["schema_version"] = "2.4"
    data["metrics"]["some_field_added_in_2_4"] = 1.0
    with pytest.raises(ValidationError, match="Upgrade Calbredge") as excinfo:
        ResultV2.model_validate(data)
    assert "some_field_added_in_2_4" not in str(excinfo.value)


# --- v1 mapping (RESULT_SCHEMA.md §5) ---------------------------------------------


def test_to_v1_produces_the_upload_record():
    result = ResultV2.model_validate(_golden())
    v1 = result.to_v1()
    assert isinstance(v1, BenchmarkResult)
    assert v1.run_id == result.result_id
    assert v1.runtime == "tensorrt-fp16"
    assert v1.latency_p50_ms == result.metrics.latency_p50_ms
    assert v1.input_shape == [1, 3, 640, 640]
    assert v1.hardware_metrics is not None
    assert v1.hardware_metrics.power_mode == "maxn_super"
    assert v1.hardware_metrics.power_domain == "module_rail"
    # v2-only key never reaches the v1 upload payload.
    assert "device_tree_model" not in v1.system_metadata
    assert v1.system_metadata["hostname"] == "jetson-orin-nano-super"
    # Legacy status is the agent's to fill (classify_stability lives there).
    assert v1.status is None


def test_to_v1_without_telemetry_or_power_mode():
    data = copy.deepcopy(_golden())
    for key in list(data["metrics"]):
        if key.startswith(("watts_", "temperature_", "gpu_memory", "system_memory")) or key in (
            "energy_joules_window",
            "joules_per_inference",
            "thermal_throttle_detected",
        ):
            data["metrics"][key] = None
    data["target"].update(power_mode="not_applicable", power_domain=None, power_source=None)
    v1 = ResultV2.model_validate(data).to_v1()
    assert v1.hardware_metrics is None


@pytest.mark.parametrize("value", ["n/a", "N/A", " n/a ", "unknown", ""])
def test_a_placeholder_version_is_rejected_with_the_null_upstream_message(value):
    data = _golden()
    data["candidate"]["runtime_version"] = value
    with pytest.raises(ValidationError, match="record null upstream"):
        ResultV2.model_validate(data)


def test_a_real_version_is_accepted():
    data = _golden()
    data["candidate"]["runtime_version"] = "10.3.0"
    assert ResultV2.model_validate(data).candidate.runtime_version == "10.3.0"
