"""Tests for `calbredge inspect` (agent/calbredge/inspect_v2.py and the CLI command)."""

import json

import pytest
from calbredge import host_v2, result_builder
from calbredge.cli import main
from calbredge.inspect_v2 import inspect_model, predict_affinity_record
from calbredge_protocol.derive_v2 import cpu_affinity_tier
from calbredge_protocol.models import BenchmarkProtocol
from calbredge_protocol.result_v2 import SCHEMA_VERSION
from click.testing import CliRunner

from tests.open.fake_hosts import ORIN_FAMILY, build_jetson_orin, build_x86, conv_model, qdq_model

pytestmark = pytest.mark.unit


@pytest.fixture
def protocol() -> BenchmarkProtocol:
    return BenchmarkProtocol(protocol_version="1.0", input_shape=[1, 3, 8, 8])


@pytest.fixture(autouse=True)
def _deterministic_host(monkeypatch):
    monkeypatch.setattr(host_v2, "_get_nvpmodel_mode", lambda: "maxn_super")
    monkeypatch.setattr(result_builder, "runtime_version", lambda family: "10.3.0")


def _inspect(tmp_path, protocol, *, pinned, runtime="tensorrt-fp16", model=None):
    root = build_jetson_orin(tmp_path / "root", pinned=pinned)
    model = model or conv_model(tmp_path / "yolo.onnx")
    return inspect_model(
        model, protocol, runtime=runtime, cwd=tmp_path, root=root, cache_dir=tmp_path / "cache"
    )


@pytest.mark.parametrize("pinned", [True, False])
def test_jetson_clocks_pinned_follows_the_host(tmp_path, protocol, pinned):
    report = _inspect(tmp_path, protocol, pinned=pinned)
    target = report["target"]
    assert target["clocks_pinned"] is pinned
    assert target["hardware_family"] == ORIN_FAMILY
    assert target["hardware_family_source"] == "device_tree"
    assert target["power_mode"] == "maxn_super"
    assert report["protocol"]["graph_kind"] == "fixed_compute"
    assert report["v2_validation"] == {"candidate": "ok", "target": "ok", "protocol": "ok"}


@pytest.mark.parametrize(
    ("runtime", "precision"),
    [("tensorrt-fp16", "fp16"), ("tensorrt", "fp32"), ("tensorrt-int8", "int8")],
)
def test_precision_follows_the_runtime_label(tmp_path, protocol, runtime, precision):
    candidate = _inspect(tmp_path, protocol, pinned=True, runtime=runtime)["candidate"]
    assert candidate["precision"] == precision
    assert candidate["runtime"] == f"tensorrt-{precision}"


def test_suffixless_runtime_takes_precision_from_the_graph(tmp_path):
    proto = BenchmarkProtocol(protocol_version="1.0", input_shape=[1, 4])
    report = _inspect(
        tmp_path, proto, pinned=True, runtime="onnxruntime", model=qdq_model(tmp_path / "q.onnx")
    )
    assert report["candidate"]["precision"] == "int8"
    assert report["inspect"]["runtime_predicted"] is True


def test_dynamic_dims_bound_from_protocol_and_measurement_fields_not_collected(tmp_path):
    proto = BenchmarkProtocol(protocol_version="1.0", input_shape=[4, 3, 8, 8])
    report = _inspect(tmp_path, proto, pinned=True)
    assert report["protocol"]["input_shape"] == [4, 3, 8, 8]
    assert report["protocol"]["io_signature"]["inputs"][0]["shape"] == ["N", 3, 8, 8]
    for field in ("power_domain", "power_source", "telemetry_measurement_boundary"):
        assert report["target"][field] is None
    assert report["candidate"]["engine_hash"] is None
    assert set(report["inspect"]["not_collected"]) >= {"engine_hash", "power_domain"}


def test_system_metadata_gains_device_tree_model(tmp_path, protocol):
    meta = _inspect(tmp_path, protocol, pinned=True)["target"]["system_metadata"]
    assert meta["device_tree_model"].startswith("NVIDIA Jetson Orin Nano")


def test_second_inspect_hits_the_contract_cache(tmp_path, protocol):
    model = conv_model(tmp_path / "yolo.onnx")
    first = _inspect(tmp_path, protocol, pinned=True, model=model)
    second = _inspect(tmp_path, protocol, pinned=True, model=model)
    assert first["inspect"]["contract_cache"].startswith("written:")
    assert second["inspect"]["contract_cache"].startswith("hit:")


def test_validation_reports_problems_instead_of_failing(tmp_path, protocol, monkeypatch):
    monkeypatch.setattr(result_builder, "runtime_version", lambda family: None)
    report = _inspect(tmp_path, protocol, pinned=True)
    assert report["v2_validation"]["candidate"] != "ok"
    assert any("runtime_version" in e for e in report["v2_validation"]["candidate"])


@pytest.mark.parametrize("spec", ["big", "none", "0,1"])
def test_predicted_affinity_uses_the_run_record_shapes(tmp_path, spec):
    record = predict_affinity_record(spec, build_x86(tmp_path))
    assert cpu_affinity_tier(record) is not None


def test_cli_prints_exactly_one_json_document(tmp_path):
    model = conv_model(tmp_path / "m.onnx", dynamic_batch=False)
    protocol_file = tmp_path / "p.yaml"
    protocol_file.write_text('protocol_version: "1.0"\ninput_shape: [1, 3, 8, 8]\n')
    result = CliRunner().invoke(
        main,
        [
            "inspect",
            "--model",
            str(model),
            "--runtime",
            "onnxruntime",
            "--protocol",
            str(protocol_file),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["protocol"]["graph_kind"] == "fixed_compute"


def test_cli_rank_mismatch_is_a_clean_error(tmp_path):
    model = conv_model(tmp_path / "m.onnx")
    protocol_file = tmp_path / "p.yaml"
    protocol_file.write_text('protocol_version: "1.0"\ninput_shape: [1, 3, 8]\n')
    result = CliRunner().invoke(
        main, ["inspect", "--model", str(model), "--protocol", str(protocol_file)]
    )
    assert result.exit_code == 1
    assert "dimensions" in result.output
