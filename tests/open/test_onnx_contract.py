"""Tests for agent/calbredge/onnx_contract.py.

docs/RESULT_SCHEMA.md §3.3.1 and §3.5.
"""

import json
from pathlib import Path

import pytest
from calbredge import onnx_contract
from calbredge.onnx_contract import (
    DERIVATION_VERSION,
    bind_input_shape,
    derive_graph_contract,
    load_graph_contract,
    sidecar_path,
)
from calbredge_protocol.graph_variant import load_graph

from tests.open.fake_hosts import conv_model, if_model, nms_model, qdq_model, topk_model

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parent / "fixtures"


def _contract(path: Path):
    return derive_graph_contract(load_graph(path))


# --- graph_kind --------------------------------------------------------------------


def test_conv_is_fixed_compute(tmp_path):
    c = _contract(conv_model(tmp_path / "m.onnx"))
    assert c.graph_kind == "fixed_compute"
    assert c.graph_kind_triggers == []


def test_nms_is_data_dependent(tmp_path):
    c = _contract(nms_model(tmp_path / "m.onnx"))
    assert (c.graph_kind, c.graph_kind_triggers) == ("data_dependent", ["NonMaxSuppression"])


def test_if_is_data_dependent(tmp_path):
    assert _contract(if_model(tmp_path / "m.onnx")).graph_kind_triggers == ["If"]


def test_ops_inside_subgraphs_are_found(tmp_path):
    assert _contract(if_model(tmp_path / "m.onnx", nested_nms=True)).graph_kind_triggers == [
        "If",
        "NonZero",
    ]


def test_topk_with_constant_k_is_fixed(tmp_path):
    assert _contract(topk_model(tmp_path / "m.onnx", constant_k=True)).graph_kind == "fixed_compute"


def test_topk_with_runtime_k_is_data_dependent(tmp_path):
    c = _contract(topk_model(tmp_path / "m.onnx", constant_k=False))
    assert c.graph_kind_triggers == ["TopK(non-constant K)"]


@pytest.mark.parametrize(
    "name", ["yolov8n.onnx", "yolov8n_seg.onnx", "rtdetr_s.onnx", "efficientdet_lite0.onnx"]
)
def test_real_fixture_models_are_fixed_compute(name):
    # Real exports are gitignored (tests/open/fixtures/README.md): this runs where
    # they are present locally and skips in CI.
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    assert _contract(path).graph_kind == "fixed_compute"


# --- precision -------------------------------------------------------------------------


def test_qdq_is_int8_despite_float_inputs_and_float_weights(tmp_path):
    c = _contract(qdq_model(tmp_path / "m.onnx"))
    assert c.io_signature["inputs"][0]["dtype"] == "float32"
    assert c.graph_precision == "int8"


def test_fp16_weights(tmp_path):
    assert _contract(conv_model(tmp_path / "m.onnx", fp16_weights=True)).graph_precision == "fp16"


def test_fp32_weights(tmp_path):
    assert _contract(conv_model(tmp_path / "m.onnx")).graph_precision == "fp32"


def test_no_weights_gives_no_precision(tmp_path):
    assert _contract(nms_model(tmp_path / "m.onnx")).graph_precision is None


def test_constant_node_weights_count(tmp_path):
    # if_model's only tensors are Constant nodes inside its branches.
    assert _contract(if_model(tmp_path / "m.onnx")).graph_precision == "fp32"


# --- io_signature / input_shape --------------------------------------------------------------


def test_io_signature_excludes_initializers_and_keeps_symbolic_dims(tmp_path):
    c = _contract(conv_model(tmp_path / "m.onnx"))
    assert c.io_signature == {
        "inputs": [{"name": "x", "dtype": "float32", "shape": ["N", 3, 8, 8]}],
        "outputs": [{"name": "y", "dtype": "float32", "shape": ["N", 4, 6, 6]}],
    }
    assert c.primary_input_dims == ["N", 3, 8, 8]


def test_bind_input_shape():
    assert bind_input_shape(["N", 3, 8, 8], [2, 3, 8, 8]) == [2, 3, 8, 8]
    # Static dims come from the graph, not the protocol.
    assert bind_input_shape([1, 3, 640, 640], [1, 3, 224, 224]) == [1, 3, 640, 640]
    with pytest.raises(ValueError, match="dimensions"):
        bind_input_shape(["N", 3, 8, 8], [1, 3, 8])


# --- sidecar cache -------------------------------------------------------------------------------


def _count_loads(monkeypatch):
    calls = []
    real = onnx_contract.load_graph

    def counting(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(onnx_contract, "load_graph", counting)
    return calls


def test_parsed_once_then_served_from_sidecar(tmp_path, monkeypatch):
    model = conv_model(tmp_path / "m.onnx")
    calls = _count_loads(monkeypatch)
    first, status1 = load_graph_contract(model, "a" * 64)
    second, status2 = load_graph_contract(model, "a" * 64)
    assert status1 == f"written:{sidecar_path(model)}"
    assert status2 == f"hit:{sidecar_path(model)}"
    assert first == second
    assert len(calls) == 1


@pytest.mark.parametrize(
    "tamper",
    [
        lambda p: p.update(model_hash="b" * 64),
        lambda p: p.update(derivation_version=DERIVATION_VERSION + 1),
        lambda p: p["contract"].pop("graph_kind"),
    ],
)
def test_stale_or_foreign_cache_is_ignored(tmp_path, monkeypatch, tamper):
    model = conv_model(tmp_path / "m.onnx")
    load_graph_contract(model, "a" * 64)
    payload = json.loads(sidecar_path(model).read_text())
    tamper(payload)
    sidecar_path(model).write_text(json.dumps(payload))
    calls = _count_loads(monkeypatch)
    _, status = load_graph_contract(model, "a" * 64)
    assert status.startswith("written:")
    assert len(calls) == 1


def test_corrupt_cache_is_ignored(tmp_path):
    model = conv_model(tmp_path / "m.onnx")
    sidecar_path(model).write_text("{not json")
    contract, status = load_graph_contract(model, "a" * 64)
    assert contract.graph_kind == "fixed_compute"
    assert status.startswith("written:")


def test_read_only_model_dir_falls_back_to_cache_dir(tmp_path, monkeypatch):
    (tmp_path / "models").mkdir()
    model = conv_model(tmp_path / "models" / "m.onnx")
    fallback = tmp_path / "cache"
    real_write = onnx_contract._write_atomic
    monkeypatch.setattr(
        onnx_contract,
        "_write_atomic",
        lambda path, payload: False if path.parent == model.parent else real_write(path, payload),
    )
    _, status = load_graph_contract(model, "c" * 64, fallback)
    assert status == f"written:{fallback / ('c' * 64 + '.json')}"
    _, status = load_graph_contract(model, "c" * 64, fallback)
    assert status.startswith("hit:")


def test_unwritable_everywhere_still_returns_the_contract(tmp_path, monkeypatch):
    model = conv_model(tmp_path / "m.onnx")
    monkeypatch.setattr(onnx_contract, "_write_atomic", lambda path, payload: False)
    contract, status = load_graph_contract(model, "a" * 64, tmp_path / "cache")
    assert status == "unwritable"
    assert contract.graph_kind == "fixed_compute"
