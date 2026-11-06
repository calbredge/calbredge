"""Content-addressed TensorRT engine cache (protocol/calbredge_protocol/engine_cache.py).

The bug this guards: engines used to be cached by model *path*, so replacing
the .onnx at the same path (a retrained model, same I/O, new weights) reused
the old engine silently: a false PASS. Every adapter test here runs the real
TensorRTAdapter.load_model against the fake tensorrt/pycuda modules from
tests/open/test_tensorrt_adapter.py.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import calbredge_protocol.engine_cache as engine_cache
import calbredge_protocol.tensorrt_adapter as trt_adapter
import pytest
from calbredge.result_builder import engine_io_mismatches
from calbredge_protocol.engine_cache import (
    RECORD_SUFFIX,
    EngineCacheError,
    compute_build_key,
    resolve_input_profile,
    verify_record,
)
from calbredge_protocol.result_v2 import ResultV2
from calbredge_protocol.tensorrt_adapter import TensorRTAdapter

from tests.open.test_tensorrt_adapter import (
    _engines,
    _make_builder_mock,
    _make_cuda_mock,
    _make_engine_mock,
    _make_trt_mock,
    _patch_modules,
    _write_onnx,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def trt_env():
    trt = _make_trt_mock()
    setattr(trt, "__version__", "10.3.0")  # fake module: attribute set dynamically
    cuda = _make_cuda_mock()
    engine, _ = _make_engine_mock(trt)
    builder, _, _, runtime, network = _make_builder_mock(trt, engine)
    builds = iter(f"engine-build-{i}".encode() for i in range(1, 100))
    builder.build_serialized_network.side_effect = lambda *a, **k: next(builds)
    with (
        _patch_modules(trt, cuda),
        patch("calbredge_protocol.tensorrt_adapter._get_nvpmodel_mode", return_value="maxn_super"),
    ):
        yield SimpleNamespace(trt=trt, builder=builder, runtime=runtime, network=network)


def _load(path, precision="fp16", **kwargs):
    adapter = TensorRTAdapter(precision=precision)
    adapter.load_model(path, **kwargs)
    return adapter


def _built(env) -> int:
    return env.builder.build_serialized_network.call_count


def _deserialized(env) -> list[bytes]:
    return [c.args[0] for c in env.runtime.deserialize_cuda_engine.call_args_list]


def _record_path(engine_path):
    return engine_path.with_name(engine_path.name[: -len(".engine")] + RECORD_SUFFIX)


# --- reuse and rebuild ---------------------------------------------------------------


def test_same_model_reuses_its_engine(tmp_path, trt_env):
    model = _write_onnx(tmp_path / "m.onnx")
    _load(model)
    _load(model)
    assert _built(trt_env) == 1
    assert _deserialized(trt_env) == [b"engine-build-1", b"engine-build-1"]


def test_retrained_model_same_io_is_rebuilt_never_reused(tmp_path, trt_env):
    """The false-PASS case: new weights, same inputs/outputs, same path."""
    model = _write_onnx(tmp_path / "m.onnx", fill=0.0)
    first = _load(model)
    _write_onnx(model, fill=1.0)  # replace the model in place
    second = _load(model)
    assert _built(trt_env) == 2
    # The first engine is never loaded for the new model.
    assert _deserialized(trt_env) == [b"engine-build-1", b"engine-build-2"]
    assert len(_engines(model, "fp16", "maxn_super")) == 2
    assert (
        first._engine_build["built_from_model_hash"]
        != second._engine_build["built_from_model_hash"]
    )
    assert second._engine_build["built_from_model_hash"] == trt_adapter._sha256_file(model)


def test_different_calibration_rebuilds(tmp_path, trt_env, monkeypatch):
    model = _write_onnx(tmp_path / "m.onnx")
    with patch("calbredge_protocol.tensorrt_adapter._make_calibrator"):
        a = _load(model, precision="int8")
        monkeypatch.setattr(
            trt_adapter, "INT8_CALIBRATION", {**trt_adapter.INT8_CALIBRATION, "n_calib": 1000}
        )
        b = _load(model, precision="int8")
    assert _built(trt_env) == 2
    assert a._engine_build["calibration_digest"] != b._engine_build["calibration_digest"]


def test_different_input_profile_rebuilds(tmp_path, trt_env):
    model = _write_onnx(tmp_path / "dyn.onnx", shape=("N", 3, 640, 640))
    trt_env.network.get_input.return_value.shape = [-1, 3, 640, 640]
    _load(model, input_shape=[1, 3, 640, 640])
    _load(model, input_shape=[2, 3, 640, 640])
    _load(model, input_shape=[1, 3, 640, 640])
    assert _built(trt_env) == 2
    assert len(_engines(model, "fp16", "maxn_super")) == 2


def test_tensorrt_version_change_rebuilds_without_manual_deletion(tmp_path, trt_env):
    model = _write_onnx(tmp_path / "m.onnx")
    _load(model)
    trt_env.trt.__version__ = "10.4.0"
    upgraded = _load(model)
    assert _built(trt_env) == 2
    assert upgraded._engine_build["tensorrt_version"] == "10.4.0"


def test_builder_config_version_bump_rebuilds(tmp_path, trt_env, monkeypatch):
    model = _write_onnx(tmp_path / "m.onnx")
    _load(model)
    monkeypatch.setattr(
        trt_adapter, "TRT_BUILDER_CONFIG_VERSION", trt_adapter.TRT_BUILDER_CONFIG_VERSION + 1
    )
    _load(model)
    assert _built(trt_env) == 2


# --- refusal ---------------------------------------------------------------------------


def test_edited_record_is_refused(tmp_path, trt_env):
    model = _write_onnx(tmp_path / "m.onnx")
    _load(model)
    (engine,) = _engines(model, "fp16", "maxn_super")
    record = json.loads(_record_path(engine).read_text())
    record["inputs"]["model_hash"] = "0" * 64  # no longer hashes to build_key
    _record_path(engine).write_text(json.dumps(record))
    with pytest.raises(EngineCacheError, match="do not hash to its build_key"):
        _load(model)


def test_record_for_another_key_is_refused(tmp_path, trt_env):
    model = _write_onnx(tmp_path / "m.onnx")
    _load(model)
    (engine,) = _engines(model, "fp16", "maxn_super")
    record = json.loads(_record_path(engine).read_text())
    record["inputs"]["precision"] = "fp32"
    record["build_key"] = compute_build_key(record["inputs"])  # self-consistent, wrong engine
    _record_path(engine).write_text(json.dumps(record))
    with pytest.raises(EngineCacheError, match="refusing to load"):
        _load(model)


def test_network_profile_disagreeing_with_the_key_is_not_cached(tmp_path, trt_env):
    model = _write_onnx(tmp_path / "m.onnx")  # declares [1, 3, 640, 640]
    trt_env.network.get_input.return_value.shape = [1, 3, 320, 320]
    with pytest.raises(EngineCacheError, match="does not match the build key"):
        _load(model)
    assert _engines(model, "fp16", "maxn_super") == []


# --- legacy engines, fallback dir, crash safety ------------------------------------------


def test_legacy_unstamped_engine_is_ignored_and_untouched(tmp_path, trt_env):
    model = _write_onnx(tmp_path / "m.onnx")
    legacy = tmp_path / "m.onnx.trt.fp16.maxn_super"
    legacy.write_bytes(b"legacy engine used by another tool")
    before = (legacy.read_bytes(), legacy.stat().st_mtime_ns)
    _load(model)
    assert _built(trt_env) == 1
    assert b"legacy engine used by another tool" not in _deserialized(trt_env)
    assert (legacy.read_bytes(), legacy.stat().st_mtime_ns) == before


def test_read_only_model_dir_uses_the_fallback_cache(tmp_path, trt_env):
    models = tmp_path / "models"
    models.mkdir()
    model = _write_onnx(models / "m.onnx")
    fallback = tmp_path / "engine-cache"
    models.chmod(0o500)
    try:
        first = _load(model, engine_cache_fallback_dir=fallback)
        _load(model, engine_cache_fallback_dir=fallback)
    finally:
        models.chmod(0o700)
    assert first._engine_path.parent == fallback
    assert _built(trt_env) == 1
    assert list(models.glob("*.engine")) == []


def test_engine_without_its_record_is_rebuilt(tmp_path, trt_env):
    model = _write_onnx(tmp_path / "m.onnx")
    _load(model)
    (engine,) = _engines(model, "fp16", "maxn_super")
    _record_path(engine).unlink()
    _load(model)
    assert _built(trt_env) == 2


def test_interrupted_write_leaves_no_engine(tmp_path, trt_env, monkeypatch):
    model = _write_onnx(tmp_path / "m.onnx")

    def crash(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(engine_cache.os, "replace", crash)
    with pytest.raises(EngineCacheError, match="could not write"):
        _load(model, engine_cache_fallback_dir=tmp_path / "fb")
    leftovers = [p.name for p in tmp_path.rglob("*") if p.is_file() and p.name != "m.onnx"]
    assert leftovers == []


# --- what gets recorded ---------------------------------------------------------------------


def test_engine_build_and_engine_io_are_recorded(tmp_path, trt_env):
    model = _write_onnx(tmp_path / "m.onnx")
    adapter = _load(model)
    build = adapter._engine_build
    (engine,) = _engines(model, "fp16", "maxn_super")
    record = json.loads(_record_path(engine).read_text())
    assert build["build_key"] == record["build_key"]
    assert engine.name.split(".")[-2] == build["build_key"][:16]
    assert build["tensorrt_version"] == "10.3.0"
    assert build["calibration_digest"] is None
    assert record["inputs"]["input_profile"] == {
        "images": {"min": [1, 3, 640, 640], "opt": [1, 3, 640, 640], "max": [1, 3, 640, 640]}
    }
    assert adapter._engine_io["inputs"] == [
        {"name": "images", "dtype": "float32", "shape": [1, 3, 640, 640]}
    ]


# --- pure functions ----------------------------------------------------------------------------


def test_resolve_input_profile_rule():
    signature = [{"name": "x", "shape": ["N", 3, 224, 224]}, {"name": "y", "shape": [1, "T"]}]
    assert resolve_input_profile(signature, [4, 3, 224, 224]) == {
        "x": {"min": [4, 3, 224, 224], "opt": [4, 3, 224, 224], "max": [4, 3, 224, 224]},
        "y": {"min": [1, 3], "opt": [1, 3], "max": [1, 3]},
    }
    assert resolve_input_profile([{"name": "x", "shape": ["N", 3]}], None)["x"]["min"] == [1, 3]


def test_build_key_is_stable():
    """Pinned: a change to canonical JSON or the key inputs would invalidate
    every cached engine; that must be a deliberate change, not an accident."""
    inputs = engine_cache.build_key_inputs(
        artifact="tensorrt_engine",
        model_hash="a" * 64,
        precision="fp16",
        power_mode="maxn_super",
        input_profile={
            "images": {"min": [1, 3, 640, 640], "opt": [1, 3, 640, 640], "max": [1, 3, 640, 640]}
        },
        calibration_digest=None,
        compiler_version="10.3.0",
        builder_config_version=1,
    )
    assert compute_build_key(inputs) == compute_build_key(json.loads(json.dumps(inputs)))
    assert compute_build_key(inputs) == PINNED_BUILD_KEY


PINNED_BUILD_KEY = "bec2123b74d9d71e841ac4ce80ea0f06367cd78dd8a87377d51b72e8e183cfca"  # gitleaks:allow  # noqa: E501


def test_verify_record_accepts_a_consistent_record():
    inputs = {"model_hash": "b" * 64}
    key = compute_build_key(inputs)
    verify_record({"build_key": key, "inputs": inputs}, key)


# --- the I/O guard and the v2 field ------------------------------------------------------------


def test_engine_io_mismatches():
    model = {
        "inputs": [{"name": "images", "dtype": "float32", "shape": ["N", 3, 640, 640]}],
        "outputs": [{"name": "output0", "dtype": "float32", "shape": [1, 84, 8400]}],
    }
    engine = {
        "inputs": [{"name": "images", "dtype": "float32", "shape": [1, 3, 640, 640]}],
        "outputs": [{"name": "output0", "dtype": "float32", "shape": [1, 84, 8400]}],
    }
    assert engine_io_mismatches(engine, model) == []  # dynamic dim may be concrete in the engine
    engine["outputs"][0]["shape"] = [1, 116, 8400]
    engine["inputs"][0]["dtype"] = "float16"
    problems = engine_io_mismatches(engine, model)
    assert len(problems) == 2
    assert engine_io_mismatches({"inputs": [], "outputs": []}, model) != []


def test_engine_build_must_come_from_the_same_model():
    from tests.open.test_result_v2 import _golden

    data = _golden()
    data["schema_version"] = "2.1"
    data["candidate"]["engine_build"] = {
        "build_key": "c" * 64,
        "built_from_model_hash": data["candidate"]["model_hash"],
        "tensorrt_version": "10.3.0",
        "calibration_digest": None,
    }
    assert ResultV2.model_validate(data).candidate.engine_build is not None
    data["candidate"]["engine_build"]["built_from_model_hash"] = "d" * 64
    with pytest.raises(ValueError, match="built from a different model"):
        ResultV2.model_validate(data)
