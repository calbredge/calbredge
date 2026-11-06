"""The build record that travels with a converted .rknn.

protocol/calbredge_protocol/rknn_artifact.py: the key, the record and what verifying it
refuses. Then convert_rknn, with a fake rknn-toolkit2: the record it writes beside the artifact.
"""

import json
from pathlib import Path

import pytest
from calbredge import __version__
from calbredge.converter import RKNN_BUILDER_CONFIG_VERSION, _rknn_toolkit_version, convert_rknn
from calbredge_protocol import rknn_artifact
from calbredge_protocol.engine_cache import compute_build_key, onnx_input_signature
from calbredge_protocol.rknn_artifact import (
    RknnArtifactError,
    load_record,
    record_path,
    verify_record,
)
from PIL import Image

from tests.open.test_rknn_convert import (
    CXR_MOBILENETV2,
    FAKE_TOOLKIT_VERSION,
    MOBILENETV2,
    YOLOV8N,
    _healthy_fused_yolo_output,
    _make_onnx_edit_spy,
    _make_rknn_api_mock,
    _patch_rknn_api,
)

pytestmark = pytest.mark.unit

MODEL_HASH = "a" * 64


def _inputs(**changes) -> dict:
    inputs = rknn_artifact.key_inputs(
        model_hash=MODEL_HASH,
        precision="fp16",
        target_platform="rk3588",
        model_inputs=[
            {
                "name": "images",
                "source_shape": [1, 3, 640, 640],
                "source_dtype": "float",
                "source_layout": "nchw",
                "converted_shape": [1, 3, 640, 640],
            }
        ],
        normalization={"mean_values": [[0, 0, 0]], "std_values": [[255, 255, 255]]},
        feed_dtype="uint8",
        graph_edits=[],
        calibration_digest=None,
        compiler_version="2.3.2",
        builder_config_version=1,
    )
    inputs.update(changes)
    return inputs


@pytest.fixture
def artifact(tmp_path) -> Path:
    path = tmp_path / "model.onnx.rknn.fp16.default"
    path.write_bytes(b"rknn bytes, as converted")
    return path


def _verify(record: dict, artifact: Path, **changes) -> None:
    expected = {
        "model_hash": MODEL_HASH,
        "artifact_sha256": rknn_artifact.file_sha256(artifact),
        "precision": "fp16",
        "target_platform": "rk3588",
        **changes,
    }
    verify_record(record, **expected)


# --- the key and the record --------------------------------------------------------------


def test_the_key_is_the_digest_of_the_inputs_and_every_input_moves_it():
    base = compute_build_key(_inputs())
    assert len(base) == 64
    assert compute_build_key(_inputs()) == base
    changed = {
        "model_hash": "b" * 64,
        "precision": "int8",
        "target_platform": "rk3576",
        "model_inputs": [],
        "normalization": {"mean_values": [[0, 0, 0]], "std_values": [[1, 1, 1]]},
        "feed_dtype": "float32",
        "graph_edits": ["nhwc_to_nchw_input_transpose"],
        "calibration_digest": "c" * 64,
        "compiler_version": "2.4.0",
        "builder_config_version": 2,
    }
    assert set(changed) == set(_inputs()) - {"artifact"}
    for name, value in changed.items():
        assert compute_build_key(_inputs(**{name: value})) != base, name


def test_the_record_sits_beside_the_artifact_and_names_its_content(artifact):
    record = rknn_artifact.write_record(artifact, _inputs(), extra={"calbredge_version": "x"})
    assert record_path(artifact) == artifact.with_name(artifact.name + ".json")
    assert load_record(artifact) == record
    assert record["record_version"] == 1
    assert record["build_key"] == compute_build_key(_inputs())
    assert record["inputs"] == _inputs()
    assert record["artifact_sha256"] == rknn_artifact.file_sha256(artifact)
    assert record["calbredge_version"] == "x"
    assert record["built_at"].endswith("+00:00")
    assert not list(artifact.parent.glob("*.partial"))


def test_a_record_that_describes_its_artifact_and_the_run_verifies(artifact):
    _verify(rknn_artifact.write_record(artifact, _inputs()), artifact)


def test_an_artifact_without_a_record_has_none(artifact):
    assert load_record(artifact) is None


@pytest.mark.parametrize("content", ["{not json", "[]", '"a string"'])
def test_an_unreadable_record_is_an_error_not_an_absence(artifact, content):
    record_path(artifact).write_text(content)
    with pytest.raises(RknnArtifactError, match="build record"):
        load_record(artifact)


# --- what verifying refuses --------------------------------------------------------------


def test_a_re_exported_model_beside_an_old_artifact_is_refused(artifact):
    """The reason the record exists: the .onnx changed, the .rknn beside it did not."""
    record = rknn_artifact.write_record(artifact, _inputs())
    with pytest.raises(RknnArtifactError, match="built from a different model"):
        _verify(record, artifact, model_hash="b" * 64)


def test_a_replaced_artifact_under_an_old_record_is_refused(artifact):
    record = rknn_artifact.write_record(artifact, _inputs())
    artifact.write_bytes(b"rknn bytes, converted again")
    with pytest.raises(RknnArtifactError, match="not the one its build record describes"):
        _verify(record, artifact)


def test_an_edited_record_is_refused(artifact):
    record = rknn_artifact.write_record(artifact, _inputs())
    record["inputs"]["compiler_version"] = "9.9.9"
    with pytest.raises(RknnArtifactError, match="do not hash to its build_key"):
        _verify(record, artifact)


def test_an_edited_record_with_a_recomputed_key_is_still_tied_to_the_run(artifact):
    """Rehashing an edited record gets past the key check, not past what the run asks for."""
    record = rknn_artifact.write_record(artifact, _inputs())
    record["inputs"]["model_hash"] = "b" * 64
    record["build_key"] = compute_build_key(record["inputs"])
    with pytest.raises(RknnArtifactError, match="built from a different model"):
        _verify(record, artifact)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"precision": "int8"}, "built for precision 'fp16', this run needs 'int8'"),
        ({"target_platform": "rk3576"}, "built for target_platform 'rk3588'"),
    ],
)
def test_an_artifact_for_another_precision_or_target_is_refused(artifact, changes, message):
    record = rknn_artifact.write_record(artifact, _inputs())
    with pytest.raises(RknnArtifactError, match=message):
        _verify(record, artifact, **changes)


@pytest.mark.parametrize("missing", ["inputs", "build_key"])
def test_a_record_without_its_inputs_or_key_is_refused(artifact, missing):
    record = rknn_artifact.write_record(artifact, _inputs())
    del record[missing]
    with pytest.raises(RknnArtifactError, match="missing its inputs or build_key"):
        _verify(record, artifact)


def test_a_tensorrt_engine_record_is_not_an_rknn_record(artifact):
    record = rknn_artifact.write_record(artifact, _inputs(artifact="tensorrt_engine"))
    with pytest.raises(RknnArtifactError, match=r"not a \.rknn"):
        _verify(record, artifact)


def test_calibration_digest_follows_the_images_their_order_and_the_dtype(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(b"image a")
    b.write_bytes(b"image b")
    base = rknn_artifact.calibration_digest([a, b], "asymmetric_quantized-8")
    assert rknn_artifact.calibration_digest([a, b], "asymmetric_quantized-8") == base
    assert rknn_artifact.calibration_digest([b, a], "asymmetric_quantized-8") != base
    assert rknn_artifact.calibration_digest([a], "asymmetric_quantized-8") != base
    assert rknn_artifact.calibration_digest([a, b], "asymmetric_quantized-16") != base
    b.write_bytes(b"image b, replaced")  # same name, different content
    assert rknn_artifact.calibration_digest([a, b], "asymmetric_quantized-8") != base


# --- convert_rknn writes it --------------------------------------------------------------


@pytest.fixture
def calibration_dataset(tmp_path) -> Path:
    image = tmp_path / "calib.png"
    Image.new("RGB", (224, 224), color=(100, 150, 200)).save(image)
    dataset = tmp_path / "dataset.txt"
    dataset.write_text("calib.png\n")  # relative: resolved against the dataset file
    return dataset


def _convert(tmp_path, model: Path, **kwargs) -> tuple[Path, dict]:
    output = tmp_path / f"{model.name}.rknn.{kwargs.get('precision', 'fp16')}.default"
    outputs = [_healthy_fused_yolo_output()] if "calibration_dataset" in kwargs else []
    with _patch_rknn_api(_make_rknn_api_mock(inference_outputs=outputs)):
        convert_rknn(model, output, **kwargs)
    record = load_record(output)
    assert record is not None
    return output, record


def test_convert_writes_a_record_that_verifies_against_the_model_and_the_artifact(tmp_path):
    output, record = _convert(tmp_path, MOBILENETV2, precision="fp16")

    verify_record(
        record,
        model_hash=rknn_artifact.file_sha256(MOBILENETV2),
        artifact_sha256=rknn_artifact.file_sha256(output),
        precision="fp16",
        target_platform="rk3588",
    )
    inputs = record["inputs"]
    assert inputs["compiler_version"] == FAKE_TOOLKIT_VERSION
    assert inputs["builder_config_version"] == RKNN_BUILDER_CONFIG_VERSION
    assert inputs["calibration_digest"] is None
    assert inputs["graph_edits"] == []
    assert inputs["feed_dtype"] == "uint8"
    assert inputs["normalization"] == {
        "mean_values": [[0, 0, 0]],
        "std_values": [[255, 255, 255]],
    }
    assert record["calbredge_version"] == __version__
    assert record["artifact_name"] == output.name
    assert json.loads(record_path(output).read_text()) == record


def test_the_record_holds_the_models_inputs_as_declared_and_as_converted(tmp_path):
    """The input contract of the artifact can be read from its record."""
    _, record = _convert(tmp_path, MOBILENETV2, precision="fp16")
    (source,) = onnx_input_signature(MOBILENETV2)
    (entry,) = record["inputs"]["model_inputs"]
    assert entry["name"] == source["name"]
    assert entry["source_shape"] == source["shape"]  # the symbolic batch kept as its name
    assert isinstance(entry["source_shape"][0], str)
    assert entry["converted_shape"] == [1, 3, 224, 224]  # the batch resolved to 1
    assert entry["source_dtype"] == "float"
    assert entry["source_layout"] == "nchw"


@pytest.mark.skipif(not CXR_MOBILENETV2.exists(), reason="needs the cxr_mobilenetv2 fixture")
def test_an_nhwc_model_records_its_layout_and_the_transpose(tmp_path):
    output = tmp_path / "cxr.onnx.rknn.fp16.default"
    api = _make_rknn_api_mock(inference_outputs=[])
    with _patch_rknn_api(api, onnx_edit=_make_onnx_edit_spy([])):
        convert_rknn(CXR_MOBILENETV2, output, precision="fp16")
    record = load_record(output)
    assert record is not None
    assert record["inputs"]["graph_edits"] == ["nhwc_to_nchw_input_transpose"]
    (entry,) = record["inputs"]["model_inputs"]
    assert entry["source_layout"] == "nhwc"
    assert entry["source_shape"][1:] == [224, 224, 3]


@pytest.mark.skipif(not YOLOV8N.exists(), reason="needs the yolov8n fixture")
def test_a_static_model_records_its_own_shape_as_converted(tmp_path):
    _, record = _convert(tmp_path, YOLOV8N, precision="fp16")
    (entry,) = record["inputs"]["model_inputs"]
    assert entry["source_shape"] == entry["converted_shape"] == [1, 3, 640, 640]


def test_an_int8_record_carries_the_calibration_digest(tmp_path, calibration_dataset, monkeypatch):
    from calbredge.validate_rknn import ValidationResult

    passed = ValidationResult(passed=True, total_bands=1, meaningful_bands=1, collapsed_bands=[])
    monkeypatch.setattr("calbredge.converter.validate_conversion", lambda *a, **k: passed)
    _, record = _convert(
        tmp_path, MOBILENETV2, precision="int8", calibration_dataset=calibration_dataset
    )
    expected = rknn_artifact.calibration_digest(
        [calibration_dataset.parent / "calib.png"], "asymmetric_quantized-8"
    )
    assert record["inputs"]["calibration_digest"] == expected
    assert record["inputs"]["precision"] == "int8"


def test_an_fp16_build_checked_against_a_dataset_has_no_calibration_digest(
    tmp_path, calibration_dataset
):
    """The dataset only feeds the correctness check for FP16: it is not part of the build."""
    _, record = _convert(
        tmp_path,
        MOBILENETV2,
        precision="fp16",
        calibration_dataset=calibration_dataset,
        skip_validate_conversion=True,
    )
    assert record["inputs"]["calibration_digest"] is None


def test_converting_again_replaces_the_artifact_and_its_record_together(tmp_path):
    output, first = _convert(tmp_path, MOBILENETV2, precision="fp16")
    api = _make_rknn_api_mock(inference_outputs=[])
    with _patch_rknn_api(api, toolkit_version="2.4.0"):
        convert_rknn(MOBILENETV2, output, precision="fp16")
    second = load_record(output)
    assert second is not None
    assert second["inputs"]["compiler_version"] == "2.4.0"
    assert second["build_key"] != first["build_key"]


def test_a_failed_export_leaves_no_record(tmp_path):
    output = tmp_path / "model.rknn"
    api = _make_rknn_api_mock(inference_outputs=[], export_ret=-1)
    with _patch_rknn_api(api), pytest.raises(RuntimeError, match="export_rknn failed"):
        convert_rknn(MOBILENETV2, output, precision="fp16")
    assert not record_path(output).exists()


def test_an_unknown_toolkit_version_stops_the_conversion_before_it_builds(tmp_path, monkeypatch):
    """The compiler version is part of the key: no artifact is built that could not say it."""
    import importlib.metadata

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(RuntimeError, match="cannot read the installed rknn-toolkit2 version"):
        _rknn_toolkit_version()

    output = tmp_path / "model.rknn"
    instances: list = []
    api = _make_rknn_api_mock(inference_outputs=[], instances=instances)
    with _patch_rknn_api(api):
        monkeypatch.setattr(
            "calbredge.converter._rknn_toolkit_version",
            lambda: (_ for _ in ()).throw(RuntimeError("cannot read the installed version")),
        )
        with pytest.raises(RuntimeError, match="cannot read the installed version"):
            convert_rknn(MOBILENETV2, output, precision="fp16")
    assert instances == []
    assert not output.exists()
