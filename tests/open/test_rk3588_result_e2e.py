"""An RK3588 run, end to end, builds a v2 result that validates.

`calbredge run --runtime rknn-fp16` through the real CLI, the real RKNNAdapter, the real
collectors and the real result builder, on a fake RK3588 host: a fake ``rknnlite`` module, a
fake rknn-toolkit-lite2 distribution, and a sysfs and device tree as the Orange Pi 5 Plus has
them. Only what the hardware provides is faked. A run that cannot fill a required field
writes result.invalid.json ("candidate.runtime_version: Input should be a valid string"); this
test builds a whole RK3588 result, so a missing field is noticed.
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest
from calbredge import cli, collector
from calbredge_protocol import rknn_artifact
from calbredge_protocol.result_v2 import ResultV2
from calbredge_protocol.rknn_adapter import RKNNAdapter
from click.testing import CliRunner

from tests.open.fake_hosts import build_rk3588, conv_model
from tests.open.test_rknn_adapter import _make_rknnlite_mock, _patch_rknnlite

pytestmark = pytest.mark.unit

LITE_VERSION = "2.3.2"


@pytest.fixture
def rk3588(tmp_path, monkeypatch) -> Path:
    """The host an RK3588 run sees: pinned as scripts/performance_mode.sh leaves it."""
    root = build_rk3588(tmp_path / "root")
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    build = cli.build_result_v2
    monkeypatch.setattr(cli, "build_result_v2", lambda **kwargs: build(**kwargs, root=root))
    collect = RKNNAdapter.collect_clock_state
    monkeypatch.setattr(
        RKNNAdapter, "collect_clock_state", lambda self, _root=None: collect(self, root)
    )
    return root


def _steady_rknnlite():
    """A fake rknnlite whose inference takes about 5 ms, so the run's p99/p50 is that of a
    steady device and run_status is stable, as a baseline needs. A fake that returns at once
    measures only scheduler noise."""
    api = _make_rknnlite_mock([np.ones((1, 1000), dtype=np.float32)])
    instant = api.RKNNLite.inference

    def inference(self, inputs):
        time.sleep(0.005)
        return instant(self, inputs)

    api.RKNNLite.inference = inference
    return api


def _lite_runtime(monkeypatch, version: str | None) -> None:
    """rknn-toolkit-lite2 installed at ``version``, or not installed at all."""
    real = collector.importlib.metadata.version

    def lookup(name):
        if name != "rknn-toolkit-lite2":
            return real(name)
        if version is None:
            raise collector.importlib.metadata.PackageNotFoundError(name)
        return version

    monkeypatch.setattr(collector.importlib.metadata, "version", lookup)


def _run(
    tmp_path: Path, *, record: bool = False, std_values: list | None = None
) -> tuple[object, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = conv_model(tmp_path / "m.onnx")
    artifact = Path(f"{model}.rknn.fp16.default")
    artifact.write_bytes(b"fake rknn cache")
    if record:  # as `calbredge convert` writes it beside the artifact
        rknn_artifact.write_record(
            artifact,
            rknn_artifact.key_inputs(
                model_hash=rknn_artifact.file_sha256(model),
                precision="fp16",
                target_platform="rk3588",
                model_inputs=[
                    {
                        "name": "x",
                        "source_shape": ["N", 3, 8, 8],
                        "source_dtype": "float",
                        "source_layout": "nchw",
                        "converted_shape": [1, 3, 8, 8],
                    }
                ],
                normalization={
                    "mean_values": [[0, 0, 0]],
                    "std_values": std_values or [[255, 255, 255]],
                },
                feed_dtype="uint8",
                graph_edits=[],
                calibration_digest=None,
                compiler_version="2.3.2",
                builder_config_version=1,
            ),
        )
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        'protocol_version: "1.0"\nwarmup_iterations: 2\nmeasurement_iterations: 20\n'
        "batch_size: 1\ninput_shape: [1, 3, 8, 8]\n"
    )
    rknnlite = _steady_rknnlite()
    with _patch_rknnlite(rknnlite):
        result = CliRunner().invoke(
            cli.main,
            [
                "run", "--model", str(model), "--runtime", "rknn-fp16", "--protocol",
                str(protocol), "--cpu-affinity", "none", "--dry-run",
                "--bundle-dir", str(tmp_path / "runs"),
            ],
        )  # fmt: skip
    assert result.exit_code == 0, result.output
    (bundle,) = [p for p in (tmp_path / "runs").iterdir() if not p.name.startswith(".")]
    return result, bundle


def test_an_rk3588_run_writes_a_valid_v2_result(tmp_path, monkeypatch, rk3588):
    _lite_runtime(monkeypatch, LITE_VERSION)

    result, bundle = _run(tmp_path)

    assert not (bundle / "result.invalid.json").exists(), (
        bundle / "result.invalid.json"
    ).read_text()
    raw = json.loads((bundle / "result.json").read_text())
    parsed = ResultV2.model_validate(raw)
    assert "failed validation" not in result.stderr

    assert parsed.candidate.runtime == "rknn-fp16"
    assert parsed.candidate.runtime_family == "rknn"
    assert parsed.candidate.runtime_version == LITE_VERSION
    assert parsed.candidate.precision == "fp16"
    # What ran, as the RK3588 collectors record it.
    assert parsed.target.clock_state is not None
    assert parsed.target.clock_state["source"] == "rk3588"
    assert parsed.target.clock_state["gpu_governor"] == "simple_ondemand"
    assert parsed.target.clocks_pinned is True
    assert parsed.target.power_mode == "not_applicable"
    assert parsed.target.system_metadata["runtime_version"] == LITE_VERSION
    assert parsed.metrics.latency_p50_ms > 0


def test_an_rk3588_run_without_a_runtime_version_is_invalid_never_a_placeholder(
    tmp_path, monkeypatch, rk3588
):
    """runtime_version is required: a run that cannot say what ran is not a valid result. It
    is recorded as invalid with the field named, and no placeholder stands in for the version."""
    _lite_runtime(monkeypatch, None)

    result, bundle = _run(tmp_path)

    assert not (bundle / "result.json").exists()
    invalid = json.loads((bundle / "result.invalid.json").read_text())
    assert [e for e in invalid["errors"] if "runtime_version" in e] == invalid["errors"]
    assert invalid["errors"][0].startswith("candidate.runtime_version")
    assert invalid["result"]["candidate"]["runtime_version"] is None
    assert "failed validation" in result.stderr


def test_a_re_exported_model_is_refused_by_the_run_itself(tmp_path, monkeypatch, rk3588):
    """Through the CLI: the run stops with the reason, and writes no result."""
    _lite_runtime(monkeypatch, LITE_VERSION)
    model = conv_model(tmp_path / "m.onnx")
    artifact = Path(f"{model}.rknn.fp16.default")
    artifact.write_bytes(b"fake rknn cache")
    rknn_artifact.write_record(
        artifact,
        rknn_artifact.key_inputs(
            model_hash="0" * 64,  # the model this .rknn was converted from, since replaced
            precision="fp16",
            target_platform="rk3588",
            model_inputs=[],
            normalization={},
            feed_dtype="uint8",
            graph_edits=[],
            calibration_digest=None,
            compiler_version="2.3.2",
            builder_config_version=1,
        ),
    )
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        'protocol_version: "1.0"\nwarmup_iterations: 2\nmeasurement_iterations: 20\n'
        "batch_size: 1\ninput_shape: [1, 3, 8, 8]\n"
    )
    with _patch_rknnlite(_make_rknnlite_mock([np.ones((1, 1000), dtype=np.float32)])):
        result = CliRunner().invoke(
            cli.main,
            [
                "run", "--model", str(model), "--runtime", "rknn-fp16", "--protocol",
                str(protocol), "--cpu-affinity", "none", "--dry-run",
                "--bundle-dir", str(tmp_path / "runs"),
            ],
        )  # fmt: skip
    assert result.exit_code != 0
    assert "Refusing to load m.onnx.rknn.fp16.default" in result.output
    assert "built from a different model than the .onnx beside it" in result.output
    assert not (tmp_path / "runs").exists() or not list((tmp_path / "runs").iterdir())
