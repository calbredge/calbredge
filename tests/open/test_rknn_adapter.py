"""Unit tests for the pure-Python logic added to RKNNAdapter in the 2026-07
adapter-hardening review (constructor input-spec args, dtype coercion, canary
input-spec resolution, and the fail-closed/skip-loudly canary behaviour), plus
the shape-wiring tests added when ``load_model`` gained an ``input_shape``
kwarg so the benchmark runner can thread ``BenchmarkProtocol.input_shape``
through uniformly for both TensorRTAdapter and RKNNAdapter.

Most of these tests touch neither ``rknnlite`` nor real hardware — code paths
that return or raise before the adapter would dereference ``self._rknn``
require no mocking. The ``TestLoadModelInputShapeWiring`` class below does
mock ``rknnlite.api.RKNNLite`` (via ``sys.modules``, mirroring
test_tensorrt_adapter.py's pattern for ``tensorrt``/``pycuda``) so that
``load_model`` — and therefore the real correctness canary — can run
end-to-end. On-device behaviour beyond that is out of scope here; see
docs/ORANGE_PI_SESSION2.md for that validation plan.
"""

import json
import sys
import time
import types
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pytest
from calbredge_protocol import rknn_artifact
from calbredge_protocol.models import BenchmarkProtocol
from calbredge_protocol.power_meter import PowerReading
from calbredge_protocol.result_v2 import ConversionInput
from calbredge_protocol.rknn_adapter import RKNNAdapter, _coerce_input_dtype

# ---------------------------------------------------------------------------
# 1. Constructor
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRKNNAdapterInit:
    def test_expected_input_shape_stored_as_tuple(self):
        adapter = RKNNAdapter(precision="fp16", expected_input_shape=[1, 224, 224, 3])
        assert adapter._expected_input_shape == (1, 224, 224, 3)
        assert isinstance(adapter._expected_input_shape, tuple)

    def test_invalid_precision_raises(self):
        with pytest.raises(ValueError, match="precision must be one of"):
            RKNNAdapter(precision="int4")


# ---------------------------------------------------------------------------
# 2. _coerce_input_dtype
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCoerceInputDtype:
    def test_lowercase_rknn_style_string(self):
        assert _coerce_input_dtype("uint8") == np.dtype(np.uint8)

    def test_uppercase_rknn_style_string(self):
        assert _coerce_input_dtype("UINT8") == np.dtype(np.uint8)

    def test_float16_rknn_style_string(self):
        assert _coerce_input_dtype("FLOAT16") == np.dtype(np.float16)

    def test_numpy_type(self):
        assert _coerce_input_dtype(np.uint8) == np.dtype(np.uint8)

    def test_numpy_dtype_instance(self):
        assert _coerce_input_dtype(np.dtype("float32")) == np.dtype(np.float32)


# ---------------------------------------------------------------------------
# 3. _resolve_canary_input_spec
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveCanaryInputSpec:
    def test_explicit_shape_defaults_dtype_to_uint8(self):
        adapter = RKNNAdapter(precision="fp16", expected_input_shape=(1, 224, 224, 3))
        shape, dtype = adapter._resolve_canary_input_spec(Path("mobilenetv2.onnx"))
        assert shape == (1, 224, 224, 3)
        assert dtype == np.dtype(np.uint8)


# ---------------------------------------------------------------------------
# 4. _run_correctness_canary — fail-closed (int8) / skip-loudly (fp16)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCorrectnessCanaryUnknownShape:
    def test_int8_with_no_shape_raises(self):
        adapter = RKNNAdapter(precision="int8")
        with pytest.raises(RuntimeError, match="Refusing to load INT8"):
            adapter._run_correctness_canary(Path("m.onnx.rknn.int8.default"), Path("m.onnx"))

    def test_fp16_with_no_shape_skips_without_touching_runtime(self):
        adapter = RKNNAdapter(precision="fp16")
        adapter._rknn = None  # the skip path must return before dereferencing this
        result = adapter._run_correctness_canary(Path("m.onnx.rknn.fp16.default"), Path("m.onnx"))
        assert result is None
        assert adapter._rknn is None


# ---------------------------------------------------------------------------
# 5. load_model(input_shape=...) — runner shape wiring
# ---------------------------------------------------------------------------


def _make_rknnlite_mock(output_arrays, expected_dtype=np.uint8, instances=None):
    """Build a fake ``rknnlite.api`` module.

    ``RKNNLite().inference(...)`` returns *output_arrays* only when every
    input array's dtype matches *expected_dtype* — otherwise it raises,
    mirroring real RKNN's dtype strictness (see
    RKNNAdapter._resolve_canary_input_spec's docstring: "a wrong dtype makes
    RKNN raise loudly"). This means the mock can no longer silently accept a
    float32 feed the way a value-blind stub would, so a test asserting on
    ``inference()``'s output actually proves the caller sent the right dtype
    rather than just that *something* was passed.

    load_rknn/init_runtime report success (return code 0), so tests can drive
    ``RKNNAdapter.load_model`` end-to-end — including the real correctness
    canary — without real hardware or the rknn-toolkit-lite2 package.

    Args:
        instances: Optional list; if given, every constructed ``RKNNLite``
                   instance appends itself to it. Lets a caller that has no
                   direct handle on the adapter (e.g. a test driving the CLI
                   through Click's CliRunner, which builds the adapter
                   internally) still inspect ``last_inference_inputs`` from
                   outside. ``None`` (the default) preserves the original
                   behaviour for callers that already hold the adapter and
                   read ``adapter._rknn`` directly.
    """
    rknnlite_api = types.ModuleType("rknnlite.api")

    class _RKNNLite:
        def __init__(self):
            if instances is not None:
                instances.append(self)

        def load_rknn(self, path):
            return 0

        def init_runtime(self):
            return 0

        def inference(self, inputs):
            for arr in inputs:
                if arr.dtype != np.dtype(expected_dtype):
                    raise ValueError(
                        f"RKNN dtype mismatch: model expects {expected_dtype}, got {arr.dtype}"
                    )
            self.last_inference_inputs = inputs
            return output_arrays

        def release(self):
            pass

    rknnlite_api.RKNNLite = _RKNNLite
    return rknnlite_api


def _patch_rknnlite(rknnlite_api_mock):
    """Return a context manager injecting the fake ``rknnlite.api`` module."""
    rknnlite = types.ModuleType("rknnlite")
    return patch.dict(
        sys.modules,
        {
            "rknnlite": rknnlite,
            "rknnlite.api": rknnlite_api_mock,
        },
    )


@pytest.fixture
def rknn_cache_files(tmp_path):
    """Write a fake .onnx path plus sibling .rknn cache files for both precisions."""
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake onnx bytes")
    (tmp_path / "model.onnx.rknn.int8.default").write_bytes(b"fake rknn cache")
    (tmp_path / "model.onnx.rknn.fp16.default").write_bytes(b"fake rknn cache")
    return onnx_path


@pytest.mark.unit
class TestLoadModelInputShapeWiring:
    """Proves BenchmarkProtocol.input_shape reaches the adapter via
    load_model(input_shape=...) — the same value agent/calbredge/cli.py
    already threads through for TensorRTAdapter — and that it lets the
    correctness canary actually run instead of fail-closing (int8) or
    loud-skipping (fp16).
    """

    def test_input_shape_kwarg_sets_expected_input_shape(self, rknn_cache_files):
        adapter = RKNNAdapter(precision="fp16")
        healthy_output = [np.ones((1, 1000), dtype=np.float32)]
        with _patch_rknnlite(_make_rknnlite_mock(healthy_output)):
            adapter.load_model(rknn_cache_files, input_shape=[1, 224, 224, 3])
        assert adapter._expected_input_shape == (1, 224, 224, 3)
        assert adapter.is_loaded is True

    def test_constructor_shape_wins_over_load_model_kwarg(self, rknn_cache_files):
        adapter = RKNNAdapter(precision="fp16", expected_input_shape=(1, 300, 300, 3))
        healthy_output = [np.ones((1, 1000), dtype=np.float32)]
        with _patch_rknnlite(_make_rknnlite_mock(healthy_output)):
            adapter.load_model(rknn_cache_files, input_shape=[1, 224, 224, 3])
        assert adapter._expected_input_shape == (1, 300, 300, 3)

    def test_int8_no_longer_raises_when_runner_shape_available(self, rknn_cache_files):
        """rknn-int8 used to fail-close in load_model with no shape supplied;
        with the runner's input_shape threaded through, it loads normally."""
        adapter = RKNNAdapter(precision="int8")
        healthy_output = [np.ones((1, 1000), dtype=np.float32)]
        with _patch_rknnlite(_make_rknnlite_mock(healthy_output)):
            adapter.load_model(rknn_cache_files, input_shape=[1, 224, 224, 3])
        assert adapter.is_loaded is True

    def test_int8_still_raises_without_runner_shape(self, rknn_cache_files):
        """Unchanged fail-closed behaviour when the runner passes no shape."""
        adapter = RKNNAdapter(precision="int8")
        healthy_output = [np.ones((1, 1000), dtype=np.float32)]
        with (
            _patch_rknnlite(_make_rknnlite_mock(healthy_output)),
            pytest.raises(RuntimeError, match="Refusing to load INT8"),
        ):
            adapter.load_model(rknn_cache_files)

    def test_fp16_runs_real_canary_instead_of_skipping(self, rknn_cache_files, capsys):
        """rknn-fp16 used to always loud-skip the canary with no shape source;
        with the runner's input_shape threaded through, it runs a real
        zero-input inference pass instead."""
        adapter = RKNNAdapter(precision="fp16")
        healthy_output = [np.ones((1, 1000), dtype=np.float32)]
        with _patch_rknnlite(_make_rknnlite_mock(healthy_output)):
            adapter.load_model(rknn_cache_files, input_shape=[1, 224, 224, 3])
        captured = capsys.readouterr()
        assert "SKIPPED" not in captured.out
        assert adapter._rknn.last_inference_inputs[0].shape == (1, 224, 224, 3)
        assert adapter._rknn.last_inference_inputs[0].dtype == np.uint8

    def test_fp16_canary_noise_fallback_warns_not_raises_on_degenerate_output(
        self, rknn_cache_files, capsys
    ):
        """Generation 3 behaviour change: with no real image resolvable (no
        canary_image_path, no sibling dataset.txt), the canary falls back to
        fixed-seed noise, and a low nonzero ratio on THAT input is now a
        loud warning, not a hard failure — a controlled on-device A/B proved
        noise alone produces false positives on legitimate sparse/
        large-object detection outputs (see _run_correctness_canary's
        docstring). This replaces the old
        test_fp16_canary_still_fails_closed_on_degenerate_output, which
        asserted the pre-fix hard-fail-on-noise behaviour."""
        adapter = RKNNAdapter(precision="fp16")
        all_zero_output = [np.zeros((1, 1000), dtype=np.float32)]
        with _patch_rknnlite(_make_rknnlite_mock(all_zero_output)):
            adapter.load_model(rknn_cache_files, input_shape=[1, 224, 224, 3])
        assert adapter.is_loaded is True
        captured = capsys.readouterr()
        assert "WARNING, not a hard failure" in captured.out
        assert "fixed-seed noise" in captured.out


# ---------------------------------------------------------------------------
# 5b. Correctness canary — real-image resolution (generation 3)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCanaryRealImage:
    """Generation 3 of the load-time canary: prefer a real representative
    image over synthetic noise when one is resolvable, and only hard-fail
    on real-image evidence — see _run_correctness_canary's docstring for why
    noise alone was demoted to a warning."""

    @staticmethod
    def _write_solid_color_image(path: Path, color: tuple[int, int, int]) -> None:
        """A flat-color image survives PIL's resize unchanged (no
        interpolation artifacts), so the resized array can be asserted
        against the exact input color — proof the real image (not noise)
        was fed to inference."""
        from PIL import Image

        img = Image.new("RGB", (8, 8), color)
        # PNG (lossless) — JPEG's chroma subsampling can shift a flat
        # color's pixel values by 1, which would defeat the exact-match
        # assertions these tests rely on to prove the real image was fed.
        img.save(path, format="PNG")

    def test_dataset_txt_convention_resolves_and_uses_real_image(self, rknn_cache_files):
        onnx_dir = rknn_cache_files.parent
        self._write_solid_color_image(onnx_dir / "calib.jpg", (200, 100, 50))
        (onnx_dir / "dataset.txt").write_text("calib.jpg\n")

        adapter = RKNNAdapter(precision="fp16")
        healthy_output = [np.ones((1, 1000), dtype=np.float32)]
        with _patch_rknnlite(_make_rknnlite_mock(healthy_output)):
            adapter.load_model(rknn_cache_files, input_shape=[1, 224, 224, 3])

        fed = adapter._rknn.last_inference_inputs[0]
        assert fed.shape == (1, 224, 224, 3)
        assert fed.dtype == np.uint8
        assert np.all(fed[0, :, :, 0] == 200)
        assert np.all(fed[0, :, :, 1] == 100)
        assert np.all(fed[0, :, :, 2] == 50)

    def test_explicit_canary_image_path_wins_over_dataset_txt(self, rknn_cache_files):
        onnx_dir = rknn_cache_files.parent
        self._write_solid_color_image(onnx_dir / "calib.jpg", (1, 1, 1))
        (onnx_dir / "dataset.txt").write_text("calib.jpg\n")

        explicit_image = onnx_dir / "explicit.jpg"
        self._write_solid_color_image(explicit_image, (222, 111, 33))

        adapter = RKNNAdapter(precision="fp16", canary_image_path=explicit_image)
        healthy_output = [np.ones((1, 1000), dtype=np.float32)]
        with _patch_rknnlite(_make_rknnlite_mock(healthy_output)):
            adapter.load_model(rknn_cache_files, input_shape=[1, 224, 224, 3])

        fed = adapter._rknn.last_inference_inputs[0]
        assert np.all(fed[0, :, :, 0] == 222)
        assert np.all(fed[0, :, :, 1] == 111)
        assert np.all(fed[0, :, :, 2] == 33)

    def test_load_model_kwarg_canary_image_path(self, rknn_cache_files):
        onnx_dir = rknn_cache_files.parent
        image_path = onnx_dir / "runner_supplied.jpg"
        self._write_solid_color_image(image_path, (9, 8, 7))

        adapter = RKNNAdapter(precision="fp16")
        healthy_output = [np.ones((1, 1000), dtype=np.float32)]
        with _patch_rknnlite(_make_rknnlite_mock(healthy_output)):
            adapter.load_model(
                rknn_cache_files,
                input_shape=[1, 224, 224, 3],
                canary_image_path=image_path,
            )
        fed = adapter._rknn.last_inference_inputs[0]
        assert np.all(fed[0, :, :, 0] == 9)

    def test_constructor_canary_image_path_wins_over_load_model_kwarg(self, rknn_cache_files):
        onnx_dir = rknn_cache_files.parent
        constructor_image = onnx_dir / "constructor.jpg"
        self._write_solid_color_image(constructor_image, (77, 88, 99))
        kwarg_image = onnx_dir / "kwarg.jpg"
        self._write_solid_color_image(kwarg_image, (1, 2, 3))

        adapter = RKNNAdapter(precision="fp16", canary_image_path=constructor_image)
        healthy_output = [np.ones((1, 1000), dtype=np.float32)]
        with _patch_rknnlite(_make_rknnlite_mock(healthy_output)):
            adapter.load_model(
                rknn_cache_files,
                input_shape=[1, 224, 224, 3],
                canary_image_path=kwarg_image,
            )
        fed = adapter._rknn.last_inference_inputs[0]
        assert np.all(fed[0, :, :, 0] == 77)

    def test_real_image_collapse_hard_fails(self, rknn_cache_files):
        """A degenerate output verified against a REAL image must still
        raise — real-image evidence is trustworthy, unlike noise."""
        onnx_dir = rknn_cache_files.parent
        self._write_solid_color_image(onnx_dir / "calib.jpg", (50, 50, 50))
        (onnx_dir / "dataset.txt").write_text("calib.jpg\n")

        adapter = RKNNAdapter(precision="fp16")
        all_zero_output = [np.zeros((1, 1000), dtype=np.float32)]
        with (
            _patch_rknnlite(_make_rknnlite_mock(all_zero_output)),
            pytest.raises(RuntimeError, match="Correctness canary FAILED"),
        ):
            adapter.load_model(rknn_cache_files, input_shape=[1, 224, 224, 3])

    def test_missing_explicit_path_falls_back_gracefully(self, rknn_cache_files, capsys):
        """A nonexistent canary_image_path must not crash load_model — it
        falls back to the dataset.txt convention, then noise."""
        nonexistent = rknn_cache_files.parent / "does_not_exist.jpg"
        adapter = RKNNAdapter(precision="fp16", canary_image_path=nonexistent)
        healthy_output = [np.ones((1, 1000), dtype=np.float32)]
        with _patch_rknnlite(_make_rknnlite_mock(healthy_output)):
            adapter.load_model(rknn_cache_files, input_shape=[1, 224, 224, 3])
        assert adapter.is_loaded is True
        captured = capsys.readouterr()
        assert "does not exist" in captured.out

    def test_no_dataset_txt_no_explicit_path_uses_noise_silently(self, rknn_cache_files, capsys):
        """No candidate image at all is the common/expected case and must
        not print spurious 'candidate image' warnings — only genuine
        resolution failures (missing explicit path, unreadable image, etc.)
        should warn."""
        adapter = RKNNAdapter(precision="fp16")
        healthy_output = [np.ones((1, 1000), dtype=np.float32)]
        with _patch_rknnlite(_make_rknnlite_mock(healthy_output)):
            adapter.load_model(rknn_cache_files, input_shape=[1, 224, 224, 3])
        captured = capsys.readouterr()
        assert "candidate image" not in captured.out

    def test_dataset_txt_present_but_image_missing_falls_back_to_noise(
        self, rknn_cache_files, capsys
    ):
        """dataset.txt exists and names an image, but the image file itself
        is missing — must warn and fall back to noise, not crash."""
        onnx_dir = rknn_cache_files.parent
        (onnx_dir / "dataset.txt").write_text("missing.jpg\n")

        adapter = RKNNAdapter(precision="fp16")
        healthy_output = [np.ones((1, 1000), dtype=np.float32)]
        with _patch_rknnlite(_make_rknnlite_mock(healthy_output)):
            adapter.load_model(rknn_cache_files, input_shape=[1, 224, 224, 3])
        assert adapter.is_loaded is True
        captured = capsys.readouterr()
        assert "Falling back to noise" in captured.out


# ---------------------------------------------------------------------------
# 6. input_dtype — canary and benchmark loop must agree, and never drift
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInputDtypeSharedSource:
    """Proves RKNNAdapter.input_dtype (read by the runner at cli.py:343 to
    synthesize benchmark-loop inputs) and the correctness canary's dtype both
    resolve from the single _resolve_input_dtype source, so they cannot
    disagree — the bug this guards against: canary passes with uint8, then
    the real measurement loop feeds float32 and RKNN dies or produces garbage
    on the first real inference.
    """

    def test_input_dtype_property_defaults_to_uint8(self):
        adapter = RKNNAdapter(precision="fp16")
        assert adapter.input_dtype == np.dtype(np.uint8)

    def test_input_dtype_property_honours_explicit_constructor_dtype(self):
        adapter = RKNNAdapter(precision="fp16", expected_input_dtype="float16")
        assert adapter.input_dtype == np.dtype(np.float16)

    def test_canary_and_measurement_loop_both_feed_uint8(self, rknn_cache_files):
        """End-to-end: load_model's canary and a post-load run_inference call
        (simulating cli.py's benchmark loop) both send uint8 to RKNNLite."""
        adapter = RKNNAdapter(precision="fp16")
        healthy_output = [np.ones((1, 1000), dtype=np.float32)]
        with _patch_rknnlite(_make_rknnlite_mock(healthy_output)):
            adapter.load_model(rknn_cache_files, input_shape=[1, 224, 224, 3])
            # The canary (run inside load_model) already fed uint8.
            assert adapter._rknn.last_inference_inputs[0].dtype == np.uint8

            # Simulate cli.py:343's synthesis: .astype(adapter.input_dtype).
            benchmark_inputs = {"0": np.random.randn(1, 224, 224, 3).astype(adapter.input_dtype)}
            adapter.run_inference(benchmark_inputs)
            assert adapter._rknn.last_inference_inputs[0].dtype == np.uint8

    def test_float32_measurement_feed_is_rejected(self, rknn_cache_files):
        """A float32 feed (the old cli.py:343 hardcoded dtype) must be
        rejected by RKNN once the model's real dtype is uint8 — proving the
        mock models dtype strictness and isn't just theater."""
        adapter = RKNNAdapter(precision="fp16")
        healthy_output = [np.ones((1, 1000), dtype=np.float32)]
        with _patch_rknnlite(_make_rknnlite_mock(healthy_output)):
            adapter.load_model(rknn_cache_files, input_shape=[1, 224, 224, 3])
            stale_float32_inputs = {"0": np.random.randn(1, 224, 224, 3).astype(np.float32)}
            with pytest.raises(ValueError, match="RKNN dtype mismatch"):
                adapter.run_inference(stale_float32_inputs)


@pytest.mark.unit
class TestBaseAdapterInputDtypeDefault:
    """Regression guard for the shared cli.py:343 path: adapters that don't
    override RuntimeAdapter.input_dtype (TensorRT, ONNX Runtime, Mock) must
    keep synthesizing float32 inputs exactly as before.
    """

    def test_mock_adapter_input_dtype_is_float32(self):
        from calbredge_protocol.mock_adapter import MockRuntimeAdapter

        assert MockRuntimeAdapter().input_dtype == np.dtype(np.float32)


# ---------------------------------------------------------------------------
# 6. collect_clock_state() — RK3588 governor collection (B2)
# ---------------------------------------------------------------------------

_CLOCK_STATE_RELPATHS = {
    "compatible": "proc/device-tree/compatible",
    "policy0": "sys/devices/system/cpu/cpufreq/policy0/scaling_governor",
    "policy4": "sys/devices/system/cpu/cpufreq/policy4/scaling_governor",
    "policy6": "sys/devices/system/cpu/cpufreq/policy6/scaling_governor",
    "npu": "sys/class/devfreq/fdab0000.npu/governor",
    "dmc": "sys/class/devfreq/dmc/governor",
    "gpu": "sys/class/devfreq/fb000000.gpu/governor",
}


_CPU_MAX_KHZ = {"policy0": 1800000, "policy4": 2400000, "policy6": 2400000}
_DEVFREQ_MAX_HZ = {"npu": 1000000000, "dmc": 2112000000, "gpu": 1000000000}


def _write_fake_sysfs(
    root: Path, *, skip: frozenset[str] | set[str] = frozenset(), governor: str = "performance"
) -> None:
    """Build a fake sysfs/procfs tree under ``root`` for collect_clock_state tests.

    Writes every path in _CLOCK_STATE_RELPATHS except those named in ``skip``, and beside each
    governor file its frequency files (kHz for a CPU policy, Hz for a devfreq node).
    ``compatible`` gets rk3588 device-tree bytes; every governor file gets ``governor``.
    """
    for key, relpath in _CLOCK_STATE_RELPATHS.items():
        if key in skip:
            continue
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        if key == "compatible":
            path.write_bytes(b"rockchip,rk3588\x00")
            continue
        path.write_text(f"{governor}\n")
        if key in _CPU_MAX_KHZ:
            khz = _CPU_MAX_KHZ[key]
            (path.parent / "scaling_cur_freq").write_text(f"{khz}\n")
            (path.parent / "scaling_min_freq").write_text("408000\n")
            (path.parent / "scaling_max_freq").write_text(f"{khz}\n")
            (path.parent / "cpuinfo_max_freq").write_text(f"{khz}\n")
        else:
            (path.parent / "cur_freq").write_text(f"{_DEVFREQ_MAX_HZ[key]}\n")
            (path.parent / "max_freq").write_text(f"{_DEVFREQ_MAX_HZ[key]}\n")


def _cpu_frequencies() -> dict:
    return {
        policy: {
            "scaling_cur_freq": khz,
            "scaling_min_freq": 408000,
            "scaling_max_freq": khz,
            "cpuinfo_max_freq": khz,
        }
        for policy, khz in _CPU_MAX_KHZ.items()
    }


@pytest.mark.unit
class TestCollectClockState:
    def test_populated_case_all_governors_performance(self, tmp_path, monkeypatch):
        monkeypatch.setattr("platform.machine", lambda: "aarch64")
        _write_fake_sysfs(tmp_path, governor="performance")
        adapter = RKNNAdapter(precision="fp16")

        result = adapter.collect_clock_state(root=tmp_path)

        assert result == {
            "captured": True,
            "source": "rk3588",
            "cpu_governors": {
                "policy0": "performance",
                "policy4": "performance",
                "policy6": "performance",
            },
            "cpu_frequencies": _cpu_frequencies(),
            "npu_governor": "performance",
            "npu_frequency": {"cur_freq": 1000000000, "max_freq": 1000000000},
            "dmc_governor": "performance",
            "dmc_frequency": {"cur_freq": 2112000000, "max_freq": 2112000000},
            "gpu_governor": "performance",
            "gpu_frequency": {"cur_freq": 1000000000, "max_freq": 1000000000},
        }

    @pytest.mark.parametrize(
        "name", ["scaling_cur_freq", "scaling_min_freq", "scaling_max_freq", "cpuinfo_max_freq"]
    )
    @pytest.mark.parametrize("damage", ["missing", "not a number"])
    def test_sentinel_when_a_cpu_frequency_cannot_be_read(
        self, tmp_path, monkeypatch, name, damage
    ):
        """clocks_pinned compares the scaling maximum with the hardware maximum, so a CPU
        frequency is a required read: never a partial dict, never a default."""
        monkeypatch.setattr("platform.machine", lambda: "aarch64")
        _write_fake_sysfs(tmp_path)
        path = (tmp_path / _CLOCK_STATE_RELPATHS["policy4"]).parent / name
        if damage == "missing":
            path.unlink()
        else:
            path.write_text("<unsupported>\n")

        result = RKNNAdapter(precision="fp16").collect_clock_state(root=tmp_path)

        assert result is not None
        assert result["captured"] is False
        assert name in result["reason"]
        assert "cpu_governors" not in result
        assert "cpu_frequencies" not in result

    @pytest.mark.parametrize("node", ["npu", "dmc", "gpu"])
    def test_an_unreadable_devfreq_frequency_is_null_not_a_failure(
        self, tmp_path, monkeypatch, node
    ):
        """Devfreq frequencies are recorded, not required: null, never 0 and never a guess."""
        monkeypatch.setattr("platform.machine", lambda: "aarch64")
        _write_fake_sysfs(tmp_path)
        directory = (tmp_path / _CLOCK_STATE_RELPATHS[node]).parent
        (directory / "cur_freq").unlink()
        (directory / "max_freq").write_text("\n")

        result = RKNNAdapter(precision="fp16").collect_clock_state(root=tmp_path)

        assert result is not None
        assert result["captured"] is True
        assert result[f"{node}_frequency"] == {"cur_freq": None, "max_freq": None}

    def test_sentinel_when_required_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("platform.machine", lambda: "aarch64")
        _write_fake_sysfs(tmp_path, skip={"npu"})
        adapter = RKNNAdapter(precision="fp16")

        result = adapter.collect_clock_state(root=tmp_path)

        assert result is not None
        assert result["captured"] is False
        assert "reason" in result
        assert "cpu_governors" not in result

    def test_returns_none_on_non_rk3588(self, tmp_path, monkeypatch):
        monkeypatch.setattr("platform.machine", lambda: "aarch64")
        _write_fake_sysfs(tmp_path)
        # Overwrite compatible with a non-rk3588 device-tree string.
        (tmp_path / _CLOCK_STATE_RELPATHS["compatible"]).write_bytes(b"nvidia,tegra234\x00")
        adapter = RKNNAdapter(precision="fp16")

        result = adapter.collect_clock_state(root=tmp_path)

        assert result is None

    def test_returns_none_on_non_aarch64_machine(self, tmp_path, monkeypatch):
        monkeypatch.setattr("platform.machine", lambda: "x86_64")
        _write_fake_sysfs(tmp_path)
        adapter = RKNNAdapter(precision="fp16")

        result = adapter.collect_clock_state(root=tmp_path)

        assert result is None

    def test_gpu_optional_missing_gpu_file_still_captured(self, tmp_path, monkeypatch):
        monkeypatch.setattr("platform.machine", lambda: "aarch64")
        _write_fake_sysfs(tmp_path, skip={"gpu"})
        adapter = RKNNAdapter(precision="fp16")

        result = adapter.collect_clock_state(root=tmp_path)

        assert result is not None
        assert result["captured"] is True
        assert "gpu_governor" not in result
        assert "gpu_frequency" not in result
        assert result["cpu_frequencies"] == _cpu_frequencies()
        assert result["cpu_governors"] == {
            "policy0": "performance",
            "policy4": "performance",
            "policy6": "performance",
        }

    def test_missing_cpu_policy_file_returns_sentinel(self, tmp_path, monkeypatch):
        monkeypatch.setattr("platform.machine", lambda: "aarch64")
        _write_fake_sysfs(tmp_path, skip={"policy4"})
        adapter = RKNNAdapter(precision="fp16")

        result = adapter.collect_clock_state(root=tmp_path)

        assert result is not None
        assert result["captured"] is False
        assert "policy4" in result["reason"] or "scaling_governor" in result["reason"]


# ---------------------------------------------------------------------------
# 7. collect_hardware_metrics(power_reading=...) — external meter wiring
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCollectHardwareMetricsExternalMeter:
    def test_no_power_reading_returns_none(self):
        """Both the base-class no-arg contract and 'no meter attached' land here."""
        adapter = RKNNAdapter(precision="fp16")
        assert adapter.collect_hardware_metrics() is None
        assert adapter.collect_hardware_metrics(None) is None

    def test_power_reading_populates_metrics_with_board_input_provenance(self):
        adapter = RKNNAdapter(precision="fp16")
        reading = PowerReading(
            watts_average=4.2, watts_peak=5.1, energy_joules_window=0.084, sample_count=20
        )

        metrics = adapter.collect_hardware_metrics(reading)

        assert metrics is not None
        assert metrics.watts_average == pytest.approx(4.2)
        assert metrics.watts_peak == pytest.approx(5.1)
        assert metrics.energy_joules_window == pytest.approx(0.084)
        assert metrics.power_source == "external_meter"
        assert metrics.power_domain == "board_input"
        # joules_per_inference is derived later in run_benchmark from p50,
        # not set here — collect_hardware_metrics has no latency to derive it from.
        assert metrics.joules_per_inference is None


# ---------------------------------------------------------------------------
# 8. run_benchmark — PowerSampler brackets exactly the measured loop
# ---------------------------------------------------------------------------


class _CountingRKNNLite:
    """Fake rknnlite.api.RKNNLite that counts total inference() calls so
    tests can assert exactly when a PowerSampler.start()/stop() happened
    relative to warmup vs. the measured loop."""

    call_count = 0  # class-level: shared across the single instance a test creates

    def load_rknn(self, path):
        return 0

    def init_runtime(self):
        return 0

    def inference(self, inputs):
        _CountingRKNNLite.call_count += 1
        return [np.ones((1, 1000), dtype=np.float32)]

    def release(self):
        pass


class _FakePowerSampler:
    """Records the inference call count at start()/stop() time, so a test
    can assert the bracket excludes warmup and covers exactly the measured
    iterations."""

    instances: ClassVar[list["_FakePowerSampler"]] = []

    def __init__(self):
        self.count_at_start: int | None = None
        self.count_at_stop: int | None = None
        _FakePowerSampler.instances.append(self)

    def start(self):
        self.count_at_start = _CountingRKNNLite.call_count
        return True

    def stop(self):
        self.count_at_stop = _CountingRKNNLite.call_count
        return PowerReading(
            watts_average=3.0, watts_peak=4.0, energy_joules_window=0.03, sample_count=5
        )


@pytest.mark.unit
class TestRunBenchmarkPowerSamplerBracketing:
    def test_sampler_brackets_measured_loop_excluding_warmup(self, rknn_cache_files, monkeypatch):
        _CountingRKNNLite.call_count = 0
        _FakePowerSampler.instances = []
        monkeypatch.setattr("calbredge_protocol.rknn_adapter.PowerSampler", _FakePowerSampler)

        rknnlite_api = types.ModuleType("rknnlite.api")
        rknnlite_api.RKNNLite = _CountingRKNNLite
        with _patch_rknnlite(rknnlite_api):
            adapter = RKNNAdapter(precision="fp16")
            adapter.load_model(rknn_cache_files, input_shape=[1, 224, 224, 3])

            proto = BenchmarkProtocol(
                protocol_version="1.0",
                input_shape=[1, 224, 224, 3],
                warmup_iterations=4,
                measurement_iterations=7,
            )
            inputs = {"0": np.zeros((1, 224, 224, 3), dtype=np.uint8)}
            adapter.run_benchmark(inputs, proto)

        assert len(_FakePowerSampler.instances) == 1
        sampler = _FakePowerSampler.instances[0]
        # load_model's fp16 correctness canary makes one inference() call of
        # its own before run_benchmark is even entered, so the absolute
        # count_at_start isn't just warmup_iterations — what matters is the
        # *interval* between start() and stop(): it must be exactly
        # measurement_iterations, covering none of the 4 warmup calls.
        assert sampler.count_at_stop - sampler.count_at_start == 7

        metrics = adapter._last_hardware_metrics
        assert metrics is not None
        assert metrics.power_source == "external_meter"
        assert metrics.power_domain == "board_input"
        assert metrics.energy_joules_window == pytest.approx(0.03)
        # joules_per_inference IS computed here (lines 824-836), from the
        # sampler's watts_average and this run's p50 — unaffected by the change.
        assert metrics.joules_per_inference is not None

    def test_no_meter_attached_hardware_metrics_stays_none(self, rknn_cache_files, monkeypatch):
        """PowerSampler.start() returning False (no meter) must not change
        run_benchmark's behaviour — the run completes and hardware_metrics is None,
        exactly as it was before the meter existed."""

        class _NoMeterSampler:
            def start(self):
                return False

            def stop(self):
                return None

        monkeypatch.setattr("calbredge_protocol.rknn_adapter.PowerSampler", _NoMeterSampler)

        rknnlite_api = types.ModuleType("rknnlite.api")
        rknnlite_api.RKNNLite = _CountingRKNNLite
        with _patch_rknnlite(rknnlite_api):
            adapter = RKNNAdapter(precision="fp16")
            adapter.load_model(rknn_cache_files, input_shape=[1, 224, 224, 3])

            proto = BenchmarkProtocol(
                protocol_version="1.0",
                input_shape=[1, 224, 224, 3],
                warmup_iterations=1,
                measurement_iterations=2,
            )
            inputs = {"0": np.zeros((1, 224, 224, 3), dtype=np.uint8)}
            stats = adapter.run_benchmark(inputs, proto)

        assert adapter._last_hardware_metrics is None
        assert stats.p50_ms >= 0.0


# ---------------------------------------------------------------------------
# 9. run_benchmark: the order of the timed loop
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTimedLoopOrder:
    """The timed loop's order: warm-up untimed and finished first; the power meter started
    after it; then, per measured iteration, t0, one inference, t1; the meter stopped and the
    telemetry collected only after the last timer read.

    Records the order of the clock reads, the inferences, the meter and the telemetry calls on
    fakes, with no change to the adapter. run_benchmark imports ``time`` inside the function,
    so the fake clock replaces ``time.perf_counter`` itself, only while run_benchmark runs.
    Fails if warm-up is timed, the timer stops before the inference, a second inference or a
    telemetry or meter call enters the timed region, or the meter starts before warm-up (each
    checked by mutating run_benchmark by hand when this test was written).

    Whether the real ``inference()`` returns only when the NPU has finished is a property of
    the device, checked on the RK3588 by tests/open/test_rknn_blocking_device.py.
    """

    WARMUP = 3
    MEASURED = 5

    def _run(self, rknn_cache_files, monkeypatch):
        events: list[str] = []
        ticks = iter(range(10_000))

        def clock() -> float:
            events.append("t")
            return float(next(ticks))

        class _RecordingRKNNLite(_CountingRKNNLite):
            def inference(self, inputs):
                events.append("inference")
                return [np.ones((1, 1000), dtype=np.float32)]

        class _RecordingSampler:
            def start(self):
                events.append("meter_start")
                return True

            def stop(self):
                events.append("meter_stop")
                return None

        monkeypatch.setattr("calbredge_protocol.rknn_adapter.PowerSampler", _RecordingSampler)
        rknnlite_api = types.ModuleType("rknnlite.api")
        rknnlite_api.RKNNLite = _RecordingRKNNLite
        with _patch_rknnlite(rknnlite_api):
            adapter = RKNNAdapter(precision="fp16")
            adapter.load_model(rknn_cache_files, input_shape=[1, 224, 224, 3])
            monkeypatch.setattr(
                adapter, "collect_hardware_metrics", lambda reading: events.append("telemetry")
            )
            monkeypatch.setattr(
                adapter, "collect_clock_state", lambda: events.append("clock_state")
            )
            proto = BenchmarkProtocol(
                protocol_version="1.0",
                input_shape=[1, 224, 224, 3],
                warmup_iterations=self.WARMUP,
                measurement_iterations=self.MEASURED,
            )
            inputs = {"0": np.zeros((1, 224, 224, 3), dtype=np.uint8)}
            events.clear()  # drop load_model's canary inference
            with monkeypatch.context() as during_run:
                during_run.setattr(time, "perf_counter", clock)
                stats = adapter.run_benchmark(inputs, proto)
        return events, stats

    def test_order(self, rknn_cache_files, monkeypatch):
        events, stats = self._run(rknn_cache_files, monkeypatch)

        first_t = events.index("t")
        last_t = len(events) - 1 - events[::-1].index("t")
        # 1. Warm-up is untimed and finished before the meter starts and the timer first reads.
        assert events[:first_t] == ["inference"] * self.WARMUP + ["meter_start"]
        # 2. Every measured iteration: t0, exactly one inference, t1, and nothing else.
        assert events[first_t : last_t + 1] == ["t", "inference", "t"] * self.MEASURED
        # 3. The meter stops and the telemetry is read only after the last timer read.
        assert events[last_t + 1 :] == ["meter_stop", "telemetry", "clock_state"]
        # 4. No other clock reads.
        assert events.count("t") == 2 * self.MEASURED
        # 5. The samples are exactly those brackets (consecutive fake readings, 1 s apart).
        assert stats.sample_count == self.MEASURED
        assert stats.samples_ms == (1000.0,) * self.MEASURED


# ---------------------------------------------------------------------------
# 10. load_model: the .rknn's build record
# ---------------------------------------------------------------------------


def _write_record(onnx_path: Path, precision: str = "fp16", **changes) -> dict:
    """The record `calbredge convert` would have written for this .onnx and its .rknn."""
    inputs = rknn_artifact.key_inputs(
        model_hash=rknn_artifact.file_sha256(onnx_path),
        precision=precision,
        target_platform="rk3588",
        model_inputs=[
            {
                "name": "input_layer_1",
                "source_shape": ["unk__654", 224, 224, 3],
                "source_dtype": "float",
                "source_layout": "nhwc",
                "converted_shape": [1, 3, 224, 224],
            }
        ],
        normalization={"mean_values": [[0, 0, 0]], "std_values": [[255, 255, 255]]},
        feed_dtype="uint8",
        graph_edits=["nhwc_to_nchw_input_transpose"],
        calibration_digest="d" * 64 if precision == "int8" else None,
        compiler_version="2.3.2",
        builder_config_version=1,
    )
    inputs.update(changes)
    return rknn_artifact.write_record(Path(f"{onnx_path}.rknn.{precision}.default"), inputs)


@pytest.mark.unit
class TestBuildRecordAtLoad:
    """A .rknn is run only if its build record says it was built from the .onnx beside it.

    The guard for "no silent substitution" on the RK3588: before this, the adapter loaded
    whatever file had the right name, so a model re-exported after conversion ran the old
    .rknn and its timings were recorded under the new model's hash.
    """

    def _load(self, onnx_path: Path, precision: str = "fp16") -> tuple[RKNNAdapter, list]:
        instances: list = []
        api = _make_rknnlite_mock([np.ones((1, 1000), dtype=np.float32)], instances=instances)
        adapter = RKNNAdapter(precision=precision)
        with _patch_rknnlite(api):
            adapter.load_model(onnx_path, input_shape=[1, 224, 224, 3])
        return adapter, instances

    def _refused(self, onnx_path: Path, precision: str = "fp16") -> str:
        instances: list = []
        api = _make_rknnlite_mock([np.ones((1, 1000), dtype=np.float32)], instances=instances)
        adapter = RKNNAdapter(precision=precision)
        with _patch_rknnlite(api), pytest.raises(ValueError, match="Refusing to load") as error:
            adapter.load_model(onnx_path, input_shape=[1, 224, 224, 3])
        assert instances == [], "the runtime must not be touched for a refused .rknn"
        return str(error.value)

    def test_a_verified_record_becomes_the_engine_build(self, rknn_cache_files, capsys):
        record = _write_record(rknn_cache_files)
        adapter, instances = self._load(rknn_cache_files)
        assert adapter._engine_build == {
            "build_key": record["build_key"],
            "built_from_model_hash": rknn_artifact.file_sha256(rknn_cache_files),
            "tensorrt_version": None,
            "calibration_digest": None,
            "conversion_input": {
                "feed_dtype": "uint8",
                "mean_values": [[0, 0, 0]],
                "std_values": [[255, 255, 255]],
                "inputs": [{"source_layout": "nhwc", "converted_shape": [1, 3, 224, 224]}],
                "graph_edits": ["nhwc_to_nchw_input_transpose"],
            },
        }
        assert len(instances) == 1
        out = capsys.readouterr().out
        key = record["build_key"][:16]
        assert f"Build record verified: model.onnx.rknn.fp16.default (build_key {key})" in out
        assert "Warning" not in out.split("Build record verified")[0]

    def test_an_int8_record_carries_its_calibration_digest(self, rknn_cache_files):
        _write_record(rknn_cache_files, precision="int8")
        adapter, _ = self._load(rknn_cache_files, precision="int8")
        assert adapter._engine_build["calibration_digest"] == "d" * 64

    def test_a_re_exported_onnx_beside_an_old_rknn_is_refused(self, rknn_cache_files):
        """The guard: the .onnx changed after conversion, the .rknn and its record did not."""
        _write_record(rknn_cache_files)
        rknn_cache_files.write_bytes(b"the same model, exported again")
        message = self._refused(rknn_cache_files)
        assert "model.onnx.rknn.fp16.default" in message
        assert "built from a different model than the .onnx beside it" in message

    def test_a_replaced_rknn_under_an_old_record_is_refused(self, rknn_cache_files):
        _write_record(rknn_cache_files)
        Path(f"{rknn_cache_files}.rknn.fp16.default").write_bytes(b"converted again, no record")
        assert "not the one its build record describes" in self._refused(rknn_cache_files)

    def test_an_edited_record_is_refused(self, rknn_cache_files):
        record = _write_record(rknn_cache_files)
        record["inputs"]["compiler_version"] = "9.9.9"
        path = rknn_artifact.record_path(Path(f"{rknn_cache_files}.rknn.fp16.default"))
        path.write_text(json.dumps(record))
        assert "do not hash to its build_key" in self._refused(rknn_cache_files)

    def test_a_record_for_another_precision_is_refused(self, rknn_cache_files):
        """An int8 artifact copied over the fp16 name, with its record."""
        _write_record(rknn_cache_files, precision="int8")
        fp16 = Path(f"{rknn_cache_files}.rknn.fp16.default")
        int8 = Path(f"{rknn_cache_files}.rknn.int8.default")
        fp16.write_bytes(int8.read_bytes())
        rknn_artifact.record_path(fp16).write_text(rknn_artifact.record_path(int8).read_text())
        assert "built for precision 'int8', this run needs 'fp16'" in self._refused(
            rknn_cache_files
        )

    def test_a_record_for_another_target_is_refused(self, rknn_cache_files):
        _write_record(rknn_cache_files, target_platform="rk3576")
        assert "built for target_platform 'rk3576', this run needs 'rk3588'" in self._refused(
            rknn_cache_files
        )

    def test_conversion_input_names_only_what_is_compared_in_canonical_order(
        self, rknn_cache_files
    ):
        """Not the input's name, source shape or dtype (io_signature has them), and not the
        toolkit version: the feed dtype, the normalization, each input's layout and converted
        shape, and the graph edits, sorted."""
        _write_record(rknn_cache_files, graph_edits=["z_edit", "a_edit", "z_edit"])
        adapter, _ = self._load(rknn_cache_files)
        conversion = adapter._engine_build["conversion_input"]
        assert set(conversion) == {
            "feed_dtype", "mean_values", "std_values", "inputs", "graph_edits",
        }  # fmt: skip
        assert set(conversion["inputs"][0]) == {"source_layout", "converted_shape"}
        assert conversion["graph_edits"] == ["a_edit", "z_edit"]
        ConversionInput.model_validate(conversion)

    @pytest.mark.parametrize(
        "missing", ["feed_dtype", "normalization", "model_inputs", "graph_edits"]
    )
    def test_a_record_that_does_not_say_how_inputs_were_converted_is_refused(
        self, rknn_cache_files, missing
    ):
        record = _write_record(rknn_cache_files)
        del record["inputs"][missing]
        record["build_key"] = rknn_artifact.compute_build_key(record["inputs"])
        path = rknn_artifact.record_path(Path(f"{rknn_cache_files}.rknn.fp16.default"))
        path.write_text(json.dumps(record))
        message = self._refused(rknn_cache_files)
        assert "does not say how the model's inputs were converted" in message
        assert missing in message

    def test_an_unreadable_record_is_refused_not_treated_as_missing(self, rknn_cache_files):
        path = rknn_artifact.record_path(Path(f"{rknn_cache_files}.rknn.fp16.default"))
        path.write_text("{truncated")
        assert "unreadable build record" in self._refused(rknn_cache_files)

    def test_a_record_without_the_onnx_to_check_it_against_is_refused(self, rknn_cache_files):
        _write_record(rknn_cache_files)
        rknn_cache_files.unlink()
        message = self._refused(rknn_cache_files)
        assert "model.onnx is needed beside it" in message

    def test_a_rknn_without_a_record_loads_with_a_warning_and_no_engine_build(
        self, rknn_cache_files, capsys
    ):
        """July's artifacts: they still run, and say plainly that no build is recorded."""
        adapter, instances = self._load(rknn_cache_files)
        assert adapter._engine_build is None
        assert len(instances) == 1
        out = capsys.readouterr().out
        assert "Warning: model.onnx.rknn.fp16.default has no build record" in out
        assert "its result records no build for it" in out
        assert "Build record verified" not in out

    def test_the_record_is_checked_before_the_runtime_loads_or_warms_up(self, rknn_cache_files):
        """Verification is part of the load, never of the timed loop."""
        _write_record(rknn_cache_files)
        events: list[str] = []
        real_verify = rknn_artifact.verify_record

        def verify(record, **kwargs):
            events.append("verify")
            return real_verify(record, **kwargs)

        api = _make_rknnlite_mock([np.ones((1, 1000), dtype=np.float32)])
        real_load = api.RKNNLite.load_rknn
        real_inference = api.RKNNLite.inference

        def load_rknn(self, path):
            events.append("load_rknn")
            return real_load(self, path)

        def inference(self, inputs):
            events.append("inference")
            return real_inference(self, inputs)

        api.RKNNLite.load_rknn = load_rknn
        api.RKNNLite.inference = inference
        adapter = RKNNAdapter(precision="fp16")
        with _patch_rknnlite(api), patch.object(rknn_artifact, "verify_record", verify):
            adapter.load_model(rknn_cache_files, input_shape=[1, 224, 224, 3])
        assert events[:2] == ["verify", "load_rknn"]
        assert events.count("verify") == 1
