"""Tests for agent/calbredge/cli.py's runtime-label resolution.

Root cause: cli.py used to relabel `onnxruntime` only (provider-based); both
TensorRTAdapter and RKNNAdapter expose a `runtime_label` property that was
never read, so e.g. bare `--runtime rknn` and `--runtime rknn-fp16` recorded
different runtime strings for the same fp16 precision. `_resolve_recorded_runtime`
is the fix; these tests exercise it directly (no hardware/SDK needed — neither
TensorRTAdapter nor RKNNAdapter imports its real backend at construction time,
only inside lazy `_import_*` helpers called from `load_model`/engine-build code
paths) plus one CLI-level regression test proving the fix is actually wired in.
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from calbredge.cli import _BARE_RUNTIME_DEFAULTS, _resolve_recorded_runtime, main
from calbredge_protocol.mock_adapter import MockRuntimeAdapter
from calbredge_protocol.rknn_adapter import RKNNAdapter
from calbredge_protocol.tensorrt_adapter import TensorRTAdapter
from click.testing import CliRunner

from tests.open.test_rknn_adapter import _make_rknnlite_mock, _patch_rknnlite

FAST_PROTOCOL = str(Path(__file__).parent / "fixtures" / "protocol_fast.yaml")


class _StubAdapter:
    """Minimal stand-in for ONNXRuntimeAdapter/TFLiteAdapter/MockRuntimeAdapter:
    no `runtime_label` property, optionally an `_active_provider` attribute."""

    def __init__(self, active_provider=None):
        if active_provider is not None:
            self._active_provider = active_provider


# ---------------------------------------------------------------------------
# Pinned strings — TensorRT/RKNN adapters relabel via their own runtime_label.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveRecordedRuntimePinnedStrings:
    def test_bare_rknn_becomes_rknn_fp16(self):
        """RKNNAdapter's bare-form default precision is fp16 (autodetect.py's
        dispatch table: partial(RKNNAdapter, precision="fp16"))."""
        adapter = RKNNAdapter(precision="fp16")
        assert _resolve_recorded_runtime("rknn", "rknn", adapter) == "rknn-fp16"

    def test_explicit_rknn_fp16_stays_rknn_fp16(self):
        adapter = RKNNAdapter(precision="fp16")
        assert _resolve_recorded_runtime("rknn-fp16", "rknn-fp16", adapter) == "rknn-fp16"

    def test_explicit_rknn_int8_stays_rknn_int8(self):
        adapter = RKNNAdapter(precision="int8")
        assert _resolve_recorded_runtime("rknn-int8", "rknn-int8", adapter) == "rknn-int8"

    def test_bare_tensorrt_becomes_tensorrt_fp32_not_fp16(self):
        """TensorRTAdapter's bare-form default precision is fp32
        (TensorRTAdapter.__init__(self, precision: str = "fp32")), asymmetric
        with RKNN's fp16 default — confirmed via source read, not assumed."""
        adapter = TensorRTAdapter()
        assert _resolve_recorded_runtime("tensorrt", "tensorrt", adapter) == "tensorrt-fp32"

    def test_explicit_tensorrt_fp16_stays_tensorrt_fp16(self):
        adapter = TensorRTAdapter(precision="fp16")
        assert (
            _resolve_recorded_runtime("tensorrt-fp16", "tensorrt-fp16", adapter) == "tensorrt-fp16"
        )

    def test_explicit_tensorrt_int8_stays_tensorrt_int8(self):
        adapter = TensorRTAdapter(precision="int8")
        assert (
            _resolve_recorded_runtime("tensorrt-int8", "tensorrt-int8", adapter) == "tensorrt-int8"
        )

    def test_autodetected_requested_is_none(self):
        """When --runtime was omitted and detect_runtime() picked the adapter,
        `requested` is None — must still relabel via runtime_label, and must
        never trip the conflict check (None is exempt, not just the bare
        defaults)."""
        adapter = RKNNAdapter(precision="int8")
        assert _resolve_recorded_runtime(None, "rknn", adapter) == "rknn-int8"


# ---------------------------------------------------------------------------
# onnxruntime special case — must be byte-identical to the pre-fix behaviour.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveRecordedRuntimeOnnxruntimeUnchanged:
    def test_cuda_provider(self):
        adapter = _StubAdapter(active_provider="CUDAExecutionProvider")
        assert (
            _resolve_recorded_runtime("onnxruntime", "onnxruntime", adapter) == "onnxruntime-cuda"
        )

    def test_cpu_provider(self):
        adapter = _StubAdapter(active_provider="CPUExecutionProvider")
        assert _resolve_recorded_runtime("onnxruntime", "onnxruntime", adapter) == "onnxruntime-cpu"

    def test_no_active_provider_attribute(self):
        """No _active_provider at all (e.g. load failed before it was set) —
        falls through unchanged, exactly as the pre-fix `if ... and
        active_provider:` guard did."""
        adapter = _StubAdapter()
        assert _resolve_recorded_runtime("onnxruntime", "onnxruntime", adapter) == "onnxruntime"

    def test_unrecognized_provider_value(self):
        adapter = _StubAdapter(active_provider="SomeOtherProvider")
        assert _resolve_recorded_runtime("onnxruntime", "onnxruntime", adapter) == "onnxruntime"


@pytest.mark.unit
class TestResolveRecordedRuntimeOtherNoLabelAdapters:
    def test_tflite_passes_through_unchanged(self):
        adapter = _StubAdapter()
        assert _resolve_recorded_runtime("tflite", "tflite", adapter) == "tflite"

    def test_mock_passes_through_unchanged(self):
        adapter = _StubAdapter()
        assert _resolve_recorded_runtime("mock", "mock", adapter) == "mock"


# ---------------------------------------------------------------------------
# Conflict rule: a precision-qualified --runtime disagreeing with the
# adapter's own runtime_label is a configuration error, never silently
# resolved. NOTE: under the current _RUNTIME_TO_ADAPTER dispatch table this
# can never actually be triggered through get_adapter() — every registered
# qualified key is constructed with exactly the precision its name says, so
# requested always equals label for real dispatch. This test constructs the
# mismatch directly as a forward-guard against the dispatch table drifting
# out of sync with itself later.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveRecordedRuntimeConflict:
    def test_mismatched_precision_raises(self):
        import click

        adapter = TensorRTAdapter(precision="fp16")  # runtime_label == "tensorrt-fp16"
        with pytest.raises(click.ClickException, match="tensorrt-int8"):
            _resolve_recorded_runtime("tensorrt-int8", "tensorrt-int8", adapter)

    def test_bare_defaults_are_exempt_from_the_conflict_check(self):
        """Bare 'rknn'/'tensorrt' are meant to be silently filled in — that is
        the whole point of the fix, not a conflict."""
        for bare in _BARE_RUNTIME_DEFAULTS:
            assert bare in ("rknn", "tensorrt")
        adapter = RKNNAdapter(precision="int8")
        # requested="rknn" (bare) vs runtime_label="rknn-int8": no raise.
        assert _resolve_recorded_runtime("rknn", "rknn", adapter) == "rknn-int8"


# ---------------------------------------------------------------------------
# CLI-level regression: the runtime stored in the printed result equals the
# adapter's own runtime_label. This is the invariant the walking skeleton's
# compatibility check will depend on. RKNN only — TensorRT requires mocking
# the tensorrt/pycuda modules deep enough to complete a full engine build
# (see test_tensorrt_adapter.py), and this environment's own existing test
# (test_agent_e2e.py::test_unsupported_runtime) already documents that a bare
# `--runtime tensorrt` CLI invocation fails before reaching this code path
# here (no tensorrt package installed) — that failure path never calls
# _resolve_recorded_runtime, so a CLI-level TensorRT case would test nothing
# beyond what TestResolveRecordedRuntimePinnedStrings already covers directly.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRknnCliRuntimeLabelMatchesStoredResult:
    def _run(self, tmp_path, cli_runtime_flag):
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"fake onnx bytes")
        precision = "int8" if "int8" in cli_runtime_flag else "fp16"
        (tmp_path / f"model.onnx.rknn.{precision}.default").write_bytes(b"fake rknn cache")

        healthy_output = [np.ones((1, 1000), dtype=np.float32)]
        rknnlite_api = _make_rknnlite_mock(healthy_output)

        runner = CliRunner()
        with _patch_rknnlite(rknnlite_api):
            result = runner.invoke(
                main,
                [
                    "run",
                    "--model",
                    str(onnx_path),
                    "--runtime",
                    cli_runtime_flag,
                    "--protocol",
                    FAST_PROTOCOL,
                    "--dry-run",
                ],
            )
        assert result.exit_code == 0, (
            f"CLI exited with code {result.exit_code}.\nOutput:\n{result.output}"
        )
        return json.loads(result.output[result.output.index("{") : result.output.rindex("}") + 1])

    def test_bare_rknn_records_rknn_fp16(self, tmp_path):
        output = self._run(tmp_path, "rknn")
        adapter = RKNNAdapter(precision="fp16")
        assert output["runtime"] == adapter.runtime_label == "rknn-fp16"

    def test_explicit_rknn_int8_records_rknn_int8(self, tmp_path):
        output = self._run(tmp_path, "rknn-int8")
        adapter = RKNNAdapter(precision="int8")
        assert output["runtime"] == adapter.runtime_label == "rknn-int8"


# ---------------------------------------------------------------------------
# _execute_benchmark extraction: adapter.unload()
# must still be called on a mid-run failure, exactly as the pre-extraction
# inline code did via its `adapter` closure variable. No existing test
# covered this before the refactor — this is that test.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterUnloadCalledOnMidRunException:
    def test_unload_called_when_measurement_loop_raises(self, tmp_path):
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"fake onnx bytes")

        unload_calls: list = []
        original_unload = MockRuntimeAdapter.unload

        def spy_unload(self):
            unload_calls.append(self)
            original_unload(self)

        def raising_run_inference(self, inputs):
            raise RuntimeError("synthetic mid-run failure")

        runner = CliRunner()
        with (
            patch.object(MockRuntimeAdapter, "unload", spy_unload),
            patch.object(MockRuntimeAdapter, "run_inference", raising_run_inference),
        ):
            result = runner.invoke(
                main,
                ["run", "--model", str(onnx_path), "--runtime", "mock", "--dry-run"],
            )

        assert result.exit_code == 1
        assert "Benchmark failed" in result.output
        assert len(unload_calls) == 1, (
            "adapter.unload() must be called from run_benchmark's finally block "
            "even when the measurement loop raises mid-run"
        )
