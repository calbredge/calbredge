"""Unit tests for HardwareMetrics model and CLI summary formatting.

All tests use only in-process logic — no hardware, runtime libraries, or
network access required.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from calbredge.cli import _clock_summary_line, _hardware_summary_lines
from calbredge_protocol import BenchmarkResult, HardwareMetrics
from calbredge_protocol.adapters import RuntimeAdapter
from pydantic import ValidationError

from tests.open.fake_hosts import build_jetson_orin, build_x86

# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing RuntimeAdapter.collect_hardware_metrics
# ---------------------------------------------------------------------------


class _StubAdapter(RuntimeAdapter):
    def load_model(self, model_path):
        pass

    def warmup(self, inputs, iterations):
        pass

    def run_inference(self, inputs):
        return {}

    def unload(self):
        pass


_SAMPLE_TEGRASTATS = (
    "RAM 1930/7620MB CPU [0%@729] EMC_FREQ 0% GR3D_FREQ 0% "
    "cpu@41.8C gpu@43.1C VDD_IN 3059mW/3059mW "
    "VDD_CPU_GPU_CV 482mW/482mW VDD_SOC 1006mW/1006mW\n"
    "RAM 1932/7620MB CPU [2%@729] EMC_FREQ 0% GR3D_FREQ 0% "
    "cpu@42.0C gpu@43.5C VDD_IN 3100mW/3100mW "
    "VDD_CPU_GPU_CV 490mW/490mW VDD_SOC 1010mW/1010mW\n"
)


def _popen_mock(output: str) -> MagicMock:
    proc = MagicMock()
    proc.communicate.return_value = (output, "")
    return proc


# ---------------------------------------------------------------------------
# HardwareMetrics model
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHardwareMetrics:
    def test_all_none_values(self):
        """HardwareMetrics with all fields None is valid."""
        hm = HardwareMetrics()
        assert hm.watts_average is None
        assert hm.watts_peak is None
        assert hm.joules_per_inference is None
        assert hm.temperature_start_celsius is None
        assert hm.temperature_end_celsius is None
        assert hm.temperature_peak_celsius is None
        assert hm.thermal_throttle_detected is None
        assert hm.gpu_memory_used_mb is None
        assert hm.gpu_memory_total_mb is None
        assert hm.system_memory_used_mb is None
        assert hm.power_mode is None
        assert hm.precision is None
        assert hm.power_source is None

    def test_validates_float_fields_correctly(self):
        """Float fields accept numeric values and store them correctly."""
        hm = HardwareMetrics(
            watts_average=15.5,
            watts_peak=20.0,
            joules_per_inference=0.0155,
            temperature_start_celsius=35.2,
            temperature_end_celsius=67.8,
            temperature_peak_celsius=72.1,
            gpu_memory_used_mb=4096.0,
            gpu_memory_total_mb=8192.0,
            system_memory_used_mb=2048.5,
        )
        assert hm.watts_average == pytest.approx(15.5)
        assert hm.watts_peak == pytest.approx(20.0)
        assert hm.joules_per_inference == pytest.approx(0.0155)
        assert hm.temperature_start_celsius == pytest.approx(35.2)
        assert hm.temperature_end_celsius == pytest.approx(67.8)
        assert hm.temperature_peak_celsius == pytest.approx(72.1)
        assert hm.gpu_memory_used_mb == pytest.approx(4096.0)
        assert hm.gpu_memory_total_mb == pytest.approx(8192.0)
        assert hm.system_memory_used_mb == pytest.approx(2048.5)

    def test_bool_field_accepted(self):
        """thermal_throttle_detected accepts bool values."""
        hm_true = HardwareMetrics(thermal_throttle_detected=True)
        hm_false = HardwareMetrics(thermal_throttle_detected=False)
        assert hm_true.thermal_throttle_detected is True
        assert hm_false.thermal_throttle_detected is False

    def test_string_fields_accepted(self):
        """power_mode, precision, power_source accept string values."""
        hm = HardwareMetrics(
            power_mode="max_performance",
            precision="fp16",
            power_source="tegrastats",
        )
        assert hm.power_mode == "max_performance"
        assert hm.precision == "fp16"
        assert hm.power_source == "tegrastats"

    def test_extra_fields_rejected(self):
        """extra='forbid' rejects unknown fields."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            HardwareMetrics(unknown_field="oops")

    def test_frozen(self):
        """Instances are immutable (frozen=True)."""
        hm = HardwareMetrics(watts_average=10.0)
        with pytest.raises(Exception):
            hm.watts_average = 20.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BenchmarkResult + HardwareMetrics integration
# ---------------------------------------------------------------------------

_BASE_RESULT_KWARGS = {
    "protocol_version": "1.0",
    "hardware_id": "a" * 64,
    "runtime": "onnxruntime",
    "model_hash": "b" * 64,
    "latency_p50_ms": 5.0,
    "latency_p90_ms": 7.0,
    "latency_p99_ms": 9.0,
    "latency_mean_ms": 5.5,
    "latency_std_ms": 0.8,
    "throughput_ips": 200.0,
    "sample_count": 100,
    "system_metadata": {"hostname": "test-device"},
}


@pytest.mark.unit
class TestBenchmarkResultWithHardwareMetrics:
    def test_accepts_hardware_metrics_none(self):
        """BenchmarkResult with hardware_metrics=None is valid."""
        result = BenchmarkResult(**_BASE_RESULT_KWARGS, hardware_metrics=None)
        assert result.hardware_metrics is None

    def test_hardware_metrics_defaults_to_none(self):
        """hardware_metrics defaults to None when omitted."""
        result = BenchmarkResult(**_BASE_RESULT_KWARGS)
        assert result.hardware_metrics is None

    def test_accepts_fully_populated_hardware_metrics(self):
        """BenchmarkResult accepts a fully populated HardwareMetrics."""
        hm = HardwareMetrics(
            watts_average=12.3,
            watts_peak=18.0,
            joules_per_inference=0.012,
            temperature_start_celsius=40.0,
            temperature_end_celsius=65.0,
            temperature_peak_celsius=68.5,
            thermal_throttle_detected=False,
            gpu_memory_used_mb=2048.0,
            gpu_memory_total_mb=4096.0,
            system_memory_used_mb=1024.0,
            power_mode="max_performance",
            precision="fp16",
            power_source="tegrastats",
        )
        result = BenchmarkResult(**_BASE_RESULT_KWARGS, hardware_metrics=hm)
        assert result.hardware_metrics is hm
        assert result.hardware_metrics.watts_average == pytest.approx(12.3)
        assert result.hardware_metrics.power_source == "tegrastats"

    def test_backward_compatible_json_without_hardware_metrics(self):
        """Existing benchmark JSON without hardware_metrics still parses correctly."""
        payload = {
            **_BASE_RESULT_KWARGS,
            "run_id": "11111111-1111-1111-1111-111111111111",
            "timestamp": "2026-01-01T00:00:00Z",
        }
        # No hardware_metrics key at all
        assert "hardware_metrics" not in payload
        result = BenchmarkResult(**payload)
        assert result.hardware_metrics is None

    def test_json_round_trip_with_hardware_metrics(self):
        """BenchmarkResult serialises and deserialises hardware_metrics correctly."""
        hm = HardwareMetrics(watts_average=10.0, power_source="nvidia-smi")
        result = BenchmarkResult(**_BASE_RESULT_KWARGS, hardware_metrics=hm)
        json_str = result.model_dump_json()
        data = json.loads(json_str)

        assert data["hardware_metrics"]["watts_average"] == pytest.approx(10.0)
        assert data["hardware_metrics"]["power_source"] == "nvidia-smi"
        assert data["hardware_metrics"]["watts_peak"] is None

        # Round-trip
        restored = BenchmarkResult.model_validate_json(json_str)
        assert restored.hardware_metrics is not None
        assert restored.hardware_metrics.watts_average == pytest.approx(10.0)

    def test_json_round_trip_hardware_metrics_none(self):
        """BenchmarkResult with hardware_metrics=None serialises to null."""
        result = BenchmarkResult(**_BASE_RESULT_KWARGS)
        data = json.loads(result.model_dump_json())
        assert data["hardware_metrics"] is None


# ---------------------------------------------------------------------------
# CLI summary helper: _hardware_summary_lines
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHardwareSummaryLines:
    def test_returns_empty_list_when_none(self):
        """No metrics → no lines printed."""
        assert _hardware_summary_lines(None) == []

    def test_includes_hardware_section_when_metrics_present(self):
        """A populated HardwareMetrics produces at least one line."""
        hm = HardwareMetrics(watts_average=15.0, power_source="tegrastats")
        lines = _hardware_summary_lines(hm)
        assert len(lines) > 0

    def test_power_line_present_when_watts_average_set(self):
        """Power line appears when watts_average is set."""
        hm = HardwareMetrics(watts_average=12.5)
        lines = _hardware_summary_lines(hm)
        power_lines = [line for line in lines if "Power:" in line]
        assert len(power_lines) == 1
        assert "12.5W avg" in power_lines[0]

    def test_watts_average_none_skips_power_line(self):
        """When watts_average is None, the Power line is not emitted
        even if other hardware fields are present."""
        hm = HardwareMetrics(
            watts_average=None,
            watts_peak=None,
            joules_per_inference=None,
            temperature_peak_celsius=70.0,
        )
        lines = _hardware_summary_lines(hm)
        power_lines = [line for line in lines if "Power:" in line]
        assert len(power_lines) == 0

    def test_excludes_hardware_section_when_all_none(self):
        """All-None HardwareMetrics produces no lines (no output at all)."""
        hm = HardwareMetrics()
        lines = _hardware_summary_lines(hm)
        assert lines == []

    def test_thermal_line_with_start_celsius_only(self):
        """Thermal line prints correctly with only temperature_start_celsius."""
        hm = HardwareMetrics(temperature_start_celsius=35.0)
        lines = _hardware_summary_lines(hm)
        thermal_lines = [line for line in lines if "Thermal:" in line]
        assert len(thermal_lines) == 1
        assert "35.0°C →" in thermal_lines[0]
        # peak should not appear since it's None
        assert "peak" not in thermal_lines[0]

    def test_thermal_line_with_start_and_peak(self):
        """Thermal line shows start → peak when both are set."""
        hm = HardwareMetrics(
            temperature_start_celsius=40.0,
            temperature_peak_celsius=72.0,
        )
        lines = _hardware_summary_lines(hm)
        thermal_lines = [line for line in lines if "Thermal:" in line]
        assert len(thermal_lines) == 1
        assert "40.0°C →" in thermal_lines[0]
        assert "72.0°C peak" in thermal_lines[0]

    def test_memory_line_gpu_and_ram(self):
        """Memory line includes both GPU and RAM when both are set."""
        hm = HardwareMetrics(
            gpu_memory_used_mb=4096.0,
            system_memory_used_mb=2048.0,
        )
        lines = _hardware_summary_lines(hm)
        memory_lines = [line for line in lines if "Memory:" in line]
        assert len(memory_lines) == 1
        assert "4096.0MB GPU" in memory_lines[0]
        assert "2048.0MB RAM" in memory_lines[0]

    def test_config_line_with_power_source(self):
        """Config line includes power source when set."""
        hm = HardwareMetrics(power_source="nvidia-smi")
        lines = _hardware_summary_lines(hm)
        config_lines = [line for line in lines if "Config:" in line]
        assert len(config_lines) == 1
        assert "measured via nvidia-smi" in config_lines[0]

    def test_full_metrics_produces_all_four_lines(self):
        """All four summary lines appear when all fields are populated."""
        hm = HardwareMetrics(
            watts_average=15.0,
            watts_peak=20.0,
            joules_per_inference=0.015,
            temperature_start_celsius=38.0,
            temperature_peak_celsius=70.0,
            gpu_memory_used_mb=4096.0,
            system_memory_used_mb=1024.0,
            power_mode="max_performance",
            precision="fp16",
            power_source="tegrastats",
        )
        lines = _hardware_summary_lines(hm)
        labels = [line.split(":")[0].strip() for line in lines]
        assert "Power" in labels
        assert "Thermal" in labels
        assert "Memory" in labels
        assert "Config" in labels


@pytest.mark.unit
class TestPostLoopPowerIsProvisional:
    """Power read once after the timed loop is labelled provisional in the console, rounded to
    two decimals, and J/inference (that reading times the p50) is left out; the values
    themselves are unchanged."""

    def _hm(self) -> HardwareMetrics:
        return HardwareMetrics(
            watts_average=7.5432,
            watts_peak=12.987,
            joules_per_inference=0.0335,
            temperature_peak_celsius=51.2,
            power_source="tegrastats",
        )

    def test_post_loop_power_is_labelled_and_has_no_j_per_inference(self):
        hm = self._hm()
        lines = _hardware_summary_lines(hm, "post_loop_snapshot")
        [power] = [line for line in lines if "Power:" in line]
        assert power == (
            "  Power:   7.54 W avg, 12.99 W peak "
            "(provisional: read after the timed loop, not during it)"
        )
        assert not any("J/inference" in line for line in lines)
        assert any("Thermal:" in line for line in lines)  # the other lines stay
        assert hm.joules_per_inference == 0.0335  # the recorded value is untouched

    def test_post_loop_without_watts_prints_no_power_line(self):
        hm = HardwareMetrics(temperature_peak_celsius=51.2)
        lines = _hardware_summary_lines(hm, "post_loop_snapshot")
        assert not any("Power:" in line for line in lines)

    @pytest.mark.parametrize("boundary", [None, "windowed_bracket"])
    def test_other_boundaries_print_as_before(self, boundary):
        lines = _hardware_summary_lines(self._hm(), boundary)
        [power] = [line for line in lines if "Power:" in line]
        assert power == "  Power:   7.5432W avg · 12.987W peak · 0.0335J/inference"
        assert "provisional" not in power


@pytest.mark.unit
class TestClockSummaryLine:
    def test_pinned_jetson(self, tmp_path):
        root = build_jetson_orin(tmp_path / "root", pinned=True)
        assert _clock_summary_line(root) == "  Clocks      pinned (jetson)"

    def test_unpinned_jetson(self, tmp_path):
        root = build_jetson_orin(tmp_path / "root", pinned=False)
        assert _clock_summary_line(root) == (
            "  Clocks      not pinned (jetson): timings may move; calbredge doctor shows the fix"
        )

    @pytest.mark.parametrize("governor", ["performance", "powersave"])
    def test_x86_is_not_determined(self, tmp_path, governor):
        root = build_x86(tmp_path / "root", governor=governor)
        assert _clock_summary_line(root) == (
            "  Clocks      not determined (linux_cpufreq): the CPU's speed can still vary; "
            "compare with care"
        )

    def test_no_collector_is_unknown(self, tmp_path):
        (tmp_path / "root").mkdir()
        line = _clock_summary_line(tmp_path / "root")
        assert line.startswith("  Clocks      unknown (")


# ---------------------------------------------------------------------------
# collect_hardware_metrics() — tegrastats parsing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCollectHardwareMetrics:
    def _run(self, output: str) -> HardwareMetrics | None:
        adapter = _StubAdapter()
        with (
            patch.object(RuntimeAdapter, "_is_jetson", return_value=True),
            patch("calbredge_protocol.adapters.subprocess.Popen", return_value=_popen_mock(output)),
            patch("calbredge_protocol.adapters.subprocess.run", side_effect=FileNotFoundError),
            patch("time.sleep"),
        ):
            return adapter.collect_hardware_metrics()

    def test_returns_none_on_non_jetson(self):
        adapter = _StubAdapter()
        with patch.object(RuntimeAdapter, "_is_jetson", return_value=False):
            assert adapter.collect_hardware_metrics() is None

    def test_returns_none_when_tegrastats_missing(self):
        adapter = _StubAdapter()
        with (
            patch.object(RuntimeAdapter, "_is_jetson", return_value=True),
            patch("calbredge_protocol.adapters.subprocess.Popen", side_effect=FileNotFoundError),
            patch("time.sleep"),
        ):
            assert adapter.collect_hardware_metrics() is None

    def test_returns_none_on_empty_output(self):
        assert self._run("") is None

    def test_returns_none_on_unparseable_output(self):
        assert self._run("some random garbage\n") is None

    def test_parses_watts_average_and_peak(self):
        result = self._run(_SAMPLE_TEGRASTATS)
        assert result is not None
        # avg of 3.059 and 3.100
        assert result.watts_average == pytest.approx(3.0795, rel=1e-3)
        assert result.watts_peak == pytest.approx(3.100, rel=1e-3)

    def test_parses_gpu_temperature_peak(self):
        result = self._run(_SAMPLE_TEGRASTATS)
        assert result is not None
        assert result.temperature_peak_celsius == pytest.approx(43.5)

    def test_no_throttle_below_85c(self):
        result = self._run(_SAMPLE_TEGRASTATS)
        assert result is not None
        assert result.thermal_throttle_detected is False

    def test_throttle_detected_above_85c(self):
        hot = "RAM 1930/7620MB cpu@86.0C gpu@86.5C VDD_IN 5000mW/5000mW\n"
        result = self._run(hot)
        assert result is not None
        assert result.thermal_throttle_detected is True
        assert result.temperature_peak_celsius == pytest.approx(86.5)

    def test_cpu_temp_used_when_no_gpu_temp(self):
        cpu_only = "RAM 1930/7620MB cpu@55.0C VDD_IN 2000mW/2000mW\n"
        result = self._run(cpu_only)
        assert result is not None
        assert result.temperature_peak_celsius == pytest.approx(55.0)

    def test_power_source_is_tegrastats(self):
        result = self._run(_SAMPLE_TEGRASTATS)
        assert result is not None
        assert result.power_source == "tegrastats"

    def test_returns_hardware_metrics_instance(self):
        result = self._run(_SAMPLE_TEGRASTATS)
        assert isinstance(result, HardwareMetrics)

    def test_power_mode_populated_when_nvpmodel_available(self):
        """power_mode is set in HardwareMetrics when nvpmodel reports a mode."""
        nvp_result = MagicMock()
        nvp_result.stdout = "NV Power Mode: MAXN_SUPER\n0\n"
        adapter = _StubAdapter()
        with (
            patch.object(RuntimeAdapter, "_is_jetson", return_value=True),
            patch("subprocess.Popen", return_value=_popen_mock(_SAMPLE_TEGRASTATS)),
            patch("calbredge_protocol.adapters.subprocess.run", return_value=nvp_result),
            patch("time.sleep"),
        ):
            result = adapter.collect_hardware_metrics()
        assert result is not None
        assert result.power_mode == "maxn_super"

    def test_power_mode_none_when_nvpmodel_unavailable(self):
        """power_mode is None when nvpmodel is not installed on Jetson."""
        adapter = _StubAdapter()
        with (
            patch.object(RuntimeAdapter, "_is_jetson", return_value=True),
            patch("subprocess.Popen", return_value=_popen_mock(_SAMPLE_TEGRASTATS)),
            patch("calbredge_protocol.adapters.subprocess.run", side_effect=FileNotFoundError),
            patch("time.sleep"),
        ):
            result = adapter.collect_hardware_metrics()
        assert result is not None
        assert result.power_mode is None

    def test_power_mode_none_on_non_jetson(self):
        """collect_hardware_metrics returns None entirely on non-Jetson hardware."""
        adapter = _StubAdapter()
        with patch.object(RuntimeAdapter, "_is_jetson", return_value=False):
            result = adapter.collect_hardware_metrics()
        assert result is None


# ---------------------------------------------------------------------------
# joules_per_inference calculation in ONNXRuntimeAdapter.run_benchmark()
# ---------------------------------------------------------------------------


def _make_onnx_adapter(hm: HardwareMetrics | None):
    """Return an ONNXRuntimeAdapter subclass with session-dependent methods stubbed out."""
    from calbredge_protocol.onnx_adapter import ONNXRuntimeAdapter

    class _Inner(ONNXRuntimeAdapter):
        def warmup(self, inputs, iterations):
            pass

        def run_inference(self, inputs):
            return {}

        def collect_hardware_metrics(self_inner):
            return hm

    return _Inner()


def _protocol():
    from calbredge_protocol.models import BenchmarkProtocol

    return BenchmarkProtocol(
        protocol_version="1.0",
        warmup_iterations=0,
        measurement_iterations=10,
        input_shape=[1, 3, 224, 224],
    )


@pytest.mark.unit
class TestJoulesPerInference:
    def test_joules_calculated_from_watts_and_p50(self):
        hm = HardwareMetrics(watts_average=10.0, power_source="tegrastats")
        adapter = _make_onnx_adapter(hm)
        stats = adapter.run_benchmark({}, _protocol())
        assert adapter._last_hardware_metrics is not None
        expected = 10.0 * (stats.p50_ms / 1000.0)
        assert adapter._last_hardware_metrics.joules_per_inference == pytest.approx(expected)

    def test_joules_none_when_watts_not_available(self):
        hm = HardwareMetrics(temperature_peak_celsius=45.0, power_source="tegrastats")
        adapter = _make_onnx_adapter(hm)
        adapter.run_benchmark({}, _protocol())
        assert adapter._last_hardware_metrics is not None
        assert adapter._last_hardware_metrics.joules_per_inference is None

    def test_joules_none_when_no_hardware_metrics(self):
        adapter = _make_onnx_adapter(None)
        adapter.run_benchmark({}, _protocol())
        assert adapter._last_hardware_metrics is None
