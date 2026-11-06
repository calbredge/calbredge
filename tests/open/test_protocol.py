"""Unit tests for the protocol package.

All tests use only in-process logic or the canonical v1.0 protocol YAML.
No runtime adapters, hardware, or network access is required.
"""

from pathlib import Path

import numpy as np
import pytest
from calbredge_protocol import (
    BenchmarkProtocol,
    BenchmarkResult,
    LatencyStats,
    MockRuntimeAdapter,
    ONNXRuntimeAdapter,
    compute_latency_stats,
    compute_throughput,
    detect_runtime,
    get_adapter,
    load_protocol,
)
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

V1_YAML = Path(__file__).parents[2] / "protocol" / "calbredge_protocol" / "protocols" / "v1.0.yaml"


# ---------------------------------------------------------------------------
# BenchmarkProtocol
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBenchmarkProtocol:
    def test_valid_defaults(self):
        """Required fields only — all defaults must be applied."""
        proto = BenchmarkProtocol(protocol_version="1.0", input_shape=[1, 3, 224, 224])

        assert proto.protocol_version == "1.0"
        assert proto.warmup_iterations == 50
        assert proto.measurement_iterations == 500
        assert proto.batch_size == 1
        assert proto.input_shape == [1, 3, 224, 224]
        assert proto.power_mode == "default"
        assert proto.runtime_config == {}

    def test_valid_custom_values(self):
        """All fields supplied explicitly."""
        proto = BenchmarkProtocol(
            protocol_version="2.0",
            warmup_iterations=5,
            measurement_iterations=50,
            batch_size=4,
            input_shape=[4, 1, 28, 28],
            power_mode="max_performance",
            runtime_config={"execution_provider": "CUDAExecutionProvider"},
        )

        assert proto.protocol_version == "2.0"
        assert proto.warmup_iterations == 5
        assert proto.measurement_iterations == 50
        assert proto.batch_size == 4
        assert proto.input_shape == [4, 1, 28, 28]
        assert proto.power_mode == "max_performance"
        assert proto.runtime_config == {"execution_provider": "CUDAExecutionProvider"}

    def test_measurement_iterations_zero_raises(self):
        with pytest.raises(ValidationError, match="measurement_iterations must be > 0"):
            BenchmarkProtocol(
                protocol_version="1.0",
                input_shape=[1, 3, 224, 224],
                measurement_iterations=0,
            )

    def test_empty_input_shape_raises(self):
        with pytest.raises(ValidationError, match="input_shape must not be empty"):
            BenchmarkProtocol(protocol_version="1.0", input_shape=[])

    def test_unknown_field_raises(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            BenchmarkProtocol(
                protocol_version="1.0",
                input_shape=[1, 3, 224, 224],
                unknown_field="oops",
            )


# ---------------------------------------------------------------------------
# load_protocol
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadProtocol:
    def test_loads_v1_yaml(self):
        """Canonical v1.0 YAML round-trips to exact field values."""
        proto = load_protocol(V1_YAML)

        assert isinstance(proto, BenchmarkProtocol)
        assert proto.protocol_version == "1.0"
        assert proto.warmup_iterations == 50
        assert proto.measurement_iterations == 500
        assert proto.batch_size == 1
        assert proto.input_shape == [1, 3, 224, 224]
        assert proto.power_mode == "default"
        assert proto.runtime_config == {}

    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError, match="Protocol file not found"):
            load_protocol(missing)

    def test_malformed_yaml_raises(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("key: [unclosed bracket\n")
        with pytest.raises(ValueError, match="Malformed YAML"):
            load_protocol(bad_yaml)

    def test_missing_required_field_raises(self, tmp_path):
        """A YAML that omits protocol_version should raise ValueError."""
        incomplete = tmp_path / "incomplete.yaml"
        incomplete.write_text("input_shape: [1, 3, 224, 224]\n")
        with pytest.raises(ValueError, match="Invalid protocol configuration"):
            load_protocol(incomplete)


# ---------------------------------------------------------------------------
# compute_latency_stats
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeLatencyStats:
    def test_known_percentiles(self):
        """100 evenly-spaced samples [10, 20] ms give exact percentile values."""
        import numpy as np

        samples = list(map(float, range(10, 110)))  # 10.0 .. 109.0 ms, 100 samples
        stats = compute_latency_stats(samples)

        assert isinstance(stats, LatencyStats)
        assert stats.sample_count == 100
        assert stats.min_ms == pytest.approx(10.0)
        assert stats.max_ms == pytest.approx(109.0)
        assert stats.p50_ms == pytest.approx(np.percentile(samples, 50))
        assert stats.p90_ms == pytest.approx(np.percentile(samples, 90))
        assert stats.p99_ms == pytest.approx(np.percentile(samples, 99))

    def test_single_sample(self):
        """A single sample: all percentiles and mean collapse to that value."""
        stats = compute_latency_stats([42.0])

        assert stats.p50_ms == pytest.approx(42.0)
        assert stats.p90_ms == pytest.approx(42.0)
        assert stats.p99_ms == pytest.approx(42.0)
        assert stats.mean_ms == pytest.approx(42.0)
        assert stats.std_ms == pytest.approx(0.0)
        assert stats.sample_count == 1

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="samples must not be empty"):
            compute_latency_stats([])


# ---------------------------------------------------------------------------
# compute_throughput
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeThroughput:
    def test_batch_size_1(self):
        """batch=1, p50=10 ms → 100.0 img/s."""
        result = compute_throughput(batch_size=1, p50_latency_ms=10.0)
        assert result == pytest.approx(100.0)

    def test_batch_size_4(self):
        """batch=4, p50=10 ms → 400.0 img/s."""
        result = compute_throughput(batch_size=4, p50_latency_ms=10.0)
        assert result == pytest.approx(400.0)

    def test_zero_latency_raises(self):
        with pytest.raises(ValueError, match="p50_latency_ms must be > 0"):
            compute_throughput(batch_size=1, p50_latency_ms=0.0)

    def test_negative_latency_raises(self):
        with pytest.raises(ValueError, match="p50_latency_ms must be > 0"):
            compute_throughput(batch_size=1, p50_latency_ms=-5.0)

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            compute_throughput(batch_size=0, p50_latency_ms=10.0)


# ---------------------------------------------------------------------------
# MockRuntimeAdapter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMockRuntimeAdapter:
    def test_mock_loads_and_unloads(self):
        """is_ready is True after load_model and False after unload."""
        adapter = MockRuntimeAdapter()
        assert adapter.is_ready is False

        adapter.load_model("any/path/model.onnx")
        assert adapter.is_ready is True

        adapter.unload()
        assert adapter.is_ready is False

    def test_mock_run_inference_returns_deterministic_output(self):
        """Same output is returned on every call — no randomness."""
        adapter = MockRuntimeAdapter()
        adapter.load_model("model.onnx")

        results = [adapter.run_inference({}) for _ in range(3)]
        assert results[0] == results[1] == results[2]
        assert results[0] == {"output": [0.1, 0.2, 0.3]}

    def test_mock_full_benchmark(self):
        """run_benchmark with v1.0 protocol returns valid LatencyStats."""
        adapter = MockRuntimeAdapter()
        adapter.load_model("model.onnx")

        protocol = BenchmarkProtocol(protocol_version="1.0", input_shape=[1, 3, 224, 224])
        stats = adapter.run_benchmark({}, protocol)

        assert isinstance(stats, LatencyStats)
        assert stats.sample_count == 500
        assert stats.p50_ms > 0

    def test_mock_warmup_zero_iterations(self):
        """warmup with iterations=0 completes without error."""
        adapter = MockRuntimeAdapter()
        adapter.load_model("model.onnx")
        adapter.warmup({}, 0)  # must not raise

    def test_mock_resolved_input_shape_is_none(self):
        """MockRuntimeAdapter has no binding source, so resolved_input_shape
        stays None (inherited from the base class) even after load_model."""
        adapter = MockRuntimeAdapter()
        adapter.load_model("model.onnx")
        assert adapter.resolved_input_shape is None


# ---------------------------------------------------------------------------
# Runtime auto-detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRuntimeAutoDetect:
    def test_detect_onnx(self):
        assert detect_runtime("models/mobilenetv2.onnx") == "onnxruntime"

    def test_detect_tensorrt(self):
        assert detect_runtime("models/mobilenetv2.engine") == "tensorrt"

    def test_detect_unknown_raises(self):
        with pytest.raises(ValueError, match="Unrecognised model file extension"):
            detect_runtime("models/model.xyz")

    def test_get_adapter_onnx(self):
        adapter = get_adapter("onnxruntime")
        assert isinstance(adapter, ONNXRuntimeAdapter)

    def test_get_adapter_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown runtime"):
            get_adapter("unknown")


# ---------------------------------------------------------------------------
# ONNXRuntimeAdapter (integration — requires real model file)
# ---------------------------------------------------------------------------

MOBILENETV2_ONNX = Path(__file__).parent / "fixtures" / "mobilenetv2.onnx"
FAST_PROTOCOL_YAML = Path(__file__).parent / "fixtures" / "protocol_fast.yaml"


@pytest.mark.integration
class TestONNXRuntimeAdapter:
    def test_onnx_adapter_loads_and_unloads(self):
        """Session is active after load_model and released after unload."""
        adapter = ONNXRuntimeAdapter()

        adapter.load_model(MOBILENETV2_ONNX)
        assert adapter.is_ready is True

        adapter.unload()
        assert adapter.is_ready is False

    def test_onnx_adapter_benchmark(self):
        """run_benchmark produces valid LatencyStats over 10 real inference passes."""
        adapter = ONNXRuntimeAdapter()
        adapter.load_model(MOBILENETV2_ONNX)

        protocol = load_protocol(FAST_PROTOCOL_YAML)
        pixel_values = np.random.rand(1, 3, 224, 224).astype(np.float32)
        inputs = {adapter.input_names[0]: pixel_values}

        stats = adapter.run_benchmark(inputs, protocol)

        assert stats.p50_ms > 0
        assert stats.sample_count == 10

        adapter.unload()

    def test_onnx_adapter_resolves_static_input_shape(self, tmp_path):
        """A model whose graph declares fully concrete input dims resolves
        that exact NCHW shape from the graph itself, not the protocol file.

        Built on the fly rather than reusing tests/open/fixtures/mobilenetv2.onnx:
        that fixture's own graph declares a *symbolic* batch_size dim (see
        test_onnx_adapter_dynamic_batch_stays_none below), so it cannot
        exercise the fully-static path this test targets.
        """
        import onnx
        from onnx import TensorProto, helper

        model_path = tmp_path / "static_shape_model.onnx"
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])
        node = helper.make_node("Identity", ["input"], ["output"])
        graph = helper.make_graph([node], "static_shape_test", [inp], [out])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        onnx.checker.check_model(model)
        onnx.save(model, str(model_path))

        adapter = ONNXRuntimeAdapter()
        adapter.load_model(model_path)

        assert adapter.resolved_input_shape == [1, 3, 224, 224]
        adapter.unload()

    def test_onnx_adapter_dynamic_batch_stays_none(self):
        """tests/open/fixtures/mobilenetv2.onnx declares a symbolic 'batch_size'
        dim (verified: session.get_inputs()[0].shape == ['batch_size', 3,
        224, 224], not four concrete ints) — not fully static, so
        resolved_input_shape must stay None and the runner falls back to
        the protocol shape, per the dynamic-axis behavior documented on
        ONNXRuntimeAdapter.load_model."""
        adapter = ONNXRuntimeAdapter()
        adapter.load_model(MOBILENETV2_ONNX)

        assert adapter.resolved_input_shape is None
        adapter.unload()


# ---------------------------------------------------------------------------
# BenchmarkResult.input_shape (PR A: first-class, artifact-resolved field)
# ---------------------------------------------------------------------------


def _benchmark_result_kwargs(**overrides):
    kwargs = {
        "protocol_version": "1.0",
        "hardware_id": "a" * 64,
        "runtime": "onnxruntime",
        "model_hash": "b" * 64,
        "latency_p50_ms": 5.0,
        "latency_p90_ms": 6.0,
        "latency_p99_ms": 7.0,
        "latency_mean_ms": 5.5,
        "latency_std_ms": 0.5,
        "throughput_ips": 200.0,
        "sample_count": 500,
        "system_metadata": {},
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.unit
class TestBenchmarkResultInputShape:
    def test_round_trips_with_input_shape_set(self):
        """A result constructed with input_shape survives a JSON round-trip."""
        result = BenchmarkResult(**_benchmark_result_kwargs(input_shape=[1, 3, 224, 224]))
        assert result.input_shape == [1, 3, 224, 224]

        restored = BenchmarkResult.model_validate_json(result.model_dump_json())
        assert restored.input_shape == [1, 3, 224, 224]

    def test_round_trips_with_input_shape_none(self):
        """A result with no input_shape (e.g. RKNN, a pre-this-field agent)
        stays None through construction and a JSON round-trip."""
        result = BenchmarkResult(**_benchmark_result_kwargs())
        assert result.input_shape is None

        restored = BenchmarkResult.model_validate_json(result.model_dump_json())
        assert restored.input_shape is None

    def test_validator_rejects_zero_dimension(self):
        """A zero dimension anywhere in input_shape is rejected."""
        with pytest.raises(ValidationError, match="positive"):
            BenchmarkResult(**_benchmark_result_kwargs(input_shape=[1, 3, 0, 640]))

    def test_validator_rejects_empty_list(self):
        """An empty (but non-None) input_shape is rejected."""
        with pytest.raises(ValidationError, match="not be empty"):
            BenchmarkResult(**_benchmark_result_kwargs(input_shape=[]))

    def test_from_run_propagates_input_shape(self):
        """from_run forwards its input_shape kwarg onto the constructed result."""
        protocol = BenchmarkProtocol(protocol_version="1.0", input_shape=[1, 3, 224, 224])
        stats = LatencyStats(
            p50_ms=5.0,
            p90_ms=6.0,
            p99_ms=7.0,
            mean_ms=5.5,
            std_ms=0.5,
            min_ms=4.0,
            max_ms=8.0,
            sample_count=500,
        )
        result = BenchmarkResult.from_run(
            protocol=protocol,
            stats=stats,
            throughput=200.0,
            hardware_id="a" * 64,
            runtime="onnxruntime",
            model_hash="b" * 64,
            metadata={},
            input_shape=[1, 3, 640, 640],
        )
        assert result.input_shape == [1, 3, 640, 640]

    def test_from_run_defaults_input_shape_to_none(self):
        """from_run without an input_shape kwarg (e.g. RKNN callers) leaves it None."""
        protocol = BenchmarkProtocol(protocol_version="1.0", input_shape=[1, 3, 224, 224])
        stats = LatencyStats(
            p50_ms=5.0,
            p90_ms=6.0,
            p99_ms=7.0,
            mean_ms=5.5,
            std_ms=0.5,
            min_ms=4.0,
            max_ms=8.0,
            sample_count=500,
        )
        result = BenchmarkResult.from_run(
            protocol=protocol,
            stats=stats,
            throughput=200.0,
            hardware_id="a" * 64,
            runtime="onnxruntime",
            model_hash="b" * 64,
            metadata={},
        )
        assert result.input_shape is None


# ---------------------------------------------------------------------------
# BenchmarkResult.clock_state (PR B1: schema+API expand phase, no collector yet)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBenchmarkResultClockState:
    def test_round_trips_with_clock_state_set(self):
        """A result constructed with a populated rk3588 clock_state dict
        survives a JSON round-trip."""
        clock_state = {
            "captured": True,
            "source": "rk3588",
            "cpu_governor": "performance",
            "npu_governor": "performance",
            "dmc_governor": "performance",
        }
        result = BenchmarkResult(**_benchmark_result_kwargs(clock_state=clock_state))
        assert result.clock_state == clock_state

        restored = BenchmarkResult.model_validate_json(result.model_dump_json())
        assert restored.clock_state == clock_state

    def test_round_trips_with_clock_state_none(self):
        """A result with no clock_state (e.g. a pre-this-field agent) stays
        None through construction and a JSON round-trip."""
        result = BenchmarkResult(**_benchmark_result_kwargs())
        assert result.clock_state is None

        restored = BenchmarkResult.model_validate_json(result.model_dump_json())
        assert restored.clock_state is None

    def test_from_run_propagates_clock_state(self):
        """from_run forwards its clock_state kwarg onto the constructed result."""
        protocol = BenchmarkProtocol(protocol_version="1.0", input_shape=[1, 3, 224, 224])
        stats = LatencyStats(
            p50_ms=5.0,
            p90_ms=6.0,
            p99_ms=7.0,
            mean_ms=5.5,
            std_ms=0.5,
            min_ms=4.0,
            max_ms=8.0,
            sample_count=500,
        )
        clock_state = {"captured": True, "source": "jetson", "jetson_clocks": True}
        result = BenchmarkResult.from_run(
            protocol=protocol,
            stats=stats,
            throughput=200.0,
            hardware_id="a" * 64,
            runtime="onnxruntime",
            model_hash="b" * 64,
            metadata={},
            clock_state=clock_state,
        )
        assert result.clock_state == clock_state

    def test_from_run_defaults_clock_state_to_none(self):
        """from_run without a clock_state kwarg (B1, no collector yet) leaves it None."""
        protocol = BenchmarkProtocol(protocol_version="1.0", input_shape=[1, 3, 224, 224])
        stats = LatencyStats(
            p50_ms=5.0,
            p90_ms=6.0,
            p99_ms=7.0,
            mean_ms=5.5,
            std_ms=0.5,
            min_ms=4.0,
            max_ms=8.0,
            sample_count=500,
        )
        result = BenchmarkResult.from_run(
            protocol=protocol,
            stats=stats,
            throughput=200.0,
            hardware_id="a" * 64,
            runtime="onnxruntime",
            model_hash="b" * 64,
            metadata={},
        )
        assert result.clock_state is None


# ---------------------------------------------------------------------------
# BenchmarkResult.cpu_affinity (standalone affinity-placement field)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBenchmarkResultCpuAffinity:
    def test_round_trips_with_cpu_affinity_set(self):
        """A result constructed with a populated pinned cpu_affinity dict
        survives a JSON round-trip."""
        cpu_affinity = {
            "requested": "big",
            "applied": [4, 5, 6, 7],
            "tier": "big",
            "capacity": 1024,
        }
        result = BenchmarkResult(**_benchmark_result_kwargs(cpu_affinity=cpu_affinity))
        assert result.cpu_affinity == cpu_affinity

        restored = BenchmarkResult.model_validate_json(result.model_dump_json())
        assert restored.cpu_affinity == cpu_affinity

    def test_round_trips_with_cpu_affinity_none(self):
        """A result with no cpu_affinity (e.g. a pre-this-field agent) stays
        None through construction and a JSON round-trip."""
        result = BenchmarkResult(**_benchmark_result_kwargs())
        assert result.cpu_affinity is None

        restored = BenchmarkResult.model_validate_json(result.model_dump_json())
        assert restored.cpu_affinity is None

    def test_from_run_propagates_cpu_affinity(self):
        """from_run forwards its cpu_affinity kwarg onto the constructed result."""
        protocol = BenchmarkProtocol(protocol_version="1.0", input_shape=[1, 3, 224, 224])
        stats = LatencyStats(
            p50_ms=5.0,
            p90_ms=6.0,
            p99_ms=7.0,
            mean_ms=5.5,
            std_ms=0.5,
            min_ms=4.0,
            max_ms=8.0,
            sample_count=500,
        )
        cpu_affinity = {"requested": "none", "applied": None}
        result = BenchmarkResult.from_run(
            protocol=protocol,
            stats=stats,
            throughput=200.0,
            hardware_id="a" * 64,
            runtime="onnxruntime",
            model_hash="b" * 64,
            metadata={},
            cpu_affinity=cpu_affinity,
        )
        assert result.cpu_affinity == cpu_affinity

    def test_from_run_defaults_cpu_affinity_to_none(self):
        """from_run without a cpu_affinity kwarg (e.g. a pre-this-field agent) leaves it None."""
        protocol = BenchmarkProtocol(protocol_version="1.0", input_shape=[1, 3, 224, 224])
        stats = LatencyStats(
            p50_ms=5.0,
            p90_ms=6.0,
            p99_ms=7.0,
            mean_ms=5.5,
            std_ms=0.5,
            min_ms=4.0,
            max_ms=8.0,
            sample_count=500,
        )
        result = BenchmarkResult.from_run(
            protocol=protocol,
            stats=stats,
            throughput=200.0,
            hardware_id="a" * 64,
            runtime="onnxruntime",
            model_hash="b" * 64,
            metadata={},
        )
        assert result.cpu_affinity is None
