"""Tests for run variance detection and status classification.

Covers the classify_run() function in protocol.adapters and the
run_status / run_status_reason fields on BenchmarkResult.
"""

import pytest
from calbredge_protocol.adapters import classify_run
from calbredge_protocol.models import BenchmarkResult


@pytest.mark.unit
class TestClassifyRun:
    def test_stable_result(self):
        """p99/p50 = 1.1 → stable."""
        status, _reason = classify_run(p50=10.0, p99=11.0)
        assert status == "stable"

    def test_inconclusive_result(self):
        """p99/p50 = 2.5 → inconclusive (between 2.0 and 3.0 thresholds)."""
        status, _reason = classify_run(p50=10.0, p99=25.0)
        assert status == "inconclusive"

    def test_unstable_result(self):
        """p99/p50 = 3.5 → unstable (above 3.0 threshold)."""
        status, _reason = classify_run(p50=10.0, p99=35.0)
        assert status == "unstable"

    def test_unstable_reason_populated(self):
        """Unstable reason must mention the ratio value and variance."""
        _, reason = classify_run(p50=10.0, p99=35.0)
        assert reason is not None
        assert "3.5" in reason
        assert "p99/p50" in reason.lower() or "ratio" in reason.lower()

    def test_inconclusive_reason_populated(self):
        """Inconclusive reason must mention the ratio value."""
        _, reason = classify_run(p50=10.0, p99=25.0)
        assert reason is not None
        assert "2.5" in reason

    def test_stable_reason_is_none(self):
        """Stable runs must have no reason string."""
        _, reason = classify_run(p50=10.0, p99=11.0)
        assert reason is None

    def test_benchmark_result_accepts_run_status(self):
        """BenchmarkResult accepts and stores run_status / run_status_reason."""
        result = BenchmarkResult(
            protocol_version="1.0",
            hardware_id="a" * 64,
            runtime="onnxruntime",
            model_hash="b" * 64,
            latency_p50_ms=5.0,
            latency_p90_ms=6.0,
            latency_p99_ms=7.0,
            latency_mean_ms=5.5,
            latency_std_ms=0.5,
            throughput_ips=200.0,
            sample_count=500,
            system_metadata={},
            run_status="unstable",
            run_status_reason="p99/p50 ratio 3.2 — high variance detected",
        )
        assert result.run_status == "unstable"
        assert result.run_status_reason is not None
        assert "3.2" in result.run_status_reason

    def test_benchmark_result_defaults_to_stable(self):
        """Results constructed without run_status default to 'stable'."""
        result = BenchmarkResult(
            protocol_version="1.0",
            hardware_id="a" * 64,
            runtime="onnxruntime",
            model_hash="b" * 64,
            latency_p50_ms=5.0,
            latency_p90_ms=6.0,
            latency_p99_ms=7.0,
            latency_mean_ms=5.5,
            latency_std_ms=0.5,
            throughput_ips=200.0,
            sample_count=500,
            system_metadata={},
        )
        assert result.run_status == "stable"
        assert result.run_status_reason is None
