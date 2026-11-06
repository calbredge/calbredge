"""End-to-end integration tests for the calbredge CLI.

These tests invoke the full CLI pipeline via Click's CliRunner and require
the real MobileNetV2 ONNX fixture. They are marked @pytest.mark.integration
and are excluded from the default CI run.
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from calbredge.cli import _load_config, classify_stability, main
from click.testing import CliRunner

from tests.open.test_rknn_adapter import _make_rknnlite_mock, _patch_rknnlite

MOBILENETV2 = str(Path(__file__).parent / "fixtures" / "mobilenetv2.onnx")
FAST_PROTOCOL = str(Path(__file__).parent / "fixtures" / "protocol_fast.yaml")

# Fields that must be present and non-null in every BenchmarkResult JSON
REQUIRED_FIELDS = [
    "run_id",
    "protocol_version",
    "hardware_id",
    "runtime",
    "model_hash",
    "latency_p50_ms",
    "latency_p90_ms",
    "latency_p99_ms",
    "latency_mean_ms",
    "latency_std_ms",
    "throughput_ips",
    "sample_count",
    "system_metadata",
    "timestamp",
]


@pytest.mark.integration
class TestAgentE2E:
    def test_full_pipeline_dry_run(self):
        """Full benchmark pipeline exits cleanly and emits all result fields."""
        runner = CliRunner()
        result = runner.invoke(
            main, ["run", "--model", MOBILENETV2, "--protocol", FAST_PROTOCOL, "--dry-run"]
        )

        assert result.exit_code == 0, (
            f"CLI exited with code {result.exit_code}.\nOutput:\n{result.output}"
        )
        for field in [
            "run_id",
            "protocol_version",
            "hardware_id",
            "runtime",
            "latency_p50_ms",
            "throughput_ips",
        ]:
            assert field in result.output, (
                f"Expected field '{field}' not found in output.\nOutput:\n{result.output}"
            )

    def test_full_pipeline_offline(self):
        """--offline flag prints the OFFLINE banner and produces valid output."""
        runner = CliRunner()
        result = runner.invoke(
            main, ["run", "--model", MOBILENETV2, "--protocol", FAST_PROTOCOL, "--offline"]
        )

        assert result.exit_code == 0, (
            f"CLI exited with code {result.exit_code}.\nOutput:\n{result.output}"
        )
        assert "OFFLINE mode" in result.output, (
            f"Expected 'OFFLINE mode' banner in output.\nOutput:\n{result.output}"
        )
        for field in ["run_id", "latency_p50_ms", "throughput_ips"]:
            assert field in result.output, (
                f"Expected field '{field}' not found in output.\nOutput:\n{result.output}"
            )

    def test_full_pipeline_saves_locally(self):
        """Result JSON is saved to ~/.calbredge/results/ and contains all required fields."""
        runner = CliRunner()
        result = runner.invoke(
            main, ["run", "--model", MOBILENETV2, "--protocol", FAST_PROTOCOL, "--dry-run"]
        )

        assert result.exit_code == 0, (
            f"CLI exited with code {result.exit_code}.\nOutput:\n{result.output}"
        )

        # Extract run_id from the JSON block in stdout
        output_data = json.loads(
            result.output[result.output.index("{") : result.output.rindex("}") + 1]
        )
        run_id = output_data["run_id"]

        saved_path = Path.home() / ".calbredge" / "results" / f"{run_id}.json"
        assert saved_path.exists(), f"Expected saved file not found: {saved_path}"

        saved = json.loads(saved_path.read_text())

        for field in REQUIRED_FIELDS:
            assert field in saved, f"Missing field '{field}' in saved JSON"
            assert saved[field] is not None, f"Field '{field}' is null in saved JSON"

        # model_label is the only field permitted to be null
        assert "model_label" in saved  # present but may be null

    def test_unsupported_runtime(self):
        """Stub runtime exits with code 1 and explains the adapter is not implemented."""
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--model", MOBILENETV2, "--runtime", "tensorrt"])

        assert result.exit_code == 1
        assert "not yet fully implemented" in result.output or "not installed" in result.output, (
            f"Expected stub/not-installed message.\nOutput:\n{result.output}"
        )

    def test_missing_model(self):
        """Non-existent model path exits with code 2 (Click path validation)."""
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--model", "nonexistent.onnx", "--dry-run"])

        assert result.exit_code != 0


@pytest.mark.unit
class TestRKNNInputDtypeThroughCLI:
    """Proves cli.py:343 actually reads ``adapter.input_dtype`` rather than a
    hardcoded ``np.float32`` literal.

    None of the ``TestAgentE2E`` tests above catch a regression here: they
    only ever exercise ``ONNXRuntimeAdapter``, whose ``input_dtype`` is
    float32 either way, so a revert of cli.py:343 to the old hardcoded
    ``.astype(np.float32)`` would leave every one of them green. This test
    drives the same Click command with ``--runtime rknn-fp16`` — an adapter
    whose ``input_dtype`` defaults to uint8 — using the dtype-strict RKNNLite
    mock from test_rknn_adapter.py (reused, not reimplemented) so a wrong
    dtype at cli.py:343 is caught two ways: the run fails outright (the mock
    raises on mismatch), and the captured instance's last input array is
    asserted to be uint8 directly.
    """

    def test_rknn_run_synthesizes_uint8_inputs(self, tmp_path):
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"fake onnx bytes")
        (tmp_path / "model.onnx.rknn.fp16.default").write_bytes(b"fake rknn cache")

        instances: list = []
        healthy_output = [np.ones((1, 1000), dtype=np.float32)]
        rknnlite_api = _make_rknnlite_mock(healthy_output, instances=instances)

        runner = CliRunner()
        with _patch_rknnlite(rknnlite_api):
            result = runner.invoke(
                main,
                [
                    "run",
                    "--model",
                    str(onnx_path),
                    "--runtime",
                    "rknn-fp16",
                    "--protocol",
                    FAST_PROTOCOL,
                    "--dry-run",
                ],
            )

        assert result.exit_code == 0, (
            f"CLI exited with code {result.exit_code}.\nOutput:\n{result.output}"
        )
        assert instances, "expected RKNNLite to have been instantiated by load_model"
        # The last inference() call is from the measurement loop, i.e.
        # cli.py:343's synthesized array — not the load_model canary.
        last_inputs = instances[-1].last_inference_inputs
        assert last_inputs[0].dtype == np.dtype(np.uint8)


# ---------------------------------------------------------------------------
# classify_stability unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClassifyStability:
    def test_ratio_exactly_2_5_is_stable(self):
        """p99/p50 == 2.5 is at the boundary and must be classified stable."""
        status, reason = classify_stability(p50=10.0, p99=25.0)
        assert status == "stable"
        assert reason is None

    def test_ratio_above_2_5_is_unstable(self):
        """p99/p50 == 2.51 crosses the threshold and must be classified unstable."""
        status, reason = classify_stability(p50=10.0, p99=25.1)
        assert status == "unstable"
        assert reason is not None
        assert "2.5" in reason

    def test_ratio_1_0_is_stable(self):
        """Perfect consistency (p99 == p50) must be classified stable."""
        status, reason = classify_stability(p50=10.0, p99=10.0)
        assert status == "stable"
        assert reason is None

    def test_p50_zero_is_unknown(self):
        """p50 <= 0 must return unknown to avoid division by zero."""
        status, reason = classify_stability(p50=0.0, p99=10.0)
        assert status == "unknown"
        assert reason is not None


# ---------------------------------------------------------------------------
# _load_config()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadConfig:
    def test_returns_empty_dict_when_file_absent(self, tmp_path):
        missing = tmp_path / "config.yaml"
        with patch("calbredge.cli.CONFIG_PATH", missing):
            assert _load_config() == {}

    def test_returns_parsed_dict(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("api_key: secret\nupload_url: https://example.com\n")
        with patch("calbredge.cli.CONFIG_PATH", cfg):
            data = _load_config()
        assert data["api_key"] == "secret"
        assert data["upload_url"] == "https://example.com"

    def test_returns_empty_dict_on_malformed_yaml(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("}{invalid yaml}{")
        with patch("calbredge.cli.CONFIG_PATH", cfg):
            assert _load_config() == {}

    def test_returns_empty_dict_when_yaml_is_not_a_mapping(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("- item1\n- item2\n")
        with patch("calbredge.cli.CONFIG_PATH", cfg):
            assert _load_config() == {}


# ---------------------------------------------------------------------------
# calbredge config subcommand
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfigCommand:
    def test_writes_config_file(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        runner = CliRunner()
        with patch("calbredge.cli.CONFIG_PATH", cfg):
            result = runner.invoke(
                main,
                ["config", "--api-key", "mykey", "--upload-url", "https://api.test/benchmarks"],
            )
        assert result.exit_code == 0, result.output
        assert "Config saved to ~/.calbredge/config.yaml" in result.output
        import yaml

        data = yaml.safe_load(cfg.read_text())
        assert data["api_key"] == "mykey"
        assert data["upload_url"] == "https://api.test/benchmarks"

    def test_merges_with_existing_config(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("api_key: oldkey\nupload_url: https://old.example.com\n")
        runner = CliRunner()
        with patch("calbredge.cli.CONFIG_PATH", cfg):
            runner.invoke(main, ["config", "--api-key", "newkey"])
        import yaml

        data = yaml.safe_load(cfg.read_text())
        assert data["api_key"] == "newkey"
        assert data["upload_url"] == "https://old.example.com"

    def test_no_flags_is_a_no_op(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("api_key: existing\n")
        runner = CliRunner()
        with patch("calbredge.cli.CONFIG_PATH", cfg):
            result = runner.invoke(main, ["config"])
        assert result.exit_code == 0
        import yaml

        data = yaml.safe_load(cfg.read_text())
        assert data["api_key"] == "existing"
