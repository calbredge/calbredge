"""Tests for the v2 run bundle: RESULT_SCHEMA.md §6.

Covers the golden v1 upload payload, samples.jsonl reproducing the recorded
statistics exactly, to_v1() equality, the code-path-set telemetry fields,
secret redaction, and the `run`/`gate` contracts that must not change.
"""

import importlib.metadata
import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from calbredge import cli
from calbredge.bundle import REDACTED, redact_argv, result_json_text, write_bundle
from calbredge.result_builder import build_result_v2
from calbredge_protocol.adapters import RuntimeAdapter
from calbredge_protocol.autodetect import get_adapter
from calbredge_protocol.loader import load_protocol
from calbredge_protocol.models import protocol_digest
from calbredge_protocol.power_meter import PowerReading
from calbredge_protocol.result_v2 import ResultV2
from calbredge_protocol.stats import compute_latency_stats
from click.testing import CliRunner

from tests.open.fake_hosts import conv_model
from tests.open.test_hardware_metrics import _SAMPLE_TEGRASTATS, _popen_mock, _StubAdapter
from tests.open.v1_golden import GOLDEN, mock_upload_payload

pytestmark = pytest.mark.unit

FAKE_KEY = "cbk-FAKE-KEY-0123456789abcdef"
FAKE_ENV_TOKEN = "env-token-value-9876543210"


def _protocol_file(tmp_path: Path, iterations: int = 20) -> Path:
    path = tmp_path / "protocol.yaml"
    path.write_text(
        'protocol_version: "1.0"\nwarmup_iterations: 2\n'
        f"measurement_iterations: {iterations}\nbatch_size: 1\ninput_shape: [1, 3, 8, 8]\n"
    )
    return path


def _only_bundle(bundle_dir: Path) -> Path:
    bundles = [p for p in bundle_dir.iterdir() if not p.name.startswith(".")]
    assert len(bundles) == 1, bundles
    return bundles[0]


def _run(tmp_path: Path, *extra: str, runtime: str = "onnxruntime", iterations: int = 20):
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = conv_model(tmp_path / "m.onnx")
    args = [
        "run", "--model", str(model), "--runtime", runtime, "--protocol",
        str(_protocol_file(tmp_path, iterations)), "--cpu-affinity", "none", "--dry-run",
        "--bundle-dir", str(tmp_path / "runs"), *extra,
    ]  # fmt: skip
    return CliRunner().invoke(cli.main, args)


def test_v1_upload_payload_is_byte_identical_to_golden(tmp_path, monkeypatch):
    payload, state = mock_upload_payload(tmp_path, monkeypatch)
    assert payload + "\n" == GOLDEN.read_text()
    # The samples ride on LatencyStats, which is never serialized into v1.
    assert len(state["stats"].samples_ms) == 10
    assert "samples" not in payload


def test_jetson_payload_difference_is_exactly_module_rail():
    """A Jetson's tegrastats rows record power_domain="module_rail", never null, and every
    other v1 hardware field is as listed here."""
    adapter = _StubAdapter()
    popen = _popen_mock(_SAMPLE_TEGRASTATS)
    with (
        patch.object(RuntimeAdapter, "_is_jetson", return_value=True),
        patch("calbredge_protocol.adapters.subprocess.Popen", return_value=popen),
        patch("calbredge_protocol.adapters.subprocess.run", side_effect=FileNotFoundError),
        patch("time.sleep"),
    ):
        hm = adapter.collect_hardware_metrics()
    assert hm is not None
    # The full v1 HardwareMetrics a Jetson run uploads.
    assert hm.model_dump() == {
        "watts_average": 3.0795000000000003,
        "watts_peak": 3.1,
        "joules_per_inference": None,
        "energy_joules_window": None,
        "temperature_start_celsius": None,
        "temperature_end_celsius": None,
        "temperature_peak_celsius": 43.5,
        "thermal_throttle_detected": False,
        "gpu_memory_used_mb": None,
        "gpu_memory_total_mb": None,
        "system_memory_used_mb": None,
        "power_mode": None,
        "precision": None,
        "power_source": "tegrastats",
        "power_domain": "module_rail",
    }
    assert adapter._last_telemetry_boundary == "post_loop_snapshot"


def test_no_telemetry_means_no_boundary():
    adapter = _StubAdapter()
    with patch.object(RuntimeAdapter, "_is_jetson", return_value=False):
        assert adapter.collect_hardware_metrics() is None
    assert adapter._last_telemetry_boundary is None


def test_rknn_meter_is_a_windowed_bracket():
    adapter = get_adapter("rknn")
    reading = PowerReading(
        watts_average=3.5, watts_peak=3.9, energy_joules_window=1.2, sample_count=40
    )
    hm = adapter.collect_hardware_metrics(reading)  # type: ignore[call-arg]
    assert hm is not None
    assert hm.power_domain == "board_input"
    assert adapter._last_telemetry_boundary == "windowed_bracket"  # type: ignore[attr-defined]
    adapter.collect_hardware_metrics(None)  # type: ignore[call-arg]
    assert adapter._last_telemetry_boundary is None  # type: ignore[attr-defined]


def test_run_console_labels_post_loop_power_and_keeps_the_recorded_values(tmp_path):
    """A Jetson-style post-loop tegrastats reading: the console labels the watts provisional
    and drops J/inference; the printed JSON and the bundle still record joules_per_inference."""
    popen = _popen_mock(_SAMPLE_TEGRASTATS)
    with (
        patch.object(RuntimeAdapter, "_is_jetson", return_value=True),
        patch("calbredge_protocol.adapters.subprocess.Popen", return_value=popen),
        patch("calbredge_protocol.adapters.subprocess.run", side_effect=FileNotFoundError),
        patch("time.sleep"),
    ):
        result = _run(tmp_path)
    assert result.exit_code == 0, result.output
    assert (
        "  Power:   3.08 W avg, 3.10 W peak (provisional: read after the timed loop, not during it)"
    ) in result.stdout
    assert "J/inference" not in result.stdout
    assert re.search(r'"joules_per_inference": [0-9]', result.stdout)  # the --dry-run JSON
    bundle = _only_bundle(tmp_path / "runs")
    name = "result.json" if (bundle / "result.json").exists() else "result.invalid.json"
    recorded = json.loads((bundle / name).read_text())
    recorded = recorded.get("result", recorded)
    assert recorded["metrics"]["joules_per_inference"] is not None
    assert recorded["target"]["telemetry_measurement_boundary"] == "post_loop_snapshot"


def test_run_summary_has_a_clocks_line(tmp_path):
    result = _run(tmp_path)
    assert result.exit_code == 0, result.output
    assert "  Clocks      " in result.stdout


def test_bundle_contents_and_recomputed_stats_match_exactly(tmp_path):
    result = _run(tmp_path, iterations=25)
    assert result.exit_code == 0, result.output
    bundle = _only_bundle(tmp_path / "runs")
    assert sorted(p.name for p in bundle.iterdir()) == [
        "environment.json",
        "logs",
        "result.json",
        "samples.jsonl",
    ]
    assert (bundle / "logs" / "calbredge.log").read_text().startswith("CPU affinity:")

    v2 = ResultV2.model_validate_json((bundle / "result.json").read_text())
    # A gate decision's candidate_result_digest hashes result_json_text(raw):
    # exactly these bytes.
    written = (bundle / "result.json").read_text()
    assert result_json_text(json.loads(written)) == written
    # 2.2: the run records the digest of the protocol it loaded.
    recorded = v2.protocol.protocol_digest
    assert recorded == protocol_digest(load_protocol(_protocol_file(tmp_path, 25)))
    lines = (bundle / "samples.jsonl").read_text().splitlines()
    assert len(lines) == v2.protocol.measurement_iterations == 25
    samples = [json.loads(line) for line in lines]
    assert [s["iteration"] for s in samples] == list(range(25))

    recomputed = compute_latency_stats([s["latency_ms"] for s in samples])
    m = v2.metrics
    recorded = (m.latency_p50_ms, m.latency_p90_ms, m.latency_p99_ms)
    recorded += (m.latency_mean_ms, m.latency_std_ms)
    r = recomputed
    assert recorded == (r.p50_ms, r.p90_ms, r.p99_ms, r.mean_ms, r.std_ms)
    assert m.sample_count == recomputed.sample_count
    assert v2.result_id == bundle.name
    assert v2.candidate.runtime == "onnxruntime-cpu"
    assert v2.target.cpu_affinity_tier == "none"


def test_environment_records_the_installed_calbredge_version(tmp_path):
    # calbredge_version is the installed version, never a hardcoded one.
    result = _run(tmp_path)
    assert result.exit_code == 0, result.output
    env = json.loads((_only_bundle(tmp_path / "runs") / "environment.json").read_text())
    assert env["calbredge_version"] == importlib.metadata.version("calbredge")
    assert env["packages"]["calbredge"] == env["calbredge_version"]


def test_to_v1_equals_the_actual_upload_payload(tmp_path, monkeypatch):
    model = conv_model(tmp_path / "m.onnx")
    state: dict = {}
    v1 = cli._execute_benchmark(
        model=model,
        runtime="onnxruntime",
        protocol_path=_protocol_file(tmp_path),
        model_label="conv",
        cpu_affinity="none",
        state=state,
        echo=lambda message: None,
    )
    raw, errors = build_result_v2(
        v1=v1,
        stats=state["stats"],
        protocol=state["protocol"],
        model_path=model,
        metadata=state["metadata"],
        engine_path=state["engine_path"],
        telemetry_measurement_boundary=state["telemetry_boundary"],
        adapter_clock_state=state["adapter_clock_state"],
        cwd=tmp_path,
        contract_cache_dir=tmp_path / "cache",
    )
    assert errors == []
    rebuilt = ResultV2.model_validate(raw).to_v1()
    # Legacy status is the agent's to fill (to_v1 documents this).
    rebuilt = rebuilt.model_copy(update={"status": v1.status, "status_reason": v1.status_reason})
    assert rebuilt.model_dump_json() == v1.model_dump_json()


def test_invalid_v2_is_written_as_result_invalid_json_and_run_still_succeeds(tmp_path):
    # The mock runtime is not a v2 runtime_family, so its v2 result cannot validate.
    result = _run(tmp_path, runtime="mock")
    assert result.exit_code == 0, result.output
    bundle = _only_bundle(tmp_path / "runs")
    assert not (bundle / "result.json").exists()
    invalid = json.loads((bundle / "result.invalid.json").read_text())
    assert any("runtime_family" in e for e in invalid["errors"])
    assert "failed validation" in result.stderr


def test_no_bundle_writes_nothing(tmp_path):
    result = _run(tmp_path, "--no-bundle")
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "runs").exists()


def test_unwritable_bundle_dir_is_a_warning_not_a_failure(tmp_path):
    read_only = tmp_path / "runs"
    read_only.mkdir()
    read_only.chmod(0o500)
    try:
        result = _run(tmp_path)
    finally:
        read_only.chmod(0o700)
    assert result.exit_code == 0, result.output
    assert "run bundle not written" in result.stderr
    assert list(read_only.iterdir()) == []


def test_run_stdout_is_unchanged_by_the_bundle(tmp_path):
    with_bundle = _run(tmp_path / "a")
    without = _run(tmp_path / "b", "--no-bundle")
    assert "Run bundle:" in with_bundle.stderr
    assert "Run bundle" not in with_bundle.stdout

    def strip_ids(text: str) -> str:
        return "\n".join(
            line for line in text.splitlines() if '"' not in line and "run_id" not in line
        )

    assert "Calbredge Result Summary" in with_bundle.stdout
    assert strip_ids(with_bundle.stdout).count("\n") == strip_ids(without.stdout).count("\n")


def test_secrets_appear_nowhere_in_the_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("CALBREDGE_API_TOKEN", FAKE_ENV_TOKEN)
    monkeypatch.setattr(
        "sys.argv",
        [
            "calbredge",
            "run",
            "--api-key",
            FAKE_KEY,
            "--upload",
            f"https://u:{FAKE_KEY}@h/x?token={FAKE_ENV_TOKEN}",
        ],
    )
    result = _run(tmp_path, "--api-key", FAKE_KEY)
    assert result.exit_code == 0, result.output
    bundle = _only_bundle(tmp_path / "runs")
    files = [p for p in bundle.rglob("*") if p.is_file()]
    assert len(files) == 4
    for path in files:
        text = path.read_text()
        assert FAKE_KEY not in text, path
        assert FAKE_ENV_TOKEN not in text, path
    env = json.loads((bundle / "environment.json").read_text())
    assert REDACTED in json.dumps(env["command_line"])
    assert "CALBREDGE_API_TOKEN" not in env["environment_variables"]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--api-key", "abc123"], ["--api-key", REDACTED]),
        (["--api-key=abc123"], [f"--api-key={REDACTED}"]),
        (["--token", "t"], ["--token", REDACTED]),
        (["https://user:pw@host/p"], [f"https://{REDACTED}@host/p"]),
        (["https://host/p?access_token=zzz&x=1"], [f"https://host/p?access_token={REDACTED}&x=1"]),
        (
            ["--model", "m.onnx", "--runtime", "tensorrt"],
            ["--model", "m.onnx", "--runtime", "tensorrt"],
        ),
    ],
)
def test_redact_argv(argv, expected):
    assert redact_argv(argv) == expected


def test_existing_bundle_is_never_overwritten(tmp_path):
    kwargs = {
        "result_id": "r1", "result": {}, "errors": [], "samples_ms": (1.0,),
        "environment": {}, "log_lines": [], "secrets": [],
    }  # fmt: skip
    write_bundle(tmp_path, **kwargs)
    with pytest.raises(FileExistsError):
        write_bundle(tmp_path, **kwargs)
