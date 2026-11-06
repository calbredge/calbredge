"""Deterministic v1 upload payload from a mock run, for the golden-payload test.

Every source of run-to-run variation is pinned in-process (clock, UUID,
timestamp, host metadata, hardware_id, CPU affinity, Jetson detection), so
the payload can be compared byte for byte with no normalization. The
fixture tests/open/fixtures/v1/golden_upload_mock.json was captured from this
function.
"""

import itertools
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import calbredge_protocol.models
import pytest
from calbredge import cli
from calbredge.collector import SystemMetadata
from calbredge_protocol.adapters import RuntimeAdapter

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = FIXTURES / "v1" / "golden_upload_mock.json"
FAST_PROTOCOL = FIXTURES / "protocol_fast.yaml"

_FIXED_NOW = datetime(2026, 9, 24, 12, 0, 0, 123456, tzinfo=timezone.utc)
_FIXED_UUID = uuid.UUID("11111111-2222-4333-8444-555555555555")
_FIXED_METADATA = SystemMetadata(
    hostname="golden-host",
    os_name="Linux",
    os_version="Ubuntu 22.04.5 LTS",
    cpu_model="x86_64",
    cpu_cores=8,
    ram_gb=15.5,
    gpu_model=None,
    python_version="3.12.3",
    runtime_name="mock",
    runtime_version="n/a",
)


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return _FIXED_NOW


def _scripted_clock():
    """perf_counter whose t1 - t0 differences follow a fixed, varied pattern."""
    counter = itertools.count()
    state = {"t": 1000.0}

    def perf_counter() -> float:
        n = next(counter)
        state["t"] += 0.0005 + 0.0001 * (n % 7)
        return state["t"]

    return perf_counter


def mock_upload_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, dict]:
    """Run the shared measurement core with the mock adapter.

    Returns (the exact bytes the uploader would POST, the pipeline state).
    """
    monkeypatch.setattr(time, "perf_counter", _scripted_clock())
    monkeypatch.setattr(uuid, "uuid4", lambda: _FIXED_UUID)
    monkeypatch.setattr(calbredge_protocol.models, "datetime", _FixedDatetime)
    monkeypatch.setattr(cli, "collect_metadata", lambda runtime: _FIXED_METADATA)
    monkeypatch.setattr(cli, "compute_hardware_id", lambda metadata: "f" * 64)
    monkeypatch.setattr(RuntimeAdapter, "_is_jetson", staticmethod(lambda *a, **k: False))

    model = tmp_path / "golden_model.onnx"
    model.write_bytes(b"calbredge golden mock model bytes\n")
    state: dict = {}
    result = cli._execute_benchmark(
        model=model,
        runtime="mock",
        protocol_path=FAST_PROTOCOL,
        model_label="golden",
        cpu_affinity="none",
        state=state,
        echo=lambda message: None,
    )
    # uploader.upload_or_save sends exactly result.model_dump_json().
    return result.model_dump_json(), state
