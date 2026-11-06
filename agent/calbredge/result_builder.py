"""Assemble a v2 result (docs/RESULT_SCHEMA.md) from a finished run.

The section builders here are shared with `calbredge inspect`, so the fields
a run records and the fields inspect predicts cannot drift apart. A run
passes what actually happened (the adapter's recorded runtime label, engine
file, telemetry boundary, affinity); inspect passes predictions.

build_result_v2 returns the raw dict plus validation errors rather than
raising: an invalid v2 result must never break the v1 path (it is written to
the bundle as result.invalid.json instead).
"""

from pathlib import Path
from typing import Any

from calbredge_protocol.canonical import runtime_family
from calbredge_protocol.derive_v2 import clocks_pinned, cpu_affinity_tier
from calbredge_protocol.models import (
    BenchmarkProtocol,
    BenchmarkResult,
    HardwareMetrics,
    protocol_digest,
)
from calbredge_protocol.result_v2 import SCHEMA_VERSION, ResultV2
from calbredge_protocol.stats import LatencyStats
from pydantic import ValidationError

from calbredge.clock_state import collect_clock_state
from calbredge.collector import SystemMetadata
from calbredge.hasher import compute_model_hash
from calbredge.host_v2 import (
    device_tree_model,
    driver_version,
    git_state,
    hardware_family,
    power_mode,
    runtime_version,
)
from calbredge.onnx_contract import GraphContract, bind_input_shape, load_graph_contract

DEFAULT_CONTRACT_CACHE_DIR = Path.home() / ".calbredge" / "contract-cache"

_ARTIFACT_FORMAT = {"onnxruntime": "onnx", "tensorrt": "engine", "rknn": "rknn", "tflite": "tflite"}


def precision_for(runtime_label: str, graph_precision: str | None) -> str | None:
    """The runtime label's precision suffix if it has one, else the graph's (§3.3.1)."""
    suffix = runtime_label.split("-", 1)[1] if "-" in runtime_label else None
    return suffix if suffix in ("fp32", "fp16", "int8") else graph_precision


def candidate_section(
    *,
    model_hash: str,
    model_label: str | None,
    engine_hash: str | None,
    runtime_label: str,
    graph_precision: str | None,
    cwd: Path,
    root: Path,
    engine_build: dict[str, Any] | None = None,
) -> dict[str, Any]:
    family = runtime_family(runtime_label)
    commit, dirty = git_state(cwd)
    return {
        "model_hash": model_hash,
        "model_label": model_label,
        "engine_hash": engine_hash,
        "git_commit": commit,
        "git_dirty": dirty,
        "runtime": runtime_label,
        "runtime_family": family,
        "runtime_version": runtime_version(family),
        "driver_version": driver_version(root),
        "precision": precision_for(runtime_label, graph_precision),
        "engine_build": engine_build,
    }


def target_section(
    *,
    hardware_id: str,
    metadata: SystemMetadata,
    clock_state: dict[str, Any] | None,
    cpu_affinity: dict[str, Any] | None,
    power_domain: str | None,
    power_source: str | None,
    telemetry_measurement_boundary: str | None,
    root: Path,
) -> dict[str, Any]:
    system_metadata = {**metadata.model_dump(), "device_tree_model": device_tree_model(root)}
    family_name, family_source = hardware_family(system_metadata, root)
    return {
        "hardware_id": hardware_id,
        "hardware_family": family_name,
        "hardware_family_source": family_source,
        "power_mode": power_mode(root),
        "power_domain": power_domain,
        "power_source": power_source,
        "telemetry_measurement_boundary": telemetry_measurement_boundary,
        "clock_state": clock_state,
        "clocks_pinned": clocks_pinned(clock_state),
        "cpu_affinity": cpu_affinity,
        "cpu_affinity_tier": cpu_affinity_tier(cpu_affinity),
        "system_metadata": system_metadata,
    }


def protocol_section(protocol: BenchmarkProtocol, contract: GraphContract) -> dict[str, Any]:
    return {
        "protocol_version": protocol.protocol_version,
        "warmup_iterations": protocol.warmup_iterations,
        "measurement_iterations": protocol.measurement_iterations,
        "batch_size": protocol.batch_size,
        "input_shape": bind_input_shape(contract.primary_input_dims, protocol.input_shape),
        "io_signature": contract.io_signature,
        "preprocessing": "none",
        "input_fixture_digest": None,
        "graph_kind": contract.graph_kind,
        "protocol_digest": protocol_digest(protocol),
    }


def engine_io_mismatches(engine_io: dict[str, Any], io_signature: dict[str, Any]) -> list[str]:
    """Where the loaded engine's I/O disagrees with the model's (names, dtypes, static dims).

    A guard on top of the content-addressed engine cache: an engine whose
    tensors don't match the model it claims to be built from is not that model.
    """
    problems = []
    for side in ("inputs", "outputs"):
        engine_specs, model_specs = engine_io.get(side, []), io_signature.get(side, [])
        if [s["name"] for s in engine_specs] != [s["name"] for s in model_specs]:
            problems.append(
                f"{side}: engine {[s['name'] for s in engine_specs]} vs "
                f"model {[s['name'] for s in model_specs]}"
            )
            continue
        for e, m in zip(engine_specs, model_specs, strict=True):
            if e["dtype"] != m["dtype"]:
                problems.append(
                    f"{side} {m['name']}: dtype engine {e['dtype']} vs model {m['dtype']}"
                )
            static = [(i, d) for i, d in enumerate(m["shape"]) if isinstance(d, int) and d > 0]
            if len(e["shape"]) != len(m["shape"]) or any(e["shape"][i] != d for i, d in static):
                problems.append(
                    f"{side} {m['name']}: shape engine {e['shape']} vs model {m['shape']}"
                )
    return problems


def _telemetry_captured(hm: HardwareMetrics | None) -> list[str]:
    if hm is None:
        return []
    captured = []
    if hm.watts_average is not None or hm.watts_peak is not None:
        captured.append("power")
    if hm.temperature_peak_celsius is not None or hm.temperature_start_celsius is not None:
        captured.append("thermal")
    if hm.gpu_memory_used_mb is not None or hm.system_memory_used_mb is not None:
        captured.append("memory")
    return captured


def _metrics_section(v1: BenchmarkResult) -> dict[str, Any]:
    hm = v1.hardware_metrics or HardwareMetrics()
    return {
        "latency_p50_ms": v1.latency_p50_ms,
        "latency_p90_ms": v1.latency_p90_ms,
        "latency_p99_ms": v1.latency_p99_ms,
        "latency_mean_ms": v1.latency_mean_ms,
        "latency_std_ms": v1.latency_std_ms,
        "sample_count": v1.sample_count,
        "throughput_ips": v1.throughput_ips,
        "watts_average": hm.watts_average,
        "watts_peak": hm.watts_peak,
        "energy_joules_window": hm.energy_joules_window,
        "joules_per_inference": hm.joules_per_inference,
        "temperature_start_celsius": hm.temperature_start_celsius,
        "temperature_end_celsius": hm.temperature_end_celsius,
        "temperature_peak_celsius": hm.temperature_peak_celsius,
        "thermal_throttle_detected": hm.thermal_throttle_detected,
        "gpu_memory_used_mb": hm.gpu_memory_used_mb,
        "gpu_memory_total_mb": hm.gpu_memory_total_mb,
        "system_memory_used_mb": hm.system_memory_used_mb,
        "run_status": v1.run_status,
        "run_status_reason": v1.run_status_reason,
        "samples_pointer": "samples.jsonl",
    }


def build_result_v2(
    *,
    v1: BenchmarkResult,
    stats: LatencyStats,
    protocol: BenchmarkProtocol,
    model_path: Path,
    metadata: SystemMetadata,
    engine_path: Path | None,
    telemetry_measurement_boundary: str | None,
    adapter_clock_state: dict[str, Any] | None,
    cwd: Path,
    engine_build: dict[str, Any] | None = None,
    engine_io: dict[str, Any] | None = None,
    root: Path = Path("/"),
    contract_cache_dir: Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """(raw v2 result, validation errors). Errors empty means ResultV2 accepts it."""
    if stats.sample_count != len(stats.samples_ms):
        raise ValueError("LatencyStats carries no per-iteration samples")
    contract, _ = load_graph_contract(
        model_path, v1.model_hash, contract_cache_dir or DEFAULT_CONTRACT_CACHE_DIR
    )
    engine_hash = (
        compute_model_hash(engine_path)
        if engine_path is not None and engine_path.exists()
        else None
    )
    family = runtime_family(v1.runtime)
    # RK3588's adapter records clock state itself; everywhere else it is collected here.
    clock_state = (
        adapter_clock_state if adapter_clock_state is not None else collect_clock_state(root)
    )
    hm = v1.hardware_metrics
    raw: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_id": v1.run_id,
        "timestamp": v1.timestamp.isoformat(),
        "workload": {
            "workload_type": "single_forward_pass",
            "work_unit": "inference",
            "workload_params": None,
        },
        "candidate": candidate_section(
            model_hash=v1.model_hash,
            model_label=v1.model_label,
            engine_hash=engine_hash,
            runtime_label=v1.runtime,
            graph_precision=contract.graph_precision,
            cwd=cwd,
            root=root,
            engine_build=engine_build,
        ),
        "target": target_section(
            hardware_id=v1.hardware_id,
            metadata=metadata,
            clock_state=clock_state,
            cpu_affinity=v1.cpu_affinity,
            power_domain=hm.power_domain if hm else None,
            power_source=hm.power_source if hm else None,
            telemetry_measurement_boundary=telemetry_measurement_boundary,
            root=root,
        ),
        "protocol": protocol_section(protocol, contract),
        "capabilities": {
            "artifact_format": _ARTIFACT_FORMAT.get(family, "onnx"),
            "telemetry": _telemetry_captured(hm),
        },
        "metrics": _metrics_section(v1),
        "workload_metrics": {},
        "quality": {"quality_status": "NOT_EVALUATED"},
    }
    errors: list[str] = []
    if engine_io is not None:
        errors += [
            f"candidate.engine_build: engine I/O does not match the model ({problem})"
            for problem in engine_io_mismatches(engine_io, contract.io_signature)
        ]
    try:
        ResultV2.model_validate(raw)
    except ValidationError as exc:
        errors += [
            f"{'.'.join(str(p) for p in e['loc']) or '<result>'}: {e['msg']}" for e in exc.errors()
        ]
    return raw, errors
