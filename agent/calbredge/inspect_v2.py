"""`calbredge inspect`: collect the v2 candidate/target/protocol fields, read-only.

Runs every v2 collector and derivation a benchmark run would, without
loading the model into a runtime, pinning CPUs, benchmarking, uploading, or
reading ~/.calbredge/config.yaml. Used to validate the collectors on real
hardware (e.g. clocks_pinned on a Jetson with clocks pinned vs unpinned).

Fields a run sets only while measuring (engine_hash, power_domain,
power_source, telemetry_measurement_boundary) are null here and listed in
``inspect.not_collected``. Fields that depend on what a run would do are
predicted and flagged (``runtime_predicted``, ``cpu_affinity_predicted``).
"""

from pathlib import Path
from typing import Any

from calbredge_protocol.autodetect import detect_runtime, get_adapter
from calbredge_protocol.canonical import runtime_family
from calbredge_protocol.models import BenchmarkProtocol
from calbredge_protocol.result_v2 import SCHEMA_VERSION, Candidate, ProtocolInfo, Target
from pydantic import BaseModel, ValidationError

from calbredge.affinity import detect_clusters, resolve_affinity
from calbredge.clock_state import collect_clock_state
from calbredge.collector import collect_metadata
from calbredge.fingerprint import compute_hardware_id
from calbredge.hasher import compute_model_hash
from calbredge.onnx_contract import load_graph_contract
from calbredge.result_builder import candidate_section, protocol_section, target_section

NOT_COLLECTED = (
    "engine_hash",
    "engine_build",
    "power_domain",
    "power_source",
    "telemetry_measurement_boundary",
)
DEFAULT_CACHE_DIR = Path.home() / ".calbredge" / "contract-cache"


def predict_runtime_label(requested: str) -> tuple[str, bool]:
    """(label, predicted). Mirrors cli._resolve_recorded_runtime without loading.

    TensorRT/RKNN expose runtime_label on a freshly constructed adapter. ONNX
    Runtime's label depends on the provider a session actually picks, so it
    is predicted from the available providers.
    """
    label = getattr(get_adapter(requested), "runtime_label", None)
    if label is not None:
        return label, False
    if requested == "onnxruntime":
        try:
            import onnxruntime as ort
        except ImportError:
            return requested, True
        cuda = "CUDAExecutionProvider" in ort.get_available_providers()
        return ("onnxruntime-cuda" if cuda else "onnxruntime-cpu"), True
    return requested, False


def predict_affinity_record(spec: str, root: Path = Path("/")) -> dict[str, Any]:
    """The cpu_affinity record `run --cpu-affinity <spec>` would write, without pinning.

    Same shapes as agent/calbredge/cli.py's record (see derive_v2.cpu_affinity_tier).
    """
    clusters = detect_clusters(root)
    cores = resolve_affinity(spec, clusters)
    if cores is not None:
        tier = spec if spec in ("big", "little") else "custom"
        record: dict[str, Any] = {"requested": spec, "applied": sorted(cores), "tier": tier}
        if tier in ("big", "little") and clusters is not None:
            record["capacity"] = clusters["capacities"][next(iter(cores))]
        return record
    if spec == "none":
        return {"requested": "none", "applied": None}
    return {"requested": spec, "applied": None, "reason": "homogeneous CPU"}


def _validation(model_cls: type[BaseModel], data: dict[str, Any]) -> str | list[str]:
    try:
        model_cls.model_validate(data)
    except ValidationError as exc:
        return [
            f"{'.'.join(str(p) for p in e['loc']) or '<section>'}: {e['msg']}" for e in exc.errors()
        ]
    return "ok"


def inspect_model(
    model_path: Path,
    protocol: BenchmarkProtocol,
    *,
    runtime: str | None = None,
    cpu_affinity: str = "big",
    cwd: Path | None = None,
    root: Path = Path("/"),
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    requested = runtime or detect_runtime(model_path)
    label, runtime_predicted = predict_runtime_label(requested)

    model_hash = compute_model_hash(model_path)
    contract, cache_status = load_graph_contract(
        model_path, model_hash, cache_dir or DEFAULT_CACHE_DIR
    )
    candidate = candidate_section(
        model_hash=model_hash,
        model_label=None,
        engine_hash=None,
        runtime_label=label,
        graph_precision=contract.graph_precision,
        cwd=cwd or Path.cwd(),
        root=root,
    )
    metadata = collect_metadata(runtime_family(label))
    target = target_section(
        hardware_id=compute_hardware_id(metadata),
        metadata=metadata,
        clock_state=collect_clock_state(root),
        cpu_affinity=predict_affinity_record(cpu_affinity, root),
        power_domain=None,
        power_source=None,
        telemetry_measurement_boundary=None,
        root=root,
    )
    protocol_fields = protocol_section(protocol, contract)

    return {
        "schema_version": SCHEMA_VERSION,
        "inspect": {
            "model": str(model_path),
            "runtime_requested": requested,
            "runtime_predicted": runtime_predicted,
            "cpu_affinity_predicted": True,
            "graph_kind_triggers": contract.graph_kind_triggers,
            "contract_cache": cache_status,
            "not_collected": list(NOT_COLLECTED),
        },
        "candidate": candidate,
        "target": target,
        "protocol": protocol_fields,
        "v2_validation": {
            "candidate": _validation(Candidate, candidate),
            "target": _validation(Target, target),
            "protocol": _validation(ProtocolInfo, protocol_fields),
        },
    }
