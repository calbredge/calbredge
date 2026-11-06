"""Result schema v2: the model behind docs/RESULT_SCHEMA.md.

The doc and this module change together; tests/open/test_schema_doc_conformance.py
fails if a field exists in one but not the other.

v2 is a local artifact. It extends, and never replaces, the v1
``BenchmarkResult`` in protocol/calbredge_protocol/models.py, which stays the upload format;
``ResultV2.to_v1`` produces it.

Every model is strict (no type coercion: a float where a string belongs is an
error, never silently converted), frozen, and rejects unknown fields.
"""

import math
import re
import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator

from calbredge_protocol.canonical import (
    is_canonical_identifier,
    is_digest,
    is_git_commit,
    is_runtime_label,
    runtime_family,
)
from calbredge_protocol.models import BenchmarkResult, HardwareMetrics

# Schema versioning (docs/RESULT_SCHEMA.md §1): "MAJOR.MINOR". A minor bump
# only ever adds optional fields. This reader understands exactly these.
SCHEMA_MAJOR = 2
# 2.1: added optional candidate.engine_build.
# 2.2: added optional protocol.protocol_digest, and compatibility_contract.protocol_digest in
#      baseline declarations.
# 2.3: added optional candidate.engine_build.conversion_input, and
#      compatibility_contract.conversion_input in baseline declarations.
SCHEMA_MINOR = 3
SCHEMA_VERSION = f"{SCHEMA_MAJOR}.{SCHEMA_MINOR}"
_SCHEMA_VERSION_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")

STRICT_CONFIG = ConfigDict(strict=True, frozen=True, extra="forbid")


class SchemaVersionError(ValueError):
    """The record's schema_version is not one this Calbredge can read."""


def check_schema_version(raw: Any, kind: str) -> None:
    """Accept ``2.0`` (and older 2.x minors); reject everything else clearly.

    A newer minor is rejected rather than read with its new fields ignored:
    silently dropping fields a newer writer considered part of the record
    could change what a baseline or result means.
    """
    if not isinstance(raw, str) or not _SCHEMA_VERSION_RE.match(raw):
        raise SchemaVersionError(
            f"{kind}: schema_version must be a quoted 'MAJOR.MINOR' string, got {raw!r}"
        )
    major, minor = (int(p) for p in raw.split("."))
    if major != SCHEMA_MAJOR:
        raise SchemaVersionError(
            f"{kind}: schema_version {raw} is not supported "
            f"(this Calbredge reads {SCHEMA_MAJOR}.x only)"
        )
    if minor > SCHEMA_MINOR:
        raise SchemaVersionError(
            f"{kind}: schema_version {raw} was written by a newer Calbredge; "
            f"this one reads up to {SCHEMA_VERSION}. Upgrade Calbredge to read it."
        )


# --- constrained string types ---------------------------------------------


def _digest(v: str) -> str:
    if not is_digest(v):
        raise ValueError("must be a lowercase hex SHA-256 digest (64 characters, no prefix)")
    return v


def _git_commit(v: str) -> str:
    if not is_git_commit(v):
        raise ValueError("must be a 40-character lowercase hex git commit")
    return v


def _identifier(v: str) -> str:
    if not is_canonical_identifier(v):
        raise ValueError(
            f"{v!r} is not in canonical form (lowercase, [a-z0-9_], see RESULT_SCHEMA.md §2)"
        )
    return v


def _runtime_label(v: str) -> str:
    if not is_runtime_label(v):
        raise ValueError(f"{v!r} is not a canonical runtime label (lowercase, hyphen-qualified)")
    return v


def _version(v: str) -> str:
    # null, never a placeholder, when the version could not be determined.
    if not v.strip() or v.strip().lower() in ("unknown", "n/a"):
        raise ValueError(
            "must be a real version string; record null upstream rather than a placeholder"
        )
    return v


def _uuid4(v: str) -> str:
    try:
        parsed = uuid.UUID(v)
    except ValueError:
        raise ValueError("must be a UUID") from None
    if parsed.version != 4 or str(parsed) != v:
        raise ValueError("must be a lowercase, hyphenated UUID4")
    return v


def _utc_timestamp(v: str) -> str:
    if not (v.endswith("Z") or v.endswith("+00:00")):
        raise ValueError("must be an ISO 8601 UTC timestamp ending in 'Z' or '+00:00'")
    try:
        datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("must be an ISO 8601 UTC timestamp") from None
    return v


def _finite(v: float) -> float:
    if math.isnan(v) or math.isinf(v):
        raise ValueError("must be a finite number")
    return v


def _positive_dims(v: list[int]) -> list[int]:
    if not v or any(d <= 0 for d in v):
        raise ValueError("must be a non-empty list of positive integers")
    return v


Digest = Annotated[str, AfterValidator(_digest)]
GitCommit = Annotated[str, AfterValidator(_git_commit)]
Identifier = Annotated[str, AfterValidator(_identifier)]
RuntimeLabel = Annotated[str, AfterValidator(_runtime_label)]
Version = Annotated[str, AfterValidator(_version)]
UUID4 = Annotated[str, AfterValidator(_uuid4)]
UTCTimestamp = Annotated[str, AfterValidator(_utc_timestamp)]
Finite = Annotated[float, AfterValidator(_finite)]
Shape = Annotated[list[int], AfterValidator(_positive_dims)]

WorkloadType = Literal["single_forward_pass", "iterative_action_policy"]
WorkUnit = Literal["inference", "decision"]
RuntimeFamily = Literal["onnxruntime", "tensorrt", "rknn", "tflite"]
Precision = Literal["fp32", "fp16", "int8"]
PowerDomain = Literal["module_rail", "board_input"]
PowerSource = Literal["tegrastats", "external_meter"]
TelemetryBoundary = Literal["post_loop_snapshot", "windowed_bracket"]
CpuAffinityTier = Literal["big", "little", "custom", "none", "homogeneous"]
GraphKind = Literal["fixed_compute", "data_dependent"]

WORK_UNIT_FOR: dict[str, str] = {
    "single_forward_pass": "inference",
    "iterative_action_policy": "decision",
}
# workload_params keys an iterative_action_policy run must record; they are
# also always-required compatibility keys.
ITERATIVE_POLICY_PARAMS = (
    "scheduler",
    "denoising_steps",
    "action_horizon",
    "control_rate_hz",
    "replan_budget",
    "fidelity_metric",
)


def check_workload_params(workload_type: str, params: dict[str, Any] | None) -> None:
    if workload_type != "iterative_action_policy":
        return
    missing = [k for k in ITERATIVE_POLICY_PARAMS if params is None or params.get(k) is None]
    if missing:
        raise ValueError(f"iterative_action_policy requires workload_params {missing}")


# --- sections ---------------------------------------------------------------


class Workload(BaseModel):
    model_config = STRICT_CONFIG

    workload_type: WorkloadType
    work_unit: WorkUnit
    workload_params: dict[str, Any] | None

    @model_validator(mode="after")
    def _consistent(self) -> "Workload":
        expected = WORK_UNIT_FOR[self.workload_type]
        if self.work_unit != expected:
            raise ValueError(
                f"work_unit for {self.workload_type} must be {expected!r}, got {self.work_unit!r}"
            )
        check_workload_params(self.workload_type, self.workload_params)
        return self


def _sorted_unique(v: list[str]) -> list[str]:
    if v != sorted(set(v)):
        raise ValueError("must be sorted and without duplicates")
    return v


class ConversionInputTensor(BaseModel):
    """One model input as a converter handled it."""

    model_config = STRICT_CONFIG

    source_layout: Literal["nchw", "nhwc"] | None
    converted_shape: Annotated[list[int], AfterValidator(_positive_dims)] | None


class ConversionInput(BaseModel):
    """What a converted model takes at inference and how its inputs were converted (schema
    2.3): the part of a conversion that changes what a run measures. Two runs whose models
    were converted with different normalization, layout handling or feed dtype do different
    work, so this is a compatibility key (RESULT_SCHEMA.md §3.3.2). ``inputs`` is in
    the model's input order; ``graph_edits`` is sorted."""

    model_config = STRICT_CONFIG

    feed_dtype: str
    mean_values: list[list[int | float]]
    std_values: list[list[int | float]]
    inputs: list[ConversionInputTensor]
    graph_edits: Annotated[list[str], AfterValidator(_sorted_unique)]


class EngineBuild(BaseModel):
    """How a compiled engine was built (schema 2.1). Generic over compilers:
    compiler-specific fields are null where they don't apply."""

    model_config = STRICT_CONFIG

    build_key: Digest
    built_from_model_hash: Digest
    tensorrt_version: Version | None
    calibration_digest: Digest | None
    # 2.3, optional: set for engines converted off the device (RKNN); absent or null elsewhere.
    conversion_input: ConversionInput | None = None


class Candidate(BaseModel):
    model_config = STRICT_CONFIG

    model_hash: Digest
    model_label: str | None
    engine_hash: Digest | None
    git_commit: GitCommit | None
    git_dirty: bool | None
    runtime: RuntimeLabel
    runtime_family: RuntimeFamily
    runtime_version: Version
    driver_version: dict[str, str] | None
    precision: Precision
    # Added in 2.1: optional, so 2.0 records (which lack it) still load.
    engine_build: EngineBuild | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "Candidate":
        if (
            self.engine_build is not None
            and self.engine_build.built_from_model_hash != self.model_hash
        ):
            raise ValueError(
                "engine_build.built_from_model_hash does not match model_hash: "
                "the engine was built from a different model"
            )
        if runtime_family(self.runtime) != self.runtime_family:
            raise ValueError(
                f"runtime_family {self.runtime_family!r} does not match runtime {self.runtime!r}"
            )
        suffix = self.runtime.split("-", 1)[1] if "-" in self.runtime else None
        if suffix in ("fp32", "fp16", "int8") and suffix != self.precision:
            raise ValueError(
                f"precision {self.precision!r} contradicts runtime label {self.runtime!r}"
            )
        if (self.git_commit is None) != (self.git_dirty is None):
            raise ValueError(
                "git_commit and git_dirty are both set (inside a repository) or both null"
            )
        return self


class Target(BaseModel):
    model_config = STRICT_CONFIG

    hardware_id: Digest
    hardware_family: Identifier
    hardware_family_source: Literal["device_tree", "derived"]
    power_mode: Identifier
    power_domain: PowerDomain | None
    power_source: PowerSource | None
    telemetry_measurement_boundary: TelemetryBoundary | None
    clock_state: dict[str, Any] | None
    clocks_pinned: bool | None
    cpu_affinity: dict[str, Any] | None
    cpu_affinity_tier: CpuAffinityTier
    system_metadata: dict[str, Any]

    @model_validator(mode="after")
    def _consistent(self) -> "Target":
        if self.cpu_affinity_tier == "custom":
            applied = (self.cpu_affinity or {}).get("applied")
            if not isinstance(applied, list) or not applied:
                raise ValueError(
                    "cpu_affinity_tier 'custom' requires cpu_affinity.applied (the core list)"
                )
        return self


class TensorSpec(BaseModel):
    model_config = STRICT_CONFIG

    name: str
    dtype: str
    # Dynamic dimensions are recorded by their symbolic name (RESULT_SCHEMA.md §3.5).
    shape: list[int | str]


class IOSignature(BaseModel):
    model_config = STRICT_CONFIG

    inputs: list[TensorSpec]
    outputs: list[TensorSpec]


class ProtocolInfo(BaseModel):
    model_config = STRICT_CONFIG

    protocol_version: Version
    warmup_iterations: int
    measurement_iterations: int
    batch_size: int
    input_shape: Shape
    io_signature: IOSignature
    preprocessing: Identifier
    input_fixture_digest: Digest | None
    graph_kind: GraphKind
    # 2.2: the protocol the run used, by digest (calbredge_protocol.models.protocol_digest);
    # absent in results written before it.
    protocol_digest: Digest | None = None

    @model_validator(mode="after")
    def _ranges(self) -> "ProtocolInfo":
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be >= 0")
        if self.measurement_iterations <= 0:
            raise ValueError("measurement_iterations must be > 0")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        return self


class Capabilities(BaseModel):
    model_config = STRICT_CONFIG

    artifact_format: Literal["onnx", "engine", "rknn", "tflite"]
    telemetry: list[Literal["power", "thermal", "memory"]]

    @model_validator(mode="after")
    def _unique(self) -> "Capabilities":
        if len(set(self.telemetry)) != len(self.telemetry):
            raise ValueError("telemetry entries must be unique")
        return self


class Metrics(BaseModel):
    model_config = STRICT_CONFIG

    latency_p50_ms: Finite
    latency_p90_ms: Finite
    latency_p99_ms: Finite
    latency_mean_ms: Finite
    latency_std_ms: Finite
    sample_count: int
    throughput_ips: Finite
    watts_average: Finite | None
    watts_peak: Finite | None
    energy_joules_window: Finite | None
    joules_per_inference: Finite | None
    temperature_start_celsius: Finite | None
    temperature_end_celsius: Finite | None
    temperature_peak_celsius: Finite | None
    thermal_throttle_detected: bool | None
    gpu_memory_used_mb: Finite | None
    gpu_memory_total_mb: Finite | None
    system_memory_used_mb: Finite | None
    run_status: Literal["stable", "inconclusive", "unstable"]
    run_status_reason: str | None
    samples_pointer: Literal["samples.jsonl"]


class Quality(BaseModel):
    model_config = STRICT_CONFIG

    quality_status: Literal["NOT_EVALUATED", "OUTPUT_FIDELITY_ONLY", "CUSTOMER_VALIDATED"]


class ResultV2(BaseModel):
    """One benchmark run's measured evidence (docs/RESULT_SCHEMA.md)."""

    model_config = STRICT_CONFIG

    schema_version: str
    result_id: UUID4
    timestamp: UTCTimestamp
    workload: Workload
    candidate: Candidate
    target: Target
    protocol: ProtocolInfo
    capabilities: Capabilities
    metrics: Metrics
    workload_metrics: dict[str, Any]
    quality: Quality

    @model_validator(mode="before")
    @classmethod
    def _schema_version_first(cls, data: Any) -> Any:
        # Checked before field validation so a newer-schema record fails with
        # "upgrade Calbredge", not with a pile of unknown-field errors.
        if isinstance(data, dict):
            check_schema_version(data.get("schema_version"), "result")
        return data

    def to_v1(self) -> BenchmarkResult:
        """The v1 upload record for this result (RESULT_SCHEMA.md §5).

        v1's legacy ``status``/``status_reason`` are left unset here: they
        come from the agent's ``classify_stability`` (agent/calbredge/cli.py),
        which this package cannot import. The agent fills them when it builds
        the upload.
        """
        m, t = self.metrics, self.target
        hw = HardwareMetrics(
            watts_average=m.watts_average,
            watts_peak=m.watts_peak,
            joules_per_inference=m.joules_per_inference,
            energy_joules_window=m.energy_joules_window,
            temperature_start_celsius=m.temperature_start_celsius,
            temperature_end_celsius=m.temperature_end_celsius,
            temperature_peak_celsius=m.temperature_peak_celsius,
            thermal_throttle_detected=m.thermal_throttle_detected,
            gpu_memory_used_mb=m.gpu_memory_used_mb,
            gpu_memory_total_mb=m.gpu_memory_total_mb,
            system_memory_used_mb=m.system_memory_used_mb,
            # v1 records power_mode only where a mode exists (Jetson).
            power_mode=None if t.power_mode == "not_applicable" else t.power_mode,
            power_source=t.power_source,
            power_domain=t.power_domain,
        )
        return BenchmarkResult(
            run_id=self.result_id,
            protocol_version=self.protocol.protocol_version,
            input_shape=self.protocol.input_shape,
            # v1 records clock_state from the RK3588 collector only (RESULT_SCHEMA.md
            # §3.4, v1 column); v2's Jetson/Linux clock_state is v2-only.
            clock_state=t.clock_state
            if t.clock_state is not None and t.clock_state.get("source") == "rk3588"
            else None,
            cpu_affinity=t.cpu_affinity,
            hardware_id=t.hardware_id,
            runtime=self.candidate.runtime,
            model_hash=self.candidate.model_hash,
            model_label=self.candidate.model_label,
            latency_p50_ms=m.latency_p50_ms,
            latency_p90_ms=m.latency_p90_ms,
            latency_p99_ms=m.latency_p99_ms,
            latency_mean_ms=m.latency_mean_ms,
            latency_std_ms=m.latency_std_ms,
            throughput_ips=m.throughput_ips,
            sample_count=m.sample_count,
            # device_tree_model is v2-only; the upload payload stays exactly v1.
            system_metadata={
                k: v for k, v in t.system_metadata.items() if k != "device_tree_model"
            },
            hardware_metrics=None if hw == HardwareMetrics() else hw,
            run_status=m.run_status,
            run_status_reason=m.run_status_reason,
            timestamp=datetime.fromisoformat(self.timestamp.replace("Z", "+00:00")),
        )
