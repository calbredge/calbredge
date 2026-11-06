import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from calbredge_protocol.stats import LatencyStats


class HardwareMetrics(BaseModel):
    """Power, thermal, and memory measurements captured during a benchmark run.

    All fields are optional so that partial hardware instrumentation is valid.
    Populated by the runtime adapter when supported hardware is present
    (e.g. Jetson via tegrastats, discrete NVIDIA GPU via nvidia-smi).
    Returns None-populated instance when hardware telemetry is unavailable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Power
    watts_average: float | None = None
    watts_peak: float | None = None
    joules_per_inference: float | None = None
    energy_joules_window: float | None = None
    """Time-integrated energy (Σ P·dt, joules) over the sampled measurement
    window. Populated only by a sampler that brackets the measurement loop
    directly (e.g. RKNNAdapter's external-meter :class:`PowerSampler`) —
    distinct from ``joules_per_inference``, which is derived from
    ``watts_average`` x p50 latency and is not replaced by this field."""

    # Thermal
    temperature_start_celsius: float | None = None
    temperature_end_celsius: float | None = None
    temperature_peak_celsius: float | None = None
    thermal_throttle_detected: bool | None = None

    # Memory
    gpu_memory_used_mb: float | None = None
    gpu_memory_total_mb: float | None = None
    system_memory_used_mb: float | None = None

    # Configuration
    power_mode: str | None = None
    precision: str | None = None
    power_source: str | None = None
    power_domain: str | None = None
    """What point in the power delivery chain was measured — e.g.
    ``"board_input"`` for an external meter on the 5V USB-C input (whole-board
    draw), as opposed to ``"module_rail"`` for an on-module rail sensor like
    Jetson's ``tegrastats`` (set since run bundles were introduced; earlier
    Jetson rows recorded ``None``). Distinct from
    ``power_source`` (which instrument measured it): two sources can share a
    domain, and results across different domains are not directly
    comparable even when both report a ``watts_average``."""


class BenchmarkProtocol(BaseModel):
    """Represents a benchmark job configuration loaded from a protocol YAML file.

    A BenchmarkProtocol defines *how* a benchmark run is conducted — iteration
    counts, input shape, batch size, etc. — but not *what* model is being
    benchmarked. The model path is supplied separately at job dispatch time.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol_version: str
    """Semantic version string identifying the protocol specification (e.g. "1.0").

    Must be declared in every protocol YAML. The version is pinned at
    job-definition time and stored in the resulting BenchmarkResult so that
    dashboard comparisons can enforce like-for-like protocol rules.
    """

    warmup_iterations: int = 50
    """Number of inference passes to run before measurements begin.

    Warmup passes are discarded and exist solely to prime the runtime (JIT
    compilation, memory allocation, cache warming). Must be >= 0; set to 0
    to disable warmup entirely.
    """

    measurement_iterations: int = 500
    """Number of inference passes whose latencies are recorded and reported.

    All p50/p90/p99 statistics are derived from exactly this many samples.
    Must be > 0.
    """

    batch_size: int = 1
    """Number of inputs processed per inference call.

    Affects throughput calculations and must match the batch dimension of
    input_shape[0] when the model has a fixed batch axis. Must be >= 1.
    """

    input_shape: list[int]
    """Tensor shape of a single input, expressed as a list of integers.

    Layout is model-dependent, not a fixed convention: most vision models are
    NCHW (e.g. [1, 3, 224, 224]), but TF/Keras and TFLite-origin exports —
    e.g. efficientnet_b0, efficientdet_lite0, cxr_mobilenetv2 — are NHWC
    (e.g. [1, 224, 224, 3]). This field stores the model's raw declared dim
    order as-is. Consumers must treat it as an ordered tuple and must not
    assume any fixed axis is the channel axis. All dimensions must be
    positive integers. The runtime adapter uses this shape to construct
    synthetic inputs during warmup and measurement.
    """

    power_mode: str = "default"
    """Power/performance mode hint passed to the runtime or device driver.

    Supported values are device-specific (e.g. "default", "max_performance",
    "power_save"). The protocol engine records the requested mode in
    BenchmarkResult but does not validate it against a fixed enum, as
    available modes vary by hardware.
    """

    runtime_config: dict = {}
    """Freeform runtime-specific settings forwarded verbatim to the RuntimeAdapter.

    Keys and values are adapter-defined (e.g. ONNX execution provider
    options, TensorRT precision flags). Defaults to an empty dict when no
    extra configuration is needed.
    """

    @field_validator("protocol_version")
    @classmethod
    def protocol_version_not_empty(cls, v: str) -> str:
        """Reject blank or whitespace-only version strings."""
        if not v.strip():
            raise ValueError("protocol_version must not be empty")
        return v

    @field_validator("warmup_iterations")
    @classmethod
    def warmup_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("warmup_iterations must be >= 0")
        return v

    @field_validator("measurement_iterations")
    @classmethod
    def measurement_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("measurement_iterations must be > 0")
        return v

    @field_validator("batch_size")
    @classmethod
    def batch_size_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("batch_size must be >= 1")
        return v

    @field_validator("input_shape")
    @classmethod
    def input_shape_valid(cls, v: list[int]) -> list[int]:
        """Require at least one dimension and all dimensions positive."""
        if not v:
            raise ValueError("input_shape must not be empty")
        if any(dim <= 0 for dim in v):
            raise ValueError("all input_shape dimensions must be positive integers")
        return v


def protocol_digest(protocol: BenchmarkProtocol) -> str:
    """SHA-256 over the RFC 8785 JSON of the validated protocol, defaults included.

    Formatting, comments, and whether a default is written out don't change it; any value
    that changes the measurement does. Recorded in v2 results (docs/RESULT_SCHEMA.md §3.5):
    two runs compare only when their digests are equal.
    """
    from calbredge_protocol.canonical import canonical_json, sha256_hex

    return sha256_hex(canonical_json(protocol.model_dump(mode="json")))


class BenchmarkResult(BaseModel):
    """Immutable record of a completed benchmark run.

    Produced by the agent after a full measurement cycle and uploaded to the
    API. Every field needed to reproduce, filter, and compare results on the
    dashboard is captured here. Instances are frozen — results may not be
    mutated after construction.

    Prefer constructing via the :meth:`from_run` class method, which assembles
    a ``BenchmarkResult`` from the component objects produced by the pipeline
    (protocol, latency stats, throughput, metadata) rather than requiring
    callers to unpack fields manually.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for this benchmark run. UUID4, auto-generated."""

    protocol_version: str
    """Protocol specification version used for this run (e.g. "1.0").

    Copied from ``BenchmarkProtocol.protocol_version``. Stored so the
    dashboard can enforce like-for-like comparisons across protocol versions.
    """

    input_shape: list[int] | None = None
    """Concrete input shape the run actually used, resolved from the model
    graph/engine by the adapter — NOT copied from the protocol file. Layout is
    model-dependent, not a fixed convention: most vision models are NCHW
    (e.g. [1, 3, 224, 224]), but TF/Keras and TFLite-origin exports — e.g.
    efficientnet_b0, efficientdet_lite0, cxr_mobilenetv2 — are NHWC (e.g.
    [1, 224, 224, 3]). This field stores the model's raw declared dim order
    as-is; consumers and any comparator must treat it as an ordered tuple and
    must not assume any fixed axis is the channel axis. None when the
    adapter had no safe binding source (e.g. RKNN), for a dynamic ONNX graph,
    or for results from agents predating this field. Comparability gate: two runs
    with different input_shape are not directly comparable. Stored as jsonb."""

    clock_state: dict | None = None
    """CPU/NPU/dmc governor and clock-pinning state captured at run time, as a
    platform-tagged dict (source: "rk3588" | "jetson"), or a
    {"captured": false, "reason": ...} sentinel when state is known-unknown, or
    None for results from agents predating this field. Comparability signal: two
    runs with different clock pinning are not directly comparable. Pinned-ness is
    derived by the comparator, not stored here. Stored as jsonb. Follows the
    plain-dict pattern of system_metadata — deliberately not a rigid sub-model,
    since the key set is platform-dependent."""

    cpu_affinity: dict | None = None
    """Host-thread CPU placement chosen by the agent's --cpu-affinity flag, as a
    plain dict, e.g. {"requested": "big", "applied": [4, 5, 6, 7], "tier": "big",
    "capacity": 1024} when pinned; {"requested": "none", "applied": null} when
    explicitly unpinned; {"requested": "big", "applied": null, "reason":
    "homogeneous CPU"} when big.LITTLE detection found nothing to pin against;
    or None for results from agents predating this field. Standalone from
    clock_state — different provenance (agent-set, every platform) vs
    clock_state's RKNN-collector governor data. Comparability signal: two runs
    with different affinity are not directly comparable. Stored as jsonb,
    same plain-dict pattern as clock_state/system_metadata."""

    hardware_id: str
    """SHA256 hardware fingerprint of the device that ran the benchmark.

    Computed by ``compute_hardware_id(metadata)``. Stable across reboots and
    runtime changes — same physical device always produces the same ID.
    """

    runtime: str
    """Canonical runtime adapter name (e.g. "onnxruntime", "tensorrt", "rknn")."""

    model_hash: str
    """SHA256 digest of the model file bytes.

    Used to deduplicate models in the API's model registry and to confirm
    that results from different devices were produced from the same weights.
    """

    model_label: str | None = None
    """Optional human-readable model name supplied by the user (e.g.
    "mobilenetv2-cxr"). Not validated or normalised — stored as-is."""

    latency_p50_ms: float
    """Median inference latency in milliseconds."""

    latency_p90_ms: float
    """90th-percentile inference latency in milliseconds."""

    latency_p99_ms: float
    """99th-percentile inference latency in milliseconds."""

    latency_mean_ms: float
    """Arithmetic mean inference latency in milliseconds."""

    latency_std_ms: float
    """Standard deviation of inference latency in milliseconds."""

    throughput_ips: float
    """Throughput in inferences per second, computed from p50 latency and batch size."""

    sample_count: int
    """Number of measurement samples used to compute latency statistics.

    Equals ``BenchmarkProtocol.measurement_iterations`` for a complete run.
    """

    system_metadata: dict
    """Serialized ``SystemMetadata`` captured at run time.

    Stored as a plain dict so ``BenchmarkResult`` has no dependency on the
    agent package. Callers should pass ``metadata.model_dump()``.
    """

    hardware_metrics: HardwareMetrics | None = None
    """Optional power, thermal, and memory telemetry collected during the run.

    Populated when the runtime adapter supports hardware instrumentation
    (e.g. Jetson via tegrastats, discrete GPU via nvidia-smi). None when
    the device does not expose hardware telemetry or the adapter does not
    implement ``collect_hardware_metrics``.
    """

    status: str | None = None
    """Run stability classification: "stable", "unstable", or "unknown".

    Derived from the p99/p50 latency ratio. A ratio > 2.5 indicates possible
    thermal throttling or system load interference. None if not yet classified.
    Serializes as null in JSON when not set.
    """

    status_reason: str | None = None
    """Human-readable explanation accompanying a non-stable status.

    Populated when status is "unstable" or "unknown". None for stable runs.
    Serializes as null in JSON when not set.
    """

    run_status: str = "stable"
    """Variance-based run classification: "stable", "inconclusive", or "unstable".

    Computed by :func:`~protocol.adapters.classify_run` from the p99/p50 latency
    ratio immediately after the measurement loop completes. Thresholds:
    - ratio > 3.0 → "unstable"
    - ratio > 2.0 → "inconclusive"
    - otherwise   → "stable"

    Defaults to "stable" so that results uploaded by older agent versions that
    predate this field are treated as stable rather than rejected.
    """

    run_status_reason: str | None = None
    """Human-readable explanation for a non-stable run_status.

    Populated when run_status is "unstable" or "inconclusive". None for stable
    runs. Serializes as null in JSON when not set.
    """

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    """UTC timestamp of when the benchmark result was assembled."""

    @field_validator("input_shape")
    @classmethod
    def input_shape_dims_positive(cls, v: list[int] | None) -> list[int] | None:
        if v is None:
            return v
        if not v:
            raise ValueError("input_shape must not be empty when provided")
        if any(dim <= 0 for dim in v):
            raise ValueError("all input_shape dimensions must be positive integers")
        return v

    @classmethod
    def from_run(
        cls,
        protocol: "BenchmarkProtocol",
        stats: "LatencyStats",
        throughput: float,
        hardware_id: str,
        runtime: str,
        model_hash: str,
        metadata: "dict | Any",
        model_label: str | None = None,
        hardware_metrics: "HardwareMetrics | None" = None,
        status: str | None = None,
        status_reason: str | None = None,
        run_status: str = "stable",
        run_status_reason: str | None = None,
        input_shape: list[int] | None = None,
        clock_state: dict | None = None,
        cpu_affinity: dict | None = None,
    ) -> "BenchmarkResult":
        """Construct a BenchmarkResult from pipeline component objects.

        This is the preferred constructor. It unpacks ``protocol`` and
        ``stats`` so callers do not need to reference individual field names.

        Args:
            protocol: The ``BenchmarkProtocol`` used for this run. Provides
                      ``protocol_version`` and ``sample_count`` via
                      ``measurement_iterations``.
            stats: ``LatencyStats`` returned by ``compute_latency_stats``.
            throughput: Inferences per second from ``compute_throughput``.
            hardware_id: SHA256 device fingerprint from ``compute_hardware_id``.
            runtime: Canonical runtime name (e.g. "onnxruntime").
            model_hash: SHA256 of the model file bytes.
            metadata: Either a ``SystemMetadata`` instance or an already-serialized
                      dict. Pydantic model instances are converted via
                      ``.model_dump()`` automatically.
            model_label: Optional human-readable model name. Defaults to None.

        Returns:
            A frozen :class:`BenchmarkResult` instance.
        """
        metadata_dict = metadata if isinstance(metadata, dict) else metadata.model_dump()
        return cls(
            protocol_version=protocol.protocol_version,
            hardware_id=hardware_id,
            runtime=runtime,
            model_hash=model_hash,
            model_label=model_label,
            latency_p50_ms=stats.p50_ms,
            latency_p90_ms=stats.p90_ms,
            latency_p99_ms=stats.p99_ms,
            latency_mean_ms=stats.mean_ms,
            latency_std_ms=stats.std_ms,
            throughput_ips=throughput,
            sample_count=stats.sample_count,
            system_metadata=metadata_dict,
            hardware_metrics=hardware_metrics,
            status=status,
            status_reason=status_reason,
            run_status=run_status,
            run_status_reason=run_status_reason,
            input_shape=input_shape,
            clock_state=clock_state,
            cpu_affinity=cpu_affinity,
        )
