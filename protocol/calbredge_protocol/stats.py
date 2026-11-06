from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LatencyStats:
    """Summary statistics derived from a set of inference latency samples.

    All timing fields are in milliseconds. Instances are frozen (immutable)
    once created.
    """

    p50_ms: float
    """Median latency — robust central tendency, unaffected by outliers."""

    p90_ms: float
    """90th-percentile latency — captures commonly degraded conditions such
    as background load or memory pressure that p50 misses."""

    p99_ms: float
    """99th-percentile latency — worst-case tail; the number that determines
    whether a system meets its real-time SLA."""

    mean_ms: float
    """Arithmetic mean latency. Included for completeness but less reliable
    than percentiles on edge hardware where thermal spikes skew the average."""

    std_ms: float
    """Standard deviation of latency samples. High std relative to mean_ms
    indicates unstable runtime conditions (thermal throttling, OS scheduling)."""

    min_ms: float
    """Fastest observed inference. Represents best-case performance under
    ideal conditions."""

    max_ms: float
    """Slowest observed inference. Together with p99_ms, identifies whether
    extreme outliers are isolated events or a persistent pattern."""

    sample_count: int
    """Number of measurement samples used to compute these statistics.
    Equals BenchmarkProtocol.measurement_iterations for a complete run."""

    samples_ms: tuple[float, ...] = ()
    """The per-iteration measurement samples these statistics were computed
    from, in measurement order, exactly as the timing loop recorded them.
    Carried here so the raw samples reach the v2 run bundle (samples.jsonl)
    without any change to the adapters' timing loops. Never serialized into
    the v1 BenchmarkResult: BenchmarkResult.from_run copies only the named
    summary fields."""


def compute_latency_stats(samples: list[float]) -> LatencyStats:
    """Compute p50/p90/p99 and summary statistics from inference latency samples.

    Args:
        samples: Latency measurements in **wall-clock milliseconds**, one value
                 per inference pass. Must contain only *measurement* samples —
                 warmup iterations must be excluded by the caller before passing
                 the list in. This function has no knowledge of the warmup
                 window and will treat every element as a valid measurement.

    Returns:
        A frozen :class:`LatencyStats` instance with percentile and summary
        statistics computed over all provided samples.

    Raises:
        ValueError: If *samples* is empty.

    Example::

        warmup_latencies = run_inference_n_times(protocol.warmup_iterations)
        measurement_latencies = run_inference_n_times(protocol.measurement_iterations)
        stats = compute_latency_stats(measurement_latencies)
        print(stats.p99_ms)
    """
    if not samples:
        raise ValueError(
            "samples must not be empty — warmup iterations must be excluded by the caller"
        )

    arr = np.array(samples, dtype=np.float64)

    return LatencyStats(
        p50_ms=float(np.percentile(arr, 50)),
        p90_ms=float(np.percentile(arr, 90)),
        p99_ms=float(np.percentile(arr, 99)),
        mean_ms=float(np.mean(arr)),
        std_ms=float(np.std(arr, ddof=0)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        sample_count=len(samples),
        samples_ms=tuple(float(s) for s in samples),
    )
