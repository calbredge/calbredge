def compute_throughput(batch_size: int, p50_latency_ms: float) -> float:
    """Compute inference throughput in images (samples) per second.

    Formula::

        throughput = batch_size / (p50_latency_ms / 1000.0)

    **Why p50 and not mean?**
    Mean latency is skewed by outlier spikes — a single thermal-throttle event
    can inflate the mean by 10–20 %, producing a throughput figure that is
    neither achievable in practice nor representative of steady-state operation.
    p50 (the median) is robust to such outliers and reflects the latency the
    device sustains for at least half of all inference calls, making the
    resulting throughput number a stable, reproducible characteristic of the
    hardware/runtime combination under test.

    Args:
        batch_size: Number of inputs processed per inference call. Must be >= 1.
        p50_latency_ms: Median inference latency in milliseconds. Must be > 0.

    Returns:
        Throughput as a ``float`` in **images per second** (inferences × batch
        size per second).

    Raises:
        ValueError: If *batch_size* is less than 1.
        ValueError: If *p50_latency_ms* is <= 0.

    Example::

        stats = compute_latency_stats(measurement_samples)
        throughput = compute_throughput(protocol.batch_size, stats.p50_ms)
        print(f"{throughput:.1f} img/s")
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if p50_latency_ms <= 0:
        raise ValueError(f"p50_latency_ms must be > 0, got {p50_latency_ms}")

    return batch_size / (p50_latency_ms / 1000.0)
