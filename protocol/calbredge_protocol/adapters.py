import platform
import re
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .models import BenchmarkProtocol, HardwareMetrics
from .stats import LatencyStats, compute_latency_stats


def _get_nvpmodel_mode() -> str | None:
    """Return the current Jetson nvpmodel power mode as a sanitized lowercase string.

    Runs ``nvpmodel -q``, parses the ``NV Power Mode: <NAME>`` line, and returns
    the name lowercased with non-alphanumeric characters replaced by underscores
    (e.g. ``"MAXN_SUPER"`` → ``"maxn_super"``).  Returns ``None`` on non-Jetson
    hardware, when ``nvpmodel`` is not installed, or when output cannot be parsed.
    """
    try:
        result = subprocess.run(
            ["nvpmodel", "-q"],  # noqa: S607 — hardcoded arg list, no shell, no user input
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None

    for line in result.stdout.splitlines():
        m = re.search(r"NV Power Mode:\s*(.+)", line)
        if m:
            raw = m.group(1).strip()
            sanitized = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
            return sanitized or None

    return None


def classify_run(p50: float, p99: float) -> tuple[str, str | None]:
    """Classify a benchmark run's variance into stable / inconclusive / unstable.

    Uses the p99/p50 latency ratio as a proxy for runtime consistency.
    A high ratio is a signature of thermal throttling, OS scheduling
    interference, or sustained resource contention during measurement.

    Thresholds:
        ratio > 3.0  → ``"unstable"``    (high variance, results unreliable)
        ratio > 2.0  → ``"inconclusive"`` (moderate variance, consider re-running)
        otherwise    → ``"stable"``

    Args:
        p50: Median latency in milliseconds.
        p99: 99th-percentile latency in milliseconds.

    Returns:
        ``(run_status, run_status_reason)`` — reason is ``None`` for stable runs.
    """
    if p50 <= 0:
        return "stable", None
    ratio = p99 / p50
    if ratio > 3.0:
        return (
            "unstable",
            f"p99/p50 ratio {ratio:.1f} — high variance detected, "
            "possible thermal throttling or system load interference",
        )
    if ratio > 2.0:
        return (
            "inconclusive",
            f"p99/p50 ratio {ratio:.1f} — moderate variance, consider re-running on an idle system",
        )
    return "stable", None


class RuntimeAdapter(ABC):
    """Abstract base class for Calbredge runtime adapters.

    A RuntimeAdapter encapsulates everything needed to load a model, run
    inference, and release resources for a specific inference runtime (e.g.
    ONNX Runtime, TensorRT, RKNN). Concrete subclasses implement the four
    abstract methods; the measurement loop is provided here as
    :meth:`run_benchmark` to guarantee every runtime is timed identically.

    Implementors must not override :meth:`run_benchmark`. All timing and
    statistical aggregation is handled by the base class to ensure
    cross-runtime comparability.
    """

    @abstractmethod
    def load_model(self, model_path: str | Path) -> None:
        """Load the model from disk and prepare it for inference.

        This method is called once before any warmup or measurement pass.
        Implementors should perform all one-time initialisation here: reading
        the model file, allocating runtime sessions, reserving GPU/NPU memory,
        and compiling or optimising the graph where applicable.

        After this method returns the adapter must be ready to accept calls to
        :meth:`warmup` and :meth:`run_inference` without further setup.

        Args:
            model_path: Filesystem path to the model file. The format is
                        runtime-specific (e.g. ``.onnx``, ``.trt``, ``.rknn``).

        Raises:
            FileNotFoundError: If the model file does not exist.
            RuntimeError: If the runtime fails to load or compile the model.
        """

    @abstractmethod
    def warmup(self, inputs: dict, iterations: int) -> None:
        """Run a number of warmup inference passes; results are discarded.

        Warmup primes the runtime for steady-state performance: JIT compilation,
        CUDA kernel launch overhead, CPU/GPU cache warming, and memory
        pre-allocation all happen here. Implementors must call
        :meth:`run_inference` (or the runtime's equivalent) exactly
        *iterations* times and discard all outputs.

        Implementors must not record latencies during warmup — timing must
        only occur inside :meth:`run_inference` as called by
        :meth:`run_benchmark`.

        Args:
            inputs: Runtime-specific input tensors keyed by input name.
            iterations: Number of warmup passes to perform. May be 0, in
                        which case this method must return immediately without
                        running any inference.
        """

    @abstractmethod
    def run_inference(self, inputs: dict) -> dict:
        """Execute a single inference pass and return the outputs.

        This method is the atomic unit of measurement. It must perform exactly
        one forward pass through the model and return the raw outputs. It must
        not include any pre- or post-processing beyond what the runtime itself
        performs internally.

        Implementors must not sleep, batch internally, or perform any operation
        unrelated to inference inside this method, as its wall-clock duration
        is recorded directly by :meth:`run_benchmark`.

        Args:
            inputs: Runtime-specific input tensors keyed by input name.

        Returns:
            A dict of output tensors keyed by output name. The exact types of
            the values are runtime-specific (e.g. numpy arrays, torch tensors).
        """

    @abstractmethod
    def unload(self) -> None:
        """Release the loaded model and free all runtime resources.

        Called once after all benchmark passes are complete. Implementors must
        release any GPU/NPU memory, close runtime sessions, and reset internal
        state so that :meth:`load_model` could be called again cleanly if
        needed.

        This method must not raise if called before :meth:`load_model` or
        after a previous call to :meth:`unload` — implementations should be
        idempotent.
        """

    resolved_input_shape: list[int] | None = None
    """Concrete input shape the adapter resolved from the loaded artifact at
    load time (e.g. [1, 3, 640, 640]), or None if it could not independently
    resolve one. Layout is model-dependent, not a fixed convention: most
    vision models are NCHW, but TF/Keras and TFLite-origin exports — e.g.
    efficientnet_b0, efficientdet_lite0, cxr_mobilenetv2 — are NHWC (e.g.
    [1, 224, 224, 3]). This is the model's raw declared dim order as-is;
    consumers must treat it as an ordered tuple and must not assume any
    fixed axis is the channel axis. Set by load_model in adapters that can
    read bound dims (ONNX, TensorRT). Left None where there is no safe
    binding source (RKNN) and by the mock adapter; the runner falls back to
    BenchmarkProtocol.input_shape. Resolved from the artifact, not copied
    from the protocol file."""

    @property
    def input_dtype(self) -> np.dtype:
        """Numpy dtype the runner should synthesize benchmark input tensors as.

        This is a property of the loaded model artifact, not the benchmark
        protocol — most runtimes (ONNX Runtime, TensorRT) take float32, so
        that is the base-class default. Adapters whose model artifact expects
        a different dtype (e.g. RKNNAdapter, whose quantized Rockchip-export
        image models take raw uint8 pixels) override this to match.
        """
        return np.dtype(np.float32)

    @staticmethod
    def _is_jetson() -> bool:
        """Return True if running on a Jetson device (aarch64 + Tegra device-tree)."""
        if platform.machine() != "aarch64":
            return False
        try:
            compatible = Path("/proc/device-tree/compatible").read_bytes()
            return b"tegra" in compatible
        except OSError:
            return False

    def collect_hardware_metrics(self) -> HardwareMetrics | None:
        """Collect power and thermal telemetry via tegrastats on Jetson hardware.

        Returns a :class:`HardwareMetrics` with watts and temperature data, or
        ``None`` on non-Jetson hardware or when tegrastats is not available.

        Also records ``self._last_telemetry_boundary``: ``"post_loop_snapshot"``
        when metrics are returned (tegrastats is read for 2 s after the timed
        loop ends, not during it), ``None`` otherwise. Read by the v2 result
        builder; RESULT_SCHEMA.md §5.4.
        """
        self._last_telemetry_boundary: str | None = None
        if not self._is_jetson():
            return None
        try:
            proc = subprocess.Popen(
                ["tegrastats", "--interval", "100"],  # noqa: S607 — hardcoded arg list, no shell, no user input
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except FileNotFoundError:
            return None
        try:
            time.sleep(2.0)
            proc.terminate()
            output, _ = proc.communicate(timeout=3)
        except Exception:
            proc.kill()
            return None
        lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
        if not lines:
            return None
        watts_samples: list[float] = []
        gpu_temps: list[float] = []
        cpu_temps: list[float] = []
        for line in lines:
            m = re.search(r"VDD_IN\s+(\d+)mW/\d+mW", line)
            if m:
                watts_samples.append(int(m.group(1)) / 1000.0)
            m = re.search(r"gpu@([\d.]+)C", line)
            if m:
                gpu_temps.append(float(m.group(1)))
            m = re.search(r"cpu@([\d.]+)C", line)
            if m:
                cpu_temps.append(float(m.group(1)))
        if not watts_samples and not gpu_temps and not cpu_temps:
            return None
        watts_average = sum(watts_samples) / len(watts_samples) if watts_samples else None
        watts_peak = max(watts_samples) if watts_samples else None
        all_temps = gpu_temps if gpu_temps else cpu_temps
        temperature_peak_celsius = max(all_temps) if all_temps else None
        thermal_throttle_detected = (
            (temperature_peak_celsius > 85.0) if temperature_peak_celsius is not None else None
        )
        metrics = HardwareMetrics(
            watts_average=watts_average,
            watts_peak=watts_peak,
            temperature_peak_celsius=temperature_peak_celsius,
            thermal_throttle_detected=thermal_throttle_detected,
            power_source="tegrastats",
            # tegrastats reports the module's VDD_IN rail, not whole-board
            # input power (compare RKNNAdapter's board_input external meter).
            power_domain="module_rail",
            power_mode=_get_nvpmodel_mode(),
        )
        self._last_telemetry_boundary = "post_loop_snapshot"
        return metrics

    def run_benchmark(self, inputs: dict, protocol: BenchmarkProtocol) -> LatencyStats:
        """Run the standardised benchmark loop and return latency statistics.

        This method is the single source of truth for how Calbredge measures
        inference latency. Subclasses may override to add sync guards or
        provider-specific timing. If overriding, call
        :meth:`collect_hardware_metrics` and return a :class:`LatencyStats`
        object.

        The sequence is:
        1. Call :meth:`warmup` with *protocol.warmup_iterations* to reach
           steady-state runtime conditions.
        2. Call :meth:`run_inference` exactly *protocol.measurement_iterations*
           times, recording the wall-clock duration of each call in milliseconds
           using :func:`time.perf_counter` (the highest-resolution timer
           available on the platform).
        3. Call :meth:`collect_hardware_metrics` to capture a post-run
           power/thermal snapshot. The result is stored on
           :attr:`_last_hardware_metrics` for the agent to attach to the
           ``BenchmarkResult``. If it returns ``None``, ``hardware_metrics``
           stays ``None`` — no change in behaviour on unsupported hardware.
        4. Pass the collected samples to :func:`compute_latency_stats` and
           return the result.

        Warmup latencies are never recorded. Only the measurement samples are
        passed to :func:`compute_latency_stats`.

        Args:
            inputs: Runtime-specific input tensors forwarded verbatim to
                    :meth:`warmup` and :meth:`run_inference`.
            protocol: Validated :class:`BenchmarkProtocol` that controls
                      iteration counts. ``protocol.warmup_iterations`` and
                      ``protocol.measurement_iterations`` are used directly.

        Returns:
            A :class:`LatencyStats` instance computed from
            *protocol.measurement_iterations* wall-clock samples.
        """
        self.warmup(inputs, protocol.warmup_iterations)

        samples: list[float] = []
        for _ in range(protocol.measurement_iterations):
            t0 = time.perf_counter()
            self.run_inference(inputs)
            t1 = time.perf_counter()
            samples.append((t1 - t0) * 1000.0)

        self._last_hardware_metrics: HardwareMetrics | None = self.collect_hardware_metrics()

        stats = compute_latency_stats(samples)
        self._last_run_status: str
        self._last_run_status_reason: str | None
        self._last_run_status, self._last_run_status_reason = classify_run(
            stats.p50_ms, stats.p99_ms
        )
        return stats
