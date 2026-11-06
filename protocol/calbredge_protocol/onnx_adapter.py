import subprocess
import time
from pathlib import Path

import numpy as np

from .adapters import RuntimeAdapter, classify_run
from .bundled import bundled_protocol_path, matching_bundled_protocol
from .models import BenchmarkProtocol, HardwareMetrics
from .stats import LatencyStats, compute_latency_stats


class ONNXRuntimeAdapter(RuntimeAdapter):
    """RuntimeAdapter implementation for ONNX Runtime inference.

    Uses ONNX Runtime's ``InferenceSession`` to load and execute ``.onnx``
    models. Execution provider selection is automatic:

    - **CUDAExecutionProvider** is used if a CUDA-capable GPU is detected by
      ONNX Runtime at session creation time.
    - **CPUExecutionProvider** is used as the fallback on all other hardware.

    No manual provider configuration is required for MVP use. Fine-grained
    provider options (e.g. CUDA device ordinal, memory limits, TensorRT
    integration) can be passed via ``BenchmarkProtocol.runtime_config`` in a
    future iteration.

    Requires ``onnxruntime`` (CPU) or ``onnxruntime-gpu`` (CUDA) to be
    installed in the active environment. The import is deferred to method
    call time so that the rest of the ``protocol`` package can be imported
    on machines where ONNX Runtime is not installed.
    """

    def __init__(self) -> None:
        self.session = None
        self.is_loaded: bool = False
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._cached_inputs: dict | None = None
        self._active_provider: str | None = None
        self._cuda_sync = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _import_ort():
        """Import and return the onnxruntime module, with a clear error on miss."""
        try:
            import onnxruntime as ort

            return ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed.\n"
                "Run: pip install onnxruntime\n"
                "     pip install calbredge[onnxruntime]  (if installed via pip)\n"
                "     pip install onnxruntime-gpu         (for CUDA support)"
            ) from exc

    def _get_cuda_sync(self):
        """Return a CUDA synchronization callable, or None if unavailable.

        Tries ``torch.cuda.synchronize`` first (preferred when PyTorch is
        present), then falls back to ``cudaDeviceSynchronize`` via ctypes and
        ``libcuda.so``. Returns ``None`` silently if neither path is available
        — CPU-only runs need no synchronization and must not be penalised.

        Only called from :meth:`load_model` when ``CUDAExecutionProvider`` is
        confirmed active.
        """
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.synchronize
        except ImportError:
            pass

        try:
            import ctypes

            libcuda = ctypes.CDLL("libcuda.so")
            return libcuda.cudaDeviceSynchronize
        except Exception:  # noqa: S110 — CUDA-sync capability probe, absence is the expected path
            pass

        return None

    @staticmethod
    def _has_nvidia_gpu() -> bool:
        """Return True if an NVIDIA GPU is detected via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],  # noqa: S607 — hardcoded arg list, no shell, no user input
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        except Exception:
            return False

    # ------------------------------------------------------------------
    # RuntimeAdapter interface
    # ------------------------------------------------------------------

    def load_model(self, model_path: str | Path) -> None:
        """Load an ONNX model and create an InferenceSession.

        Execution providers are selected automatically: CUDAExecutionProvider
        is preferred when available; CPUExecutionProvider is always included as
        the final fallback. ONNX Runtime silently skips any provider that is
        not available on the current machine, so specifying both in priority
        order is safe on all platforms.

        Stores input and output names from the session metadata for use in
        :meth:`run_inference`.

        Args:
            model_path: Path to the ``.onnx`` model file.

        Raises:
            RuntimeError: If ``onnxruntime`` is not installed.
            Exception: Any exception raised by ``ort.InferenceSession``
                       (e.g. invalid model file, unsupported opset).
        """
        ort = self._import_ort()

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(str(model_path), providers=providers)
        except Exception as exc:
            print(
                f"✗ Failed to load model: {exc}\n"
                "Common causes:\n"
                "- Model file is corrupted or incomplete\n"
                "- ONNX opset version not supported by this runtime version\n"
                "- Run: pip install --upgrade onnxruntime"
            )
            raise

        self._input_names = [inp.name for inp in self.session.get_inputs()]

        # Resolve the concrete input shape from the graph's declared dims.
        # Only set when every dim is a concrete positive int; a symbolic/None
        # (dynamic) axis means the graph does not pin the shape, so stay None
        # and let the runner fall back to the protocol shape.
        _dims = self.session.get_inputs()[0].shape
        self.resolved_input_shape = (
            [int(d) for d in _dims]
            if _dims and all(isinstance(d, int) and d > 0 for d in _dims)
            else None
        )
        self._output_names = [out.name for out in self.session.get_outputs()]
        self._active_provider = self.session.get_providers()[0]

        if self._active_provider == "CUDAExecutionProvider":
            self._cuda_sync = self._get_cuda_sync()
        else:
            self._cuda_sync = None
            if self._has_nvidia_gpu():
                if self._is_jetson():
                    print(
                        "Jetson detected — standard onnxruntime does not include "
                        "CUDA support. Install NVIDIA's Jetson build:\n"
                        "pip install https://pypi.jetson-ai-lab.io/jp6/cu126/+f/4eb/"
                        "e6a8902dc7708/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl\n"
                        "Then re-run your benchmark."
                    )
                else:
                    print(
                        "⚠ WARNING: CUDA is available on this system but CUDAExecutionProvider "
                        "failed to load. Running on CPU instead. This may indicate a missing cuDNN "
                        "installation. See: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html"
                    )
            else:
                print("ℹ No GPU detected — running on CPUExecutionProvider")

        self.is_loaded = True

    @staticmethod
    def _bind_shape(name: str, dims: list, requested: tuple[int, ...] | None) -> tuple[int, ...]:
        """The shape fed to one input: the model's static dimensions, each dynamic one bound.

        A dynamic dimension (``None``, a name, or ``<= 0``) takes the requested shape's value
        at that position, the same rule the TensorRT adapter builds its engine for, so the
        shape that runs is the shape the result records. Without a requested shape it is 1.

        Raises:
            RuntimeError: If the requested shape has another rank than the model's input, or
                differs from one of its static dimensions.
        """
        dynamic = [d is None or not isinstance(d, int) or d <= 0 for d in dims]
        if requested is None:
            return tuple(1 if dyn else d for d, dyn in zip(dims, dynamic, strict=True))

        def fix() -> str:
            """The refusal's advice: the bundled protocol for the model's input, by init's
            matching, when one exists."""
            match = matching_bundled_protocol(
                ["dynamic" if dyn else d for d, dyn in zip(dims, dynamic, strict=True)]
            )
            if match is None:
                return "Use a protocol whose input_shape matches"
            return (
                f"Use --protocol {bundled_protocol_path(match[0])}, the bundled protocol with "
                f"the model's shape {match[1]}, or another protocol whose input_shape matches"
            )

        if len(requested) != len(dims):
            raise RuntimeError(
                f"The protocol's input_shape {list(requested)} has {len(requested)} dimensions, "
                f"but the model's input {name!r} has {len(dims)}: {list(dims)}. {fix()} "
                "the model's input."
            )
        clashes = [
            j
            for j, (d, dyn, r) in enumerate(zip(dims, dynamic, requested, strict=True))
            if not dyn and d != r
        ]
        if clashes:
            raise RuntimeError(
                f"The protocol's input_shape {list(requested)} contradicts the model's input "
                f"{name!r}, {list(dims)}, at {'dimensions' if len(clashes) > 1 else 'dimension'} "
                f"{', '.join(str(j) for j in clashes)}, which the model fixes. {fix()} the "
                "model's static dimensions."
            )
        return tuple(r if dyn else d for d, dyn, r in zip(dims, dynamic, requested, strict=True))

    def _prepare_inputs(self, inputs: dict) -> dict:
        """Return a correctly-shaped synthetic input dict for this model.

        Shape is read from ``session.get_inputs()`` metadata on first call
        and cached for subsequent calls so that random array generation does
        not appear in the timed benchmark loop.

        The first input's shape is the model's, with each dynamic dimension
        (``None``, a name, or ``<= 0``) taken from the array *inputs* holds for
        it, which the runner builds from the protocol's ``input_shape``. A
        protocol shape that contradicts the model's rank or a static dimension
        is refused (:meth:`_bind_shape`). The shape that runs is recorded as
        ``resolved_input_shape``. Any other input, and the first one when
        *inputs* holds no array for it, takes 1 for a dynamic dimension.

        Only the shape of *inputs* is used: the benchmark feeds random data
        of the model's own input dtype.

        Args:
            inputs: Input dict keyed by ONNX input name, as passed to
                    :meth:`warmup` and :meth:`run_inference`.

        Returns:
            Dict mapping each ONNX input name to a random numpy array of the
            bound shape and the model's input dtype.

        Raises:
            RuntimeError: If the protocol's shape contradicts the model's input.
        """
        if self._cached_inputs is not None:
            return self._cached_inputs

        _DTYPE_MAP = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(uint8)": np.uint8,
        }

        result: dict = {}
        for i, inp in enumerate(self.session.get_inputs()):
            given = inputs.get(inp.name) if i == 0 else None
            requested = tuple(int(d) for d in np.shape(given)) if given is not None else None
            shape = self._bind_shape(inp.name, list(inp.shape), requested)
            dtype = _DTYPE_MAP.get(inp.type, np.float32)
            result[inp.name] = np.random.randn(*shape).astype(dtype)
            if i == 0:
                self.resolved_input_shape = list(shape)

        self._cached_inputs = result
        return result

    def warmup(self, inputs: dict, iterations: int) -> None:
        """Run *iterations* inference passes to prime the runtime; results discarded.

        Handles ``iterations=0`` cleanly by returning immediately. Inputs are
        normalised via :meth:`_prepare_inputs` before each pass.

        Args:
            inputs: Input dict keyed by ONNX input name, or positionally by
                    value count (see :meth:`_prepare_inputs`).
            iterations: Number of warmup passes. Must be >= 0.
        """
        prepared = self._prepare_inputs(inputs)
        for _ in range(iterations):
            self.session.run(self._output_names, prepared)
        if self._cuda_sync:
            self._cuda_sync()

    def run_inference(self, inputs: dict) -> dict:
        """Execute a single ONNX Runtime inference pass and return the outputs.

        Inputs are normalised via :meth:`_prepare_inputs` before the session
        call, so callers are not required to know the exact ONNX input names.

        Args:
            inputs: Input dict keyed by ONNX input name, or positionally by
                    value count (see :meth:`_prepare_inputs`).

        Returns:
            Dict mapping each output name to its corresponding numpy array.
        """
        prepared = self._prepare_inputs(inputs)
        outputs = self.session.run(self._output_names, prepared)
        return dict(zip(self._output_names, outputs))

    def run_benchmark(self, inputs: dict, protocol: BenchmarkProtocol) -> LatencyStats:
        """Run the benchmark loop with CUDA sync guards on GPU runs.

        Overrides the base-class loop to insert ``self._cuda_sync()`` between
        each ``run_inference`` call and the ``t1`` sample.  On CUDA runs this
        blocks the host thread until the GPU kernel has finished, so
        ``perf_counter`` captures actual GPU latency rather than kernel
        *submission* latency.  On CPU runs ``self._cuda_sync`` is ``None`` and
        the added branches are zero-cost.

        The post-warmup sync (inherited from :meth:`warmup`) ensures GPU clocks
        have settled before the first timed iteration begins.
        """
        self.warmup(inputs, protocol.warmup_iterations)

        samples: list[float] = []
        for _ in range(protocol.measurement_iterations):
            t0 = time.perf_counter()
            self.run_inference(inputs)
            if self._cuda_sync:
                self._cuda_sync()
            t1 = time.perf_counter()
            samples.append((t1 - t0) * 1000.0)

        self._last_hardware_metrics = self.collect_hardware_metrics()
        stats = compute_latency_stats(samples)
        if self._last_hardware_metrics and self._last_hardware_metrics.watts_average is not None:
            p50_seconds = stats.p50_ms / 1000.0
            self._last_hardware_metrics = HardwareMetrics(
                **{
                    **self._last_hardware_metrics.model_dump(),
                    "joules_per_inference": self._last_hardware_metrics.watts_average * p50_seconds,
                }
            )
        self._last_run_status, self._last_run_status_reason = classify_run(
            stats.p50_ms, stats.p99_ms
        )
        print(f"✓ Benchmark complete — provider: {self._active_provider}")
        return stats

    def unload(self) -> None:
        """Release the InferenceSession and reset internal state.

        Idempotent — safe to call before :meth:`load_model` or multiple times.
        ONNX Runtime releases session resources when the object is garbage
        collected; setting ``self.session = None`` drops the reference
        immediately.
        """
        self.session = None
        self._input_names = []
        self._output_names = []
        self._cached_inputs = None
        self._active_provider = None
        self._cuda_sync = None
        self.is_loaded = False

    # ------------------------------------------------------------------
    # Additional API
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """Return ``True`` if an InferenceSession is active."""
        return self.session is not None

    @property
    def input_names(self) -> list[str]:
        """Return the model's expected input names in session order.

        Returns an empty list before :meth:`load_model` is called or after
        :meth:`unload`. Use this to inspect expected names before constructing
        the inputs dict for :meth:`run_inference`.

        Example::

            adapter.load_model("model.onnx")
            print(adapter.input_names)  # e.g. ["input", "attention_mask"]
        """
        return list(self._input_names)
