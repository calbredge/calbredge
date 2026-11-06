"""TensorRT runtime adapter for Calbredge.

Builds a TensorRT engine from a .onnx model file, caches it to disk, and runs
timed inference using the TensorRT 10.x Python API.

Requirements (target hardware only — not needed on dev machines):
  - NVIDIA Jetson (JetPack 6.x) or a CUDA-capable GPU with TensorRT SDK
  - ``tensorrt`` Python package (included in JetPack or TensorRT SDK)
  - ``pycuda`` Python package (pip install pycuda)

Both tensorrt and pycuda are imported lazily so the rest of Calbredge can
be imported on any machine without those packages installed.
"""

import time
from pathlib import Path

import numpy as np

from .adapters import RuntimeAdapter, _get_nvpmodel_mode, classify_run
from .canonical import canonical_json, sha256_hex
from .engine_cache import (
    EngineCacheError,
    build_key_inputs,
    cache_dirs,
    compute_build_key,
    engine_filename,
    find_cached,
    onnx_input_signature,
    resolve_input_profile,
    verify_record,
    write_cached,
)
from .models import BenchmarkProtocol, HardwareMetrics
from .stats import LatencyStats, compute_latency_stats

_PRECISION_VALUES = frozenset({"fp32", "fp16", "int8"})


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


# Bump whenever _build_engine's builder configuration changes (flags,
# workspace size, profile rule, network creation flags): it is part of the
# engine build key, so every previously cached engine is rebuilt.
TRT_BUILDER_CONFIG_VERSION = 1

# The INT8 calibration recipe (_NoiseCalibrator, n_calib=500, unseeded noise).
# Its digest is part of the build key; change it with the calibrator.
INT8_CALIBRATION = {"method": "random_normal", "n_calib": 500, "seeded": False}


def _trt_dtype_to_numpy(trt_dtype, trt):
    """Map a TensorRT DataType to the corresponding numpy dtype."""
    mapping = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT8: np.int8,
        trt.DataType.INT32: np.int32,
        trt.DataType.BOOL: np.bool_,
        trt.DataType.UINT8: np.uint8,
    }
    return mapping.get(trt_dtype, np.float32)


def _make_calibrator(trt, cuda, input_shapes: dict, input_dtypes: dict, n_calib: int = 500):
    """Return an IInt8EntropyCalibrator2 that uses random-noise inputs.

    ``input_shapes`` and ``input_dtypes`` are dicts keyed by binding name.
    The calibrator feeds ``n_calib`` random batches and then signals completion.
    """

    class _NoiseCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            # Pre-allocate one device buffer per input; reused across batches
            self._d_bufs = {}
            for name, shape in input_shapes.items():
                arr = np.zeros(shape, dtype=input_dtypes[name])
                self._d_bufs[name] = cuda.mem_alloc(arr.nbytes)
            # Pre-generate all calibration batches on the host
            self._batches = [
                {n: np.random.randn(*s).astype(input_dtypes[n]) for n, s in input_shapes.items()}
                for _ in range(n_calib)
            ]
            self._idx = 0

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self._idx >= len(self._batches):
                return None
            batch = self._batches[self._idx]
            self._idx += 1
            ptrs = []
            for name in names:
                cuda.memcpy_htod(self._d_bufs[name], np.ascontiguousarray(batch[name]))
                ptrs.append(int(self._d_bufs[name]))
            return ptrs

        def read_calibration_cache(self):
            return None

        def write_calibration_cache(self, cache):
            pass

    return _NoiseCalibrator()


class TensorRTAdapter(RuntimeAdapter):
    """RuntimeAdapter for TensorRT FP32 / FP16 / INT8 inference on NVIDIA hardware.

    Accepts a ``.onnx`` model file, builds (or loads from cache) a TensorRT
    engine for the requested precision, allocates pinned host and device
    buffers, and runs asynchronous inference on a CUDA stream.

    Engine files are cached alongside the model, keyed by both precision and the
    active nvpmodel power mode to avoid TensorRT warnings when the engine was
    built under a different clock configuration:
    ``model.onnx.trt.fp16.maxn_super``, ``model.onnx.trt.fp32.25w``, etc.
    The first run at each (precision, power-mode) combination compiles the engine
    (30–90 s); subsequent runs load from cache instantly.

    For INT8, calibration uses 500 random-noise inputs of the correct shape and
    dtype — fully reproducible, no external dataset required.  See
    ``docs/PROTOCOL.md §13`` for the rationale.

    Args:
        precision: One of ``"fp32"``, ``"fp16"``, or ``"int8"``.
    """

    def __init__(self, precision: str = "fp32") -> None:
        if precision not in _PRECISION_VALUES:
            raise ValueError(
                f"precision must be one of {sorted(_PRECISION_VALUES)}, got {precision!r}"
            )
        self._precision = precision
        self._engine = None
        self._context = None
        self._stream = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        # name → (host_ndarray, device_ptr)
        self._input_buffers: dict = {}
        self._output_buffers: dict = {}
        self.is_loaded: bool = False
        self._trt = None  # tensorrt module, set after import
        self._cuda = None  # pycuda.driver module, set after import

    # ------------------------------------------------------------------
    # Lazy imports
    # ------------------------------------------------------------------

    @staticmethod
    def _import_trt():
        try:
            import tensorrt as trt  # type: ignore

            return trt
        except ImportError as exc:
            raise RuntimeError(
                "tensorrt is not installed. TensorRT requires NVIDIA GPU hardware "
                "(Jetson with JetPack 6.x or a desktop GPU with TensorRT SDK). "
                "See https://developer.nvidia.com/tensorrt for installation instructions."
            ) from exc

    @staticmethod
    def _import_pycuda():
        try:
            import pycuda.autoinit  # noqa: F401 — initialises CUDA context
            import pycuda.driver as cuda  # type: ignore

            return cuda
        except ImportError as exc:
            raise RuntimeError(
                "pycuda is not installed. Install it with: pip install pycuda\n"
                "pycuda is required by the TensorRT adapter for CUDA memory management."
            ) from exc

    # ------------------------------------------------------------------
    # RuntimeAdapter interface
    # ------------------------------------------------------------------

    def load_model(  # type: ignore[override]
        self,
        model_path: str | Path,
        input_shape: list[int] | None = None,
        *,
        model_hash: str | None = None,
        input_signature: list[dict] | None = None,
        engine_cache_fallback_dir: Path | None = None,
    ) -> None:
        """Build or load a cached TensorRT engine from a .onnx file.

        The cache is content-addressed (protocol/calbredge_protocol/engine_cache.py): an engine is
        reused only if its build key, over the model's content hash, precision,
        power mode, resolved input profile, calibration, TensorRT version and
        TRT_BUILDER_CONFIG_VERSION, matches. Anything else builds a new engine
        under a new name; nothing is overwritten. ``model_hash`` and
        ``input_signature`` are computed from the file when not supplied.

        On first call for a given (model, precision) pair the ONNX model is
        compiled into a TensorRT engine — this can take 30–90 s. The serialised
        engine is saved to ``<model_path>.trt.<precision>`` so subsequent calls
        load from disk and complete in seconds.

        Args:
            model_path: Path to a ``.onnx`` model file.
            input_shape: Concrete input tensor shape (e.g. ``[1, 3, 224, 224]``)
                used to resolve dynamic dimensions when building the optimization
                profile.  Should match ``BenchmarkProtocol.input_shape``.
                If ``None``, dynamic dimensions fall back to ``1``.

        Raises:
            ValueError: If ``model_path`` does not have a ``.onnx`` extension.
            FileNotFoundError: If the model file does not exist.
            RuntimeError: If ``tensorrt`` or ``pycuda`` is not installed, if
                          engine compilation fails, or if FP16 or INT8 is
                          requested on a GPU without fast support for it.
        """
        model_path = Path(model_path)

        if model_path.suffix.lower() != ".onnx":
            raise ValueError(
                f"TensorRTAdapter requires a .onnx file but received {model_path.suffix!r}. "
                "Provide an ONNX model — TensorRT will compile it to an engine on first run."
            )
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        trt = self._import_trt()
        cuda = self._import_pycuda()
        self._trt = trt
        self._cuda = cuda
        # Before the cache lookup, so an engine cached on such a GPU is refused too.
        self._require_fast_precision(trt)

        power_mode = _get_nvpmodel_mode() or "default"
        if model_hash is None:
            model_hash = _sha256_file(model_path)
        if input_signature is None:
            try:
                input_signature = onnx_input_signature(model_path)
            except Exception as exc:
                raise RuntimeError(f"Failed to parse ONNX model {model_path.name}: {exc}") from exc
        trt_version = str(getattr(trt, "__version__", "unknown"))
        calibration_digest = (
            sha256_hex(canonical_json(INT8_CALIBRATION)) if self._precision == "int8" else None
        )
        key_inputs = build_key_inputs(
            artifact="tensorrt_engine",
            model_hash=model_hash,
            precision=self._precision,
            power_mode=power_mode,
            input_profile=resolve_input_profile(input_signature, input_shape),
            calibration_digest=calibration_digest,
            compiler_version=trt_version,
            builder_config_version=TRT_BUILDER_CONFIG_VERSION,
        )
        build_key = compute_build_key(key_inputs)
        filename = engine_filename(model_path, self._precision, power_mode, build_key)
        dirs = cache_dirs(model_path, engine_cache_fallback_dir)

        cached = find_cached(dirs, filename)
        if cached is not None:
            engine_path, record = cached
            verify_record(record, build_key)
            print(
                f"  Loading cached TensorRT engine: {engine_path.name} "
                f"(precision={self._precision}, power_mode={power_mode})"
            )
            logger = trt.Logger(trt.Logger.WARNING)
            trt_runtime = trt.Runtime(logger)
            with open(engine_path, "rb") as f:
                engine_bytes = f.read()
            self._engine = trt_runtime.deserialize_cuda_engine(engine_bytes)
        else:
            print(
                f"  Building TensorRT {self._precision.upper()} engine "
                f"for power mode {power_mode!r} (30–90 s on first run) …"
            )
            self._engine, engine_path = self._build_engine(
                trt,
                cuda,
                model_path,
                input_shape,
                dirs=dirs,
                filename=filename,
                build_key=build_key,
                key_inputs=key_inputs,
            )
        # The engine file actually loaded; hashed into the v2 result's engine_hash.
        self._engine_path = engine_path
        # What the engine was built from, recorded in the v2 result (candidate.engine_build).
        self._engine_build = {
            "build_key": build_key,
            "built_from_model_hash": model_hash,
            "tensorrt_version": trt_version,
            "calibration_digest": calibration_digest,
        }

        if self._engine is None:
            raise RuntimeError(
                f"TensorRT engine build failed for {model_path.name} "
                f"(precision={self._precision}). Check the log above for details."
            )

        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()

        # Discover input / output tensor names
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            else:
                self._output_names.append(name)

        # The engine's own I/O, cross-checked against the model's in the v2 result.
        self._engine_io = {
            "inputs": [self._tensor_spec(n, trt) for n in self._input_names],
            "outputs": [self._tensor_spec(n, trt) for n in self._output_names],
        }

        self._allocate_buffers(trt, cuda)
        self.is_loaded = True

    def _require_fast_precision(self, trt) -> None:
        """Refuse FP16 or INT8 on a GPU that TensorRT says has no fast support for it.

        The precision flag allows TensorRT to pick FP16 or INT8 kernels; it doesn't force
        them. On such a GPU the engine may run other kernels while the result records fp16
        or int8. A missing attribute is refused the same way: the support can't be checked.
        """
        attribute = {"fp16": "platform_has_fast_fp16", "int8": "platform_has_fast_int8"}.get(
            self._precision
        )
        if attribute is None:
            return
        builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
        supported = getattr(builder, attribute, None) is True
        del builder
        if not supported:
            name = self._precision.upper()
            raise RuntimeError(
                f"This GPU has no fast {name} support (TensorRT's {attribute} is not True), "
                f"so an {name} engine could run other kernels while the result records "
                f"{self._precision}. Refusing to build or load it. "
                "Run with --runtime tensorrt-fp32 on this GPU."
            )

    def _build_engine(
        self,
        trt,
        cuda,
        model_path: Path,
        input_shape: list[int] | None = None,
        *,
        dirs: list[Path],
        filename: str,
        build_key: str,
        key_inputs: dict,
    ):
        """Compile a TensorRT engine from an ONNX file and cache it (content-addressed).

        Returns (engine, engine_path).
        """
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        with open(model_path, "rb") as f:
            onnx_bytes = f.read()

        if not parser.parse(onnx_bytes):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(
                f"Failed to parse ONNX model {model_path.name}:\n" + "\n".join(errors)
            )

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        # Add an optimization profile so TRT can handle dynamic input shapes.
        # For each dimension that is dynamic (-1), use the corresponding value
        # from input_shape (falling back to 1) as the concrete min/opt/max.
        protocol_shape = input_shape or []
        profile = builder.create_optimization_profile()
        network_profile: dict = {}
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            concrete_shape = tuple(
                s if s > 0 else (protocol_shape[j] if j < len(protocol_shape) else 1)
                for j, s in enumerate(tensor.shape)
            )
            profile.set_shape(
                tensor.name,
                min=concrete_shape,
                opt=concrete_shape,
                max=concrete_shape,
            )
            network_profile[tensor.name] = {
                "min": list(concrete_shape),
                "opt": list(concrete_shape),
                "max": list(concrete_shape),
            }
        config.add_optimization_profile(profile)
        # The build key was computed from the model's ONNX input signature
        # before building; the profile TensorRT actually got must be the same,
        # or the key would not describe this engine.
        if network_profile != key_inputs["input_profile"]:
            raise EngineCacheError(
                f"input profile from the parsed network {network_profile} does not match "
                f"the build key's {key_inputs['input_profile']}; refusing to cache this engine"
            )

        # load_model has already refused FP16 and INT8 where the GPU has no fast support.
        if self._precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)

        elif self._precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)

            # Gather input metadata from the parsed network for calibration
            calib_shapes: dict = {}
            calib_dtypes: dict = {}
            for i in range(network.num_inputs):
                inp = network.get_input(i)
                shape = tuple(
                    s if s > 0 else (protocol_shape[j] if j < len(protocol_shape) else 1)
                    for j, s in enumerate(inp.shape)
                )
                calib_shapes[inp.name] = shape
                calib_dtypes[inp.name] = _trt_dtype_to_numpy(inp.dtype, trt)

            config.int8_calibrator = _make_calibrator(
                trt, cuda, calib_shapes, calib_dtypes, n_calib=500
            )

        try:
            serialized = builder.build_serialized_network(network, config)
        except Exception as exc:
            if self._precision == "int8":
                raise RuntimeError(
                    "INT8 engine build failed — random noise calibration may not be "
                    "suitable for this model architecture. Try tensorrt-fp16 instead."
                ) from exc
            raise

        if serialized is None:
            if self._precision == "int8":
                raise RuntimeError(
                    "INT8 engine build failed — random noise calibration may not be "
                    "suitable for this model architecture. Try tensorrt-fp16 instead."
                )
            raise RuntimeError(
                "TensorRT builder returned None — engine serialization failed. "
                "This usually means the model contains unsupported operators or the "
                "requested precision is not supported on this GPU."
            )

        # Persist to the content-addressed cache, atomically, with its build record.
        engine_path = write_cached(
            dirs,
            filename,
            bytes(serialized),
            build_key=build_key,
            inputs=key_inputs,
            extra={"network_input_profile": network_profile},
        )
        print(f"  Engine cached → {engine_path}")

        trt_runtime = trt.Runtime(logger)
        return trt_runtime.deserialize_cuda_engine(bytes(serialized)), engine_path

    def _tensor_spec(self, name: str, trt) -> dict:
        if self._engine is None:
            raise RuntimeError("no engine loaded")
        return {
            "name": name,
            "dtype": np.dtype(_trt_dtype_to_numpy(self._engine.get_tensor_dtype(name), trt)).name,
            "shape": [int(d) for d in self._engine.get_tensor_shape(name)],
        }

    def _allocate_buffers(self, trt, cuda) -> None:
        """Allocate host numpy arrays and device buffers for all I/O tensors.

        Input host buffers are pre-filled with random float32 noise.  The same
        buffers are reused for every inference call to avoid allocating inside
        the timed measurement loop.

        This method runs unconditionally after :meth:`load_model` obtains an
        engine via EITHER path (a fresh :meth:`_build_engine` compile or a
        cache-hit deserialize), with the engine already assigned to
        ``self._engine`` — making it the single place to resolve
        ``resolved_input_shape`` from the engine's own bound tensor shape so
        cached-engine runs record the same provenance as a fresh build,
        instead of falling back to the protocol shape every time after the
        first run of a given (model, precision, power_mode).
        """
        for idx, name in enumerate(self._input_names):
            raw_shape = tuple(self._engine.get_tensor_shape(name))
            if idx == 0:
                # Primary input only. A dynamic dim (TRT reports -1 here,
                # same convention as everywhere else in this file) means the
                # loaded engine does not pin the shape — leave None rather
                # than recording -1.
                self.resolved_input_shape = (
                    [int(d) for d in raw_shape] if all(d > 0 for d in raw_shape) else None
                )
            shape = tuple(1 if d <= 0 else d for d in raw_shape)
            dtype = _trt_dtype_to_numpy(self._engine.get_tensor_dtype(name), trt)
            h_buf = np.random.randn(*shape).astype(dtype)
            d_buf = cuda.mem_alloc(h_buf.nbytes)
            self._input_buffers[name] = (h_buf, d_buf)
            self._context.set_tensor_address(name, int(d_buf))

        for name in self._output_names:
            shape = tuple(1 if d <= 0 else d for d in self._engine.get_tensor_shape(name))
            dtype = _trt_dtype_to_numpy(self._engine.get_tensor_dtype(name), trt)
            h_buf = np.empty(shape, dtype=dtype)
            d_buf = cuda.mem_alloc(h_buf.nbytes)
            self._output_buffers[name] = (h_buf, d_buf)
            self._context.set_tensor_address(name, int(d_buf))

    def warmup(self, inputs: dict, iterations: int) -> None:
        """Queue *iterations* async inference passes; sync before returning.

        Input host buffers were pre-filled in :meth:`_allocate_buffers`.
        The ``inputs`` parameter is accepted for interface compatibility but
        the TRT adapter ignores it — it uses the pre-allocated device buffers.
        """
        if iterations == 0:
            return
        for name, (h_buf, d_buf) in self._input_buffers.items():
            self._cuda.memcpy_htod_async(d_buf, h_buf, self._stream)
        for _ in range(iterations):
            self._context.execute_async_v3(self._stream.handle)
        self._stream.synchronize()

    def run_inference(self, inputs: dict) -> dict:
        """Submit one async inference pass and queue output copies to host.

        Does *not* synchronise the stream.  The caller (:meth:`run_benchmark`)
        calls ``stream.synchronize()`` before stopping the timer so that the
        GPU-to-host copy is included in the measured latency.

        Returns:
            Dict of output name → host numpy array (may not be fully populated
            until the caller synchronises the stream).
        """
        for name, (h_buf, d_buf) in self._input_buffers.items():
            self._cuda.memcpy_htod_async(d_buf, h_buf, self._stream)
        self._context.execute_async_v3(self._stream.handle)
        for name, (h_buf, d_buf) in self._output_buffers.items():
            self._cuda.memcpy_dtoh_async(h_buf, d_buf, self._stream)
        return {name: h_buf for name, (h_buf, _) in self._output_buffers.items()}

    def run_benchmark(self, inputs: dict, protocol: BenchmarkProtocol) -> LatencyStats:
        """Run the benchmark loop with CUDA stream synchronization.

        Overrides the base-class loop to call ``stream.synchronize()`` between
        each :meth:`run_inference` submission and the ``t1`` sample, ensuring
        the timer captures actual GPU execution latency (including the
        device-to-host output copy) rather than just kernel submission latency.
        """
        self.warmup(inputs, protocol.warmup_iterations)

        samples: list[float] = []
        for _ in range(protocol.measurement_iterations):
            t0 = time.perf_counter()
            self.run_inference(inputs)
            self._stream.synchronize()
            t1 = time.perf_counter()
            samples.append((t1 - t0) * 1000.0)

        self._last_hardware_metrics: HardwareMetrics | None = self.collect_hardware_metrics()
        stats = compute_latency_stats(samples)

        if (
            self._last_hardware_metrics is not None
            and self._last_hardware_metrics.watts_average is not None
        ):
            p50_seconds = stats.p50_ms / 1000.0
            self._last_hardware_metrics = HardwareMetrics(
                **{
                    **self._last_hardware_metrics.model_dump(),
                    "joules_per_inference": self._last_hardware_metrics.watts_average * p50_seconds,
                }
            )

        self._last_run_status: str
        self._last_run_status_reason: str | None
        self._last_run_status, self._last_run_status_reason = classify_run(
            stats.p50_ms, stats.p99_ms
        )
        print(f"✓ Benchmark complete — TensorRT {self._precision.upper()}")
        return stats

    def unload(self) -> None:
        """Release the engine, context, stream, and CUDA buffers.

        Idempotent — safe to call before :meth:`load_model` or multiple times.
        Device memory is freed when the pycuda DeviceAllocation objects are
        garbage-collected; this method drops all references immediately.
        """
        self._context = None
        self._engine = None
        self._stream = None
        self._input_buffers = {}
        self._output_buffers = {}
        self._input_names = []
        self._output_names = []
        self._trt = None
        self._cuda = None
        self.is_loaded = False

    # ------------------------------------------------------------------
    # Additional API
    # ------------------------------------------------------------------

    @property
    def runtime_label(self) -> str:
        """Canonical runtime string for use in BenchmarkResult.runtime."""
        return f"tensorrt-{self._precision}"

    @property
    def input_names(self) -> list[str]:
        """Return model input tensor names (empty before :meth:`load_model`)."""
        return list(self._input_names)
