"""RKNN runtime adapter for Calbredge.

Loads a pre-converted .rknn model file for a Rockchip NPU (RK3588 and related
platforms) and runs timed inference using the RKNN-Toolkit-Lite2 Python API.

The adapter accepts an .onnx path from the benchmark runner and resolves it to a
sibling .rknn cache file named ``<model>.onnx.rknn.<precision>.<power_mode>``,
following the same convention as TensorRTAdapter.  Conversion of .onnx → .rknn
is a separate, out-of-band step handled by ``calbredge convert`` (which must run
on x86 — the RKNN-Toolkit2 conversion package is not available for aarch64).
If the expected cache file is not present, ``load_model`` raises with an
actionable error pointing at the conversion command.

Requirements (target hardware only — not needed on dev machines):
  - Rockchip NPU device (RK3588 Orange Pi 5 Plus is the reference platform)
  - ``rknn-toolkit-lite2`` Python package on the device (pip install rknn-toolkit-lite2)
  - Underlying ``librknnrt.so`` at v2.3.2 or newer — see docs/ORANGE_PI_SETUP.md
    for the runtime library replacement step, which is required on the Orange
    Pi vendor OS image (its bundled runtime is v1.4.0 and too old).

``rknnlite.api`` is imported lazily so the rest of Calbredge can be imported on
any machine without the package installed.

``collect_hardware_metrics`` reports power via an external KM003C USB meter
(see ``protocol/calbredge_protocol/power_meter.py``) bracketing the measured loop — RK3588 has
no on-chip rail telemetry equivalent to Jetson's tegrastats, so this is
whole-board input power (``power_domain="board_input"``), not module power.
The meter is optional reference instrumentation: with none attached,
``hardware_metrics`` stays ``None`` exactly as before, and the run is
unaffected.

Deferred (not implemented yet):
  - RK3588 thermal telemetry via /sys/class/thermal — temperature_*/
    thermal_throttle_detected stay None on every RKNN result regardless of
    whether a power meter is attached.
  - Full fp16-vs-quantized correctness canary: this version implements only the
    lightweight per-tensor nonzero-count check, which catches the specific
    silent-zero-output bug documented in TROUBLESHOOTING.md §7.  The full
    canary — running the fp16 baseline alongside and comparing outputs — is
    deferred to a follow-up implementation.

Correctness-canary generation history — see ``_run_correctness_canary``'s
docstring for the full account, but in short: an all-zero-input canary
(generation 1) could not separate healthy from broken; the fixed-seed-noise
canary that replaced it (generation 2) itself turned out to produce false
positives on legitimate sparse/large-object detection outputs (yolov8m INT8's
P5 class-confidence scale, found in a controlled on-device A/B). Generation 3
(current) prefers a real representative image when one is available via
``canary_image_path`` or the sibling-``dataset.txt`` convention, and demotes
the noise-only fallback to a loud warning instead of a hard failure.
"""

import platform
from pathlib import Path

import numpy as np

from . import rknn_artifact
from .adapters import RuntimeAdapter, classify_run
from .models import BenchmarkProtocol, HardwareMetrics
from .power_meter import PowerReading, PowerSampler
from .stats import LatencyStats, compute_latency_stats

_PRECISION_VALUES = frozenset({"fp16", "int8"})
# The target this adapter runs on, and the one `calbredge convert` builds for by default: a
# .rknn's build record must name it.
_TARGET_PLATFORM = "rk3588"

# Sysfs root defaults to "/" (the real filesystem). Tests redirect
# collect_clock_state's ``root`` argument to a tmp_path with the same
# relative layout so no test depends on real hardware.
_SYSFS_ROOT = Path("/")

_DEVICE_TREE_COMPATIBLE_RELPATH = "proc/device-tree/compatible"

# 3 clusters on RK3588: A55x4 (policy0), A76x2 (policy4), A76x2 (policy6).
_CPU_GOVERNOR_RELPATHS = {
    "policy0": "sys/devices/system/cpu/cpufreq/policy0/scaling_governor",
    "policy4": "sys/devices/system/cpu/cpufreq/policy4/scaling_governor",
    "policy6": "sys/devices/system/cpu/cpufreq/policy6/scaling_governor",
}
_NPU_GOVERNOR_RELPATH = "sys/class/devfreq/fdab0000.npu/governor"
_DMC_GOVERNOR_RELPATH = "sys/class/devfreq/dmc/governor"
# The GPU is idle while the NPU runs a model: its governor and frequency are recorded but
# clocks_pinned (protocol/calbredge_protocol/derive_v2.py) does not require them, and their
# absence must not fail the collector.
_GPU_GOVERNOR_RELPATH = "sys/class/devfreq/fb000000.gpu/governor"
# Per CPU policy, in kHz, beside scaling_governor. clocks_pinned compares scaling_max_freq
# with cpuinfo_max_freq (the hardware maximum), so these are required reads.
_CPU_FREQUENCY_FILES = (
    "scaling_cur_freq",
    "scaling_min_freq",
    "scaling_max_freq",
    "cpuinfo_max_freq",
)
# Per devfreq node (NPU, DMC, GPU), in Hz, beside governor. Recorded, not required.
_DEVFREQ_FREQUENCY_FILES = ("cur_freq", "max_freq")


def _read_governor(path: Path) -> str:
    """Read and strip a sysfs governor file's contents."""
    return path.read_text().strip()


def _read_devfreq_frequencies(governor_path: Path) -> dict[str, int | None]:
    """cur_freq and max_freq beside a devfreq governor file; null where one can't be read."""
    frequencies: dict[str, int | None] = {}
    for name in _DEVFREQ_FREQUENCY_FILES:
        try:
            frequencies[name] = int((governor_path.parent / name).read_text().strip())
        except (OSError, ValueError):
            frequencies[name] = None
    return frequencies


# Minimum acceptable ratio of nonzero elements per output tensor before the
# adapter treats a run as a silent quantization failure. Empirical anchor:
# during YOLOv8n INT8 validation on 2026-07-14, a broken stock-ONNX export
# produced classification-head outputs that were exactly 0/705,600 nonzero
# (see TROUBLESHOOTING.md §7). A healthy Rockchip-export model produced
# 100% nonzero on the same tensors. The 0.01 threshold catches the
# pathological case with wide margin while tolerating legitimately sparse
# outputs (e.g. objectness/confidence-sum tensors that are naturally near-zero
# for background regions).
#
# This threshold is still meaningful when the canary has a REAL representative
# image to run against. It is NOT reliable against synthetic noise: a
# controlled on-device A/B (2026-07-30)
# found yolov8m INT8's P5-scale class-confidence tensor (1, 80, 20, 20) scores
# a 0.0037 nonzero ratio on fixed-seed noise — below this threshold, so it
# would hard-fail — while the SAME artifact on a real calibration image
# reports a healthy max confidence of 0.9050 at that same tensor. Noise
# legitimately contains nothing resembling a large object, so a sparse
# large-object detection head correctly produces near-zero output on it; that
# is not evidence of a broken conversion. See
# ``RKNNAdapter._run_correctness_canary`` for how this threshold is now
# applied conditionally on whether real-image or noise input was used.
_CANARY_MIN_NONZERO_RATIO = 0.01


def _get_rk3588_power_mode() -> str:
    """Return a sanitised label for the current RK3588 power configuration.

    RK3588 has no vendor-provided named power modes (unlike Jetson's nvpmodel).
    Custom power tiers are engineered via sysfs — CPU governor writes and
    devfreq caps.  This helper is a placeholder that returns ``"default"``
    until custom power-profile support lands.

    Returning a stable label here allows the .rknn cache file naming to
    include a power-mode dimension from day one, so cached artefacts do not
    need to be renamed later when power-mode support is added.
    """
    # TODO(rk3588-power): Once custom power-profile support lands, read the
    # active profile marker — e.g. a file
    # written by the power-mode script into /run/calbredge/power_mode — and
    # return it here.  Falling back to "default" preserves cache validity for
    # runs done before power-mode integration.
    return "default"


class RKNNAdapter(RuntimeAdapter):
    """RuntimeAdapter for RKNN FP16 / INT8 inference on Rockchip NPU hardware.

    Accepts a ``.onnx`` model path from the benchmark runner and resolves it
    to a sibling pre-converted ``.rknn`` cache file:
    ``model.onnx.rknn.fp16.default``, ``model.onnx.rknn.int8.default``, etc.
    Unlike TensorRTAdapter, the adapter does not compile the model — the
    RKNN-Toolkit2 conversion package is x86-only and cannot run on the target
    device.  ``.rknn`` files must be produced ahead of time via
    ``calbredge convert``.

    On ``load_model`` the adapter also runs a lightweight correctness canary:
    after initialising the runtime, it performs one inference pass and
    verifies that every output tensor has at least
    ``_CANARY_MIN_NONZERO_RATIO`` nonzero elements. This catches the
    silent-quantization-failure mode documented in TROUBLESHOOTING.md §7 in
    which INT8-quantized YOLO-family models produce all-zero classification
    branches while reporting successful builds and fast, "stable" benchmark
    timing. The canary prefers a real representative image (via
    ``canary_image_path`` or a sibling ``dataset.txt``) when one is
    available, and only hard-fails on that real-image evidence; when no real
    image is available it falls back to fixed-seed random noise and reports
    any low-nonzero-ratio result as a warning rather than a hard failure —
    see ``_run_correctness_canary``'s docstring for why noise alone is not
    trustworthy evidence of a broken conversion.

    The canary's input shape is resolved from (in priority order) the
    ``expected_input_shape`` constructor arg, then the ``input_shape`` kwarg
    passed to ``load_model`` — which the benchmark runner populates from
    ``BenchmarkProtocol.input_shape``, the same value it uses to synthesize
    warmup/measurement inputs for every adapter (see agent/calbredge/cli.py).

    Args:
        precision: One of ``"fp16"`` or ``"int8"``.  Note that RKNN-Toolkit2
                   v2.3.2 does not expose INT4 as a documented quantization
                   target on RK3588; that dtype is intentionally omitted.
        canary_image_path: Optional path to a real representative image the
                   correctness canary should verify against instead of
                   synthetic noise. See ``_run_correctness_canary`` and
                   ``load_model``'s ``canary_image_path`` kwarg (explicit
                   constructor arg wins over the kwarg, same precedence as
                   ``expected_input_shape``).
    """

    def __init__(
        self,
        precision: str = "fp16",
        expected_input_shape: tuple[int, ...] | None = None,
        expected_input_dtype: str | np.dtype | None = None,
        canary_image_path: str | Path | None = None,
    ) -> None:
        if precision not in _PRECISION_VALUES:
            raise ValueError(
                f"precision must be one of {sorted(_PRECISION_VALUES)}, got {precision!r}. "
                "Note: RKNN-Toolkit2 v2.3.2 does not expose INT4 as a documented "
                "quantized_dtype on RK3588 — see TROUBLESHOOTING.md §7."
            )
        self._precision = precision
        # Canary input spec. Explicit args win over the model-registry lookup in
        # _resolve_canary_input_spec. There is deliberately no hardcoded default
        # shape here — see _run_correctness_canary for why an unknown shape must
        # fail closed (int8) or skip loudly (fp16) rather than guess.
        self._expected_input_shape: tuple[int, ...] | None = (
            tuple(expected_input_shape) if expected_input_shape is not None else None
        )
        self._expected_input_dtype: np.dtype | None = (
            _coerce_input_dtype(expected_input_dtype) if expected_input_dtype is not None else None
        )
        # Real-image override for the correctness canary (see
        # _run_correctness_canary). Explicit constructor arg wins over
        # load_model's canary_image_path kwarg, mirroring
        # _expected_input_shape's precedence rule below.
        self._canary_image_path: Path | None = (
            Path(canary_image_path) if canary_image_path is not None else None
        )
        self._rknn = None  # RKNNLite instance, populated in load_model
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self.is_loaded: bool = False
        self._rknnlite = None  # module reference, set after lazy import

    # ------------------------------------------------------------------
    # Lazy imports
    # ------------------------------------------------------------------

    @staticmethod
    def _import_rknnlite():
        try:
            from rknnlite.api import RKNNLite  # type: ignore

            return RKNNLite
        except ImportError as exc:
            raise RuntimeError(
                "rknn-toolkit-lite2 is not installed. The RKNN adapter requires "
                "an aarch64 Rockchip NPU device (e.g. Orange Pi 5 Plus / RK3588). "
                "On the device: pip install rknn-toolkit-lite2==2.3.2\n"
                "See docs/ORANGE_PI_SETUP.md for full setup, including the "
                "librknnrt.so runtime replacement step."
            ) from exc

    # ------------------------------------------------------------------
    # Cache resolution
    # ------------------------------------------------------------------

    def _verify_build_record(self, model_path: Path, rknn_path: Path) -> dict | None:
        """Check the .rknn against its build record; the result's ``engine_build``, or None.

        `calbredge convert` writes the record beside the .rknn
        (protocol/calbredge_protocol/rknn_artifact.py). With a record, the .rknn must be the file
        it names, built from the .onnx at ``model_path`` (which is hashed
        here), for this precision and target: a model re-exported after
        conversion is refused, never run. Without one the .rknn still loads,
        with a warning and no ``engine_build``, so its result does not show
        which model the .rknn was built from.

        Raises:
            ValueError: If the record is unreadable, or does not describe this
                        .rknn, this model or this run. The message names the
                        check that failed.
        """
        try:
            record = rknn_artifact.load_record(rknn_path)
            if record is None:
                print(
                    f"Warning: {rknn_path.name} has no build record "
                    f"({rknn_artifact.record_path(rknn_path).name}), so nothing shows which "
                    "model it was built from. It will run, but its result records no build for "
                    "it. Convert the model again with `calbredge convert` "
                    "and copy the .rknn and its record to this device."
                )
                return None
            rknn_artifact.verify_record(
                record,
                model_hash=rknn_artifact.file_sha256(model_path),
                artifact_sha256=rknn_artifact.file_sha256(rknn_path),
                precision=self._precision,
                target_platform=_TARGET_PLATFORM,
            )
        except rknn_artifact.RknnArtifactError as exc:
            raise ValueError(f"Refusing to load {rknn_path.name}: {exc}") from exc
        except FileNotFoundError as exc:
            raise ValueError(
                f"Refusing to load {rknn_path.name}: it has a build record, and "
                f"{model_path.name} is needed beside it to check which model it was built from"
            ) from exc
        inputs = record["inputs"]
        try:
            # How the model's inputs were converted: a compatibility key, since a different
            # normalization, layout handling or feed dtype is different measured work.
            conversion_input = {
                "feed_dtype": inputs["feed_dtype"],
                "mean_values": inputs["normalization"]["mean_values"],
                "std_values": inputs["normalization"]["std_values"],
                "inputs": [
                    {
                        "source_layout": entry["source_layout"],
                        "converted_shape": entry["converted_shape"],
                    }
                    for entry in inputs["model_inputs"]
                ],
                "graph_edits": sorted(set(inputs["graph_edits"])),
            }
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Refusing to load {rknn_path.name}: its build record does not say how the "
                f"model's inputs were converted ({exc!r}); convert the model again"
            ) from exc
        print(f"  Build record verified: {rknn_path.name} (build_key {record['build_key'][:16]})")
        return {
            "build_key": record["build_key"],
            "built_from_model_hash": inputs["model_hash"],
            "tensorrt_version": None,
            "calibration_digest": inputs.get("calibration_digest"),
            "conversion_input": conversion_input,
        }

    def _resolve_rknn_path(self, model_path: Path) -> Path:
        """Resolve the .onnx path to its sibling .rknn cache file.

        Cache pattern: ``<model_path>.rknn.<precision>.<power_mode>``.  Matches
        TensorRTAdapter's engine cache pattern for consistency.
        """
        power_mode = _get_rk3588_power_mode()
        return Path(str(model_path) + f".rknn.{self._precision}.{power_mode}")

    # ------------------------------------------------------------------
    # RuntimeAdapter interface
    # ------------------------------------------------------------------

    def load_model(
        self,
        model_path: str | Path,
        input_shape: list[int] | tuple[int, ...] | None = None,
        canary_image_path: str | Path | None = None,
    ) -> None:
        """Load a pre-converted .rknn cache file corresponding to the .onnx path.

        Does not compile the model — the RKNN-Toolkit2 conversion package is
        x86-only and cannot run on the target device.  If the expected .rknn
        cache file is not present, raises FileNotFoundError with a pointer to
        the conversion command.

        After loading, runs the correctness canary described in the class
        docstring.

        Args:
            model_path: Path to a ``.onnx`` model file.  The adapter resolves
                        this to a sibling ``.rknn`` cache file — the .onnx
                        itself is not read by this adapter (only its path is
                        used to derive the cache filename).
            input_shape: Concrete input tensor shape (e.g. ``[1, 224, 224, 3]``),
                         used to build the correctness canary's zero-filled
                         dummy input. Mirrors ``TensorRTAdapter.load_model``'s
                         ``input_shape`` parameter so the benchmark runner can
                         pass ``BenchmarkProtocol.input_shape`` uniformly to
                         both adapters. Ignored if ``expected_input_shape`` was
                         already supplied to the constructor — explicit
                         constructor args always win (see
                         ``_resolve_canary_input_spec``).
            canary_image_path: Path to a real representative image the
                         correctness canary should run against instead of
                         synthetic noise (see ``_run_correctness_canary``).
                         Ignored if ``canary_image_path`` was already supplied
                         to the constructor — explicit constructor args always
                         win, same precedence rule as ``input_shape`` above.
                         When neither is given, the canary falls back to the
                         sibling-``dataset.txt`` convention next to
                         ``model_path``, then to noise if that isn't found
                         either.

        Raises:
            ValueError: If ``model_path`` does not have a ``.onnx`` extension.
            FileNotFoundError: If the resolved .rknn cache file does not exist.
            RuntimeError: If ``rknn-toolkit-lite2`` is not installed, if the
                          runtime fails to load the model, or if the
                          correctness canary detects a degenerate output.
        """
        model_path = Path(model_path)

        if self._expected_input_shape is None and input_shape is not None:
            self._expected_input_shape = tuple(input_shape)

        if self._canary_image_path is None and canary_image_path is not None:
            self._canary_image_path = Path(canary_image_path)

        if model_path.suffix.lower() != ".onnx":
            raise ValueError(
                f"RKNNAdapter expects a .onnx path but received {model_path.suffix!r}. "
                "The .onnx path is used to locate the sibling .rknn cache file and to "
                "check it against the .rknn's build record."
            )

        rknn_path = self._resolve_rknn_path(model_path)
        # The converted model actually loaded; hashed into the v2 result's engine_hash.
        self._engine_path = rknn_path
        if not rknn_path.exists():
            raise FileNotFoundError(
                f"RKNN cache file not found: {rknn_path.name}\n"
                f"Run: calbredge convert --model {model_path.name} "
                f"--precision {self._precision} --target rk3588\n"
                "Conversion must be run on an x86 host — rknn-toolkit2 does not "
                "support aarch64. See docs/ORANGE_PI_SETUP.md for details."
            )
        # What the .rknn was built from, recorded in the v2 result (candidate.engine_build);
        # None when it has no build record. Checked before anything is loaded.
        self._engine_build = self._verify_build_record(model_path, rknn_path)

        RKNNLite = self._import_rknnlite()
        self._rknnlite = RKNNLite

        self._rknn = RKNNLite()
        ret = self._rknn.load_rknn(str(rknn_path))
        if ret != 0:
            raise RuntimeError(
                f"RKNNLite.load_rknn failed with code {ret} for {rknn_path.name}. "
                "The .rknn file may be corrupted or built for a different target "
                "platform. Re-run calbredge convert to regenerate it."
            )

        ret = self._rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(
                f"RKNNLite.init_runtime failed with code {ret}. "
                "Check that the NPU driver is loaded (dmesg | grep -i rknpu) and "
                "that the current user is a member of the 'render' group. "
                "See docs/ORANGE_PI_SETUP.md for driver setup."
            )

        # NOTE (2026-07 adapter review): input-metadata discovery used to call
        # self._rknn.rknn_runtime.get_input_attrs() behind a hasattr(self._rknn,
        # "rknn_runtime") guard. That guard was unsafe — it only confirmed the
        # attribute existed, not that get_input_attrs is a real method returning
        # dicts, so a mismatched real RKNNLite object would raise an uncaught
        # AttributeError instead of falling through cleanly. get_input_attrs is
        # also not a documented method of the Python RKNNLite API. The canary's
        # input shape/dtype now comes from explicit constructor args, the
        # runner's load_model(input_shape=...) kwarg, or _lookup_input_spec
        # (see _resolve_canary_input_spec) instead.
        #
        # TODO(rknn-input-attrs): re-add a runtime probe here, wrapped in
        # try/except so it can never crash load_model, only if the on-device
        # probe turns up a documented, dict-returning input-attr query.
        #
        # resolved_input_shape is intentionally left None (inherited from the
        # base class) until TODO(rknn-input-attrs) above lands — there is no
        # safe binding-shape source here, so the runner falls back to the
        # protocol shape.
        #
        # RKNN inference is positional, so a single input name is sufficient.
        self._input_names = ["0"]

        # Correctness canary. See class docstring and TROUBLESHOOTING.md §7.
        self._run_correctness_canary(rknn_path, model_path)

        self.is_loaded = True

    def _resolve_input_dtype(self) -> np.dtype:
        """Single source of truth for the dtype RKNN inference expects.

        Explicit ``expected_input_dtype`` (constructor arg) wins; otherwise
        defaults to uint8 — the standard image models this adapter targets
        take raw uint8 pixels. Shared by the :attr:`input_dtype` property
        (read by the benchmark runner to synthesize measurement/warmup
        inputs) and :meth:`_resolve_canary_input_spec` (used by the
        correctness canary), so the canary and the benchmark loop can never
        disagree about which dtype to feed the model — RKNN is strict about
        dtype, so a mismatch between the two would make a canary-passed model
        die (or produce garbage) on the first real inference.
        """
        return self._expected_input_dtype or np.dtype(np.uint8)

    @property
    def input_dtype(self) -> np.dtype:
        """Numpy dtype the benchmark runner should synthesize inputs as.

        Overrides :class:`RuntimeAdapter`'s float32 default. Resolved from
        :meth:`_resolve_input_dtype` — the same source the correctness canary
        uses — so this can never drift from what :meth:`load_model` already
        validated the model against.
        """
        return self._resolve_input_dtype()

    def _resolve_canary_input_spec(
        self, model_path: Path
    ) -> tuple[tuple[int, ...] | None, np.dtype | None]:
        """Resolve the (shape, dtype) used to build the canary's zero input.

        Priority: explicit constructor args, then the model-registry lookup in
        _lookup_input_spec. There is deliberately no hardcoded fallback shape —
        RKNN inference is lax about input shape but strict about dtype, so a
        wrong shape can be silently accepted and produce meaningless-but-nonzero
        output that passes the canary, defeating the exact check it exists for.
        A wrong dtype makes RKNN raise loudly instead, so once a shape is known,
        defaulting dtype to uint8 (via _resolve_input_dtype) is safe — the
        standard image models this adapter targets take raw uint8 pixels.
        """
        shape = self._expected_input_shape
        dtype = self._expected_input_dtype
        if shape is None:
            spec = self._lookup_input_spec(model_path)
            if spec is not None:
                shape, reg_dtype = spec
                if dtype is None:
                    dtype = reg_dtype
        if shape is not None and dtype is None:
            dtype = self._resolve_input_dtype()
        return shape, dtype

    @staticmethod
    def _lookup_input_spec(
        model_path: Path,
    ) -> tuple[tuple[int, ...], np.dtype] | None:
        """Look up (input_shape, numpy_dtype) for a model, keyed only by path.

        There is deliberately no model-registry-backed implementation here.
        Calbredge's only model_registry (the API's table) stores
        model_hash/model_label and has no shape or dtype columns, and adding
        one would create a second, driftable source of truth alongside
        BenchmarkProtocol.input_shape — the value the runner already resolves
        per run and threads through ``load_model(input_shape=...)`` (see
        agent/calbredge/cli.py). That runner-supplied value is read by
        _resolve_canary_input_spec before this method is ever consulted, so
        this always returns None: the canary either has a shape (constructor
        arg or load_model's input_shape) or it fails closed (int8) / skips
        loudly (fp16) — it never silently guesses one.
        """
        return None

    def _run_correctness_canary(self, rknn_path: Path, model_path: Path) -> None:
        """One-shot inference; verify each output has nonzero content.

        GENERATION 3 of this canary. History:

        Generation 1 used an all-zero input tensor. It could not separate
        healthy from broken: measured on a real broken artifact, zero input
        gives a 0.0038 vs 0.0 nonzero ratio for healthy vs. broken — not
        separable by any threshold.

        Generation 2 (fixed-seed random noise, ``np.random.default_rng(0)``)
        replaced it and fixed that specific problem. But a controlled
        on-device A/B (2026-07-30) then
        proved generation 2 has its OWN false-positive mode: yolov8m INT8's
        P5-scale class-confidence tensor (1, 80, 20, 20) scores a 0.0037
        nonzero ratio on fixed-seed noise — below _CANARY_MIN_NONZERO_RATIO,
        so generation 2 hard-failed it — while the SAME artifact, run on a
        real calibration image, reports a healthy max confidence of 0.9050 at
        that same tensor. yolov8n and yolov8s happened to produce nonzero
        values on the identical noise input at the same scale, but that is
        coincidence, not evidence of correctness: synthetic noise contains
        nothing resembling a large object, so a sparse large-object detection
        head legitimately produces near-zero output on it regardless of
        whether the conversion is healthy. Noise cannot reliably exercise
        that class of output at all.

        Generation 3 (this version) prefers a REAL representative image,
        resolved via (in order): ``canary_image_path`` (constructor arg or
        ``load_model`` kwarg), then a sibling ``dataset.txt`` next to
        ``model_path`` — the same calibration-dataset convention
        ``convert_rknn``/``check_detection_presence`` use at conversion time
        (agent/calbredge/converter.py's ``_first_calibration_image``: first
        non-blank line, relative entries resolve against the dataset file's
        own parent directory). When a real image resolves and loads
        successfully, the nonzero-ratio check runs exactly as before and
        still hard-fails (raises) below threshold — a collapse on real image
        content is genuine evidence, not a noise artifact.

        When no real image is available (no path given or resolvable, no
        sibling dataset.txt, Pillow not installed, or the image fails to
        load — not every deployment ships a calibration image next to the
        .rknn at load time), this falls back to generation 2's fixed-seed
        noise input, but the verdict is now WARN-ONLY: a low nonzero ratio on
        noise is loudly reported, never raised, since generation 2's own
        false-positive proves that verdict is not trustworthy evidence of a
        broken conversion by itself.

        This remains a coarse, whole-tensor nonzero-ratio check regardless of
        input source — it cannot catch a fused single-output model whose
        non-class channels (e.g. box regression) stay populated while only
        the class-confidence sub-range collapses to zero; that is exactly
        what happened on both on-device yolov8n int8 artifacts, and this
        canary passed anyway. The conversion-time detection-presence check in
        ``calbredge convert`` (``check_detection_presence``,
        agent/calbredge/converter.py) is what catches that pattern — it runs
        once at build time on the x86 host, asking the just-built model
        whether it reports a detection at all on a real calibration image.
        This load-time canary remains a cheap sanity screen, not a
        substitute for that check.

        The input shape/dtype needed to build the dummy tensor is resolved via
        _resolve_canary_input_spec (constructor args, then model registry) and
        is never guessed with a hardcoded fallback. When the shape is unknown:
          * int8 → raise. int8 is exactly where the silent-zero quantization
            failure lives, so loading it without the canary is unsafe.
          * fp16 → skip loudly. fp16 cannot exhibit the zeroing bug, so skipping
            is low-risk, but it's always announced, never silent.
        """
        shape, dtype = self._resolve_canary_input_spec(model_path)

        if shape is None:
            if self._precision == "int8":
                raise RuntimeError(
                    f"Refusing to load INT8 model {rknn_path.name}: the "
                    "correctness canary needs the model's input shape and none "
                    "was supplied. INT8 is exactly where the silent-zero "
                    "quantization failure occurs (TROUBLESHOOTING.md §7), so "
                    "loading it without the canary is unsafe. Pass "
                    "expected_input_shape=(...) to RKNNAdapter (e.g. "
                    "(1, 640, 640, 3) for YOLO, (1, 224, 224, 3) for MobileNet) "
                    "or register the model's input spec."
                )
            print(
                f"⚠ RKNN correctness canary SKIPPED for {rknn_path.name}: no "
                "input shape available. FP16 cannot exhibit the INT8 zeroing "
                "bug, so this is allowed — but pass expected_input_shape=(...) "
                "to enable the canary. Output names will populate on first "
                "inference."
            )
            return

        dummy, used_real_image, image_source = self._build_canary_input(shape, dtype, model_path)

        try:
            outputs = self._rknn.inference(inputs=[dummy])
        except Exception as exc:
            raise RuntimeError(
                f"Correctness canary failed to run inference on {rknn_path.name}: "
                f"{exc}. The model may expect a different input shape than the "
                f"canary's {image_source} input. If this is a legitimate case, "
                "check the model's input shape and layout (docs/CALBREDGE_DOCUMENTATION.md, "
                "'Input shapes, layout, and protocols'); the canary cannot currently be "
                "disabled."
            ) from exc

        if outputs is None or len(outputs) == 0:
            raise RuntimeError(
                f"Correctness canary: {rknn_path.name} loaded and ran, but "
                f"inference returned "
                f"{'None' if outputs is None else 'an empty output list'}. "
                "The cause is not determinable from here; known possibilities: "
                "the RKNN build produced no output nodes, a librknnrt.so version "
                "mismatch between build and runtime, or a target/model "
                "incompatibility that loads but does not execute. Re-run "
                "`calbredge convert` and confirm librknnrt.so matches the "
                "rknn-toolkit2 version used to build the .rknn."
            )

        # Record output tensor count for input_names/output_names symmetry.
        self._output_names = [str(i) for i in range(len(outputs))]

        # Verify every output tensor has meaningfully nonzero content.
        failing: list[tuple[int, np.ndarray, float]] = []
        for i, out in enumerate(outputs):
            if out.size == 0:
                continue  # empty tensor — no signal to check
            nonzero_ratio = float(np.count_nonzero(out)) / float(out.size)
            if nonzero_ratio < _CANARY_MIN_NONZERO_RATIO:
                failing.append((i, out, nonzero_ratio))

        if not failing:
            return

        detail = "\n".join(
            f"  - output[{i}] shape={out.shape}: {np.count_nonzero(out)}/{out.size} "
            f"nonzero (ratio {ratio:.4f} < threshold {_CANARY_MIN_NONZERO_RATIO})"
            for i, out, ratio in failing
        )

        if used_real_image:
            raise RuntimeError(
                f"Correctness canary FAILED for {rknn_path.name} (verified "
                f"against a real image: {image_source}):\n{detail}\n"
                "This is the silent-quantization-failure signature described "
                "in TROUBLESHOOTING.md §7 — the model was likely converted "
                "from a stock ONNX export whose fused sigmoid/DFL/concat "
                "output collapses under INT8 quantization. Re-export using "
                "Rockchip's modified graph (see rknn_model_zoo) or use fp16 "
                "precision. Benchmark results from this model would be "
                "silently invalid and are refused."
            )

        print(
            f"⚠ RKNN correctness canary: {rknn_path.name} has low/zero nonzero "
            f"content on some outputs, but verification used {image_source} — "
            "NOT a real representative image — so this is a WARNING, not a "
            "hard failure:\n"
            f"{detail}\n"
            "Synthetic noise cannot reliably exercise sparse/large-object "
            "detection outputs (a real on-device A/B found yolov8m INT8's "
            "P5 class-confidence tensor scores 0.0037 on noise vs. a healthy "
            "0.9050 max confidence on a real image; see docs/TROUBLESHOOTING.md, "
            "'Load-time canary false-positives on synthetic noise'), so a low "
            "ratio here is not trustworthy evidence "
            "of a broken conversion. Pass canary_image_path=(...) to "
            "RKNNAdapter, or place a dataset.txt next to the .onnx model "
            "(same convention convert_rknn uses), to verify against a real "
            "image instead."
        )

    def _build_canary_input(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype,
        model_path: Path,
    ) -> tuple[np.ndarray, bool, str]:
        """Resolve the canary's input tensor: a real image if one is
        available, else fixed-seed noise.

        Returns ``(array, used_real_image, source_description)``. Real-image
        resolution never raises — any failure along the way (no path
        resolvable, Pillow missing, image unreadable, an incompatible shape)
        prints a warning explaining why and falls back to noise, since "no
        calibration image at load time" is an expected, not exceptional,
        deployment state (see _run_correctness_canary's docstring).
        """
        image_path = _resolve_canary_image_path(model_path, self._canary_image_path)
        if image_path is None:
            return (
                _make_canary_dummy(shape, dtype),
                False,
                ("fixed-seed noise (np.random.default_rng(0))"),
            )

        if len(shape) != 4 or shape[-1] != 3:
            print(
                f"⚠ RKNN canary: found candidate image {image_path} but the "
                f"resolved input shape {shape} is not (N, H, W, 3) — real-image "
                "verification only supports 3-channel NHWC shapes. Falling "
                "back to noise."
            )
            return (
                _make_canary_dummy(shape, dtype),
                False,
                ("fixed-seed noise (np.random.default_rng(0))"),
            )

        image = _load_calibration_image_hwc(image_path, height=shape[1], width=shape[2])
        if image is None:
            return (
                _make_canary_dummy(shape, dtype),
                False,
                ("fixed-seed noise (np.random.default_rng(0))"),
            )

        dummy = image[np.newaxis, ...].astype(dtype)
        return dummy, True, str(image_path)

    def warmup(self, inputs: dict, iterations: int) -> None:
        """Run *iterations* warmup passes; results discarded.

        Handles ``iterations == 0`` by returning immediately.

        Args:
            inputs: Dict keyed by input name.  RKNNLite uses positional inputs,
                    so the values are passed in ``self._input_names`` order.
            iterations: Number of warmup passes.  Must be >= 0.
        """
        if iterations == 0:
            return
        input_list = self._prepare_input_list(inputs)
        for _ in range(iterations):
            self._rknn.inference(inputs=input_list)

    def run_inference(self, inputs: dict) -> dict:
        """Execute one inference pass and return outputs keyed by index.

        RKNN's Python API returns a list of numpy arrays without named outputs
        (unlike ONNX Runtime's dict).  This adapter keys them by their
        positional index as a string (``"0"``, ``"1"``, ...) to fit the
        RuntimeAdapter interface contract.

        Args:
            inputs: Dict keyed by input name (positional index as string).

        Returns:
            Dict of output index (as string) → numpy array.
        """
        input_list = self._prepare_input_list(inputs)
        outputs = self._rknn.inference(inputs=input_list)
        return {str(i): out for i, out in enumerate(outputs)}

    def _prepare_input_list(self, inputs: dict) -> list:
        """Convert the input dict to the positional list RKNNLite expects."""
        if not self._input_names:
            # We never discovered input names — just pass values in dict order
            return list(inputs.values())
        return [inputs[name] for name in self._input_names]

    def unload(self) -> None:
        """Release the RKNNLite runtime.  Idempotent."""
        if self._rknn is not None:
            try:
                self._rknn.release()
            except Exception:  # noqa: S110 — best-effort teardown, must not propagate
                # release() is best-effort; failures during teardown must not
                # propagate.
                pass
        self._rknn = None
        self._rknnlite = None
        self._input_names = []
        self._output_names = []
        self.is_loaded = False

    # ------------------------------------------------------------------
    # Hardware metrics
    # ------------------------------------------------------------------

    # TODO(rk3588-metrics): collect_hardware_metrics below covers power via
    # the external KM003C USB meter but not
    # thermal — RK3588 thermal telemetry from /sys/class/thermal is still
    # unimplemented, so temperature_*/thermal_throttle_detected stay None on
    # every RKNN result regardless of whether a meter is attached.

    def collect_hardware_metrics(
        self, power_reading: PowerReading | None = None
    ) -> HardwareMetrics | None:
        """Build HardwareMetrics from an external-meter reading, if any.

        Unlike the Jetson/base-class override, this has no way to sample
        hardware on its own — RK3588 has no sysfs power rail equivalent to
        tegrastats, and the external meter only makes sense read from a
        background thread bracketing the measurement loop (see
        :class:`~protocol.power_meter.PowerSampler`). ``run_benchmark`` below
        owns that bracketing and passes the resulting ``power_reading`` in.

        Called with no argument — directly, or via any caller relying on the
        base-class no-arg contract — this returns ``None``, identical to
        today's inherited behaviour, since there is no bracketed window to
        report on. It also returns ``None`` whenever ``power_reading`` itself
        is ``None`` (no meter was attached, pyusb missing, or a USB error —
        see PowerSampler.start), which is the normal, fully-supported
        end-user case: the run completes exactly as it does without a meter.

        Records ``self._last_telemetry_boundary``: ``"windowed_bracket"`` when
        a reading is returned (the sampler brackets the timed loop), ``None``
        otherwise. RESULT_SCHEMA.md §5.4.
        """
        self._last_telemetry_boundary: str | None = None
        if power_reading is None:
            return None
        metrics = HardwareMetrics(
            watts_average=power_reading.watts_average,
            watts_peak=power_reading.watts_peak,
            energy_joules_window=power_reading.energy_joules_window,
            power_source="external_meter",
            # Whole-board draw at the 5V USB-C input — NOT a module/rail
            # reading like Jetson's tegrastats. Stored, not just commented,
            # so Jetson-vs-RK3588 power comparisons stay honest.
            power_domain="board_input",
        )
        self._last_telemetry_boundary = "windowed_bracket"
        return metrics

    # ------------------------------------------------------------------
    # Clock state
    # ------------------------------------------------------------------

    @staticmethod
    def _is_rk3588(root: Path = _SYSFS_ROOT) -> bool:
        """Return True if running on an RK3588 device (aarch64 + rk3588 device-tree).

        Mirrors :meth:`RuntimeAdapter._is_jetson`'s pattern. ``root`` is the
        sysfs/procfs root to check under — defaults to ``/`` (the real
        filesystem) and is overridden by tests to point at a tmp_path.
        """
        if platform.machine() != "aarch64":
            return False
        try:
            compatible = (root / _DEVICE_TREE_COMPATIBLE_RELPATH).read_bytes()
            return b"rk3588" in compatible
        except OSError:
            return False

    def collect_clock_state(self, root: Path = _SYSFS_ROOT) -> dict | None:
        """Collect CPU/NPU/dmc/GPU governor and frequency state from RK3588 sysfs.

        Returns ``None`` on non-RK3588 hardware (no-op — Jetson/x86 runs are
        unaffected). On RK3588, returns either the populated dict:

            {"captured": True, "source": "rk3588",
             "cpu_governors": {"policy0": ..., "policy4": ..., "policy6": ...},
             "cpu_frequencies": {"policy0": {"scaling_cur_freq": ..., "scaling_min_freq": ...,
                                             "scaling_max_freq": ..., "cpuinfo_max_freq": ...},
                                 ...},
             "npu_governor": ..., "npu_frequency": {"cur_freq": ..., "max_freq": ...},
             "dmc_governor": ..., "dmc_frequency": {...},
             "gpu_governor": ..., "gpu_frequency": {...}}

        or, if any *required* read fails (file missing, permission denied,
        unreadable, not a number), the sentinel ``{"captured": False,
        "reason": ...}`` rather than a partial dict — values are never
        fabricated or defaulted. Required: every CPU policy's governor and
        frequencies, and the NPU and DMC governors (what clocks_pinned is
        derived from). CPU frequencies are kHz, devfreq frequencies Hz. A
        devfreq frequency that can't be read is recorded as null. The GPU is
        optional: if its governor can't be read, the populated dict is still
        returned, without ``gpu_governor`` and ``gpu_frequency``.

        Args:
            root: Sysfs/procfs root to read under. Defaults to ``/`` (the
                  real filesystem); tests redirect this to a tmp_path with
                  the same relative layout.
        """
        if not self._is_rk3588(root):
            return None

        try:
            cpu_governors = {
                policy: _read_governor(root / relpath)
                for policy, relpath in _CPU_GOVERNOR_RELPATHS.items()
            }
            npu_governor = _read_governor(root / _NPU_GOVERNOR_RELPATH)
            dmc_governor = _read_governor(root / _DMC_GOVERNOR_RELPATH)
        except OSError as exc:
            return {
                "captured": False,
                "reason": f"failed to read {exc.filename}: {exc.strerror or exc}",
            }

        cpu_frequencies: dict[str, dict[str, int]] = {}
        for policy, relpath in _CPU_GOVERNOR_RELPATHS.items():
            cpu_frequencies[policy] = {}
            for name in _CPU_FREQUENCY_FILES:
                path = (root / relpath).parent / name
                try:
                    cpu_frequencies[policy][name] = int(path.read_text().strip())
                except (OSError, ValueError) as exc:
                    return {"captured": False, "reason": f"failed to read {path}: {exc}"}

        state: dict = {
            "captured": True,
            "source": "rk3588",
            "cpu_governors": cpu_governors,
            "cpu_frequencies": cpu_frequencies,
            "npu_governor": npu_governor,
            "npu_frequency": _read_devfreq_frequencies(root / _NPU_GOVERNOR_RELPATH),
            "dmc_governor": dmc_governor,
            "dmc_frequency": _read_devfreq_frequencies(root / _DMC_GOVERNOR_RELPATH),
        }
        try:
            state["gpu_governor"] = _read_governor(root / _GPU_GOVERNOR_RELPATH)
        except OSError:
            pass
        else:
            state["gpu_frequency"] = _read_devfreq_frequencies(root / _GPU_GOVERNOR_RELPATH)
        return state

    # ------------------------------------------------------------------
    # run_benchmark override — carries joules_per_inference calc
    # ------------------------------------------------------------------

    def run_benchmark(self, inputs: dict, protocol: BenchmarkProtocol) -> LatencyStats:
        """Run the benchmark loop and attach joules_per_inference when possible.

        Mirrors the base-class loop exactly (RKNN's inference call is fully
        synchronous, so no stream-sync gymnastics like TensorRT needs), then
        derives joules_per_inference from the p50 latency and any watts_average
        that ``collect_hardware_metrics`` returned.

        A :class:`~protocol.power_meter.PowerSampler` is started immediately
        after warmup and stopped immediately after the last measured
        iteration, so it brackets exactly the 500 measured iterations and
        never includes warmup. ``PowerSampler.start`` is a no-op (returns
        ``False``, samples nothing) whenever no KM003C meter is attached —
        the loop below runs identically either way.
        """
        import time as _time  # local import to avoid a global time dependency

        self.warmup(inputs, protocol.warmup_iterations)

        sampler = PowerSampler()
        sampler.start()

        samples: list[float] = []
        for _ in range(protocol.measurement_iterations):
            t0 = _time.perf_counter()
            self.run_inference(inputs)
            t1 = _time.perf_counter()
            samples.append((t1 - t0) * 1000.0)

        power_reading = sampler.stop()

        self._last_hardware_metrics: HardwareMetrics | None = self.collect_hardware_metrics(
            power_reading
        )
        self._last_clock_state: dict | None = self.collect_clock_state()
        stats = compute_latency_stats(samples)

        if (
            self._last_hardware_metrics is not None
            and self._last_hardware_metrics.watts_average is not None
        ):
            p50_seconds = stats.p50_ms / 1000.0
            self._last_hardware_metrics = HardwareMetrics(
                **{
                    **self._last_hardware_metrics.model_dump(),
                    "joules_per_inference": (
                        self._last_hardware_metrics.watts_average * p50_seconds
                    ),
                }
            )

        self._last_run_status: str
        self._last_run_status_reason: str | None
        self._last_run_status, self._last_run_status_reason = classify_run(
            stats.p50_ms, stats.p99_ms
        )
        print(f"✓ Benchmark complete — RKNN {self._precision.upper()}")
        return stats

    # ------------------------------------------------------------------
    # Additional API
    # ------------------------------------------------------------------

    @property
    def runtime_label(self) -> str:
        """Canonical runtime string for use in BenchmarkResult.runtime."""
        return f"rknn-{self._precision}"

    @property
    def input_names(self) -> list[str]:
        """Return model input tensor names (positional index as string)."""
        return list(self._input_names)


def _make_canary_dummy(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Build the fixed-seed random-noise input used by the load-time canary.

    Deterministic (seed 0) so the canary's pass/fail behaviour is
    reproducible across runs. Integer dtypes get uniform noise spanning the
    dtype's full range (e.g. 0-255 for uint8, matching raw image bytes);
    float dtypes get standard-normal noise.
    """
    rng = np.random.default_rng(0)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return rng.integers(info.min, int(info.max) + 1, size=shape, dtype=dtype)
    return rng.standard_normal(size=shape).astype(dtype)


def _first_calibration_image_path(dataset_path: Path) -> Path | None:
    """Return the first image path listed in a calibration dataset text file.

    Mirrors agent/calbredge/converter.py's ``_first_calibration_image``
    exactly (same relative-path resolution against the dataset file's own
    parent directory) — this module intentionally does not import that
    function, since ``protocol`` must not depend on ``agent`` (protocol is
    installed and usable standalone; agent depends on protocol, not the
    other way around).

    Returns ``None`` (never raises) if ``dataset_path`` doesn't exist or
    lists no images — this is a best-effort convention lookup for the
    load-time canary, not a required conversion-time input.
    """
    if not dataset_path.exists():
        return None
    for line in dataset_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        image_path = Path(line)
        if not image_path.is_absolute():
            image_path = dataset_path.parent / image_path
        return image_path
    return None


def _resolve_canary_image_path(model_path: Path, explicit: Path | None) -> Path | None:
    """Resolve a real image path for the correctness canary, or None.

    Priority: an explicit ``canary_image_path`` (constructor arg or
    ``load_model`` kwarg) if it points at a file that exists; otherwise a
    sibling ``dataset.txt`` next to ``model_path`` (the same
    calibration-dataset convention ``convert_rknn``/
    ``check_detection_presence`` use at conversion time — see
    ``_first_calibration_image_path`` above). Returns None, never raises, if
    neither resolves — the caller falls back to noise.
    """
    if explicit is not None:
        if explicit.exists():
            return explicit
        print(
            f"⚠ RKNN canary: canary_image_path={explicit} does not exist. "
            "Falling back to the dataset.txt convention, then noise."
        )

    dataset_path = model_path.parent / "dataset.txt"
    return _first_calibration_image_path(dataset_path)


def _load_calibration_image_hwc(image_path: Path, height: int, width: int) -> np.ndarray | None:
    """Load and resize a real image to (height, width, 3) uint8 RGB, or None.

    Mirrors agent/calbredge/converter.py's ``_load_calibration_image``
    exactly. Never raises. Pillow is a ``protocol`` dependency, so it is
    normally installed; the import stays guarded so that an environment
    without it falls back to noise instead of crashing ``load_model``. Any
    failure (missing package, unreadable file, corrupt image) is reported
    and treated as "no real image available".
    """
    try:
        from PIL import Image
    except ImportError:
        print(
            f"⚠ RKNN canary: found candidate image {image_path} but Pillow is "
            "not installed (pip install Pillow) — falling back to noise."
        )
        return None

    try:
        image = Image.open(image_path).convert("RGB").resize((width, height))
        return np.array(image, dtype=np.uint8)
    except Exception as exc:
        print(
            f"⚠ RKNN canary: failed to load candidate image {image_path}: "
            f"{exc}. Falling back to noise."
        )
        return None


def _rknn_dtype_to_numpy(dtype_str: str) -> np.dtype:
    """Map an RKNN dtype string to the corresponding numpy dtype.

    RKNN reports input types as strings like ``"UINT8"``, ``"FLOAT16"``,
    ``"FLOAT32"``, ``"INT8"``.  This helper is deliberately narrow — the
    common CNN input case is uint8 (raw image bytes), which is what the
    Rockchip example graphs expect.  Unknown types fall back to uint8, since
    that's the safe default for the canary's random-noise input path.
    """
    mapping = {
        "UINT8": np.uint8,
        "INT8": np.int8,
        "FLOAT16": np.float16,
        "FLOAT32": np.float32,
        "INT32": np.int32,
    }
    return mapping.get(dtype_str.upper(), np.uint8)


def _coerce_input_dtype(dtype: str | np.dtype) -> np.dtype:
    """Coerce a user-supplied canary input dtype to a numpy dtype.

    Accepts numpy dtypes/types, RKNN-style upper-case names ("UINT8",
    "FLOAT16", ...), and ordinary numpy dtype strings ("uint8"). RKNN-style
    names are routed through _rknn_dtype_to_numpy; anything else is handed to
    numpy directly.
    """
    if isinstance(dtype, str) and dtype.upper() in {
        "UINT8",
        "INT8",
        "FLOAT16",
        "FLOAT32",
        "INT32",
    }:
        return np.dtype(_rknn_dtype_to_numpy(dtype))
    return np.dtype(dtype)
