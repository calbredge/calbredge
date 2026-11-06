"""Layout-agnostic RKNN conversion-validity check.

Detects a specific failure mode of ONNX -> RKNN INT8 conversion: the build
succeeds, the output tensors have the right shapes, but the values are
silently garbage -- e.g. a class-confidence channel-group that reported real
(0.0-1.0) values in the float reference collapsing to exactly zero after
quantization. This has happened on a real fused YOLO export (see
check_detection_presence's docstring for the full incident and metric
lineage: v1 nonzero-ratio, v2 max-ratio, v3 element-wise zeros, v4 per-
element relative error, v5 detection-presence -- four generations, each
fooled by the same fused box(~600)+class(0-1) tensor a different way).

check_detection_presence (v5) is the production gate today, but it is
YOLO-specific: it looks for class-confidence channels by shape heuristic and
explicitly skips (with a warning) any model it doesn't recognize. This
module is a *different, complementary* check with a different goal: work on
ANY model -- detector, classifier, or an architecture nobody's described to
it -- with zero knowledge of which output channels mean what.

WHY THIS DESIGN, NOT THE OBVIOUS ONES
--------------------------------------
This is the result of six exploratory calibrations (see
scripts/scratch_rknn_*_calibration.py on branch
scratch/rknn-cosine-calibration for the full numeric evidence), each ruling
out one failure mode:

1. Whole-output cosine similarity: FAILED. A fused (1, 84, 8400) YOLO output
   with 4 box channels (values up to ~637) and 80 class channels (0-1)
   scored 0.9999995 on a build with a *total* class-score collapse -- the
   corrupted sub-range contributes a vanishingly small fraction of the
   tensor's squared norm.
2. rknn-toolkit2's built-in per-layer accuracy_analysis() cosine: FAILED for
   the same reason, one level down -- per-LAYER cosine is still a whole-
   tensor aggregate at that layer, and the class-score tensor for a single
   scale is itself large enough (672,000 elements) to dilute the same way.
   The collapsed build's final-layer cosine (0.9999) was *higher* than the
   healthy build's own class-score-layer cosine (0.9889-0.9994) at two of
   its three detection heads -- a naive per-layer threshold would flag the
   healthy model and clear the broken one.
3. Whole-channel-group sparsity (fraction of near-zero elements): FAILED.
   Both the healthy and collapsed build have ~80.6% near-zero class-score
   elements in the FLOAT reference already (most anchors are confidently-
   suppressed background) -- quantization pushes nearly all of that already-
   near-zero population to exact zero in both models alike, and the real
   collapse (a few dozen genuine detections) is too small a fraction of
   672,000 elements to move the aggregate.
4. Top-k peak-agreement (does quant retain the float reference's own top-k
   positions) DID separate a collapsed YOLO detector from a healthy one
   (100% retention vs 0%) and did NOT false-positive on a healthy
   classifier -- but needed per-tensor semantic knowledge (which tensor is
   "the class scores") to know which slice to rank, so it isn't layout-
   agnostic by itself.
5. Layout-agnostic per-tensor/per-channel-band/per-channel grouping with a
   "meaningful" gate scored relative to the TENSOR's own overall float max:
   FAILED. For the fused detector tensor, the tensor max (~637) comes from
   box channels, so no class channel (~0.89) ever cleared 10% of that --
   the collapse was filtered out before ever being scored. Zero groups
   flagged for the known-bad model at any granularity.
6. Within-group (not tensor-relative) scoring with a signed max: caught the
   collapse correctly, but false-positived on a classifier's signed logits
   -- a value moving 0.058 -> -0.521 (magnitude GREW) satisfied
   "signed_max < 5% of a positive float max" despite nothing collapsing.

THIS MODULE implements the design that survived all six: score each channel
BAND (not whole-tensor, not single-channel) using its own ABSOLUTE-VALUE
(sign-invariant) peak, compared only against that same band's own float
peak -- no cross-band, cross-tensor, or cross-model reference value enters
the comparison. Band-level (not per-channel) is the validated operating
point: per-channel granularity re-introduces false positives on a
classifier's shared-scale logits (individual near-threshold logits round to
exactly zero under coarse int8 resolution as an expected, benign artifact,
not a conversion defect), while grouping 8 channels together washes that
out.

INTEGRATION CONSTRAINT -- READ BEFORE WIRING THIS IN
-----------------------------------------------------
RKNN simulator inference (``init_runtime()`` with no ``target=``) only works
on a model still in the SAME in-memory session that called ``build()`` --
reloading a saved .rknn file via ``load_rknn()`` and simulating it raises
"not support inference on the simulator" unconditionally (confirmed
empirically, see scratch_rknn_cosine_calibration.py's module docstring).
So the real call site for this check is immediately after
``rknn.build()`` succeeds, while the caller still holds that live ``RKNN``
object -- exactly where ``_check_build_for_collapse`` is called from inside
``convert_rknn`` today. ``validate_conversion`` below accepts that live
object directly (the ``rknn=`` parameter); it does NOT accept a path to a
saved .rknn file, because that path cannot work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

BAND_SIZE = 8
MEANINGFUL_FLOOR = 0.05
COLLAPSE_RATIO = 0.05


@dataclass
class CollapsedBand:
    """One channel band whose signal died between float and quant."""

    output_index: int
    channel_range: tuple[int, int]
    float_energy: float
    quant_energy: float


@dataclass
class ValidationResult:
    """Diagnosable result of validate_conversion -- not just a boolean."""

    passed: bool
    collapsed_bands: list[CollapsedBand] = field(default_factory=list)
    meaningful_bands: int = 0
    total_bands: int = 0
    reason: str | None = (
        None  # set for structural (shape/count) failures, distinct from band collapse
    )

    @property
    def verdict(self) -> str:
        return "pass" if self.passed else "fail"


def _align_outputs_by_shape(
    float_outputs: list[np.ndarray], quant_outputs: list[np.ndarray]
) -> tuple[list[tuple[int, np.ndarray, np.ndarray]] | None, str | None]:
    """Pair float reference tensors to quant simulator tensors by shape.

    RKNN's ``inference()`` output order is not guaranteed to match the ONNX
    graph's declared output order (established empirically across the
    scratch calibrations -- see scratch_rknn_topk_calibration.py and
    scratch_rknn_layout_agnostic_calibration.py). Shape matching is the
    layout-agnostic workaround: greedily pair each float tensor with an
    unused quant tensor of the identical shape.

    A count or shape mismatch is itself a validation FAILURE (per the
    proven design): a conversion that silently changes output structure is
    broken by definition, whether or not any band inside would look
    collapsed. Returns (None, reason) in that case.
    """
    if len(float_outputs) != len(quant_outputs):
        return None, (
            f"output count mismatch: {len(float_outputs)} float reference output(s) "
            f"vs {len(quant_outputs)} quantized simulator output(s)"
        )

    quant_by_shape: dict[tuple[int, ...], list[np.ndarray]] = {}
    for q in quant_outputs:
        q = np.asarray(q)
        quant_by_shape.setdefault(tuple(q.shape), []).append(q)

    pairs: list[tuple[int, np.ndarray, np.ndarray]] = []
    for i, f in enumerate(float_outputs):
        f = np.asarray(f)
        matches = quant_by_shape.get(tuple(f.shape))
        if not matches:
            return None, (
                f"no quantized output with shape {tuple(f.shape)} to match float "
                f"reference output #{i} -- conversion changed output shape/structure"
            )
        pairs.append((i, f, matches.pop(0)))
    return pairs, None


def _channel_bands(num_channels: int, band_size: int) -> list[tuple[int, int]]:
    return [(s, min(s + band_size, num_channels)) for s in range(0, num_channels, band_size)]


def _as_channel_major(arr: np.ndarray) -> np.ndarray:
    """Ensure at least (batch, channel) dims so axis=1 is always the channel axis.

    All three validated cases (fused detector, unfused detector, classifier)
    are NC.. (batch, channel, ...spatial) -- axis 0 batch, axis 1 channel.
    A bare 1-D output (no batch dim) is reshaped to (1, N) so axis 1 still
    means channel.
    """
    if arr.ndim >= 2:
        return arr
    return arr.reshape(1, -1)


def _analyze_band(
    float_arr: np.ndarray,
    quant_arr: np.ndarray,
    channel_range: tuple[int, int],
    meaningful_floor: float,
    collapse_ratio: float,
) -> tuple[bool, bool, float, float]:
    """WITHIN-BAND, SIGN-INVARIANT: abs-max compared only to itself.

    No cross-band, cross-tensor, or cross-model reference value is used --
    this is precisely what distinguishes the working design from the two
    that preceded it (see module docstring, designs #5 and #6).
    """
    s, e = channel_range
    fband = float_arr[:, s:e, ...].ravel()
    qband = quant_arr[:, s:e, ...].ravel()

    float_energy = float(np.abs(fband).max()) if fband.size else 0.0
    quant_energy = float(np.abs(qband).max()) if qband.size else 0.0

    meaningful = float_energy > meaningful_floor
    collapsed = meaningful and (quant_energy < collapse_ratio * float_energy)

    return meaningful, collapsed, float_energy, quant_energy


def _score_aligned_outputs(
    pairs: list[tuple[int, np.ndarray, np.ndarray]],
    band_size: int,
    meaningful_floor: float,
    collapse_ratio: float,
) -> ValidationResult:
    collapsed_bands: list[CollapsedBand] = []
    meaningful_count = 0
    total_bands = 0

    for output_index, f, q in pairs:
        f = _as_channel_major(f)
        q = _as_channel_major(q)
        num_channels = f.shape[1]
        for channel_range in _channel_bands(num_channels, band_size):
            total_bands += 1
            meaningful, collapsed, float_energy, quant_energy = _analyze_band(
                f, q, channel_range, meaningful_floor, collapse_ratio
            )
            if meaningful:
                meaningful_count += 1
            if collapsed:
                collapsed_bands.append(
                    CollapsedBand(
                        output_index=output_index,
                        channel_range=channel_range,
                        float_energy=float_energy,
                        quant_energy=quant_energy,
                    )
                )

    return ValidationResult(
        passed=not collapsed_bands,
        collapsed_bands=collapsed_bands,
        meaningful_bands=meaningful_count,
        total_bands=total_bands,
    )


def _build_rknn_insession(onnx_path: Path, build_params: dict[str, Any]):
    """Build an RKNN in-session from an ONNX file, for callers that don't
    already have a live rknn object (see module docstring's integration
    constraint -- simulator inference requires this exact in-session state).

    build_params keys: target_platform, mean_values, std_values,
    quantized_dtype (optional), calibration_dataset, plus optional
    inputs / input_size_list for models with symbolic input dims.
    """
    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    config_kwargs = {
        "target_platform": build_params["target_platform"],
        "mean_values": build_params["mean_values"],
        "std_values": build_params["std_values"],
    }
    if "quantized_dtype" in build_params:
        config_kwargs["quantized_dtype"] = build_params["quantized_dtype"]
    rknn.config(**config_kwargs)

    ret = rknn.load_onnx(
        model=str(onnx_path),
        inputs=build_params.get("inputs"),
        input_size_list=build_params.get("input_size_list"),
    )
    if ret != 0:
        rknn.release()
        raise RuntimeError(f"load_onnx failed ({ret}) for {onnx_path}")

    ret = rknn.build(do_quantization=True, dataset=str(build_params["calibration_dataset"]))
    if ret != 0:
        rknn.release()
        raise RuntimeError(f"build failed ({ret}) for {onnx_path}")

    return rknn


def validate_conversion(
    onnx_path: Path,
    image_hwc_uint8: np.ndarray,
    mean_values: list[list[float]],
    std_values: list[list[float]],
    rknn=None,
    *,
    build_params: dict[str, Any] | None = None,
    band_size: int = BAND_SIZE,
    meaningful_floor: float = MEANINGFUL_FLOOR,
    collapse_ratio: float = COLLAPSE_RATIO,
) -> ValidationResult:
    """Layout-agnostic check: did this RKNN conversion silently corrupt the model?

    Args:
        onnx_path: path to the original ONNX model (the float reference).
        image_hwc_uint8: one real representative image, resized to the
            model's input (H, W), uint8, HWC -- the same array shape
            ``rknn.inference()`` and ``_load_calibration_image`` already use
            elsewhere in this codebase. NOT a synthetic/noise input: this
            check (like check_detection_presence) needs a real image, since
            zero/noise input cannot separate a healthy model from a broken
            one (see check_detection_presence's docstring).
        mean_values, std_values: the SAME normalization passed to
            ``rknn.config()`` for this conversion (e.g. ``[[0, 0, 0]]`` /
            ``[[255, 255, 255]]`` for the YOLO convention used elsewhere in
            this codebase, or ``[[127.5]*3]`` / ``[[127.5]*3]`` for a
            classifier normalized to [-1, 1]) -- needed to build the
            matching float-reference input for onnxruntime, since RKNN
            applies this normalization internally to a raw uint8 input.
        rknn: a LIVE in-session RKNN object with ``build()`` already called
            (see module docstring's integration constraint). If None,
            ``build_params`` must be supplied and a model is built
            internally from ``onnx_path``.
        build_params: only used when ``rknn`` is None. See
            ``_build_rknn_insession`` for required keys.
        band_size, meaningful_floor, collapse_ratio: tunable sensitivity,
            defaulted to the values validated across six calibrations.

    Returns:
        ValidationResult -- verdict plus enough detail (which output tensor,
        which channel range, the actual float/quant energy values) to
        diagnose a FAIL, not just report one.
    """
    import onnxruntime

    owns_rknn = rknn is None
    if owns_rknn:
        if build_params is None:
            raise ValueError("build_params is required when rknn= is not supplied")
        rknn = _build_rknn_insession(onnx_path, build_params)

    try:
        mean = np.asarray(mean_values[0], dtype=np.float32).reshape(1, -1, 1, 1)
        std = np.asarray(std_values[0], dtype=np.float32).reshape(1, -1, 1, 1)
        nchw_float = image_hwc_uint8.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        nchw_normalized = (nchw_float - mean) / std

        session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        float_outputs = session.run(None, {session.get_inputs()[0].name: nchw_normalized})

        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"rknn.init_runtime (simulator) failed with code {ret}")
        quant_outputs = rknn.inference(inputs=[image_hwc_uint8[np.newaxis, ...]])
        quant_outputs = [np.asarray(q) for q in quant_outputs]

        pairs, mismatch_reason = _align_outputs_by_shape(float_outputs, quant_outputs)
        if mismatch_reason is not None:
            return ValidationResult(passed=False, reason=mismatch_reason)

        return _score_aligned_outputs(pairs, band_size, meaningful_floor, collapse_ratio)
    finally:
        if owns_rknn:
            rknn.release()
