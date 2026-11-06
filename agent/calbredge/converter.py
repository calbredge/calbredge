"""Model conversion utilities for Calbredge.

Converts Keras (.h5) and PyTorch (.pth / .pt) models to ONNX format, and
ONNX models to .rknn artifacts for Rockchip NPU inference, so they can be
benchmarked with the Calbredge protocol.

Dependencies are checked lazily inside each function so that missing
optional packages produce clear pip install instructions rather than
import-time errors.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from calbredge_protocol import rknn_artifact

from calbredge import __version__
from calbredge.validate_rknn import ValidationResult, validate_conversion


def convert_keras(model_path: Path, output_path: Path) -> None:
    """Convert a Keras .h5 model to ONNX (opset 17) using tf2onnx.

    Input shape is auto-detected from the saved model.  Both channels-last
    (H, W, C) and channels-first (C, H, W) layouts are handled transparently
    by tf2onnx.

    Args:
        model_path: Path to the .h5 model file.
        output_path: Destination path for the output .onnx file.
    """
    try:
        import tf2onnx
        import tf2onnx.convert
    except ImportError:
        print("tf2onnx is not installed. Run: pip install tf2onnx")
        sys.exit(1)

    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow is not installed. Run: pip install tensorflow")
        sys.exit(1)

    try:
        import onnx
    except ImportError:
        print("onnx is not installed. Run: pip install onnx")
        sys.exit(1)

    print(f"Loading Keras model from {model_path} …")
    try:
        model = tf.keras.models.load_model(str(model_path))
    except (ValueError, OSError) as exc:
        msg = str(exc)
        if "No model config" in msg or "Unable to open file" in msg:
            print(
                f"Error: {exc}\n\n"
                "The .h5 file appears to be weights-only: "
                "the model architecture is not embedded.\n"
                "Calbredge needs the full model (architecture + weights).\n"
                "Re-save with:  model.save('model.h5')  or  model.save('model.keras')\n"
                "For weights-only files, reconstruct the model in Python, call\n"
                "model.load_weights(path), then save the full model before converting."
            )
            sys.exit(1)
        raise

    # ── Input shape ───────────────────────────────────────────────────────
    raw_input = model.input_shape
    if isinstance(raw_input, list):
        raw_input = raw_input[0]  # first input of multi-input model
    input_shape = tuple(1 if d is None else d for d in raw_input)

    # ── Output shape ──────────────────────────────────────────────────────
    raw_output = model.output_shape
    if isinstance(raw_output, list):
        raw_output = raw_output[0]
    output_shape = tuple(1 if d is None else d for d in raw_output)

    data_format = tf.keras.backend.image_data_format()

    print(f"Input shape:  {input_shape}  ({data_format})")
    print(f"Output shape: {output_shape}")

    # ── Convert ───────────────────────────────────────────────────────────
    print("Converting to ONNX (opset 17) …")
    model_proto, _ = tf2onnx.convert.from_keras(model, opset=17)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "wb") as fh:
        fh.write(model_proto.SerializeToString())

    # ── Validate ──────────────────────────────────────────────────────────
    print("Validating ONNX model …")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    print(f"✓ Conversion successful → {output_path}")


def convert_pytorch(
    model_path: Path,
    output_path: Path,
    input_shape: tuple[int, ...],
) -> None:
    """Convert a PyTorch model (.pth / .pt) to ONNX (opset 17).

    Supports both full torch.nn.Module saves and TorchScript saves.
    State-dict-only .pth files are not supported — the architecture must
    be embedded in the file.

    Args:
        model_path: Path to the .pth or .pt model file.
        output_path: Destination path for the output .onnx file.
        input_shape: Tuple of ints e.g. (1, 3, 224, 224).
    """
    try:
        import torch
    except ImportError:
        print("torch is not installed. Run: pip install torch")
        sys.exit(1)

    try:
        import onnx
    except ImportError:
        print("onnx is not installed. Run: pip install onnx")
        sys.exit(1)

    print(f"Loading PyTorch model from {model_path} …")

    # ── Load model ────────────────────────────────────────────────────────
    model = None

    # Try TorchScript first (.pt or .pth saved with torch.jit.save)
    try:
        model = torch.jit.load(str(model_path), map_location="cpu")
        print("Detected: TorchScript model")
    except Exception:  # noqa: S110 — format-detection probe, falls through to the next loader on failure
        pass

    if model is None:
        try:
            loaded = torch.load(str(model_path), map_location="cpu")
        except Exception as exc:
            print(f"Error: Failed to load model: {exc}")
            sys.exit(1)

        if isinstance(loaded, torch.nn.Module):
            model = loaded
            print("Detected: torch.nn.Module")
        elif isinstance(loaded, dict):
            print(
                "Error: The .pth file contains a state_dict (weights only), not a full model.\n"
                "Calbredge requires the architecture to be embedded in the file.\n"
                "Save the full model with one of:\n"
                "  torch.save(model, 'model.pth')                            # full module\n"
                "  torch.jit.save(torch.jit.script(model), 'model.pt')      # TorchScript"
            )
            sys.exit(1)
        else:
            print(f"Error: Unexpected object type '{type(loaded).__name__}' in .pth file.")
            sys.exit(1)

    model.eval()

    # ── Export ────────────────────────────────────────────────────────────
    print(f"Input shape:  {input_shape}")
    dummy = torch.zeros(*input_shape)

    print("Converting to ONNX (opset 17) …")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
    )

    # ── Validate ──────────────────────────────────────────────────────────
    print("Validating ONNX model …")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    # Read output shape from ONNX graph
    out_shape_proto = onnx_model.graph.output[0].type.tensor_type.shape
    output_shape = tuple(d.dim_value if d.dim_value != 0 else "?" for d in out_shape_proto.dim)
    print(f"Output shape: {output_shape}")
    print(f"✓ Conversion successful → {output_path}")


# ---------------------------------------------------------------------------
# RKNN conversion-time correctness check
# ---------------------------------------------------------------------------
#
# Mirrors the TensorRT pattern of refusing a bad INT8 build at *build* time
# rather than trusting a .rknn file that loads and runs without error. See
# docs/TROUBLESHOOTING.md §7: a stock Ultralytics YOLO export fuses
# sigmoid/DFL/concat into one output tensor, and RKNN's per-tensor INT8
# quantization collapses the class-confidence sub-range of that tensor to
# exactly zero — silently, with box coordinates still nonzero, so the
# load-time canary in protocol/calbredge_protocol/rknn_adapter.py (a whole-tensor nonzero-ratio
# check) cannot catch it. Rather than comparing the quantized output against
# a float reference — four prior generations of that approach all failed,
# see check_detection_presence's docstring for the full lineage — this
# check runs the just-built model's own simulator inference on one
# representative calibration image and asks a question with a known answer:
# did any class score survive quantization at all.


def find_class_scores(outputs: list[np.ndarray]) -> list[np.ndarray] | None:
    """Locate the class-confidence tensor(s) in a YOLO model's raw output.

    Two known output layouts:

    - Fused (stock Ultralytics export): exactly one output tensor, shape
      ``(1, 4 + nc, anchors)`` — 4 box channels concatenated with nc class
      channels along axis 1. ``nc`` is derived from the tensor's own shape
      (accepting 5..~1000 total channels) rather than hardcoded, since
      custom-trained models have class counts other than the COCO 80.
      Returns a single-element list: ``[output[:, 4:, :]]``.
    - Unfused (Rockchip zoo export): multiple output tensors, one DFL box
      tensor per scale (channel dim 64) and one score-sum tensor per scale
      (channel dim 1) alongside the class-score tensors. The class-score
      tensors are identified by elimination — ndim 4, channel dim not in
      (64, 1) — and cross-checked by requiring every candidate to share the
      same channel count (nc, shared across scales), rather than hardcoding
      80. Returns those tensors.

    Anything else (a classifier's single ``(1, 1000)`` vector, an
    unrecognized multi-output layout) returns ``None`` — this function
    never guesses.

    Args:
        outputs: Raw output tensors from a model (float or quantized).

    Returns:
        A list of class-score tensors, or ``None`` if the layout isn't
        recognized.
    """
    if len(outputs) == 1:
        out = outputs[0]
        if out.ndim == 3 and out.shape[0] == 1:
            channels = out.shape[1]
            if 5 <= channels <= 1000:
                return [out[:, 4:, :]]
        return None

    candidates = [o for o in outputs if o.ndim == 4]
    class_tensors = [o for o in candidates if o.shape[1] not in (64, 1)]
    if not class_tensors:
        return None
    if len({o.shape[1] for o in class_tensors}) != 1:
        return None
    return class_tensors


def check_detection_presence(
    quant_outputs: list[np.ndarray], min_score: float = 0.3
) -> tuple[bool, str | None]:
    """This check catches one specific bug: quantization silently zeroing a
    YOLO model's class scores. For any other model type it skips with a
    warning instead of guessing.

    This is metric v5, and the last one. It is not a new theory of
    correctness — it is the manual bus.jpg probe that has been 5-for-5
    correct on real artifacts all week, moved into convert_rknn. Every
    prior generation instead tried to compare the quantized output tensor
    against a float reference tensor, at a different level of aggregation
    each time, and every one failed for the same underlying reason:

    1. Nonzero-ratio (load-time canary, protocol/calbredge_protocol/rknn_adapter.py): masked
       — the surviving box channels kept the tensor's overall nonzero
       ratio high while every class-confidence value was exactly zero.
    2. Per-tensor max-ratio: masked the same way — float max ~= quant max
       ~= 637 (the box value), ratio ~1.0, passed a build confirmed dead
       on device on 2026-07-20 (bus.jpg: box max 636.91, cls max 0.0, cls
       nonzero-count 0).
    3. Element-wise zero check: false-alarmed on its first true-negative
       test, refusing the healthy Rockchip zoo yolov8n export because a
       handful of small values legitimately round to zero in a
       wide-dynamic-range tensor.
    4. Cosine similarity (per-tensor, whole-vector): evaluated by hand
       against the exact fused box(~600)+class(0.5-0.9->0) construction
       from failure #2 and rejected before ever being merged — cos =
       0.9999995. The corrupted class sub-range contributes
       ~1/930,000th of the tensor's squared norm, so zeroing it moves
       cosine by an amount indistinguishable from float noise.
    5. Per-element relative error, aggregated as a per-tensor fraction
       (v4, shipped and then reverted on 2026-07-20): passed the
       confirmed-dead stock fused export — the 26 dead class scores were
       diluted invisible in a 33,761-element denominator (0.52% dead
       fraction, under the 2% threshold) — and refused the healthy
       Rockchip zoo export (2.94% — rounding noise on
       small-relative-to-step values, over the threshold).

    Conclusion: every semantic-blind tensor comparison above (v1 nonzero-
    ratio, v2 max-ratio, v3 element-wise zeros, cosine, v4 relative error)
    fails because a rounded-to-zero value and a collapsed-to-zero value
    are the same number; the distinction is in what the number *means*,
    and no tensor-shaped arithmetic on the number itself can recover that
    meaning.

    The fix: stop comparing tensors and ask the quantized model a question
    with a known answer. A YOLO model's class-confidence output, run on a
    real calibration image, should report at least one class above a
    detection threshold — that's what "the model detected something"
    means. A model whose class scores were zeroed by quantization reports
    a hard 0.0 regardless of the image; a healthy model reports something
    close to its training-time confidence (~0.87 on the bus.jpg fixture
    used to validate this all week). The two real outcomes are ~0.87 vs.
    exactly 0.0, which is why a crude 0.3 threshold is safe — there is no
    realistic middle ground to tune against.

    Args:
        quant_outputs: Output tensors from the quantized (int8 or fp16)
                       RKNN build, for one representative calibration
                       image.
        min_score: Minimum class-confidence score that must appear
                   somewhere in the class-score tensors for the check to
                   pass.

    Returns:
        ``(passed, reason)``. If the output layout isn't recognized (see
        find_class_scores), this prints a loud warning and returns
        ``(True, None)`` — fail open and loud, never guess; this is the
        documented behavior for classifiers (MobileNetV2 etc.) and
        unknown architectures. Otherwise ``reason`` reports the observed
        top class score on both pass and fail — evidence-on-pass stays,
        because it's what caught v2 and v4 both silently shipping a blind
        spot.
    """
    class_tensors = find_class_scores(quant_outputs)
    if class_tensors is None:
        print("⚠ unknown output layout — detection check SKIPPED, verify on-device")
        return True, None

    top = max(float(np.max(t)) for t in class_tensors)

    if top < min_score:
        return False, (
            f"No class score reached {min_score} (max observed: {top:.4f}). "
            "This is the INT8 quantization-collapse pattern from "
            "TROUBLESHOOTING.md §7: class-confidence scores were zeroed by "
            "quantization even though other output channels remain "
            "populated."
        )

    return True, (f"Detection check passed — max class score {top:.4f} >= {min_score}.")


def _calibration_image_paths(calibration_dataset: Path | str) -> list[Path]:
    """Return every image path listed in a calibration dataset text file, in
    order.

    Relative entries resolve against the dataset file's own parent directory,
    not the process's current working directory — a dataset.txt containing
    `calib_images/coco_1.jpg` must find that image regardless of where
    `calbredge convert` is invoked from, as long as the two travel together.
    Absolute entries are returned unchanged.
    """
    dataset_path = Path(calibration_dataset)
    paths: list[Path] = []
    for line in dataset_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        image_path = Path(line)
        if not image_path.is_absolute():
            image_path = dataset_path.parent / image_path
        paths.append(image_path)
    return paths


def _first_calibration_image(calibration_dataset: Path | str) -> Path:
    """Return the first image path listed in a calibration dataset text file.

    See _calibration_image_paths for the relative-path resolution rule.
    """
    paths = _calibration_image_paths(calibration_dataset)
    if not paths:
        raise ValueError(f"Calibration dataset {Path(calibration_dataset)} lists no images.")
    return paths[0]


def _onnx_input_hw(onnx_path: Path) -> tuple[int, int]:
    """Read the (height, width) the ONNX model's first input declares (NCHW).

    Dynamic/symbolic dimensions fall back to 224, matching Calbredge's own
    default protocol input shape.

    Assumes NCHW — i.e. that the last two dims are (H, W) — which is correct
    for most vision models but wrong for NHWC (channel-last) exports, e.g.
    efficientnet_b0, efficientdet_lite0, and cxr_mobilenetv2 (see
    protocol/calbredge_protocol/models.py's input_shape docstring). For NHWC, dims[-1] is the
    channel count, not width. Rather than silently mislabel it, this raises
    when the trailing dimension looks like a channel count (<= 4) or the
    shape otherwise has too few dimensions to resolve unambiguously — it
    never guesses or transposes NHWC layouts.
    """
    import onnxruntime

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    dims = [d if isinstance(d, int) and d > 0 else 224 for d in shape]

    if len(dims) < 3:
        raise ValueError(
            f"Cannot resolve H/W for {onnx_path.name}: input shape {shape} has fewer than "
            "3 dimensions, so there is no unambiguous (H, W) pair. Supply the layout "
            "explicitly rather than guessing."
        )

    height, width = dims[-2], dims[-1]
    if width <= 4:
        raise ValueError(
            f"Cannot unambiguously resolve H/W for {onnx_path.name}: input shape {shape} "
            f"has trailing dims ({height}, {width}), and a last dimension of {width} looks "
            "like a channel count, not a width — this is likely an NHWC (channel-last) "
            "input. _onnx_input_hw assumes NCHW (dims[-2], dims[-1]) = (H, W) and does not "
            "guess or transpose NHWC layouts; the caller must supply the layout explicitly."
        )
    return height, width


# Dimension sizes that plausibly denote an image channel count. Shared by
# _onnx_input_hw's ambiguity check above and _detect_nhwc_image_input below,
# so the two functions can never disagree about what "looks like a channel
# count" means.
_CHANNEL_LIKE_DIM_SIZES = frozenset({1, 3, 4})

# The onnx_edit einops-style equation used to transpose a detected NHWC
# input to NCHW at the graph level: (a, b, c, d) = (N, H, W, C) becomes
# (a, d, b, c) = (N, C, H, W).
_NHWC_TO_NCHW_TRANSFORM = "a,b,c,d->a,d,b,c"

# Bumped whenever convert_rknn's own settings or graph edits change what it builds from the
# same model (the normalization, the NHWC transpose, the quantized dtype): part of the build
# key in the .rknn's build record (protocol/calbredge_protocol/rknn_artifact.py).
RKNN_BUILDER_CONFIG_VERSION = 1
# What every artifact convert_rknn builds takes at inference: raw pixels, normalized inside
# the model by the mean and std values below (see the comment where they are set).
_RKNN_FEED_DTYPE = "uint8"
_RKNN_QUANTIZED_DTYPE = "asymmetric_quantized-8"


def _rknn_toolkit_version() -> str:
    """The installed rknn-toolkit2 version, which is part of the build key.

    Raises:
        RuntimeError: If it cannot be read. An artifact whose compiler version is unknown
                      would get a build record that says less than it claims, so the
                      conversion stops before building.
    """
    import importlib.metadata

    try:
        return importlib.metadata.version("rknn-toolkit2")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "cannot read the installed rknn-toolkit2 version, which the .rknn's build record "
            "needs; install it with pip (pip install rknn-toolkit2==2.3.2)"
        ) from exc


def _record_model_inputs(
    onnx_path: Path, load_onnx_path: Path, nhwc_input_name: str | None
) -> list[dict]:
    """One entry per model input for the build record: what the source model declares, and
    the shape the conversion used (after the NHWC transpose, a symbolic batch resolved to 1)."""
    from calbredge_protocol.engine_cache import onnx_input_signature
    from calbredge_protocol.graph_variant import load_graph
    from onnx import TensorProto

    graph = load_graph(onnx_path).graph
    initializers = {t.name for t in graph.initializer}
    dtypes = {
        v.name: TensorProto.DataType.Name(v.type.tensor_type.elem_type).lower()
        for v in graph.input
        if v.name not in initializers
    }
    names, sizes = _resolve_load_onnx_inputs(load_onnx_path)
    resolved = dict(zip(names or [], sizes or [], strict=False))
    converted = {s["name"]: s["shape"] for s in onnx_input_signature(load_onnx_path)}
    entries = []
    for source in onnx_input_signature(onnx_path):
        name, shape = source["name"], source["shape"]
        if name == nhwc_input_name:
            layout = "nhwc"
        elif len(shape) == 4 and shape[1] in _CHANNEL_LIKE_DIM_SIZES:
            layout = "nchw"
        else:
            layout = None
        entries.append(
            {
                "name": name,
                "source_shape": shape,
                "source_dtype": dtypes.get(name),
                "source_layout": layout,
                "converted_shape": resolved.get(name, converted.get(name)),
            }
        )
    return entries


def _detect_nhwc_image_input(onnx_path: Path) -> tuple[str, int, int, int] | None:
    """Detect whether onnx_path's first input is an unambiguously NHWC
    (channel-last) image input that needs an onnx_edit transpose before
    rknn.load_onnx/build ever see it.

    THE REGRESSION THIS REPLACES: reordering only rknn.load_onnx's
    input_size_list to NCHW order (the prior fix — see git history for
    the now-removed _reorder_input_size_list_to_nchw) satisfies
    load_onnx's mean_values axis check (rknn.config()'s mean_values/
    std_values are one value per channel, and load_onnx checks their
    length against axis 1 of whatever shape it's given, NCHW-positionally,
    regardless of the model's actual declared layout), but then
    rknn.build's graph executor is told the input is NCHW while the ONNX
    graph's Conv weights genuinely expect NHWC data — confirmed failing on
    cxr_mobilenetv2 with "Input channels C is not equal to kernel
    channels * group. C: 224 kernel channels: 3 group: 1". The fix is to
    make the graph ITSELF genuinely NCHW throughout, via
    rknn.utils.onnx_edit (see convert_rknn), not just the shape descriptor
    handed to load_onnx.

    Ambiguity handling matches _onnx_input_hw's philosophy above: never
    guess. A shape is treated as unambiguously NHWC only when its last
    dimension is a plausible channel count (1, 3, or 4) AND no other
    dimension between the batch axis and the last axis is equally
    plausible as a channel count — e.g. ``[1, 3, 3, 3]`` could be NCHW
    channels=3 with a 3x3 spatial size just as easily as NHWC with a 3x3
    spatial size and channels=3, so this raises rather than picking one.
    A shape whose last dimension is NOT channel-like (e.g. YOLOv8's
    ``[1, 3, 640, 640]``, last dim 640) is assumed already NCHW-ordered and
    this returns ``None`` — this is what keeps already-correct models
    untouched. A symbolic (non-batch) height or width dimension is also not
    guessed — it raises, same as a symbolic non-batch dimension anywhere
    else in this module.

    Args:
        onnx_path: Path to the source .onnx model.

    Returns:
        ``(input_name, height, width, channels)`` for the first input if it
        is unambiguously NHWC, else ``None``.

    Raises:
        ValueError: if the channel axis is ambiguous, or a non-batch
                    height/width dimension is symbolic.
    """
    import onnxruntime

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    shape = list(inp.shape)

    if len(shape) < 3:
        return None  # no distinct "last axis" vs. "axis 1" to compare

    channel = shape[-1]
    if not (isinstance(channel, int) and channel in _CHANNEL_LIKE_DIM_SIZES):
        return None  # last axis isn't channel-like — assume already NCHW

    # Candidates for "this dim could also plausibly be the channel axis":
    # every axis strictly between the batch axis (0, always excluded — it's
    # fixed at 1 by Calbredge's batch_size=1 convention, not a layout
    # signal) and the last axis (already tested above).
    ambiguous_axis = next(
        (
            axis
            for axis in range(1, len(shape) - 1)
            if isinstance(shape[axis], int) and shape[axis] in _CHANNEL_LIKE_DIM_SIZES
        ),
        None,
    )
    if ambiguous_axis is not None:
        raise ValueError(
            f"Cannot unambiguously resolve the channel axis for {onnx_path.name}: "
            f"input '{inp.name}' has shape {shape}, where both the last "
            f"dimension ({channel}) and axis {ambiguous_axis} "
            f"({shape[ambiguous_axis]}) look like plausible channel counts "
            "(1, 3, or 4). This module never guesses at a layout it can't "
            "read off the graph unambiguously (see _onnx_input_hw's "
            "docstring) — supply the layout explicitly rather than "
            "guessing."
        )

    height, width = shape[-3], shape[-2]
    if not (isinstance(height, int) and isinstance(width, int)):
        raise ValueError(
            f"Cannot resolve the NHWC transpose for {onnx_path.name}: input "
            f"'{inp.name}' has shape {shape}, where the height and/or width "
            "dimension is symbolic. This module never guesses at a "
            "dimension it can't read off the graph — supply the shape "
            "explicitly rather than guessing."
        )

    return inp.name, height, width, channel


def _resolve_load_onnx_inputs(
    onnx_path: Path,
) -> tuple[list[str] | None, list[list[int]] | None]:
    """Resolve rknn.load_onnx's ``inputs``/``input_size_list`` override for
    a symbolic (dynamic) input dimension.

    rknn.load_onnx rejects one outright (e.g. MobileNetV2's
    ``['batch_size', 3, 224, 224]`` export) with ``ValueError: The input
    shape [...] is not support!``. Calbredge always benchmarks at
    batch_size=1 (see PROTOCOL.md), so a symbolic dimension at axis 0 (the
    batch axis) is resolved to 1. A symbolic dimension anywhere else is not
    guessed — consistent with _onnx_input_hw's docstring above, this module
    never guesses at a layout it can't read off the graph — and raises
    instead, so the caller must supply the shape explicitly.

    NOTE: NHWC (channel-last) layouts are handled upstream of this
    function, not here — see _detect_nhwc_image_input and convert_rknn's
    onnx_edit-based graph transpose. By the time onnx_path reaches this
    function it is either already NCHW, or is convert_rknn's onnx_edit-
    transposed copy of a formerly-NHWC model — either way, this function
    only ever resolves a symbolic batch dimension; it does not reorder
    axes.

    Args:
        onnx_path: Path to the .onnx model that will be passed to
                   rknn.load_onnx (the original file, or convert_rknn's
                   onnx_edit-transposed copy of a formerly-NHWC model).

    Returns:
        ``(None, None)`` if every graph input is already fully concrete —
        i.e. load_onnx needs no override and the caller should invoke it
        exactly as before. Otherwise the parallel ``(inputs,
        input_size_list)`` lists rknn.load_onnx expects — input names and
        their resolved shapes (symbolic batch dimension resolved to 1).
    """
    import onnxruntime

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_inputs = session.get_inputs()

    inputs: list[str] = []
    input_size_list: list[list[int]] = []
    override_needed = False

    for inp in onnx_inputs:
        shape = list(inp.shape)
        for axis, dim in enumerate(shape):
            if isinstance(dim, int) and dim > 0:
                continue
            if axis == 0:
                shape[axis] = 1
                override_needed = True
                continue
            raise ValueError(
                f"Cannot resolve rknn.load_onnx input shape for {onnx_path.name}: "
                f"input '{inp.name}' has a symbolic dimension {dim!r} at axis "
                f"{axis} (shape {inp.shape}) that isn't the batch dimension. Only "
                "a symbolic batch dimension (axis 0) is resolved automatically, "
                "to 1 — any other symbolic axis must be supplied explicitly "
                "rather than guessed."
            )

        inputs.append(inp.name)
        input_size_list.append(shape)

    if not override_needed:
        return None, None

    return inputs, input_size_list


def _load_calibration_image(image_path: Path, width: int, height: int) -> np.ndarray:
    """Load and resize a calibration image to (height, width, 3) uint8 RGB."""
    try:
        from PIL import Image
    except ImportError:
        print("Pillow is not installed. Run: pip install Pillow")
        sys.exit(1)

    image = Image.open(image_path).convert("RGB").resize((width, height))
    return np.array(image, dtype=np.uint8)


def _build_nchw_calibration_dataset(
    calibration_dataset: Path | str,
    tmp_dir: Path,
    height: int,
    width: int,
) -> Path:
    """Build an NCHW calibration dataset for rknn.build()'s ``dataset``
    argument, for a model whose input onnx_edit just transposed to NCHW.

    rknn.utils.onnx_edit's own bundled dataset transformer emits warnings
    and produces an EMPTY output file for a model with a symbolic batch
    dimension (confirmed manually on cxr_mobilenetv2) — so this bypasses it
    entirely. Each image listed in the original *calibration_dataset* is
    loaded via PIL, resized to the model's declared (height, width),
    transposed HWC -> CHW, given a batch dim, and saved as a standalone
    .npy file next to the edited .onnx — rknn.build() accepts .npy paths in
    its dataset list, not just raw images, which is exactly what this
    relies on.

    Args:
        calibration_dataset: the ORIGINAL dataset.txt (listing real image
                              files) — not an onnx_edit-generated one.
        tmp_dir: scratch directory to write the .npy files and the new
                 dataset.txt into. Caller owns cleanup.
        height, width: the model's declared NHWC input spatial size, from
                       _detect_nhwc_image_input on the pre-edit graph.

    Returns:
        Path to a new dataset.txt listing the generated .npy files, for
        rknn.build(dataset=...).
    """
    npy_dir = tmp_dir / "calib_npy"
    npy_dir.mkdir(parents=True, exist_ok=True)

    npy_paths: list[Path] = []
    for image_path in _calibration_image_paths(calibration_dataset):
        hwc = _load_calibration_image(image_path, width, height)
        chw = np.ascontiguousarray(hwc.transpose(2, 0, 1))[np.newaxis, ...]
        npy_path = npy_dir / f"{image_path.stem}.npy"
        np.save(npy_path, chw)
        npy_paths.append(npy_path)

    if not npy_paths:
        raise ValueError(f"Calibration dataset {calibration_dataset} lists no images.")

    dataset_txt = tmp_dir / "dataset_nchw.txt"
    dataset_txt.write_text("\n".join(str(p) for p in npy_paths) + "\n")
    return dataset_txt


def _run_quantized_reference(rknn, image_hwc_uint8: np.ndarray) -> list[np.ndarray]:
    """Run one image through the just-built RKNN model via its on-host simulator.

    Requires ``rknn.build()`` to have already succeeded on *rknn*.
    ``init_runtime()`` with no ``target=`` runs in simulator mode — no NPU or
    Orange Pi connection needed, so this executes entirely on the x86
    conversion host.
    """
    ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"rknn.init_runtime (simulator) failed with code {ret}.")
    return rknn.inference(inputs=[image_hwc_uint8[np.newaxis, ...]])


def _load_representative_image(onnx_path: Path, calibration_dataset: Path | str) -> np.ndarray:
    """Load the calibration dataset's first image, resized to the ONNX
    model's declared input (H, W).

    Shared by both conversion-time correctness checks (detection-presence
    and the general layout-agnostic validator) so they see the exact same
    representative image — zero/noise input cannot separate a healthy model
    from a broken one here (measured on the real incident: 0.0038 vs 0.0
    nonzero ratio on zero input; see protocol/calbredge_protocol/rknn_adapter.py's load-time
    canary docstring).
    """
    image_path = _first_calibration_image(calibration_dataset)
    height, width = _onnx_input_hw(onnx_path)
    return _load_calibration_image(image_path, width, height)


def _check_build_for_collapse(
    rknn,
    onnx_path: Path,
    calibration_dataset: Path | str,
) -> tuple[bool, str | None]:
    """Run the conversion-time detection-presence correctness check."""
    image = _load_representative_image(onnx_path, calibration_dataset)
    quant_outputs = _run_quantized_reference(rknn, image)
    return check_detection_presence(quant_outputs)


def _format_validation_failure(result: ValidationResult, output_path: Path) -> str:
    """Render a ValidationResult FAIL verdict into a refusal message with
    enough detail (which output tensor, which channel range, the actual
    float/quant energy values) to diagnose it, matching
    check_detection_presence's evidence-on-fail pattern."""
    if result.reason is not None:
        detail = result.reason
    else:
        bands = "\n".join(
            f"  - output #{band.output_index}, channels {band.channel_range}: "
            f"float peak {band.float_energy:.4f} -> quant peak {band.quant_energy:.4f}"
            for band in result.collapsed_bands
        )
        detail = (
            f"{len(result.collapsed_bands)} channel band(s) collapsed during "
            f"quantization ({result.meaningful_bands}/{result.total_bands} bands "
            f"had real signal in the float reference):\n{bands}"
        )
    return (
        f"General conversion validator detected corruption — refusing to emit "
        f"{output_path.name}.\n{detail}\n"
        "This is the layout-agnostic check in agent/calbredge/validate_rknn.py: "
        "it compares the float ONNX reference against the quantized RKNN "
        "simulator output in channel bands, independent of architecture. A "
        "collapsed band means real signal in the float reference went to "
        "(near-)zero after quantization."
    )


def convert_rknn(
    onnx_path: Path,
    output_path: Path,
    precision: str = "fp16",
    target_platform: str = "rk3588",
    calibration_dataset: Path | str | None = None,
    skip_validate_conversion: bool = False,
) -> None:
    """Convert an ONNX model to a .rknn artifact for Rockchip NPU inference.

    Runs entirely on an x86 conversion host — rknn-toolkit2 does not support
    aarch64. Mirrors the TensorRT pattern of catching a bad INT8 build at
    *build* time: after a successful build, if a calibration dataset is
    available, this runs one representative image through the just-built
    model (via rknn-toolkit2's on-host simulator) and refuses to emit the
    .rknn if either of two correctness checks flags it:

    1. check_detection_presence: YOLO-specific, looks for a detection-shaped
       output and checks its class-confidence scores weren't zeroed by INT8
       quantization. Fails open (warns and skips) on any output layout it
       doesn't recognize — e.g. a classifier. fp16 builds run this check as
       a warning only — fp16 should not collapse, and if it does, something
       else is wrong and is worth investigating, but it is not this
       function's failure mode to enforce.
    2. validate_conversion (agent/calbredge/validate_rknn.py): general and
       layout-agnostic — compares the float ONNX reference against the
       quantized RKNN simulator output in channel bands, on any
       architecture, so it also catches corruption on models
       check_detection_presence fails open on. Unlike check_detection_presence,
       a FAIL here always blocks the export, on both fp16 and int8 builds:
       this check has no known false-positive history (unlike
       check_detection_presence's four reverted prior generations — see its
       docstring) and isn't scoped to the INT8-specific collapse pattern, so
       there's no fp16 carve-out to make. It runs unless
       skip_validate_conversion is set — it adds a full onnxruntime
       inference pass plus a second RKNN simulator pass to every conversion,
       so a fast dev-loop iteration may want to skip it, but it should
       always run before trusting an artifact.

    Both checks' results are always printed, even when one of them fails
    open — a silent skip on an unrecognized layout should never be the only
    thing the user sees.

    Before any of the above: if the model's input is unambiguously NHWC
    (channel-last — e.g. cxr_mobilenetv2, efficientnet_b0,
    efficientdet_lite0; see _detect_nhwc_image_input), the ONNX graph is
    rewritten to genuine NCHW via rknn.utils.onnx_edit before load_onnx/
    build ever see it, and (for an int8 build) the calibration dataset is
    regenerated in matching NCHW .npy form. This replaced an earlier fix
    that only reordered rknn.load_onnx's input_size_list argument — that
    satisfied load_onnx's mean_values axis check but left rknn.build's
    graph executor believing the input was NCHW while the graph's own Conv
    weights still expected NHWC data, corrupting the build.

    Args:
        onnx_path: Path to the source .onnx model.
        output_path: Destination path for the output .rknn file.
        precision: "fp16" or "int8". int8 requires calibration_dataset.
        target_platform: RKNN target platform string, e.g. "rk3588".
        calibration_dataset: Path to a text file listing calibration image
                             paths, one per line. Required for int8 builds
                             (RKNN quantization is sensitive to calibration
                             data quality — do not use random noise, unlike
                             the compromise Calbredge makes for TensorRT
                             INT8). The first entry doubles as the
                             representative image for both correctness
                             checks. When omitted, both correctness checks
                             are skipped entirely.
        skip_validate_conversion: If True, skip the general layout-agnostic
                             validate_conversion check (see above) even when
                             a calibration_dataset is supplied. Intended for
                             fast/dev iteration only — check_detection_presence
                             still runs. Defaults to False: validate_conversion
                             runs by default, since it is the only one of the
                             two checks that doesn't fail open on non-YOLO
                             architectures.

    Raises:
        ValueError: If precision is invalid, int8 is requested without a
                    calibration_dataset, the ONNX model has a symbolic
                    dimension outside the batch axis (see
                    _resolve_load_onnx_inputs — a symbolic batch dimension
                    itself is resolved automatically to 1, not an error), or
                    an NHWC input shape's channel axis can't be resolved
                    unambiguously (see _detect_nhwc_image_input).
        RuntimeError: If rknn.load_onnx, rknn.build, or rknn.export_rknn
                      fails, or if either conversion-time correctness check
                      detects corruption (check_detection_presence for an
                      int8 build, or validate_conversion for either
                      precision).
    """
    if precision not in ("fp16", "int8"):
        raise ValueError(f"precision must be 'fp16' or 'int8', got {precision!r}")

    if precision == "int8" and not calibration_dataset:
        raise ValueError(
            "calibration_dataset is required for --precision int8 (a text "
            "file listing calibration image paths, one per line). Do not "
            "use random noise for RKNN INT8 calibration — unlike TensorRT, "
            "RKNN quantization is sensitive to calibration data quality; "
            "use representative real images. See docs/ORANGE_PI_SETUP.md."
        )

    try:
        from rknn.api import RKNN
        from rknn.utils import onnx_edit
    except ImportError:
        print(
            "rknn-toolkit2 is not installed. RKNN conversion runs on an x86 "
            "host only (rknn-toolkit2 does not support aarch64).\n"
            "Install it from a clone with the converter's constraints file "
            "(onnx 1.18.0, setuptools<81, CPU-only torch):\n"
            "  pip install -c constraints/rknn-convert-x86_64-py312.txt "
            "--extra-index-url https://download.pytorch.org/whl/cpu "
            'rknn-toolkit2 ./protocol "./agent[rknn-convert]"\n'
            "See docs/QUICKSTART.md (Rockchip RK3588, step 1)."
        )
        sys.exit(1)

    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    toolkit_version = _rknn_toolkit_version()

    rknn = RKNN(verbose=False)
    with tempfile.TemporaryDirectory(prefix="calbredge_rknn_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        try:
            print(f"Configuring RKNN for {target_platform} ({precision}) …")
            # mean_values/std_values must be identical across precisions: the
            # adapter assumes every RKNN artifact takes raw uint8 input
            # (RKNNAdapter._resolve_input_dtype defaults to uint8 regardless of
            # precision). rknn.config() defaults mean=0/std=1 (no normalization)
            # when unset, so leaving them unset on the int8 branch would
            # silently give int8 and fp16 builds different input contracts.
            config_kwargs = {
                "target_platform": target_platform,
                "mean_values": [[0, 0, 0]],
                "std_values": [[255, 255, 255]],
            }
            if precision == "int8":
                config_kwargs["quantized_dtype"] = _RKNN_QUANTIZED_DTYPE
            rknn.config(**config_kwargs)

            # load_onnx_path/reference_onnx_path default to the original
            # model. If the model's input is unambiguously NHWC, both are
            # switched to an onnx_edit-transposed copy below — see
            # _detect_nhwc_image_input's docstring for why an
            # input_size_list-only reorder (the prior fix) isn't enough:
            # it satisfies load_onnx's mean_values check but leaves
            # rknn.build's graph executor believing the input is NCHW while
            # the graph's own Conv weights still expect NHWC data.
            load_onnx_path = onnx_path
            reference_onnx_path = onnx_path
            build_dataset = calibration_dataset

            graph_edits: list[str] = []
            nhwc_input = _detect_nhwc_image_input(onnx_path)
            if nhwc_input is not None:
                graph_edits.append("nhwc_to_nchw_input_transpose")
                input_name, height, width, _channels = nhwc_input
                print(
                    f"Detected NHWC input '{input_name}' ({height}x{width}) — "
                    "editing the ONNX graph to genuine NCHW via "
                    "rknn.utils.onnx_edit …"
                )
                edited_onnx_path = tmp_dir / "model_nchw.onnx"
                onnx_edit(
                    model=str(onnx_path),
                    export_path=str(edited_onnx_path),
                    inputs_transform={input_name: _NHWC_TO_NCHW_TRANSFORM},
                )
                if not edited_onnx_path.exists():
                    raise RuntimeError(
                        f"rknn.utils.onnx_edit did not produce {edited_onnx_path} "
                        f"for {onnx_path.name}'s NHWC-to-NCHW transpose."
                    )
                load_onnx_path = edited_onnx_path
                reference_onnx_path = edited_onnx_path

                if precision == "int8" and calibration_dataset is not None:
                    print("Rebuilding calibration dataset in NCHW layout for rknn.build() …")
                    build_dataset = _build_nchw_calibration_dataset(
                        calibration_dataset, tmp_dir, height, width
                    )

            print(f"Loading ONNX model from {load_onnx_path} …")
            inputs, input_size_list = _resolve_load_onnx_inputs(load_onnx_path)
            if inputs is None:
                ret = rknn.load_onnx(model=str(load_onnx_path))
            else:
                ret = rknn.load_onnx(
                    model=str(load_onnx_path), inputs=inputs, input_size_list=input_size_list
                )
            if ret != 0:
                raise RuntimeError(
                    f"rknn.load_onnx failed with code {ret}. Confirm onnx==1.18.0 "
                    "is installed exactly — newer onnx releases break load_onnx. "
                    "See docs/ORANGE_PI_SETUP.md."
                )

            print(f"Building {precision.upper()} RKNN model …")
            if precision == "int8":
                ret = rknn.build(do_quantization=True, dataset=str(build_dataset))
            else:
                ret = rknn.build(do_quantization=False)
            if ret != 0:
                raise RuntimeError(f"rknn.build failed with code {ret}.")

            if calibration_dataset is not None:
                print("Running conversion-time correctness check on one representative image …")
                passed, reason = _check_build_for_collapse(
                    rknn, reference_onnx_path, calibration_dataset
                )
                if not passed:
                    if precision == "int8":
                        raise RuntimeError(
                            f"INT8 quantization collapse detected — refusing to "
                            f"emit {output_path.name}.\n{reason}\n"
                            "This is the stock-Ultralytics-export failure mode "
                            "documented in TROUBLESHOOTING.md §7: the fused "
                            "sigmoid/DFL/concat output collapses under "
                            "per-tensor INT8 quantization. Use Rockchip's "
                            "reference ONNX export (rknn_model_zoo/examples/"
                            "yolov8/model/download_model.sh) instead of the "
                            "stock Ultralytics export, or the "
                            "airockchip/ultralytics_yolov8 fork for a "
                            "custom-trained model — or use rknn-fp16 instead."
                        )
                    print(
                        f"Warning: {reason}\n"
                        "Proceeding with the FP16 export anyway — FP16 should "
                        "not exhibit this collapse pattern, so investigate before "
                        "trusting this artifact."
                    )
                elif reason is not None:
                    print(f"✓ Conversion-time correctness check passed. {reason}")
                else:
                    print(
                        "✓ Conversion-time correctness check skipped (see "
                        "warning above) — proceeding without a detection-"
                        "presence verdict."
                    )

                if skip_validate_conversion:
                    print(
                        "Skipping general conversion validator "
                        "(skip_validate_conversion=True) — no corruption verdict "
                        "beyond the detection-presence check above."
                    )
                else:
                    print(
                        "Running general conversion validator (float ONNX vs. "
                        "RKNN simulator, layout-agnostic channel-band check) …"
                    )
                    image = _load_representative_image(reference_onnx_path, calibration_dataset)
                    validation_result = validate_conversion(
                        reference_onnx_path,
                        image,
                        mean_values=config_kwargs["mean_values"],
                        std_values=config_kwargs["std_values"],
                        rknn=rknn,
                    )
                    if not validation_result.passed:
                        raise RuntimeError(
                            _format_validation_failure(validation_result, output_path)
                        )
                    print(
                        "✓ General conversion validator passed "
                        f"({validation_result.meaningful_bands}/"
                        f"{validation_result.total_bands} channel bands had real "
                        "signal, none collapsed)."
                    )

            print(f"Exporting → {output_path} …")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            ret = rknn.export_rknn(str(output_path))
            if ret != 0:
                raise RuntimeError(f"rknn.export_rknn failed with code {ret}.")

            # The build record travels with the artifact: the device that runs it verifies
            # the two against each other and against the .onnx it is given.
            record = rknn_artifact.write_record(
                output_path,
                rknn_artifact.key_inputs(
                    model_hash=rknn_artifact.file_sha256(onnx_path),
                    precision=precision,
                    target_platform=target_platform,
                    model_inputs=_record_model_inputs(
                        onnx_path, load_onnx_path, nhwc_input[0] if nhwc_input else None
                    ),
                    normalization={
                        "mean_values": config_kwargs["mean_values"],
                        "std_values": config_kwargs["std_values"],
                    },
                    feed_dtype=_RKNN_FEED_DTYPE,
                    graph_edits=graph_edits,
                    calibration_digest=(
                        rknn_artifact.calibration_digest(
                            _calibration_image_paths(calibration_dataset), _RKNN_QUANTIZED_DTYPE
                        )
                        if precision == "int8" and calibration_dataset is not None
                        else None
                    ),
                    compiler_version=toolkit_version,
                    builder_config_version=RKNN_BUILDER_CONFIG_VERSION,
                ),
                extra={"calbredge_version": __version__, "artifact_name": output_path.name},
            )

            print(f"✓ Conversion successful → {output_path}")
            print(
                f"✓ Build record → {rknn_artifact.record_path(output_path)} "
                f"(build_key {record['build_key'][:16]}); copy both files to the device"
            )
        finally:
            rknn.release()
