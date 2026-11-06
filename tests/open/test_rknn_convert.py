"""Unit tests for RKNN conversion-time correctness validation.

The real end-to-end conversion (rknn.api.RKNN.config/load_onnx/build) needs
rknn-toolkit2, which is x86-only and not installed in this dev environment —
those code paths are exercised here with a mocked ``rknn.api`` module
(injected via ``sys.modules``, the same pattern used in
test_rknn_adapter.py), never with the check logic itself mocked.

This is metric v5, the last generation of the RKNN conversion-time
correctness check. Four prior generations (v1 nonzero-ratio, v2 max-ratio,
v3 element-wise zeros, and a rejected-pre-merge cosine-similarity proposal)
each compared the quantized output tensor against a float ONNX reference at
some level of aggregation and each was fooled by the same fused
box(~600)+class(0-1) output tensor in a different way — see
check_detection_presence's docstring in agent/calbredge/converter.py for
the full lineage. Generation v4, per-element relative error aggregated as a
per-tensor fraction, shipped and was then reverted on 2026-07-20: it passed
a confirmed-dead stock fused export (26 dead class scores diluted invisible
in a 33,761-element denominator, 0.52% dead fraction) and refused a healthy
Rockchip zoo export (2.94% — legitimate rounding noise). v5 stops comparing
tensors entirely and asks the quantized model a question with a known
answer: does its class-confidence output report a detection at all, on a
real calibration image. find_class_scores and check_detection_presence are
pure functions and are tested directly against fixtures built from the real
incident shapes and values.
"""

import contextlib
import importlib.util
import shutil
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from calbredge.converter import (
    _build_nchw_calibration_dataset,
    _check_build_for_collapse,
    _detect_nhwc_image_input,
    _first_calibration_image,
    _load_calibration_image,
    _onnx_input_hw,
    _resolve_load_onnx_inputs,
    _run_quantized_reference,
    check_detection_presence,
    convert_rknn,
    find_class_scores,
)
from calbredge.validate_rknn import ValidationResult
from PIL import Image

MOBILENETV2 = Path(__file__).parent / "fixtures" / "mobilenetv2.onnx"
# MOBILENETV2's declared input is ['batch_size', 3, 224, 224] -- a symbolic
# batch dimension (dim_param, not dim_value) -- which is what makes it also
# useful as the symbolic-input-shape fixture below, not just the classifier
# fixture used elsewhere in this file.
YOLOV8N = Path(__file__).parent / "fixtures" / "yolov8n.onnx"
# YOLOV8N's declared input is [1, 3, 640, 640] -- fully concrete, no
# symbolic dimension -- the fixed-shape counterpart used to confirm the
# already-working path is untouched.
CXR_MOBILENETV2 = Path(__file__).parent / "fixtures" / "cxr_mobilenetv2.onnx"
# CXR_MOBILENETV2's declared input is ['unk__654', 224, 224, 3] -- both a
# symbolic batch dimension AND an NHWC (channel-last) layout at once, the
# real fixture behind the rknn.load_onnx mean_values axis-mismatch bug
# (input_size_list=[[1, 224, 224, 3]] makes the toolkit read axis 1 -- 224
# -- as the channel count and reject mean_values=[0, 0, 0] as the wrong
# length). Confirms both resolutions compose correctly.

RKNN_TOOLKIT2_INSTALLED = importlib.util.find_spec("rknn") is not None


# ---------------------------------------------------------------------------
# 1. find_class_scores — pure output-layout detector
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindClassScores:
    def test_fused_yolo_shape_extracts_class_rows(self):
        """Fused single-output tensor, real incident shape (1, 84, 8400):
        4 box rows + 80 class rows. Class rows are 4:84."""
        out = np.zeros((1, 84, 8400), dtype=np.float32)
        out[:, :4, :] = 600.0  # box sub-range
        out[:, 4:, :] = 0.7  # class sub-range

        result = find_class_scores([out])

        assert result is not None
        assert len(result) == 1
        assert result[0].shape == (1, 80, 8400)
        assert np.all(result[0] == 0.7)

    def test_custom_class_count_derives_nc_not_hardcoded(self):
        """A custom-trained model with 12 classes: fused shape
        (1, 4+12, 8400) = (1, 16, 8400). Class rows must be derived as 4:16,
        not hardcoded to the COCO 84/80 split."""
        out = np.zeros((1, 16, 8400), dtype=np.float32)
        out[:, 4:, :] = 0.55

        result = find_class_scores([out])

        assert result is not None
        assert len(result) == 1
        assert result[0].shape == (1, 12, 8400)
        assert np.all(result[0] == 0.55)

    def test_classifier_single_2d_output_returns_none(self):
        """A classifier's (1, 1000) output isn't a YOLO layout at all —
        must return None rather than guessing."""
        out = np.random.default_rng(0).uniform(size=(1, 1000)).astype(np.float32)

        assert find_class_scores([out]) is None

    def test_healthy_rockchip_unfused_selects_only_class_tensors(self):
        """Real Rockchip zoo 9-tensor unfused shape: 3 scales, each with a
        DFL box tensor (1,64,S,S), a class tensor (1,80,S,S), and a
        score-sum tensor (1,1,S,S), at S = 80/40/20. Exactly the three
        80-channel tensors must be selected."""
        rng = np.random.default_rng(1)
        outputs = []
        expected_class_tensors = []
        for s in (80, 40, 20):
            box = rng.uniform(-500, 500, size=(1, 64, s, s)).astype(np.float32)
            cls = rng.uniform(0, 0.9, size=(1, 80, s, s)).astype(np.float32)
            score_sum = rng.uniform(0, 1, size=(1, 1, s, s)).astype(np.float32)
            outputs.extend([box, cls, score_sum])
            expected_class_tensors.append(cls)

        result = find_class_scores(outputs)

        assert result is not None
        assert len(result) == 3
        for tensor in result:
            assert tensor.shape[1] == 80
        for expected in expected_class_tensors:
            assert any(np.array_equal(expected, actual) for actual in result)

    def test_unrecognized_multi_output_layout_returns_none(self):
        """Multiple outputs where nothing survives the 64/1-channel
        elimination (e.g. all ndim==4 tensors share channel dim 64, or none
        are ndim==4) must return None, not guess."""
        outputs = [
            np.zeros((1, 64, 80, 80), dtype=np.float32),
            np.zeros((1, 64, 40, 40), dtype=np.float32),
        ]
        assert find_class_scores(outputs) is None

        outputs_2d = [np.zeros((1, 1000), dtype=np.float32), np.zeros((1, 4), dtype=np.float32)]
        assert find_class_scores(outputs_2d) is None


# ---------------------------------------------------------------------------
# 2. check_detection_presence — the pure detection-presence check
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCheckDetectionPresence:
    def test_dead_fused_output_fails(self):
        """Real incident shape and values: fused (1, 84, 8400), box rows
        populated (~600), class rows 4:84 all exactly zero — the confirmed
        on-device-dead bus.jpg build from 2026-07-20 (box max 636.91, cls
        max 0.0, cls nonzero-count 0). Every prior metric (v1-v4) mishandled
        this artifact in one direction or the other; this must fail."""
        out = np.zeros((1, 84, 8400), dtype=np.float32)
        out[:, :4, :] = 600.0
        out[:, 4:, :] = 0.0

        passed, reason = check_detection_presence([out])

        assert passed is False
        assert "0.0000" in reason

    def test_healthy_fused_output_passes_with_evidence(self):
        """Healthy fused output, class max ~0.87 (the known bus.jpg
        confidence used to validate this check all week) — must pass, and
        the evidence must show the observed top score."""
        out = np.zeros((1, 84, 8400), dtype=np.float32)
        out[:, :4, :] = 600.0
        out[:, 4:, :] = 0.1
        out[0, 10, 500] = 0.87

        passed, reason = check_detection_presence([out])

        assert passed is True
        assert "0.8700" in reason

    def test_healthy_rockchip_unfused_passes(self):
        """9-tensor Rockchip zoo shape (real shapes: (1,64,S,S), (1,80,S,S),
        (1,1,S,S) at S=80/40/20), 80-channel tensors with max ~0.9 — must
        pass."""
        rng = np.random.default_rng(2)
        outputs = []
        for s in (80, 40, 20):
            box = rng.uniform(-500, 500, size=(1, 64, s, s)).astype(np.float32)
            cls = rng.uniform(0, 0.3, size=(1, 80, s, s)).astype(np.float32)
            score_sum = rng.uniform(0, 1, size=(1, 1, s, s)).astype(np.float32)
            outputs.extend([box, cls, score_sum])
        outputs[1][0, 5, 0, 0] = 0.9  # one real detection in the first scale's class tensor

        passed, reason = check_detection_presence(outputs)

        assert passed is True
        assert "0.9000" in reason

    def test_dead_rockchip_unfused_fails(self):
        """Same 9-tensor Rockchip shapes, but the 80-channel class tensors
        are all zero while the 64-channel (DFL box) and 1-channel
        (score-sum) tensors remain populated — the unfused-collapse case no
        prior metric was even tested against. Must fail."""
        rng = np.random.default_rng(3)
        outputs = []
        for s in (80, 40, 20):
            box = rng.uniform(-500, 500, size=(1, 64, s, s)).astype(np.float32)
            cls = np.zeros((1, 80, s, s), dtype=np.float32)
            score_sum = rng.uniform(0, 1, size=(1, 1, s, s)).astype(np.float32)
            outputs.extend([box, cls, score_sum])

        passed, reason = check_detection_presence(outputs)

        assert passed is False
        assert "0.0000" in reason

    def test_custom_class_count_fused_works(self):
        """Custom-trained model, fused (1, 4+12, 8400) — class rows 4:16
        derived, not hardcoded, and the check still passes on a healthy
        score."""
        out = np.zeros((1, 16, 8400), dtype=np.float32)
        out[:, :4, :] = 600.0
        out[0, 8, 100] = 0.75

        passed, reason = check_detection_presence([out])

        assert passed is True
        assert "0.7500" in reason

    def test_classifier_shape_skips_with_warning(self, capsys):
        """A classifier's (1, 1000) output isn't a YOLO layout —
        find_class_scores returns None, so the check fails open: a loud
        warning is printed and the check passes (True, None) rather than
        guessing. This is the documented behavior for MobileNetV2 etc."""
        out = np.random.default_rng(0).uniform(size=(1, 1000)).astype(np.float32)

        passed, reason = check_detection_presence([out])

        assert passed is True
        assert reason is None
        captured = capsys.readouterr()
        assert "SKIPPED" in captured.out

    def test_custom_min_score_threshold(self):
        """A custom min_score is respected: a score that passes the default
        0.3 threshold fails once the threshold is raised above it."""
        out = np.zeros((1, 84, 8400), dtype=np.float32)
        out[:, :4, :] = 600.0
        out[0, 10, 500] = 0.5

        passed, _ = check_detection_presence([out])
        assert passed is True

        passed, reason = check_detection_presence([out], min_score=0.6)
        assert passed is False
        assert "0.5000" in reason


# ---------------------------------------------------------------------------
# 3. Calibration dataset / image plumbing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFirstCalibrationImage:
    def test_returns_first_nonblank_line(self, tmp_path):
        dataset = tmp_path / "dataset.txt"
        dataset.write_text("\n/images/a.jpg\n/images/b.jpg\n")
        assert _first_calibration_image(dataset) == Path("/images/a.jpg")

    def test_empty_dataset_raises(self, tmp_path):
        dataset = tmp_path / "dataset.txt"
        dataset.write_text("\n\n")
        with pytest.raises(ValueError, match="lists no images"):
            _first_calibration_image(dataset)

    def test_relative_entry_resolves_against_dataset_parent_not_cwd(self, tmp_path, monkeypatch):
        """Regression test: a dataset.txt entry like `calib_images/coco_1.jpg`
        must resolve relative to dataset.txt's own directory, not the
        process's CWD. This bit during the 2026-07-21 acceptance runs — the
        same dataset.txt worked on the Pi (run from beside calib_images/) and
        failed on the laptop (run from elsewhere)."""
        dataset_dir = tmp_path / "device_a"
        dataset_dir.mkdir()
        dataset = dataset_dir / "dataset.txt"
        dataset.write_text("calib_images/coco_1.jpg\n")

        # Run from an unrelated CWD that has no calib_images/ at all — the
        # old CWD-relative behavior would return a path that doesn't resolve
        # to anything under dataset_dir.
        elsewhere = tmp_path / "device_b"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        result = _first_calibration_image(dataset)
        assert result == dataset_dir / "calib_images" / "coco_1.jpg"


class _FakeOnnxInput:
    def __init__(self, shape, name="input"):
        self.shape = shape
        self.name = name


class _FakeOnnxSession:
    """Stand-in for onnxruntime.InferenceSession exposing only get_inputs()."""

    def __init__(self, shape, name="input"):
        self._shape = shape
        self._name = name

    def get_inputs(self):
        return [_FakeOnnxInput(self._shape, self._name)]


@pytest.mark.unit
class TestOnnxInputHw:
    def test_reads_height_and_width_from_real_fixture(self):
        height, width = _onnx_input_hw(MOBILENETV2)
        assert (height, width) == (224, 224)

    def test_nchw_unambiguous_shape_returns_height_width(self):
        """[1,3,640,640] (NCHW) — trailing dims (640, 640), width=640 does not
        look like a channel count, so the existing correct behavior holds."""
        with patch("onnxruntime.InferenceSession", return_value=_FakeOnnxSession([1, 3, 640, 640])):
            height, width = _onnx_input_hw(Path("dummy.onnx"))
        assert (height, width) == (640, 640)

    def test_nhwc_shape_raises_instead_of_mislabeling_channels_as_width(self):
        """[1,224,224,3] (NHWC) — dims[-1]=3 looks like a channel count, so the
        function must fail loud rather than silently return (224, 3)."""
        with (
            patch("onnxruntime.InferenceSession", return_value=_FakeOnnxSession([1, 224, 224, 3])),
            pytest.raises(ValueError, match="channel count"),
        ):
            _onnx_input_hw(Path("dummy.onnx"))

    def test_too_few_dims_raises(self):
        """A shape with fewer than 3 dims has no unambiguous (H, W) pair."""
        with (
            patch("onnxruntime.InferenceSession", return_value=_FakeOnnxSession([1, 1000])),
            pytest.raises(ValueError, match="fewer than"),
        ):
            _onnx_input_hw(Path("dummy.onnx"))


@pytest.mark.unit
class TestResolveLoadOnnxInputs:
    """rknn.load_onnx rejects a symbolic input dimension outright (real bug:
    ValueError: The input shape ['batch_size', 3, 224, 224] of 'pixel_values'
    is not support!, confirmed on MobileNetV2's ONNX export). This resolves
    a symbolic BATCH dimension to Calbredge's always-batch_size=1 convention
    and leaves an already-concrete shape (YOLOv8n) untouched."""

    def test_concrete_shape_returns_none_none(self):
        """YOLOv8n's fully concrete [1, 3, 640, 640] input needs no
        override — load_onnx already handles it — so the caller must be
        told to invoke load_onnx exactly as before, with no override."""
        inputs, input_size_list = _resolve_load_onnx_inputs(YOLOV8N)
        assert inputs is None
        assert input_size_list is None

    def test_symbolic_batch_dim_resolves_to_one(self):
        """MobileNetV2's real declared input is ['batch_size', 3, 224, 224]
        — a symbolic dimension (dim_param, not dim_value) at axis 0. It
        must resolve to a concrete batch_size=1, since Calbredge always
        benchmarks at batch_size=1."""
        inputs, input_size_list = _resolve_load_onnx_inputs(MOBILENETV2)
        assert inputs == ["pixel_values"]
        assert input_size_list == [[1, 3, 224, 224]]

    def test_symbolic_non_batch_dim_raises(self):
        """A symbolic dimension outside the batch axis is not guessed —
        this module never guesses at a layout it can't read off the graph
        (see _onnx_input_hw's docstring above) — so it must raise rather
        than silently pick a value for an unknown height."""
        with (
            patch(
                "onnxruntime.InferenceSession",
                return_value=_FakeOnnxSession([1, 3, "height", 224], name="pixels"),
            ),
            pytest.raises(ValueError, match="isn't the batch dimension"),
        ):
            _resolve_load_onnx_inputs(Path("dummy.onnx"))

    def test_nhwc_shape_only_resolves_batch_dim_no_longer_reorders(self):
        """cxr_mobilenetv2's real declared input is ['unk__654', 224, 224, 3]
        — a symbolic batch dim AND an NHWC layout at once. NHWC handling now
        lives entirely upstream of this function (see
        _detect_nhwc_image_input and convert_rknn's onnx_edit-based graph
        transpose) — by the time onnx_path reaches _resolve_load_onnx_inputs
        it is either already NCHW, or is convert_rknn's onnx_edit-
        transposed copy. Called directly on the ORIGINAL (still-NHWC)
        fixture, this function only resolves the symbolic batch dimension
        and leaves the NHWC axis order untouched — it no longer reorders."""
        inputs, input_size_list = _resolve_load_onnx_inputs(CXR_MOBILENETV2)
        assert inputs == ["input_layer_1"]
        assert input_size_list == [[1, 224, 224, 3]]

    def test_already_nchw_shape_not_reordered(self):
        """YOLOv8n's [1, 3, 640, 640] is already NCHW (last dim 640 isn't a
        plausible channel count) — confirms no override is needed here
        regardless of the NHWC-handling change elsewhere. Same assertion as
        test_concrete_shape_returns_none_none above; kept as its own test
        for clarity."""
        inputs, input_size_list = _resolve_load_onnx_inputs(YOLOV8N)
        assert inputs is None
        assert input_size_list is None


@pytest.mark.unit
class TestDetectNhwcImageInput:
    """Direct tests of the NHWC-detection helper that replaced
    _reorder_input_size_list_to_nchw — see its docstring in
    agent/calbredge/converter.py for why an input_size_list-only reorder
    isn't enough (it satisfies rknn.load_onnx's mean_values axis check but
    leaves rknn.build's graph executor believing the input is NCHW while
    the graph's own Conv weights still expect NHWC data)."""

    def test_unambiguous_nhwc_returns_name_height_width_channels(self):
        """[1, 224, 224, 3]: last dim (3) is channel-like, axis 1 (224) is
        not — unambiguous NHWC."""
        with patch(
            "onnxruntime.InferenceSession",
            return_value=_FakeOnnxSession([1, 224, 224, 3], name="pixels"),
        ):
            result = _detect_nhwc_image_input(Path("dummy.onnx"))
        assert result == ("pixels", 224, 224, 3)

    def test_already_nchw_shape_returns_none(self):
        """[1, 3, 640, 640]: last dim (640) is not channel-like, so this is
        assumed already NCHW-ordered — not detected as NHWC."""
        with patch(
            "onnxruntime.InferenceSession",
            return_value=_FakeOnnxSession([1, 3, 640, 640], name="images"),
        ):
            assert _detect_nhwc_image_input(Path("dummy.onnx")) is None

    def test_ambiguous_shape_raises_instead_of_guessing(self):
        """[1, 3, 3, 3]: last dim (3) AND axis 1 (3) are both plausible
        channel counts — genuinely ambiguous, so this must raise rather
        than silently pick one interpretation."""
        with (
            patch(
                "onnxruntime.InferenceSession",
                return_value=_FakeOnnxSession([1, 3, 3, 3], name="input"),
            ),
            pytest.raises(ValueError, match="Cannot unambiguously resolve the channel axis"),
        ):
            _detect_nhwc_image_input(Path("dummy.onnx"))

    def test_single_channel_shape_detected(self):
        """[1, 224, 224, 1]: a plausible single-channel (grayscale) NHWC
        shape — last dim (1) is channel-like, axis 1 (224) is not."""
        with patch(
            "onnxruntime.InferenceSession",
            return_value=_FakeOnnxSession([1, 224, 224, 1], name="input"),
        ):
            result = _detect_nhwc_image_input(Path("dummy.onnx"))
        assert result == ("input", 224, 224, 1)

    def test_too_few_dims_returns_none(self):
        """A shape with fewer than 3 dims has no distinct "axis 1" vs. "last
        axis" to compare, so this returns None rather than raising or
        guessing — mirrors _onnx_input_hw's fewer-than-3-dims guard."""
        with patch(
            "onnxruntime.InferenceSession",
            return_value=_FakeOnnxSession([1, 1000], name="logits"),
        ):
            assert _detect_nhwc_image_input(Path("dummy.onnx")) is None

    def test_symbolic_height_or_width_raises(self):
        """A symbolic (non-batch) height dimension can't be read off the
        graph, so this must raise rather than guess."""
        with (
            patch(
                "onnxruntime.InferenceSession",
                return_value=_FakeOnnxSession(["batch", "height", 224, 3], name="input"),
            ),
            pytest.raises(ValueError, match="height and/or width"),
        ):
            _detect_nhwc_image_input(Path("dummy.onnx"))

    def test_real_cxr_mobilenetv2_fixture_detected(self):
        result = _detect_nhwc_image_input(CXR_MOBILENETV2)
        assert result == ("input_layer_1", 224, 224, 3)

    def test_real_yolov8n_fixture_not_detected(self):
        assert _detect_nhwc_image_input(YOLOV8N) is None


@pytest.mark.unit
class TestBuildNchwCalibrationDataset:
    def test_generates_one_npy_per_image_in_nchw_layout(self, tmp_path):
        img1 = tmp_path / "a.png"
        img2 = tmp_path / "b.png"
        Image.new("RGB", (300, 300), color=(1, 2, 3)).save(img1)
        Image.new("RGB", (300, 300), color=(4, 5, 6)).save(img2)
        dataset = tmp_path / "dataset.txt"
        dataset.write_text(f"{img1}\n{img2}\n")

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        new_dataset = _build_nchw_calibration_dataset(dataset, out_dir, height=224, width=224)

        lines = [line for line in new_dataset.read_text().splitlines() if line]
        assert len(lines) == 2
        for line in lines:
            arr = np.load(line)
            assert arr.shape == (1, 3, 224, 224)
            assert arr.dtype == np.uint8

    def test_empty_dataset_raises(self, tmp_path):
        dataset = tmp_path / "dataset.txt"
        dataset.write_text("\n\n")
        with pytest.raises(ValueError, match="lists no images"):
            _build_nchw_calibration_dataset(dataset, tmp_path, height=224, width=224)


@pytest.mark.unit
class TestLoadCalibrationImage:
    def test_loads_and_resizes_to_hwc_uint8(self, tmp_path):
        image_path = tmp_path / "sample.png"
        Image.new("RGB", (100, 50), color=(10, 20, 30)).save(image_path)

        result = _load_calibration_image(image_path, width=64, height=48)

        assert result.shape == (48, 64, 3)
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# 4. Quantized reference runner
# ---------------------------------------------------------------------------


class _FakeRKNN:
    """Minimal fake rknn.api.RKNN instance for testing the plumbing helpers
    directly (not through sys.modules injection)."""

    def __init__(self, outputs, init_runtime_ret=0):
        self._outputs = outputs
        self._init_runtime_ret = init_runtime_ret
        self.init_runtime_called = False
        self.last_inference_inputs = None

    def init_runtime(self):
        self.init_runtime_called = True
        return self._init_runtime_ret

    def inference(self, inputs):
        self.last_inference_inputs = inputs
        return self._outputs


@pytest.mark.unit
class TestRunQuantizedReference:
    def test_adds_batch_dim_and_calls_init_runtime_then_inference(self):
        rknn = _FakeRKNN(outputs=[np.ones((1, 1000), dtype=np.float32)])
        image = np.zeros((224, 224, 3), dtype=np.uint8)

        outputs = _run_quantized_reference(rknn, image)

        assert rknn.init_runtime_called
        assert rknn.last_inference_inputs[0].shape == (1, 224, 224, 3)
        assert outputs[0].shape == (1, 1000)

    def test_init_runtime_failure_raises(self):
        rknn = _FakeRKNN(outputs=[], init_runtime_ret=-1)
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="init_runtime"):
            _run_quantized_reference(rknn, image)


@pytest.mark.unit
class TestCheckBuildForCollapse:
    def test_wires_quantized_reference_to_detection_presence(self, tmp_path):
        """End-to-end for the wiring function (real image loading, fake
        rknn) — proves the plumbing feeds check_detection_presence
        correctly without mocking the check itself. The fake RKNN returns
        a classifier-shaped output, which find_class_scores doesn't
        recognize, so the check fails open (skips, passes)."""
        image_path = tmp_path / "sample.png"
        Image.new("RGB", (224, 224), color=(128, 128, 128)).save(image_path)
        dataset = tmp_path / "dataset.txt"
        dataset.write_text(str(image_path) + "\n")

        rknn = _FakeRKNN(outputs=[np.zeros((1, 1001), dtype=np.float32)])

        passed, reason = _check_build_for_collapse(rknn, MOBILENETV2, dataset)

        assert passed is True
        assert reason is None

    def test_wires_dead_fused_output_to_failure(self, tmp_path):
        """Same wiring, but with a dead fused-YOLO-shaped output — must
        fail, proving the plumbing surfaces a real detection-presence
        failure end-to-end."""
        image_path = tmp_path / "sample.png"
        Image.new("RGB", (224, 224), color=(128, 128, 128)).save(image_path)
        dataset = tmp_path / "dataset.txt"
        dataset.write_text(str(image_path) + "\n")

        out = np.zeros((1, 84, 8400), dtype=np.float32)
        out[:, :4, :] = 600.0
        rknn = _FakeRKNN(outputs=[out])

        passed, reason = _check_build_for_collapse(rknn, MOBILENETV2, dataset)

        assert passed is False
        assert reason is not None


# ---------------------------------------------------------------------------
# 5. convert_rknn orchestration — mocked rknn.api.RKNN via sys.modules
# ---------------------------------------------------------------------------


def _make_rknn_api_mock(
    inference_outputs, build_ret=0, load_onnx_ret=0, export_ret=0, instances=None
):
    """Build a fake ``rknn.api`` module whose RKNN instance's inference()
    returns *inference_outputs* for the correctness check.

    Args:
        instances: Optional list; if given, every constructed RKNN instance
                   appends itself to it, letting a test inspect e.g.
                   ``config_calls`` from outside convert_rknn's internal
                   ``rknn = RKNN(...)`` instance.
    """
    rknn_api = types.ModuleType("rknn.api")

    class _RKNN:
        def __init__(self, verbose=False):
            self.exported_to = None
            self.config_calls: list[dict] = []
            self.load_onnx_calls: list[dict] = []
            self.build_calls: list[dict] = []
            if instances is not None:
                instances.append(self)

        def config(self, **kwargs):
            self.config_calls.append(kwargs)

        def load_onnx(self, model, inputs=None, input_size_list=None):
            self.load_onnx_calls.append(
                {"model": model, "inputs": inputs, "input_size_list": input_size_list}
            )
            return load_onnx_ret

        def build(self, do_quantization, dataset=None):
            self.build_calls.append({"do_quantization": do_quantization, "dataset": dataset})
            return build_ret

        def init_runtime(self):
            return 0

        def inference(self, inputs):
            return inference_outputs

        def export_rknn(self, path):
            self.exported_to = path
            Path(path).write_bytes(b"fake rknn bytes")
            return export_ret

        def release(self):
            pass

    rknn_api.RKNN = _RKNN
    return rknn_api


def _make_onnx_edit_spy(calls: list[dict]):
    """Fake rknn.utils.onnx_edit: records its call kwargs and copies the
    source .onnx to export_path unchanged.

    A real onnx_edit call rewrites the graph itself to genuine NCHW; this
    fake doesn't need to reproduce that (rknn.load_onnx/build are
    themselves mocked in these tests via _make_rknn_api_mock) — it only
    needs to leave a file at export_path so downstream onnxruntime-based
    inspection (_resolve_load_onnx_inputs) has something to open, and to
    record what convert_rknn asked it to do.
    """

    def _onnx_edit(model, export_path, inputs_transform):
        calls.append(
            {"model": model, "export_path": export_path, "inputs_transform": inputs_transform}
        )
        shutil.copy(model, export_path)

    return _onnx_edit


FAKE_TOOLKIT_VERSION = "2.3.2"


@contextlib.contextmanager
def _patch_rknn_api(rknn_api_mock, onnx_edit=None, toolkit_version=FAKE_TOOLKIT_VERSION):
    """A fake rknn-toolkit2: its modules, and the installed version its build record names."""
    rknn_pkg = types.ModuleType("rknn")
    rknn_utils = types.ModuleType("rknn.utils")
    rknn_utils.onnx_edit = onnx_edit if onnx_edit is not None else _make_onnx_edit_spy([])
    modules = {"rknn": rknn_pkg, "rknn.api": rknn_api_mock, "rknn.utils": rknn_utils}
    with (
        patch.dict(sys.modules, modules),
        patch("calbredge.converter._rknn_toolkit_version", return_value=toolkit_version),
    ):
        yield


@pytest.fixture
def calibration_dataset(tmp_path):
    image_path = tmp_path / "calib.png"
    Image.new("RGB", (224, 224), color=(100, 150, 200)).save(image_path)
    dataset_path = tmp_path / "dataset.txt"
    dataset_path.write_text(str(image_path) + "\n")
    return dataset_path


@pytest.fixture
def passing_validate_conversion():
    """Patches calbredge.converter.validate_conversion (the general,
    layout-agnostic conversion validator) to a canned PASS.

    Used by tests below convert_rknn's orchestration of
    check_detection_presence, which use fabricated YOLO-shaped fake RKNN
    outputs against the real MOBILENETV2 fixture (a classifier) purely to
    drive check_detection_presence's pass/fail/skip branches — those
    fabricated shapes don't match MOBILENETV2's real (1, 1001) float
    output, so validate_conversion would otherwise fail on a spurious
    output-shape mismatch that has nothing to do with what each of those
    tests is actually exercising. TestConvertRknnValidateConversionWiring
    below tests validate_conversion's real wiring and refusal behavior
    without this patch.
    """
    with patch(
        "calbredge.converter.validate_conversion",
        return_value=ValidationResult(passed=True, meaningful_bands=1, total_bands=1),
    ) as mock:
        yield mock


def _dead_fused_yolo_output():
    """Real incident shape and values: fused (1, 84, 8400), box rows
    populated (~600), class rows all exactly zero."""
    out = np.zeros((1, 84, 8400), dtype=np.float32)
    out[:, :4, :] = 600.0
    return out


def _healthy_fused_yolo_output():
    """Fused (1, 84, 8400) with a real detection: class max ~0.87."""
    out = np.zeros((1, 84, 8400), dtype=np.float32)
    out[:, :4, :] = 600.0
    out[:, 4:, :] = 0.1
    out[0, 10, 500] = 0.87
    return out


@pytest.mark.unit
class TestConvertRknnOrchestration:
    def test_int8_dead_detection_raises_and_never_writes_output(
        self, tmp_path, calibration_dataset
    ):
        output_path = tmp_path / "model.rknn"
        rknn_api = _make_rknn_api_mock(inference_outputs=[_dead_fused_yolo_output()])
        with (
            _patch_rknn_api(rknn_api),
            pytest.raises(RuntimeError, match="INT8 quantization collapse"),
        ):
            convert_rknn(
                MOBILENETV2,
                output_path,
                precision="int8",
                calibration_dataset=calibration_dataset,
            )
        assert not output_path.exists()

    def test_int8_healthy_build_writes_output(
        self, tmp_path, calibration_dataset, passing_validate_conversion
    ):
        output_path = tmp_path / "model.rknn"
        rknn_api = _make_rknn_api_mock(inference_outputs=[_healthy_fused_yolo_output()])
        with _patch_rknn_api(rknn_api):
            convert_rknn(
                MOBILENETV2,
                output_path,
                precision="int8",
                calibration_dataset=calibration_dataset,
            )
        assert output_path.exists()

    def test_int8_unrecognized_layout_skips_and_writes_output(
        self, tmp_path, calibration_dataset, passing_validate_conversion
    ):
        """MobileNetV2-shaped output (classifier, not YOLO): the check
        skips with a warning rather than guessing, and the build proceeds —
        this is the documented fail-open behavior for non-YOLO models."""
        output_path = tmp_path / "model.rknn"
        rknn_api = _make_rknn_api_mock(inference_outputs=[np.zeros((1, 1001), dtype=np.float32)])
        with _patch_rknn_api(rknn_api):
            convert_rknn(
                MOBILENETV2,
                output_path,
                precision="int8",
                calibration_dataset=calibration_dataset,
            )
        assert output_path.exists()

    def test_fp16_dead_detection_warns_but_still_writes_output(
        self, tmp_path, calibration_dataset, capsys, passing_validate_conversion
    ):
        output_path = tmp_path / "model.rknn"
        rknn_api = _make_rknn_api_mock(inference_outputs=[_dead_fused_yolo_output()])
        with _patch_rknn_api(rknn_api):
            convert_rknn(
                MOBILENETV2,
                output_path,
                precision="fp16",
                calibration_dataset=calibration_dataset,
            )
        assert output_path.exists()
        captured = capsys.readouterr()
        assert "Warning" in captured.out

    def test_int8_without_calibration_dataset_raises_before_importing_rknn(self, tmp_path):
        output_path = tmp_path / "model.rknn"
        with pytest.raises(ValueError, match="calibration_dataset is required"):
            convert_rknn(MOBILENETV2, output_path, precision="int8")

    def test_both_precisions_send_identical_mean_std_to_config(
        self, tmp_path, calibration_dataset, passing_validate_conversion
    ):
        """int8 and fp16 artifacts must share the same input contract — the
        adapter assumes uint8 raw pixels for both — so both branches must
        configure identical mean_values/std_values. Only quantized_dtype
        should differ."""
        healthy_output = [_healthy_fused_yolo_output()]

        int8_instances: list = []
        rknn_api_int8 = _make_rknn_api_mock(
            inference_outputs=healthy_output, instances=int8_instances
        )
        with _patch_rknn_api(rknn_api_int8):
            convert_rknn(
                MOBILENETV2,
                tmp_path / "int8.rknn",
                precision="int8",
                calibration_dataset=calibration_dataset,
            )

        fp16_instances: list = []
        rknn_api_fp16 = _make_rknn_api_mock(
            inference_outputs=healthy_output, instances=fp16_instances
        )
        with _patch_rknn_api(rknn_api_fp16):
            convert_rknn(
                MOBILENETV2,
                tmp_path / "fp16.rknn",
                precision="fp16",
                calibration_dataset=calibration_dataset,
            )

        int8_config = int8_instances[-1].config_calls[-1]
        fp16_config = fp16_instances[-1].config_calls[-1]

        assert int8_config["mean_values"] == fp16_config["mean_values"] == [[0, 0, 0]]
        assert int8_config["std_values"] == fp16_config["std_values"] == [[255, 255, 255]]
        assert "quantized_dtype" in int8_config
        assert "quantized_dtype" not in fp16_config

    def test_missing_rknn_toolkit_exits_with_clear_message(self, tmp_path, capsys):
        """No sys.modules patch here — this dev environment genuinely does
        not have rknn-toolkit2 installed, so this exercises the real
        ImportError path."""
        output_path = tmp_path / "model.rknn"
        with pytest.raises(SystemExit):
            convert_rknn(MOBILENETV2, output_path, precision="fp16")
        captured = capsys.readouterr()
        assert "pip install -c constraints/rknn-convert-x86_64-py312.txt" in captured.out
        assert "--extra-index-url https://download.pytorch.org/whl/cpu" in captured.out
        root = Path(__file__).resolve().parents[2]
        assert (root / "constraints" / "rknn-convert-x86_64-py312.txt").is_file()

    def test_no_protobuf_variable_is_asked_for(self, tmp_path, capsys, monkeypatch):
        """The constraints file pins a protobuf the toolkit converts with under the default
        implementation, so conversion neither needs nor mentions the variable."""
        monkeypatch.delenv("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", raising=False)
        rknn_api = _make_rknn_api_mock(inference_outputs=[])
        with _patch_rknn_api(rknn_api):
            convert_rknn(MOBILENETV2, tmp_path / "model.rknn", precision="fp16")

        assert "PROTOCOL_BUFFERS" not in capsys.readouterr().out


@pytest.mark.unit
class TestConvertRknnLoadOnnxInputOverride:
    """End-to-end confirmation of the load_onnx wiring: a symbolic-batch-
    dimension ONNX model (MobileNetV2) gets a concrete inputs/
    input_size_list override, and an already-concrete ONNX model (YOLOv8n
    — the path that already worked) gets none, unchanged from before this
    fix. Uses precision="fp16" with no calibration_dataset so neither
    correctness check runs, isolating this test to the load_onnx call
    itself."""

    def test_symbolic_batch_dim_model_gets_concrete_override(self, tmp_path):
        output_path = tmp_path / "model.rknn"
        instances: list = []
        rknn_api = _make_rknn_api_mock(inference_outputs=[], instances=instances)
        with _patch_rknn_api(rknn_api):
            convert_rknn(MOBILENETV2, output_path, precision="fp16")

        assert output_path.exists()
        load_onnx_call = instances[-1].load_onnx_calls[-1]
        assert load_onnx_call["inputs"] == ["pixel_values"]
        assert load_onnx_call["input_size_list"] == [[1, 3, 224, 224]]

    def test_fixed_shape_model_gets_no_override(self, tmp_path):
        output_path = tmp_path / "model.rknn"
        instances: list = []
        rknn_api = _make_rknn_api_mock(inference_outputs=[], instances=instances)
        with _patch_rknn_api(rknn_api):
            convert_rknn(YOLOV8N, output_path, precision="fp16")

        assert output_path.exists()
        load_onnx_call = instances[-1].load_onnx_calls[-1]
        assert load_onnx_call["inputs"] is None
        assert load_onnx_call["input_size_list"] is None


@pytest.mark.unit
class TestConvertRknnOnnxEditNhwcTranspose:
    """Confirms the onnx_edit-based NHWC-to-NCHW graph transpose (replacing
    the input_size_list-only reorder that broke rknn.build's constant-
    folding on genuinely-NHWC-graph models — see
    _detect_nhwc_image_input's docstring) is wired into
    convert_rknn: a genuinely-NHWC model (cxr_mobilenetv2) routes through
    rknn.utils.onnx_edit before load_onnx, and an already-NCHW model
    (YOLOv8n) does not. Uses precision="fp16" with no calibration_dataset
    so neither correctness check runs, isolating these tests to the
    onnx_edit/load_onnx wiring itself."""

    def test_nhwc_model_routes_through_onnx_edit(self, tmp_path):
        output_path = tmp_path / "model.rknn"
        instances: list = []
        rknn_api = _make_rknn_api_mock(inference_outputs=[], instances=instances)
        edit_calls: list = []
        with _patch_rknn_api(rknn_api, onnx_edit=_make_onnx_edit_spy(edit_calls)):
            convert_rknn(CXR_MOBILENETV2, output_path, precision="fp16")

        assert output_path.exists()
        assert len(edit_calls) == 1
        assert edit_calls[0]["model"] == str(CXR_MOBILENETV2)
        assert edit_calls[0]["inputs_transform"] == {"input_layer_1": "a,b,c,d->a,d,b,c"}

        load_onnx_call = instances[-1].load_onnx_calls[-1]
        assert load_onnx_call["model"] == edit_calls[0]["export_path"]
        assert load_onnx_call["model"] != str(CXR_MOBILENETV2)

    def test_already_nchw_model_does_not_call_onnx_edit(self, tmp_path):
        output_path = tmp_path / "model.rknn"
        instances: list = []
        rknn_api = _make_rknn_api_mock(inference_outputs=[], instances=instances)
        edit_calls: list = []
        with _patch_rknn_api(rknn_api, onnx_edit=_make_onnx_edit_spy(edit_calls)):
            convert_rknn(YOLOV8N, output_path, precision="fp16")

        assert output_path.exists()
        assert edit_calls == []
        load_onnx_call = instances[-1].load_onnx_calls[-1]
        assert load_onnx_call["model"] == str(YOLOV8N)


# ---------------------------------------------------------------------------
# 6. validate_conversion wiring — the general, layout-agnostic check
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConvertRknnValidateConversionWiring:
    """Confirms validate_conversion (agent/calbredge/validate_rknn.py) is
    actually wired into convert_rknn's happy path — not just imported and
    unused — and that a FAIL verdict blocks export exactly like
    check_detection_presence's own refusal, including on layouts
    check_detection_presence itself fails open on (its documented blind
    spot, and the whole reason this second check exists)."""

    def test_validate_conversion_invoked_with_live_rknn_and_matching_config(
        self, tmp_path, calibration_dataset
    ):
        """Spy on validate_conversion: confirm it's called once, with the
        SAME live in-session rknn object convert_rknn just built (not a
        reload — see validate_rknn.py's integration-constraint docstring),
        the float ONNX path, and the exact mean/std passed to
        rknn.config() for this build."""
        output_path = tmp_path / "model.rknn"
        instances: list = []
        rknn_api = _make_rknn_api_mock(
            inference_outputs=[_healthy_fused_yolo_output()], instances=instances
        )
        passing_result = ValidationResult(passed=True, meaningful_bands=5, total_bands=5)

        with (
            _patch_rknn_api(rknn_api),
            patch(
                "calbredge.converter.validate_conversion", return_value=passing_result
            ) as mock_validate,
        ):
            convert_rknn(
                MOBILENETV2,
                output_path,
                precision="int8",
                calibration_dataset=calibration_dataset,
            )

        assert output_path.exists()
        mock_validate.assert_called_once()
        args, kwargs = mock_validate.call_args
        assert args[0] == MOBILENETV2
        assert isinstance(args[1], np.ndarray)
        assert kwargs["mean_values"] == [[0, 0, 0]]
        assert kwargs["std_values"] == [[255, 255, 255]]
        assert kwargs["rknn"] is instances[-1]

    def test_fail_blocks_export_when_detection_check_fails_open(
        self, tmp_path, calibration_dataset, capsys
    ):
        """Real (unmocked) validate_conversion, run against the real
        MobileNetV2 fixture. check_detection_presence fails open on this
        classifier-shaped output (ndim 2, not a recognized YOLO layout) —
        without validate_conversion, this build would proceed and silently
        write a corrupted artifact. The fake RKNN's inference() returns all
        zeros at the exact (1, 1001) shape the real ONNX float reference
        produces for this calibration image (verified: 126/126 channel
        bands carry real signal, min/max roughly -3.7/5.5), so
        validate_conversion sees total collapse across every meaningful
        band and must refuse the export — proving it, not
        check_detection_presence, is what caught this."""
        output_path = tmp_path / "model.rknn"
        dead_classifier_output = np.zeros((1, 1001), dtype=np.float32)
        rknn_api = _make_rknn_api_mock(inference_outputs=[dead_classifier_output])

        with (
            _patch_rknn_api(rknn_api),
            pytest.raises(RuntimeError, match="General conversion validator"),
        ):
            convert_rknn(
                MOBILENETV2,
                output_path,
                precision="fp16",
                calibration_dataset=calibration_dataset,
            )

        assert not output_path.exists()
        captured = capsys.readouterr()
        assert "SKIPPED" in captured.out  # check_detection_presence failed open
        assert "Running general conversion validator" in captured.out

    def test_skip_flag_bypasses_check_and_allows_export(
        self, tmp_path, calibration_dataset, capsys
    ):
        """With skip_validate_conversion=True, the same all-zero classifier
        output that the previous test proves validate_conversion refuses is
        never checked by it — check_detection_presence still fails open on
        the classifier layout, and (with the general check skipped) the
        build proceeds and exports."""
        output_path = tmp_path / "model.rknn"
        dead_classifier_output = np.zeros((1, 1001), dtype=np.float32)
        rknn_api = _make_rknn_api_mock(inference_outputs=[dead_classifier_output])

        with _patch_rknn_api(rknn_api):
            convert_rknn(
                MOBILENETV2,
                output_path,
                precision="fp16",
                calibration_dataset=calibration_dataset,
                skip_validate_conversion=True,
            )

        assert output_path.exists()
        captured = capsys.readouterr()
        assert "Skipping general conversion validator" in captured.out


@pytest.mark.skipif(
    not RKNN_TOOLKIT2_INSTALLED,
    reason="requires real rknn-toolkit2 (x86-only), not installed in this environment",
)
@pytest.mark.integration
class TestConvertRknnRealToolkit:
    def test_convert_rknn_against_real_toolkit(self):
        """Placeholder for a true end-to-end run against real
        rknn-toolkit2. Skipped wherever the toolkit isn't importable — this
        environment doesn't have it, so this always skips here."""
        pytest.skip("real-toolkit conversion run not exercised in this environment")
