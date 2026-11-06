"""Acceptance tests for the layout-agnostic RKNN conversion validator.

These are the proof the design in agent/calbredge/validate_rknn.py works --
they encode the three ground-truth cases established across six exploratory
calibrations (scripts/scratch_rknn_*_calibration.py on branch
scratch/rknn-cosine-calibration): a known-collapsed detector conversion that
must FAIL, a known-healthy detector conversion that must PASS, and a
known-healthy classifier conversion that must PASS (the false-positive
check -- this is the case that broke two earlier design iterations, see
validate_rknn.py's module docstring).

Real end-to-end: builds each RKNN model in-session with the real
rknn-toolkit2 (x86 simulator, no NPU/device needed) and runs the real
onnxruntime float reference, exactly the same way as
scripts/scratch_rknn_signinvariant_calibration.py, which produced the
numbers these tests assert against. No mocking of rknn.api here -- unlike
tests/open/test_rknn_convert.py's mocked-toolkit unit tests, these need the real
toolkit's real quantization behavior, since that behavior is exactly what's
under test. Skipped wherever rknn-toolkit2 or the calibration model files
aren't available (this dev environment: ~/rknn-env; the models live outside
the repo, at the paths the scratch calibrations used -- see MODEL_PATHS
below).
"""

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
from calbredge.validate_rknn import validate_conversion
from onnx import version_converter
from PIL import Image

RKNN_TOOLKIT2_INSTALLED = importlib.util.find_spec("rknn") is not None

# Same model paths used throughout scripts/scratch_rknn_*_calibration.py on
# scratch/rknn-cosine-calibration. All but the classifier live outside the
# repo (this specific dev machine's model cache), so these tests skip
# gracefully wherever they're absent rather than failing.
ZOO_ONNX = Path.home() / "rknn_model_zoo/examples/yolov8/model/yolov8n.onnx"
STOCK_ONNX = Path.home() / "yolov8n_stock_KNOWN_BAD.onnx"
CLASSIFIER_ONNX = Path(__file__).parent / "fixtures" / "mobilenetv2.onnx"
CALIB_DATASET = Path.home() / "dataset.txt"
REAL_IMAGE = Path.home() / "pi_backup_20260721" / "bus.jpg"

TARGET_PLATFORM = "rk3588"
STOCK_ONNX_FOR_RKNN_BUILD = Path(tempfile.gettempdir()) / "test_validate_rknn_stock_opset19.onnx"

MODEL_FILES_PRESENT = all(
    p.exists() for p in (ZOO_ONNX, STOCK_ONNX, CLASSIFIER_ONNX, CALIB_DATASET, REAL_IMAGE)
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RKNN_TOOLKIT2_INSTALLED,
        reason="requires real rknn-toolkit2 (x86-only) -- run inside ~/rknn-env",
    ),
    pytest.mark.skipif(
        not MODEL_FILES_PRESENT,
        reason=(
            "requires the calibration model files from scripts/scratch_rknn_*_calibration.py "
            f"(zoo={ZOO_ONNX.exists()}, stock={STOCK_ONNX.exists()}, "
            f"classifier={CLASSIFIER_ONNX.exists()}, dataset={CALIB_DATASET.exists()}, "
            f"image={REAL_IMAGE.exists()})"
        ),
    ),
]


def _ensure_stock_onnx_opset19():
    """rknn-toolkit2==2.3.2's load_onnx refuses opset > 19; STOCK_ONNX is
    opset 20. Same downgrade used by every scratch calibration script --
    verified there to be byte-identical to the original under onnxruntime."""
    if STOCK_ONNX_FOR_RKNN_BUILD.exists():
        return
    model = onnx.load(str(STOCK_ONNX))
    converted = version_converter.convert_version(model, 19)
    onnx.save(converted, str(STOCK_ONNX_FOR_RKNN_BUILD))


def _input_hw(onnx_path: Path) -> tuple[int, int]:
    model = onnx.load(str(onnx_path))
    dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim if d.dim_value]
    return dims[-2], dims[-1]


def _real_image_hwc_uint8(onnx_path: Path) -> np.ndarray:
    h, w = _input_hw(onnx_path)
    image = Image.open(REAL_IMAGE).convert("RGB").resize((w, h))
    return np.array(image, dtype=np.uint8)


def _build_rknn_insession(onnx_path: Path, mean, std, inputs=None, input_size_list=None):
    """Mirrors validate_rknn._build_rknn_insession -- built independently
    here so the test exercises the SAME live-object integration path a real
    caller (converter.py, right after its own rknn.build()) would use,
    rather than routing through validate_conversion's internal-build path."""
    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    rknn.config(
        target_platform=TARGET_PLATFORM,
        mean_values=mean,
        std_values=std,
        quantized_dtype="asymmetric_quantized-8",
    )
    ret = rknn.load_onnx(model=str(onnx_path), inputs=inputs, input_size_list=input_size_list)
    assert ret == 0, f"load_onnx failed ({ret}) for {onnx_path}"
    ret = rknn.build(do_quantization=True, dataset=str(CALIB_DATASET))
    assert ret == 0, f"build failed ({ret}) for {onnx_path}"
    return rknn


class TestValidateConversionAcceptance:
    def test_bad_detector_fails(self):
        """Known-collapsed conversion: stock (fused-export) yolov8n int8.
        Ground truth from scratch_rknn_signinvariant_calibration.py: class
        channels 4, 9, 15, 29, 62 (0-indexed, all inside the class-score
        range 4:84) collapse from real float values (0.077-0.891) to exactly
        0.0 after quantization. Must FAIL, with collapsed bands whose
        channel ranges fall inside [4, 84)."""
        _ensure_stock_onnx_opset19()
        image = _real_image_hwc_uint8(STOCK_ONNX_FOR_RKNN_BUILD)
        rknn = _build_rknn_insession(STOCK_ONNX_FOR_RKNN_BUILD, [[0, 0, 0]], [[255, 255, 255]])
        try:
            result = validate_conversion(
                STOCK_ONNX,
                image,
                [[0, 0, 0]],
                [[255, 255, 255]],
                rknn=rknn,
            )
        finally:
            rknn.release()

        assert result.verdict == "fail"
        assert result.reason is None, "should fail on band collapse, not structural mismatch"
        assert len(result.collapsed_bands) > 0

        for band in result.collapsed_bands:
            start, end = band.channel_range
            assert start >= 4, (
                f"collapsed band {band.channel_range} falls outside the class range 4:84"
            )
            assert end <= 84, (
                f"collapsed band {band.channel_range} falls outside the class range 4:84"
            )
            assert band.quant_energy == 0.0
            assert band.float_energy > 0.0

    def test_good_detector_passes(self):
        """Known-healthy conversion: Rockchip zoo (unfused-export) yolov8n
        int8. Must PASS -- 0 collapsed bands."""
        image = _real_image_hwc_uint8(ZOO_ONNX)
        rknn = _build_rknn_insession(ZOO_ONNX, [[0, 0, 0]], [[255, 255, 255]])
        try:
            result = validate_conversion(
                ZOO_ONNX,
                image,
                [[0, 0, 0]],
                [[255, 255, 255]],
                rknn=rknn,
            )
        finally:
            rknn.release()

        assert result.verdict == "pass"
        assert result.reason is None
        assert result.collapsed_bands == []
        assert result.meaningful_bands > 0, (
            "sanity check: the check saw real signal, not an empty model"
        )

    def test_healthy_classifier_passes(self):
        """Known-healthy conversion, different model family: mobilenetv2
        int8, normalized mean=127.5 std=127.5 (NOT the YOLO mean=0/std=255
        convention -- see validate_rknn.py's module docstring on why this
        must be sign-invariant: raw signed logits, unlike detector
        sigmoid/box outputs, are never non-negative).

        This is the false-positive check: two earlier design iterations
        (tensor-relative meaningful gate, then signed-max collapse) each
        passed the two detector cases above but broke on exactly this case.
        Exercises validate_conversion's internal-build path (build_params=,
        rknn=None) rather than test_bad/good_detector's live-object path,
        for coverage of both call conventions.
        """
        image = _real_image_hwc_uint8(
            CLASSIFIER_ONNX
        )  # 224x224, from the model's own declared input
        build_params = {
            "target_platform": TARGET_PLATFORM,
            "mean_values": [[127.5, 127.5, 127.5]],
            "std_values": [[127.5, 127.5, 127.5]],
            "quantized_dtype": "asymmetric_quantized-8",
            "calibration_dataset": CALIB_DATASET,
            "inputs": ["pixel_values"],
            "input_size_list": [[1, 3, 224, 224]],
        }

        result = validate_conversion(
            CLASSIFIER_ONNX,
            image,
            [[127.5, 127.5, 127.5]],
            [[127.5, 127.5, 127.5]],
            build_params=build_params,
        )

        assert result.verdict == "pass"
        assert result.reason is None
        assert result.collapsed_bands == []
        assert result.meaningful_bands > 0
