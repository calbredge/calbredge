#!/usr/bin/env python3
"""Download and export benchmark models to ONNX format in tests/open/fixtures/.

Usage:
    python scripts/download_models.py

Models:
    1. YOLOv8s          — via ultralytics (requires torch; pip install ultralytics)
    2. EfficientNetB0   — via tf.keras.applications + tf2onnx
    3. EfficientDet-Lite0 — TFLite download from MediaPipe + tf2onnx
    4. YOLOv8m          — via ultralytics
    5. YOLOv8n-seg      — via ultralytics (segmentation)
    6. RT-DETR-S        — via optimum (HF PekingU/rtdetr_r18vd); falls back to
                          transformers + torch.onnx.export

Any model that fails is skipped with a printed error; the others continue.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

FIXTURES = Path(__file__).parent.parent / "tests" / "open" / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

PASS = "\033[32m✓\033[0m"  # noqa: S105  a coloured check mark for the output, not a password
FAIL = "\033[31m✗\033[0m"


def _pip(package: str) -> None:
    # A fixed argv with no shell, and every caller passes a literal package name.
    subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pip", "install", package, "-q"],
        check=True,
    )


def _verify(path: Path) -> None:
    """Load with onnxruntime and print input shape/dtype and file size."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for inp in sess.get_inputs():
        print(f"    input  {inp.name!r:36s}  shape={inp.shape}  dtype={inp.type}")
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"    size   {size_mb:.1f} MB")


def _export_yolo(model_name: str, dest: Path) -> None:
    """Export any ultralytics YOLO model to ONNX and copy to dest."""
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        print("  Installing ultralytics (requires torch — may be slow) …")
        _pip("ultralytics")
        from ultralytics import YOLO  # type: ignore

    with tempfile.TemporaryDirectory() as _tmp:
        _orig = os.getcwd()
        os.chdir(_tmp)
        try:
            _yolo = YOLO(f"{model_name}.pt")
            _exported = _yolo.export(format="onnx")
            _src = Path(str(_exported))
            if not _src.exists():
                _candidates = list(Path(_tmp).rglob("*.onnx"))
                if not _candidates:
                    raise FileNotFoundError("export produced no .onnx file")
                _src = _candidates[0]
            shutil.copy(_src, dest)
        finally:
            os.chdir(_orig)


# ─── 1. YOLOv8s ───────────────────────────────────────────────────────────────

print("[1/6] YOLOv8s")
_dest = FIXTURES / "yolov8s.onnx"
try:
    _export_yolo("yolov8s", _dest)
    print(f"  {PASS} saved → {_dest}")
    _verify(_dest)
except Exception as _e:
    print(f"  {FAIL} FAILED: {_e}")


# ─── 2. EfficientNet-B0 ───────────────────────────────────────────────────────

print("\n[2/6] EfficientNet-B0")
_dest = FIXTURES / "efficientnet_b0.onnx"
try:
    import onnx  # type: ignore
    import tensorflow as tf  # type: ignore
    import tf2onnx  # type: ignore

    print("  Building tf.keras EfficientNetB0 (imagenet weights) …")
    _model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=True,
        input_shape=(224, 224, 3),
    )

    _input_sig = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name="input"),)
    print("  Converting to ONNX via tf2onnx …")
    _proto, _ = tf2onnx.convert.from_keras(_model, input_signature=_input_sig, opset=13)
    onnx.save(_proto, str(_dest))

    print(f"  {PASS} saved → {_dest}")
    _verify(_dest)

except Exception as _e:
    print(f"  {FAIL} FAILED: {_e}")


# ─── 3. EfficientDet-Lite0 ────────────────────────────────────────────────────

print("\n[3/6] EfficientDet-Lite0")
_dest = FIXTURES / "efficientdet_lite0.onnx"

_TFLITE_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
)

try:
    import onnx  # type: ignore
    import tf2onnx  # type: ignore

    _fd, _tflite_path = tempfile.mkstemp(suffix=".tflite")
    os.close(_fd)
    try:
        print("  Downloading TFLite model from MediaPipe …")
        resp = requests.get(_TFLITE_URL, timeout=60)
        resp.raise_for_status()
        Path(_tflite_path).write_bytes(resp.content)
        size_kb = Path(_tflite_path).stat().st_size / 1024
        print(f"  Downloaded {size_kb:.0f} KB")

        print("  Converting TFLite → ONNX via tf2onnx …")
        _proto, _ = tf2onnx.convert.from_tflite(_tflite_path, opset=13)
        onnx.save(_proto, str(_dest))
    finally:
        if Path(_tflite_path).exists():
            os.unlink(_tflite_path)

    print(f"  {PASS} saved → {_dest}")
    _verify(_dest)

except Exception as _e:
    print(f"  {FAIL} FAILED: {_e}")


# ─── 4. YOLOv8m ───────────────────────────────────────────────────────────────

print("\n[4/6] YOLOv8m")
_dest = FIXTURES / "yolov8m.onnx"
try:
    _export_yolo("yolov8m", _dest)
    print(f"  {PASS} saved → {_dest}")
    _verify(_dest)
except Exception as _e:
    print(f"  {FAIL} FAILED: {_e}")


# ─── 5. YOLOv8n-seg ───────────────────────────────────────────────────────────

print("\n[5/6] YOLOv8n-seg")
_dest = FIXTURES / "yolov8n_seg.onnx"
try:
    _export_yolo("yolov8n-seg", _dest)
    print(f"  {PASS} saved → {_dest}")
    _verify(_dest)
except Exception as _e:
    print(f"  {FAIL} FAILED: {_e}")


# ─── 6. RT-DETR-S ─────────────────────────────────────────────────────────────

print("\n[6/6] RT-DETR-S  (PekingU/rtdetr_r18vd)")
_dest = FIXTURES / "rtdetr_s.onnx"
_HF_MODEL = "PekingU/rtdetr_r18vd"

try:
    # Primary path: optimum
    _optimum_err: Exception | None = None
    try:
        from optimum.exporters.onnx import main_export  # type: ignore

        with tempfile.TemporaryDirectory() as _tmp_dir:
            print("  Exporting via optimum (task=object-detection) …")
            main_export(
                _HF_MODEL,
                output=Path(_tmp_dir),
                task="object-detection",
                no_post_process=True,
            )
            _candidates = sorted(Path(_tmp_dir).glob("*.onnx"))
            if not _candidates:
                raise FileNotFoundError("optimum produced no .onnx in output dir")
            _src = next((c for c in _candidates if c.name == "model.onnx"), _candidates[0])
            shutil.copy(_src, _dest)
    except Exception as _oe:
        _optimum_err = _oe
        print(f"  optimum path failed ({_oe}) — falling back to transformers …")

        import torch  # type: ignore

        try:
            import onnxscript  # type: ignore  # noqa: F401
        except ImportError:
            print("  Installing onnxscript (required by torch.onnx in PyTorch 2.x) …")
            _pip("onnxscript")

        # transformers ≥5.x uses RTDetrForObjectDetection (all-caps RT)
        try:
            from transformers import RTDetrForObjectDetection  # type: ignore
        except ImportError:
            from transformers import (
                RtDetrForObjectDetection as RTDetrForObjectDetection,  # type: ignore
            )

        class _RTDETRWrapper(torch.nn.Module):
            def __init__(self, m: torch.nn.Module) -> None:
                super().__init__()
                self.m = m

            def forward(self, pixel_values: torch.Tensor):  # type: ignore[override]
                out = self.m(pixel_values=pixel_values)
                return out.logits, out.pred_boxes

        import onnx as _onnx  # type: ignore

        print(f"  Loading {_HF_MODEL} via transformers …")
        _hf_model = RTDetrForObjectDetection.from_pretrained(_HF_MODEL)
        _hf_model.eval()
        _wrapper = _RTDETRWrapper(_hf_model)

        _dummy = torch.randn(1, 3, 640, 640)
        print("  Exporting via torch.onnx.export …")
        with tempfile.TemporaryDirectory() as _tmp_dir:
            _tmp_onnx = Path(_tmp_dir) / "rtdetr_s.onnx"
            with torch.no_grad():
                torch.onnx.export(
                    _wrapper,
                    (_dummy,),
                    str(_tmp_onnx),
                    input_names=["pixel_values"],
                    output_names=["logits", "pred_boxes"],
                    dynamic_axes={"pixel_values": {0: "batch"}},
                    opset_version=18,
                )
            # torch dynamo exporter may write weights to a sidecar .data file;
            # reload with external data inlined, then save as a single file.
            _proto = _onnx.load(str(_tmp_onnx), load_external_data=True)
            print("  Consolidating external data into single file …")
            _onnx.save(_proto, str(_dest), save_as_external_data=False)

    if _dest.exists():
        print(f"  {PASS} saved → {_dest}")
        _verify(_dest)
    else:
        print(f"  {FAIL} FAILED: destination file not written")

except Exception as _e:
    print(f"  {FAIL} FAILED: {_e}")


# ─── Summary ──────────────────────────────────────────────────────────────────

print("\n── Summary ─────────────────────────────────────────")
for _name, _fname in [
    ("YOLOv8s", "yolov8s.onnx"),
    ("EfficientNet-B0", "efficientnet_b0.onnx"),
    ("EfficientDet-Lite0", "efficientdet_lite0.onnx"),
    ("YOLOv8m", "yolov8m.onnx"),
    ("YOLOv8n-seg", "yolov8n_seg.onnx"),
    ("RT-DETR-S", "rtdetr_s.onnx"),
]:
    _p = FIXTURES / _fname
    if _p.exists():
        print(f"  {PASS}  {_name:25s}  {_p.stat().st_size / (1024 * 1024):.1f} MB")
    else:
        print(f"  {FAIL}  {_name:25s}  missing")
print("────────────────────────────────────────────────────")
