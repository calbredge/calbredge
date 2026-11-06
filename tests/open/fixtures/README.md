# tests/open/fixtures/

`*.onnx` in this directory are gitignored.
A fresh clone has none of them on disk.

## What the test suite actually needs

Only **`mobilenetv2.onnx`** is loaded by any test (`tests/open/test_agent_e2e.py`,
`tests/open/test_protocol.py`, `tests/open/test_rknn_convert.py`). The other eight files
that show up here on a fully set-up dev machine (`cxr_mobilenetv2.onnx`,
`efficientdet_lite0.onnx`, `efficientnet_b0.onnx`, `rtdetr_s.onnx`,
`yolov8m.onnx`, `yolov8n.onnx`, `yolov8n_seg.onnx`, `yolov8s.onnx`) are the
canonical benchmarking models. They are
used for real benchmark runs, not by anything under `tests/`. You do not need
them to run `pytest`.

## mobilenetv2.onnx: auto-synthesized if missing

`tests/open/conftest.py` writes a tiny synthetic ONNX classifier to this path at
collection time if the real file isn't already there. It has the same input
contract as a real MobileNetV2 export (`(1, 3, 224, 224)` float32 in, a 2D
output) but random weights: no test in this suite asserts specific output
values, only that a model with this shape loads and runs, so the stand-in is
sufficient for `pytest` to pass on a fresh clone with no manual step and no
network access.

If you need the **real** pretrained MobileNetV2 export (e.g. to eyeball
actual inference output, or to benchmark it), no download script produces it
today: no stable public URL for it is recorded anywhere in this repo, and
`scripts/download_models.py` does not cover it (see below). Source one
yourself (e.g. `torchvision.models.mobilenet_v2` or
`tf.keras.applications.MobileNetV2` exported to ONNX opset 17) and drop it at
`tests/open/fixtures/mobilenetv2.onnx`: the synthetic fallback only fires when the
file is absent, so a real file you place here is always preferred.

## The full canonical set (not needed for tests)

`scripts/download_models.py` fetches 6 of the 9 canonical models
(`yolov8s`, `efficientnet_b0`, `efficientdet_lite0`, `yolov8m`,
`yolov8n_seg`, `rtdetr_s`). `mobilenetv2.onnx`, `cxr_mobilenetv2.onnx`, and
`yolov8n.onnx` currently have no provisioning script. None of
this is required to run the test suite.
