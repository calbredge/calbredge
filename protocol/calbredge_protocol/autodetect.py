from collections.abc import Callable
from functools import partial
from pathlib import Path

from .adapters import RuntimeAdapter
from .mock_adapter import MockRuntimeAdapter
from .onnx_adapter import ONNXRuntimeAdapter
from .rknn_adapter import RKNNAdapter
from .tensorrt_adapter import TensorRTAdapter
from .tflite_adapter import TFLiteAdapter

# Maps lowercase file extension (including dot) to canonical runtime name.
_EXTENSION_TO_RUNTIME: dict[str, str] = {
    ".onnx": "onnxruntime",
    ".engine": "tensorrt",
    ".tflite": "tflite",
    ".rknn": "rknn",
}

# Maps canonical runtime name to a zero-argument factory that returns a fresh
# adapter instance.  Class types are valid factories (calling them invokes
# __init__); functools.partial is used for precision-qualified TensorRT variants.
_RUNTIME_TO_ADAPTER: dict[str, Callable[[], RuntimeAdapter]] = {
    "onnxruntime": ONNXRuntimeAdapter,
    "tensorrt": TensorRTAdapter,  # defaults to fp32
    "tensorrt-fp32": partial(TensorRTAdapter, precision="fp32"),
    "tensorrt-fp16": partial(TensorRTAdapter, precision="fp16"),
    "tensorrt-int8": partial(TensorRTAdapter, precision="int8"),
    "rknn": partial(RKNNAdapter, precision="fp16"),
    "rknn-fp16": partial(RKNNAdapter, precision="fp16"),
    "rknn-int8": partial(RKNNAdapter, precision="int8"),
    "tflite": TFLiteAdapter,
    "mock": MockRuntimeAdapter,
}


def detect_runtime(model_path: str | Path) -> str:
    """Infer the runtime name from a model file's extension.

    Args:
        model_path: Path to the model file. Only the file extension is
                    inspected; the file does not need to exist.

    Returns:
        A canonical runtime name string suitable for passing to
        :func:`get_adapter`.

    Raises:
        ValueError: If the file extension is not mapped to a known runtime.
                    The error message lists all supported extensions.

    Supported mappings::

        .onnx    → "onnxruntime"
        .engine  → "tensorrt"
        .tflite  → "tflite"
        .rknn    → "rknn"

    Example::

        runtime = detect_runtime("models/mobilenetv2.onnx")
        # → "onnxruntime"
    """
    ext = Path(model_path).suffix.lower()
    if ext not in _EXTENSION_TO_RUNTIME:
        supported = ", ".join(sorted(_EXTENSION_TO_RUNTIME.keys()))
        raise ValueError(
            f"Unrecognised model file extension {ext!r}. Supported extensions: {supported}"
        )
    return _EXTENSION_TO_RUNTIME[ext]


def get_adapter(runtime: str) -> RuntimeAdapter:
    """Return a fresh RuntimeAdapter instance for the given runtime name.

    Args:
        runtime: Canonical runtime name, as returned by :func:`detect_runtime`
                 or supplied directly by the caller (e.g. from a CLI flag).
                 Case-sensitive.

    Returns:
        A new, unloaded :class:`~protocol.adapters.RuntimeAdapter` instance
        for the requested runtime.

    Raises:
        ValueError: If *runtime* is not a known runtime name. The error
                    message lists all available runtime names.

    Available runtimes::

        "onnxruntime"    → ONNXRuntimeAdapter
        "tensorrt"       → TensorRTAdapter(precision="fp32")
        "tensorrt-fp32"  → TensorRTAdapter(precision="fp32")
        "tensorrt-fp16"  → TensorRTAdapter(precision="fp16")
        "tensorrt-int8"  → TensorRTAdapter(precision="int8")
        "rknn"           → RKNNAdapter(precision="fp16")
        "rknn-fp16"      → RKNNAdapter(precision="fp16")
        "rknn-int8"      → RKNNAdapter(precision="int8")
        "tflite"         → TFLiteAdapter  (stub)
        "mock"           → MockRuntimeAdapter

    Example::

        adapter = get_adapter("onnxruntime")
        adapter.load_model("models/mobilenetv2.onnx")
    """
    if runtime not in _RUNTIME_TO_ADAPTER:
        available = ", ".join(sorted(_RUNTIME_TO_ADAPTER.keys()))
        raise ValueError(f"Unknown runtime {runtime!r}. Available runtimes: {available}")
    return _RUNTIME_TO_ADAPTER[runtime]()
