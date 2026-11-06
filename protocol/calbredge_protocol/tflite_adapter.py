from pathlib import Path

from .adapters import RuntimeAdapter


class TFLiteAdapter(RuntimeAdapter):
    """RuntimeAdapter stub for TFLite / LiteRT inference on ARM edge devices.

    This is a stub implementation. The class satisfies the
    :class:`~protocol.adapters.RuntimeAdapter` interface for architecture
    completeness. All four inference methods raise :exc:`NotImplementedError`
    until a full implementation is provided.

    **Primary deployment targets:**

    - Raspberry Pi (all models) and other ARM single-board computers.
    - ARM Cortex-A microprocessors running Linux.
    - Mobile SoCs (Qualcomm, MediaTek) where the TFLite delegate ecosystem
      provides hardware-accelerated inference via NNAPI, GPU, or Hexagon DSP.

    **Full implementation requirements:**

    - ``tflite_runtime`` Python package (``pip install tflite-runtime``).
      Available as a pre-built wheel for ARM (armv7l, aarch64) and x86_64.
      See https://www.tensorflow.org/lite/guide/python for installation.

    **Model format notes:**

    ``.tflite`` models are typically post-training quantized to INT8 for
    efficient edge deployment. INT8 quantization reduces model size by ~4x
    and improves inference speed by 2-4x on ARM CPUs with SIMD support,
    at a small accuracy cost. A full adapter implementation should surface
    the quantization type (FLOAT32, INT8, FLOAT16) in the collected metadata.

    **LiteRT rebranding note:**

    TensorFlow Lite was officially rebranded as **LiteRT** by Google in 2024.
    However, ``tflite_runtime`` remains the install package name for
    backward compatibility, and the interpreter API is unchanged. Calbredge
    uses "tflite" as the canonical runtime name and ``tflite_runtime`` as the
    import for this reason.
    """

    def load_model(self, model_path: str | Path) -> None:
        """Load a TFLite flatbuffer model and create an Interpreter.

        A full implementation would:
        - Import ``tflite_runtime.interpreter`` and instantiate
          ``tflite.Interpreter(model_path=str(model_path))``.
        - Call ``interpreter.allocate_tensors()`` to prepare input/output
          buffers.
        - Retrieve input and output tensor details via
          ``get_input_details()`` / ``get_output_details()``.

        Raises:
            RuntimeError: Always — ``tflite_runtime`` is not installed or this
                          stub has not yet been replaced with a full
                          implementation.
        """
        try:
            import tflite_runtime.interpreter as tflite  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "tflite_runtime is not installed. "
                "Install it with: pip install tflite-runtime  "
                "(pre-built wheels available for ARM and x86_64 — "
                "see https://www.tensorflow.org/lite/guide/python). "
                "TFLite / LiteRT targets ARM edge devices including Raspberry Pi."
            ) from exc

        raise NotImplementedError(
            "TFLiteAdapter.load_model is not yet implemented. "
            "A full implementation should create a tflite.Interpreter, call "
            "allocate_tensors(), and store input/output tensor details."
        )

    def warmup(self, inputs: dict, iterations: int) -> None:
        """Run warmup inference passes to prime the TFLite interpreter.

        A full implementation would populate the input tensor buffers via
        ``interpreter.set_tensor()``, call ``interpreter.invoke()`` *iterations*
        times, and discard all outputs.

        Raises:
            NotImplementedError: Always — stub not yet implemented.
        """
        raise NotImplementedError(
            "TFLiteAdapter.warmup is not yet implemented. "
            "A full implementation should call interpreter.set_tensor() for each "
            "input, invoke the interpreter, and discard outputs for each iteration."
        )

    def run_inference(self, inputs: dict) -> dict:
        """Execute a single TFLite inference pass and return outputs.

        A full implementation would set input tensors via
        ``interpreter.set_tensor()``, call ``interpreter.invoke()``, then
        retrieve outputs with ``interpreter.get_tensor()`` and return them
        as a dict keyed by tensor name.

        Raises:
            NotImplementedError: Always — stub not yet implemented.
        """
        raise NotImplementedError(
            "TFLiteAdapter.run_inference is not yet implemented. "
            "A full implementation should call interpreter.set_tensor(), "
            "interpreter.invoke(), and return outputs via interpreter.get_tensor() "
            "keyed by output tensor name."
        )

    def unload(self) -> None:
        """Release the TFLite interpreter and free associated memory.

        A full implementation would delete the interpreter object, allowing
        the Python GC to release the underlying flatbuffer and tensor memory.

        Raises:
            NotImplementedError: Always — stub not yet implemented.
        """
        raise NotImplementedError(
            "TFLiteAdapter.unload is not yet implemented. "
            "A full implementation should delete the interpreter reference "
            "to allow the GC to release the flatbuffer and tensor allocations."
        )
