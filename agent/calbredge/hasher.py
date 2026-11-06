import hashlib
from pathlib import Path

_CHUNK_SIZE = 8192


def compute_model_hash(model_path: str | Path) -> str:
    """Compute the SHA256 hash of a model file using chunked reads.

    Reads the file in chunks of 8 KB rather than loading the entire file
    into memory at once. This is important for large model files: a
    production TensorRT engine or ONNX model can easily exceed 500 MB, and
    reading it whole would spike peak RSS accordingly. Chunked hashing keeps
    memory usage flat and predictable regardless of model size.

    Args:
        model_path: Path to the model file. Accepts both ``str`` and
                    :class:`pathlib.Path`.

    Returns:
        A 64-character lowercase hex string (SHA256 digest) uniquely
        identifying the exact bytes of the model file.

    Raises:
        FileNotFoundError: If *model_path* does not exist or is not a file.

    Example::

        model_hash = compute_model_hash("models/mobilenetv2.onnx")
        # → "3a7bd3e2360a3d29aa..."
    """
    path = Path(model_path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK_SIZE):
            digest.update(chunk)

    return digest.hexdigest()
