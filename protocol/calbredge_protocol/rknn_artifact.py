"""The build record that travels with a converted ``.rknn`` model.

A TensorRT engine is built on the device that runs it, so that device can recompute the
engine's build key and find the engine by it (protocol/calbredge_protocol/engine_cache.py).
A ``.rknn`` is converted on an x86 host and copied to the board, which knows neither the
toolkit version nor the calibration that produced it. So the record is written at conversion,
beside the artifact (``<artifact>.json``), and holds:

- ``inputs``: everything that determines the artifact (source model hash, precision, target
  platform, the model's inputs as converted, normalization, graph edits, calibration digest,
  toolkit version, converter configuration version);
- ``build_key``: the SHA-256 of the RFC 8785 canonical JSON of ``inputs``, computed as
  TensorRT's is;
- ``artifact_sha256``: the SHA-256 of the ``.rknn`` file itself, which is what binds the
  record to one file. The artifact keeps its name and may be overwritten by a reconversion: a
  record that does not describe the file beside it is refused.

``verify_record`` checks what the board can see: the inputs hash to the key, the artifact is
the one recorded, and it was built from the model, precision and target this run asks for.
"""

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from calbredge_protocol.canonical import canonical_json, sha256_hex
from calbredge_protocol.engine_cache import compute_build_key

ARTIFACT = "rknn_model"
RECORD_VERSION = 1
RECORD_SUFFIX = ".json"


class RknnArtifactError(RuntimeError):
    """A ``.rknn`` build record is unreadable or does not describe what it sits beside."""


def record_path(rknn_path: Path) -> Path:
    return Path(str(rknn_path) + RECORD_SUFFIX)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibration_digest(image_paths: list[Path], quantized_dtype: str) -> str:
    """What an INT8 build was calibrated with: each image's content, in order, and the dtype."""
    return sha256_hex(
        canonical_json(
            {
                "quantized_dtype": quantized_dtype,
                "images": [file_sha256(p) for p in image_paths],
            }
        )
    )


def key_inputs(
    *,
    model_hash: str,
    precision: str,
    target_platform: str,
    model_inputs: list[dict[str, Any]],
    normalization: dict[str, Any],
    feed_dtype: str,
    graph_edits: list[str],
    calibration_digest: str | None,
    compiler_version: str,
    builder_config_version: int,
) -> dict[str, Any]:
    """Everything that determines a ``.rknn``. ``model_inputs`` is one dict per model input:
    its name, the shape and dtype the source model declares, its layout where it could be
    read, and the shape it was converted at."""
    return {
        "artifact": ARTIFACT,
        "model_hash": model_hash,
        "precision": precision,
        "target_platform": target_platform,
        "model_inputs": model_inputs,
        "normalization": normalization,
        "feed_dtype": feed_dtype,
        "graph_edits": graph_edits,
        "calibration_digest": calibration_digest,
        "compiler_version": compiler_version,
        "builder_config_version": builder_config_version,
    }


def _write_atomic(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".partial")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_record(
    rknn_path: Path, inputs: dict[str, Any], extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Write the record beside ``rknn_path`` (which must exist) and return it."""
    record = {
        "record_version": RECORD_VERSION,
        "build_key": compute_build_key(inputs),
        "inputs": inputs,
        "artifact_sha256": file_sha256(rknn_path),
        "built_at": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }
    _write_atomic(record_path(rknn_path), (json.dumps(record, indent=2) + "\n").encode("utf-8"))
    return record


def load_record(rknn_path: Path) -> dict[str, Any] | None:
    """The record beside ``rknn_path``, or None if there is none."""
    path = record_path(rknn_path)
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RknnArtifactError(f"unreadable build record {path}: {exc}") from exc
    if not isinstance(record, dict):
        raise RknnArtifactError(f"build record {path} is not a JSON object")
    return record


def verify_record(
    record: dict[str, Any],
    *,
    model_hash: str,
    artifact_sha256: str,
    precision: str,
    target_platform: str,
) -> None:
    """Refuse a record that was edited, or that does not describe this artifact and this run."""
    inputs = record.get("inputs")
    recorded_key = record.get("build_key")
    if not isinstance(inputs, dict) or not isinstance(recorded_key, str):
        raise RknnArtifactError("build record is missing its inputs or build_key")
    if compute_build_key(inputs) != recorded_key:
        raise RknnArtifactError(
            "build record's inputs do not hash to its build_key (edited or corrupted record)"
        )
    if inputs.get("artifact") != ARTIFACT:
        raise RknnArtifactError(f"build record is for {inputs.get('artifact')!r}, not a .rknn")
    if record.get("artifact_sha256") != artifact_sha256:
        raise RknnArtifactError(
            "the .rknn file is not the one its build record describes (the artifact or the "
            "record was replaced without the other); convert the model again"
        )
    if inputs.get("model_hash") != model_hash:
        raise RknnArtifactError(
            "the .rknn was built from a different model than the .onnx beside it (the model "
            "was re-exported after conversion); convert the model again"
        )
    for name, wanted in (("precision", precision), ("target_platform", target_platform)):
        if inputs.get(name) != wanted:
            raise RknnArtifactError(
                f"the .rknn was built for {name} {inputs.get(name)!r}, this run needs {wanted!r}"
            )
