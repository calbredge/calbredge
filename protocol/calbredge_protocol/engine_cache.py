"""Content-addressed cache for compiled engines (TensorRT today).

An engine is reused only if it was built from exactly the same inputs: the
cache filename carries ``build_key[:16]``, where ``build_key`` is the SHA-256
of the RFC 8785 canonical JSON of everything that determines the engine
(model content, precision, power mode, resolved input profile, calibration,
compiler version, builder configuration). A different model, profile,
calibration or TensorRT version is a different filename, so a stale engine
can never be picked up; nothing is ever overwritten or deleted.

Each engine has a build record next to it (``<engine>.json``) holding the full
key inputs. On load the key is recomputed from the record and must equal both
the record's own ``build_key`` and the key the caller expects, or the load is
refused (a hand-copied or edited record).

Legacy engines named ``<model>.onnx.trt.<precision>.<power_mode>`` are never
read or written here; tools still using those names are unaffected.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from calbredge_protocol.canonical import canonical_json, sha256_hex

ENGINE_SUFFIX = ".engine"
RECORD_SUFFIX = ".engine.json"
ENGINE_CACHE_FALLBACK_DIR = Path.home() / ".calbredge" / "engine-cache"


class EngineCacheError(RuntimeError):
    """A cached engine's build record does not match what it claims to be."""


def resolve_input_profile(
    input_signature: list[dict[str, Any]], protocol_input_shape: list[int] | None
) -> dict[str, dict[str, list[int]]]:
    """min/opt/max shape per input, by the same rule the TensorRT builder uses.

    Static dimensions come from the model. A dynamic dimension (a symbolic
    name, or a non-positive value) takes the protocol shape's value at that
    position, else 1. min = opt = max: engines are built for one shape.
    """
    protocol_shape = protocol_input_shape or []
    profile: dict[str, dict[str, list[int]]] = {}
    for spec in input_signature:
        shape = [
            d
            if isinstance(d, int) and d > 0
            else (protocol_shape[j] if j < len(protocol_shape) else 1)
            for j, d in enumerate(spec["shape"])
        ]
        profile[spec["name"]] = {"min": list(shape), "opt": list(shape), "max": list(shape)}
    return profile


def build_key_inputs(
    *,
    artifact: str,
    model_hash: str,
    precision: str,
    power_mode: str,
    input_profile: dict[str, dict[str, list[int]]],
    calibration_digest: str | None,
    compiler_version: str,
    builder_config_version: int,
) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "model_hash": model_hash,
        "precision": precision,
        "power_mode": power_mode,
        "input_profile": input_profile,
        "calibration_digest": calibration_digest,
        "compiler_version": compiler_version,
        "builder_config_version": builder_config_version,
    }


def compute_build_key(inputs: dict[str, Any]) -> str:
    return sha256_hex(canonical_json(inputs))


def engine_filename(model_path: Path, precision: str, power_mode: str, build_key: str) -> str:
    return f"{model_path.name}.trt.{precision}.{power_mode}.{build_key[:16]}{ENGINE_SUFFIX}"


def cache_dirs(model_path: Path, fallback_dir: Path | None = None) -> list[Path]:
    """Where engines are looked up and written: next to the model, then the fallback."""
    return [model_path.parent, fallback_dir or ENGINE_CACHE_FALLBACK_DIR]


def find_cached(dirs: list[Path], filename: str) -> tuple[Path, dict[str, Any]] | None:
    """(engine path, build record) from the first dir holding both, else None.

    An engine without its record is treated as absent (e.g. a crash between
    the two renames) and gets rebuilt.
    """
    for directory in dirs:
        engine = directory / filename
        record = directory / (filename[: -len(ENGINE_SUFFIX)] + RECORD_SUFFIX)
        if engine.is_file() and record.is_file():
            try:
                return engine, json.loads(record.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise EngineCacheError(f"unreadable build record {record}: {exc}") from exc
    return None


def verify_record(record: dict[str, Any], expected_key: str) -> None:
    """Refuse a record whose inputs don't hash to its key, or whose key isn't the expected one."""
    inputs = record.get("inputs")
    recorded_key = record.get("build_key")
    if not isinstance(inputs, dict) or not isinstance(recorded_key, str):
        raise EngineCacheError("build record is missing its inputs or build_key")
    if compute_build_key(inputs) != recorded_key:
        raise EngineCacheError(
            "build record's inputs do not hash to its build_key (edited or corrupted record); "
            "refusing to load the engine"
        )
    if recorded_key != expected_key:
        raise EngineCacheError(
            f"cached engine was built for build_key {recorded_key[:16]}, "
            f"this run needs {expected_key[:16]}; refusing to load it"
        )


def _write_atomic(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".partial")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_cached(
    dirs: list[Path],
    filename: str,
    engine_bytes: bytes,
    *,
    build_key: str,
    inputs: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write engine then record atomically into the first writable dir; return the engine path."""
    record = {
        "build_key": build_key,
        "inputs": inputs,
        "built_at": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }
    last_error: OSError | None = None
    for directory in dirs:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            engine = directory / filename
            _write_atomic(engine, engine_bytes)
            _write_atomic(
                directory / (filename[: -len(ENGINE_SUFFIX)] + RECORD_SUFFIX),
                (json.dumps(record, indent=2) + "\n").encode("utf-8"),
            )
            return engine
        except OSError as exc:
            last_error = exc
    raise EngineCacheError(f"could not write the engine to any cache dir: {last_error}")


def onnx_input_signature(model_path: Path) -> list[dict[str, Any]]:
    """Graph inputs (initializers excluded) with symbolic dims kept as names.

    Only used when the caller does not supply the signature (the agent passes
    the cached ONNX contract, so a cache hit never re-parses a large model).
    """
    from calbredge_protocol.graph_variant import load_graph

    graph = load_graph(model_path).graph
    initializers = {t.name for t in graph.initializer}
    signature = []
    for value in graph.input:
        if value.name in initializers:
            continue
        dims = value.type.tensor_type.shape.dim
        signature.append(
            {
                "name": value.name,
                "shape": [
                    d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in dims
                ],
            }
        )
    return signature
