from pathlib import Path

import yaml
from pydantic import ValidationError

from .models import BenchmarkProtocol


def load_protocol(path: str | Path) -> BenchmarkProtocol:
    """Load and validate a benchmark protocol from a YAML file.

    Reads the YAML file at *path*, parses its contents, and validates them
    against the BenchmarkProtocol schema. All defaults defined on the model
    are applied automatically for any omitted fields.

    Args:
        path: Filesystem path to the protocol YAML file. Accepts both
              ``str`` and :class:`pathlib.Path`.

    Returns:
        A validated, frozen :class:`BenchmarkProtocol` instance.

    Raises:
        FileNotFoundError: If *path* does not exist or is not a file.
        ValueError: If the file contains invalid YAML syntax or its contents
                    fail BenchmarkProtocol validation. The underlying error
                    is included in the message for debugging.

    Example::

        from calbredge_protocol import load_protocol

        proto = load_protocol("benchmarks/mobilenet_v2.yaml")
        print(proto.measurement_iterations)  # 500
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Protocol file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Protocol path is not a file: {path}")

    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed YAML in protocol file '{path}': {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"Protocol file '{path}' must contain a YAML mapping at the top level, "
            f"got {type(raw).__name__}"
        )

    try:
        return BenchmarkProtocol(**raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid protocol configuration in '{path}':\n{exc}") from exc
