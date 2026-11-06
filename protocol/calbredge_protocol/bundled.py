"""The bundled protocols, and which one fits a model's input.

`calbredge init` chooses a project's protocol with this matching, and the ONNX Runtime adapter
names the bundled protocol in a refusal, so the two always agree.
"""

from importlib.resources import files
from pathlib import Path

import yaml

DEFAULT_PROTOCOL = "v1.0.yaml"  # preferred when two bundled protocols share a shape


def bundled_protocols() -> list[tuple[str, list[int]]]:
    """(file name, input_shape) of every bundled protocol, the default first."""
    directory = files("calbredge_protocol.protocols")
    names = sorted(
        (p.name for p in directory.iterdir() if p.name.endswith(".yaml")),
        key=lambda name: (name != DEFAULT_PROTOCOL, name),
    )
    found = []
    for name in names:
        with (directory / name).open(encoding="utf-8") as f:
            shape = yaml.safe_load(f).get("input_shape")
        found.append((name, list(shape)))
    return found


def matching_bundled_protocol(dims: list[int | str]) -> tuple[str, list[int]] | None:
    """The bundled protocol for these input dims, (file name, input_shape), or None.

    Dims that are not ints are dynamic. A dynamic batch axis (the first) is fine: every
    bundled protocol measures batch 1, and runs bind it the same way. Any other dynamic axis
    matches nothing.
    """
    fixed = [d if isinstance(d, int) else None for d in dims]
    if not dims or any(d is None for d in fixed[1:]):
        return None
    wanted = [1, *fixed[1:]]
    for name, shape in bundled_protocols():
        if shape == wanted and fixed[0] in (None, shape[0]):
            return name, shape
    return None


def bundled_protocol_path(name: str) -> Path:
    """Where a bundled protocol file is installed, for --protocol."""
    return Path(str(files("calbredge_protocol.protocols") / name))
