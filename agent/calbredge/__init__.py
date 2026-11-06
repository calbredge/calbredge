"""Calbredge agent — edge AI inference benchmarking CLI."""

import importlib.metadata as _metadata

try:
    # Single source of truth: the installed distribution's version, i.e. the
    # `version` in agent/pyproject.toml. Never hardcode it here: it drifted
    # (0.1.1 packages reported 0.1.0, in --version and in every environment.json).
    __version__ = _metadata.version("calbredge")
except _metadata.PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0+unknown"

from calbredge.fingerprint import compute_hardware_id
from calbredge.hasher import compute_model_hash
from calbredge.uploader import save_result_locally, upload_or_save

__all__ = [
    "__version__",
    "compute_hardware_id",
    "compute_model_hash",
    "save_result_locally",
    "upload_or_save",
]
