"""Host facts for v2 results that v1's SystemMetadata does not collect.

docs/RESULT_SCHEMA.md §3.3 (git_commit, git_dirty, runtime_version,
driver_version), §3.4 (power_mode) and §4 (hardware_family). Every function
returns None, never a placeholder, for anything it cannot determine.

v1's SystemMetadata (agent/calbredge/collector.py) is deliberately left
unchanged: it is the upload payload, and v2 adds device_tree_model only to its
own system_metadata dict.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from calbredge_protocol.adapters import _get_nvpmodel_mode
from calbredge_protocol.canonical import canonical_identifier, is_git_commit
from calbredge_protocol.derive_v2 import derived_hardware_family

from calbredge.clock_state import detect_platform
from calbredge.collector import _get_runtime_version

_L4T_RELEASE = re.compile(r"#\s*R(\d+)\s*\(release\),\s*REVISION:\s*([0-9.]+)")
_PLACEHOLDER_VERSIONS = {"", "unknown", "n/a"}


def device_tree_model(root: Path = Path("/")) -> str | None:
    """``/proc/device-tree/model`` without its trailing NUL, or None."""
    try:
        raw = (root / "proc/device-tree/model").read_bytes()
    except OSError:
        return None
    model = raw.replace(b"\x00", b"").decode("utf-8", "replace").strip()
    return model or None


def hardware_family(system_metadata: dict[str, Any], root: Path = Path("/")) -> tuple[str, str]:
    """(hardware_family, hardware_family_source), RESULT_SCHEMA.md §4.

    The device tree always wins when present; it names the whole board,
    carrier included.
    """
    model = device_tree_model(root)
    if model is not None:
        return canonical_identifier(model), "device_tree"
    return derived_hardware_family(system_metadata), "derived"


def _git(cwd: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(  # noqa: S603 — fixed git argv, cwd only; no shell, no user-controlled command
            ["git", "-C", str(cwd), *args],  # noqa: S607 — git resolved from PATH by design
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def git_state(cwd: Path) -> tuple[str | None, bool | None]:
    """(git_commit, git_dirty) for the working directory, (None, None) outside a repo.

    Dirty means a modified or staged **tracked** file. Untracked files are
    ignored on purpose, so build output and scratch files don't mark every
    run dirty.
    """
    head = _git(cwd, "rev-parse", "HEAD")
    commit = head.strip() if head else None
    if commit is None or not is_git_commit(commit):
        return None, None
    status = _git(cwd, "status", "--porcelain", "--untracked-files=no")
    if status is None:
        return None, None
    return commit, bool(status.strip())


def l4t_version(root: Path = Path("/")) -> str | None:
    """``# R36 (release), REVISION: 4.7, ...`` -> ``36.4.7``."""
    try:
        first = (root / "etc/nv_tegra_release").read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return None
    match = _L4T_RELEASE.match(first)
    return f"{match.group(1)}.{match.group(2)}" if match else None


def cuda_version(root: Path = Path("/")) -> str | None:
    try:
        data = json.loads((root / "usr/local/cuda/version.json").read_text(encoding="utf-8"))
        version = data["cuda"]["version"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return version if isinstance(version, str) and version else None


def nvidia_driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],  # noqa: S607 — hardcoded argv, no shell
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.strip().splitlines()[0].strip() or None


def driver_version(
    root: Path = Path("/"), *, query_nvidia_smi: bool = True
) -> dict[str, str] | None:
    """Platform-specific driver/toolkit versions: ``l4t``, ``cuda``, ``nvidia_driver``."""
    versions = {
        "l4t": l4t_version(root),
        "cuda": cuda_version(root),
        "nvidia_driver": nvidia_driver_version() if query_nvidia_smi else None,
    }
    found = {k: v for k, v in versions.items() if v is not None}
    return found or None


def runtime_version(runtime_name: str) -> str | None:
    """The runtime package's version, or None; never "unknown"/"n/a"."""
    version = _get_runtime_version(runtime_name)
    return None if version.strip().lower() in _PLACEHOLDER_VERSIONS else version


def power_mode(root: Path = Path("/")) -> str | None:
    """Jetson: the nvpmodel mode (None if unreadable). Elsewhere: ``not_applicable``."""
    if detect_platform(root) == "jetson":
        return _get_nvpmodel_mode()
    return "not_applicable"


def git_toplevel(cwd: Path) -> Path | None:
    """The root of the git repository containing ``cwd``, or None."""
    out = _git(cwd, "rev-parse", "--show-toplevel")
    return Path(out.strip()) if out and out.strip() else None


def git_tracked(path: Path) -> bool | None:
    """Whether git tracks this file; None outside a repository or when git can't say."""
    if git_toplevel(path.parent) is None:
        return None
    out = _git(path.parent, "ls-files", "--", path.name)
    return None if out is None else bool(out.strip())


def git_user_name(cwd: Path) -> str | None:
    """``git config user.name`` as seen from ``cwd``, or None if unset."""
    out = _git(cwd, "config", "user.name")
    if out is None or not out.strip():
        return None
    return out.strip()


def classify_host(
    os_name: str, device_tree_model: str | None, clock_source: str | None, machine: str
) -> tuple[str, str, str]:
    """(os, device class, accelerator) for the support-matrix lookup."""
    if (device_tree_model or "").startswith("NVIDIA Jetson"):
        device_class = "jetson"
    elif clock_source == "rk3588":
        device_class = "rk3588"
    else:
        device_class = machine.lower() or "unknown"
    return os_name.lower(), device_class, "none"
