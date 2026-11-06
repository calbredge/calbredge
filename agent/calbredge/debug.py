"""Debug info collection and formatting for --debug flag output.

Gathers environment details that an engineer can paste directly into a
GitHub issue to help diagnose benchmark failures or runtime mismatches.
All collection is defensive — any individual failure falls back to a
safe placeholder rather than crashing.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Individual collectors
# ---------------------------------------------------------------------------


def _ort_info() -> tuple[str, bool, list[str]]:
    """Return (version, cuda_available, providers).

    Returns ("not installed", False, []) when onnxruntime is absent.
    """
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        return ort.__version__, "CUDAExecutionProvider" in providers, providers
    except ImportError:
        return "not installed", False, []
    except Exception:
        return "unknown", False, []


def _jetpack_version() -> str:
    """Attempt to detect JetPack version.

    Checks /etc/nv_tegra_release first (present on older JetPack), then
    falls back to ``dpkg -l nvidia-jetpack``.  Returns "not detected" on
    non-Jetson hardware or any failure.
    """
    try:
        tegra = Path("/etc/nv_tegra_release")
        if tegra.exists():
            first_line = tegra.read_text().splitlines()[0]
            # Format: "# R35 (release), REVISION: 4.1, ..."
            return first_line.strip().lstrip("# ")
    except Exception:  # noqa: S110 — non-Jetson-hardware probe, absence is the expected path
        pass

    try:
        result = subprocess.run(
            ["dpkg", "-l", "nvidia-jetpack"],  # noqa: S607 — hardcoded arg list, no shell, no user input
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            if line.startswith("ii"):
                parts = line.split()
                if len(parts) >= 3:
                    return parts[2]  # version column
    except Exception:  # noqa: S110 — dpkg-based JetPack probe, absence is the expected path
        pass

    return "not detected"


def _tegrastats_status() -> str:
    """Return "available" if tegrastats is found in PATH, else "not available"."""
    return "available" if shutil.which("tegrastats") else "not available"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def format_run_debug_block(
    *,
    calbredge_version: str,
    model_path: Path | None,
    model_hash: str | None,
    runtime: str | None,
    protocol_version: str | None,
    hardware_id: str | None,
    hostname: str | None,
    os_version: str | None,
    cpu_model: str | None,
    cpu_cores: int | None,
    ram_gb: float | None,
    gpu_model: str | None,
    error_message: str | None = None,
) -> str:
    """Build the full debug block string for the ``run`` command.

    All fields are optional — fields that could not be collected are shown
    as "unavailable".  The caller decides which fields it was able to
    gather before or after a failure.
    """
    ort_version, cuda_available, providers = _ort_info()

    def _v(val: object, suffix: str = "") -> str:
        if val is None:
            return "unavailable"
        return f"{val}{suffix}"

    hw_id_display = (hardware_id[:16] + "...") if hardware_id else "unavailable"
    hash_display = (model_hash[:12] + "...") if model_hash else "unavailable"
    model_display = model_path.name if model_path else "unavailable"
    gpu_display = gpu_model if gpu_model else "none detected"
    cpu_display = f"{cpu_model} ({cpu_cores} cores)" if cpu_model and cpu_cores else _v(cpu_model)
    ram_display = f"{ram_gb} GB" if ram_gb is not None else "unavailable"

    lines = [
        "--- Calbredge Debug Info ---",
        f"calbredge version:     {calbredge_version}",
        f"python version:        {platform.python_version()}",
        f"platform:              {platform.platform(terse=True)}",
        f"onnxruntime version:   {ort_version}",
        f"cuda available:        {'true' if cuda_available else 'false'}",
        f"available providers:   {providers}",
        "",
        f"hardware_id:           {hw_id_display}",
        f"hostname:              {_v(hostname)}",
        f"os:                    {_v(os_version)}",
        f"cpu:                   {cpu_display}",
        f"ram:                   {ram_display}",
        f"gpu:                   {gpu_display}",
        "",
        f"model:                 {model_display}",
        f"model_hash:            {hash_display}",
        f"runtime:               {_v(runtime)}",
        f"protocol:              {_v(protocol_version)}",
        "",
        f"jetpack_version:       {_jetpack_version()}",
        f"tegrastats:            {_tegrastats_status()}",
    ]

    if error_message:
        lines += ["", f"error:                 {error_message}"]

    lines.append("----------------------------")
    return "\n".join(lines)


def format_convert_debug_block(
    *,
    input_path: Path | None,
    error_message: str | None = None,
) -> str:
    """Build the debug block string for the ``convert`` command."""

    def _pkg_version(name: str) -> str:
        try:
            import importlib.metadata

            return importlib.metadata.version(name)
        except Exception:
            return "not installed"

    lines = [
        "--- Calbredge Convert Debug Info ---",
        f"python version:    {platform.python_version()}",
        f"tf2onnx version:   {_pkg_version('tf2onnx')}",
        f"onnx version:      {_pkg_version('onnx')}",
        f"input file:        {input_path or 'unavailable'}",
    ]

    if error_message:
        lines.append(f"error:             {error_message}")

    lines.append("------------------------------------")
    return "\n".join(lines)
