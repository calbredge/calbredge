"""System metadata collector for Calbredge benchmark results.

Gathers host hardware and software details at benchmark run time. All
collection failures are handled gracefully — a field that cannot be
determined is set to "unknown" rather than crashing the benchmark.
"""

import importlib
import importlib.metadata
import platform
import socket
import subprocess

import psutil
from calbredge_protocol.canonical import runtime_family
from pydantic import BaseModel, ConfigDict


class SystemMetadata(BaseModel):
    """Hardware and software context captured at benchmark run time.

    Attached to every BenchmarkResult so that dashboard comparisons can
    be filtered and grouped by hardware configuration without relying on
    user-supplied labels.
    """

    model_config = ConfigDict(frozen=True)

    hostname: str
    """Network hostname of the device running the benchmark."""

    os_name: str
    """Operating system family name (e.g. "Linux", "Windows", "Darwin")."""

    os_version: str
    """Human-readable OS distribution / release string (e.g. "Ubuntu 22.04")."""

    cpu_model: str
    """CPU model name as reported by the hardware (e.g. "ARM Cortex-A78AE")."""

    cpu_cores: int
    """Number of logical CPU cores available to the process."""

    ram_gb: float
    """Total installed RAM in gigabytes, rounded to one decimal place."""

    gpu_model: str | None
    """GPU model name as reported by nvidia-smi, or None if no NVIDIA GPU is
    detected or nvidia-smi is unavailable."""

    python_version: str
    """Python interpreter version string (e.g. "3.12.3")."""

    runtime_name: str
    """Canonical runtime name passed to the benchmark (e.g. "onnxruntime")."""

    runtime_version: str
    """Version string of the runtime package, or "unknown" if not determinable."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_RUNTIME_PACKAGE_MAP: dict[str, str] = {
    "onnxruntime": "onnxruntime",
    "tensorrt": "tensorrt",
    "tflite": "tflite_runtime",
    "mock": None,  # no real package to inspect
}
# Families whose version is the installed distribution's. RKNN inference runs through
# rknn-toolkit-lite2 (the rknnlite module, which RKNNAdapter imports); the module named rknn
# is rknn-toolkit2, the x86 converter, which is never on the device and never runs a model.
_RUNTIME_DISTRIBUTION_MAP: dict[str, str] = {
    "rknn": "rknn-toolkit-lite2",
}


def _get_runtime_version(runtime_name: str) -> str:
    # By family: a precision-qualified label ("tensorrt-fp16") names the same package.
    family = runtime_family(runtime_name)
    distribution = _RUNTIME_DISTRIBUTION_MAP.get(family)
    if distribution is not None:
        try:
            return importlib.metadata.version(distribution)
        except Exception:
            return "unknown"
    package = _RUNTIME_PACKAGE_MAP.get(family)
    if package is None:
        return "n/a"
    try:
        mod = importlib.import_module(package)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "unknown"


def _get_gpu_model() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],  # noqa: S607 — hardcoded arg list, no shell, no user input
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            name = result.stdout.strip().splitlines()[0].strip()
            return name or None
    except Exception:  # noqa: S110 — best-effort GPU-model probe, absence is the expected path
        pass
    return None


def _get_os_version() -> str:
    try:
        info = platform.freedesktop_os_release()
        return info.get("PRETTY_NAME", platform.version())
    except (AttributeError, OSError):
        pass
    # Fallback: works on macOS and Windows too
    return platform.platform(terse=True)


def _get_cpu_model() -> str:
    # platform.processor() is reliable on most platforms; empty on some ARM Linux
    model = platform.processor()
    if model:
        return model
    # Fallback: read /proc/cpuinfo on Linux
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("Model name") or line.startswith("Hardware"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def collect_metadata(runtime_name: str) -> SystemMetadata:
    """Collect system hardware and software metadata for a benchmark result.

    All fields are collected defensively — any failure falls back to a safe
    default ("unknown" or None) so that a metadata collection hiccup never
    aborts a benchmark run.

    Args:
        runtime_name: Canonical runtime name for the current benchmark run
                      (e.g. "onnxruntime"). Used to determine which package
                      to inspect for the runtime version.

    Returns:
        A frozen :class:`SystemMetadata` instance populated with the current
        host's hardware and software details.
    """
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = "unknown"

    try:
        os_name = platform.system() or "unknown"
    except Exception:
        os_name = "unknown"

    try:
        os_version = _get_os_version()
    except Exception:
        os_version = "unknown"

    try:
        cpu_model = _get_cpu_model()
    except Exception:
        cpu_model = "unknown"

    try:
        cpu_cores = psutil.cpu_count(logical=True) or 0
    except Exception:
        cpu_cores = 0

    try:
        ram_bytes = psutil.virtual_memory().total
        ram_gb = round(ram_bytes / (1024**3), 1)
    except Exception:
        ram_gb = 0.0

    gpu_model = _get_gpu_model()

    try:
        python_version = platform.python_version()
    except Exception:
        python_version = "unknown"

    runtime_version = _get_runtime_version(runtime_name)

    return SystemMetadata(
        hostname=hostname,
        os_name=os_name,
        os_version=os_version,
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        gpu_model=gpu_model,
        python_version=python_version,
        runtime_name=runtime_name,
        runtime_version=runtime_version,
    )
