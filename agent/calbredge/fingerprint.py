import hashlib
from pathlib import Path

from calbredge.collector import SystemMetadata


def _stable_cpu_count() -> int:
    """Return the total number of CPUs in the system, including offline ones.

    Reads /sys/devices/system/cpu/possible which lists every CPU the kernel
    knows about regardless of online/offline state. This is stable across
    Jetson power mode switches that hotplug cores on and off.

    Falls back to psutil.cpu_count(logical=True) on non-Linux systems or if
    the sysfs file is unavailable.
    """
    try:
        text = Path("/sys/devices/system/cpu/possible").read_text().strip()
        # Format examples: "0-7", "0-3,6-9", "0"
        total = 0
        for part in text.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                total += int(hi) - int(lo) + 1
            else:
                total += 1
        if total > 0:
            return total
    except (OSError, ValueError):
        pass
    import psutil

    return psutil.cpu_count(logical=True) or 0


def compute_hardware_id(metadata: SystemMetadata) -> str:
    """Compute a stable SHA256 hardware identifier from system metadata.

    Produces a deterministic hex digest that uniquely identifies a physical
    device. The same device always produces the same ``hardware_id`` regardless
    of reboots, power mode changes, or runtime upgrades, allowing the dashboard
    to group results by device across multiple benchmark runs.

    **Fields included in the hash** (in fixed order):

    - ``hostname`` — distinguishes multiple devices on the same network.
    - ``cpu_model`` — CPU family / model string.
    - ``cpu_cores`` — *total* hardware CPU count including offline cores.
      Read from ``/sys/devices/system/cpu/possible`` so that Jetson power
      mode switches (which hotplug cores on/off) do not alter the hash.
    - ``ram_gb`` — total installed RAM.
    - ``gpu_model`` — GPU name, or the string "None" if no GPU is present.

    **Fields deliberately excluded:**

    - ``os_name``, ``os_version`` — OS identity, not hardware identity.
      An OS reinstall or kernel update should not create a new device record,
      and the kernel version embedded in some ``os_version`` strings changes
      independently of the hardware.
    - ``runtime_name``, ``runtime_version`` — the same hardware running
      ONNX Runtime vs TensorRT must produce the same ``hardware_id`` so that
      cross-runtime comparisons are attributed to a single device record.
    - ``python_version`` — a Python upgrade must not create a new device record.

    **Stability guarantees:**

    - Stable across reboots and power mode changes (including Jetson nvpmodel).
    - Stable across OS upgrades and runtime/Python version changes.
    - Changes if the CPU, RAM, or GPU is physically replaced.
    - Changes if the hostname is renamed.

    Args:
        metadata: A :class:`~calbredge.collector.SystemMetadata` instance as
                  returned by :func:`~calbredge.collector.collect_metadata`.

    Returns:
        A 64-character lowercase hex string (SHA256 digest).
    """
    fingerprint_str = "|".join(
        [
            metadata.hostname,
            metadata.cpu_model,
            str(_stable_cpu_count()),
            str(metadata.ram_gb),
            str(metadata.gpu_model),  # "None" if no GPU
        ]
    )
    return hashlib.sha256(fingerprint_str.encode("utf-8")).hexdigest()
