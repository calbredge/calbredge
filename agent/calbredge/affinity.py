"""CPU affinity detection and application for Calbredge benchmark runs.

Detects ARM big.LITTLE clusters via sysfs core-capacity values and pins the
benchmark process to a requested cluster. Confirmed on-device (RK3588):
runtime threads spawned by RKNNLite's ``init_runtime()`` inherit the calling
process's affinity mask when the pin is applied before ``load_model()`` — so
callers must apply the pin before loading the model, not just before the
measurement loop, or runtime-internal threads will miss it.
"""

import os
import re
from pathlib import Path

_CPU_DIR_RE = re.compile(r"cpu\d+$")


def detect_clusters(sysfs_root: str | Path = "/") -> dict | None:
    r"""Detect big.LITTLE CPU clusters via cpuN/cpu_capacity under sysfs_root.

    Reads ``cpu_capacity`` for every ``cpuN`` directory under
    ``<sysfs_root>/sys/devices/system/cpu/``, matched by the ``cpu\d+`` regex
    rather than a ``cpu*`` glob — a glob would also match sibling directories
    like ``cpuidle`` and ``cpufreq`` that aren't per-core.

    Returns ``{"big": [cpu_ids], "little": [cpu_ids], "capacities": {cpu_id:
    capacity}}``, where "big" is every core at the maximum observed capacity
    (unioning multiple clusters that share the top capacity value, e.g.
    RK3588's two A76 policy clusters) and "little" is everything below it.

    Returns ``None`` — fail soft — when no ``cpu_capacity`` file is readable
    (x86, or a kernel without ``arch_topology``/EAS support) or when every
    readable capacity is equal (homogeneous CPU, nothing to pin against). A
    wrong pin is worse than no pin, so ambiguity always resolves to ``None``
    rather than guessing.

    Args:
        sysfs_root: Root to read the ``sys/devices/system/cpu`` tree under.
                    Defaults to ``/`` (the real filesystem); tests redirect
                    this to a tmp_path with the same relative layout.
    """
    cpu_root = Path(sysfs_root) / "sys" / "devices" / "system" / "cpu"
    try:
        entries = list(cpu_root.iterdir())
    except OSError:
        return None

    capacities: dict[int, int] = {}
    for entry in entries:
        if not _CPU_DIR_RE.fullmatch(entry.name):
            continue
        try:
            capacity = int((entry / "cpu_capacity").read_text().strip())
        except (OSError, ValueError):
            continue
        capacities[int(entry.name[len("cpu") :])] = capacity

    if not capacities or len(set(capacities.values())) < 2:
        return None

    max_capacity = max(capacities.values())
    big = sorted(cpu_id for cpu_id, cap in capacities.items() if cap == max_capacity)
    little = sorted(cpu_id for cpu_id, cap in capacities.items() if cap != max_capacity)

    return {"big": big, "little": little, "capacities": capacities}


def resolve_affinity(spec: str, clusters: dict | None) -> set[int] | None:
    """Resolve a --cpu-affinity spec string to a concrete core set.

    ``spec`` is one of:
      - ``"none"``: explicit no-pin. Always returns ``None``.
      - ``"big"`` / ``"little"``: the corresponding cluster from
        ``clusters``, or ``None`` (fail soft) if ``clusters`` is ``None``
        (homogeneous or undetectable CPU).
      - a comma-separated list of core ids (e.g. ``"0,2,4"``): parsed
        directly, independent of ``clusters``. Raises ``ValueError`` if any
        entry is not an integer.

    Args:
        spec: The raw --cpu-affinity value.
        clusters: The dict returned by :func:`detect_clusters`, or ``None``.

    Returns:
        The concrete set of CPU ids to pin to, or ``None`` to mean "don't
        pin".
    """
    spec = spec.strip()
    if spec == "none":
        return None
    if spec in ("big", "little"):
        if clusters is None:
            return None
        return set(clusters[spec])

    try:
        return {int(part.strip()) for part in spec.split(",")}
    except ValueError as exc:
        raise ValueError(
            f"Invalid --cpu-affinity value {spec!r}: expected 'none', 'big', 'little', "
            "or a comma-separated list of core ids (e.g. '0,2,4')."
        ) from exc


def apply_affinity(core_set: set[int] | None) -> set[int] | None:
    """Pin the current process to core_set via os.sched_setaffinity.

    Returns the set actually applied, or ``None`` if no pin was applied —
    either because ``core_set`` is ``None`` (caller resolved to "don't
    pin"), or because ``os.sched_setaffinity`` doesn't exist on this
    platform (non-Linux; fails soft rather than raising).

    Must be called before ``adapter.load_model()`` to cover runtime-internal
    threads spawned during model load (e.g. RKNNLite's ``init_runtime()``)
    — Linux threads inherit the creating thread's affinity mask at creation
    time, so a pin applied afterward would miss any thread already spawned.
    """
    if core_set is None:
        return None
    if not hasattr(os, "sched_setaffinity"):
        return None
    os.sched_setaffinity(0, core_set)
    return core_set
