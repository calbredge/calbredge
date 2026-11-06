"""Pure derivations of v2 keys from data a run already recorded.

Nothing here reads the host: every function takes values that a collector
wrote into the result at run time and derives a compatibility key from them
(docs/RESULT_SCHEMA.md §3.4.1, §3.4.2, §4). Backfilled or hand-written
values must never be fed in; they are annotation, not evidence.
"""

from typing import Any

from calbredge_protocol.canonical import canonical_identifier

# Clock sources whose state is recorded but for which no rule shows the speed held fixed:
# clocks_pinned is None there whatever the state says, and this is why.
UNDETERMINED_REASON = {
    "linux_cpufreq": "no rule shows this CPU's speed is fixed",
}


def cpu_affinity_tier(cpu_affinity: dict[str, Any] | None) -> str | None:
    """Map the agent's recorded ``cpu_affinity`` dict to a tier.

    The five shapes are the ones agent/calbredge/cli.py writes: a pinned
    record carries ``tier`` (big/little/custom); an explicit ``--cpu-affinity
    none`` records ``requested: "none"``; a big.LITTLE request on a
    homogeneous CPU records ``reason: "homogeneous CPU"``. Anything else is
    unrecognized and yields None rather than a guess.
    """
    if not cpu_affinity:
        return None
    tier = cpu_affinity.get("tier")
    if tier in ("big", "little", "custom"):
        return tier
    if cpu_affinity.get("applied") is None:
        if cpu_affinity.get("requested") == "none":
            return "none"
        if cpu_affinity.get("reason") == "homogeneous CPU":
            return "homogeneous"
    return None


def _all_performance(governors: Any) -> bool | None:
    if not isinstance(governors, dict) or not governors:
        return None
    return all(g == "performance" for g in governors.values())


def clocks_pinned(clock_state: dict[str, Any] | None) -> bool | None:
    """Whether the run's clocks were held fixed, per RESULT_SCHEMA.md §3.4.1.

    Never assumed: missing or uncaptured clock state is None, not False and
    not True.

    - ``rk3588``: every CPU policy governor, the NPU governor and the DMC
      (memory) governor are ``performance``, and every CPU policy's
      scaling_max_freq equals its cpuinfo_max_freq (a ``performance`` governor
      under a lowered maximum is not pinned). The CPU, the accelerator that
      runs the model, and memory: the GPU is idle during RKNN inference, so
      its governor is recorded and not required. If the governors say pinned
      and the CPU frequencies were not recorded, the answer is None, never
      True.
    - ``linux_cpufreq``: always None (``UNDETERMINED_REASON``). Neither the
      governors nor turbo nor the scaling limits show the speed held fixed:
      thermal and power limits still lower it below what cpufreq asks for.
    - ``jetson``: every CPU policy has scaling_min_freq == scaling_max_freq
      AND the GPU devfreq has min_freq == max_freq. Not governor-based: the
      governor stays ``schedutil`` when jetson_clocks pins. Other devfreq
      nodes and EMC do not count (EMC is recorded separately in
      ``clock_state["emc"]``).
    """
    if not clock_state or clock_state.get("captured") is not True:
        return None
    source = clock_state.get("source")
    if source == "rk3588":
        cpu = _all_performance(clock_state.get("cpu_governors"))
        if cpu is None:
            return None
        others = [clock_state.get(k) for k in ("npu_governor", "dmc_governor")]
        if any(g is None for g in others):
            return None
        if not (cpu and all(g == "performance" for g in others)):
            return False
        return _rk3588_cpu_at_hardware_max(clock_state)
    if source in UNDETERMINED_REASON:
        return None
    if source == "jetson":
        return _jetson_pinned(clock_state)
    return None


def _rk3588_cpu_at_hardware_max(clock_state: dict[str, Any]) -> bool | None:
    """Whether every CPU policy's scaling maximum is its hardware maximum; None if unrecorded."""
    governors = clock_state.get("cpu_governors")
    frequencies = clock_state.get("cpu_frequencies")
    if not isinstance(governors, dict) or not isinstance(frequencies, dict):
        return None
    if set(frequencies) != set(governors):
        return None
    at_max = True
    for policy in frequencies.values():
        if not isinstance(policy, dict):
            return None
        scaling_max, hardware_max = policy.get("scaling_max_freq"), policy.get("cpuinfo_max_freq")
        if not _is_frequency(scaling_max) or not _is_frequency(hardware_max):
            return None
        at_max = at_max and scaling_max == hardware_max
    return at_max


def _is_frequency(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _jetson_pinned(clock_state: dict[str, Any]) -> bool | None:
    policies = clock_state.get("cpu_policies")
    gpu = clock_state.get("gpu")
    if not isinstance(policies, dict) or not policies or not isinstance(gpu, dict):
        return None
    try:
        cpu = all(p["scaling_min_freq"] == p["scaling_max_freq"] for p in policies.values())
        gpu_pinned = gpu["min_freq"] == gpu["max_freq"]
    except (KeyError, TypeError):
        return None
    return cpu and gpu_pinned


def derived_hardware_family(system_metadata: dict[str, Any]) -> str:
    """Fallback ``hardware_family`` for hosts without a device tree (§4).

    ``gpu_model`` of None contributes the string "None", matching how
    ``compute_hardware_id`` stringifies it.
    """
    parts = [
        str(system_metadata["cpu_model"]),
        str(system_metadata["cpu_cores"]),
        str(system_metadata["ram_gb"]),
        str(system_metadata["gpu_model"]),
    ]
    return canonical_identifier("|".join(parts))
