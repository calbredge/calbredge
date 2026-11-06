"""Tests for protocol/calbredge_protocol/derive_v2.py: v2 keys derived from recorded data."""

import pytest
from calbredge_protocol.derive_v2 import (
    UNDETERMINED_REASON,
    clocks_pinned,
    cpu_affinity_tier,
    derived_hardware_family,
)

pytestmark = pytest.mark.unit


# The record shapes agent/calbredge/cli.py writes for --cpu-affinity.
@pytest.mark.parametrize(
    ("record", "tier"),
    [
        ({"requested": "big", "applied": [4, 5, 6, 7], "tier": "big", "capacity": 1024}, "big"),
        (
            {"requested": "little", "applied": [0, 1, 2, 3], "tier": "little", "capacity": 414},
            "little",
        ),
        ({"requested": "4,5", "applied": [4, 5], "tier": "custom"}, "custom"),
        ({"requested": "none", "applied": None}, "none"),
        ({"requested": "big", "applied": None, "reason": "homogeneous CPU"}, "homogeneous"),
        ({"requested": "big", "applied": None, "reason": "something new"}, None),
        (None, None),
        ({}, None),
    ],
)
def test_cpu_affinity_tier(record, tier):
    assert cpu_affinity_tier(record) == tier


def _rk3588_frequencies(**scaling_max):
    hardware_max = {"policy0": 1800000, "policy4": 2400000, "policy6": 2400000}
    return {
        policy: {
            "scaling_cur_freq": scaling_max.get(policy, khz),
            "scaling_min_freq": 408000,
            "scaling_max_freq": scaling_max.get(policy, khz),
            "cpuinfo_max_freq": khz,
        }
        for policy, khz in hardware_max.items()
    }


def _rk3588(**overrides):
    """What the collector records after scripts/performance_mode.sh: the GPU at its default."""
    state = {
        "captured": True,
        "source": "rk3588",
        "cpu_governors": {
            "policy0": "performance",
            "policy4": "performance",
            "policy6": "performance",
        },
        "cpu_frequencies": _rk3588_frequencies(),
        "npu_governor": "performance",
        "npu_frequency": {"cur_freq": 1000000000, "max_freq": 1000000000},
        "dmc_governor": "performance",
        "dmc_frequency": {"cur_freq": 2112000000, "max_freq": 2112000000},
        "gpu_governor": "simple_ondemand",
        "gpu_frequency": {"cur_freq": 300000000, "max_freq": 1000000000},
    }
    state.update(overrides)
    return state


def test_rk3588_cpu_npu_and_memory_at_performance_is_pinned():
    assert clocks_pinned(_rk3588()) is True


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "cpu_governors": {
                "policy0": "performance",
                "policy4": "schedutil",
                "policy6": "performance",
            }
        },
        {"npu_governor": "rknpu_ondemand"},
        {"dmc_governor": "dmc_ondemand"},
    ],
)
def test_rk3588_any_scaling_governor_is_unpinned(overrides):
    assert clocks_pinned(_rk3588(**overrides)) is False


@pytest.mark.parametrize("gpu", ["simple_ondemand", "performance", "powersave"])
def test_rk3588_gpu_governor_is_recorded_not_required(gpu):
    """The GPU is idle while the NPU runs the model: CPU, the accelerator, and memory decide."""
    assert clocks_pinned(_rk3588(gpu_governor=gpu)) is True


def test_rk3588_without_any_gpu_record_is_still_pinned():
    state = _rk3588()
    del state["gpu_governor"], state["gpu_frequency"]
    assert clocks_pinned(state) is True


@pytest.mark.parametrize("policy", ["policy0", "policy4", "policy6"])
def test_rk3588_performance_under_a_lowered_scaling_maximum_is_unpinned(policy):
    capped = _rk3588(cpu_frequencies=_rk3588_frequencies(**{policy: 1200000}))
    assert capped["cpu_governors"][policy] == "performance"
    assert clocks_pinned(capped) is False


def _without_frequencies():
    state = _rk3588()
    del state["cpu_frequencies"]
    return state


def _frequencies_with(policy: str, **changes):
    frequencies = _rk3588_frequencies()
    frequencies[policy] = {**frequencies[policy], **changes}
    return frequencies


@pytest.mark.parametrize(
    "state",
    [
        _without_frequencies(),  # a result recorded before frequencies were collected
        _rk3588(cpu_frequencies={}),
        _rk3588(cpu_frequencies="1800000"),
        _rk3588(cpu_frequencies={"policy0": _rk3588_frequencies()["policy0"]}),  # two missing
        _rk3588(cpu_frequencies={**_rk3588_frequencies(), "policy4": None}),
        _rk3588(cpu_frequencies=_frequencies_with("policy4", scaling_max_freq=None)),
        _rk3588(cpu_frequencies=_frequencies_with("policy6", cpuinfo_max_freq="2400000")),
        _rk3588(cpu_frequencies=_frequencies_with("policy6", cpuinfo_max_freq=0)),
        _rk3588(cpu_frequencies=_frequencies_with("policy0", scaling_max_freq=True)),
    ],
)
def test_rk3588_missing_cpu_frequencies_are_unknown_never_pinned(state):
    assert clocks_pinned(state) is None


def test_rk3588_unpinned_governors_are_unpinned_even_without_frequencies():
    state = _without_frequencies()
    state["npu_governor"] = "rknpu_ondemand"
    assert clocks_pinned(state) is False


@pytest.mark.parametrize("key", ["npu_governor", "dmc_governor"])
def test_rk3588_missing_required_governor_is_unknown(key):
    state = _rk3588()
    del state[key]
    assert clocks_pinned(state) is None


@pytest.mark.parametrize(
    "state",
    [
        None,
        {"captured": False, "reason": "failed to read /sys/..."},
        _rk3588(cpu_governors={}),
        # Jetson has no collector yet; its definition lands with it.
        {"captured": True, "source": "jetson"},
        {"captured": True, "source": "something_else"},
    ],
)
def test_clock_state_that_proves_nothing_is_none(state):
    assert clocks_pinned(state) is None


@pytest.mark.parametrize(
    "governors",
    [
        {"policy0": "performance"},
        {"policy0": "performance", "policy1": "performance"},
        {"policy0": "powersave"},
        {"policy0": "performance", "policy1": "schedutil"},
    ],
)
def test_linux_cpufreq_is_never_pinned_or_unpinned(governors):
    """No rule shows a generic Linux CPU's speed held fixed: the governors, turbo and the
    scaling limits don't, since thermal and power limits still lower it. So null, never true,
    and never false either: the governor doesn't decide it."""
    state = {"captured": True, "source": "linux_cpufreq", "cpu_governors": governors}
    assert clocks_pinned(state) is None
    assert "linux_cpufreq" in UNDETERMINED_REASON


def test_derived_hardware_family_is_canonical_and_stable():
    meta = {
        "cpu_model": "AMD Ryzen 9 5950X 16-Core Processor",
        "cpu_cores": 32,
        "ram_gb": 62.7,
        "gpu_model": None,
    }
    assert derived_hardware_family(meta) == "amd_ryzen_9_5950x_16_core_processor_32_62_7_none"
    meta["gpu_model"] = "NVIDIA GeForce RTX 3090"
    assert derived_hardware_family(meta).endswith("nvidia_geforce_rtx_3090")
