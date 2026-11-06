"""Tests for agent/calbredge/clock_state.py and the clocks_pinned it feeds."""

import pytest
from calbredge.clock_state import collect_clock_state, detect_platform
from calbredge_protocol.derive_v2 import clocks_pinned

from tests.open.fake_hosts import build_jetson_orin, build_rk3588, build_x86

pytestmark = pytest.mark.unit


def test_platform_detection(tmp_path):
    assert detect_platform(build_jetson_orin(tmp_path / "j", pinned=True)) == "jetson"
    assert detect_platform(build_rk3588(tmp_path / "r")) == "rk3588"
    assert detect_platform(build_x86(tmp_path / "x")) == "linux"
    assert detect_platform(tmp_path / "empty") is None


def test_jetson_pinned(tmp_path):
    state = collect_clock_state(build_jetson_orin(tmp_path, pinned=True))
    assert state is not None
    assert state["source"] == "jetson"
    assert set(state["cpu_policies"]) == {"policy0", "policy2"}
    # jetson_clocks pins without changing the governor.
    assert state["cpu_policies"]["policy2"]["scaling_governor"] == "schedutil"
    assert state["gpu"] == {
        "node": "17000000.gpu",
        "min_freq": 1020000000,
        "max_freq": 1020000000,
        "cur_freq": 1020000000,
    }
    assert clocks_pinned(state) is True


def test_jetson_unpinned(tmp_path):
    state = collect_clock_state(build_jetson_orin(tmp_path, pinned=False))
    assert clocks_pinned(state) is False


def test_jetson_cpu_pinned_but_gpu_free_is_unpinned(tmp_path):
    root = build_jetson_orin(tmp_path, pinned=True)
    (root / "sys/class/devfreq/17000000.gpu/min_freq").write_text("306000000\n")
    assert clocks_pinned(collect_clock_state(root)) is False


def test_jetson_non_gpu_devfreq_nodes_are_ignored(tmp_path):
    # vic stays min < max in the fixture; pinned must still be True.
    assert clocks_pinned(collect_clock_state(build_jetson_orin(tmp_path, pinned=True))) is True


def test_jetson_emc_unreadable_unprivileged(tmp_path):
    emc = collect_clock_state(build_jetson_orin(tmp_path, pinned=True))["emc"]  # type: ignore[index]
    assert emc["pinned"] is None
    assert emc["source"] is None
    assert "not readable" in emc["reason"]


@pytest.mark.parametrize(("min_rate", "pinned"), [("3199000000", True), ("204000000", False)])
def test_jetson_emc_from_debugfs_when_readable(tmp_path, min_rate, pinned):
    root = build_jetson_orin(tmp_path, pinned=True)
    emc = root / "sys/kernel/debug/bpmp/debug/clk/emc"
    emc.mkdir(parents=True)
    (emc / "min_rate").write_text(min_rate + "\n")
    (emc / "max_rate").write_text("3199000000\n")
    state = collect_clock_state(root)
    assert state["emc"]["pinned"] is pinned  # type: ignore[index]
    assert state["emc"]["source"] == "debugfs"  # type: ignore[index]
    # EMC never feeds clocks_pinned.
    assert clocks_pinned(state) is True


def test_other_jetson_gpu_node_is_uncaptured(tmp_path):
    state = collect_clock_state(build_jetson_orin(tmp_path, pinned=True, gpu_node="17000000.gv11b"))
    assert state == {"captured": False, "reason": state["reason"]}  # type: ignore[index]
    assert "17000000.gpu" in state["reason"]  # type: ignore[index]
    assert clocks_pinned(state) is None


def test_unreadable_gpu_is_uncaptured(tmp_path):
    root = build_jetson_orin(tmp_path, pinned=True)
    (root / "sys/class/devfreq/17000000.gpu/max_freq").unlink()
    state = collect_clock_state(root)
    assert state["captured"] is False  # type: ignore[index]
    assert clocks_pinned(state) is None


def test_rk3588_uses_the_existing_collector(tmp_path, monkeypatch):
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    state = collect_clock_state(build_rk3588(tmp_path))
    assert state is not None
    assert state["source"] == "rk3588"
    assert clocks_pinned(state) is True


def test_rk3588_gpu_governor_is_recorded_and_does_not_decide(tmp_path, monkeypatch):
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    state = collect_clock_state(build_rk3588(tmp_path / "default"))
    assert state is not None
    assert state["gpu_governor"] == "simple_ondemand"
    assert clocks_pinned(state) is True
    without = collect_clock_state(build_rk3588(tmp_path / "none", gpu_governor=None))
    assert without is not None
    assert "gpu_governor" not in without
    assert clocks_pinned(without) is True


def test_rk3588_after_a_governor_reset_is_unpinned(tmp_path, monkeypatch):
    """What cpufrequtils leaves at boot, or after its service restarts: everything on demand."""
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    state = collect_clock_state(build_rk3588(tmp_path, governor="ondemand"))
    assert state is not None
    assert state["npu_governor"] == "rknpu_ondemand"
    assert clocks_pinned(state) is False


def test_rk3588_lowered_scaling_maximum_is_unpinned(tmp_path, monkeypatch):
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    host = build_rk3588(tmp_path, scaling_max_khz={"policy4": 1200000})
    state = collect_clock_state(host)
    assert state is not None
    assert state["cpu_governors"]["policy4"] == "performance"
    assert state["cpu_frequencies"]["policy4"]["scaling_max_freq"] == 1200000
    assert clocks_pinned(state) is False


@pytest.mark.parametrize("governor", ["performance", "powersave"])
def test_x86_linux_cpufreq(tmp_path, governor):
    """The governors are recorded; clocks_pinned is never true or false on x86, at any governor."""
    state = collect_clock_state(build_x86(tmp_path, governor=governor))
    assert state == {
        "captured": True,
        "source": "linux_cpufreq",
        "cpu_governors": {"policy0": governor, "policy1": governor},
    }
    assert clocks_pinned(state) is None


def test_no_cpufreq_at_all(tmp_path):
    assert collect_clock_state(tmp_path) is None
