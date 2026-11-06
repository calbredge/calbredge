"""The RK3588's declared performance state: the pin script, the boot unit, what
doctor says, and what the gate does when the governors are reset during a run.

The collector and the clocks_pinned rule are tested in tests/open/test_rknn_adapter.py,
tests/open/test_clock_state.py and tests/open/test_v2_derivations.py.
"""

import configparser
import os
import subprocess
from pathlib import Path

import pytest
from calbredge import doctor
from calbredge.clock_state import collect_clock_state
from calbredge_protocol.derive_v2 import clocks_pinned

from tests.open.fake_hosts import RK3588_CPU_MAX_KHZ, build_rk3588

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "performance_mode.sh"
UNIT = ROOT / "examples" / "systemd" / "calbredge-rk3588-clocks.service"
INSTALLED_SCRIPT = "/usr/local/sbin/calbredge-performance-mode"


@pytest.fixture(autouse=True)
def _aarch64(monkeypatch):
    monkeypatch.setattr("platform.machine", lambda: "aarch64")


def _pin(host: Path, *, euid: int = 0) -> subprocess.CompletedProcess:
    env = {**os.environ, "CALBREDGE_SYSFS_ROOT": str(host), "CALBREDGE_TEST_EUID": str(euid)}
    return subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=30, check=False
    )


def _tree(host: Path) -> dict[str, str]:
    return {str(p.relative_to(host)): p.read_text() for p in sorted(host.rglob("*")) if p.is_file()}


def test_the_script_pins_a_board_as_it_boots(tmp_path):
    """From what cpufrequtils leaves at boot to a state clocks_pinned accepts."""
    host = build_rk3588(tmp_path, governor="ondemand", scaling_max_khz={"policy4": 1200000})
    assert clocks_pinned(collect_clock_state(host)) is False

    done = _pin(host)

    assert done.returncode == 0, done.stderr
    state = collect_clock_state(host)
    assert state is not None
    assert clocks_pinned(state) is True
    assert set(state["cpu_governors"].values()) == {"performance"}
    assert state["npu_governor"] == state["dmc_governor"] == "performance"
    for policy, hardware_max in RK3588_CPU_MAX_KHZ.items():
        assert state["cpu_frequencies"][policy]["scaling_max_freq"] == hardware_max


def test_the_script_leaves_the_gpu_alone_and_prints_its_governor(tmp_path):
    host = build_rk3588(tmp_path, governor="ondemand")
    done = _pin(host)
    assert done.returncode == 0, done.stderr
    assert (host / "sys/class/devfreq/fb000000.gpu/governor").read_text() == "simple_ondemand\n"
    assert "gpu: simple_ondemand @ 1000000000 Hz (not pinned" in done.stdout
    assert "policy4: performance" in done.stdout
    assert "npu: performance" in done.stdout
    assert "dmc: performance" in done.stdout


@pytest.mark.parametrize(
    "missing",
    [
        "sys/class/devfreq/fdab0000.npu/governor",
        "sys/class/devfreq/dmc/governor",
        "sys/devices/system/cpu/cpufreq/policy6/scaling_governor",
        "sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq",
        "sys/devices/system/cpu/cpufreq/policy4/cpuinfo_max_freq",
    ],
)
def test_a_missing_node_fails_clearly_and_changes_nothing(tmp_path, missing):
    """At boot a node may not exist yet: the unit must fail, not half pin the board."""
    host = build_rk3588(tmp_path, governor="ondemand")
    (host / missing).unlink()
    before = _tree(host)

    done = _pin(host)

    assert done.returncode == 1
    assert done.stderr.startswith("performance_mode: ")
    assert missing in done.stderr
    assert _tree(host) == before


def test_no_cpu_policies_is_an_error(tmp_path):
    (tmp_path / "sys/devices/system/cpu/cpufreq").mkdir(parents=True)
    done = _pin(tmp_path)
    assert done.returncode == 1
    assert "no CPU policies" in done.stderr


def test_the_script_refuses_to_run_without_root(tmp_path):
    host = build_rk3588(tmp_path, governor="ondemand")
    before = _tree(host)
    done = _pin(host, euid=1000)
    assert done.returncode == 1
    assert "run with sudo" in done.stderr
    assert _tree(host) == before


def test_a_missing_gpu_node_does_not_stop_the_script(tmp_path):
    host = build_rk3588(tmp_path, governor="ondemand", gpu_governor=None)
    done = _pin(host)
    assert done.returncode == 0, done.stderr
    assert "gpu: unreadable" in done.stdout
    assert clocks_pinned(collect_clock_state(host)) is True


def test_the_boot_unit_runs_the_script_once_after_cpufrequtils():
    unit = configparser.ConfigParser()
    unit.optionxform = str  # systemd keys are case-sensitive
    unit.read_string(UNIT.read_text())
    assert unit["Service"]["Type"] == "oneshot"
    assert unit["Service"]["RemainAfterExit"] == "yes"
    assert unit["Service"]["ExecStart"] == INSTALLED_SCRIPT
    # cpufrequtils.service sets the governors from /etc/default/cpufrequtils at boot.
    assert unit["Unit"]["After"].split() == ["cpufrequtils.service", "loadcpufreq.service"]
    assert unit["Install"]["WantedBy"] == "multi-user.target"


def test_the_units_install_notes_name_the_script_and_the_path_it_runs():
    text = UNIT.read_text()
    assert f"sudo install -m 755 scripts/performance_mode.sh {INSTALLED_SCRIPT}" in text
    assert f"sudo systemctl restart {UNIT.name}" in text
    assert UNIT.name == doctor.RK3588_CLOCKS_UNIT


def test_doctor_reports_a_pinned_rk3588_with_the_gpu_at_its_default(tmp_path):
    check = doctor._clocks(build_rk3588(tmp_path))
    assert (check.status, check.detail) == ("ok", "yes (rk3588)")


def test_doctor_names_the_script_and_the_unit_for_an_unpinned_rk3588(tmp_path):
    check = doctor._clocks(build_rk3588(tmp_path, governor="ondemand"))
    assert check.status == "fail"
    assert check.detail == "no (rk3588): timings would move"
    assert check.fix == (
        "run: sudo scripts/performance_mode.sh from a Calbredge checkout (the CPU, NPU and DMC "
        "governors to performance; until the next reboot), or install "
        "calbredge-rk3588-clocks.service from examples/systemd (at every boot)"
    )
    assert "GPU" not in check.fix


def test_doctor_names_a_restart_when_the_unit_is_installed(tmp_path):
    """The unit stays active after boot, so only a restart re-pins after a governor reset."""
    host = build_rk3588(tmp_path, governor="ondemand")
    unit = host / "etc" / "systemd" / "system" / "calbredge-rk3588-clocks.service"
    unit.parent.mkdir(parents=True)
    unit.write_text(UNIT.read_text())
    check = doctor._clocks(host)
    assert check.fix is not None
    assert check.fix.startswith("run: sudo systemctl restart calbredge-rk3588-clocks.service")


def test_doctor_fails_a_lowered_scaling_maximum(tmp_path):
    check = doctor._clocks(build_rk3588(tmp_path, scaling_max_khz={"policy0": 1200000}))
    assert check.status == "fail"
