"""Collect clock_state for v2 results (docs/RESULT_SCHEMA.md §3.4, §3.4.1).

Every read here works unprivileged: CI cannot sudo, and neither can most
users' release pipelines. Anything that cannot be read is recorded as such;
nothing is assumed or defaulted. ``clocks_pinned`` is derived from this
output by protocol/calbredge_protocol/derive_v2.py, never here.

``root`` is the filesystem root to read under (tests point it at a fake
sysfs/procfs tree built in tmp_path).
"""

from pathlib import Path
from typing import Any

# Orin's GPU devfreq node. Other Jetson generations name it differently; until
# one is verified they record no GPU state and clocks_pinned stays null.
JETSON_GPU_DEVFREQ_NODES = ("17000000.gpu",)

_CPUFREQ = Path("sys/devices/system/cpu/cpufreq")
_DEVFREQ = Path("sys/class/devfreq")
_EMC_DEBUGFS = Path("sys/kernel/debug/bpmp/debug/clk/emc")


def device_tree_compatible(root: Path = Path("/")) -> list[str]:
    try:
        raw = (root / "proc/device-tree/compatible").read_bytes()
    except OSError:
        return []
    return [c.decode("utf-8", "replace") for c in raw.split(b"\x00") if c]


def detect_platform(root: Path = Path("/")) -> str | None:
    """``jetson``, ``rk3588``, ``linux`` (plain cpufreq) or None."""
    compatible = device_tree_compatible(root)
    if any(c.startswith("nvidia,tegra") for c in compatible):
        return "jetson"
    if "rockchip,rk3588" in compatible:
        return "rk3588"
    if (root / _CPUFREQ).is_dir():
        return "linux"
    return None


def collect_clock_state(root: Path = Path("/")) -> dict[str, Any] | None:
    platform = detect_platform(root)
    if platform == "jetson":
        return collect_jetson_clock_state(root)
    if platform == "rk3588":
        # The RK3588 collector already exists on the RKNN adapter; reuse it
        # through the factory (the agent never imports concrete adapters).
        from calbredge_protocol.autodetect import get_adapter

        return get_adapter("rknn").collect_clock_state(root)  # type: ignore[attr-defined]
    if platform == "linux":
        return collect_linux_cpufreq_clock_state(root)
    return None


def _uncaptured(reason: str) -> dict[str, Any]:
    return {"captured": False, "reason": reason}


def _policies(root: Path) -> list[Path]:
    base = root / _CPUFREQ
    if not base.is_dir():
        return []
    return sorted(
        (p for p in base.iterdir() if p.name.startswith("policy") and p.name[6:].isdigit()),
        key=lambda p: int(p.name[6:]),
    )


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def collect_jetson_clock_state(root: Path = Path("/")) -> dict[str, Any]:
    """Per-policy CPU min/max/governor, the GPU devfreq min/max/cur, and EMC.

    clocks_pinned (derive_v2) uses only CPU and GPU min == max: jetson_clocks
    pins by raising the minimum to the maximum while the governor stays
    ``schedutil``. EMC is recorded separately and never feeds clocks_pinned.
    """
    policies = _policies(root)
    if not policies:
        return _uncaptured(f"no cpufreq policies under /{_CPUFREQ}")
    try:
        cpu_policies = {
            p.name: {
                "related_cpus": _read(p / "related_cpus"),
                "scaling_min_freq": int(_read(p / "scaling_min_freq")),
                "scaling_max_freq": int(_read(p / "scaling_max_freq")),
                "scaling_governor": _read(p / "scaling_governor"),
            }
            for p in policies
        }
    except (OSError, ValueError) as exc:
        return _uncaptured(f"failed to read cpufreq: {exc}")

    node = next((n for n in JETSON_GPU_DEVFREQ_NODES if (root / _DEVFREQ / n).is_dir()), None)
    if node is None:
        return _uncaptured(
            f"no known Jetson GPU devfreq node (supported: {', '.join(JETSON_GPU_DEVFREQ_NODES)})"
        )
    gpu_dir = root / _DEVFREQ / node
    try:
        gpu = {
            "node": node,
            "min_freq": int(_read(gpu_dir / "min_freq")),
            "max_freq": int(_read(gpu_dir / "max_freq")),
            "cur_freq": int(_read(gpu_dir / "cur_freq")),
        }
    except (OSError, ValueError) as exc:
        return _uncaptured(f"failed to read GPU devfreq {node}: {exc}")

    return {
        "captured": True,
        "source": "jetson",
        "cpu_policies": cpu_policies,
        "gpu": gpu,
        "emc": _jetson_emc(root),
    }


def _jetson_emc(root: Path) -> dict[str, Any]:
    """EMC pinning from BPMP debugfs, which needs root on stock JetPack.

    Unprivileged, this is ``{"pinned": null, ...}`` with the reason. There is
    no unprivileged fallback: tegrastats omits EMC_FREQ when not root, and the
    MAXN_SUPER nvpmodel profile declares no EMC cap to compare against, so an
    indirect answer would be a guess.
    """
    emc = root / _EMC_DEBUGFS
    try:
        min_rate = int(_read(emc / "min_rate"))
        max_rate = int(_read(emc / "max_rate"))
    except (OSError, ValueError) as exc:
        return {
            "pinned": None,
            "source": None,
            "reason": f"EMC debugfs not readable: {exc.__class__.__name__}",
        }
    return {
        "pinned": min_rate == max_rate,
        "source": "debugfs",
        "min_rate": min_rate,
        "max_rate": max_rate,
    }


def collect_linux_cpufreq_clock_state(root: Path = Path("/")) -> dict[str, Any]:
    policies = _policies(root)
    if not policies:
        return _uncaptured(f"no cpufreq policies under /{_CPUFREQ}")
    try:
        governors = {p.name: _read(p / "scaling_governor") for p in policies}
    except OSError as exc:
        return _uncaptured(f"failed to read cpufreq governors: {exc}")
    return {"captured": True, "source": "linux_cpufreq", "cpu_governors": governors}
