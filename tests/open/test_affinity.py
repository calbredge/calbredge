"""Unit tests for CPU affinity detection/resolution/application.

detect_clusters and resolve_affinity are pure functions tested against a
fake sysfs tree under tmp_path — no dependency on real /sys or a real
big.LITTLE host. apply_affinity is tested with os.sched_setaffinity
monkeypatched so the suite never actually changes this process's affinity.
"""

from pathlib import Path
from typing import ClassVar

import pytest
from calbredge.affinity import apply_affinity, detect_clusters, resolve_affinity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_cpu_capacity(root: Path, cpu_id: int, capacity: int) -> None:
    cpu_dir = root / "sys" / "devices" / "system" / "cpu" / f"cpu{cpu_id}"
    cpu_dir.mkdir(parents=True, exist_ok=True)
    (cpu_dir / "cpu_capacity").write_text(f"{capacity}\n")


def _write_rk3588_like(root: Path) -> None:
    """4 little (A55, capacity 397) + 4 big (A76 across two policy clusters, capacity 1024)."""
    for cpu_id in range(4):
        _write_cpu_capacity(root, cpu_id, 397)
    for cpu_id in range(4, 8):
        _write_cpu_capacity(root, cpu_id, 1024)


# ---------------------------------------------------------------------------
# detect_clusters
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDetectClusters:
    def test_rk3588_like_splits_big_and_little(self, tmp_path):
        _write_rk3588_like(tmp_path)

        result = detect_clusters(sysfs_root=tmp_path)

        assert result["big"] == [4, 5, 6, 7]
        assert result["little"] == [0, 1, 2, 3]
        assert result["capacities"] == {
            0: 397,
            1: 397,
            2: 397,
            3: 397,
            4: 1024,
            5: 1024,
            6: 1024,
            7: 1024,
        }

    def test_homogeneous_returns_none(self, tmp_path):
        for cpu_id in range(4):
            _write_cpu_capacity(tmp_path, cpu_id, 1024)

        assert detect_clusters(sysfs_root=tmp_path) is None

    def test_missing_capacity_files_returns_none(self, tmp_path):
        cpu_dir = tmp_path / "sys" / "devices" / "system" / "cpu" / "cpu0"
        cpu_dir.mkdir(parents=True)
        # No cpu_capacity file written — simulates x86 or an old kernel.

        assert detect_clusters(sysfs_root=tmp_path) is None

    def test_missing_cpu_tree_returns_none(self, tmp_path):
        # tmp_path exists but has no sys/devices/system/cpu at all.
        assert detect_clusters(sysfs_root=tmp_path) is None

    def test_ignores_non_cpuN_siblings(self, tmp_path):
        """cpuidle/cpufreq siblings must not be mistaken for per-core dirs."""
        _write_rk3588_like(tmp_path)
        cpu_root = tmp_path / "sys" / "devices" / "system" / "cpu"
        (cpu_root / "cpuidle").mkdir()
        (cpu_root / "cpufreq").mkdir()

        result = detect_clusters(sysfs_root=tmp_path)

        assert result["big"] == [4, 5, 6, 7]
        assert result["little"] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# resolve_affinity
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveAffinity:
    RK3588_CLUSTERS: ClassVar[dict] = {
        "big": [4, 5, 6, 7],
        "little": [0, 1, 2, 3],
        "capacities": {i: 397 if i < 4 else 1024 for i in range(8)},
    }

    def test_none_spec_always_returns_none(self):
        assert resolve_affinity("none", self.RK3588_CLUSTERS) is None
        assert resolve_affinity("none", None) is None

    def test_big_spec_returns_big_cluster(self):
        assert resolve_affinity("big", self.RK3588_CLUSTERS) == {4, 5, 6, 7}

    def test_little_spec_returns_little_cluster(self):
        assert resolve_affinity("little", self.RK3588_CLUSTERS) == {0, 1, 2, 3}

    def test_big_spec_fails_soft_when_clusters_none(self):
        assert resolve_affinity("big", None) is None

    def test_little_spec_fails_soft_when_clusters_none(self):
        assert resolve_affinity("little", None) is None

    def test_explicit_core_list_parses_to_set(self):
        assert resolve_affinity("0,2", self.RK3588_CLUSTERS) == {0, 2}
        assert resolve_affinity("0,2", None) == {0, 2}

    def test_malformed_spec_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid --cpu-affinity"):
            resolve_affinity("garbage", self.RK3588_CLUSTERS)


# ---------------------------------------------------------------------------
# apply_affinity
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApplyAffinity:
    def test_none_core_set_does_not_call_sched_setaffinity(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "os.sched_setaffinity", lambda pid, mask: calls.append((pid, mask)), raising=False
        )

        result = apply_affinity(None)

        assert result is None
        assert calls == []

    def test_core_set_calls_sched_setaffinity_with_pid_zero(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "os.sched_setaffinity", lambda pid, mask: calls.append((pid, mask)), raising=False
        )

        result = apply_affinity({4, 5, 6, 7})

        assert result == {4, 5, 6, 7}
        assert calls == [(0, {4, 5, 6, 7})]

    def test_returns_none_when_sched_setaffinity_unavailable(self, monkeypatch):
        monkeypatch.delattr("os.sched_setaffinity", raising=False)

        result = apply_affinity({4, 5, 6, 7})

        assert result is None
