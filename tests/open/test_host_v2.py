"""Tests for agent/calbredge/host_v2.py."""

import os
import subprocess

import pytest
from calbredge import host_v2
from calbredge.host_v2 import (
    cuda_version,
    device_tree_model,
    driver_version,
    git_state,
    hardware_family,
    l4t_version,
    power_mode,
    runtime_version,
)

from tests.open.fake_hosts import ORIN_FAMILY, ORIN_MODEL, build_jetson_orin, build_x86

pytestmark = pytest.mark.unit

X86_META = {"cpu_model": "x86_64", "cpu_cores": 8, "ram_gb": 7.5, "gpu_model": None}


def test_device_tree_model_strips_trailing_nul(tmp_path):
    assert device_tree_model(build_jetson_orin(tmp_path, pinned=True)) == ORIN_MODEL


def test_hardware_family_from_device_tree(tmp_path):
    root = build_jetson_orin(tmp_path, pinned=True)
    assert hardware_family(X86_META, root) == (ORIN_FAMILY, "device_tree")


def test_hardware_family_derived_without_device_tree(tmp_path):
    assert hardware_family(X86_META, build_x86(tmp_path)) == ("x86_64_8_7_5_none", "derived")


def test_l4t_and_cuda(tmp_path):
    root = build_jetson_orin(tmp_path, pinned=True)
    assert l4t_version(root) == "36.4.7"
    assert cuda_version(root) == "12.6.11"
    assert driver_version(root, query_nvidia_smi=False) == {"l4t": "36.4.7", "cuda": "12.6.11"}


@pytest.mark.parametrize("content", ["", "garbage\n", "# R36 (release)\n"])
def test_malformed_l4t_release_is_none(tmp_path, content):
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc/nv_tegra_release").write_text(content)
    assert l4t_version(tmp_path) is None


def test_no_driver_facts_is_none(tmp_path):
    assert driver_version(tmp_path, query_nvidia_smi=False) is None


@pytest.mark.parametrize(
    ("raw", "expected"), [("1.23.0", "1.23.0"), ("unknown", None), ("n/a", None), ("", None)]
)
def test_runtime_version_never_a_placeholder(monkeypatch, raw, expected):
    monkeypatch.setattr(host_v2, "_get_runtime_version", lambda name: raw)
    assert runtime_version("onnxruntime") == expected


def test_power_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(host_v2, "_get_nvpmodel_mode", lambda: "maxn_super")
    assert power_mode(build_jetson_orin(tmp_path / "j", pinned=True)) == "maxn_super"
    assert power_mode(build_x86(tmp_path / "x")) == "not_applicable"
    monkeypatch.setattr(host_v2, "_get_nvpmodel_mode", lambda: None)
    assert power_mode(tmp_path / "j") is None  # Jetson but nvpmodel unreadable: never a placeholder


def _git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "tracked.txt").write_text("v1\n")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_git_clean(repo):
    commit, dirty = git_state(repo)
    assert commit is not None
    assert len(commit) == 40
    assert dirty is False


def test_git_untracked_files_do_not_make_it_dirty(repo):
    (repo / "build_output.bin").write_text("x")
    assert git_state(repo)[1] is False


def test_git_modified_tracked_file_is_dirty(repo):
    (repo / "tracked.txt").write_text("v2\n")
    assert git_state(repo)[1] is True


def test_git_staged_change_is_dirty(repo):
    (repo / "new.txt").write_text("x")
    _git(repo, "add", "new.txt")
    assert git_state(repo)[1] is True


def test_outside_a_repo(tmp_path, monkeypatch):
    outside = tmp_path / "not_a_repo"
    outside.mkdir()
    # Stop git from finding any repository enclosing tmp_path.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    assert git_state(outside) == (None, None)


def test_git_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert git_state(tmp_path) == (None, None)


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected"),
    [(0, "560.35.03\n560.35.03\n", "560.35.03"), (9, "", None), (0, " \n", None)],
    ids=["one line per GPU", "nvidia-smi fails", "no output"],
)
def test_the_nvidia_driver_is_the_first_gpus_or_none(
    tmp_path, monkeypatch, returncode, stdout, expected
):
    done = subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="")
    monkeypatch.setattr(host_v2.subprocess, "run", lambda *args, **kwargs: done)
    assert host_v2.nvidia_driver_version() == expected
    # A host with no L4T or CUDA files: the driver is the only fact, or there is none.
    assert driver_version(tmp_path) == (None if expected is None else {"nvidia_driver": expected})


def test_no_nvidia_smi_is_no_driver_version(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(host_v2.subprocess, "run", missing)
    assert host_v2.nvidia_driver_version() is None


def test_a_commit_whose_status_cannot_be_read_has_no_git_state(repo, monkeypatch):
    """Never a commit with a guessed dirty flag: both are null together."""
    real = host_v2._git
    monkeypatch.setattr(
        host_v2, "_git", lambda cwd, *args: None if args[0] == "status" else real(cwd, *args)
    )
    assert git_state(repo) == (None, None)


def test_the_repository_root_and_what_git_tracks(repo, tmp_path_factory, monkeypatch):
    nested = repo / "models"
    nested.mkdir()
    (nested / "m.onnx").write_bytes(b"onnx")
    assert host_v2.git_toplevel(nested) == repo
    assert host_v2.git_tracked(repo / "tracked.txt") is True
    assert host_v2.git_tracked(nested / "m.onnx") is False
    outside = tmp_path_factory.mktemp("outside")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(outside.parent))
    assert host_v2.git_toplevel(outside) is None
    assert host_v2.git_tracked(outside / "m.onnx") is None


def test_git_user_name_is_the_configured_one_or_none(repo, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    assert host_v2.git_user_name(repo) is None
    _git(repo, "config", "user.name", "Release Engineering")
    assert host_v2.git_user_name(repo) == "Release Engineering"
    _git(repo, "config", "user.name", " ")
    assert host_v2.git_user_name(repo) is None
