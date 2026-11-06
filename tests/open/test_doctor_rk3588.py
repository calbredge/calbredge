"""`calbredge doctor` on an RK3588: the RKNN binding, the runtime library it
loads, and whether this user can reach the NPU.

rknn-toolkit-lite2 is only a Python binding. librknnrt.so is not a pip package, and a vendor
image can ship 1.4.0 under a 2.3.2 binding, so the binding's version says nothing about what
executes: doctor reads the library's own version and refuses a mismatch.
"""

import os
from pathlib import Path

import calbredge
import pytest
from calbredge import doctor
from calbredge.doctor import Check, exit_code, librknnrt_version, report_text, run_checks

from tests.open.fake_hosts import build_jetson_orin, build_rk3588

pytestmark = pytest.mark.unit

BINDING = "2.3.2"


def _library(root: Path, version: str | None, rel: str = "usr/lib/librknnrt.so") -> Path:
    """A librknnrt.so that states its version the way the real one does, among other bytes."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    stated = f"librknnrt version: {version} (429f97ae6b@2025-04-09T09:09:27)" if version else ""
    path.write_bytes(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 64 + stated.encode() + b"\x00rknn\x00")
    return path


def _render_nodes(root: Path, mode: int = 0o660) -> list[Path]:
    nodes = []
    for name in ("renderD128", "renderD129"):
        node = root / "dev" / "dri" / name
        node.parent.mkdir(parents=True, exist_ok=True)
        node.write_bytes(b"")
        node.chmod(mode)
        nodes.append(node)
    return nodes


@pytest.fixture
def board(tmp_path, monkeypatch) -> Path:
    """A healthy Orange Pi 5 Plus: pinned, the binding and a matching library, the NPU open."""
    root = build_rk3588(tmp_path / "root")
    _library(root, BINDING)
    _render_nodes(root)
    dists = {
        "rknn-toolkit-lite2": BINDING,
        "numpy": "2.2.6",
        "onnx": "1.23.1",
        "calbredge": calbredge.__version__,
        "calbredge-protocol": calbredge.__version__,
    }
    present = {"rknnlite", "numpy", "onnx"}
    monkeypatch.setattr(doctor, "_dist_version", lambda name: dists.get(name))
    monkeypatch.setattr(doctor, "_module_version", lambda name: dists.get(name))
    monkeypatch.setattr(doctor, "_has_module", lambda name: name in present)
    monkeypatch.setattr(doctor, "_which", lambda cmd: None)
    monkeypatch.setattr(doctor, "_platform", lambda: ("Linux", "aarch64"))
    monkeypatch.setattr(doctor, "_user", lambda: "accept")
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    return root


def _named(root: Path, tmp_path: Path) -> dict[str, Check]:
    checks = run_checks(root=root, home=tmp_path / "home", system_metadata=dict)
    return {c.name: c for c in checks}


# --- the whole report --------------------------------------------------------------------


def test_a_healthy_rk3588_is_ready_with_the_three_rknn_checks(board, tmp_path):
    checks = run_checks(root=board, home=tmp_path / "home", system_metadata=dict)
    named = {c.name: c for c in checks}
    assert named["rknn-toolkit-lite2"] == Check("rknn-toolkit-lite2", "ok", BINDING)
    assert named["librknnrt"].status == "ok"
    assert named["librknnrt"].detail == f"{BINDING} ({board / 'usr/lib/librknnrt.so'})"
    assert named["NPU access"] == Check("NPU access", "ok", "renderD128, renderD129")
    assert named["clocks pinned"].detail == "yes (rk3588)"
    assert "rknn-fp16 supported" in named["support tier"].detail
    assert exit_code(checks) == 0
    assert report_text(checks).endswith("Ready.")


def test_a_jetson_report_has_none_of_them(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "_platform", lambda: ("Linux", "aarch64"))
    root = build_jetson_orin(tmp_path / "root", pinned=True)
    names = {c.name for c in run_checks(root=root, home=tmp_path / "home", system_metadata=dict)}
    assert not names & {"rknn-toolkit-lite2", "librknnrt", "NPU access"}


# --- librknnrt -----------------------------------------------------------------------------


def test_the_librarys_own_version_is_read_from_the_file(tmp_path):
    assert librknnrt_version(tmp_path) is None
    path = _library(tmp_path, "2.3.2")
    assert librknnrt_version(tmp_path) == (path, "2.3.2")


def test_the_first_library_on_the_loaders_path_is_the_one_reported(tmp_path):
    _library(tmp_path, "1.4.0", "usr/local/lib/librknnrt.so")
    first = _library(tmp_path, "2.3.2", "usr/lib/librknnrt.so")
    assert librknnrt_version(tmp_path) == (first, "2.3.2")


def test_a_missing_library_fails_and_says_where_one_comes_from(board, tmp_path):
    (board / "usr/lib/librknnrt.so").unlink()
    check = _named(board, tmp_path)["librknnrt"]
    assert (check.status, check.detail) == ("fail", "not installed (no librknnrt.so)")
    assert check.fix == (
        "install the one for rknn-toolkit-lite2 2.3.2 from the airockchip/rknn-toolkit2 "
        "repository, rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so: "
        "sudo cp librknnrt.so /usr/lib/librknnrt.so && sudo ldconfig"
    )
    assert exit_code(list(_named(board, tmp_path).values())) == 7


def test_the_vendor_images_old_library_under_a_new_binding_fails(board, tmp_path):
    """1.4.0 under a 2.3.2 toolkit: the binding's version does not identify what executes."""
    path = _library(board, "1.4.0")
    check = _named(board, tmp_path)["librknnrt"]
    assert check.status == "fail"
    assert check.detail == (
        f"1.4.0 at {path}, but rknn-toolkit-lite2 is 2.3.2: models converted for 2.3 may not "
        "load, or may crash"
    )
    assert check.fix is not None
    assert "the one for rknn-toolkit-lite2 2.3.2" in check.fix


@pytest.mark.parametrize("version", ["2.2.0", "3.3.2", "2.4.0"])
def test_a_different_major_or_minor_fails(board, tmp_path, version):
    _library(board, version)
    assert _named(board, tmp_path)["librknnrt"].status == "fail"


@pytest.mark.parametrize("version", ["2.3.0", "2.3.2", "2.3.9"])
def test_the_same_major_and_minor_is_ok_whatever_the_patch(board, tmp_path, version):
    _library(board, version)
    check = _named(board, tmp_path)["librknnrt"]
    assert check.status == "ok"
    assert check.detail.startswith(version)


def test_a_library_that_states_no_version_fails_never_passes_unread(board, tmp_path):
    path = _library(board, None)
    check = _named(board, tmp_path)["librknnrt"]
    assert (check.status, check.detail) == ("fail", f"{path}: its version can't be read")


# --- the binding ---------------------------------------------------------------------------


def test_a_missing_binding_fails_and_names_the_kit(board, tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "_dist_version", lambda name: None)
    named = _named(board, tmp_path)
    check = named["rknn-toolkit-lite2"]
    assert check.status == "fail"
    assert check.fix is not None
    assert "sudo ./install.sh" in check.fix
    assert "constraints/rk3588-jammy-py310.txt" in check.fix
    # With no binding there is nothing to match the library against: it is still reported.
    assert named["librknnrt"].status == "ok"


# --- NPU access ----------------------------------------------------------------------------


def test_no_render_node_means_the_npu_driver_is_not_loaded(board, tmp_path):
    for node in (board / "dev/dri").iterdir():
        node.unlink()
    check = _named(board, tmp_path)["NPU access"]
    assert check.status == "fail"
    assert "the NPU driver (rknpu) isn't loaded" in check.detail


@pytest.mark.skipif(os.geteuid() == 0, reason="root can open any node")
def test_a_user_outside_the_groups_is_told_to_join_them(board, tmp_path):
    for node in (board / "dev/dri").iterdir():
        node.chmod(0o000)
    check = _named(board, tmp_path)["NPU access"]
    assert check.status == "fail"
    assert check.detail == "accept can't open /dev/dri/renderD128, /dev/dri/renderD129"
    assert check.fix == "sudo usermod -aG video,render accept, then log in again"
    assert exit_code(list(_named(board, tmp_path).values())) == 7


@pytest.mark.skipif(os.geteuid() == 0, reason="root can open any node")
def test_one_node_the_user_can_open_is_enough(board, tmp_path):
    (board / "dev/dri/renderD128").chmod(0o000)
    assert _named(board, tmp_path)["NPU access"].status == "ok"
