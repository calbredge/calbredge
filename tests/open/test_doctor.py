"""`calbredge doctor`: host readiness from the collectors a run uses."""

import json
import os
import subprocess

import click
import pytest
from calbredge import cli, doctor, term
from calbredge.doctor import Check, cached_engines, exit_code, report_json, report_text, run_checks
from calbredge.term import Style
from calbredge_protocol.engine_cache import compute_build_key
from click.testing import CliRunner

from tests.open.fake_hosts import build_jetson_orin, build_x86, conv_model, nms_model

pytestmark = pytest.mark.unit

JETSON_MODULES = {"tensorrt": "10.3.0", "numpy": "1.26.4", "onnx": "1.23.0"}
JETSON_DISTS = {
    "protobuf": "7.34.1",
    "calbredge": doctor.calbredge.__version__,
    "calbredge-protocol": doctor.calbredge.__version__,
}


@pytest.fixture
def jetson_probes(monkeypatch):
    """A healthy JetPack 6 software stack, on top of a fake /proc and /sys tree."""
    modules = dict(JETSON_MODULES)
    dists = dict(JETSON_DISTS)
    present = {"tensorrt", "pycuda", "numpy", "onnx"}
    monkeypatch.setattr(doctor, "_module_version", lambda name: modules.get(name))
    monkeypatch.setattr(doctor, "_has_module", lambda name: name in present)
    monkeypatch.setattr(doctor, "_dist_version", lambda name: dists.get(name))
    monkeypatch.setattr(doctor, "_which", lambda cmd: f"/usr/bin/{cmd}")
    monkeypatch.setattr(doctor, "_power_mode", lambda root: "maxn_super")
    monkeypatch.setattr(doctor, "_platform", lambda: ("Linux", "aarch64"))
    return {"modules": modules, "dists": dists, "present": present}


def _checks(tmp_path, *, pinned: bool, **kwargs) -> list[Check]:
    root = build_jetson_orin(tmp_path / "root", pinned=pinned)
    return run_checks(root=root, home=tmp_path / "home", system_metadata=dict, **kwargs)


def _by_name(checks: list[Check]) -> dict[str, Check]:
    return {c.name: c for c in checks}


def test_pinned_jetson_is_ready(tmp_path, jetson_probes):
    checks = _checks(tmp_path, pinned=True)
    assert exit_code(checks) == 0
    named = _by_name(checks)
    assert named["clocks pinned"].status == "ok"
    assert named["power mode"].detail == "maxn_super"
    assert named["hardware family"].detail.startswith(
        "nvidia_jetson_orin_nano_engineering_reference_developer_kit_super (device_tree)"
    )
    assert named["JetPack"].detail == "L4T 36.4.7, CUDA 12.6.11"
    assert "module_rail" in named["power telemetry"].detail
    tiers = named["support tier"].detail
    assert "linux/jetson" in tiers
    assert "tensorrt-fp16 supported" in tiers
    assert "tensorrt-int8 results_only" in tiers
    assert report_text(checks).endswith("Ready.")


def test_unpinned_jetson_is_not_ready_and_shows_jetson_clocks(tmp_path, jetson_probes):
    checks = _checks(tmp_path, pinned=False)
    assert exit_code(checks) == 7
    clocks = _by_name(checks)["clocks pinned"]
    assert clocks.status == "fail"
    assert clocks.fix is not None
    assert "sudo jetson_clocks" in clocks.fix
    text = report_text(checks)
    assert "fix: run: sudo jetson_clocks" in text
    assert "Not ready: 1 check(s) failed (exit 7)." in text


def test_unpinned_jetson_without_the_unit_names_jetson_clocks_and_install_sh(
    tmp_path, jetson_probes
):
    clocks = _by_name(_checks(tmp_path, pinned=False))["clocks pinned"]
    assert clocks.fix == (
        "run: sudo jetson_clocks (until the next reboot), or sudo ./install.sh "
        "--pin-clocks-at-boot from the Calbredge kit (now and at every boot)"
    )


def test_unpinned_jetson_with_the_unit_names_a_restart(tmp_path, jetson_probes):
    """The unit stays active after its first run, so only a restart re-pins after a mode change."""
    root = build_jetson_orin(tmp_path / "root", pinned=False)
    unit = root / "etc" / "systemd" / "system" / "calbredge-jetson-clocks.service"
    unit.parent.mkdir(parents=True)
    unit.write_text("[Service]\nType=oneshot\n")
    checks = run_checks(root=root, home=tmp_path / "home", system_metadata=dict)
    clocks = _by_name(checks)["clocks pinned"]
    assert clocks.fix == (
        "run: sudo systemctl restart calbredge-jetson-clocks.service (re-pins for the current "
        "power mode; needed after every power-mode change)"
    )
    assert "sudo jetson_clocks" not in clocks.fix


@pytest.mark.parametrize(
    ("change", "name", "fix"),
    [
        (lambda p: p["modules"].update(numpy="2.1.0"), "numpy", "numpy<2"),
        (lambda p: p["present"].discard("pycuda"), "pycuda", '"calbredge[jetson]"'),
        (lambda p: p["modules"].pop("tensorrt"), "TensorRT", "--system-site-packages"),
        (
            lambda p: p["dists"].update({"calbredge-protocol": "0.1.1"}),
            "package versions",
            "together",
        ),
    ],
)
def test_broken_jetson_stack_fails_with_its_fix(tmp_path, jetson_probes, change, name, fix):
    change(jetson_probes)
    checks = _checks(tmp_path, pinned=True)
    check = _by_name(checks)[name]
    assert check.status == "fail"
    assert check.fix is not None
    assert fix in check.fix
    assert exit_code(checks) == 7


def test_missing_tegrastats_is_a_warning_not_a_failure(tmp_path, jetson_probes, monkeypatch):
    monkeypatch.setattr(doctor, "_which", lambda cmd: None)
    checks = _checks(tmp_path, pinned=True)
    assert _by_name(checks)["power telemetry"].status == "warn"
    assert exit_code(checks) == 0


def test_config_values_are_never_printed(tmp_path, jetson_probes):
    config = tmp_path / "home" / ".calbredge" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("api_key: SECRET-KEY-123\nupload_url: https://internal.example/api\n")
    checks = _checks(tmp_path, pinned=True)
    assert _by_name(checks)["config file"].detail == "present; upload URL set: yes"
    for output in (report_text(checks), report_json(checks)):
        assert "SECRET-KEY-123" not in output
        assert "internal.example" not in output


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
def test_unwritable_state_dir_fails(tmp_path, jetson_probes):
    base = tmp_path / "home" / ".calbredge"
    (base / "runs").mkdir(parents=True)
    (base / "runs").chmod(0o500)
    try:
        check = _by_name(_checks(tmp_path, pinned=True))["~/.calbredge writable"]
    finally:
        (base / "runs").chmod(0o700)
    assert check.status == "fail"
    assert str(base / "runs") in check.detail


@pytest.fixture
def x86_probes(monkeypatch):
    """An x86 host with ONNX Runtime CPU and matching package versions."""
    dists = dict(JETSON_DISTS)
    monkeypatch.setattr(doctor, "_has_module", lambda name: name == "onnxruntime")
    monkeypatch.setattr(
        doctor, "_module_version", lambda name: {"numpy": "2.2.6", "onnx": "1.23.0"}.get(name)
    )
    monkeypatch.setattr(doctor, "_dist_version", lambda name: dists.get(name))
    monkeypatch.setattr(doctor, "_platform", lambda: ("Linux", "x86_64"))
    monkeypatch.setattr(doctor, "predict_runtime_label", lambda r: ("onnxruntime-cpu", True))
    return dists


_X86_METADATA = {"cpu_model": "x86_64", "cpu_cores": 8, "ram_gb": 7.5, "gpu_model": None}


def _x86_checks(tmp_path, governor: str = "powersave") -> list[Check]:
    root = build_x86(tmp_path / "root", governor=governor)
    return run_checks(root=root, home=tmp_path / "home", system_metadata=lambda: _X86_METADATA)


@pytest.mark.parametrize("governor", ["powersave", "performance"])
def test_x86_host_with_onnxruntime_is_ready_and_reports_results_only(
    tmp_path, x86_probes, governor
):
    checks = _x86_checks(tmp_path, governor)
    named = _by_name(checks)
    assert named["hardware family"].detail == "x86_64_8_7_5_none (derived)"
    clocks = named["clocks pinned"]
    assert clocks.status == "warn"
    assert clocks.detail.startswith(
        "not determined (linux_cpufreq): no rule shows this CPU's speed is fixed"
    )
    performance = 2 if governor == "performance" else 0
    assert f"governors: performance on {performance} of 2 policies" in clocks.detail
    assert "scaling_governor" not in (clocks.fix or "")  # the governor doesn't fix the speed
    assert "clocks_pinned as null" in (clocks.fix or "")
    support = named["support tier"]
    assert support.status == "ok"  # information, not readiness
    assert "onnxruntime-cpu results_only" in support.detail
    assert "gating needs a supported combination" in support.detail
    assert "JetPack" not in named
    assert named["TensorRT"].status == "ok"  # not needed off Jetson
    assert exit_code(checks) == doctor.EXIT_READY
    assert report_text(checks).endswith("Ready.")


def test_a_host_with_no_runtime_is_not_ready(tmp_path, x86_probes, monkeypatch):
    monkeypatch.setattr(doctor, "_has_module", lambda name: False)
    checks = _x86_checks(tmp_path)
    support = _by_name(checks)["support tier"]
    assert support.status == "fail"
    assert "no runtime installed" in support.detail
    assert exit_code(checks) == doctor.EXIT_NOT_READY


def test_a_host_with_no_cpufreq_warns_off_jetson_and_rk3588(tmp_path, x86_probes):
    root = tmp_path / "root"
    root.mkdir()
    checks = run_checks(root=root, home=tmp_path / "home", system_metadata=lambda: _X86_METADATA)
    clocks = _by_name(checks)["clocks pinned"]
    assert clocks.status == "warn"
    assert clocks.detail.startswith("unknown: ")
    assert exit_code(checks) == doctor.EXIT_READY


def test_an_unreadable_clock_state_on_a_jetson_still_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "detect_platform", lambda root: "jetson")
    monkeypatch.setattr(
        doctor, "collect_clock_state", lambda root: {"captured": False, "reason": "no gpu node"}
    )
    clocks = doctor._clocks(tmp_path)
    assert clocks.status == "fail"
    assert clocks.detail == "unknown: no gpu node"


# --- --model ---------------------------------------------------------------------------


def _model(tmp_path, build, name):
    (tmp_path / "m").mkdir(exist_ok=True)
    return build(tmp_path / "m" / name)


def _cache_engine(model, *, model_hash, power_mode="maxn_super", trt="10.3.0", tamper=False):
    inputs = {
        "artifact": "tensorrt_engine",
        "model_hash": model_hash,
        "precision": "fp16",
        "power_mode": power_mode,
        "input_profile": {},
        "calibration_digest": None,
        "compiler_version": trt,
        "builder_config_version": 1,
    }
    key = compute_build_key(inputs)
    name = f"{model.name}.trt.fp16.{power_mode}.{key[:16]}"
    (model.parent / f"{name}.engine").write_bytes(b"engine")
    record = {"build_key": "0" * 64 if tamper else key, "inputs": inputs}
    (model.parent / f"{name}.engine.json").write_text(json.dumps(record))


def test_model_contract_and_a_usable_cached_engine(tmp_path, jetson_probes):
    model = _model(tmp_path, conv_model, "yolo.onnx")
    _cache_engine(model, model_hash=doctor._sha256_file(model))
    named = _by_name(_checks(tmp_path, pinned=True, model=model))
    assert named["model graph_kind"].detail == "fixed_compute"
    assert "float32" in named["model io_signature"].detail
    engine = named["cached engine"]
    assert engine.status == "ok"
    assert ".trt.fp16.maxn_super." in engine.detail


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_hash": "f" * 64},  # built from a different model saved at this path
        {"power_mode": "15w"},
        {"trt": "10.0.0"},
        {"tamper": True},
    ],
)
def test_unusable_cached_engines_are_a_warning(tmp_path, jetson_probes, kwargs):
    model = _model(tmp_path, conv_model, "yolo.onnx")
    _cache_engine(model, **{"model_hash": doctor._sha256_file(model), **kwargs})
    checks = _checks(tmp_path, pinned=True, model=model)
    engine = _by_name(checks)["cached engine"]
    assert engine.status == "warn"
    assert "1 other engine(s)" in engine.detail
    assert exit_code(checks) == 0


def test_cached_engines_looks_where_the_adapter_looks(tmp_path):
    model = _model(tmp_path, conv_model, "yolo.onnx")
    home = tmp_path / "home"
    fallback = home / ".calbredge" / "engine-cache"
    fallback.mkdir(parents=True)
    (fallback / f"{model.name}.trt.fp16.default.0123456789abcdef.engine").write_bytes(b"e")
    (fallback / f"{model.name}.trt.fp16.default.0123456789abcdef.engine.json").write_text("{}")
    (model.parent / f"{model.name}.trt.fp16.maxn_super").write_bytes(b"legacy, unstamped")
    found = cached_engines(model, home)
    assert [e["engine"].split("/")[-1] for e in found] == [
        f"{model.name}.trt.fp16.default.0123456789abcdef.engine"
    ]
    assert found[0]["intact"] is False


def test_data_dependent_model_warns(tmp_path, jetson_probes):
    model = _model(tmp_path, nms_model, "nms.onnx")
    check = _by_name(_checks(tmp_path, pinned=True, model=model))["model graph_kind"]
    assert check.status == "warn"
    assert "data_dependent" in check.detail


def test_doctor_writes_nothing_next_to_the_model(tmp_path, jetson_probes):
    model = _model(tmp_path, conv_model, "yolo.onnx")
    _checks(tmp_path, pinned=True, model=model)
    assert sorted(p.name for p in model.parent.iterdir()) == ["yolo.onnx"]
    assert not (tmp_path / "home").exists()


def test_doctor_never_runs_sudo(tmp_path, monkeypatch):
    calls = []
    real_run = subprocess.run

    def spy(args, *a, **k):
        calls.append(args)
        return real_run(args, *a, **k)

    monkeypatch.setattr(subprocess, "run", spy)
    run_checks(root=build_jetson_orin(tmp_path / "root", pinned=False), home=tmp_path / "home")
    assert not [c for c in calls if "sudo" in (c if isinstance(c, str) else " ".join(c))]


# --- the command -----------------------------------------------------------------------


def test_cli_json_is_one_document_and_exit_code_follows_readiness(monkeypatch):
    failing = [Check("clocks pinned", "fail", "no (jetson)", "run: sudo jetson_clocks")]
    monkeypatch.setattr(doctor, "run_checks", lambda **k: failing)
    result = CliRunner().invoke(cli.main, ["doctor", "--json"])
    assert result.exit_code == 7
    document = json.loads(result.stdout)
    assert document["ready"] is False
    assert document["exit_code"] == 7
    assert document["checks"][0]["fix"] == "run: sudo jetson_clocks"


def test_cli_ready_exits_0(monkeypatch):
    monkeypatch.setattr(doctor, "run_checks", lambda **k: [Check("x", "warn", "y")])
    result = CliRunner().invoke(cli.main, ["doctor"])
    assert result.exit_code == 0
    assert "Ready." in result.stdout


def test_cli_usage_error_is_exit_2():
    assert CliRunner().invoke(cli.main, ["doctor", "--no-such-flag"]).exit_code == 2


# --- terminal styling: plain output unchanged; styled only on a terminal ------------

_STYLE_CHECKS = [
    Check("clocks pinned", "ok", "yes (jetson)"),
    Check("power mode", "warn", "unknown", "set a mode with nvpmodel"),
    Check("TensorRT", "fail", "not importable", "install JetPack"),
]
# Exactly what report_text printed before styling existed.
_PLAIN = (
    "OK    clocks pinned  yes (jetson)\n"
    "WARN  power mode     unknown\n"
    "                     fix: set a mode with nvpmodel\n"
    "FAIL  TensorRT       not importable\n"
    "                     fix: install JetPack\n"
    "\n"
    "Not ready: 1 check(s) failed (exit 7)."
)


def test_plain_report_is_unchanged():
    assert report_text(_STYLE_CHECKS) == _PLAIN
    assert report_text(_STYLE_CHECKS, Style()) == _PLAIN


def test_styled_report_adds_only_symbols_and_colour():
    styled = report_text(_STYLE_CHECKS, Style(color=True))
    assert "\x1b[" in styled
    lines = click.unstyle(styled).splitlines()
    assert lines[0] == "✓ OK    clocks pinned  yes (jetson)"
    assert lines[1] == "⚠ WARN  power mode     unknown"
    assert lines[2] == "                       fix: set a mode with nvpmodel"
    assert lines[3] == "✗ FAIL  TensorRT       not importable"
    assert lines[-1] == "Not ready: 1 check(s) failed (exit 7)."


@pytest.mark.parametrize("env", [{}, {"GITHUB_ACTIONS": "true"}, {"NO_COLOR": "1"}])
def test_cli_output_is_plain_when_not_a_terminal(monkeypatch, env):
    monkeypatch.setattr(doctor, "run_checks", lambda **k: _STYLE_CHECKS)
    for key in ("NO_COLOR", "GITHUB_ACTIONS"):
        monkeypatch.delenv(key, raising=False)
    result = CliRunner().invoke(cli.main, ["doctor"], env=env)
    assert result.exit_code == 7
    assert result.stdout == _PLAIN + "\n"


def test_cli_is_styled_on_a_terminal_but_never_with_json_or_no_color(monkeypatch):
    monkeypatch.setattr(doctor, "run_checks", lambda **k: _STYLE_CHECKS)
    monkeypatch.setattr(term, "_is_tty", lambda stream: True)
    monkeypatch.delenv("NO_COLOR", raising=False)
    styled = CliRunner().invoke(cli.main, ["doctor"])
    assert "\x1b[" in styled.stdout
    assert click.unstyle(styled.stdout).startswith("✓ OK    clocks pinned")
    assert styled.exit_code == 7
    as_json = CliRunner().invoke(cli.main, ["doctor", "--json"])
    assert "\x1b[" not in as_json.stdout
    assert json.loads(as_json.stdout)["exit_code"] == 7
    no_color = CliRunner().invoke(cli.main, ["doctor"], env={"NO_COLOR": "1"})
    assert no_color.stdout == _PLAIN + "\n"
