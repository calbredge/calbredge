"""`calbredge init` per platform: which runner the project's workflow targets,
what an RK3588 project needs beside its model, and hosts that are neither.

The Jetson's workflow comes from the release gate's init hook and is tested with it.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from calbredge import init
from calbredge.doctor import Check
from calbredge.project import read_project
from calbredge_protocol import rknn_artifact
from calbredge_protocol.rknn_adapter import RKNNAdapter

from tests.open.test_init import _GIT_ENV, _git, _init, _model

pytestmark = pytest.mark.unit
ARTIFACT = "m.onnx.rknn.fp16.default"


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    monkeypatch.chdir(root)
    for key, value in _GIT_ENV.items():
        monkeypatch.setenv(key, value)
    return root


@pytest.fixture
def runs(monkeypatch) -> list:
    """A fake discovery run that reports the runtime it was asked for."""
    calls: list = []

    def fake(model, runtime, protocol):
        calls.append((model, runtime, protocol))
        return SimpleNamespace(
            run_id="r-1",
            run_status="stable",
            run_status_reason=None,
            latency_p50_ms=5.9871,
            runtime=runtime,
        )

    monkeypatch.setattr(init, "discovery_run", fake)
    return calls


def _host(monkeypatch, platform: str | None, *, pinned: bool = True) -> None:
    """Stand for a Jetson, an RK3588, or (None) an x86 host, with its clocks pinned or not."""
    monkeypatch.setattr(init, "detected_platform", lambda: platform)
    monkeypatch.setattr("calbredge.clock_state.detect_platform", lambda root: platform or "linux")
    source = platform or "linux_cpufreq"
    check = (
        Check("clocks pinned", "ok", f"yes ({source})")
        if pinned
        else Check(
            "clocks pinned",
            "fail",
            f"no ({source}): timings would move",
            "run: sudo scripts/performance_mode.sh from a Calbredge checkout",
        )
    )
    monkeypatch.setattr(init, "clock_check", lambda: check)


def _rk3588_model(repo: Path, *, artifact: bool = True, record: bool = True) -> Path:
    """mobilenetv2's input shape, with the converted model and its record as asked."""
    model = _model(repo / "models" / "m.onnx", ["N", 3, 224, 224])
    converted = model.parent / ARTIFACT
    if artifact:
        converted.write_bytes(b"rknn bytes")
    if record:
        rknn_artifact.record_path(converted).write_text("{}")
    return model


def _nothing_written(repo: Path) -> bool:
    return not (repo / ".calbredge").exists() and not (repo / ".github").exists()


@pytest.mark.parametrize("detected", ["jetson", "rk3588"])
def test_the_platform_is_detected_on_a_target_and_may_be_named(monkeypatch, detected):
    monkeypatch.setattr(init, "detected_platform", lambda: detected)
    assert init.resolve_platform(None) == detected
    assert init.resolve_platform(detected) == detected


def test_a_platform_that_contradicts_the_host_is_refused(monkeypatch):
    monkeypatch.setattr(init, "detected_platform", lambda: "jetson")
    with pytest.raises(init.InitError, match="doesn't match this host, which is a jetson"):
        init.resolve_platform("rk3588")


def test_a_host_that_is_neither_must_say_which(monkeypatch):
    monkeypatch.setattr(init, "detected_platform", lambda: None)
    with pytest.raises(init.InitError, match=r"pass --platform jetson\|rk3588"):
        init.resolve_platform(None)
    assert init.resolve_platform("rk3588") == "rk3588"
    assert init.resolve_platform("jetson") == "jetson"


@pytest.mark.parametrize(
    ("host", "expected"),
    [("jetson", "jetson"), ("rk3588", "rk3588"), ("linux", None), (None, None)],
)
def test_only_a_jetson_or_an_rk3588_is_a_detected_platform(monkeypatch, host, expected):
    monkeypatch.setattr("calbredge.clock_state.detect_platform", lambda root: host)
    assert init.detected_platform() == expected


@pytest.mark.parametrize(
    ("host", "runtime"),
    [
        ("jetson", "tensorrt-fp16"),
        ("rk3588", "rknn-fp16"),
        ("linux", "onnxruntime"),
        (None, "onnxruntime"),
    ],
)
def test_the_default_runtime_is_the_hosts_accelerator(monkeypatch, host, runtime):
    monkeypatch.setattr("calbredge.clock_state.detect_platform", lambda root: host)
    assert init.default_runtime() == runtime


def test_the_converted_model_init_looks_for_is_the_one_the_adapter_loads(tmp_path):
    model = tmp_path / "m.onnx"
    for label, precision in (("rknn-fp16", "fp16"), ("rknn-int8", "int8")):
        adapter = RKNNAdapter(precision=precision)
        assert init.rknn_artifact_path(model, label) == adapter._resolve_rknn_path(model)
    assert init.rknn_artifact_path(model, "rknn").name == "m.onnx.rknn.fp16.default"


@pytest.mark.parametrize(
    ("artifact", "record", "message"),
    [
        (False, False, f"has no converted model beside it ({ARTIFACT})"),
        (True, False, f"{ARTIFACT} has no build record ({ARTIFACT}.json)"),
    ],
)
def test_an_rk3588_project_needs_the_converted_model_and_its_record(
    repo, runs, monkeypatch, artifact, record, message
):
    _host(monkeypatch, "rk3588")
    _rk3588_model(repo, artifact=artifact, record=record)

    result = _init("--model", "models/m.onnx")

    assert result.exit_code == 1
    assert message in result.output
    assert "Nothing was written." in result.output
    assert (
        "On an x86 host, run: calbredge convert --framework rknn --model m.onnx "
        f"--precision fp16 --output {ARTIFACT}"
    ) in result.output
    assert f"then copy {ARTIFACT} and {ARTIFACT}.json beside models/m.onnx." in result.output
    assert _nothing_written(repo)
    assert runs == []


def test_example_is_refused_on_an_rk3588_before_anything_is_fetched(repo, runs, monkeypatch):
    _host(monkeypatch, "rk3588")

    def fetch(*args, **kwargs):
        raise AssertionError("the example must not be downloaded")

    monkeypatch.setattr("calbredge.example_model.fetch", fetch)
    result = _init("--example")
    assert result.exit_code == 1
    assert "--example downloads only the .onnx, and an RK3588 can't convert it" in result.output
    assert "calbredge convert --framework rknn" in result.output
    assert _nothing_written(repo)
    assert not (repo / "models").exists()


def test_unpinned_clocks_on_an_rk3588_get_the_rk3588_fix(repo, runs, monkeypatch):
    _host(monkeypatch, "rk3588", pinned=False)
    _rk3588_model(repo)
    result = _init("--model", "models/m.onnx")
    assert result.exit_code == init.EXIT_NOT_READY
    assert "clocks pinned: no (rk3588)" in result.output
    assert "sudo scripts/performance_mode.sh" in result.output
    assert "jetson_clocks" not in result.output
    assert _nothing_written(repo)


def test_another_runtime_on_an_rk3588_needs_no_converted_model(repo, runs, monkeypatch):
    _host(monkeypatch, "rk3588")
    _rk3588_model(repo, artifact=False, record=False)
    result = _init("--model", "models/m.onnx", "--runtime", "onnxruntime")
    assert result.exit_code == 0, result.output
    assert f"Commit {ARTIFACT}" not in result.output


def test_an_unknown_platform_is_a_usage_error(repo, runs, monkeypatch):
    _host(monkeypatch, None)
    result = _init("--model", "models/m.onnx", "--platform", "hailo8")
    assert result.exit_code == 2


def test_without_a_hook_init_on_an_rk3588_sets_up_an_rknn_project(repo, runs, monkeypatch):
    monkeypatch.setattr(init, "load_hooks", list)
    _host(monkeypatch, "rk3588")
    model = _rk3588_model(repo)

    result = _init("--model", "models/m.onnx")

    assert result.exit_code == 0, result.output
    project = read_project(repo / ".calbredge" / "project.yaml")
    assert [t.runtime for t in project.targets] == ["rknn-fp16"]
    assert project.policy is None
    assert runs == [(model, "rknn-fp16", repo / ".calbredge" / "protocol.yaml")]
    assert "runtime rknn-fp16" in result.output
    assert (
        f"Commit {ARTIFACT} and {ARTIFACT}.json with m.onnx (in models/): the board can't "
        "convert, and the record names this exact .onnx, so the three travel together."
    ) in result.output
    assert "--runtime rknn-fp16 --protocol .calbredge/protocol.yaml --offline" in result.output
    # The converted model and its record are committed, so init must not ignore them.
    assert "rknn" not in (repo / ".gitignore").read_text()
    assert not (repo / ".github").exists()


@pytest.mark.parametrize("governor", ["powersave", "performance"])
def test_on_x86_the_clock_warning_does_not_stop_init(repo, runs, monkeypatch, tmp_path, governor):
    """doctor's real clock check on an x86 tree is a warning, and init goes ahead on it."""
    from calbredge import doctor

    from tests.open.fake_hosts import build_x86

    monkeypatch.setattr(init, "load_hooks", list)
    monkeypatch.setattr(init, "detected_platform", lambda: None)
    root = build_x86(tmp_path / "root", governor=governor)
    monkeypatch.setattr(init, "clock_check", lambda: doctor._clocks(root))
    assert init.clock_check().status == "warn"
    model = _model(repo / "models" / "m.onnx", ["N", 3, 224, 224])

    result = _init("--model", "models/m.onnx", "--runtime", "onnxruntime")

    assert result.exit_code == 0, result.output
    assert (repo / ".calbredge" / "project.yaml").is_file()
    assert runs == [(model, "onnxruntime", repo / ".calbredge" / "protocol.yaml")]


def test_only_a_clock_fail_stops_init(repo, runs, monkeypatch):
    monkeypatch.setattr(init, "load_hooks", list)
    monkeypatch.setattr(init, "detected_platform", lambda: None)
    _model(repo / "models" / "m.onnx", ["N", 3, 224, 224])
    monkeypatch.setattr(
        init, "clock_check", lambda: Check("clocks pinned", "fail", "unknown: x", "fix it")
    )

    result = _init("--model", "models/m.onnx", "--runtime", "onnxruntime")

    assert result.exit_code == init.EXIT_NOT_READY
    assert "clocks pinned: unknown: x. Nothing was written; fix it" in result.output
    assert _nothing_written(repo)
