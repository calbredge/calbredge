"""The runtime probe (agent/calbredge/runtime_probe.py).

The parent side is tested against small child scripts that behave as a runtime does: one
prints librknnrt's unsupported-operator line, one dies on a signal, one hangs. The child's own
steps are tested in-process on a fake adapter.
"""

import subprocess
import sys

import pytest
from calbredge import runtime_probe
from calbredge.runtime_probe import ProbeResult, needs_probe, probe

pytestmark = pytest.mark.unit

# What librknnrt 2.3.2 prints on stderr, nine times, at init_runtime for rtdetr_s (2026-10-05).
GRIDSAMPLE_LINE = (
    "E RKNN: [18:43:19.369] Unsupport CPU op: GridSample in this librknnrt.so, please try to "
    "register custom op by callingrknn_register_custom_ops or If using rknn, update to the "
    "latest toolkit2 and runtime from: https://console.zbox.filez.com/l/I00fc3 (PWD: rknn)."
)
PRINT_UNSUPPORTED = f"import sys\nfor _ in range(9): print({GRIDSAMPLE_LINE!r}, file=sys.stderr)\n"
SEGFAULT = "import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)\n"


def _run(code: str, **kwargs) -> ProbeResult:
    return probe(None, "rknn-fp16", None, "big", command=[sys.executable, "-c", code], **kwargs)


@pytest.mark.parametrize(
    ("label", "probed"),
    [("rknn-fp16", True), ("rknn-int8", True), ("tensorrt-fp16", False), ("mock", False)],
)
def test_only_rknn_is_probed(label, probed):
    assert needs_probe(label) is probed


@pytest.mark.parametrize("stream", ["stderr", "stdout"])
def test_unsupported_op_lines_name_the_op_once(stream):
    """Both of the child's streams are scanned: librknnrt 2.3.2 uses stderr, and nothing
    promises another version will."""
    result = _run(PRINT_UNSUPPORTED.replace("sys.stderr", f"sys.{stream}"))
    assert result.unsupported_ops == ("GridSample",)
    assert result.crashed is None


@pytest.mark.parametrize("stream", ["stderr", "stdout"])
def test_unsupported_op_lines_are_found_when_the_child_hangs(stream):
    code = PRINT_UNSUPPORTED.replace("sys.stderr", f"sys.{stream}")
    code += f"sys.{stream}.flush()\nimport time\ntime.sleep(30)\n"
    result = _run(code, timeout_s=3)
    assert result.crashed == "timed out after 3 s"
    assert result.unsupported_ops == ("GridSample",)


def test_an_unsupported_op_is_still_reported_when_the_child_then_crashes():
    """rtdetr_s: the load's own canary inference segfaults after the lines are printed."""
    result = _run(PRINT_UNSUPPORTED + SEGFAULT)
    assert result.crashed == "killed by SIGSEGV"
    assert result.unsupported_ops == ("GridSample",)


def test_two_unsupported_ops_are_both_named_in_order():
    code = "import sys\n" + "".join(
        f"print('E RKNN: Unsupport CPU op: {op} in this librknnrt.so', file=sys.stderr)\n"
        for op in ("GridSample", "Einsum", "GridSample")
    )
    assert _run(code).unsupported_ops == ("GridSample", "Einsum")


def test_a_child_killed_by_a_signal_is_a_probe_crash():
    result = _run(SEGFAULT)
    assert result.unsupported_ops == ()
    assert result.crashed == "killed by SIGSEGV"
    assert result.returncode is None


def test_a_child_that_hangs_is_a_probe_crash():
    result = _run("import time\ntime.sleep(30)\n", timeout_s=0.5)
    assert result.unsupported_ops == ()
    assert result.crashed == "timed out after 0.5 s"
    assert result.seconds < 10


def test_a_clean_child_reports_nothing_wrong():
    result = _run("print('ok')\n")
    assert (result.unsupported_ops, result.crashed, result.returncode) == ((), None, 0)
    assert "ok" in result.output


def test_an_ordinary_python_error_in_the_child_is_neither_finding():
    """The caller's in-process load then raises the same error it has always reported."""
    result = _run("raise FileNotFoundError('no .rknn beside the model')\n")
    assert (result.unsupported_ops, result.crashed, result.returncode) == ((), None, 1)
    assert "FileNotFoundError" in result.output


def test_the_child_command_carries_the_gates_own_arguments(tmp_path, monkeypatch):
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"], seen["kwargs"] = argv, kwargs
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(runtime_probe.subprocess, "run", fake_run)
    model, protocol = tmp_path / "m.onnx", tmp_path / "p.yaml"
    probe(model, "rknn-fp16", protocol, "4,5")
    argv = seen["argv"]
    assert argv[:3] == [sys.executable, "-c", runtime_probe._CHILD]
    assert argv[3:] == [
        "--model", str(model), "--runtime", "rknn-fp16", "--cpu-affinity", "4,5",
        "--protocol", str(protocol),
    ]  # fmt: skip
    assert seen["kwargs"]["timeout"] == runtime_probe.PROBE_TIMEOUT_S

    probe(model, "rknn-fp16", None, "big")
    assert "--protocol" not in seen["argv"]


def test_the_child_imports_the_installed_package_not_a_directory_named_calbredge(tmp_path):
    """Started from a directory that holds a ``calbredge`` folder (a checkout in the home
    directory), the child must still find the installed package."""
    (tmp_path / "calbredge").mkdir()
    done = subprocess.run(
        [sys.executable, "-c", runtime_probe._CHILD, "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert "calbredge.runtime_probe" in done.stdout


def test_the_child_pins_then_loads_then_runs_one_inference(tmp_path, monkeypatch):
    import numpy as np

    events: list = []

    class _Adapter:
        input_names = ("0",)
        input_dtype = np.dtype("uint8")

        def load_model(self, model, input_shape):
            events.append(("load", model, tuple(input_shape)))

        def run_inference(self, inputs):
            ((name, array),) = inputs.items()
            events.append(("inference", name, array.shape, str(array.dtype)))

        def unload(self):
            events.append("unload")

    monkeypatch.setattr("calbredge.affinity.detect_clusters", lambda: {"big": [4, 5]})
    monkeypatch.setattr(
        "calbredge.affinity.resolve_affinity", lambda spec, clusters: (spec, clusters)
    )
    monkeypatch.setattr(
        "calbredge.affinity.apply_affinity", lambda cores: events.append(("pin", cores))
    )
    monkeypatch.setattr("calbredge_protocol.autodetect.get_adapter", lambda runtime: _Adapter())
    protocol = tmp_path / "p.yaml"
    protocol.write_text(
        'protocol_version: "1.0"\nwarmup_iterations: 1\nmeasurement_iterations: 1\n'
        "input_shape: [1, 3, 8, 8]\n"
    )
    model = tmp_path / "m.onnx"

    argv = ["--model", str(model), "--runtime", "rknn-fp16", "--protocol", str(protocol)]
    code = runtime_probe.main([*argv, "--cpu-affinity", "big"])

    assert code == 0
    assert events == [
        ("pin", ("big", {"big": [4, 5]})),
        ("load", model, (1, 3, 8, 8)),
        ("inference", "0", (1, 3, 8, 8), "uint8"),
        "unload",
    ]


def test_output_and_signal_fallbacks():
    """A timeout hands back bytes or nothing, and a signal number may have no name."""
    assert runtime_probe._text(None) == ""
    assert runtime_probe._text(b"E RKNN: \xff") == "E RKNN: \ufffd"
    assert runtime_probe._text("already text") == "already text"
    assert runtime_probe._signal_name(11) == "SIGSEGV"
    assert runtime_probe._signal_name(250) == "signal 250"


def test_the_probe_reports_how_its_child_ended(tmp_path):
    def child(code: str) -> ProbeResult:
        return runtime_probe.probe(
            tmp_path / "m.onnx", "rknn-fp16", None, "none", command=[sys.executable, "-c", code]
        )

    assert child("print('ok')\n").returncode == 0
    assert child("raise ValueError('no')\n").returncode == 1
    killed = child("import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n")
    assert (killed.returncode, killed.crashed) == (None, "killed by SIGKILL")
