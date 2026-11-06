"""`calbredge init` without a hook, and how it treats one (the ``calbredge.init_hooks`` group).

Without calbredge-gate, init writes only the open files; the gate's own hook is tested with
the gate.
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from calbredge import cli, init
from calbredge.doctor import Check
from calbredge.project import read_project
from click.testing import CliRunner
from onnx import TensorProto, helper, save

pytestmark = pytest.mark.unit

_GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.invalid",
}


def _model(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = [1, 3, 224, 224]
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "identity",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )
    save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), str(path))
    return path


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    """A fresh git repository on branch main, as the working directory, on a pinned Jetson."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "-C", str(root), "init", "-q", "-b", "main"], check=True)
    monkeypatch.chdir(root)
    for key, value in _GIT_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(init, "detected_platform", lambda: "jetson")
    monkeypatch.setattr(init, "clock_check", lambda: Check("clocks pinned", "ok", "yes (jetson)"))
    return root


def _run(monkeypatch, status: str = "stable") -> None:
    monkeypatch.setattr(
        init,
        "discovery_run",
        lambda model, runtime, protocol: SimpleNamespace(
            run_id="r-1",
            run_status=status,
            run_status_reason=None if status == "stable" else "p99/p50 ratio 3.1",
            latency_p50_ms=1.0123,
            runtime="tensorrt-fp16",
        ),
    )


def _init(*args: str):
    return CliRunner().invoke(cli.main, ["init", *args])


def _files(root: Path) -> list[str]:
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and ".git/" not in p.relative_to(root).as_posix() + "/"
    )


def test_an_open_refusal_after_the_hooks_check_writes_nothing(repo, monkeypatch):
    """Unpinned clocks are checked after the hook has agreed: its files aren't written either."""
    _run(monkeypatch)
    monkeypatch.setattr(init, "clock_check", lambda: Check("clocks pinned", "fail", "no (jetson)"))
    monkeypatch.setattr(init, "pin_fix", lambda: "pin the clocks first")
    _model(repo / "m.onnx")
    result = _init("--model", "m.onnx", "--runtime", "tensorrt-fp16")
    assert result.exit_code == init.EXIT_NOT_READY
    assert _files(repo) == ["m.onnx"]


_OPEN_SUMMARY = """
Wrote the project (paths from the repository root):
  .calbredge/project.yaml  (runtime tensorrt-fp16)
  .calbredge/protocol.yaml  (v1.0.yaml, input [1, 3, 224, 224])
  .gitignore  (created: engines, contract caches and CI output are never committed)

OK discovery run r-1: stable, p50 1.012 ms, runtime tensorrt-fp16
The project is measured. Measure it again with: calbredge run --model m.onnx --runtime tensorrt-fp16 --protocol .calbredge/protocol.yaml --offline
"""  # noqa: E501


@pytest.fixture
def no_hooks(monkeypatch):
    monkeypatch.setattr(init, "load_hooks", list)


def test_without_the_gate_init_writes_the_open_files_only(repo, monkeypatch, no_hooks):
    _run(monkeypatch)
    _model(repo / "m.onnx")
    result = _init("--model", "m.onnx", "--runtime", "tensorrt-fp16")
    assert result.exit_code == 0, result.output
    assert result.stderr == _OPEN_SUMMARY
    assert _files(repo) == [
        ".calbredge/project.yaml",
        ".calbredge/protocol.yaml",
        ".gitignore",
        "m.onnx",
        "m.onnx.calbredge-contract.json",
    ]
    project = read_project(repo / ".calbredge" / "project.yaml")
    assert project.policy is None
    assert project.quality_evidence is None
    assert project.target.baseline is None
    assert "policy: null\n" in project.file.read_text()


def test_without_the_gate_an_unstable_run_ends_with_the_same_line(repo, monkeypatch, no_hooks):
    _run(monkeypatch, "unstable")
    _model(repo / "m.onnx")
    result = _init("--model", "m.onnx", "--runtime", "tensorrt-fp16")
    assert result.exit_code == 0, result.output
    assert "WARN discovery run r-1: unstable (p99/p50 ratio 3.1)" in result.stderr
    assert result.stderr.endswith(_OPEN_SUMMARY.rsplit("\n", 2)[1] + "\n")
    assert "baseline" not in result.stderr


def test_without_the_gate_an_existing_project_names_nothing_else(repo, monkeypatch, no_hooks):
    _run(monkeypatch)
    _model(repo / "m.onnx")
    assert _init("--model", "m.onnx", "--runtime", "tensorrt-fp16").exit_code == 0
    again = _init("--model", "m.onnx", "--runtime", "tensorrt-fp16")
    assert again.exit_code == 1
    assert again.output == (
        f"Error: {repo / '.calbredge' / 'project.yaml'} already exists; pass --force to write "
        "it again\n"
    )


def test_without_the_gate_a_host_that_is_neither_target_needs_no_platform(
    repo, monkeypatch, no_hooks
):
    monkeypatch.setattr(init, "detected_platform", lambda: None)
    _run(monkeypatch)
    _model(repo / "m.onnx")
    result = _init("--model", "m.onnx", "--runtime", "onnxruntime")
    assert result.exit_code == 0, result.output
    assert not (repo / ".github").exists()


def test_without_the_gate_the_branch_and_platform_flags_are_accepted_and_unused(
    repo, monkeypatch, no_hooks
):
    _run(monkeypatch)
    _model(repo / "m.onnx")
    result = _init(
        "--model", "m.onnx", "--runtime", "tensorrt-fp16", "--branch", "x y", "--platform", "rk3588"
    )
    assert result.exit_code == 0, result.output
    assert not (repo / ".github").exists()


def test_a_broken_hook_stops_init_before_anything_is_written(repo, monkeypatch):
    def broken():
        raise ImportError("no module named calbredge_gate.invented_hook")

    monkeypatch.setattr(init, "load_hooks", broken)
    _run(monkeypatch)
    _model(repo / "m.onnx")
    result = _init("--model", "m.onnx", "--runtime", "tensorrt-fp16")
    assert result.exit_code == 1
    assert isinstance(result.exception, ImportError)
    assert _files(repo) == ["m.onnx"]


def test_importing_open_init_loads_no_gate_module():
    code = (
        "import sys, calbredge.init; "
        "print([m for m in sys.modules if m.split('.')[0] == 'calbredge_gate'])"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"


class _NotesHook:
    """A hook as another package would write one: a file of its own, a project field, a line."""

    replaced_by_force = "the notes are replaced too"

    def __init__(self) -> None:
        self.checked: list[init.InitRequest] = []

    def check(self, request: init.InitRequest) -> None:
        self.checked.append(request)

    def write(self, request: init.InitRequest, outcome: init.Outcome) -> dict[str, str]:
        for rel, note in ((".calbredge/policy.yaml", "limits to edit"), ("NOTES.md", "")):
            init._put(
                request.project_dir,
                rel,
                "written by the hook\n",
                force=request.force,
                outcome=outcome,
                shown=rel,
                note=note,
            )
        return {"policy": ".calbredge/policy.yaml"}

    def closing(self, outcome: init.Outcome, result) -> list[str]:
        return [f"Next: approve run {result.run_id}."]


def test_a_hooks_files_field_and_closing_lines_are_used(repo, monkeypatch):
    """What init does with any hook, shown on one that is not the gate's."""
    hook = _NotesHook()
    monkeypatch.setattr(init, "load_hooks", lambda: [hook])
    _run(monkeypatch)
    _model(repo / "m.onnx")
    (repo / "NOTES.md").write_text("the user's own\n")

    result = _init("--model", "m.onnx", "--runtime", "tensorrt-fp16", "--branch", "release")

    assert result.exit_code == 0, result.output
    [request] = hook.checked
    assert (request.project_dir, request.branch, request.force) == (repo, "release", False)
    # The hook's file is written and named in the project file, with its note in the summary.
    assert (repo / ".calbredge" / "policy.yaml").read_text() == "written by the hook\n"
    assert read_project(repo / ".calbredge" / "project.yaml").policy == ".calbredge/policy.yaml"
    assert "  .calbredge/policy.yaml  (limits to edit)\n" in result.stderr
    # A file that was already there is kept, and the summary says so.
    assert (repo / "NOTES.md").read_text() == "the user's own\n"
    assert "Kept (already there; --force replaces them):\n  NOTES.md\n" in result.stderr
    # The hook's lines close the summary, in place of the open one.
    assert result.stderr.rstrip().endswith("Next: approve run r-1.")
    assert "The project is measured." not in result.stderr
