"""A project whose model file is git-ignored: init, approve and gate need only the file on disk.

Calbredge identifies the model by its SHA-256, not by git, so a model kept out of the
repository (fetched into place before the gate, e.g. by a CI step) works like a committed one.
Dirty-tree checks count modified tracked files only, so an ignored model, or a new one fetched
over it, never marks a run dirty. The docs rely on this (QUICKSTART, HOW_IT_WORKS).

init extends the project's .gitignore only after the discovery run: done before, a committed
.gitignore would be a modified tracked file while the run measured, the run would be recorded
dirty, and approve would refuse it.
"""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from calbredge import cli, init
from calbredge.doctor import Check
from calbredge.host_v2 import git_state
from calbredge.project import read_project
from calbredge_protocol.loader import load_protocol
from calbredge_protocol.models import protocol_digest
from click.testing import CliRunner

from tests.open.bundles import _bundle
from tests.open.test_init import _GIT_ENV, FAST_PROTOCOL, _git, _model

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def on_a_jetson(monkeypatch):
    """These tests stand for a Jetson, as their clock checks say: init detects its platform."""
    monkeypatch.setattr(init, "detected_platform", lambda: "jetson")


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    """A git repository whose committed .gitignore ignores models/."""
    root = tmp_path / "repo"
    root.mkdir()
    for key, value in _GIT_ENV.items():
        monkeypatch.setenv(key, value)
    _git(root, "init", "-q", "-b", "main")
    (root / ".gitignore").write_text("models/\n")
    _git(root, "add", ".gitignore")
    _git(root, "commit", "-q", "-m", "ignore models")
    return root


def _ignored(root: Path, rel: str) -> bool:
    done = subprocess.run(["git", "-C", str(root), "check-ignore", "-q", rel], check=False)
    return done.returncode == 0


def test_init_takes_an_ignored_model_and_its_discovery_run_is_clean(repo, monkeypatch):
    from calbredge import init
    from calbredge.doctor import Check

    monkeypatch.setattr(init, "clock_check", lambda: Check("clocks pinned", "ok", "yes (jetson)"))
    project_dir = repo / "detector"
    _model(project_dir / "models" / "m.onnx", [1, 3, 224, 224])
    assert _ignored(repo, "detector/models/m.onnx")
    monkeypatch.chdir(project_dir)

    result = CliRunner().invoke(
        cli.main,
        ["init", "--model", "models/m.onnx", "--runtime", "mock", "--protocol", str(FAST_PROTOCOL)],
    )
    assert result.exit_code == 0, result.output
    assert read_project(project_dir / ".calbredge" / "project.yaml").model == "models/m.onnx"
    [bundle] = [p for p in Path(cli.DEFAULT_BUNDLE_DIR).iterdir() if p.is_dir()]
    # The mock runtime's result isn't a valid v2 result (no precision), so it is kept as
    # result.invalid.json; its git fields are recorded all the same.
    candidate = json.loads((bundle / "result.invalid.json").read_text())["result"]["candidate"]
    assert candidate["git_commit"] is not None
    assert candidate["git_dirty"] is False


def test_replacing_an_ignored_model_never_makes_the_tree_dirty(repo):
    model = _model(repo / "models" / "m.onnx", [1, 3, 224, 224])
    commit, dirty = git_state(repo)
    assert commit is not None and dirty is False
    _model(model, [1, 3, 320, 320])  # a new candidate fetched over the old one
    assert git_state(repo) == (commit, False)


_INIT = ["init", "--model", "models/m.onnx", "--runtime", "tensorrt-fp16"]


@pytest.fixture
def init_host(monkeypatch):
    """Pinned clocks, as init's clock check sees them."""
    monkeypatch.setattr(init, "clock_check", lambda: Check("clocks pinned", "ok", "yes (jetson)"))
    monkeypatch.setenv("USER", "tester")


def _recording_run(tmp_path: Path, monkeypatch) -> list[tuple[str | None, bool | None]]:
    """A fake discovery run that records the git state at the moment it measures, as a bundle."""
    seen: list[tuple[str | None, bool | None]] = []

    def fake(model, runtime, protocol):
        commit, dirty = git_state(Path.cwd())
        seen.append((commit, dirty))
        _bundle(
            tmp_path / "runs",
            model,
            1,
            "2026-10-01T10:00:00Z",
            candidate={"git_commit": commit, "git_dirty": dirty},
            protocol={"protocol_digest": protocol_digest(load_protocol(protocol))},
        )
        return SimpleNamespace(
            run_id="r-1",
            run_status="stable",
            run_status_reason=None,
            latency_p50_ms=1.0,
            runtime="tensorrt-fp16",
        )

    monkeypatch.setattr(init, "discovery_run", fake)
    return seen


@pytest.mark.parametrize(
    "error",
    [RuntimeError("tensorrt failed"), KeyboardInterrupt()],
    ids=["RuntimeError", "KeyboardInterrupt"],
)
def test_a_failed_or_interrupted_discovery_run_still_updates_the_gitignore(
    repo, init_host, monkeypatch, error
):
    _model(repo / "models" / "m.onnx", [1, 3, 224, 224])
    monkeypatch.chdir(repo)

    def fail(model, runtime, protocol):
        raise error

    monkeypatch.setattr(init, "discovery_run", fail)
    result = CliRunner().invoke(cli.main, _INIT)
    assert result.exit_code == 1
    if isinstance(error, RuntimeError):
        assert "the discovery run failed: tensorrt failed" in result.output
    else:
        assert "Aborted!" in result.output
    lines = (repo / ".gitignore").read_text().splitlines()
    assert lines[0] == "models/"
    assert [p for p in init.IGNORED if p in lines] == list(init.IGNORED)
    assert (repo / ".calbredge" / "project.yaml").is_file()


def test_an_unwritable_gitignore_is_a_warning_naming_the_file_and_lines(
    tmp_path, repo, init_host, monkeypatch
):
    _model(repo / "models" / "m.onnx", [1, 3, 224, 224])
    monkeypatch.chdir(repo)
    _recording_run(tmp_path, monkeypatch)

    def refuse(project_dir):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(init, "update_gitignore", refuse)
    result = CliRunner().invoke(cli.main, _INIT)
    assert result.exit_code == 0, result.output
    assert f"could not update {repo / '.gitignore'}" in result.stderr
    assert "Add these lines to it by hand" in result.stderr
    assert " ".join(init.IGNORED) in result.stderr
    assert "  .gitignore" not in result.stderr
