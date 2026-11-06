"""`calbredge init` (agent/calbredge/init.py).

The clock check and the discovery run are replaced with fakes (no Jetson in CI), except in
one end-to-end test that measures with the mock runtime for real.
"""

import hashlib
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import calbredge.example_model as example_model
import pytest
from calbredge import cli, init
from calbredge.doctor import Check
from calbredge.project import read_project
from click.testing import CliRunner
from onnx import TensorProto, helper, save

pytestmark = pytest.mark.unit

FAST_PROTOCOL = Path(__file__).parent / "fixtures" / "protocol_fast.yaml"
_GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.invalid",
}


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args],
        env={**os.environ, **_GIT_ENV},
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout


def _model(path: Path, shape: list, inputs: int = 1) -> Path:
    """An Identity graph with the given input shape (named dims are dynamic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [f"x{i}" for i in range(inputs)]
    graph = helper.make_graph(
        [helper.make_node("Identity", [n], [f"y{i}"]) for i, n in enumerate(names)],
        "identity",
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, shape) for n in names],
        [helper.make_tensor_value_info(f"y{i}", TensorProto.FLOAT, shape) for i in range(inputs)],
    )
    save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), str(path))
    return path


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    """A fresh git repository on branch main, as the working directory."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    monkeypatch.chdir(root)
    for key, value in _GIT_ENV.items():
        monkeypatch.setenv(key, value)
    return root


@pytest.fixture(autouse=True)
def on_a_jetson(monkeypatch):
    """These tests stand for a Jetson (the clock check below says so); the RK3588 and the
    hosts that are neither are in tests/open/test_init_platform.py."""
    monkeypatch.setattr(init, "detected_platform", lambda: "jetson")


@pytest.fixture
def pinned(monkeypatch):
    monkeypatch.setattr(init, "clock_check", lambda: Check("clocks pinned", "ok", "yes (jetson)"))


@pytest.fixture
def runs(monkeypatch):
    """A fake discovery run; returns the list of calls."""
    calls = []

    def fake(model, runtime, protocol, status="stable"):
        calls.append((model, runtime, protocol))
        return SimpleNamespace(
            run_id="r-1",
            run_status=status,
            run_status_reason=None if status == "stable" else "p99/p50 ratio 3.1",
            latency_p50_ms=1.0123,
            runtime="tensorrt-fp16",
        )

    monkeypatch.setattr(init, "discovery_run", fake)
    return calls


def _init(*args: str):
    return CliRunner().invoke(cli.main, ["init", *args])


@pytest.mark.parametrize(
    ("dims", "expected"),
    [
        ([1, 3, 224, 224], "v1.0.yaml"),  # not protocol224.yaml: the default wins a tie
        (["batch_size", 3, 224, 224], "v1.0.yaml"),  # a dynamic batch measures batch 1
        ([1, 3, 320, 320], "protocol320.yaml"),
        ([1, 3, 640, 640], "yolo640.yaml"),
        ([1, 224, 224, 3], "protocol224_nhwc.yaml"),
        (["N", 320, 320, 3], "protocol320_nhwc.yaml"),
    ],
)
def test_the_protocol_matches_the_input_shape(dims, expected):
    name, shape = init.choose_protocol(dims)
    assert name == expected
    assert shape == [1, *dims[1:]]


@pytest.mark.parametrize(
    ("dims", "message"),
    [
        ([1, 3, "H", "W"], "dynamic dimensions besides the batch"),
        ([1, 3, 512, 512], "no bundled protocol has input shape"),
        ([2, 3, 224, 224], "no bundled protocol has input shape"),
        ([1, 3, 224], "no bundled protocol has input shape"),
    ],
)
def test_other_shapes_say_pass_protocol(dims, message):
    with pytest.raises(init.InitError, match=message) as refused:
        init.choose_protocol(dims)
    assert "pass --protocol" in refused.value.message


def test_the_remotes_default_branch_wins(repo):
    _git(repo, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/develop")
    assert init.default_branch(repo) == "develop"


def test_a_given_protocol_is_copied(repo, pinned, runs):
    _model(repo / "m.onnx", [1, 3, 224, 224])
    result = _init(
        "--model", "m.onnx", "--runtime", "tensorrt-fp16", "--protocol", str(FAST_PROTOCOL)
    )
    assert result.exit_code == 0, result.output
    copied = (repo / ".calbredge" / "protocol.yaml").read_text()
    assert copied == f"{init.ORIGIN_PREFIX}protocol_fast.yaml.\n{FAST_PROTOCOL.read_text()}"


def test_a_protocol_for_another_shape_is_refused(repo, pinned, runs):
    _model(repo / "m.onnx", [1, 3, 320, 320])
    result = _init(
        "--model", "m.onnx", "--runtime", "tensorrt-fp16", "--protocol", str(FAST_PROTOCOL)
    )
    assert result.exit_code == 1
    assert (
        "measures input [1, 3, 224, 224], but the model's input is [1, 3, 320, 320]"
        in result.output
    )
    assert not (repo / ".calbredge").exists()


def _nothing_written(repo: Path) -> None:
    assert not (repo / ".calbredge").exists()
    assert not (repo / ".github").exists()
    assert not (repo / "scripts").exists()


def test_unpinned_clocks_are_refused_with_the_fix(repo, runs, monkeypatch, tmp_path):
    monkeypatch.setattr(init, "_HOST_ROOT", tmp_path / "host")  # no boot unit installed
    monkeypatch.setattr(
        init,
        "clock_check",
        lambda: Check("clocks pinned", "fail", "no (jetson): timings would move", "x"),
    )
    _model(repo / "m.onnx", [1, 3, 224, 224])
    result = _init("--model", "m.onnx", "--runtime", "tensorrt-fp16")
    assert result.exit_code == init.EXIT_NOT_READY == 7
    assert "clocks pinned: no (jetson)" in result.output
    assert "sudo ./install.sh --pin-clocks-at-boot" in result.output
    assert "sudo jetson_clocks" in result.output
    assert "Nothing was written" in result.output
    _nothing_written(repo)
    assert runs == []


@pytest.mark.parametrize("unit", [False, True], ids=["no unit", "unit installed"])
def test_the_clock_fix_names_a_restart_when_the_unit_is_installed(
    repo, runs, monkeypatch, tmp_path, unit
):
    root = tmp_path / "host"
    if unit:
        path = root / "etc" / "systemd" / "system" / "calbredge-jetson-clocks.service"
        path.parent.mkdir(parents=True)
        path.write_text("[Service]\n")
    monkeypatch.setattr(init, "_HOST_ROOT", root)
    monkeypatch.setattr(
        init,
        "clock_check",
        lambda: Check("clocks pinned", "fail", "no (jetson): timings would move", "x"),
    )
    _model(repo / "m.onnx", [1, 3, 224, 224])
    result = _init("--model", "m.onnx", "--runtime", "tensorrt-fp16")
    assert result.exit_code == 7
    restart = "sudo systemctl restart calbredge-jetson-clocks.service"
    if unit:
        assert restart in result.output
        assert "needed after every power-mode change" in result.output
    else:
        assert restart not in result.output
        assert "sudo jetson_clocks" in result.output
        assert "sudo ./install.sh --pin-clocks-at-boot" in result.output
    _nothing_written(repo)


def test_outside_a_git_repository_is_refused(tmp_path, pinned, runs, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    _model(tmp_path / "m.onnx", [1, 3, 224, 224])
    result = _init("--model", "m.onnx", "--runtime", "tensorrt-fp16")
    assert result.exit_code == 1
    assert "not inside a git repository" in result.output
    assert "git init" in result.output
    assert not (tmp_path / ".calbredge").exists()


def test_a_model_outside_the_repository_is_refused(repo, tmp_path, pinned, runs):
    outside = _model(tmp_path / "elsewhere" / "m.onnx", [1, 3, 224, 224])
    result = _init("--model", str(outside), "--runtime", "tensorrt-fp16")
    assert result.exit_code == 1
    assert "outside the project directory" in result.output
    _nothing_written(repo)


@pytest.mark.parametrize(
    ("shape", "inputs", "message"),
    [
        ([1, 3, 512, 512], 1, "pass --protocol"),
        ([1, 3, 224, 224], 2, "has 2 inputs"),
    ],
)
def test_a_model_no_bundled_protocol_fits_is_refused(repo, pinned, runs, shape, inputs, message):
    _model(repo / "m.onnx", shape, inputs=inputs)
    result = _init("--model", "m.onnx", "--runtime", "tensorrt-fp16")
    assert result.exit_code == 1
    assert message in result.output
    _nothing_written(repo)


def test_model_or_example_exactly(repo, pinned, runs):
    assert _init().exit_code == 2
    _model(repo / "m.onnx", [1, 3, 224, 224])
    assert _init("--model", "m.onnx", "--example").exit_code == 2


def test_an_existing_project_needs_force(repo, pinned, runs):
    _model(repo / "m.onnx", [1, 3, 224, 224])
    assert _init("--model", "m.onnx", "--runtime", "tensorrt-fp16").exit_code == 0
    again = _init("--model", "m.onnx", "--runtime", "tensorrt-fp16")
    assert again.exit_code == 1
    assert "already exists; pass --force" in again.output
    assert len(runs) == 1
    assert _init("--model", "m.onnx", "--runtime", "tensorrt-fp32", "--force").exit_code == 0
    assert read_project(repo / ".calbredge" / "project.yaml").target.runtime == "tensorrt-fp32"


def test_a_different_protocol_file_needs_force(repo, pinned, runs):
    existing = repo / ".calbredge" / "protocol.yaml"
    existing.parent.mkdir(parents=True)
    existing.write_text(FAST_PROTOCOL.read_text())
    _model(repo / "m.onnx", [1, 3, 224, 224])
    result = _init("--model", "m.onnx", "--runtime", "tensorrt-fp16")
    assert result.exit_code == 1
    assert "protocol.yaml exists and differs" in result.output
    assert existing.read_text() == FAST_PROTOCOL.read_text()


def _serve(monkeypatch, content: bytes) -> None:
    monkeypatch.setattr(example_model, "MODEL_URL", "https://models.example/m.onnx")
    monkeypatch.setattr(example_model.time, "sleep", lambda s: None)

    class _Response:
        def __init__(self):
            self._chunks = [content, b""]

        def read(self, n):
            return self._chunks.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(example_model.urllib.request, "urlopen", lambda url, timeout: _Response())


def test_example_fetches_the_pinned_model_then_sets_up(repo, tmp_path, pinned, runs, monkeypatch):
    content = _model(tmp_path / "src.onnx", ["N", 3, 224, 224]).read_bytes()
    _serve(monkeypatch, content)
    monkeypatch.setattr(example_model, "MODEL_SHA256", hashlib.sha256(content).hexdigest())
    result = _init("--example", "--runtime", "tensorrt-fp16")
    assert result.exit_code == 0, result.output
    fetched = repo / "models" / "mobilenetv2-12.onnx"
    assert fetched.read_bytes() == content
    assert "downloaded, SHA-256" in result.stderr
    assert read_project(repo / ".calbredge" / "project.yaml").model == "models/mobilenetv2-12.onnx"
    assert runs[0][0].resolve() == fetched.resolve()


def test_example_with_the_wrong_bytes_writes_nothing(repo, tmp_path, pinned, runs, monkeypatch):
    _serve(monkeypatch, b"not the pinned model")
    result = _init("--example", "--runtime", "tensorrt-fp16")
    assert result.exit_code == 1
    assert "example model not fetched, nothing was written" in result.output
    assert not (repo / "models" / "mobilenetv2-12.onnx").exists()
    _nothing_written(repo)
    assert runs == []


def test_example_is_https_only(repo, pinned, runs, monkeypatch):
    monkeypatch.setattr(example_model, "MODEL_URL", "http://models.example/m.onnx")
    result = _init("--example", "--runtime", "tensorrt-fp16")
    assert result.exit_code == 1
    assert "only https" in result.output
    _nothing_written(repo)


def test_the_discovery_run_writes_a_bundle_like_run(repo, pinned):
    _model(repo / "m.onnx", [1, 3, 224, 224])
    result = _init("--model", "m.onnx", "--runtime", "mock", "--protocol", str(FAST_PROTOCOL))
    assert result.exit_code == 0, result.output
    assert "Run bundle: " in result.stderr
    bundles = [p for p in Path(cli.DEFAULT_BUNDLE_DIR).iterdir() if p.is_dir()]
    assert len(bundles) == 1
    assert (bundles[0] / "samples.jsonl").read_text().count("\n") == 10
    assert "discovery run" in result.stderr
    assert not (repo / ".calbredge" / "baselines").exists()


def _init_in(directory: Path, monkeypatch, *args: str):
    directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(directory)
    return _init(*args)


def test_a_project_inside_another_is_refused(repo, pinned, runs, monkeypatch):
    _model(repo / "a" / "m.onnx", [1, 3, 224, 224])
    assert (
        _init_in(
            repo / "a", monkeypatch, "--model", "m.onnx", "--runtime", "tensorrt-fp16"
        ).exit_code
        == 0
    )
    _model(repo / "a" / "sub" / "m.onnx", [1, 3, 224, 224])
    nested = _init_in(
        repo / "a" / "sub", monkeypatch, "--model", "m.onnx", "--runtime", "tensorrt-fp16"
    )
    assert nested.exit_code == 1
    assert f"inside the project in {(repo / 'a').resolve()}" in nested.output
    assert not (repo / "a" / "sub" / ".calbredge").exists()
    assert not (repo / ".github" / "workflows" / "calbredge-a-sub.yml").exists()
    forced = _init_in(
        repo / "a" / "sub",
        monkeypatch,
        "--model",
        "m.onnx",
        "--runtime",
        "tensorrt-fp16",
        "--force",
    )
    assert forced.exit_code == 0, forced.output


def test_a_project_around_another_is_refused(repo, pinned, runs, monkeypatch):
    _model(repo / "a" / "m.onnx", [1, 3, 224, 224])
    assert (
        _init_in(
            repo / "a", monkeypatch, "--model", "m.onnx", "--runtime", "tensorrt-fp16"
        ).exit_code
        == 0
    )
    _model(repo / "m.onnx", [1, 3, 224, 224])
    around = _init_in(repo, monkeypatch, "--model", "m.onnx", "--runtime", "tensorrt-fp16")
    assert around.exit_code == 1
    assert "already holds project(s) below it (a/.calbredge/project.yaml)" in around.output
    assert not (repo / ".calbredge").exists()


def test_the_model_must_be_in_the_project_directory(repo, pinned, runs, monkeypatch):
    _model(repo / "shared" / "m.onnx", [1, 3, 224, 224])
    result = _init_in(
        repo / "a", monkeypatch, "--model", "../shared/m.onnx", "--runtime", "tensorrt-fp16"
    )
    assert result.exit_code == 1
    assert "outside the project directory" in result.output
    assert not (repo / "a" / ".calbredge").exists()


def test_example_goes_into_the_project_directory(repo, tmp_path, pinned, runs, monkeypatch):
    content = _model(tmp_path / "src.onnx", ["N", 3, 224, 224]).read_bytes()
    _serve(monkeypatch, content)
    monkeypatch.setattr(example_model, "MODEL_SHA256", hashlib.sha256(content).hexdigest())
    result = _init_in(repo / "demo", monkeypatch, "--example", "--runtime", "tensorrt-fp16")
    assert result.exit_code == 0, result.output
    assert (repo / "demo" / "models" / "mobilenetv2-12.onnx").read_bytes() == content
    assert not (repo / "models").exists()


def test_commands_find_the_nearest_project_from_nested_folders(repo, pinned, runs, monkeypatch):
    from calbredge.project import load_project

    for sub in ("a", "b"):
        _model(repo / sub / "m.onnx", [1, 3, 224, 224])
        _init_in(repo / sub, monkeypatch, "--model", "m.onnx", "--runtime", "tensorrt-fp16")
    deep = repo / "b" / "src" / "deep"
    deep.mkdir(parents=True)
    assert load_project(deep).root == (repo / "b").resolve()
    assert load_project(repo / "a").root == (repo / "a").resolve()


_IGNORES = ["*.engine", "*.engine.json", "*.calbredge-contract.json", "calbredge-out/"]


def test_a_fresh_init_writes_the_gitignore(repo, pinned, runs):
    _model(repo / "m.onnx", [1, 3, 224, 224])
    result = _init("--model", "m.onnx", "--runtime", "tensorrt-fp16")
    assert result.exit_code == 0, result.output
    lines = (repo / ".gitignore").read_text().splitlines()
    assert lines[0].startswith("# Calbredge: derived files")
    assert lines[1:] == _IGNORES
    assert "  .gitignore  (created: engines, contract caches and CI output" in result.stderr


def test_rerunning_init_never_duplicates_the_ignores(repo, pinned, runs):
    _model(repo / "m.onnx", [1, 3, 224, 224])
    _init("--model", "m.onnx", "--runtime", "tensorrt-fp16")
    before = (repo / ".gitignore").read_text()
    again = _init("--model", "m.onnx", "--runtime", "tensorrt-fp16", "--force")
    assert again.exit_code == 0, again.output
    assert (repo / ".gitignore").read_text() == before
    assert "  .gitignore" not in again.stderr  # unchanged, so not listed


@pytest.mark.parametrize("user", ["node_modules/\n*.engine\n", "node_modules/\n*.engine"])
def test_a_users_gitignore_is_kept_and_only_extended(repo, pinned, runs, user):
    (repo / ".gitignore").write_text(user)
    _model(repo / "m.onnx", [1, 3, 224, 224])
    result = _init("--model", "m.onnx", "--runtime", "tensorrt-fp16")
    assert result.exit_code == 0, result.output
    text = (repo / ".gitignore").read_text()
    assert text.startswith("node_modules/\n*.engine\n")
    lines = text.splitlines()
    assert lines.count("*.engine") == 1
    assert [p for p in _IGNORES if p in lines] == _IGNORES
    assert "  .gitignore  (updated:" in result.stderr


def test_each_project_gets_its_own_gitignore(repo, pinned, runs, monkeypatch):
    _model(repo / "a" / "m.onnx", [1, 3, 224, 224])
    result = _init_in(repo / "a", monkeypatch, "--model", "m.onnx", "--runtime", "tensorrt-fp16")
    assert result.exit_code == 0, result.output
    assert (repo / "a" / ".gitignore").read_text().splitlines()[1:] == _IGNORES
    assert not (repo / ".gitignore").exists()
    assert "  a/.gitignore  (created:" in result.stderr


# --- paths the gate's tests used to be the only ones to reach ----------------------------


def test_without_a_remote_the_current_branch_is_the_default(repo):
    assert init.default_branch(repo) == "main"
    _git(repo, "checkout", "-q", "-b", "release")
    assert init.default_branch(repo) == "release"


def test_the_default_branch_is_main_when_git_cannot_run(repo, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(init.subprocess, "run", missing)
    assert init.default_branch(repo) == "main"


def test_a_protocol_that_does_not_load_is_refused(repo, pinned, runs):
    _model(repo / "m.onnx", [1, 3, 224, 224])
    (repo / "broken.yaml").write_text("protocol_version: [unclosed\n")
    result = _init("--model", "m.onnx", "--runtime", "tensorrt-fp16", "--protocol", "broken.yaml")
    assert result.exit_code == 1
    assert "could not load protocol broken.yaml" in result.output
    assert not (repo / ".calbredge").exists()
    assert runs == []


def test_a_runtime_no_adapter_knows_is_refused_before_anything_is_written(repo, pinned, runs):
    _model(repo / "m.onnx", [1, 3, 224, 224])
    result = _init("--model", "m.onnx", "--runtime", "hailort-fp16")
    assert result.exit_code == 1
    assert "hailort" in result.output
    assert not (repo / ".calbredge").exists()
    assert runs == []


def test_the_summary_says_when_the_run_recorded_another_runtime(repo, pinned, runs, monkeypatch):
    """The fake run reports tensorrt-fp16 whatever it is asked for."""
    monkeypatch.setattr(init, "load_hooks", list)
    _model(repo / "m.onnx", [1, 3, 224, 224])
    result = _init("--model", "m.onnx", "--runtime", "onnxruntime")
    assert result.exit_code == 0, result.output
    written = read_project(repo / ".calbredge" / "project.yaml").target.runtime
    assert written.startswith("onnxruntime")
    assert (
        f"WARN the run recorded runtime tensorrt-fp16, but project.yaml says {written}: "
        "measure the project with the project's runtime"
    ) in result.output


def test_the_clock_check_is_doctors():
    check = init.clock_check()
    assert check.name == "clocks pinned"
    assert check.status in ("ok", "warn", "fail")
