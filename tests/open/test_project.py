"""The project file, .calbredge/project.yaml (agent/calbredge/project.py).

Strict loader, discovery by walking up from the current directory, the writer, and the
atomic update of a target's baseline pointer.
"""

import re
from pathlib import Path

import pytest
import yaml
from calbredge import project as project_module
from calbredge.project import (
    Project,
    ProjectError,
    Target,
    find_project_file,
    load_project,
    read_project,
    render_project,
    set_baseline,
    write_project,
)

pytestmark = pytest.mark.unit

VALID = {
    "schema_version": "1",
    "model": "models/m.onnx",
    "protocol": ".calbredge/protocol.yaml",
    "policy": ".calbredge/policy.yaml",
    "quality_evidence": None,
    "targets": [{"runtime": "tensorrt-fp16", "baseline": None}],
}


def _write(root: Path, data) -> Path:
    file = root / ".calbredge" / "project.yaml"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(data if isinstance(data, str) else yaml.safe_dump(data, sort_keys=False))
    return file


def _with(**changes) -> dict:
    return {**VALID, **changes}


# --- loading ----------------------------------------------------------------------------


def test_a_valid_file_loads(tmp_path):
    loaded = read_project(_write(tmp_path, VALID))
    assert loaded.root == tmp_path
    assert loaded.model == "models/m.onnx"
    assert loaded.target == Target(runtime="tensorrt-fp16", baseline=None)
    assert loaded.quality_evidence is None
    assert loaded.path(loaded.model) == tmp_path / "models" / "m.onnx"
    assert loaded.file == tmp_path / ".calbredge" / "project.yaml"


def test_quality_evidence_may_be_omitted(tmp_path):
    data = dict(VALID)
    del data["quality_evidence"]
    assert read_project(_write(tmp_path, data)).quality_evidence is None


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (_with(modle="x.onnx"), "unknown key(s) modle"),
        (
            _with(targets=[{"runtime": "tensorrt-fp16", "baselin": None}]),
            "unknown key(s) in targets[0]",
        ),
        ({k: v for k, v in VALID.items() if k != "protocol"}, "missing key(s) protocol"),
        (_with(schema_version=1), 'quoted whole number, e.g. "1"'),
        (_with(schema_version="one"), "quoted whole number"),
        (_with(model="/abs/m.onnx"), "relative to the project directory"),
        (_with(model=""), "model must be a path"),
        (_with(policy=True), "policy must be a path"),
        (_with(targets=[]), "targets must be a list with one entry"),
        (_with(targets={"runtime": "tensorrt-fp16"}), "targets must be a list"),
        (_with(targets=[{"runtime": "TensorRT FP16"}]), "must be a runtime label"),
        (_with(targets=[{"baseline": None}]), "must be a runtime label"),
        (_with(targets=[{"runtime": "tensorrt-fp16", "baseline": "/abs/b.yaml"}]), "relative"),
        ("- just\n- a list\n", "must be a YAML mapping"),
    ],
)
def test_invalid_files_are_refused(tmp_path, data, message):
    with pytest.raises(ProjectError, match=re.escape(message)):
        read_project(_write(tmp_path, data))


def test_a_newer_schema_says_upgrade(tmp_path):
    # A newer file may also use keys this version doesn't know: the version is checked first.
    data = _with(schema_version="2", targets_by_hardware=[])
    with pytest.raises(ProjectError, match="upgrade calbredge"):
        read_project(_write(tmp_path, data))


def test_more_targets_than_supported_says_upgrade(tmp_path):
    two = [{"runtime": "tensorrt-fp16"}, {"runtime": "tensorrt-fp32"}]
    with pytest.raises(ProjectError, match="upgrade calbredge for multi-target"):
        read_project(_write(tmp_path, _with(targets=two)))


def test_unreadable_yaml_is_a_project_error(tmp_path):
    with pytest.raises(ProjectError, match="could not read"):
        read_project(_write(tmp_path, "model: [unclosed\n"))


# --- discovery --------------------------------------------------------------------------


def test_the_file_is_found_by_walking_up(tmp_path, monkeypatch):
    file = _write(tmp_path, VALID)
    nested = tmp_path / "src" / "deep" / "er"
    nested.mkdir(parents=True)
    assert find_project_file(nested) == file
    monkeypatch.chdir(nested)
    assert load_project().root == tmp_path


def test_the_nearest_project_wins(tmp_path):
    _write(tmp_path, VALID)
    inner_file = _write(tmp_path / "sub", _with(model="inner.onnx"))
    (tmp_path / "sub" / "x").mkdir()
    assert find_project_file(tmp_path / "sub" / "x") == inner_file


def test_a_calbredge_dir_without_a_project_file_is_skipped(tmp_path):
    (tmp_path / "home" / ".calbredge" / "runs").mkdir(parents=True)  # the data directory
    with pytest.raises(ProjectError, match="run `calbredge init`"):
        find_project_file(tmp_path / "home")


def test_a_missing_file_says_run_init(tmp_path):
    with pytest.raises(
        ProjectError, match=r"no \.calbredge/project\.yaml in .* run `calbredge init`"
    ):
        load_project(tmp_path)


# --- writing ----------------------------------------------------------------------------


def _project(root, **changes) -> Project:
    fields = {
        "root": root,
        "model": "models/m.onnx",
        "protocol": ".calbredge/protocol.yaml",
        "policy": ".calbredge/policy.yaml",
        "targets": (Target("tensorrt-fp16"),),
        **changes,
    }
    return Project(**fields)


def test_a_project_without_a_policy_is_written_and_read(tmp_path):
    """The policy is optional: open init alone writes none."""
    written = _project(tmp_path, policy=None)
    file = write_project(written)
    assert "policy: null\n" in file.read_text()
    assert read_project(file) == written
    file.write_text(file.read_text().replace("policy: null\n", ""))
    assert read_project(file).policy is None


def test_the_writer_round_trips(tmp_path):
    written = _project(tmp_path, quality_evidence="qa/evidence.yaml")
    file = write_project(written)
    assert read_project(file) == written
    text = file.read_text()
    assert text.startswith("schema_version: '1'\nmodel: models/m.onnx\n")
    assert list(yaml.safe_load(text)) == list(VALID)


def test_the_writer_quotes_what_yaml_would_misread(tmp_path):
    tricky = _project(tmp_path, model="on", policy="1.0")
    assert read_project(write_project(tricky)) == tricky


def test_the_writer_never_overwrites_by_accident(tmp_path):
    write_project(_project(tmp_path))
    with pytest.raises(ProjectError, match="already exists"):
        write_project(_project(tmp_path, model="other.onnx"))
    write_project(_project(tmp_path, model="other.onnx"), overwrite=True)
    assert read_project(tmp_path / ".calbredge" / "project.yaml").model == "other.onnx"


def test_the_writer_refuses_an_invalid_project(tmp_path):
    with pytest.raises(ProjectError, match="relative"):
        write_project(_project(tmp_path, model="/abs.onnx"))
    assert not (tmp_path / ".calbredge" / "project.yaml").exists()


# --- the baseline pointer ---------------------------------------------------------------


def test_set_baseline_changes_only_the_pointer(tmp_path):
    file = write_project(_project(tmp_path, quality_evidence="qa.yaml"))
    before = read_project(file)
    updated = set_baseline(file, ".calbredge/baselines/b1.yaml")
    assert updated.target.baseline == ".calbredge/baselines/b1.yaml"
    assert read_project(file) == updated
    assert updated.model == before.model
    assert updated.quality_evidence == "qa.yaml"
    assert updated.target.runtime == before.target.runtime
    # A second approval moves the pointer again; nothing else is written or left behind.
    set_baseline(file, ".calbredge/baselines/b2.yaml")
    assert read_project(file).target.baseline == ".calbredge/baselines/b2.yaml"
    assert sorted(p.name for p in file.parent.iterdir()) == ["project.yaml"]


def test_set_baseline_rereads_the_file(tmp_path):
    file = write_project(_project(tmp_path))
    file.write_text(render_project(_project(tmp_path, policy="edited/policy.yaml")))
    assert set_baseline(file, "b.yaml").policy == "edited/policy.yaml"


def test_set_baseline_refuses_an_absolute_pointer(tmp_path):
    file = write_project(_project(tmp_path))
    with pytest.raises(ProjectError, match="relative"):
        set_baseline(file, "/abs/b.yaml")
    assert read_project(file).target.baseline is None


def test_a_failed_write_leaves_the_old_file_whole(tmp_path, monkeypatch):
    file = write_project(_project(tmp_path))
    original = file.read_bytes()

    def failing_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(project_module.os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        set_baseline(file, "b.yaml")
    assert file.read_bytes() == original
    assert sorted(p.name for p in file.parent.iterdir()) == ["project.yaml"]
