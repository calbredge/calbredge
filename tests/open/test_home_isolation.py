"""The autouse isolation fixture (tests/open/conftest.py) keeps tests out of the real ~/.calbredge.

The real home directory is read from the password database, not HOME, so
these checks see through the fixture's own redirection.
"""

import json
import os
import pwd
import subprocess
import sys
from pathlib import Path

import pytest
from calbredge import cli
from click.testing import CliRunner

from tests.open.fake_hosts import conv_model

pytestmark = pytest.mark.unit

REAL_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir)
FAST_PROTOCOL = str(Path(__file__).parent / "fixtures" / "protocol_fast.yaml")


def _run_args(model: Path) -> list[str]:
    return ["run", "--model", str(model), "--runtime", "mock", "--protocol", FAST_PROTOCOL]


def _assert_saved_only_under_test_home(run_id: str) -> None:
    assert (Path.home() / ".calbredge" / "results" / f"{run_id}.json").is_file()
    assert not (REAL_HOME / ".calbredge" / "results" / f"{run_id}.json").exists()


def test_home_is_redirected():
    assert Path.home() != REAL_HOME
    assert not cli.CONFIG_PATH.is_relative_to(REAL_HOME / ".calbredge")


def test_the_real_config_is_never_read():
    """A real config can hold an upload URL and API key; tests must not see it."""
    assert cli._load_config() == {}


def test_in_process_run_saves_results_under_the_test_home(tmp_path):
    result = CliRunner().invoke(
        cli.main, [*_run_args(conv_model(tmp_path / "m.onnx")), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    saved = list((Path.home() / ".calbredge" / "results").glob("*.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text())["runtime"] == "mock"
    _assert_saved_only_under_test_home(saved[0].stem)


def test_subprocess_run_saves_results_under_the_test_home(tmp_path):
    """Subprocesses inherit HOME, so they are isolated too."""
    result = subprocess.run(
        [
            sys.executable, "-c", "from calbredge.cli import main; main()",
            *_run_args(conv_model(tmp_path / "m.onnx")), "--offline",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    saved = list((Path.home() / ".calbredge" / "results").glob("*.json"))
    assert saved, "the subprocess saved no result under the test HOME"
    for path in saved:
        _assert_saved_only_under_test_home(path.stem)
