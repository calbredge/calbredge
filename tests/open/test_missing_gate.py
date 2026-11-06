"""`calbredge gate` and `calbredge baseline` without the release gate package installed.

The open CLI has neither command. It says which package provides the one typed and exits 2,
as click does for any unknown command. The message is raised in the unknown-command branch of
the command lookup, which no measurement reaches: a built-in command is found before it.
"""

from pathlib import Path

import click
import pytest
from calbredge.cli import _PluginGroup, main
from click.testing import CliRunner

pytestmark = pytest.mark.unit

FAST_PROTOCOL = str(Path(__file__).parent / "fixtures" / "protocol_fast.yaml")


@pytest.fixture
def no_gate(monkeypatch):
    """The open pair alone: no package registers a command."""
    monkeypatch.setattr(_PluginGroup, "_plugins", lambda self: {})


@pytest.mark.parametrize("args", [["gate"], ["baseline"], ["baseline", "approve", "--yes"]])
def test_a_gate_command_without_the_gate_names_the_package_and_exits_2(no_gate, args):
    done = CliRunner().invoke(main, args)
    assert done.exit_code == 2
    assert f"`calbredge {args[0]}` needs the release gate package, calbredge-gate" in done.output
    assert "not installed here" in done.output
    assert "No such command" not in done.output


def test_any_other_unknown_command_is_clicks_own_refusal(no_gate):
    done = CliRunner().invoke(main, ["gatee"])
    assert done.exit_code == 2
    assert "No such command 'gatee'" in done.output
    assert "calbredge-gate" not in done.output


def test_help_lists_no_gate_command_without_the_gate(no_gate):
    done = CliRunner().invoke(main, ["--help"])
    assert done.exit_code == 0
    assert set(main.list_commands(click.Context(main))) == set(
        click.Group.list_commands(main, click.Context(main))
    )


@pytest.fixture
def unknown_branch_forbidden(monkeypatch):
    """Fail if the lookup looks for a plugin or reaches the unknown-command branch."""

    def reached(*_args, **_kwargs):
        raise AssertionError("the command lookup went past the built-in commands")

    monkeypatch.setattr(_PluginGroup, "_plugins", reached)
    monkeypatch.setattr(_PluginGroup, "_unknown_command", reached)


def test_every_built_in_command_is_found_before_the_unknown_command_branch(
    unknown_branch_forbidden,
):
    ctx = click.Context(main)
    built_in = click.Group.list_commands(main, ctx)
    assert {"run", "doctor", "init", "inspect", "convert"} <= set(built_in)
    for name in built_in:
        assert main.get_command(ctx, name) is not None, name


def test_a_measurement_never_reaches_the_unknown_command_branch(unknown_branch_forbidden, tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake onnx bytes")
    done = CliRunner().invoke(
        main,
        [
            "run",
            "--model",
            str(model),
            "--runtime",
            "mock",
            "--protocol",
            FAST_PROTOCOL,
            "--dry-run",
        ],
    )
    assert done.exit_code == 0, done.output
    assert "p50" in done.output.lower()
