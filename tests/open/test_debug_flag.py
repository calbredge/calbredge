"""Tests for the --debug flag on calbredge run and calbredge convert commands."""

import platform

from calbredge.cli import main
from click.testing import CliRunner


class TestDebugFlagHelp:
    def test_run_debug_flag_in_help(self):
        """--debug flag is documented in calbredge run --help."""
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--debug" in result.output

    def test_convert_debug_flag_in_help(self):
        """--debug flag is documented in calbredge convert --help."""
        runner = CliRunner()
        result = runner.invoke(main, ["convert", "--help"])
        assert result.exit_code == 0
        assert "--debug" in result.output


class TestRunDebugFlag:
    def test_debug_missing_model_prints_block_and_error(self):
        """--debug prints the debug block and error message when the model file is missing."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "--model", "nonexistent_model.onnx", "--dry-run", "--debug"],
        )

        # Run must fail (model not found)
        assert result.exit_code != 0

        # Debug block header and footer must be present
        assert "--- Calbredge Debug Info ---" in result.output
        assert "----------------------------" in result.output

        # Error message must appear in the block
        assert "error:" in result.output
        assert "nonexistent_model" in result.output or "No such file" in result.output

    def test_debug_block_contains_expected_fields(self):
        """Debug block includes hardware_id, python version, and onnxruntime version."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "--model", "nonexistent_model.onnx", "--dry-run", "--debug"],
        )

        # The block should always contain software environment fields
        assert "python version:" in result.output
        assert platform.python_version() in result.output

        assert "onnxruntime version:" in result.output

        # hardware_id line present (value may be "unavailable" if metadata failed,
        # but the label must be there)
        assert "hardware_id:" in result.output

        # Platform line
        assert "platform:" in result.output
