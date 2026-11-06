"""Unit tests for calbredge convert CLI command.

All tests use mocks — no real model files, GPU, or optional dependencies
(tf2onnx, torch) are required.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from calbredge.cli import main
from click.testing import CliRunner

# ─── Help / registration tests ────────────────────────────────────────────────


class TestConvertRegistered:
    def test_convert_in_top_level_help(self):
        """'convert' subcommand appears in calbredge --help."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0, result.output
        assert "convert" in result.output

    def test_convert_help_lists_model_option(self):
        """convert --help shows --model option."""
        runner = CliRunner()
        result = runner.invoke(main, ["convert", "--help"])
        assert result.exit_code == 0, result.output
        assert "--model" in result.output

    def test_convert_help_lists_output_option(self):
        """convert --help shows --output option."""
        runner = CliRunner()
        result = runner.invoke(main, ["convert", "--help"])
        assert result.exit_code == 0, result.output
        assert "--output" in result.output

    def test_convert_help_lists_framework_option(self):
        """convert --help shows --framework option."""
        runner = CliRunner()
        result = runner.invoke(main, ["convert", "--help"])
        assert result.exit_code == 0, result.output
        assert "--framework" in result.output

    def test_convert_help_lists_input_shape_option(self):
        """convert --help shows --input-shape option."""
        runner = CliRunner()
        result = runner.invoke(main, ["convert", "--help"])
        assert result.exit_code == 0, result.output
        assert "--input-shape" in result.output

    def test_convert_description_in_help(self):
        """convert --help contains the expected description."""
        runner = CliRunner()
        result = runner.invoke(main, ["convert", "--help"])
        assert result.exit_code == 0, result.output
        assert "ONNX" in result.output


# ─── Missing input file ────────────────────────────────────────────────────────


class TestConvertMissingFile:
    def test_missing_model_file_exits_nonzero(self):
        """Non-existent --model path is rejected by Click path validation."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "convert",
                "--model",
                "does_not_exist.h5",
                "--output",
                "out.onnx",
                "--framework",
                "keras",
            ],
        )
        assert result.exit_code != 0

    def test_missing_output_arg_exits_nonzero(self):
        """Omitting --output causes a non-zero exit."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("model.h5").write_bytes(b"dummy")
            result = runner.invoke(
                main,
                ["convert", "--model", "model.h5", "--framework", "keras"],
            )
        assert result.exit_code != 0

    def test_missing_framework_arg_exits_nonzero(self):
        """Omitting --framework causes a non-zero exit."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("model.h5").write_bytes(b"dummy")
            result = runner.invoke(
                main,
                ["convert", "--model", "model.h5", "--output", "out.onnx"],
            )
        assert result.exit_code != 0

    def test_invalid_framework_exits_nonzero(self):
        """An unknown framework name causes a non-zero exit."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("model.h5").write_bytes(b"dummy")
            result = runner.invoke(
                main,
                [
                    "convert",
                    "--model",
                    "model.h5",
                    "--output",
                    "out.onnx",
                    "--framework",
                    "caffe",
                ],
            )
        assert result.exit_code != 0


# ─── PyTorch input-shape validation ───────────────────────────────────────────


class TestConvertPyTorchInputShape:
    def test_pytorch_requires_input_shape(self):
        """--input-shape is required for pytorch; omitting it gives a usage error."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("model.pth").write_bytes(b"dummy")
            result = runner.invoke(
                main,
                [
                    "convert",
                    "--model",
                    "model.pth",
                    "--output",
                    "out.onnx",
                    "--framework",
                    "pytorch",
                ],
            )
        assert result.exit_code != 0
        assert "input-shape" in result.output.lower() or "input_shape" in result.output.lower()

    def test_pytorch_invalid_input_shape_exits_nonzero(self):
        """Non-integer --input-shape string is rejected."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("model.pth").write_bytes(b"dummy")
            # Patch torch to avoid missing-dep exit before shape validation
            torch_mock = MagicMock()
            with patch.dict(sys.modules, {"torch": torch_mock}):
                result = runner.invoke(
                    main,
                    [
                        "convert",
                        "--model",
                        "model.pth",
                        "--output",
                        "out.onnx",
                        "--framework",
                        "pytorch",
                        "--input-shape",
                        "1,three,224,224",
                    ],
                )
        assert result.exit_code != 0


# ─── Missing dependency handling ─────────────────────────────────────────────


class TestConvertMissingDependencies:
    def test_missing_tf2onnx_prints_pip_install(self):
        """Missing tf2onnx prints 'pip install tf2onnx' and exits with code 1."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("model.h5").write_bytes(b"dummy")
            # Block tf2onnx by setting to None in sys.modules
            blocked = {"tf2onnx": None, "tf2onnx.convert": None}
            with patch.dict(sys.modules, blocked):
                result = runner.invoke(
                    main,
                    [
                        "convert",
                        "--model",
                        "model.h5",
                        "--output",
                        "out.onnx",
                        "--framework",
                        "keras",
                    ],
                )
        assert "pip install tf2onnx" in result.output
        assert result.exit_code == 1

    def test_missing_torch_prints_pip_install(self):
        """Missing torch prints 'pip install torch' and exits with code 1."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("model.pth").write_bytes(b"dummy")
            with patch.dict(sys.modules, {"torch": None}):
                result = runner.invoke(
                    main,
                    [
                        "convert",
                        "--model",
                        "model.pth",
                        "--output",
                        "out.onnx",
                        "--framework",
                        "pytorch",
                        "--input-shape",
                        "1,3,224,224",
                    ],
                )
        assert "pip install torch" in result.output
        assert result.exit_code == 1

    def test_missing_tensorflow_prints_pip_install(self):
        """Missing tensorflow (with tf2onnx present) prints install message."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("model.h5").write_bytes(b"dummy")
            # tf2onnx present but tensorflow missing
            tf2onnx_mock = MagicMock()
            with patch.dict(
                sys.modules,
                {
                    "tf2onnx": tf2onnx_mock,
                    "tf2onnx.convert": tf2onnx_mock.convert,
                    "tensorflow": None,
                },
            ):
                result = runner.invoke(
                    main,
                    [
                        "convert",
                        "--model",
                        "model.h5",
                        "--output",
                        "out.onnx",
                        "--framework",
                        "keras",
                    ],
                )
        assert "pip install tensorflow" in result.output
        assert result.exit_code == 1


# ─── Successful conversion (fully mocked) ────────────────────────────────────


class TestConvertSuccessMocked:
    def test_keras_conversion_success_path(self):
        """Full keras conversion path succeeds when all deps are mocked."""
        runner = CliRunner()

        # Build a minimal fake Keras model
        fake_model = MagicMock()
        fake_model.input_shape = (None, 224, 224, 3)
        fake_model.output_shape = (None, 10)

        fake_tf = MagicMock()
        fake_tf.keras.models.load_model.return_value = fake_model
        fake_tf.keras.backend.image_data_format.return_value = "channels_last"

        fake_proto = MagicMock()
        fake_proto.SerializeToString.return_value = b"\x00" * 16

        fake_tf2onnx = MagicMock()
        fake_tf2onnx.convert.from_keras.return_value = (fake_proto, None)

        fake_onnx_model = MagicMock()
        fake_onnx = MagicMock()
        fake_onnx.load.return_value = fake_onnx_model
        fake_onnx.checker.check_model.return_value = None  # no exception = valid

        with runner.isolated_filesystem():
            Path("model.h5").write_bytes(b"dummy")
            with patch.dict(
                sys.modules,
                {
                    "tf2onnx": fake_tf2onnx,
                    "tf2onnx.convert": fake_tf2onnx.convert,
                    "tensorflow": fake_tf,
                    "tensorflow.keras": fake_tf.keras,
                    "onnx": fake_onnx,
                },
            ):
                result = runner.invoke(
                    main,
                    [
                        "convert",
                        "--model",
                        "model.h5",
                        "--output",
                        "out.onnx",
                        "--framework",
                        "keras",
                    ],
                )

        assert result.exit_code == 0, result.output
        assert "✓" in result.output
        assert "out.onnx" in result.output

    def test_pytorch_conversion_success_path(self):
        """Full pytorch conversion path succeeds when all deps are mocked."""
        runner = CliRunner()

        fake_module = MagicMock(spec=["eval"])
        fake_module.eval.return_value = fake_module

        fake_torch = MagicMock()
        # Make jit.load fail so we fall through to torch.load
        fake_torch.jit.load.side_effect = Exception("not torchscript")
        fake_torch.load.return_value = fake_module
        fake_torch.nn.Module = object  # isinstance check: fake_module is not a Module
        fake_torch.zeros.return_value = MagicMock()

        # Make isinstance(loaded, torch.nn.Module) return True
        fake_torch.nn.Module = type(fake_module)

        fake_onnx_output = MagicMock()
        fake_onnx_output.graph.output[0].type.tensor_type.shape.dim = [
            MagicMock(dim_value=1),
            MagicMock(dim_value=10),
        ]
        fake_onnx = MagicMock()
        fake_onnx.load.return_value = fake_onnx_output
        fake_onnx.checker.check_model.return_value = None

        with runner.isolated_filesystem():
            Path("model.pth").write_bytes(b"dummy")
            with patch.dict(
                sys.modules,
                {"torch": fake_torch, "onnx": fake_onnx},
            ):
                result = runner.invoke(
                    main,
                    [
                        "convert",
                        "--model",
                        "model.pth",
                        "--output",
                        "out.onnx",
                        "--framework",
                        "pytorch",
                        "--input-shape",
                        "1,3,224,224",
                    ],
                )

        assert result.exit_code == 0, result.output
        assert "✓" in result.output
        assert "out.onnx" in result.output
