"""The header `calbredge run` prints: the protocol file, its input, and where an init copy
came from.
"""

import re

import pytest
from calbredge import cli
from click.testing import CliRunner

from tests.open.fake_hosts import conv_model

pytestmark = pytest.mark.unit


def _summary_lines(result) -> list[str]:
    return [
        line
        for line in result.stderr.splitlines()
        if not line.startswith(("Gate inputs", "Warning:"))
    ]


def test_the_run_header_names_the_protocol_file_and_input(tmp_path):
    protocol = tmp_path / "yolo-like.yaml"
    protocol.write_text(
        'protocol_version: "1.0"\nwarmup_iterations: 2\nmeasurement_iterations: 5\n'
        "batch_size: 1\ninput_shape: [1, 3, 8, 8]\n"
    )
    model = conv_model(tmp_path / "m.onnx")
    result = CliRunner().invoke(
        cli.main,
        [
            "run",
            "--model",
            str(model),
            "--runtime",
            "mock",
            "--protocol",
            str(protocol),
            "--cpu-affinity",
            "none",
            "--dry-run",
            "--no-bundle",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Protocol: yolo-like.yaml, input [1, 3, 8, 8], 2 warmup + 5 measured" in result.output
    assert not re.search(r"Protocol: v\d", result.output)


def test_the_run_header_names_where_an_init_copy_came_from(tmp_path):
    from calbredge.init import ORIGIN_PREFIX

    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        f"{ORIGIN_PREFIX}v1.0.yaml.\n"
        'protocol_version: "1.0"\nwarmup_iterations: 2\nmeasurement_iterations: 5\n'
        "batch_size: 1\ninput_shape: [1, 3, 8, 8]\n"
    )
    model = conv_model(tmp_path / "m.onnx")
    result = CliRunner().invoke(
        cli.main,
        ["run", "--model", str(model), "--runtime", "mock", "--protocol", str(protocol),
         "--cpu-affinity", "none", "--dry-run", "--no-bundle"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert (
        "Protocol: protocol.yaml (from v1.0.yaml), input [1, 3, 8, 8], 2 warmup + 5 measured"
        in result.output
    )
