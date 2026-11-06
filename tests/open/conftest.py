"""Fixtures for the open tests, and the ones every test directory shares.

Nothing here imports the release gate or the API: tests/open runs with only calbredge-protocol
and calbredge installed. The other test directories import the two autouse fixtures below
and add their own.
"""

import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Test fixture provisioning
# ---------------------------------------------------------------------------
#
# tests/open/fixtures/*.onnx are gitignored — a
# fresh clone has none of them. Of the nine canonical models, exactly one,
# mobilenetv2.onnx, is actually loaded by this test suite (see
# tests/open/fixtures/README.md for how that was verified). No test asserts
# specific output values from it — only that a model with a
# (1, 3, 224, 224) float32 input loads and runs — so if the real ~14MB
# pretrained export isn't present locally, synthesize a tiny stand-in graph
# with the same input/output contract instead of failing the whole suite.
# This runs once, at collection time, before any test can touch the file.


def _ensure_mobilenetv2_fixture() -> None:
    dest = Path(__file__).parent / "fixtures" / "mobilenetv2.onnx"
    if dest.exists():
        return

    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    rng = np.random.default_rng(0)
    conv_w = (rng.standard_normal((8, 3, 3, 3)) * 0.05).astype(np.float32)
    gemm_w = (rng.standard_normal((1000, 8)) * 0.05).astype(np.float32)
    gemm_b = np.zeros(1000, dtype=np.float32)

    # Input name "pixel_values" and the symbolic "batch_size" dim (dim_param,
    # not a concrete int) both mirror the real, gitignored mobilenetv2.onnx
    # exactly — load-bearing for tests that assert on either (e.g.
    # test_onnx_adapter_dynamic_batch_stays_none, which requires
    # resolved_input_shape to stay None; test_symbolic_batch_dim_resolves_to_one
    # / test_symbolic_batch_dim_model_gets_concrete_override, which assert the
    # resolved input name is "pixel_values"). A concrete batch dim or a
    # different name here only "passes" locally because the real file masks
    # the stub — it fails the moment the real file is absent, as in CI.
    inputs = [
        helper.make_tensor_value_info(
            "pixel_values", TensorProto.FLOAT, ["batch_size", 3, 224, 224]
        )
    ]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1000])]
    initializers = [
        helper.make_tensor("conv_w", TensorProto.FLOAT, conv_w.shape, conv_w.flatten()),
        helper.make_tensor("gemm_w", TensorProto.FLOAT, gemm_w.shape, gemm_w.flatten()),
        helper.make_tensor("gemm_b", TensorProto.FLOAT, gemm_b.shape, gemm_b.flatten()),
    ]
    nodes = [
        helper.make_node(
            "Conv",
            ["pixel_values", "conv_w"],
            ["conv_out"],
            kernel_shape=[3, 3],
            strides=[2, 2],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node("Relu", ["conv_out"], ["relu_out"]),
        helper.make_node("GlobalAveragePool", ["relu_out"], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "gemm_w", "gemm_b"], ["output"], transB=1),
    ]
    graph = helper.make_graph(
        nodes, "synthetic_mobilenetv2_stand_in", inputs, outputs, initializer=initializers
    )
    model = helper.make_model(
        graph,
        producer_name="calbredge-test-fixture-synth",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)

    dest.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dest))


def _ensure_identity_fixture(filename: str, input_name: str, shape: list) -> None:
    """Synthesize a minimal, runnable ONNX stand-in with a single Identity
    node so tests that only inspect input signature (name/shape) via
    onnxruntime.InferenceSession — never real inference — can run without
    the real, gitignored fixture. shape entries may be ints (concrete dims)
    or strings (symbolic dim_param, e.g. a dynamic batch axis) per
    onnx.helper.make_tensor_value_info's convention.
    """
    dest = Path(__file__).parent / "fixtures" / filename
    if dest.exists():
        return

    import onnx
    from onnx import TensorProto, helper

    inputs = [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, shape)]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)]
    nodes = [helper.make_node("Identity", [input_name], ["output"])]

    graph = helper.make_graph(nodes, f"synthetic_{dest.stem}_stand_in", inputs, outputs)
    model = helper.make_model(
        graph,
        producer_name="calbredge-test-fixture-synth",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)

    dest.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dest))


_ensure_mobilenetv2_fixture()

# yolov8n.onnx's real declared input is [1, 3, 640, 640] — fully concrete,
# NCHW, no symbolic dimension (see tests/open/test_rknn_convert.py's YOLOV8N
# comment). Used to confirm the already-concrete/already-NCHW code paths are
# left untouched.
_ensure_identity_fixture("yolov8n.onnx", "images", [1, 3, 640, 640])

# cxr_mobilenetv2.onnx's real declared input is ['unk__654', 224, 224, 3] —
# a symbolic batch dimension AND an NHWC (channel-last) layout at once (see
# tests/open/test_rknn_convert.py's CXR_MOBILENETV2 comment). The input name
# "input_layer_1" is load-bearing — asserted directly by
# TestDetectNhwcImageInput.test_real_cxr_mobilenetv2_fixture_detected and
# TestResolveLoadOnnxInputs.test_nhwc_shape_only_resolves_batch_dim_no_longer_reorders.
_ensure_identity_fixture("cxr_mobilenetv2.onnx", "input_layer_1", ["unk__654", 224, 224, 3])


@pytest.fixture(autouse=True)
def _plain_terminal_env(monkeypatch):
    """Start every test with no colour settings from the environment running it.

    GitHub Actions sets GITHUB_ACTIONS=true, which calbredge.term treats as "style on", so
    output would carry colour codes in CI but not locally. Tests that exercise styling set
    these variables themselves (tests/open/test_term.py, tests/open/test_doctor.py).
    """
    for name in ("GITHUB_ACTIONS", "NO_COLOR", "FORCE_COLOR"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _isolate_v2_run_artifacts(tmp_path_factory, monkeypatch):
    """Keep every test out of the real ~/.calbredge.

    HOME points at a temporary directory, so anything that resolves
    ``Path.home()`` when it runs (the local results cache, ~/.calbredge/results)
    and every subprocess a test starts (it inherits HOME) stays out of the
    developer's home directory. Paths computed at import time, before HOME
    changed, are patched one by one: the run bundle directory, the contract and
    engine caches, and the config file. The config file matters
    most: a real one can hold an upload URL and API key, and a test that
    reached `run`'s upload step with it would upload to that server.
    """
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("calbredge.cli.CONFIG_PATH", home / ".calbredge" / "config.yaml")
    monkeypatch.setattr("calbredge.cli.DEFAULT_BUNDLE_DIR", tmp_path_factory.mktemp("runs"))
    cache = tmp_path_factory.mktemp("contract-cache")
    monkeypatch.setattr("calbredge.result_builder.DEFAULT_CONTRACT_CACHE_DIR", cache)
    monkeypatch.setattr("calbredge.inspect_v2.DEFAULT_CACHE_DIR", cache)
    monkeypatch.setattr(
        "calbredge_protocol.engine_cache.ENGINE_CACHE_FALLBACK_DIR",
        tmp_path_factory.mktemp("engine-cache"),
    )


# ---------------------------------------------------------------------------
# The open-only run
# ---------------------------------------------------------------------------
#
# CI's open job installs calbredge-protocol and calbredge, never calbredge-gate, sets
# CALBREDGE_OPEN_ONLY=1 and runs tests/open. These hooks fail that run, never skip, when the
# gate is reachable, when a test imported it, when a test skipped because it is missing, and
# when fewer tests are collected than CALBREDGE_OPEN_TESTS_MIN.

OPEN_ONLY_ENV = "CALBREDGE_OPEN_ONLY"
OPEN_TESTS_MIN_ENV = "CALBREDGE_OPEN_TESTS_MIN"
GATE_PACKAGE = "calbredge_gate"
GATE_DISTRIBUTION = "calbredge-gate"
EXTENSION_GROUPS = ("calbredge.commands", "calbredge.init_hooks")
_GATE_SKIPS: list[str] = []


def _open_only() -> bool:
    return os.environ.get(OPEN_ONLY_ENV) == "1"


def open_only_problems(collected: int | None) -> list[str]:
    """What is wrong with an open-only run, one line each; an empty list passes."""
    problems = []
    if importlib.util.find_spec(GATE_PACKAGE) is not None:
        problems.append(f"{GATE_PACKAGE} is importable: the open job must not install the gate")
    loaded = sorted(m for m in sys.modules if m.split(".")[0] == GATE_PACKAGE)
    if loaded:
        problems.append(f"a test imported gate code: {', '.join(loaded)}")
    for group in EXTENSION_GROUPS:
        owners = sorted(ep.value for ep in importlib.metadata.entry_points(group=group))
        if owners:
            problems.append(f"{group} has entry points without the gate: {', '.join(owners)}")
    problems += [f"skipped because the gate is missing: {skip}" for skip in _GATE_SKIPS]
    minimum = os.environ.get(OPEN_TESTS_MIN_ENV)
    if minimum is not None and collected is not None and collected < int(minimum):
        problems.append(
            f"{collected} tests collected, fewer than {minimum} ({OPEN_TESTS_MIN_ENV}): an open"
            " test was lost, or lower the number in the open job with the tests removed"
        )
    return problems


def pytest_runtest_logreport(report):
    if not _open_only() or not report.skipped or hasattr(report, "wasxfail"):
        return
    reason = str(report.longrepr[2] if isinstance(report.longrepr, tuple) else report.longrepr)
    if GATE_PACKAGE in reason or GATE_DISTRIBUTION in reason:
        _GATE_SKIPS.append(f"{report.nodeid}: {reason}")


def pytest_sessionfinish(session, exitstatus):
    if not _open_only():
        return
    # The count holds for the whole directory only, not for a run of one file or -k.
    whole = [Path(arg).resolve() for arg in session.config.args] == [Path(__file__).parent]
    problems = open_only_problems(session.testscollected if whole else None)
    if problems:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        for problem in problems:
            reporter.write_line(f"open-only run: {problem}", red=True)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
