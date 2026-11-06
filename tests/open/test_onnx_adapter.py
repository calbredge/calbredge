"""The ONNX Runtime adapter feeds the shape the result records.

A dynamic input dimension takes the protocol's value, the rule the TensorRT adapter builds its
engine for; a protocol shape that contradicts the model's rank or a static dimension is
refused before anything runs; a static model runs its own shape, as before. These run real
ONNX Runtime CPU sessions on tiny Identity graphs.
"""

import json

import numpy as np
import pytest
from calbredge import init
from calbredge.cli import main
from calbredge_protocol.bundled import bundled_protocol_path
from calbredge_protocol.models import BenchmarkProtocol
from calbredge_protocol.onnx_adapter import ONNXRuntimeAdapter
from click.testing import CliRunner

pytest.importorskip("onnxruntime")


def _identity(path, inputs: dict[str, list]):
    """An ONNX graph passing each input straight to an output; dims may be names (dynamic)."""
    import onnx
    from onnx import TensorProto, helper

    nodes = [helper.make_node("Identity", [n], [f"{n}_out"]) for n in inputs]
    graph = helper.make_graph(
        nodes,
        "identity",
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in inputs.items()],
        [
            helper.make_tensor_value_info(f"{n}_out", TensorProto.FLOAT, s)
            for n, s in inputs.items()
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def _loaded(path) -> ONNXRuntimeAdapter:
    adapter = ONNXRuntimeAdapter()
    adapter.load_model(path)
    return adapter


def _protocol(shape: list[int]) -> BenchmarkProtocol:
    return BenchmarkProtocol(
        protocol_version="1.0",
        warmup_iterations=1,
        measurement_iterations=3,
        batch_size=shape[0],
        input_shape=shape,
    )


def _feed(shape) -> dict:
    return {"x": np.zeros(shape, dtype=np.float32)}


@pytest.mark.unit
class TestDynamicDimensionsTakeTheProtocolsValue:
    def test_a_dynamic_batch_of_4_runs_as_4(self, tmp_path):
        adapter = _loaded(_identity(tmp_path / "m.onnx", {"x": ["N", 3, 8, 8]}))

        outputs = adapter.run_inference(_feed((4, 3, 8, 8)))

        assert outputs["x_out"].shape == (4, 3, 8, 8)
        assert adapter.resolved_input_shape == [4, 3, 8, 8]

    def test_dynamic_spatial_dimensions_are_bound(self, tmp_path):
        adapter = _loaded(_identity(tmp_path / "m.onnx", {"x": [1, 3, "H", "W"]}))

        outputs = adapter.run_inference(_feed((1, 3, 32, 16)))

        assert outputs["x_out"].shape == (1, 3, 32, 16)
        assert adapter.resolved_input_shape == [1, 3, 32, 16]

    def test_the_benchmark_runs_the_protocols_shape(self, tmp_path):
        adapter = _loaded(_identity(tmp_path / "m.onnx", {"x": ["N", 3, 8, 8]}))
        seen: list[tuple] = []
        run = adapter.session.run
        adapter.session.run = lambda names, feed: seen.append(feed["x"].shape) or run(names, feed)

        adapter.run_benchmark(_feed((2, 3, 8, 8)), _protocol([2, 3, 8, 8]))

        assert set(seen) == {(2, 3, 8, 8)}  # warm-up and every measured iteration
        assert len(seen) == 1 + 3

    def test_without_a_given_array_a_dynamic_dimension_is_1(self, tmp_path):
        adapter = _loaded(_identity(tmp_path / "m.onnx", {"x": ["N", 3, 8, 8]}))

        outputs = adapter.run_inference({})

        assert outputs["x_out"].shape == (1, 3, 8, 8)

    def test_a_second_input_takes_1_for_a_dynamic_dimension(self, tmp_path):
        path = _identity(tmp_path / "m.onnx", {"x": ["N", 3, 8, 8], "mask": ["N", 4]})
        adapter = _loaded(path)

        outputs = adapter.run_inference(_feed((2, 3, 8, 8)))

        assert outputs["x_out"].shape == (2, 3, 8, 8)
        assert outputs["mask_out"].shape == (1, 4)


@pytest.mark.unit
class TestAContradictionIsRefused:
    def test_another_rank_is_refused(self, tmp_path):
        adapter = _loaded(_identity(tmp_path / "m.onnx", {"x": ["N", 3, 8, 8]}))

        with pytest.raises(
            RuntimeError, match=r"has 3 dimensions, but the model's input 'x' has 4"
        ):
            adapter.run_inference(_feed((1, 3, 8)))

    def test_a_static_dimension_that_differs_is_refused(self, tmp_path):
        adapter = _loaded(_identity(tmp_path / "m.onnx", {"x": ["N", 3, 8, 8]}))

        with pytest.raises(
            RuntimeError, match=r"contradicts the model's input 'x'.*dimensions 2, 3"
        ):
            adapter.run_inference(_feed((1, 3, 224, 224)))

    def test_the_refusal_comes_before_anything_runs(self, tmp_path):
        adapter = _loaded(_identity(tmp_path / "m.onnx", {"x": [1, 3, 8, 8]}))
        calls: list = []
        adapter.session.run = lambda *a: calls.append(a)

        with pytest.raises(RuntimeError, match="contradicts"):
            adapter.run_benchmark(_feed((1, 3, 16, 16)), _protocol([1, 3, 16, 16]))

        assert calls == []


@pytest.mark.unit
class TestTheRefusalNamesTheBundledProtocol:
    """A refusal names the bundled protocol whose input_shape fits the model, chosen by the
    same matching `calbredge init` uses, as a --protocol path that exists."""

    @staticmethod
    def _refusal(adapter, shape) -> str:
        with pytest.raises(RuntimeError) as caught:
            adapter.run_inference(_feed(shape))
        return str(caught.value)

    @pytest.mark.parametrize(
        ("dims", "protocol_shape", "bundled"),
        [
            ([1, 3, 640, 640], (1, 3, 224, 224), "yolo640.yaml"),  # yolov8n with the default
            (["N", 3, 224, 224], (1, 3, 320, 320), "v1.0.yaml"),  # the default wins a tie
            ([1, 224, 224, 3], (1, 3, 224, 224), "protocol224_nhwc.yaml"),  # the other layout
        ],
    )
    def test_a_static_contradiction_names_init_s_choice(
        self, dims, protocol_shape, bundled, tmp_path
    ):
        adapter = _loaded(_identity(tmp_path / "m.onnx", {"x": dims}))

        message = self._refusal(adapter, protocol_shape)

        assert init.choose_protocol(dims)[0] == bundled  # init chooses the same file
        path = bundled_protocol_path(bundled)
        assert f"Use --protocol {path}, the bundled protocol with the model's shape" in message
        assert path.is_file()

    def test_a_rank_contradiction_names_it_too(self, tmp_path):
        adapter = _loaded(_identity(tmp_path / "m.onnx", {"x": [1, 3, 640, 640]}))

        message = self._refusal(adapter, (1, 640, 640))

        assert f"Use --protocol {bundled_protocol_path('yolo640.yaml')}" in message

    def test_no_bundled_protocol_fits_and_none_is_named(self, tmp_path):
        adapter = _loaded(_identity(tmp_path / "m.onnx", {"x": [1, 3, 8, 8]}))

        message = self._refusal(adapter, (1, 3, 224, 224))

        assert "--protocol" not in message
        assert "Use a protocol whose input_shape matches the model's static dimensions." in message

    def test_a_dynamic_spatial_model_names_none(self, tmp_path):
        adapter = _loaded(_identity(tmp_path / "m.onnx", {"x": [1, 3, "H", "W"]}))

        message = self._refusal(adapter, (2, 3, 224, 224))

        assert "--protocol" not in message


@pytest.mark.unit
class TestStaticModelsAreUnchanged:
    def test_a_static_model_runs_its_own_shape(self, tmp_path):
        adapter = _loaded(_identity(tmp_path / "m.onnx", {"x": [1, 3, 8, 8]}))
        assert adapter.resolved_input_shape == [1, 3, 8, 8]  # set at load, as before

        outputs = adapter.run_inference(_feed((1, 3, 8, 8)))

        assert outputs["x_out"].shape == (1, 3, 8, 8)
        assert adapter.resolved_input_shape == [1, 3, 8, 8]

    def test_a_dynamic_model_resolves_no_shape_at_load(self, tmp_path):
        adapter = _loaded(_identity(tmp_path / "m.onnx", {"x": ["N", 3, 8, 8]}))

        assert adapter.resolved_input_shape is None


@pytest.mark.unit
def test_calbredge_run_records_the_shape_that_ran(tmp_path):
    """End to end: a dynamic batch with a batch-2 protocol runs, and records, batch 2."""
    model = _identity(tmp_path / "m.onnx", {"x": ["N", 3, 8, 8]})
    protocol = tmp_path / "p.yaml"
    protocol.write_text(
        'protocol_version: "1.0"\nwarmup_iterations: 1\nmeasurement_iterations: 3\n'
        "batch_size: 2\ninput_shape: [2, 3, 8, 8]\n"
    )
    seen: list[tuple] = []
    real_prepare = ONNXRuntimeAdapter._prepare_inputs

    def prepare(self, inputs):
        prepared = real_prepare(self, inputs)
        seen.append(prepared["x"].shape)
        return prepared

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ONNXRuntimeAdapter, "_prepare_inputs", prepare)
        result = CliRunner().invoke(
            main,
            [
                "run",
                "--model",
                str(model),
                "--runtime",
                "onnxruntime",
                "--protocol",
                str(protocol),
                "--offline",
            ],
        )

    assert result.exit_code == 0, result.output
    assert set(seen) == {(2, 3, 8, 8)}
    payload, _ = json.JSONDecoder().raw_decode(result.stdout[result.stdout.index("{") :])
    assert payload["input_shape"] == [2, 3, 8, 8]


@pytest.mark.unit
def test_calbredge_run_refuses_a_contradicting_protocol(tmp_path):
    model = _identity(tmp_path / "m.onnx", {"x": [1, 3, 8, 8]})
    protocol = tmp_path / "p.yaml"
    protocol.write_text(
        'protocol_version: "1.0"\nwarmup_iterations: 1\nmeasurement_iterations: 3\n'
        "batch_size: 1\ninput_shape: [1, 3, 224, 224]\n"
    )

    result = CliRunner().invoke(
        main,
        [
            "run",
            "--model",
            str(model),
            "--runtime",
            "onnxruntime",
            "--protocol",
            str(protocol),
            "--offline",
        ],
    )

    assert result.exit_code != 0
    assert "contradicts the model's input 'x'" in (result.output + str(result.exception))


@pytest.mark.unit
def test_the_named_protocol_is_a_command_that_runs(tmp_path):
    """The --protocol the refusal names is accepted by calbredge run for the same model."""
    model = _identity(tmp_path / "m.onnx", {"x": [1, 3, 320, 320]})
    wrong = tmp_path / "p.yaml"
    wrong.write_text('protocol_version: "1.0"\nbatch_size: 1\ninput_shape: [1, 3, 224, 224]\n')
    args = ["run", "--model", str(model), "--runtime", "onnxruntime", "--offline"]

    refused = CliRunner().invoke(main, [*args, "--protocol", str(wrong)])
    path = bundled_protocol_path("protocol320.yaml")
    assert f"Use --protocol {path}," in refused.output + str(refused.exception)

    ran = CliRunner().invoke(main, [*args, "--protocol", str(path)])
    assert ran.exit_code == 0, ran.output
