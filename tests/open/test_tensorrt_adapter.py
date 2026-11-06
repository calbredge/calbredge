"""Unit tests for TensorRTAdapter.

All tests mock ``tensorrt`` and ``pycuda`` so no CUDA hardware or TRT
installation is required.  The mocks are injected via ``patch.dict(sys.modules)``
before any import of the adapter so that the lazy-import helpers inside
TensorRTAdapter pick up the fakes.
"""

import subprocess
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from calbredge_protocol.adapters import _get_nvpmodel_mode
from calbredge_protocol.autodetect import get_adapter
from calbredge_protocol.tensorrt_adapter import (
    TensorRTAdapter,
    _make_calibrator,
    _trt_dtype_to_numpy,
)

# ---------------------------------------------------------------------------
# Shared mock-module builders
# ---------------------------------------------------------------------------


def _make_trt_mock():
    """Build a fake ``tensorrt`` module with minimal enum/class structure."""
    trt = types.ModuleType("tensorrt")

    class _Logger:
        WARNING = 0

        def __init__(self, level):
            pass

    class _TensorIOMode:
        INPUT = 0
        OUTPUT = 1

    class _DataType:
        FLOAT = 0
        HALF = 1
        INT8 = 2
        INT32 = 3
        BOOL = 4
        UINT8 = 5

    class _BuilderFlag:
        FP16 = 1
        INT8 = 2

    class _MemoryPoolType:
        WORKSPACE = 0

    class _NetworkDefinitionCreationFlag:
        EXPLICIT_BATCH = 0

    class _IInt8EntropyCalibrator2:
        def __init__(self):
            pass

    trt.Logger = _Logger
    trt.TensorIOMode = _TensorIOMode
    trt.DataType = _DataType
    trt.BuilderFlag = _BuilderFlag
    trt.MemoryPoolType = _MemoryPoolType
    trt.NetworkDefinitionCreationFlag = _NetworkDefinitionCreationFlag
    trt.IInt8EntropyCalibrator2 = _IInt8EntropyCalibrator2
    return trt


def _make_engine_mock(trt_mock, input_name="images", output_name="output0"):
    """Return a mock TRT engine with one input and one output tensor."""
    engine = MagicMock()
    engine.num_io_tensors = 2

    names = [input_name, output_name]
    shapes = {input_name: [1, 3, 640, 640], output_name: [1, 84, 8400]}

    engine.get_tensor_name.side_effect = lambda i: names[i]
    engine.get_tensor_mode.side_effect = lambda n: (
        trt_mock.TensorIOMode.INPUT if n == input_name else trt_mock.TensorIOMode.OUTPUT
    )
    engine.get_tensor_shape.side_effect = lambda n: shapes[n]
    engine.get_tensor_dtype.return_value = trt_mock.DataType.FLOAT

    context = MagicMock()
    engine.create_execution_context.return_value = context
    return engine, context


def _make_cuda_mock():
    """Return a mock ``pycuda.driver`` module."""
    cuda = MagicMock(name="pycuda.driver")
    stream = MagicMock()
    stream.handle = 42
    cuda.Stream.return_value = stream
    cuda.mem_alloc.return_value = MagicMock()
    return cuda


def _make_builder_mock(trt_mock, engine_mock, serialized_bytes=b"fake_engine"):
    """Return a mock TRT Builder that produces ``engine_mock``."""
    network = MagicMock()
    network.num_inputs = 1
    inp = MagicMock()
    inp.name = "images"
    inp.shape = [1, 3, 640, 640]
    inp.dtype = trt_mock.DataType.FLOAT
    network.get_input.return_value = inp
    network.num_outputs = 1

    parser = MagicMock()
    parser.parse.return_value = True
    parser.num_errors = 0

    config = MagicMock()
    builder = MagicMock()
    builder.platform_has_fast_fp16 = True
    builder.platform_has_fast_int8 = True
    builder.create_network.return_value = network
    builder.create_builder_config.return_value = config
    builder.build_serialized_network.return_value = serialized_bytes

    trt_runtime = MagicMock()
    trt_runtime.deserialize_cuda_engine.return_value = engine_mock

    trt_mock.Builder = MagicMock(return_value=builder)
    trt_mock.OnnxParser = MagicMock(return_value=parser)
    trt_mock.Runtime = MagicMock(return_value=trt_runtime)

    return builder, parser, config, trt_runtime, network


def _patch_modules(trt_mock, cuda_mock):
    """Return a context manager that injects fake TRT and pycuda into sys.modules."""
    pycuda = types.ModuleType("pycuda")
    pycuda_autoinit = types.ModuleType("pycuda.autoinit")
    return patch.dict(
        sys.modules,
        {
            "tensorrt": trt_mock,
            "pycuda": pycuda,
            "pycuda.driver": cuda_mock,
            "pycuda.autoinit": pycuda_autoinit,
        },
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_onnx(path, shape=(1, 3, 640, 640), fill=0.0):
    """A real, tiny ONNX model (Identity plus one unused weight) whose input
    matches the mocked network: the engine build key is computed from the
    model's actual input signature and content hash. ``fill`` changes the
    weight values (same I/O, different content)."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weight = numpy_helper.from_array(np.full((4,), fill, dtype=np.float32), "unused_weight")
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["output0"])],
        "identity",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, list(shape))],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(shape))],
        initializer=[weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def _engines(onnx_path, precision, power_mode):
    """Content-addressed engine files for one (model, precision, power mode)."""
    return sorted(onnx_path.parent.glob(f"{onnx_path.name}.trt.{precision}.{power_mode}.*.engine"))


@pytest.fixture
def onnx_file(tmp_path):
    """Write a minimal real .onnx file and return its path."""
    return _write_onnx(tmp_path / "model.onnx")


@pytest.fixture
def trt_mocks():
    """Yield (trt_mock, cuda_mock, engine_mock, context_mock) with sys.modules patched.

    Also stubs ``_get_nvpmodel_mode`` to return ``"default"`` so tests don't
    invoke sudo or depend on Jetson hardware.
    """
    trt = _make_trt_mock()
    cuda = _make_cuda_mock()
    engine, context = _make_engine_mock(trt)
    _make_builder_mock(trt, engine)
    with (
        _patch_modules(trt, cuda),
        patch("calbredge_protocol.tensorrt_adapter._get_nvpmodel_mode", return_value=None),
    ):
        yield trt, cuda, engine, context


# ---------------------------------------------------------------------------
# 1. Init / precision validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTensorRTAdapterInit:
    def test_default_precision_is_fp32(self):
        adapter = TensorRTAdapter()
        assert adapter._precision == "fp32"
        assert adapter.runtime_label == "tensorrt-fp32"

    def test_fp16_precision(self):
        adapter = TensorRTAdapter(precision="fp16")
        assert adapter._precision == "fp16"
        assert adapter.runtime_label == "tensorrt-fp16"

    def test_int8_precision(self):
        adapter = TensorRTAdapter(precision="int8")
        assert adapter._precision == "int8"
        assert adapter.runtime_label == "tensorrt-int8"

    def test_invalid_precision_raises(self):
        with pytest.raises(ValueError, match="precision must be one of"):
            TensorRTAdapter(precision="bf16")

    def test_input_dtype_defaults_to_float32(self):
        """Regression guard: TensorRTAdapter doesn't override input_dtype, so
        cli.py:343's synthesized inputs stay float32 — unchanged behaviour
        now that the CLI reads adapter.input_dtype instead of hardcoding
        np.float32 directly."""
        adapter = TensorRTAdapter()
        assert adapter.input_dtype == np.dtype(np.float32)

    def test_not_loaded_initially(self):
        adapter = TensorRTAdapter()
        assert not adapter.is_loaded
        assert adapter.input_names == []


# ---------------------------------------------------------------------------
# 2. Import errors
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestImportErrors:
    def test_missing_tensorrt_raises_clear_error(self, onnx_file):
        cuda = _make_cuda_mock()
        pycuda = types.ModuleType("pycuda")
        pycuda_autoinit = types.ModuleType("pycuda.autoinit")
        # tensorrt absent from sys.modules and also blocked from real import
        with patch.dict(
            sys.modules,
            {
                "pycuda": pycuda,
                "pycuda.driver": cuda,
                "pycuda.autoinit": pycuda_autoinit,
            },
        ):
            sys.modules.pop("tensorrt", None)
            adapter = TensorRTAdapter()
            with pytest.raises(RuntimeError, match="tensorrt is not installed"):
                adapter.load_model(onnx_file)

    def test_missing_pycuda_raises_clear_error(self, onnx_file):
        trt = _make_trt_mock()
        # pycuda absent — importing it should fail
        with patch.dict(sys.modules, {"tensorrt": trt}):
            sys.modules.pop("pycuda", None)
            sys.modules.pop("pycuda.driver", None)
            sys.modules.pop("pycuda.autoinit", None)
            adapter = TensorRTAdapter()
            with pytest.raises(RuntimeError, match="pycuda is not installed"):
                adapter.load_model(onnx_file)


# ---------------------------------------------------------------------------
# 3. load_model — non-ONNX and missing file errors
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadModelErrors:
    def test_non_onnx_extension_raises_value_error(self, tmp_path, trt_mocks):
        _trt, _cuda, _engine, _context = trt_mocks
        engine_file = tmp_path / "model.engine"
        engine_file.write_bytes(b"bytes")
        adapter = TensorRTAdapter()
        with pytest.raises(ValueError, match=r"\.onnx"):
            adapter.load_model(engine_file)

    def test_missing_model_raises_file_not_found(self, trt_mocks):
        adapter = TensorRTAdapter()
        with pytest.raises(FileNotFoundError):
            adapter.load_model("/nonexistent/path/model.onnx")

    def test_parser_failure_raises_runtime_error(self, tmp_path, trt_mocks):
        trt, _cuda, _engine, _context = trt_mocks
        # Override parser to report a failure
        failing_parser = MagicMock()
        failing_parser.parse.return_value = False
        failing_parser.num_errors = 1
        failing_parser.get_error.return_value = "Unsupported op"
        trt.OnnxParser = MagicMock(return_value=failing_parser)

        p = tmp_path / "bad.onnx"
        p.write_bytes(b"bad onnx")
        adapter = TensorRTAdapter()
        with pytest.raises(RuntimeError, match="Failed to parse"):
            adapter.load_model(p)


# ---------------------------------------------------------------------------
# 4. Engine caching
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEngineCaching:
    def test_engine_built_and_cache_file_written(self, onnx_file, trt_mocks):
        _trt, _cuda, _engine, _context = trt_mocks
        adapter = TensorRTAdapter(precision="fp16")
        adapter.load_model(onnx_file)

        (cache_path,) = _engines(onnx_file, "fp16", "default")
        assert cache_path.read_bytes() == b"fake_engine"
        assert cache_path.with_name(cache_path.name[: -len(".engine")] + ".engine.json").is_file()

    def test_cached_engine_loaded_without_rebuilding(self, onnx_file, trt_mocks):
        trt, _cuda, _engine, _context = trt_mocks

        # First load builds and caches (engine + build record).
        TensorRTAdapter(precision="fp32").load_model(onnx_file)
        (cache_path,) = _engines(onnx_file, "fp32", "default")
        cache_path.write_bytes(b"cached_engine_bytes")
        trt.Builder.reset_mock()
        trt.Runtime.return_value.deserialize_cuda_engine.reset_mock()

        adapter = TensorRTAdapter(precision="fp32")
        adapter.load_model(onnx_file)

        # Builder should NOT have been called because the cache matched
        trt.Builder.assert_not_called()
        # Runtime.deserialize_cuda_engine should have been called with cache bytes
        trt.Runtime.return_value.deserialize_cuda_engine.assert_called_once_with(
            b"cached_engine_bytes"
        )

    def test_fp32_and_fp16_use_separate_cache_files(self, onnx_file, trt_mocks):
        _trt, _cuda, _engine, _context = trt_mocks

        adapter32 = TensorRTAdapter(precision="fp32")
        adapter32.load_model(onnx_file)

        adapter16 = TensorRTAdapter(precision="fp16")
        adapter16.load_model(onnx_file)

        assert len(_engines(onnx_file, "fp32", "default")) == 1
        assert len(_engines(onnx_file, "fp16", "default")) == 1

    def test_resolved_input_shape_matches_on_fresh_build_and_cache_hit(self, onnx_file, trt_mocks):
        """Regression guard: resolved_input_shape used to be set only inside
        _build_engine, so a cache-hit load (every run after the first for a
        given model/precision/power_mode) left it None and silently fell back
        to the protocol shape. It is now set in _allocate_buffers, which runs
        unconditionally after either path obtains an engine — so a fresh
        build and a subsequent cache-hit load of the SAME engine must agree."""
        trt, _cuda, _engine, _context = trt_mocks

        # First adapter: no cache file exists yet -> exercises _build_engine.
        fresh = TensorRTAdapter(precision="fp16")
        fresh.load_model(onnx_file)
        assert len(_engines(onnx_file, "fp16", "default")) == 1  # the fresh-build path

        # Second adapter, same model/precision: cache_path already exists ->
        # exercises the deserialize branch, never calling _build_engine again.
        trt.Builder.reset_mock()
        cached = TensorRTAdapter(precision="fp16")
        cached.load_model(onnx_file)
        # Confirms this really was the cache-hit path. The builder is still created once, for
        # the fast-FP16 check, but builds nothing.
        trt.Builder.return_value.create_network.assert_not_called()
        trt.Builder.return_value.build_serialized_network.assert_not_called()

        assert fresh.resolved_input_shape == [1, 3, 640, 640]
        assert cached.resolved_input_shape == [1, 3, 640, 640]
        assert fresh.resolved_input_shape == cached.resolved_input_shape

    def test_resolved_input_shape_none_when_engine_reports_dynamic_dim(self, onnx_file, trt_mocks):
        """A dynamic dim (-1) on the loaded engine's bound input shape must
        leave resolved_input_shape None rather than recording -1 — same
        contract as the ONNX adapter for a dynamic/unpinned axis."""
        _trt, _cuda, engine, _context = trt_mocks
        engine.get_tensor_shape.side_effect = lambda n: (
            [-1, 3, 640, 640] if n == "images" else [1, 84, 8400]
        )

        adapter = TensorRTAdapter(precision="fp16")
        adapter.load_model(onnx_file)

        assert adapter.resolved_input_shape is None


# ---------------------------------------------------------------------------
# 5. Precision flags
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPrecisionFlags:
    def test_fp16_flag_set_on_builder_config(self, onnx_file, trt_mocks):
        trt, _cuda, engine, _context = trt_mocks
        builder, *_ = _make_builder_mock(trt, engine)

        adapter = TensorRTAdapter(precision="fp16")
        adapter.load_model(onnx_file)

        config = builder.create_builder_config.return_value
        config.set_flag.assert_called_with(trt.BuilderFlag.FP16)

    @patch("calbredge_protocol.tensorrt_adapter._make_calibrator")
    def test_int8_flag_and_calibrator_set(self, mock_calib, onnx_file, trt_mocks):
        trt, _cuda, engine, _context = trt_mocks
        builder, *_ = _make_builder_mock(trt, engine)

        adapter = TensorRTAdapter(precision="int8")
        adapter.load_model(onnx_file)

        config = builder.create_builder_config.return_value
        config.set_flag.assert_called_with(trt.BuilderFlag.INT8)
        assert config.int8_calibrator is not None

    def test_fp32_sets_no_precision_flag(self, onnx_file, trt_mocks):
        trt, _cuda, engine, _context = trt_mocks
        builder, *_ = _make_builder_mock(trt, engine)

        adapter = TensorRTAdapter(precision="fp32")
        adapter.load_model(onnx_file)

        config = builder.create_builder_config.return_value
        config.set_flag.assert_not_called()


# ---------------------------------------------------------------------------
# 5b. Optimization profile
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOptimizationProfile:
    def test_optimization_profile_always_added(self, onnx_file, trt_mocks):
        trt, _cuda, engine, _context = trt_mocks
        builder, *_ = _make_builder_mock(trt, engine)

        adapter = TensorRTAdapter(precision="fp32")
        adapter.load_model(onnx_file)

        config = builder.create_builder_config.return_value
        config.add_optimization_profile.assert_called_once()

    def test_profile_set_shape_called_for_each_input(self, onnx_file, trt_mocks):
        trt, _cuda, engine, _context = trt_mocks
        builder, *_ = _make_builder_mock(trt, engine)

        adapter = TensorRTAdapter(precision="fp32")
        adapter.load_model(onnx_file)

        profile = builder.create_optimization_profile.return_value
        profile.set_shape.assert_called_once()

    def test_dynamic_dim_resolved_from_input_shape(self, onnx_file, trt_mocks):
        trt, _cuda, engine, _context = trt_mocks
        builder, _, _, _, network = _make_builder_mock(trt, engine)

        # Override the network input to have a dynamic batch dim
        inp = MagicMock()
        inp.name = "images"
        inp.shape = [-1, 3, 640, 640]
        inp.dtype = trt.DataType.FLOAT
        network.get_input.return_value = inp

        adapter = TensorRTAdapter(precision="fp32")
        adapter.load_model(onnx_file, input_shape=[1, 3, 640, 640])

        profile = builder.create_optimization_profile.return_value
        profile.set_shape.assert_called_once_with(
            "images",
            min=(1, 3, 640, 640),
            opt=(1, 3, 640, 640),
            max=(1, 3, 640, 640),
        )

    def test_dynamic_dim_falls_back_to_1_when_no_input_shape(self, tmp_path, trt_mocks):
        onnx_file = _write_onnx(tmp_path / "dyn.onnx", shape=("N", 3, 224, 224))
        trt, _cuda, engine, _context = trt_mocks
        builder, _, _, _, network = _make_builder_mock(trt, engine)

        inp = MagicMock()
        inp.name = "images"
        inp.shape = [-1, 3, 224, 224]
        inp.dtype = trt.DataType.FLOAT
        network.get_input.return_value = inp

        adapter = TensorRTAdapter(precision="fp32")
        adapter.load_model(onnx_file)  # no input_shape

        profile = builder.create_optimization_profile.return_value
        profile.set_shape.assert_called_once_with(
            "images",
            min=(1, 3, 224, 224),
            opt=(1, 3, 224, 224),
            max=(1, 3, 224, 224),
        )


# ---------------------------------------------------------------------------
# 5c. INT8 build failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInt8BuildFailure:
    @patch("calbredge_protocol.tensorrt_adapter._make_calibrator")
    def test_int8_exception_raises_clear_runtime_error(self, mock_calib, onnx_file, trt_mocks):
        trt, _cuda, engine, _context = trt_mocks
        builder, *_ = _make_builder_mock(trt, engine)
        builder.build_serialized_network.side_effect = RuntimeError("quantizationBase")

        adapter = TensorRTAdapter(precision="int8")
        with pytest.raises(RuntimeError, match="INT8 engine build failed"):
            adapter.load_model(onnx_file)

    @patch("calbredge_protocol.tensorrt_adapter._make_calibrator")
    def test_int8_none_result_raises_clear_runtime_error(self, mock_calib, onnx_file, trt_mocks):
        trt, _cuda, engine, _context = trt_mocks
        _builder, *_ = _make_builder_mock(trt, engine, serialized_bytes=None)

        adapter = TensorRTAdapter(precision="int8")
        with pytest.raises(RuntimeError, match="INT8 engine build failed"):
            adapter.load_model(onnx_file)

    def test_fp32_none_result_raises_generic_error(self, onnx_file, trt_mocks):
        trt, _cuda, engine, _context = trt_mocks
        _builder, *_ = _make_builder_mock(trt, engine, serialized_bytes=None)

        adapter = TensorRTAdapter(precision="fp32")
        with pytest.raises(RuntimeError, match="engine serialization failed"):
            adapter.load_model(onnx_file)

    def test_fp16_exception_propagates_unchanged(self, onnx_file, trt_mocks):
        trt, _cuda, engine, _context = trt_mocks
        builder, *_ = _make_builder_mock(trt, engine)
        builder.build_serialized_network.side_effect = ValueError("unexpected")

        adapter = TensorRTAdapter(precision="fp16")
        with pytest.raises(ValueError, match="unexpected"):
            adapter.load_model(onnx_file)


# ---------------------------------------------------------------------------
# 5d. FP16 and INT8 need the GPU's fast support
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFastPrecisionRequired:
    """FP16 or INT8 on a GPU without fast support is refused, built or cached, never warned
    about: the precision flag only allows those kernels, so the result could record fp16 or
    int8 for an engine that runs others."""

    @pytest.mark.parametrize(
        ("precision", "attribute", "name"),
        [("fp16", "platform_has_fast_fp16", "FP16"), ("int8", "platform_has_fast_int8", "INT8")],
    )
    def test_refused_without_fast_support_and_nothing_is_built(
        self, precision, attribute, name, onnx_file, trt_mocks
    ):
        trt, _cuda, engine, _context = trt_mocks
        builder, *_ = _make_builder_mock(trt, engine)
        setattr(builder, attribute, False)

        with pytest.raises(RuntimeError, match=f"no fast {name} support"):
            TensorRTAdapter(precision=precision).load_model(onnx_file)

        builder.create_network.assert_not_called()
        builder.build_serialized_network.assert_not_called()
        trt.Runtime.return_value.deserialize_cuda_engine.assert_not_called()
        assert _engines(onnx_file, precision, "default") == []

    def test_a_cached_engine_is_refused_too(self, onnx_file, trt_mocks):
        trt, _cuda, engine, _context = trt_mocks
        builder, *_ = _make_builder_mock(trt, engine)
        TensorRTAdapter(precision="fp16").load_model(onnx_file)
        assert len(_engines(onnx_file, "fp16", "default")) == 1
        trt.Runtime.return_value.deserialize_cuda_engine.reset_mock()
        builder.platform_has_fast_fp16 = False

        with pytest.raises(RuntimeError, match="no fast FP16 support"):
            TensorRTAdapter(precision="fp16").load_model(onnx_file)

        trt.Runtime.return_value.deserialize_cuda_engine.assert_not_called()

    def test_a_missing_attribute_is_refused(self, onnx_file, trt_mocks):
        trt, _cuda, engine, _context = trt_mocks
        builder, *_ = _make_builder_mock(trt, engine)
        del builder.platform_has_fast_fp16

        with pytest.raises(RuntimeError, match="no fast FP16 support"):
            TensorRTAdapter(precision="fp16").load_model(onnx_file)

    def test_each_precision_checks_its_own_support(self, onnx_file, trt_mocks):
        trt, _cuda, engine, _context = trt_mocks
        builder, *_ = _make_builder_mock(trt, engine)
        builder.platform_has_fast_int8 = False

        adapter = TensorRTAdapter(precision="fp16")
        adapter.load_model(onnx_file)

        assert adapter.is_loaded

    def test_fp32_is_never_checked(self, onnx_file, trt_mocks):
        trt, _cuda, engine, _context = trt_mocks
        builder, *_ = _make_builder_mock(trt, engine)
        builder.platform_has_fast_fp16 = False
        builder.platform_has_fast_int8 = False

        adapter = TensorRTAdapter(precision="fp32")
        adapter.load_model(onnx_file)

        assert adapter.is_loaded
        assert len(_engines(onnx_file, "fp32", "default")) == 1


# ---------------------------------------------------------------------------
# 6. load_model state after successful load
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadModelSuccess:
    def test_is_loaded_after_load_model(self, onnx_file, trt_mocks):
        _trt, _cuda, _engine, _context = trt_mocks
        adapter = TensorRTAdapter()
        adapter.load_model(onnx_file)
        assert adapter.is_loaded

    def test_input_names_populated(self, onnx_file, trt_mocks):
        _trt, _cuda, _engine, _context = trt_mocks
        adapter = TensorRTAdapter()
        adapter.load_model(onnx_file)
        assert adapter.input_names == ["images"]

    def test_output_names_populated(self, onnx_file, trt_mocks):
        _trt, _cuda, _engine, _context = trt_mocks
        adapter = TensorRTAdapter()
        adapter.load_model(onnx_file)
        assert adapter._output_names == ["output0"]

    def test_device_buffers_allocated_for_inputs_and_outputs(self, onnx_file, trt_mocks):
        _trt, cuda, _engine, _context = trt_mocks
        adapter = TensorRTAdapter()
        adapter.load_model(onnx_file)
        # One call per input tensor + one per output tensor
        assert cuda.mem_alloc.call_count == 2

    def test_set_tensor_address_called_for_all_tensors(self, onnx_file, trt_mocks):
        _trt, _cuda, _engine, context = trt_mocks
        adapter = TensorRTAdapter()
        adapter.load_model(onnx_file)
        # set_tensor_address once per input + once per output
        assert context.set_tensor_address.call_count == 2


# ---------------------------------------------------------------------------
# 7. warmup
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWarmup:
    def test_warmup_zero_iterations_is_noop(self, onnx_file, trt_mocks):
        _trt, _cuda, _engine, context = trt_mocks
        adapter = TensorRTAdapter()
        adapter.load_model(onnx_file)
        context.execute_async_v3.reset_mock()

        adapter.warmup({}, 0)

        context.execute_async_v3.assert_not_called()

    def test_warmup_calls_execute_n_times(self, onnx_file, trt_mocks):
        _trt, _cuda, _engine, context = trt_mocks
        adapter = TensorRTAdapter()
        adapter.load_model(onnx_file)
        context.execute_async_v3.reset_mock()

        adapter.warmup({}, 5)

        assert context.execute_async_v3.call_count == 5

    def test_warmup_synchronizes_stream(self, onnx_file, trt_mocks):
        _trt, cuda, _engine, _context = trt_mocks
        stream = cuda.Stream.return_value

        adapter = TensorRTAdapter()
        adapter.load_model(onnx_file)
        stream.synchronize.reset_mock()

        adapter.warmup({}, 3)

        stream.synchronize.assert_called_once()


# ---------------------------------------------------------------------------
# 8. run_inference
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunInference:
    def test_run_inference_calls_execute_async(self, onnx_file, trt_mocks):
        _trt, cuda, _engine, context = trt_mocks
        adapter = TensorRTAdapter()
        adapter.load_model(onnx_file)
        context.execute_async_v3.reset_mock()

        adapter.run_inference({})

        context.execute_async_v3.assert_called_once_with(cuda.Stream.return_value.handle)

    def test_run_inference_returns_output_dict(self, onnx_file, trt_mocks):
        _trt, _cuda, _engine, _context = trt_mocks
        adapter = TensorRTAdapter()
        adapter.load_model(onnx_file)

        result = adapter.run_inference({})

        assert isinstance(result, dict)
        assert "output0" in result

    def test_run_inference_queues_dtoh_copy(self, onnx_file, trt_mocks):
        _trt, cuda, _engine, _context = trt_mocks
        adapter = TensorRTAdapter()
        adapter.load_model(onnx_file)
        cuda.memcpy_dtoh_async.reset_mock()

        adapter.run_inference({})

        cuda.memcpy_dtoh_async.assert_called_once()


# ---------------------------------------------------------------------------
# 9. run_benchmark
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunBenchmark:
    def _make_protocol(self):
        from calbredge_protocol.models import BenchmarkProtocol

        return BenchmarkProtocol(
            protocol_version="1.0",
            warmup_iterations=2,
            measurement_iterations=5,
            input_shape=[1, 3, 640, 640],
        )

    def test_run_benchmark_returns_latency_stats(self, onnx_file, trt_mocks):
        from calbredge_protocol.stats import LatencyStats

        _trt, _cuda, _engine, _context = trt_mocks
        adapter = TensorRTAdapter()
        adapter.load_model(onnx_file)

        stats = adapter.run_benchmark({}, self._make_protocol())

        assert isinstance(stats, LatencyStats)
        assert stats.sample_count == 5

    def test_stream_synchronize_called_per_measurement(self, onnx_file, trt_mocks):
        _trt, cuda, _engine, _context = trt_mocks
        stream = cuda.Stream.return_value
        adapter = TensorRTAdapter()
        adapter.load_model(onnx_file)

        protocol = self._make_protocol()
        # Reset after load_model warmup calls
        stream.synchronize.reset_mock()

        adapter.run_benchmark({}, protocol)

        # synchronize is called once per measurement iteration (inside run_benchmark loop)
        # plus once at end of warmup
        assert stream.synchronize.call_count >= protocol.measurement_iterations

    def test_run_benchmark_sets_run_status(self, onnx_file, trt_mocks):
        _trt, _cuda, _engine, _context = trt_mocks
        adapter = TensorRTAdapter()
        adapter.load_model(onnx_file)

        adapter.run_benchmark({}, self._make_protocol())

        assert hasattr(adapter, "_last_run_status")
        assert adapter._last_run_status in {"stable", "inconclusive", "unstable"}

    def test_run_benchmark_calls_collect_hardware_metrics(self, onnx_file, trt_mocks):
        _trt, _cuda, _engine, _context = trt_mocks
        adapter = TensorRTAdapter()
        adapter.load_model(onnx_file)

        with patch.object(adapter, "collect_hardware_metrics", return_value=None) as mock_hw:
            adapter.run_benchmark({}, self._make_protocol())

        mock_hw.assert_called_once()

    def test_joules_calculated_when_watts_available(self, onnx_file, trt_mocks):
        from calbredge_protocol.models import HardwareMetrics

        _trt, _cuda, _engine, _context = trt_mocks
        adapter = TensorRTAdapter()
        adapter.load_model(onnx_file)

        fake_hm = HardwareMetrics(watts_average=10.0)
        with patch.object(adapter, "collect_hardware_metrics", return_value=fake_hm):
            stats = adapter.run_benchmark({}, self._make_protocol())

        assert adapter._last_hardware_metrics is not None
        assert adapter._last_hardware_metrics.joules_per_inference is not None
        expected = 10.0 * (stats.p50_ms / 1000.0)
        assert abs(adapter._last_hardware_metrics.joules_per_inference - expected) < 1e-9


# ---------------------------------------------------------------------------
# 10. unload
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTimedLoopOrder:
    """The timed loop's order: warm-up untimed and synced first; then, per measured iteration,
    t0, the host-to-device copies, execute, the device-to-host copies, stream sync, t1.

    Records the order of the clock reads and the CUDA calls on fakes, with no change to the
    adapter: the clock is the module's own ``time`` name, the CUDA calls are the mocks. Fails if
    the sync moves after the timer stops, a copy leaves the timed region, or warm-up is timed
    (each checked by mutating run_benchmark by hand when this test was written).
    """

    WARMUP = 2
    MEASURED = 5

    def _protocol(self):
        from calbredge_protocol.models import BenchmarkProtocol

        return BenchmarkProtocol(
            protocol_version="1.0",
            warmup_iterations=self.WARMUP,
            measurement_iterations=self.MEASURED,
            input_shape=[1, 3, 640, 640],
        )

    def _record(self, adapter, cuda, context, monkeypatch) -> list[str]:
        import calbredge_protocol.tensorrt_adapter as tensorrt_adapter

        events: list[str] = []
        ticks = iter(range(10_000))

        def clock() -> float:
            events.append("t")
            return float(next(ticks))

        monkeypatch.setattr(tensorrt_adapter, "time", types.SimpleNamespace(perf_counter=clock))
        cuda.memcpy_htod_async.side_effect = lambda *a, **k: events.append("htod")
        cuda.memcpy_dtoh_async.side_effect = lambda *a, **k: events.append("dtoh")
        context.execute_async_v3.side_effect = lambda *a, **k: events.append("exec")
        cuda.Stream.return_value.synchronize.side_effect = lambda *a, **k: events.append("sync")
        return events

    def _check(self, events, stats, inputs: int, outputs: int):
        iteration = ["t", *["htod"] * inputs, "exec", *["dtoh"] * outputs, "sync", "t"]
        warmup = [*["htod"] * inputs, *["exec"] * self.WARMUP, "sync"]
        first_t = events.index("t")
        # 1. Warm-up is untimed and finished (synced) before the first timer read.
        assert events[:first_t] == warmup
        # 2. Every measured iteration: t0, copies in, execute, copies out, sync, t1.
        assert events[first_t:] == iteration * self.MEASURED
        # 3. No other clock reads.
        assert events.count("t") == 2 * self.MEASURED
        # 4. The samples are exactly those brackets (consecutive fake readings, 1 s apart).
        assert stats.sample_count == self.MEASURED
        assert stats.samples_ms == (1000.0,) * self.MEASURED

    def test_one_input_one_output(self, onnx_file, trt_mocks, monkeypatch):
        _, cuda, _, context = trt_mocks
        adapter = TensorRTAdapter()
        adapter.load_model(onnx_file)
        events = self._record(adapter, cuda, context, monkeypatch)

        stats = adapter.run_benchmark({}, self._protocol())

        self._check(events, stats, inputs=1, outputs=1)

    def test_two_inputs_two_outputs(self, onnx_file, trt_mocks, monkeypatch):
        """Every copy loop stays inside the timed region, not only the first copy."""
        _, cuda, _, context = trt_mocks
        adapter = TensorRTAdapter()
        adapter.load_model(onnx_file)
        adapter._input_buffers = {
            name: (np.zeros(4, dtype=np.float32), MagicMock()) for name in ("a", "b")
        }
        adapter._output_buffers = {
            name: (np.zeros(4, dtype=np.float32), MagicMock()) for name in ("y", "z")
        }
        events = self._record(adapter, cuda, context, monkeypatch)

        stats = adapter.run_benchmark({}, self._protocol())

        self._check(events, stats, inputs=2, outputs=2)


@pytest.mark.unit
class TestUnload:
    def test_unload_clears_is_loaded(self, onnx_file, trt_mocks):
        _trt, _cuda, _engine, _context = trt_mocks
        adapter = TensorRTAdapter()
        adapter.load_model(onnx_file)
        adapter.unload()
        assert not adapter.is_loaded

    def test_unload_clears_buffers_and_names(self, onnx_file, trt_mocks):
        _trt, _cuda, _engine, _context = trt_mocks
        adapter = TensorRTAdapter()
        adapter.load_model(onnx_file)
        adapter.unload()
        assert adapter._input_buffers == {}
        assert adapter._output_buffers == {}
        assert adapter._input_names == []
        assert adapter._output_names == []

    def test_unload_before_load_is_safe(self):
        adapter = TensorRTAdapter()
        adapter.unload()  # must not raise
        assert not adapter.is_loaded

    def test_double_unload_is_safe(self, onnx_file, trt_mocks):
        _trt, _cuda, _engine, _context = trt_mocks
        adapter = TensorRTAdapter()
        adapter.load_model(onnx_file)
        adapter.unload()
        adapter.unload()  # must not raise


# ---------------------------------------------------------------------------
# 11. autodetect integration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAutodetectPrecisionVariants:
    def test_get_adapter_tensorrt_returns_fp32(self):
        adapter = get_adapter("tensorrt")
        assert isinstance(adapter, TensorRTAdapter)
        assert adapter._precision == "fp32"

    def test_get_adapter_tensorrt_fp16(self):
        adapter = get_adapter("tensorrt-fp16")
        assert isinstance(adapter, TensorRTAdapter)
        assert adapter._precision == "fp16"
        assert adapter.runtime_label == "tensorrt-fp16"

    def test_get_adapter_tensorrt_fp32(self):
        adapter = get_adapter("tensorrt-fp32")
        assert isinstance(adapter, TensorRTAdapter)
        assert adapter._precision == "fp32"

    def test_get_adapter_tensorrt_int8(self):
        adapter = get_adapter("tensorrt-int8")
        assert isinstance(adapter, TensorRTAdapter)
        assert adapter._precision == "int8"
        assert adapter.runtime_label == "tensorrt-int8"

    def test_unknown_runtime_raises(self):
        with pytest.raises(ValueError, match="Unknown runtime"):
            get_adapter("tensorrt-bf16")


# ---------------------------------------------------------------------------
# 12. _trt_dtype_to_numpy helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTrtDtypeToNumpy:
    def test_float_maps_to_float32(self):
        trt = _make_trt_mock()
        assert _trt_dtype_to_numpy(trt.DataType.FLOAT, trt) is np.float32

    def test_half_maps_to_float16(self):
        trt = _make_trt_mock()
        assert _trt_dtype_to_numpy(trt.DataType.HALF, trt) is np.float16

    def test_int8_maps_to_int8(self):
        trt = _make_trt_mock()
        assert _trt_dtype_to_numpy(trt.DataType.INT8, trt) is np.int8

    def test_unknown_dtype_defaults_to_float32(self):
        trt = _make_trt_mock()
        assert _trt_dtype_to_numpy(999, trt) is np.float32


# ---------------------------------------------------------------------------
# 13. _make_calibrator
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMakeCalibrator:
    def test_calibrator_get_batch_size_returns_1(self):
        trt = _make_trt_mock()
        cuda = _make_cuda_mock()
        cal = _make_calibrator(
            trt,
            cuda,
            input_shapes={"images": (1, 3, 224, 224)},
            input_dtypes={"images": np.float32},
            n_calib=3,
        )
        assert cal.get_batch_size() == 1

    def test_calibrator_returns_none_after_n_batches(self):
        trt = _make_trt_mock()
        cuda = _make_cuda_mock()
        n = 3
        cal = _make_calibrator(
            trt,
            cuda,
            input_shapes={"images": (1, 3, 4, 4)},
            input_dtypes={"images": np.float32},
            n_calib=n,
        )
        for _ in range(n):
            result = cal.get_batch(["images"])
            assert result is not None
        assert cal.get_batch(["images"]) is None

    def test_calibrator_read_calibration_cache_returns_none(self):
        trt = _make_trt_mock()
        cuda = _make_cuda_mock()
        cal = _make_calibrator(
            trt,
            cuda,
            input_shapes={"images": (1, 3, 4, 4)},
            input_dtypes={"images": np.float32},
        )
        assert cal.read_calibration_cache() is None


# ---------------------------------------------------------------------------
# 14. _get_nvpmodel_mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetNvpmodelMode:
    def _run(self, stdout: str):
        mock_result = MagicMock()
        mock_result.stdout = stdout
        with patch("calbredge_protocol.adapters.subprocess.run", return_value=mock_result):
            return _get_nvpmodel_mode()

    def test_parses_maxn_super(self):
        assert self._run("NV Power Mode: MAXN_SUPER\n0\n") == "maxn_super"

    def test_parses_25w(self):
        assert self._run("NV Power Mode: 25W\n3\n") == "25w"

    def test_parses_7w(self):
        assert self._run("NV Power Mode: 7W\n5\n") == "7w"

    def test_returns_none_on_empty_output(self):
        assert self._run("") is None

    def test_returns_none_when_nvpmodel_not_found(self):
        with patch(
            "calbredge_protocol.adapters.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            assert _get_nvpmodel_mode() is None

    def test_returns_none_on_timeout(self):
        with patch(
            "calbredge_protocol.adapters.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="nvpmodel", timeout=5),
        ):
            assert _get_nvpmodel_mode() is None

    def test_cache_filename_includes_power_mode(self, onnx_file, trt_mocks):
        # trt_mocks patches _get_nvpmodel_mode → "default"
        _trt, _cuda, _engine, _context = trt_mocks
        adapter = TensorRTAdapter(precision="fp16")
        adapter.load_model(onnx_file)
        assert len(_engines(onnx_file, "fp16", "default")) == 1

    def test_different_power_modes_produce_different_cache_files(self, onnx_file, trt_mocks):
        _trt, _cuda, _engine, _context = trt_mocks

        with patch(
            "calbredge_protocol.tensorrt_adapter._get_nvpmodel_mode", return_value="maxn_super"
        ):
            adapter = TensorRTAdapter(precision="fp32")
            adapter.load_model(onnx_file)

        with patch("calbredge_protocol.tensorrt_adapter._get_nvpmodel_mode", return_value="25w"):
            adapter2 = TensorRTAdapter(precision="fp32")
            adapter2.load_model(onnx_file)

        assert len(_engines(onnx_file, "fp32", "maxn_super")) == 1
        assert len(_engines(onnx_file, "fp32", "25w")) == 1
