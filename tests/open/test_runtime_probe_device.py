"""On-device check of the runtime probe, on the RK3588.

rtdetr_s converts to RKNN FP16, loads and initialises, then segfaults at inference because
librknnrt 2.3.2 has no GridSample operator. The probe must turn that into a refusal that names
the operator, with this process still alive; a model the runtime can run must pass. The time
each probe takes is printed (``-s``): it is what the probe adds to a gate run.

It needs rknn-toolkit-lite2 and the models in the home directory, each with its sibling
``<model>.onnx.rknn.fp16.default``; a case skips when its model is missing. Run it with
``--noconftest``, so tests/open/conftest.py's fixtures do not move HOME under the runtime.

    ~/calbredge-venv/bin/pytest --noconftest -s -rs \\
        ~/calbredge-main/tests/open/test_runtime_probe_device.py
"""

import importlib.util
from pathlib import Path

import calbredge_protocol
import pytest
from calbredge.runtime_probe import probe

PROTOCOLS = Path(calbredge_protocol.__file__).parent / "protocols"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        importlib.util.find_spec("rknnlite") is None,
        reason="requires rknn-toolkit-lite2 (RK3588 only), not installed in this environment",
    ),
]


def _model(name: str) -> Path:
    model = Path.home() / f"{name}.onnx"
    rknn = Path(f"{model}.rknn.fp16.default")
    if not (model.exists() and rknn.exists()):
        pytest.skip(f"requires {model} (present={model.exists()}) and {rknn.name}")
    return model


def test_rtdetr_s_is_refused_for_gridsample_and_the_crash_is_contained():
    result = probe(_model("rtdetr_s"), "rknn-fp16", PROTOCOLS / "yolo640.yaml", "big")
    print(
        f"\nruntime probe, rtdetr_s: unsupported {list(result.unsupported_ops)}, "
        f"child {result.crashed or 'exited'}, {result.seconds:.2f} s"
    )
    assert result.unsupported_ops == ("GridSample",)
    # The child died as the caller would have; this process is the one still running.
    assert result.crashed is not None
    assert result.crashed.startswith("killed by")


@pytest.mark.parametrize(
    ("name", "protocol_file"),
    [("mobilenetv2", "protocol224.yaml"), ("yolov8m", "yolo640.yaml")],
)
def test_a_model_the_runtime_can_run_passes(name, protocol_file):
    result = probe(_model(name), "rknn-fp16", PROTOCOLS / protocol_file, "big")
    print(f"\nruntime probe, {name}: {result.seconds:.2f} s")
    assert result.unsupported_ops == (), result.output
    assert result.crashed is None, result.output
