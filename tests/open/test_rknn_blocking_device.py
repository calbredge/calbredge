"""On-device check that RKNN ``inference()`` returns only when the NPU has finished.

RKNNAdapter.run_benchmark reads the timer straight after ``run_inference`` with no explicit
sync, on the assumption that the call blocks. tests/open/test_rknn_adapter.py::TestTimedLoopOrder
guards the order of the loop on fakes; this file checks the assumption itself, on the RK3588,
through the adapter's own ``run_inference``, pinned to the big cores before the load as
``calbredge run`` is by default. Both checks have to pass.

It needs rknn-toolkit-lite2 and ``~/mobilenetv2.onnx`` with its sibling
``mobilenetv2.onnx.rknn.fp16.default``, and skips without either. Run it with the governors
pinned (``sudo ~/performance_mode.sh``) and ``--noconftest``, so tests/open/conftest.py's
fixtures do not move HOME under the runtime. ``-s`` shows the medians.

    ~/calbredge-venv/bin/pytest --noconftest -s -rs \\
        ~/calbredge-main/tests/open/test_rknn_blocking_device.py
"""

import importlib.util
import os
import statistics
import time
from pathlib import Path

import calbredge_protocol
import numpy as np
import pytest
from calbredge.affinity import apply_affinity, detect_clusters, resolve_affinity
from calbredge_protocol.loader import load_protocol
from calbredge_protocol.rknn_adapter import RKNNAdapter

MODEL = Path.home() / "mobilenetv2.onnx"
RKNN_FILE = Path(str(MODEL) + ".rknn.fp16.default")
PROTOCOL = Path(calbredge_protocol.__file__).parent / "protocols" / "protocol224.yaml"

RKNNLITE_INSTALLED = importlib.util.find_spec("rknnlite") is not None
MODEL_FILES_PRESENT = MODEL.exists() and RKNN_FILE.exists()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RKNNLITE_INSTALLED,
        reason="requires rknn-toolkit-lite2 (RK3588 only), not installed in this environment",
    ),
    pytest.mark.skipif(
        not MODEL_FILES_PRESENT,
        reason=f"requires {MODEL} (present={MODEL.exists()}) and {RKNN_FILE.name} "
        f"(present={RKNN_FILE.exists()})",
    ),
]

SETTLE_S = 0.2
WARMUP = 20
ALTERNATIONS = 50
TIMED = 100
ISOLATION_SLEEP_S = 0.05

# An output must sit within this fraction of the distance between the two references to count
# as its own input's result. The NPU is deterministic, so the expected distance is 0; a result
# belonging to the other input sits at 1.0.
OWN_REFERENCE_FRACTION = 0.01

# If inference() returned with a fraction f of the work still running, a back-to-back bracket
# would still contain it (the next call has to wait for the NPU), while a call made after a
# 50 ms sleep would not (the work finishes during the sleep, outside the bracket), so
# isolated / back-to-back would be about 1 - f. 0.8 fails the check when a fifth or more of
# the work is outside the bracket, and sits ten times MobileNetV2's run-to-run noise (about
# 2%) below 1.0, so noise cannot fail it. Isolated calls being somewhat slower passes.
#
# The comparison only means something with the process pinned to the big cores, which the
# adapter fixture does. Unpinned on the RK3588, a call after 50 ms of idle took about 1.86
# times a back-to-back one (9.67 ms against 5.19 ms, 2026-10-05): the scheduler had moved the
# idle thread, most likely to a LITTLE core, and a penalty that size would hide a partly
# asynchronous return. Pinned to cores 4-7 the ratio was 1.009.
MIN_ISOLATED_OVER_BACK_TO_BACK = 0.8


@pytest.fixture(scope="module")
def adapter():
    # Pinned before the load, as _execute_benchmark does, so librknnrt's threads inherit it.
    unpinned = os.sched_getaffinity(0)
    pinned = apply_affinity(resolve_affinity("big", detect_clusters()))
    print(f"\nrknn blocking, CPU affinity: {sorted(pinned) if pinned else 'not pinned'}")
    try:
        shape = load_protocol(PROTOCOL).input_shape
        rknn = RKNNAdapter(precision="fp16")
        rknn.load_model(MODEL, input_shape=shape)
        yield rknn, shape
        rknn.unload()
    finally:
        os.sched_setaffinity(0, unpinned)


def _flat(outputs: dict) -> np.ndarray:
    return np.concatenate(
        [np.asarray(outputs[k], dtype=np.float64).ravel() for k in sorted(outputs)]
    )


def _timed_ms(rknn: RKNNAdapter, inputs: dict, sleep_s: float) -> list[float]:
    samples = []
    for _ in range(TIMED):
        if sleep_s:
            time.sleep(sleep_s)
        t0 = time.perf_counter()
        rknn.run_inference(inputs)
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)
    return samples


def test_each_output_is_its_own_inputs_result(adapter):
    """Alternating two inputs back to back, every returned output is the result for the input
    just given, so the result is complete when inference() returns, not one call late."""
    rknn, shape = adapter
    rng = np.random.default_rng(0)
    feeds = [
        {"0": np.zeros(shape, dtype=rknn.input_dtype)},
        {"0": rng.integers(0, 256, size=shape).astype(rknn.input_dtype)},
    ]

    references = []
    for feed in feeds:
        rknn.run_inference(feed)
        time.sleep(SETTLE_S)
        references.append(_flat(rknn.run_inference(feed)))
        time.sleep(SETTLE_S)
    apart = float(np.max(np.abs(references[0] - references[1])))
    assert apart > 0, "the two inputs give the same output, so this check can show nothing"

    worst = 0.0
    for i in range(2 * ALTERNATIONS):
        own = references[i % 2]
        distance = float(np.max(np.abs(_flat(rknn.run_inference(feeds[i % 2])) - own)))
        worst = max(worst, distance)
        assert distance <= OWN_REFERENCE_FRACTION * apart, (
            f"call {i}: the output is {distance:.6g} from its own input's reference "
            f"(the two references are {apart:.6g} apart)"
        )
    print(
        f"\nrknn blocking, outputs: {2 * ALTERNATIONS} alternating calls, worst distance from "
        f"own reference {worst:.6g}, references {apart:.6g} apart"
    )


def test_isolated_calls_are_not_faster_than_back_to_back(adapter):
    """A call made after a sleep takes no less time than one made back to back, so no part of
    the work is finishing outside the timed bracket."""
    rknn, shape = adapter
    inputs = {"0": np.random.default_rng(0).integers(0, 256, size=shape).astype(rknn.input_dtype)}
    for _ in range(WARMUP):
        rknn.run_inference(inputs)

    back_to_back = statistics.median(_timed_ms(rknn, inputs, 0.0))
    isolated = statistics.median(_timed_ms(rknn, inputs, ISOLATION_SLEEP_S))
    ratio = isolated / back_to_back
    print(
        f"\nrknn blocking, timing: back-to-back median {back_to_back:.4f} ms, isolated median "
        f"{isolated:.4f} ms, isolated / back-to-back {ratio:.3f} (n={TIMED} each, "
        f"threshold {MIN_ISOLATED_OVER_BACK_TO_BACK})"
    )

    assert ratio >= MIN_ISOLATED_OVER_BACK_TO_BACK, (
        f"isolated calls read {ratio:.3f} of back-to-back: inference() may be returning before "
        "the NPU has finished"
    )
