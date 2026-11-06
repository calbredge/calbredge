"""Tests for IterativePolicyRunner (protocol/calbredge_protocol/iterative_policy.py).

protocol/calbredge_protocol/iterative_policy.py imports torch, which is not a declared runtime
dependency of protocol, agent or api and is not installed in CI. It is the only module that
imports torch at module level (onnx_adapter.py and converter.py import it lazily, inside
functions), so torch's module-level API surface is fully stubbed here
rather than the test being skipped — these tests assert on
IterativePolicyRunner's own arithmetic and control flow (horizon validation,
Pass A/Pass B decomposition, the deadline verdict), not on real CUDA/tensor
behavior.

The stub is installed via monkeypatch.setitem(sys.modules, ...) and
calbredge_protocol.iterative_policy is imported by the iterative_policy fixture, after
the stub is in place — a top-level `import torch` runs at module-import time,
before any fixture can run. Real torch IS installed locally (ultralytics
pulls it in) and protocol/calbredge_protocol/onnx_adapter.py imports it lazily, so nothing
stub-bound may outlive a test: monkeypatch restores sys.modules["torch"],
and the fixture evicts the stub-bound calbredge_protocol.iterative_policy module so a
later import re-binds to whatever torch is really there.
"""

import importlib
import sys
import types

import pytest


def _install_fake_torch(monkeypatch):
    # types.ModuleType doesn't declare these attributes statically, so
    # assigning them is only a real module's dynamic nature, not a type
    # error — pyright flags it because it can't know this is a deliberate
    # stub, not real module state.
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        synchronize=lambda: None,
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 0,
        memory_reserved=lambda: 0,
    )
    fake_torch.backends = types.SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        cudnn=types.SimpleNamespace(benchmark=False)
    )
    fake_torch.isfinite = lambda _tensor: types.SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        all=lambda: True
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return fake_torch


@pytest.fixture
def iterative_policy(monkeypatch):
    """calbredge_protocol.iterative_policy, imported fresh against the torch stub.

    Teardown order matters: this fixture's cleanup runs before
    monkeypatch's (it depends on monkeypatch), so the stub-bound module is
    dropped first, then monkeypatch restores any pre-test entries.
    """
    import calbredge_protocol

    _install_fake_torch(monkeypatch)
    monkeypatch.delitem(sys.modules, "calbredge_protocol.iterative_policy", raising=False)
    module = importlib.import_module("calbredge_protocol.iterative_policy")
    yield module
    sys.modules.pop("calbredge_protocol.iterative_policy", None)
    if getattr(calbredge_protocol, "iterative_policy", None) is module:
        delattr(calbredge_protocol, "iterative_policy")


class _FakeModule:
    """Stand-in for an nn.Module: exposes a patchable .forward, nothing else.

    _install_hooks/_remove_hooks swap this attribute directly rather than
    going through __call__, so a plain lambda is a faithful enough stand-in.
    """

    def __init__(self):
        self.forward = lambda *args, **kwargs: None


class _FakeTensor:
    """Stand-in tensor: only .shape is ever read by IterativePolicyRunner."""

    def __init__(self, shape):
        self.shape = shape


class _FakeHandle:
    """Stand-in for the policy handle IterativePolicyRunner composes over.

    predict_action calls encoder_module.forward once and step_module.forward
    N times per decision, mirroring one encode + N denoise steps, so Pass
    B's hook-based decomposition has real per-call timings to decompose
    instead of an empty sink.
    """

    def __init__(self, *, H=8, Ta=4, To=2, N=3):
        self.H = H
        self.Ta = Ta
        self.To = To
        self.N = N
        self.sched = "ddim"
        self.amp = False
        self.obs = object()
        self.encoder_module = _FakeModule()
        self.step_module = _FakeModule()

    def predict_action(self, _obs):
        self.encoder_module.forward()
        for _ in range(self.N):
            self.step_module.forward()
        return {
            "action": _FakeTensor((1, self.Ta, 2)),
            "action_pred": _FakeTensor((1, self.H, 2)),
        }


@pytest.mark.unit
def test_iterative_policy_runner_produces_a_complete_result(iterative_policy):
    IterativePolicyRunner = iterative_policy.IterativePolicyRunner

    handle = _FakeHandle()
    runner = IterativePolicyRunner(handle, warmup=2, iters=5)
    result = runner.run()

    assert result["H"] == handle.H
    assert result["Ta"] == handle.Ta
    assert result["To"] == handle.To
    assert result["N_denoise"] == handle.N
    assert result["unet_calls_per_decision"] == handle.N
    assert result["T_decision_ms"]["n"] == 5
    assert result["verdict"] in ("runs-meets-deadline", "runs-unstable", "runs-too-slow")
    # No tegrastats binary on CI: TegrastatsWindowSampler degrades to null
    # power (protocol/calbredge_protocol/power_window.py), never fails the run.
    assert result["watts_avg"] is None
    assert result["power_samples"] == 0


@pytest.mark.unit
def test_iterative_policy_runner_rejects_wrong_action_horizon(iterative_policy, monkeypatch):
    IterativePolicyRunner = iterative_policy.IterativePolicyRunner

    handle = _FakeHandle(Ta=4)
    bad_action = _FakeTensor((1, handle.Ta + 1, 2))  # wrong horizon
    monkeypatch.setattr(
        handle,
        "predict_action",
        lambda _obs: {"action": bad_action, "action_pred": _FakeTensor((1, handle.H, 2))},
    )
    runner = IterativePolicyRunner(handle, warmup=1, iters=1)

    with pytest.raises(AssertionError, match="action horizon"):
        runner.run()


@pytest.mark.unit
def test_iterative_policy_runner_rejects_wrong_pred_horizon(iterative_policy, monkeypatch):
    IterativePolicyRunner = iterative_policy.IterativePolicyRunner

    handle = _FakeHandle(H=8)
    bad_pred = _FakeTensor((1, handle.H + 1, 2))  # wrong pred horizon
    monkeypatch.setattr(
        handle,
        "predict_action",
        lambda _obs: {"action": _FakeTensor((1, handle.Ta, 2)), "action_pred": bad_pred},
    )
    runner = IterativePolicyRunner(handle, warmup=1, iters=1)

    with pytest.raises(AssertionError, match="pred horizon"):
        runner.run()
