# protocol/workloads/iterative_policy.py
import math
import time

import numpy as np
import torch

from .power_window import TegrastatsWindowSampler


def _pct(samples_s):
    """Percentiles in ms from seconds. Byte-for-byte match to spike_dp.pct
    (np.percentile, linear interpolation)."""
    v = np.asarray(samples_s) * 1e3
    return {
        "p50": round(float(np.percentile(v, 50)), 3),
        "p90": round(float(np.percentile(v, 90)), 3),
        "p99": round(float(np.percentile(v, 99)), 3),
        "max": round(float(v.max()), 3),
        "n": len(v),
    }


class IterativePolicyRunner:
    """Measures an action-chunked iterative policy: one decision = one chunk =
    N sequential denoise steps, NOT one forward pass. Sibling to RuntimeAdapter,
    not a subclass, because RuntimeAdapter fixes the measured unit at one forward
    pass. Composes a policy handle exposing:
        .predict_action(obs) -> {"action": (1,Ta,Da), "action_pred": (1,H,Da)}
        .obs                  (prebuilt synthetic obs, reused every decision)
        .H .Ta .To .N .sched .amp
        .encoder_module .step_module   (for optional Pass-B decomposition)

    Pass A: end-to-end T_decision, sync only around the whole decision, no inner
    hooks. Headline number and seam-check target.
    Pass B: decomposition with per-call syncs on encoder/step submodules. Heavier
    (syncs kill overlap), used ONLY for the encode/step breakdown, kept separate
    exactly as the spike does.
    """

    def __init__(self, handle, *, warmup=5, iters=50, fc=10.0, rail="VDD_IN"):
        self.h = handle
        self.warmup = warmup
        self.iters = iters
        self.fc = fc
        self.rail = rail
        self._orig = {}

    def _decide(self):
        return self.h.predict_action(self.h.obs)

    def _install_hooks(self, enc_sink, step_sink):
        sync = torch.cuda.synchronize
        self._orig = {}
        for mod, sink in ((self.h.encoder_module, enc_sink), (self.h.step_module, step_sink)):
            orig = mod.forward
            self._orig[mod] = orig

            def make(orig, sink):
                def f(*x, **kw):
                    sync()
                    t0 = time.perf_counter()
                    y = orig(*x, **kw)
                    sync()
                    sink.append(time.perf_counter() - t0)
                    return y

                return f

            mod.forward = make(orig, sink)

    def _remove_hooks(self):
        for mod, orig in self._orig.items():
            mod.forward = orig
        self._orig = {}

    def run(self):
        h = self.h
        sync = torch.cuda.synchronize
        torch.backends.cudnn.benchmark = True

        # cold inference: cuDNN autotune + shape/finite validation (spike Step 4)
        sync()
        r = self._decide()
        sync()
        if r["action"].shape[1] != h.Ta:
            raise AssertionError("action horizon != Ta")
        if r["action_pred"].shape[1] != h.H:
            raise AssertionError("pred horizon != H")
        if not torch.isfinite(r["action_pred"]).all():
            raise AssertionError("non-finite actions (check amp)")

        for _ in range(self.warmup):
            self._decide()
        sync()

        # Pass A: end-to-end, bracketed by the windowed sampler
        power = TegrastatsWindowSampler(rail=self.rail).start()
        torch.cuda.reset_peak_memory_stats()
        T = []
        for _ in range(self.iters):
            sync()
            t0 = time.perf_counter()
            self._decide()
            sync()
            T.append(time.perf_counter() - t0)
        peak_alloc_mb = torch.cuda.max_memory_allocated() / 2**20
        power.stop()

        # Pass B: decomposition with inner hooks (separate pass, per spike)
        enc_t, step_t = [], []
        self._install_hooks(enc_t, step_t)
        nB = max(10, self.iters // 5)
        TB = []
        for _ in range(nB):
            sync()
            t0 = time.perf_counter()
            self._decide()
            sync()
            TB.append(time.perf_counter() - t0)
        self._remove_hooks()
        calls = len(step_t) // nB if nB else 0
        other = (
            [tb - sum(step_t[i * calls : (i + 1) * calls]) - enc_t[i] for i, tb in enumerate(TB)]
            if calls and len(enc_t) == nB
            else []
        )

        A = _pct(T)
        fc = self.fc
        future = h.H - (h.To - 1)
        hard = future / (2 * fc) * 1e3
        naive = h.Ta / fc * 1e3

        def stale(ms):
            return math.ceil(ms / 1e3 * fc)

        W = power.watts_avg

        return {
            "sched": h.sched,
            "amp": h.amp,
            "H": h.H,
            "Ta": h.Ta,
            "To": h.To,
            "N_denoise": h.N,
            "unet_calls_per_decision": calls,
            "T_decision_ms": A,
            "T_decision_synced_ms": _pct(TB),
            "T_encode_ms": _pct(enc_t) if enc_t else None,
            "T_denoise_step_ms": _pct(step_t) if step_t else None,
            "T_other_ms(sched+norm+decode)": _pct(other) if other else None,
            "sync_inflation_pct": round((np.median(TB) / np.median(T) - 1) * 100, 1),
            "cuda_peak_alloc_MB": round(peak_alloc_mb),
            "cuda_reserved_MB": round(torch.cuda.memory_reserved() / 2**20),
            "rail": self.rail,
            "watts_avg": round(W, 3) if W else None,
            "watts_peak": round(power.watts_peak, 3) if power.watts_peak else None,
            "power_samples": len(power.watts),
            "sys_RAM_peak_MB": power.ram_peak_mb,
            # guarded on W (watts_avg), not this property directly, since both
            # derive from the same self.watts truthiness — pyright can't see
            # that equivalence across two separate property reads.
            "energy_joules_window": round(power.energy_joules_window, 3)  # pyright: ignore[reportCallIssue, reportArgumentType]
            if W
            else None,
            "J_per_chunk_p50": round(W * A["p50"] / 1e3, 3) if W else None,
            "J_per_executed_action": round(W * A["p50"] / 1e3 / h.Ta, 4) if W else None,
            "deadline_naive_ms": naive,
            "deadline_hard_gapless_ms": hard,
            "stale_actions_p50": stale(A["p50"]),
            "stale_actions_p99": stale(A["p99"]),
            "fresh_actions_left_p99": future - stale(A["p99"]),
            "verdict": (
                "runs-meets-deadline"
                if A["p99"] <= hard and A["p99"] / A["p50"] <= 2.0
                else "runs-unstable"
                if A["p50"] <= hard
                else "runs-too-slow"
            ),
        }
