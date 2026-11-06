from pathlib import Path

from .adapters import RuntimeAdapter


class MockRuntimeAdapter(RuntimeAdapter):
    """Deterministic fake RuntimeAdapter for CI and local development.

    This adapter satisfies the full RuntimeAdapter interface without requiring
    any model file, runtime library, or edge hardware. It is intended
    exclusively for unit tests, CI pipelines, and local development on
    non-edge machines.

    **Do not use MockRuntimeAdapter in production benchmarks.** Its latency
    values reflect Python method-call overhead on the host machine, not real
    inference performance. Results produced by this adapter are meaningless as
    hardware benchmarks and must never be uploaded to the dashboard as genuine
    measurements.

    The adapter produces a fixed, deterministic output on every call to
    :meth:`run_inference` so that test assertions are stable across runs and
    environments.
    """

    def __init__(self) -> None:
        self.model_path: str | Path | None = None
        self.is_loaded: bool = False

    # ------------------------------------------------------------------
    # RuntimeAdapter interface
    # ------------------------------------------------------------------

    def load_model(self, model_path: str | Path) -> None:
        """Record the model path and mark the adapter as loaded.

        Does not check whether *model_path* exists on disk — the mock
        requires no real model file.

        Args:
            model_path: Any string or Path value; stored as-is.
        """
        self.model_path = model_path
        self.is_loaded = True

    def warmup(self, inputs: dict, iterations: int) -> None:
        """Run *iterations* fake inference passes and discard all results.

        Handles ``iterations=0`` cleanly by returning immediately.

        Args:
            inputs: Forwarded verbatim to :meth:`run_inference`; not inspected.
            iterations: Number of warmup passes. Must be >= 0.
        """
        for _ in range(iterations):
            self.run_inference(inputs)

    def run_inference(self, inputs: dict) -> dict:
        """Return a fixed, deterministic output dict without any added latency.

        The returned value is always ``{"output": [0.1, 0.2, 0.3]}``.
        No randomness, no sleep — the only latency measured by
        :meth:`~protocol.adapters.RuntimeAdapter.run_benchmark` is the real
        CPU time of this Python method call.

        Args:
            inputs: Ignored. The mock does not inspect or validate inputs.

        Returns:
            ``{"output": [0.1, 0.2, 0.3]}`` on every call.
        """
        return {"output": [0.1, 0.2, 0.3]}

    def unload(self) -> None:
        """Release the mock model and reset internal state.

        Idempotent — safe to call before :meth:`load_model` or multiple times.
        """
        self.is_loaded = False
        self.model_path = None

    # ------------------------------------------------------------------
    # Additional API
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """Return ``True`` if a model has been loaded and not yet unloaded."""
        return self.is_loaded
