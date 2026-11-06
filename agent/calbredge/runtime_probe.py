"""One inference in a subprocess before a benchmark, so a runtime crash can be reported.

A model with an operator the RKNN runtime cannot run loads and initialises, then segfaults
inside ``inference()``: the calling process dies with nothing recorded. A caller therefore
first runs one load, init and inference in a child process (``probe``), with the same adapter,
the same ``.rknn`` lookup, the same CPU affinity and the same input as the benchmark, and gets
back a plain ``ProbeResult`` saying what happened, read from outside:

- ``unsupported_ops``: librknnrt names operators it cannot run at ``init_runtime``. Any such
  line on the child's stdout or stderr is collected, whether or not the child then crashed.
- ``crashed``: set when the child was killed by a signal, or was still running at the timeout.
- Anything else, including an ordinary Python error in the child, sets neither: the caller's
  in-process load then raises the same error it always has.

What a result means is the caller's decision; the gate turns the first two into verdicts.
Only the runtime families in ``PROBED_RUNTIME_FAMILIES`` are probed. The benchmark itself stays
in the caller's own process and is not changed by any of this.
"""

import argparse
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROBED_RUNTIME_FAMILIES = frozenset({"rknn"})
# One load, init and inference; generous, because a hung probe is itself the finding.
PROBE_TIMEOUT_S = 120.0

# "E RKNN: [18:43:19.369] Unsupport CPU op: GridSample in this librknnrt.so, please try ..."
UNSUPPORTED_OP = re.compile(r"Unsupport(?:ed)? CPU op: (\S+) in this librknnrt")

# Not ``-m``: that puts the working directory first on sys.path, where a directory named
# ``calbredge`` (a source checkout in the home directory, say) would be imported instead of the
# installed package.
_CHILD = (
    "import sys; sys.path[:] = [p for p in sys.path if p]; "
    "from calbredge.runtime_probe import main; sys.exit(main(sys.argv[1:]))"
)


@dataclass(frozen=True)
class ProbeResult:
    unsupported_ops: tuple[str, ...]
    crashed: str | None
    output: str
    seconds: float
    # The child's exit code; None when it was killed by a signal or timed out. Nonzero means
    # it ended with an ordinary error, which the caller's own load then reports.
    returncode: int | None = None


def needs_probe(runtime_label: str) -> bool:
    return runtime_label.split("-", 1)[0] in PROBED_RUNTIME_FAMILIES


def _text(stream: str | bytes | None) -> str:
    if stream is None:
        return ""
    return stream.decode("utf-8", errors="replace") if isinstance(stream, bytes) else stream


def _signal_name(number: int) -> str:
    try:
        return signal.Signals(number).name
    except ValueError:
        return f"signal {number}"


def probe(
    model: Path,
    runtime_label: str,
    protocol_path: Path | None,
    cpu_affinity: str,
    *,
    timeout_s: float = PROBE_TIMEOUT_S,
    command: list[str] | None = None,
) -> ProbeResult:
    """Run the child and report what it printed and how it ended. ``command`` is for tests."""
    argv = command or [
        sys.executable,
        "-c",
        _CHILD,
        "--model",
        str(model),
        "--runtime",
        runtime_label,
        "--cpu-affinity",
        cpu_affinity,
        *(["--protocol", str(protocol_path)] if protocol_path is not None else []),
    ]
    started = time.monotonic()
    crashed = None
    returncode = None
    try:
        done = subprocess.run(  # noqa: S603 — fixed argv built here; no shell
            argv, capture_output=True, text=True, errors="replace", timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired as exc:
        output = _text(exc.stdout) + _text(exc.stderr)
        crashed = f"timed out after {timeout_s:g} s"
    else:
        output = done.stdout + done.stderr
        if done.returncode < 0:
            crashed = f"killed by {_signal_name(-done.returncode)}"
        else:
            returncode = done.returncode
    return ProbeResult(
        unsupported_ops=tuple(dict.fromkeys(UNSUPPORTED_OP.findall(output))),
        crashed=crashed,
        output=output,
        seconds=time.monotonic() - started,
        returncode=returncode,
    )


def _no_core_dump() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, ValueError, OSError):
        pass


def main(argv: list[str] | None = None) -> int:
    """The child: load, init and one inference, as ``_execute_benchmark`` sets them up."""
    import numpy as np
    from calbredge_protocol.autodetect import get_adapter

    from calbredge.affinity import apply_affinity, detect_clusters, resolve_affinity
    from calbredge.cli import _load_protocol_or_default

    parser = argparse.ArgumentParser(prog="calbredge.runtime_probe")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--cpu-affinity", default="big")
    args = parser.parse_args(argv)

    _no_core_dump()
    # Before the load, so the runtime's own threads inherit the pin.
    apply_affinity(resolve_affinity(args.cpu_affinity, detect_clusters()))
    protocol = _load_protocol_or_default(args.protocol)
    # Any: the probed adapters take input_shape, which the base class's load_model does not.
    adapter: Any = get_adapter(args.runtime)
    adapter.load_model(args.model, input_shape=protocol.input_shape)
    names = getattr(adapter, "input_names", None)
    inputs = {
        names[0] if names else "input": np.random.randn(*protocol.input_shape).astype(
            adapter.input_dtype
        )
    }
    adapter.run_inference(inputs)
    adapter.unload()
    return 0
