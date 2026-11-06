import importlib.metadata
import inspect
import json
import sys
import warnings
from importlib.resources import as_file, files
from pathlib import Path

import click
import numpy as np
import yaml

from calbredge import __version__
from calbredge.affinity import apply_affinity, detect_clusters, resolve_affinity
from calbredge.bundle import DEFAULT_BUNDLE_DIR, build_environment, secret_values, write_bundle
from calbredge.clock_state import collect_clock_state
from calbredge.collector import collect_metadata
from calbredge.converter import convert_keras, convert_pytorch, convert_rknn
from calbredge.debug import format_convert_debug_block, format_run_debug_block
from calbredge.fingerprint import compute_hardware_id
from calbredge.hasher import compute_model_hash
from calbredge import result_builder
from calbredge.onnx_contract import load_graph_contract
from calbredge.result_builder import build_result_v2
from calbredge.uploader import save_result_locally, upload_or_save
from calbredge_protocol import (
    BenchmarkProtocol,
    BenchmarkResult,
    HardwareMetrics,
    compute_throughput,
    load_protocol,
)
from calbredge_protocol.autodetect import detect_runtime, get_adapter
from calbredge_protocol.derive_v2 import UNDETERMINED_REASON, clocks_pinned

# Name of the canonical v1.0 protocol, shipped as package data of
# protocol.protocols (see protocol/pyproject.toml's [tool.setuptools.package-data]).
# Resolved via importlib.resources at use-time rather than a module-level Path:
# load_protocol() needs a real filesystem path, but the package may be inside a
# zipped wheel, so materializing it has to happen through an as_file() context
# right before the read — a __file__-relative path only works for editable/
# source-tree installs and silently breaks a plain `pip install calbredge`.
_DEFAULT_PROTOCOL_NAME = "v1.0.yaml"

# Sourced from BenchmarkProtocol's own field defaults so this help text can't
# drift from the protocol it describes.
_DEFAULT_WARMUP_ITERATIONS = BenchmarkProtocol.model_fields["warmup_iterations"].default
_DEFAULT_MEASUREMENT_ITERATIONS = BenchmarkProtocol.model_fields["measurement_iterations"].default


# ─── Config ───────────────────────────────────────────────────────────────────

CONFIG_PATH = Path.home() / ".calbredge" / "config.yaml"


def _load_config() -> dict:
    """Load ~/.calbredge/config.yaml, returning an empty dict if absent or unreadable."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        with CONFIG_PATH.open() as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ─── Help formatter ───────────────────────────────────────────────────────────


class _SpacedHelpFormatter(click.HelpFormatter):
    """HelpFormatter that inserts a blank line between each option entry."""

    def write_dl(self, rows, col_max: int = 30, col_spacing: int = 2) -> None:
        spaced: list = []
        for i, row in enumerate(rows):
            spaced.append(row)
            if i < len(rows) - 1:
                spaced.append(("", ""))
        super().write_dl(spaced, col_max=col_max, col_spacing=col_spacing)


class _SpacedContext(click.Context):
    formatter_class = _SpacedHelpFormatter


class _SpacedCommand(click.Command):
    context_class = _SpacedContext


# ─── Stability classification ─────────────────────────────────────────────────


def classify_stability(p50: float, p99: float) -> tuple[str, str | None]:
    """Classify a benchmark run as stable, unstable, or unknown.

    Uses the p99/p50 latency ratio as a proxy for runtime consistency.
    A high ratio indicates that tail latencies are disproportionately worse
    than the median — a signature of thermal throttling, OS scheduling
    interference, or sustained resource contention.

    Args:
        p50: Median latency in milliseconds.
        p99: 99th-percentile latency in milliseconds.

    Returns:
        A tuple of (status, status_reason) where status_reason is None for
        stable runs and a human-readable explanation otherwise.
    """
    if p50 <= 0:
        return "unknown", "p50 is zero or negative"
    ratio = p99 / p50
    if ratio > 2.5:
        return (
            "unstable",
            f"p99/p50 ratio: {ratio:.1f} — possible thermal throttling or system load",
        )
    return "stable", None


# ─── Runtime label resolution ────────────────────────────────────────────────

# The two dispatch-table keys (protocol/calbredge_protocol/autodetect.py's _RUNTIME_TO_ADAPTER)
# with an implicit, adapter-chosen default precision: bare "rknn" defaults to
# fp16, bare "tensorrt" defaults to fp32. Every OTHER precision-qualified key
# (tensorrt-fp16, rknn-int8, etc.) is constructed with exactly the precision
# its name says, so it must already equal that adapter's own runtime_label —
# see _resolve_recorded_runtime's conflict check below.
_BARE_RUNTIME_DEFAULTS = ("rknn", "tensorrt")


def _resolve_recorded_runtime(requested: str | None, resolved: str, adapter) -> str:
    """Compute the runtime string to store on BenchmarkResult.runtime.

    Why this exists: cli.py used to relabel onnxruntime only (provider-based),
    while TensorRTAdapter/RKNNAdapter both expose a runtime_label property that
    was never read, so e.g. bare `--runtime rknn` and `--runtime rknn-fp16`
    recorded different runtime strings for the same fp16 precision.

    Args:
        requested: the raw --runtime value the user or config file gave,
                   or None if neither gave one (i.e. it came from
                   detect_runtime()'s extension-based autodetection).
        resolved: the runtime string actually passed to get_adapter() —
                  `requested or detect_runtime(model)`.
        adapter: the adapter instance returned by get_adapter(resolved).

    Returns:
        The runtime string to store on the result.

    Raises:
        click.ClickException: if requested is an already precision-qualified
            string (e.g. "tensorrt-int8") that disagrees with the adapter's
            own runtime_label — a configuration error, never silently
            resolved in either direction.
    """
    label = getattr(adapter, "runtime_label", None)
    if label is not None:
        if requested is not None and requested not in _BARE_RUNTIME_DEFAULTS and requested != label:
            raise click.ClickException(
                f"--runtime {requested!r} does not match the adapter's resolved "
                f"precision ({label!r}). Pass {label!r}, or drop the precision "
                "suffix to use the adapter's default."
            )
        return label
    # No runtime_label (onnxruntime, tflite, mock): preserve the one existing
    # special case, qualifying onnxruntime by its active execution provider so
    # the dashboard can distinguish GPU vs CPU runs. Everything else (tflite,
    # mock) passes through unchanged, exactly as before this fix.
    if resolved == "onnxruntime":
        active_provider = getattr(adapter, "_active_provider", None)
        if active_provider == "CUDAExecutionProvider":
            return "onnxruntime-cuda"
        if active_provider == "CPUExecutionProvider":
            return "onnxruntime-cpu"
    return resolved


# ─── Benchmark execution core (shared by `run` and `gate`) ────────────────────


def _protocol_origin(protocol_path: Path) -> str | None:
    """The source `calbredge init` recorded in the first line of a protocol copy, if any."""
    from calbredge.init import ORIGIN_PREFIX

    try:
        with protocol_path.open(encoding="utf-8") as f:
            first = f.readline().strip()
    except OSError:
        return None
    if first.startswith(ORIGIN_PREFIX):
        return first[len(ORIGIN_PREFIX) :].rstrip(".") or None
    return None


def _load_protocol_or_default(protocol_path: Path | None) -> BenchmarkProtocol:
    if protocol_path is None:
        with as_file(
            files("calbredge_protocol.protocols") / _DEFAULT_PROTOCOL_NAME
        ) as default_path:
            return load_protocol(default_path)
    return load_protocol(protocol_path)


def _execute_benchmark(
    *,
    model: Path,
    runtime: str | None,
    protocol_path: Path | None,
    model_label: str | None,
    cpu_affinity: str,
    state: dict,
    echo=click.echo,
) -> BenchmarkResult:
    """Run one full benchmark pass and return the assembled BenchmarkResult.

    This is the measurement core shared by `run` and `gate`: CPU-affinity
    resolution, adapter dispatch, protocol load, model load, the timed
    benchmark loop, and BenchmarkResult assembly. It contains no upload
    logic — callers own that, and (via `state`) their own `--debug`
    diagnostics and `adapter.unload()` cleanup on failure.

    `echo` is the progress-line sink, defaulting to `click.echo` (stdout) so
    `run`'s output is byte-identical to before this function existed. `gate`
    passes an echo that writes to stderr instead, so its stdout is exactly
    one JSON document and nothing else — no progress lines interleaved.

    `state` is a plain dict the caller owns and passes in (start with `{}`).
    It is written progressively as each step succeeds: `state["adapter"]`,
    `state["resolved_runtime"]`, `state["protocol_version"]`,
    `state["metadata"]`, `state["model_hash"]`, `state["hardware_id"]`,
    `state["run_status"]`, `state["run_status_reason"]`, `state["stats"]`,
    `state["throughput"]`, `state["affinity_record"]`. This is not optional
    bookkeeping: a caller whose try/finally wraps this call needs whatever
    was computed before an exception, exactly as the pre-extraction inline
    code did — `run`'s `--debug` block and its `adapter.unload()` on a
    mid-run failure both depend on it.

    Raises whatever the underlying steps raise (ValueError, FileNotFoundError,
    NotImplementedError, or any adapter-specific exception) — callers decide
    how to present those; this function does not catch or reinterpret them.
    """
    # 0b. Resolve and apply CPU affinity — must happen before the adapter
    # loads the model so runtime-internal threads (e.g. RKNNLite's
    # init_runtime()) inherit the pin; applying it later would miss any
    # thread already spawned.
    clusters = detect_clusters()
    applied_affinity = apply_affinity(resolve_affinity(cpu_affinity, clusters))
    if applied_affinity is not None:
        tier = cpu_affinity if cpu_affinity in ("big", "little") else "custom"
        affinity_record: dict = {
            "requested": cpu_affinity,
            "applied": sorted(applied_affinity),
            "tier": tier,
        }
        if tier in ("big", "little") and clusters is not None:
            affinity_record["capacity"] = clusters["capacities"][next(iter(applied_affinity))]
        cores_str = ",".join(str(c) for c in affinity_record["applied"])
        if tier in ("big", "little"):
            cap = affinity_record.get("capacity")
            cap_str = f" (cap {cap})" if cap is not None else ""
            echo(
                f"CPU affinity: pinned to {cpu_affinity} cores [{cores_str}]{cap_str} "
                "— reproducible; for un-pinned deployment estimate re-run with "
                "--cpu-affinity none"
            )
        else:
            echo(f"CPU affinity: pinned to cores [{cores_str}] (--cpu-affinity {cpu_affinity})")
    elif cpu_affinity == "none":
        affinity_record = {"requested": "none", "applied": None}
        echo("CPU affinity: none (un-pinned; ~9% run-to-run variance expected on big.LITTLE)")
    else:
        affinity_record = {
            "requested": cpu_affinity,
            "applied": None,
            "reason": "homogeneous CPU",
        }
        echo("CPU affinity: not needed (all cores are the same type)")
    state["affinity_record"] = affinity_record

    # 1. Resolve runtime
    resolved_runtime = runtime or detect_runtime(model)
    state["resolved_runtime"] = resolved_runtime
    echo(f"Runtime:  {resolved_runtime}")

    # 2. Get adapter instance
    adapter = get_adapter(resolved_runtime)
    state["adapter"] = adapter

    # 3. Load protocol. A bare path materializes as-is; the packaged
    # default has to be extracted from the installed package first
    # (works whether it's on disk or inside a zipped wheel) — the
    # resulting BenchmarkProtocol object is fully parsed and independent
    # of the file, so it's fine for the context to close right after.
    if protocol_path is None:
        with as_file(
            files("calbredge_protocol.protocols") / _DEFAULT_PROTOCOL_NAME
        ) as default_path:
            protocol = load_protocol(default_path)
    else:
        protocol = load_protocol(protocol_path)
    state["protocol_version"] = f"v{protocol.protocol_version}"
    # The file, not protocol_version: every bundled protocol says 1.0. A copy made by
    # `calbredge init` also names where it came from.
    protocol_name = protocol_path.name if protocol_path is not None else _DEFAULT_PROTOCOL_NAME
    origin = _protocol_origin(protocol_path) if protocol_path is not None else None
    if origin and origin != protocol_name:
        protocol_name = f"{protocol_name} (from {origin})"
    echo(
        f"Protocol: {protocol_name}, input {list(protocol.input_shape)}, "
        f"{protocol.warmup_iterations} warmup + {protocol.measurement_iterations} measured"
    )

    # 4. Collect system metadata
    metadata = collect_metadata(resolved_runtime)
    state["metadata"] = metadata

    # 5. Compute model hash
    echo(f"Hashing model: {model.name} …")
    model_hash = compute_model_hash(model)
    state["model_hash"] = model_hash

    # 6. Compute hardware_id
    hardware_id = compute_hardware_id(metadata)
    state["hardware_id"] = hardware_id

    # 7. Load model
    echo("Loading model …")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")
        # Pass input_shape for TRT adapters so _build_engine can resolve
        # dynamic dims when creating the optimization profile.
        if "model_hash" in inspect.signature(adapter.load_model).parameters:
            # Content-addressed engine cache (TensorRT): the build key needs the
            # model's hash and input signature. The signature comes from the
            # cached ONNX contract, so a cache hit never re-parses a large model,
            # and a miss parses it before any engine is loaded.
            contract, _ = load_graph_contract(
                model, model_hash, result_builder.DEFAULT_CONTRACT_CACHE_DIR
            )
            adapter.load_model(
                model,
                input_shape=protocol.input_shape,
                model_hash=model_hash,
                input_signature=contract.io_signature["inputs"],
            )
        elif hasattr(adapter, "_precision"):
            adapter.load_model(model, input_shape=protocol.input_shape)
        else:
            adapter.load_model(model)

    # 8. Build synthetic inputs
    input_name = (
        adapter.input_names[0]
        if hasattr(adapter, "input_names") and adapter.input_names
        else "input"
    )
    inputs = {input_name: np.random.randn(*protocol.input_shape).astype(adapter.input_dtype)}

    # 9. Run benchmark
    echo(
        f"Benchmarking ({protocol.warmup_iterations} warmup + "
        f"{protocol.measurement_iterations} measurement) …"
    )
    stats = adapter.run_benchmark(inputs, protocol)
    state["stats"] = stats

    # 10. Compute throughput
    throughput = compute_throughput(protocol.batch_size, stats.p50_ms)
    state["throughput"] = throughput

    # 10b. Hardware metrics collected inside run_benchmark(); retrieve result
    hardware_metrics = adapter._last_hardware_metrics

    # 10b2. Clock state (RK3588 only) collected inside RKNNAdapter.run_benchmark();
    # other adapters never set this attribute, so it stays None for them.
    clock_state = getattr(adapter, "_last_clock_state", None)

    # 10b3. What the v2 run bundle needs from this run (RESULT_SCHEMA.md §6).
    state["protocol"] = protocol
    state["adapter_clock_state"] = clock_state
    state["engine_path"] = getattr(adapter, "_engine_path", None)
    state["telemetry_boundary"] = getattr(adapter, "_last_telemetry_boundary", None)
    state["engine_build"] = getattr(adapter, "_engine_build", None)
    state["engine_io"] = getattr(adapter, "_engine_io", None)

    # 10c. Classify run stability from p99/p50 ratio (legacy 2-tier, 2.5 threshold)
    status, status_reason = classify_stability(stats.p50_ms, stats.p99_ms)

    # 10d. Read 3-tier run_status computed by the adapter (3.0/2.0 thresholds)
    run_status: str = getattr(adapter, "_last_run_status", "stable")
    run_status_reason: str | None = getattr(adapter, "_last_run_status_reason", None)
    state["run_status"] = run_status
    state["run_status_reason"] = run_status_reason

    # 10e. Resolve the runtime string to record — adapter.runtime_label for
    # TensorRT/RKNN (so one precision always records one string), the
    # onnxruntime provider-qualify special case otherwise. See _resolve_recorded_runtime's docstring.
    resolved_runtime = _resolve_recorded_runtime(runtime, resolved_runtime, adapter)
    state["resolved_runtime"] = resolved_runtime

    # 11. Assemble BenchmarkResult
    # Prefer the adapter's artifact-resolved shape; fall back to the protocol
    # shape only when the adapter could not resolve one (RKNN, dynamic ONNX).
    effective_input_shape = getattr(adapter, "resolved_input_shape", None) or list(
        protocol.input_shape
    )
    result = BenchmarkResult.from_run(
        protocol=protocol,
        stats=stats,
        throughput=throughput,
        hardware_id=hardware_id,
        runtime=resolved_runtime,
        model_hash=model_hash,
        model_label=model_label,
        metadata=metadata,
        hardware_metrics=hardware_metrics,
        status=status,
        status_reason=status_reason,
        run_status=run_status,
        run_status_reason=run_status_reason,
        input_shape=effective_input_shape,
        clock_state=clock_state,
        cpu_affinity=affinity_record,
    )
    return result


# ─── Hardware metrics summary ─────────────────────────────────────────────────


# The console's note for power read once after the timed loop (telemetry boundary
# post_loop_snapshot), until power is sampled during the loop. The recorded values are unchanged.
_POST_LOOP_POWER_NOTE = "provisional: read after the timed loop, not during it"


def _hardware_summary_lines(
    hm: HardwareMetrics | None, telemetry_boundary: str | None = None
) -> list[str]:
    """Return printable summary lines for hardware metrics.

    Returns an empty list when hm is None.  Individual lines are omitted
    when all values that would appear on that line are None. Power read after the
    timed loop (post_loop_snapshot) is labelled provisional, and J/inference, which
    multiplies that reading by the p50, is left out of the console.
    """
    if hm is None:
        return []

    lines: list[str] = []

    if telemetry_boundary == "post_loop_snapshot":
        watts = [
            f"{value:.2f} W {label}"
            for value, label in ((hm.watts_average, "avg"), (hm.watts_peak, "peak"))
            if value is not None
        ]
        if watts:
            lines.append(f"  Power:   {', '.join(watts)} ({_POST_LOOP_POWER_NOTE})")
        return lines + _non_power_summary_lines(hm)

    # Power line: only if at least one power field is set
    power_parts: list[str] = []
    if hm.watts_average is not None:
        power_parts.append(f"{hm.watts_average}W avg")
    if hm.watts_peak is not None:
        power_parts.append(f"{hm.watts_peak}W peak")
    if hm.joules_per_inference is not None:
        power_parts.append(f"{hm.joules_per_inference}J/inference")
    if power_parts:
        lines.append(f"  Power:   {' · '.join(power_parts)}")
    return lines + _non_power_summary_lines(hm)


def _non_power_summary_lines(hm: HardwareMetrics) -> list[str]:
    """The thermal, memory and config lines of the hardware summary."""
    lines: list[str] = []

    # Thermal line: only if at least one temperature field is set
    thermal_parts: list[str] = []
    if hm.temperature_start_celsius is not None:
        thermal_parts.append(f"{hm.temperature_start_celsius}°C →")
    if hm.temperature_peak_celsius is not None:
        thermal_parts.append(f"{hm.temperature_peak_celsius}°C peak")
    if thermal_parts:
        lines.append(f"  Thermal: {' '.join(thermal_parts)}")

    # Memory line: only if at least one memory field is set
    memory_parts: list[str] = []
    if hm.gpu_memory_used_mb is not None:
        memory_parts.append(f"{hm.gpu_memory_used_mb}MB GPU")
    if hm.system_memory_used_mb is not None:
        memory_parts.append(f"{hm.system_memory_used_mb}MB RAM")
    if memory_parts:
        lines.append(f"  Memory:  {' · '.join(memory_parts)}")

    # Config line: only if at least one config field is set
    config_parts: list[str] = []
    if hm.power_mode is not None:
        config_parts.append(hm.power_mode)
    if hm.precision is not None:
        config_parts.append(hm.precision)
    if hm.power_source is not None:
        config_parts.append(f"measured via {hm.power_source}")
    if config_parts:
        lines.append(f"  Config:  {' · '.join(config_parts)}")

    return lines


def _clock_summary_line(root: Path) -> str:
    """The summary's clocks line, from the same collector the run bundle and doctor use."""
    state = collect_clock_state(root)
    pinned = clocks_pinned(state)
    source = (state or {}).get("source") or "unknown"
    if pinned is True:
        return f"  Clocks      pinned ({source})"
    if pinned is False:
        return (
            f"  Clocks      not pinned ({source}): timings may move; calbredge doctor shows the fix"
        )
    if source in UNDETERMINED_REASON:
        return (
            f"  Clocks      not determined ({source}): the CPU's speed can still vary; "
            "compare with care"
        )
    reason = (state or {}).get("reason") or "no clock state collector for this host"
    return f"  Clocks      unknown ({reason})"


# ─── CLI group ────────────────────────────────────────────────────────────────

# Commands other packages contribute, registered in their pyproject.toml. The
# release gate (`gate`, `baseline`) arrives this way, so this open module never
# imports gate code (the open/commercial split; tests/open/test_open_gate_boundary.py).
COMMANDS_ENTRY_POINT_GROUP = "calbredge.commands"
# The commands the release gate package registers, named here only to say so when
# the package is absent.
_GATE_COMMANDS = ("gate", "baseline")


class _PluginGroup(click.Group):
    """The `calbredge` group plus entry-point commands, loaded on first lookup.

    Loading lazily (in get_command, not at import) keeps `import calbredge.cli`
    free of gate modules, and avoids an import cycle: a plugin imports this
    module for the shared benchmark core. A built-in command wins over a
    plugin of the same name.
    """

    def _plugins(self) -> dict[str, importlib.metadata.EntryPoint]:
        return {
            ep.name: ep for ep in importlib.metadata.entry_points(group=COMMANDS_ENTRY_POINT_GROUP)
        }

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted({*super().list_commands(ctx), *self._plugins()})

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command
        entry_point = self._plugins().get(cmd_name)
        if entry_point is not None:
            return entry_point.load()
        return self._unknown_command(ctx, cmd_name)

    def _unknown_command(self, ctx: click.Context, cmd_name: str) -> None:
        """No built-in command and no plugin has this name, so nothing is measured from here.

        Click reports an unknown command itself (exit 2); one of the release gate's
        commands gets a message that says which package provides it, with the same exit.
        """
        if cmd_name in _GATE_COMMANDS:
            raise click.UsageError(
                f"`calbredge {cmd_name}` needs the release gate package, calbredge-gate, "
                "which is not installed here. It is separately licensed; "
                "`calbredge run` measures without it.",
                ctx,
            )


@click.group(cls=_PluginGroup)
@click.version_option(version=__version__, prog_name="calbredge")
def main() -> None:
    """Calbredge: measure AI model inference on edge hardware.

    Measure a model's inference latency under a fixed protocol, and convert
    models for the target.
    """


# ─── run subcommand ───────────────────────────────────────────────────────────


def _build_run_v2(state: dict, result: BenchmarkResult, model: Path) -> tuple[dict, list[str]]:
    """The finished run's v2 result (raw dict) and its validation errors."""
    return build_result_v2(
        v1=result,
        stats=state["stats"],
        protocol=state["protocol"],
        model_path=model,
        metadata=state["metadata"],
        engine_path=state.get("engine_path"),
        telemetry_measurement_boundary=state.get("telemetry_boundary"),
        adapter_clock_state=state.get("adapter_clock_state"),
        cwd=Path.cwd(),
        engine_build=state.get("engine_build"),
        engine_io=state.get("engine_io"),
    )


def _write_run_bundle(
    *,
    state: dict,
    result: BenchmarkResult,
    model: Path,
    bundle_dir: Path | None,
    log_lines: list[str],
    api_key: str | None = None,
    prebuilt: tuple[dict, list[str]] | None = None,
) -> None:
    """Write the v2 run bundle for a finished run (RESULT_SCHEMA.md §6).

    Never raises and never writes to stdout: a bundle problem is a stderr
    warning and must not change `run`'s upload or exit code, or `gate`'s
    verdict. An invalid v2 result is written as result.invalid.json.
    """
    try:
        raw, errors = prebuilt if prebuilt is not None else _build_run_v2(state, result, model)
        path = write_bundle(
            bundle_dir or DEFAULT_BUNDLE_DIR,
            result_id=result.run_id,
            result=raw,
            errors=errors,
            samples_ms=state["stats"].samples_ms,
            environment=build_environment(list(sys.argv), __version__),
            log_lines=log_lines,
            secrets=secret_values([api_key]),
        )
    except Exception as exc:
        click.echo(f"Warning: run bundle not written: {exc}", err=True)
        return
    if errors:
        click.echo(
            f"Warning: v2 result failed validation ({len(errors)} error(s)); "
            f"written as {path / 'result.invalid.json'}",
            err=True,
        )
    click.echo(f"Run bundle: {path}", err=True)


_BUNDLE_DIR_HELP = (
    "Directory to write the v2 run bundle into, as <dir>/<result_id>/ "
    "(default: ~/.calbredge/runs). In CI, e.g. --bundle-dir ./runs for artifact upload."
)


@main.command("run", cls=_SpacedCommand)
@click.option(
    "--model",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help=(
        "Path to the model file to benchmark. "
        "Accepted formats: .onnx (ONNX Runtime), .engine (TensorRT), "
        ".tflite (TFLite / LiteRT), .rknn (RKNN)."
    ),
)
@click.option(
    "--runtime",
    default=None,
    show_default=False,
    help=(
        "Runtime adapter to use (onnxruntime, tensorrt, tflite, rknn, mock). "
        "If omitted, the runtime is auto-detected from the model file extension."
    ),
)
@click.option(
    "--protocol",
    "protocol_path",
    default=None,
    show_default=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Path to a benchmark protocol YAML file. "
        f"Defaults to the canonical v1.0 protocol ({_DEFAULT_WARMUP_ITERATIONS} warmup, "
        f"{_DEFAULT_MEASUREMENT_ITERATIONS} measurement iterations, "
        "batch_size=1). The default protocol file is bundled with the package."
    ),
)
@click.option(
    "--upload",
    default=None,
    show_default=False,
    help=(
        "URL of the API endpoint to upload results to "
        "(e.g. https://api.your-instance.example.com/api/benchmarks). "
        "If omitted, results are saved locally only."
    ),
)
@click.option(
    "--api-key",
    default=None,
    envvar="CALBREDGE_API_KEY",
    show_default=False,
    help=(
        "API key for uploading results. Can also be set via CALBREDGE_API_KEY environment variable."
    ),
)
@click.option(
    "--model-label",
    default=None,
    show_default=False,
    help=(
        "Human-readable label for the model (e.g. yolov8n, mobilenetv2). "
        "Stored in the benchmark result and shown on the dashboard. "
        "If omitted, the field is left null."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help=(
        "Development convenience flag: run the benchmark and print results locally "
        "without uploading. Overrides --upload. "
        "For an explicit security guarantee with no network calls, use --offline instead."
    ),
)
@click.option(
    "--offline",
    is_flag=True,
    default=False,
    help=(
        "Run in fully offline mode. This process will make zero network calls. "
        "Results are saved locally to ~/.calbredge/results/ and printed to stdout. "
        "Use this mode to verify the agent collects no data and sends nothing externally. "
        "Network monitoring tools will confirm no outbound connections. "
        "Takes precedence over --upload if both are provided. "
        "Distinct from --dry-run: --dry-run is a development convenience; "
        "--offline is an explicit security guarantee."
    ),
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help=(
        "Print a debug info block after the run (or on failure) containing "
        "environment details suitable for pasting into a GitHub issue. "
        "Includes Python version, onnxruntime version, CUDA providers, "
        "hardware fingerprint, and JetPack/tegrastats status."
    ),
)
@click.option(
    "--cpu-affinity",
    default="big",
    show_default=True,
    help=(
        "CPU cores to pin the benchmark process to: 'big' (default — pins to "
        "the highest-capacity cluster on detected big.LITTLE ARM, no-ops on "
        "homogeneous CPUs), 'little', 'none' (no pin), or a comma-separated "
        "core-id list (e.g. '4,5,6,7'). Applied before the model loads so "
        "runtime-internal threads inherit it. Pinning to big cores fixes a "
        "confirmed ~9% run-to-run latency variance on RK3588; re-run with "
        "'none' to estimate real (un-pinned) deployment latency."
    ),
)
@click.option(
    "--bundle-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help=_BUNDLE_DIR_HELP,
)
@click.option("--no-bundle", is_flag=True, default=False, help="Do not write a v2 run bundle.")
def run_benchmark(
    model: Path,
    runtime: str | None,
    protocol_path: Path | None,
    upload: str | None,
    api_key: str | None,
    model_label: str | None,
    dry_run: bool,
    offline: bool,
    debug: bool,
    cpu_affinity: str,
    bundle_dir: Path | None,
    no_bundle: bool,
) -> None:
    """Run an inference benchmark on a model file."""
    if offline:
        click.echo("Running in OFFLINE mode — no network calls will be made.")
        if upload:
            click.echo(
                "Warning: --upload was provided but is ignored because --offline takes precedence.",
                err=True,
            )

    # Debug state — collected progressively (via `state`) so the block is
    # useful even on failure.
    _debug_error: str | None = None

    adapter = None
    resolved_runtime = runtime or "(unknown)"
    state: dict = {}
    log_lines: list[str] = []
    finished: BenchmarkResult | None = None

    def _logged_echo(message: str) -> None:
        log_lines.append(message)
        click.echo(message)

    try:
        # 0. Merge CLI flags with config file (CLI always wins)
        _cfg = _load_config()
        if api_key is None:
            api_key = _cfg.get("api_key")
        if upload is None:
            upload = _cfg.get("upload_url")
        if model_label is None:
            model_label = _cfg.get("model_label")
        if runtime is None:
            runtime = _cfg.get("default_runtime")

        # 0b-11. Measurement core, shared with `gate` — see _execute_benchmark.
        result = _execute_benchmark(
            model=model,
            runtime=runtime,
            protocol_path=protocol_path,
            model_label=model_label,
            cpu_affinity=cpu_affinity,
            state=state,
            echo=_logged_echo,
        )
        adapter = state["adapter"]
        resolved_runtime = state["resolved_runtime"]
        stats = state["stats"]
        throughput = state["throughput"]
        run_status: str = state.get("run_status", "stable")
        run_status_reason: str | None = state.get("run_status_reason")

        # 12. Print JSON for dry-run / offline
        if dry_run or offline:
            click.echo(result.model_dump_json(indent=2))
            save_result_locally(result)

        # 13. Upload or save
        else:
            outcome = upload_or_save(result, upload, api_key, debug=debug)
            if outcome.get("uploaded"):
                destination = f"uploaded (run_id={outcome['run_id']})"
            else:
                destination = f"saved locally → {outcome['path']}"

        # 14. Summary table
        providers_line = ""
        if resolved_runtime == "onnxruntime" and hasattr(adapter, "session") and adapter.session:
            providers = adapter.session.get_providers()
            providers_line = ", ".join(providers)

        click.echo("")
        click.echo("─" * 52)
        click.echo("  Calbredge Result Summary")
        click.echo("─" * 52)
        click.echo(f"  Model       {model.name}")
        click.echo(f"  Runtime     {resolved_runtime}")
        if providers_line:
            click.echo(f"  Providers   {providers_line}")
        click.echo(f"  p50         {stats.p50_ms:.2f} ms")
        click.echo(f"  p90         {stats.p90_ms:.2f} ms")
        click.echo(f"  p99         {stats.p99_ms:.2f} ms")
        click.echo(f"  Throughput  {throughput:.1f} img/s")
        if run_status == "stable":
            click.echo("  Status      stable")
        else:
            suffix = f" — {run_status_reason}" if run_status_reason else ""
            label = click.style(
                f"{run_status}{suffix}", fg="yellow" if run_status == "inconclusive" else "red"
            )
            click.echo(f"  Status      {label}")
        click.echo(_clock_summary_line(Path("/")))
        click.echo(f"  hardware_id {result.hardware_id[:16]}…")
        click.echo(f"  run_id      {result.run_id}")
        if dry_run or offline:
            click.echo(f"  Saved       ~/.calbredge/results/{result.run_id}.json")
        else:
            click.echo(f"  Result      {destination}")
        click.echo("─" * 52)

        hw_lines = _hardware_summary_lines(result.hardware_metrics, state.get("telemetry_boundary"))
        if hw_lines:
            click.echo("  Hardware Metrics")
            click.echo("─" * 52)
            for line in hw_lines:
                click.echo(line)
            click.echo("─" * 52)
        finished = result

    except NotImplementedError:
        resolved_runtime = state.get("resolved_runtime", resolved_runtime)
        _debug_error = f"Runtime adapter '{resolved_runtime}' is not yet fully implemented."
        click.echo(
            f"Error: {_debug_error} "
            f"This adapter is stubbed for architecture completeness but does not support "
            f"real inference yet. Please use --runtime onnxruntime for MVP testing.",
            err=True,
        )
        sys.exit(1)
    except FileNotFoundError as exc:
        _debug_error = str(exc)
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except ValueError as exc:
        _debug_error = str(exc)
        click.echo(f"Configuration error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        _debug_error = str(exc)
        click.echo(f"Benchmark failed: {exc}", err=True)
        sys.exit(1)
    finally:
        adapter = state.get("adapter")
        if adapter is not None:
            try:
                adapter.unload()
            except NotImplementedError:
                pass  # stub adapters raise NotImplementedError — ignore
            except Exception:  # noqa: S110 — best-effort teardown, must not mask the original exception
                pass  # unload errors should never mask the original exception

        # After unload, so a cache-miss ONNX parse never overlaps the loaded engine.
        if finished is not None and not no_bundle:
            _write_run_bundle(
                state=state,
                result=finished,
                model=model,
                bundle_dir=bundle_dir,
                log_lines=log_lines,
                api_key=api_key,
            )

        if debug:
            _debug_metadata = state.get("metadata")
            block = format_run_debug_block(
                calbredge_version=__version__,
                model_path=model,
                model_hash=state.get("model_hash"),
                runtime=state.get("resolved_runtime", resolved_runtime),
                protocol_version=state.get("protocol_version"),
                hardware_id=state.get("hardware_id"),
                hostname=_debug_metadata.hostname if _debug_metadata else None,
                os_version=_debug_metadata.os_version if _debug_metadata else None,
                cpu_model=_debug_metadata.cpu_model if _debug_metadata else None,
                cpu_cores=_debug_metadata.cpu_cores if _debug_metadata else None,
                ram_gb=_debug_metadata.ram_gb if _debug_metadata else None,
                gpu_model=_debug_metadata.gpu_model if _debug_metadata else None,
                error_message=_debug_error,
            )
            click.echo(block)


# ─── config subcommand ───────────────────────────────────────────────────────


@main.command("inspect", cls=_SpacedCommand)
@click.option(
    "--model",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the source .onnx model.",
)
@click.option(
    "--runtime",
    default=None,
    help=(
        "Runtime the fields are collected for (e.g. tensorrt-fp16, onnxruntime). "
        "If omitted, auto-detected from the model file extension. Never read from "
        "~/.calbredge/config.yaml."
    ),
)
@click.option(
    "--protocol",
    "protocol_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Benchmark protocol YAML. Defaults to the canonical v1.0 protocol.",
)
@click.option(
    "--cpu-affinity",
    default="big",
    show_default=True,
    help="Predict the cpu_affinity record `run --cpu-affinity` would write (nothing is pinned).",
)
def inspect_cmd(
    model: Path, runtime: str | None, protocol_path: Path | None, cpu_affinity: str
) -> None:
    """Print the v2 candidate/target/protocol fields for this host and model as JSON.

    Read-only: no model is loaded into a runtime, nothing is benchmarked,
    pinned or uploaded, and ~/.calbredge/config.yaml is not read. Fields a
    run only sets while measuring are null and listed under
    inspect.not_collected. Stdout is exactly one JSON document.
    """
    from calbredge.inspect_v2 import inspect_model

    try:
        if protocol_path is None:
            with as_file(
                files("calbredge_protocol.protocols") / _DEFAULT_PROTOCOL_NAME
            ) as default_path:
                protocol = load_protocol(default_path)
        else:
            protocol = load_protocol(protocol_path)
        report = inspect_model(model, protocol, runtime=runtime, cpu_affinity=cpu_affinity)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(report, indent=2))


@main.command("doctor", cls=_SpacedCommand)
@click.option(
    "--model",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Also check this .onnx: graph kind, inputs, and whether an engine is cached for it.",
)
@click.option("--json", "as_json", is_flag=True, default=False, help="Print one JSON document.")
def doctor_cmd(model: Path | None, as_json: bool) -> None:
    """Check this host is ready to measure: clocks, runtimes, support tier, versions.

    Read-only: nothing is pinned, installed, written or uploaded, sudo is never
    run, and no value from ~/.calbredge/config.yaml is printed. Every failing
    check shows its fix. Exit 0 when ready (warnings allowed), 7 when a check
    fails.
    """
    from calbredge.doctor import exit_code, report_json, report_text, run_checks
    from calbredge.term import Style

    checks = run_checks(model=model)
    if as_json:
        click.echo(report_json(checks))
    else:
        # Styled only on a terminal: doctor's stdout may be read by scripts, so not in CI.
        style = Style.for_stream(sys.stdout, github_actions=False)
        style.echo(report_text(checks, style))
    sys.exit(exit_code(checks))


@main.command("init", cls=_SpacedCommand)
@click.option(
    "--model",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The .onnx to measure, inside this directory. Give this or --example.",
)
@click.option(
    "--example",
    is_flag=True,
    default=False,
    help=(
        "Download the example model (MobileNetV2 1.0, Apache 2.0, pinned by SHA-256) into "
        "models/ and set up the project for it."
    ),
)
@click.option(
    "--protocol",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Measure with this protocol file instead of the bundled one matching the model's "
        "input shape. Needed when no bundled protocol fits."
    ),
)
@click.option(
    "--runtime",
    default=None,
    help=(
        "Runtime to measure with (default: tensorrt-fp16 on a Jetson, rknn-fp16 on an RK3588, "
        "else onnxruntime)."
    ),
)
@click.option(
    "--platform",
    default=None,
    type=click.Choice(["jetson", "rk3588"]),
    help=(
        "The target the project is for. Detected on a Jetson or an RK3588. With the release "
        "gate installed it names the self-hosted runner the CI workflow is for, and is "
        "required on any other host."
    ),
)
@click.option(
    "--branch",
    default=None,
    help=(
        "With the release gate installed: the branch whose pushes the CI workflow gates "
        "(default: the repository's default branch)."
    ),
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help=(
        "Write the project again. With the release gate installed, the policy, workflow and "
        "retry script are replaced too."
    ),
)
def init_cmd(
    model: Path | None,
    example: bool,
    protocol: Path | None,
    runtime: str | None,
    platform: str | None,
    branch: str | None,
    force: bool,
) -> None:
    """Make this directory a project for one model, then measure it once.

    Run it in the directory that holds the model, inside a git repository; a repository can
    hold several projects side by side. Writes .calbredge/project.yaml and
    .calbredge/protocol.yaml (the bundled protocol matching the model's input shape, or
    --protocol) here, then runs one measurement into ~/.calbredge/runs.

    Refuses, writing nothing, when the clocks aren't pinned (exit 7, as `calbredge doctor`),
    when no bundled protocol fits the model, when the project already exists, or when this
    directory is inside another project or holds one (--force for these). On an RK3588 the
    model's converted .rknn and its build record must already be beside it
    (`calbredge convert`, on an x86 host), and --example is refused.

    With the release gate (calbredge-gate) installed, init also writes that package's files
    (a starter policy, the project's CI workflow and its retry script), keeps any that
    already exist unless --force, and on a host that is neither a Jetson nor an RK3588 needs
    --platform to say which runner the workflow is for.
    """
    from calbredge import example_model
    from calbredge.init import (
        PROTOCOL_FILE,
        InitError,
        discovery_run,
        prepare,
        record_gitignore,
        summary,
    )
    from calbredge.term import Style

    if (model is None) == (not example):
        raise click.UsageError("give exactly one of --model or --example")

    def fetch(dest: Path) -> None:
        try:
            how = example_model.fetch(example_model.MODEL_URL, dest, example_model.MODEL_SHA256)
        except (example_model.HashMismatchError, example_model.InsecureURLError, OSError) as exc:
            raise InitError(f"example model not fetched, nothing was written: {exc}") from exc
        click.echo(f"{dest}: {how}, SHA-256 {example_model.MODEL_SHA256} verified", err=True)

    outcome, model_path, requested = prepare(
        cwd=Path.cwd(),
        model=model,
        example=example,
        protocol=protocol,
        runtime=runtime,
        branch=branch,
        force=force,
        fetch_example=fetch,
        platform=platform,
    )
    try:
        result = discovery_run(model_path, requested, outcome.root / PROTOCOL_FILE)
    except Exception as exc:
        raise click.ClickException(
            f"the project files are written, but the discovery run failed: {exc}\n"
            f"Fix that, then measure with: calbredge run --model {outcome.model} "
            f"--runtime {requested} --protocol {PROTOCOL_FILE} --offline"
        ) from exc
    finally:
        # After the run, so a committed .gitignore isn't modified while it measures.
        record_gitignore(outcome)
    style = Style.for_stream(sys.stderr)
    style.echo(summary(outcome, result, style), err=True)


@main.command("config", cls=_SpacedCommand)
@click.option(
    "--api-key",
    default=None,
    help="API key for uploading benchmark results.",
)
@click.option(
    "--upload-url",
    default=None,
    help="URL of the API endpoint to upload results to (e.g. https://api.your-instance.example.com/api/benchmarks).",
)
def config_cmd(api_key: str | None, upload_url: str | None) -> None:
    """Create or update ~/.calbredge/config.yaml with persistent settings.

    Run this once after install so you never need to pass --api-key or
    --upload on every benchmark run.
    """
    existing = _load_config()
    if api_key is not None:
        existing["api_key"] = api_key
    if upload_url is not None:
        existing["upload_url"] = upload_url
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w") as f:
        yaml.dump(existing, f, default_flow_style=False, allow_unicode=True)
    click.echo("Config saved to ~/.calbredge/config.yaml")


# ─── convert subcommand ───────────────────────────────────────────────────────


@main.command("convert")
@click.option(
    "--model",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the source model file (.h5 for Keras, .pth / .pt for PyTorch, .onnx for rknn).",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Destination path for the output file (.onnx, or .rknn for --framework rknn).",
)
@click.option(
    "--framework",
    required=True,
    type=click.Choice(["keras", "pytorch", "rknn"], case_sensitive=False),
    help=(
        "Source framework of the model, or 'rknn' to convert an existing "
        ".onnx model to a .rknn artifact for Rockchip NPU inference."
    ),
)
@click.option(
    "--input-shape",
    default=None,
    help=(
        "Comma-separated input tensor shape, e.g. 1,3,224,224. "
        "Required for --framework pytorch. "
        "Ignored for --framework keras (shape is auto-detected from the model) "
        "and --framework rknn (shape is read from the .onnx model)."
    ),
)
@click.option(
    "--precision",
    default="fp16",
    type=click.Choice(["fp16", "int8"], case_sensitive=False),
    help="RKNN precision. Only used with --framework rknn. Defaults to fp16.",
)
@click.option(
    "--target",
    "target_platform",
    default="rk3588",
    show_default=True,
    help="RKNN target platform. Only used with --framework rknn.",
)
@click.option(
    "--calibration-dataset",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Path to a text file listing calibration image paths, one per line. "
        "Required for --framework rknn --precision int8. Also used, for any "
        "precision, as the source of the one representative image for the "
        "conversion-time correctness check (see TROUBLESHOOTING.md §7)."
    ),
)
@click.option(
    "--skip-validate-conversion",
    is_flag=True,
    default=False,
    help=(
        "Skip the general layout-agnostic conversion validator that normally "
        "runs after the detection-presence check for --framework rknn. It adds "
        "a full onnxruntime inference pass plus a second RKNN simulator pass "
        "to every conversion — skip only for fast dev-loop iteration, never "
        "for a build you intend to trust. Only used with --framework rknn."
    ),
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help=(
        "Print a debug info block showing Python version, tf2onnx version, "
        "onnx version, and input file path. On failure, also prints the error details."
    ),
)
def convert_cmd(
    model: Path,
    output: Path,
    framework: str,
    input_shape: str | None,
    precision: str,
    target_platform: str,
    calibration_dataset: Path | None,
    skip_validate_conversion: bool,
    debug: bool,
) -> None:
    """Convert a model to ONNX format for benchmarking, or ONNX to RKNN."""
    _convert_error: str | None = None

    try:
        if framework == "pytorch":
            if input_shape is None:
                raise click.UsageError(
                    "--input-shape is required for --framework pytorch "
                    "(PyTorch weight files do not store the input shape).\n"
                    "Example: --input-shape 1,3,224,224"
                )
            try:
                parsed_shape = tuple(int(x.strip()) for x in input_shape.split(","))
            except ValueError:
                raise click.BadParameter(
                    "must be comma-separated integers, e.g. 1,3,224,224",
                    param_hint="--input-shape",
                )
            if not all(d > 0 for d in parsed_shape):
                raise click.BadParameter(
                    "all dimensions must be positive integers",
                    param_hint="--input-shape",
                )
            convert_pytorch(model, output, parsed_shape)

        elif framework == "rknn":
            convert_rknn(
                model,
                output,
                precision=precision,
                target_platform=target_platform,
                calibration_dataset=calibration_dataset,
                skip_validate_conversion=skip_validate_conversion,
            )

        else:  # keras
            convert_keras(model, output)

    except (click.UsageError, click.BadParameter):
        raise  # let Click handle its own exceptions normally
    except Exception as exc:
        _convert_error = str(exc)
        raise
    finally:
        if debug:
            block = format_convert_debug_block(
                input_path=model,
                error_message=_convert_error,
            )
            click.echo(block)


if __name__ == "__main__":
    main()
