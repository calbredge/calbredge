"""`calbredge doctor`: is this host ready to measure?

Open code (not the commercial gate): host readiness only. It never imports gate code and
never checks a baseline declaration; `calbredge gate --preflight` does that.
Read-only: it writes nothing, pins nothing, never runs sudo, and never prints
a value from ~/.calbredge/config.yaml. Every check reuses the collectors a
run records with (clock_state, host_v2, derive_v2, the support matrix, the
engine cache), so doctor and a run can't disagree about the host.

A check is ``ok``, ``warn`` or ``fail``; any ``fail`` means not ready (exit 7). Readiness is
readiness to measure: whether the host's combination can be gated is reported, never failed,
and clocks fail only on a platform with a pinning rule (Jetson, RK3588).
"""

import getpass
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from calbredge_protocol.derive_v2 import UNDETERMINED_REASON, clocks_pinned
from calbredge_protocol.engine_cache import ENGINE_SUFFIX, RECORD_SUFFIX, compute_build_key
from calbredge_protocol.support_matrix import execution_provider_for, lookup, status_of

import calbredge
from calbredge.clock_state import collect_clock_state, detect_platform
from calbredge.host_v2 import (
    classify_host,
    cuda_version,
    device_tree_model,
    hardware_family,
    l4t_version,
    power_mode,
)
from calbredge.inspect_v2 import predict_runtime_label
from calbredge.term import Style

EXIT_READY = 0
EXIT_NOT_READY = 7

Status = Literal["ok", "warn", "fail"]


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str
    fix: str | None = None


# --- probes: the few facts a fake host tree can't provide (tests replace these) -----


def _module_version(name: str) -> str | None:
    """The module's __version__ if it imports, else None."""
    if importlib.util.find_spec(name) is None:
        return None
    try:
        module = __import__(name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _has_module(name: str) -> bool:
    """Importable, without importing it (pycuda's autoinit would create a CUDA context)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _which(command: str) -> str | None:
    return shutil.which(command)


def _power_mode(root: Path) -> str | None:
    return power_mode(root)


def _platform() -> tuple[str, str]:
    return platform.system(), platform.machine()


# --- checks --------------------------------------------------------------------------

# The Jetson's clocks-at-boot unit (examples/systemd/calbredge-jetson-clocks.service): its
# name, and the path where doctor looks for it once installed, relative to the host root.
JETSON_CLOCKS_UNIT = "calbredge-jetson-clocks.service"
_JETSON_CLOCKS_UNIT_PATH = Path("etc/systemd/system") / JETSON_CLOCKS_UNIT


def jetson_pin_command(root: Path) -> str:
    """The command that pins a Jetson's clocks now: restart the kit's unit if it is installed.

    The unit is a oneshot that stays active after its first run, so `systemctl start` does
    nothing once it has run, and a power-mode change since then has unpinned the clocks.
    """
    if (root / _JETSON_CLOCKS_UNIT_PATH).is_file():
        return f"sudo systemctl restart {JETSON_CLOCKS_UNIT}"
    return "sudo jetson_clocks"


def jetson_pin_fix(root: Path) -> str:
    """How to pin a Jetson's clocks now, for doctor's and init's fix text."""
    if (root / _JETSON_CLOCKS_UNIT_PATH).is_file():
        return (
            f"run: {jetson_pin_command(root)} (re-pins for the current power "
            "mode; needed after every power-mode change)"
        )
    return (
        "run: sudo jetson_clocks (until the next reboot), or sudo ./install.sh "
        "--pin-clocks-at-boot from the Calbredge kit (now and at every boot)"
    )


# The RK3588 boot unit (examples/systemd), relative to the host root.
RK3588_CLOCKS_UNIT = "calbredge-rk3588-clocks.service"
_RK3588_CLOCKS_UNIT_PATH = Path("etc/systemd/system") / RK3588_CLOCKS_UNIT


def rk3588_pin_fix(root: Path) -> str:
    """How to pin an RK3588's CPU, NPU and memory clocks now, for doctor's fix text."""
    if (root / _RK3588_CLOCKS_UNIT_PATH).is_file():
        return (
            f"run: sudo systemctl restart {RK3588_CLOCKS_UNIT} (sets the CPU, NPU and DMC "
            "governors to performance and the CPU scaling maximum to the hardware maximum)"
        )
    return (
        "run: sudo scripts/performance_mode.sh from a Calbredge checkout (the CPU, NPU and DMC "
        f"governors to performance; until the next reboot), or install {RK3588_CLOCKS_UNIT} "
        "from examples/systemd (at every boot)"
    )


# Platforms with a pinning rule: there, clocks that aren't pinned, or can't be read, mean the
# host isn't ready. Elsewhere clocks_pinned is null and a run is still a measurement.
_PINNING_PLATFORMS = ("jetson", "rk3588")

_UNDETERMINED_FIX = (
    "nothing to change: results here record clocks_pinned as null, and timings move with the "
    "CPU's speed, so compare runs with care"
)


def _governor_count(state: dict[str, Any]) -> str:
    governors = state.get("cpu_governors")
    if not isinstance(governors, dict) or not governors:
        return ""
    performance = sum(g == "performance" for g in governors.values())
    return f"; governors: performance on {performance} of {len(governors)} policies"


def _clocks(root: Path) -> Check:
    state = collect_clock_state(root)
    pinned = clocks_pinned(state)
    source = (state or {}).get("source") or detect_platform(root) or "unknown"
    if pinned is True:
        return Check("clocks pinned", "ok", f"yes ({source})")
    if source in UNDETERMINED_REASON:
        detail = f"not determined ({source}): {UNDETERMINED_REASON[source]}"
        return Check(
            "clocks pinned", "warn", detail + _governor_count(state or {}), _UNDETERMINED_FIX
        )
    if pinned is False:
        if source == "jetson":
            fix = jetson_pin_fix(root)
        elif source == "rk3588":
            fix = rk3588_pin_fix(root)
        else:
            fix = "pin the CPU and accelerator clocks"
        return Check("clocks pinned", "fail", f"no ({source}): timings would move", fix)
    reason = (state or {}).get("reason", "no clock state collector for this host")
    if detect_platform(root) not in _PINNING_PLATFORMS:
        return Check("clocks pinned", "warn", f"unknown: {reason}", _UNDETERMINED_FIX)
    return Check(
        "clocks pinned",
        "fail",
        f"unknown: {reason}",
        "timings can't be trusted when the clock state is unknown; run on a host whose clocks "
        "can be read",
    )


def _power(root: Path, jetson: bool) -> Check:
    mode = _power_mode(root)
    if not jetson:
        return Check("power mode", "ok", mode or "not_applicable")
    if mode is None:
        return Check("power mode", "warn", "nvpmodel could not be read", "check: nvpmodel -q")
    return Check("power mode", "ok", mode)


def _family(root: Path, system_metadata: Callable[[], dict[str, Any]]) -> Check:
    if device_tree_model(root) is not None:
        family, source = hardware_family({}, root)
    else:
        family, source = hardware_family(system_metadata(), root)
    return Check("hardware family", "ok", f"{family} ({source})")


def _jetpack(root: Path) -> Check:
    l4t, cuda = l4t_version(root), cuda_version(root)
    detail = f"L4T {l4t or 'unknown'}, CUDA {cuda or 'unknown'}"
    return Check("JetPack", "ok" if l4t and cuda else "warn", detail)


def _tensorrt(jetson: bool) -> Check:
    version = _module_version("tensorrt")
    if version is not None:
        return Check("TensorRT", "ok", version)
    if jetson:
        return Check(
            "TensorRT",
            "fail",
            "not importable",
            "TensorRT comes from JetPack: create the venv with --system-site-packages",
        )
    return Check("TensorRT", "ok", "not installed (not needed on this host)")


def _pycuda(jetson: bool) -> Check:
    if _has_module("pycuda"):
        return Check("pycuda", "ok", "importable")
    if jetson:
        return Check(
            "pycuda",
            "fail",
            "not importable (the TensorRT adapter needs it)",
            "PATH=/usr/local/cuda/bin:$PATH pip install -c constraints/jetson-jp6.2-py310.txt "
            '"calbredge[jetson]"',
        )
    return Check("pycuda", "ok", "not installed (not needed on this host)")


def _numpy(jetson: bool) -> Check:
    version = _module_version("numpy")
    if version is None:
        return Check("numpy", "fail", "not importable", "reinstall calbredge-protocol")
    major = int(version.split(".")[0]) if version.split(".")[0].isdigit() else 0
    if jetson and major >= 2:
        return Check(
            "numpy",
            "fail",
            f"{version}: JetPack's TensorRT and onnxruntime-gpu builds need numpy 1.x",
            'pip install "numpy<2" (or install with -c constraints/jetson-jp6.2-py310.txt)',
        )
    return Check("numpy", "ok", version)


def _onnx_protobuf() -> Check:
    onnx_version = _module_version("onnx")
    protobuf = _dist_version("protobuf")
    detail = f"onnx {onnx_version or 'missing'}, protobuf {protobuf or 'missing'}"
    if onnx_version is None:
        return Check("onnx", "fail", detail, "reinstall calbredge (onnx is a dependency)")
    return Check("onnx", "ok", detail)


def _tegrastats(jetson: bool) -> Check:
    if not jetson:
        return Check("power telemetry", "ok", "not applicable (no tegrastats on this host)")
    if _which("tegrastats") is not None:
        return Check("power telemetry", "ok", "tegrastats present; power_domain module_rail")
    return Check(
        "power telemetry",
        "warn",
        "tegrastats not found: power and energy will be null (latency is unaffected)",
        "install JetPack's tegrastats (nvidia-l4t-tools)",
    )


# --- RK3588: the RKNN runtime ------------------------------------------------------------

RKNN_BINDING = "rknn-toolkit-lite2"
# Where librknnrt.so is installed, relative to the host root, in the order the loader looks.
_LIBRKNNRT_PATHS = (
    Path("usr/lib/librknnrt.so"),
    Path("usr/lib/aarch64-linux-gnu/librknnrt.so"),
    Path("usr/local/lib/librknnrt.so"),
)
_LIBRKNNRT_VERSION = re.compile(rb"librknnrt version: (\d+)\.(\d+)\.(\d+)")
_LIBRKNNRT_SOURCE = (
    "from the airockchip/rknn-toolkit2 repository, "
    "rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so"
)


def librknnrt_version(root: Path = Path("/")) -> tuple[Path, str | None] | None:
    """(path, version) of the installed librknnrt.so, the library that executes RKNN models.

    None when no library is installed; a None version when the file doesn't state one. The
    version is the string the library itself prints at load ("librknnrt version: 2.3.2 ...").
    """
    for rel in _LIBRKNNRT_PATHS:
        path = root / rel
        if not path.is_file():
            continue
        try:
            found = _LIBRKNNRT_VERSION.search(path.read_bytes())
        except OSError:
            return path, None
        return path, ".".join(g.decode() for g in found.groups()) if found else None
    return None


def _rknn_binding() -> Check:
    version = _dist_version(RKNN_BINDING)
    if version is not None and _has_module("rknnlite"):
        return Check(RKNN_BINDING, "ok", version)
    return Check(
        RKNN_BINDING,
        "fail",
        "not installed: RKNN models can't be run",
        'install Calbredge from the kit (sudo ./install.sh), or pip install "calbredge[rk3588]" '
        "with -c constraints/rk3588-jammy-py310.txt",
    )


def _librknnrt(root: Path) -> Check:
    """The runtime library must be installed and match the binding's major.minor version.

    rknn-toolkit-lite2 is only the Python binding; librknnrt.so is not a pip package, and a
    vendor image can ship one several versions older than the binding, which then fails or
    crashes on models converted for the newer one.
    """
    binding = _dist_version(RKNN_BINDING)
    wanted = ".".join(binding.split(".")[:2]) if binding else None
    which = f"the one for {RKNN_BINDING} {binding}" if binding else "the current one"
    install = (
        f"install {which} {_LIBRKNNRT_SOURCE}: sudo cp librknnrt.so /usr/lib/librknnrt.so "
        "&& sudo ldconfig"
    )
    found = librknnrt_version(root)
    if found is None:
        return Check("librknnrt", "fail", "not installed (no librknnrt.so)", install)
    path, version = found
    if version is None:
        return Check("librknnrt", "fail", f"{path}: its version can't be read", install)
    if wanted is not None and ".".join(version.split(".")[:2]) != wanted:
        return Check(
            "librknnrt",
            "fail",
            f"{version} at {path}, but {RKNN_BINDING} is {binding}: models converted for "
            f"{wanted} may not load, or may crash",
            install,
        )
    return Check("librknnrt", "ok", f"{version} ({path})")


def _user() -> str:
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return "<user>"


def _npu_access(root: Path) -> Check:
    """The NPU is reached through the DRM render nodes, which the render group can open."""
    nodes = sorted((root / "dev/dri").glob("renderD*"))
    if not nodes:
        return Check(
            "NPU access",
            "fail",
            "no DRM render node under /dev/dri: the NPU driver (rknpu) isn't loaded",
            "boot the vendor kernel that carries the rknpu driver (dmesg | grep -i rknpu)",
        )
    if any(os.access(node, os.R_OK | os.W_OK) for node in nodes):
        return Check("NPU access", "ok", ", ".join(node.name for node in nodes))
    return Check(
        "NPU access",
        "fail",
        f"{_user()} can't open {', '.join(str(Path('/dev/dri') / n.name) for n in nodes)}",
        f"sudo usermod -aG video,render {_user()}, then log in again",
    )


def _installed_runtime_labels() -> list[tuple[str, str | None]]:
    """(runtime label, precision) for each runtime that can run here."""
    labels: list[tuple[str, str | None]] = []
    if _has_module("tensorrt"):
        labels += [(f"tensorrt-{p}", p) for p in ("fp16", "fp32", "int8")]
    if _has_module("onnxruntime"):
        labels.append((predict_runtime_label("onnxruntime")[0], None))
    if _has_module("rknnlite"):
        labels += [("rknn-fp16", "fp16"), ("rknn-int8", "int8")]
    return labels


def _support(root: Path) -> Check:
    system, machine = _platform()
    state = collect_clock_state(root) or {}
    os_name, device_class, accelerator = classify_host(
        system, device_tree_model(root), state.get("source"), machine
    )
    tiers = []
    for label, precision in _installed_runtime_labels():
        row = lookup(
            label.split("-")[0],
            execution_provider_for(label),
            os_name,
            device_class,
            accelerator,
            precision=precision,
            workload_type="single_forward_pass",
        )
        tiers.append((label, status_of(row)))
    host = f"{os_name}/{device_class}"
    if not tiers:
        return Check("support tier", "fail", f"{host}: no runtime installed", "install a runtime")
    detail = f"{host}: " + ", ".join(f"{label} {status}" for label, status in tiers)
    if any(status == "supported" for _, status in tiers):
        return Check("support tier", "ok", detail)
    if any(status == "results_only" for _, status in tiers):
        # Information, not readiness: a results-only combination measures and records.
        return Check(
            "support tier",
            "ok",
            detail + " (measured and recorded; gating needs a supported combination, "
            "docs/SUPPORTED_PLATFORMS.md)",
        )
    return Check(
        "support tier",
        "warn",
        detail + " (no supported or results-only combination installed)",
        "measure with a supported runtime and target (docs/SUPPORTED_PLATFORMS.md)",
    )


def _writable(path: Path) -> tuple[bool, str]:
    if path.exists():
        return os.access(path, os.W_OK), "exists"
    parent = path.parent
    while not parent.exists():
        parent = parent.parent
    return os.access(parent, os.W_OK), "will be created"


def _state_dirs(home: Path) -> Check:
    # The decision directory is the gate's; its path is repeated here, not
    # imported, so this open module never imports gate code.
    base = home / ".calbredge"
    notes, bad = [], []
    for name in ("runs", "decisions", "engine-cache"):
        ok, how = _writable(base / name)
        notes.append(f"{name} {how}")
        if not ok:
            bad.append(str(base / name))
    if bad:
        return Check(
            "~/.calbredge writable",
            "fail",
            "not writable: " + ", ".join(bad),
            "make them writable by this user (a sudo run may have created them as root)",
        )
    return Check("~/.calbredge writable", "ok", ", ".join(notes))


def _config(home: Path) -> Check:
    """Present and upload URL set, as yes/no only: values are never read out."""
    path = home / ".calbredge" / "config.yaml"
    if not path.is_file():
        return Check("config file", "ok", "absent (runs save locally)")
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        url_set = bool(isinstance(data, dict) and data.get("upload_url"))
    except Exception:
        return Check("config file", "warn", "present but not readable as YAML")
    return Check("config file", "ok", f"present; upload URL set: {'yes' if url_set else 'no'}")


def _versions() -> Check:
    agent, proto = _dist_version("calbredge"), _dist_version("calbredge-protocol")
    detail = f"calbredge {agent}, calbredge-protocol {proto}, __version__ {calbredge.__version__}"
    if agent is not None and agent == proto == calbredge.__version__:
        return Check("package versions", "ok", detail)
    return Check(
        "package versions",
        "fail",
        detail + " (must all match)",
        "reinstall both packages together: pip install -e protocol/ -e agent/ "
        "(or the same release's two wheels)",
    )


# --- --model checks ------------------------------------------------------------------


def _model_contract(model: Path) -> list[Check]:
    import onnx

    from calbredge.onnx_contract import derive_graph_contract

    try:
        contract = derive_graph_contract(onnx.load(str(model)))
    except Exception as exc:
        return [Check("model", "fail", f"not a readable ONNX model: {exc}", "pass the .onnx")]
    inputs = ", ".join(
        f"{t['name']} {t['dtype']} {t['shape']}" for t in contract.io_signature["inputs"]
    )
    checks = [Check("model io_signature", "ok", f"inputs: {inputs}")]
    if contract.graph_kind == "fixed_compute":
        checks.append(Check("model graph_kind", "ok", "fixed_compute"))
    else:
        checks.append(
            Check(
                "model graph_kind",
                "warn",
                f"data_dependent ({', '.join(contract.graph_kind_triggers)}): the work per "
                "inference depends on the input (docs/RESULT_SCHEMA.md §3.5)",
            )
        )
    return checks


def _sha256_file(path: Path) -> str:
    from calbredge.hasher import compute_model_hash

    return compute_model_hash(path)


def cached_engines(model: Path, home: Path) -> list[dict[str, Any]]:
    """Every TensorRT engine cached for this model file, with what its record says.

    Looks where the adapter looks: next to the model, then ~/.calbredge/engine-cache.
    """
    model_hash = _sha256_file(model)
    found = []
    for directory in (model.parent, home / ".calbredge" / "engine-cache"):
        for engine in sorted(directory.glob(f"{model.name}.trt.*{ENGINE_SUFFIX}")):
            record_path = engine.with_name(engine.name[: -len(ENGINE_SUFFIX)] + RECORD_SUFFIX)
            entry: dict[str, Any] = {"engine": str(engine), "record": record_path.is_file()}
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
                inputs = record["inputs"]
                entry.update(
                    intact=compute_build_key(inputs) == record["build_key"],
                    this_model=inputs.get("model_hash") == model_hash,
                    precision=inputs.get("precision"),
                    power_mode=inputs.get("power_mode"),
                    tensorrt=inputs.get("compiler_version"),
                )
            except (OSError, ValueError, KeyError, TypeError):
                entry.update(intact=False, this_model=False)
            found.append(entry)
    return found


def _engine_power_mode(root: Path) -> str:
    """The power mode an engine is keyed on, by the adapter's rule (nvpmodel, else default)."""
    mode = _power_mode(root)
    return "default" if mode in (None, "not_applicable") else mode


def _usable(engine: dict[str, Any], mode: str, trt: str | None) -> bool:
    return bool(
        engine.get("intact")
        and engine.get("this_model")
        and engine.get("power_mode") == mode
        and (trt is None or engine.get("tensorrt") == trt)
    )


def _engines(model: Path, home: Path, root: Path) -> Check:
    engines = cached_engines(model, home)
    mode = _engine_power_mode(root)
    trt = _module_version("tensorrt")
    usable = [e for e in engines if _usable(e, mode, trt)]
    if usable:
        detail = "; ".join(
            f"{Path(e['engine']).name} ({e['precision']}, {e['power_mode']})" for e in usable
        )
        return Check("cached engine", "ok", detail)
    stale = len(engines) - len(usable)
    note = (
        f" ({stale} other engine(s) for this path were built from a different model, "
        "power mode or TensorRT)"
        if stale
        else ""
    )
    return Check(
        "cached engine",
        "warn",
        f"none for this model at power mode {mode}{note}: the first run builds one (30-90 s)",
    )


# --- entry point ---------------------------------------------------------------------


def run_checks(
    *,
    root: Path = Path("/"),
    home: Path | None = None,
    model: Path | None = None,
    system_metadata: Callable[[], dict[str, Any]] | None = None,
) -> list[Check]:
    home = home or Path.home()
    platform_name = detect_platform(root)
    jetson = platform_name == "jetson"

    def default_metadata() -> dict[str, Any]:
        from calbredge.collector import collect_metadata

        return collect_metadata("mock").model_dump()

    checks = [
        _clocks(root),
        _power(root, jetson),
        _family(root, system_metadata or default_metadata),
    ]
    if jetson:
        checks.append(_jetpack(root))
    checks += [
        _tensorrt(jetson),
        _pycuda(jetson),
        _numpy(jetson),
        _onnx_protobuf(),
        _tegrastats(jetson),
    ]
    if platform_name == "rk3588":
        checks += [_rknn_binding(), _librknnrt(root), _npu_access(root)]
    checks += [
        _support(root),
        _state_dirs(home),
        _config(home),
        _versions(),
    ]
    if model is not None:
        checks += _model_contract(model)
        checks.append(_engines(model, home, root))
    return checks


def exit_code(checks: list[Check]) -> int:
    return EXIT_NOT_READY if any(c.status == "fail" for c in checks) else EXIT_READY


def report_json(checks: list[Check]) -> str:
    code = exit_code(checks)
    return json.dumps(
        {"ready": code == EXIT_READY, "exit_code": code, "checks": [asdict(c) for c in checks]},
        indent=2,
    )


def report_text(checks: list[Check], style: Style | None = None) -> str:
    """The readiness table. Plain by default; a styled one adds symbols and colour only."""
    style = style or Style()
    width = max(len(c.name) for c in checks)
    status_width = 4 + style.mark_width
    lines = []
    for c in checks:
        lines.append(f"{style.label(c.status.upper(), 4)}  {c.name:<{width}}  {c.detail}")
        if c.fix and c.status != "ok":
            lines.append(f"{'':<{status_width}}  {'':<{width}}  fix: {c.fix}")
    failed = sum(c.status == "fail" for c in checks)
    lines.append("")
    lines.append(
        style.paint("Ready.", "OK")
        if not failed
        else style.paint(f"Not ready: {failed} check(s) failed (exit {EXIT_NOT_READY}).", "FAIL")
    )
    return "\n".join(lines)
