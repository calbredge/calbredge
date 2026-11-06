"""`calbredge init`: make the current directory a project that measures one model.

Open code: it never imports the gate. A project is a directory inside a git repository
holding .calbredge/ (one repository can hold several, side by side, one model each). In the
current directory init writes .calbredge/project.yaml and the protocol matching the model's
input shape (.calbredge/protocol.yaml). Then it measures the model once (the discovery run),
and only after that adds the derived-file patterns to the project's .gitignore: a committed
.gitignore extended before the run would be a modified tracked file, so the run would be
recorded git_dirty and could not become a baseline.

Other packages add to a project through the ``calbredge.init_hooks`` entry-point group
(``InitHook``). A hook checks before anything is written, writes its own files after the
protocol and before project.yaml, may set project.yaml's optional fields, and ends the
summary. The release gate's hook writes the policy, the CI workflow and the retry script.
With no hook installed init writes the open files only.

Refuses, before writing anything, when doctor's clock check fails (a Jetson or an RK3588
whose clocks aren't pinned or can't be read; elsewhere the clocks are only a warning), when
the model's input shape matches no bundled protocol (pass --protocol), when the project
already exists, when it would sit inside another project or around one (pass --force), or
when a hook refuses. A file a hook writes is kept if it is already there, unless --force.
"""

import contextlib
import importlib.metadata
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from importlib.resources import files
from pathlib import Path
from typing import Any, Protocol

import click
from calbredge_protocol.bundled import bundled_protocols, matching_bundled_protocol
from calbredge_protocol.loader import load_protocol

from calbredge.hasher import compute_model_hash
from calbredge.onnx_contract import load_graph_contract
from calbredge.project import PROJECT_DIR, PROJECT_FILE, Project, Target, write_project
from calbredge.term import Style

EXIT_NOT_READY = 7  # the same code as `calbredge doctor`: this host can't measure yet

# Other packages' parts of init, registered in their pyproject.toml (see InitHook).
INIT_HOOKS_ENTRY_POINT_GROUP = "calbredge.init_hooks"

PROTOCOL_FILE = f"{PROJECT_DIR}/protocol.yaml"
EXAMPLE_MODEL = "models/mobilenetv2-12.onnx"
# The first line init writes into the copied protocol; cli's run header reads it back.
ORIGIN_PREFIX = "# Copied by `calbredge init` from "
GITIGNORE = ".gitignore"
# Derived files a project's runs leave next to the model or in its directory: TensorRT
# engines and their build records, the ONNX contract cache, and CI output. Never committed.
IGNORED = ("*.engine", "*.engine.json", "*.calbredge-contract.json", "calbredge-out/")
_IGNORE_HEADER = "# Calbredge: derived files (engines, contract caches, CI output)"

# The targets a project can be set up for; a hook is given the one this project is for.
PLATFORMS = ("jetson", "rk3588")

_PIN_FIX = (
    "pin the clocks first: `sudo ./install.sh --pin-clocks-at-boot` from the Calbredge kit "
    "(pins them at every boot), or `sudo jetson_clocks` (until the next reboot)"
)


class InitError(click.ClickException):
    """init refused; nothing was written."""


class NotReadyError(InitError):
    exit_code = EXIT_NOT_READY


@dataclass(frozen=True)
class InitRequest:
    """What a hook is asked to set up: where the project is and what the command was given."""

    project_dir: Path
    repo_root: Path
    project_rel: str  # the project directory, relative to the repository root ("" at the root)
    platform: str  # the target the project is for, a member of PLATFORMS
    branch: str | None  # --branch as given
    force: bool


class InitHook(Protocol):
    """Another package's part of init, registered in ``calbredge.init_hooks``."""

    #: What --force replaces besides the project file, for the refusal that names it.
    replaced_by_force: str

    def check(self, request: InitRequest) -> None:
        """Refuse (raise InitError) what this hook can't set up. Writes nothing."""
        ...

    def write(self, request: InitRequest, outcome: "Outcome") -> dict[str, str]:
        """Write this hook's files with ``_put``; return fields to set in project.yaml."""
        ...

    def closing(self, outcome: "Outcome", result: Any) -> list[str]:
        """The summary's last lines, after the discovery run's."""
        ...


def load_hooks() -> list[InitHook]:
    """Every installed hook, in the order of their entry-point names (tests replace it)."""
    found = importlib.metadata.entry_points(group=INIT_HOOKS_ENTRY_POINT_GROUP)
    return [entry_point.load() for entry_point in sorted(found, key=lambda ep: ep.name)]


@dataclass
class Outcome:
    """What init did, for its summary. Written and kept files are repository-relative."""

    root: Path  # the project directory
    repo_root: Path
    project_rel: str  # the project directory, relative to the repository root ("" at the root)
    gitignore: str = "unchanged"  # created, updated or unchanged
    model: str = ""
    requested_runtime: str = ""
    written: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)  # a written file's bracketed note
    protocol_name: str = ""
    input_shape: list[int] = field(default_factory=list)
    runtime: str = ""
    converted_model: str = ""  # RKNN: the .rknn beside the model, to commit with its record
    hooks: list[InitHook] = field(default_factory=list)


# --- the repository -----------------------------------------------------------------------


def _git(cwd: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(  # noqa: S603 — fixed git argv; no shell, no user-controlled command
            ["git", "-C", str(cwd), *args],  # noqa: S607 — git resolved from PATH by design
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def repo_root(cwd: Path) -> Path:
    """The git top level: the root that repository-wide files and shown paths are relative to."""
    top = _git(cwd, "rev-parse", "--show-toplevel")
    if not top:
        raise InitError(
            f"{cwd} is not inside a git repository: the project's files are meant "
            "to be committed. Run `git init` (or clone your repository) first."
        )
    return Path(top).resolve()


def enclosing_project(project_dir: Path, root: Path) -> Path | None:
    """A project file in a parent directory of project_dir, up to the repository root."""
    for directory in project_dir.parents:
        if directory != root and root not in directory.parents:
            break
        candidate = directory / PROJECT_DIR / PROJECT_FILE
        if candidate.is_file():
            return candidate
    return None


def enclosed_projects(project_dir: Path) -> list[str]:
    """Project files below project_dir (tracked or not, unless git-ignored), relative to it."""
    listed = _git(
        project_dir,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        f":(glob)**/{PROJECT_DIR}/{PROJECT_FILE}",
    )
    own = f"{PROJECT_DIR}/{PROJECT_FILE}"
    return sorted(p for p in (listed or "").splitlines() if p and p != own)


def default_branch(root: Path) -> str:
    """The branch pushes gate on: the remote's default, else the current branch, else main."""
    remote = _git(root, "symbolic-ref", "--short", "refs/remotes/origin/HEAD")
    if remote and "/" in remote:
        return remote.split("/", 1)[1]
    return _git(root, "symbolic-ref", "--short", "HEAD") or "main"


# --- the model and its protocol -----------------------------------------------------------


def _bundled_protocols() -> list[tuple[str, list[int]]]:
    """(file name, input_shape) of every bundled protocol, the default first."""
    return bundled_protocols()


def model_input_dims(model: Path) -> list[int | str]:
    """The model's single input's dims; named dims are dynamic."""
    contract, _ = load_graph_contract(model, compute_model_hash(model))
    inputs = contract.io_signature.get("inputs", [])
    if len(inputs) != 1:
        raise InitError(
            f"{model.name} has {len(inputs)} inputs; a bundled protocol describes one. "
            "Write a protocol for it and pass --protocol."
        )
    return list(contract.primary_input_dims)


def choose_protocol(dims: list[int | str]) -> tuple[str, list[int]]:
    """The bundled protocol for these input dims: (file name, input_shape).

    A dynamic batch axis (the first) is fine: every bundled protocol measures batch 1, and
    runs bind it the same way. Any other dynamic axis, or a shape no bundled protocol has,
    is refused with "pass --protocol".
    """
    if any(not isinstance(d, int) for d in dims[1:]):
        raise InitError(
            f"the model's input {dims} has dynamic dimensions besides the batch, so no "
            "bundled protocol fits it; pass --protocol with the input shape to measure"
        )
    match = matching_bundled_protocol(dims)
    if match is not None:
        return match
    supported = ", ".join(str(s) for _, s in _bundled_protocols())
    raise InitError(
        f"no bundled protocol has input shape {dims} (bundled: {supported}); pass --protocol"
    )


def _check_given_protocol(protocol: Path, dims: list[int | str]) -> list[int]:
    try:
        shape = list(load_protocol(protocol).input_shape)
    except Exception as exc:
        raise InitError(f"could not load protocol {protocol}: {exc}") from exc
    if len(shape) != len(dims) or any(
        isinstance(d, int) and d != s for d, s in zip(dims, shape, strict=False)
    ):
        raise InitError(
            f"protocol {protocol} measures input {shape}, but the model's input is {dims}"
        )
    return shape


# --- files --------------------------------------------------------------------------------


def update_gitignore(project_dir: Path) -> str:
    """Add the derived-file patterns to the project's .gitignore; say what happened.

    Returns "created", "updated" or "unchanged".

    Idempotent: a pattern already there (as its own line) isn't added again, and nothing the
    user wrote is changed or removed.
    """
    path = project_dir / GITIGNORE
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    present = {line.strip() for line in existing.splitlines()}
    missing = [p for p in IGNORED if p not in present]
    if not missing:
        return "unchanged"
    block = "\n".join(([] if _IGNORE_HEADER in present else [_IGNORE_HEADER]) + missing) + "\n"
    separator = "" if not existing or existing.endswith("\n") else "\n"
    blank = "\n" if existing.strip() else ""
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text(existing + separator + blank + block, encoding="utf-8")
    os.replace(tmp, path)
    return "updated" if existing else "created"


def record_gitignore(outcome: Outcome) -> None:
    """Update the project's .gitignore and record it for the summary; call it after the
    discovery run, even a failed or interrupted one.

    After, not before: extending a committed .gitignore makes it a modified tracked file, and a
    run measured while it is modified is recorded git_dirty, which `baseline approve` refuses.
    The run's engine and contract cache are untracked, so they never make it dirty either way.
    A failure to write is a warning naming the file and the lines to add, never an error that
    would hide the run's own outcome.
    """
    path = outcome.root / GITIGNORE
    try:
        outcome.gitignore = update_gitignore(outcome.root)
    except OSError as exc:
        style = Style.for_stream(sys.stderr)
        click.echo(
            f"{style.label('WARN')} could not update {path}: {exc}. Add these lines to it by "
            f"hand so derived files are never committed: {' '.join(IGNORED)}",
            err=True,
        )
        return
    if outcome.gitignore != "unchanged":
        prefix = f"{outcome.project_rel}/" if outcome.project_rel else ""
        outcome.written.append(f"{prefix}{GITIGNORE}")


def _relative(path: Path, project_dir: Path) -> str:
    try:
        return path.resolve().relative_to(project_dir).as_posix()
    except ValueError:
        raise InitError(
            f"the model {path} is outside the project directory {project_dir}; put it in the "
            "project (commit it, or use Git LFS for a large model), or run init where it is"
        ) from None


def _put(
    base: Path,
    rel: str,
    text: str,
    *,
    force: bool,
    outcome: Outcome,
    shown: str,
    mode: int = 0o644,
    note: str = "",
) -> None:
    """Write base/rel, keeping an existing file unless force; record it as shown.

    ``note`` is what the summary puts in brackets after a written file.
    """
    target = base / rel
    if target.exists() and not force:
        outcome.kept.append(shown)
        return
    if note:
        outcome.notes[shown] = note
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.partial")
    tmp.write_text(text, encoding="utf-8")
    tmp.chmod(mode)
    os.replace(tmp, target)
    outcome.written.append(shown)


# --- the command --------------------------------------------------------------------------


# The host's filesystem root, for the clock fix's unit lookup (tests point it elsewhere).
_HOST_ROOT = Path("/")


def pin_fix() -> str:
    """How to pin this Jetson's clocks now: doctor's fix, which names the boot unit if any."""
    from calbredge.doctor import jetson_pin_fix

    return f"pin the clocks first: {jetson_pin_fix(_HOST_ROOT).removeprefix('run: ')}"


def clock_check():
    """doctor's clock check for this host (tests replace it)."""
    from calbredge.doctor import _clocks

    return _clocks(Path("/"))


_DEFAULT_RUNTIMES = {"jetson": "tensorrt-fp16", "rk3588": "rknn-fp16"}


def default_runtime() -> str:
    """What this host measures with: its accelerator's FP16 runtime, else ONNX Runtime."""
    from calbredge.clock_state import detect_platform

    return _DEFAULT_RUNTIMES.get(detect_platform(Path("/")) or "", "onnxruntime")


def detected_platform() -> str | None:
    """This host's platform if it is one a workflow can target, else None (tests replace it)."""
    from calbredge.clock_state import detect_platform

    found = detect_platform(_HOST_ROOT)
    return found if found in PLATFORMS else None


def resolve_platform(explicit: str | None) -> str:
    """The platform a hook sets the project up for.

    Detected on a Jetson or an RK3588, where --platform is optional and must agree. Any other
    host can only prepare a project for one of them, so it has to say which.
    """
    detected = detected_platform()
    names = " or ".join(PLATFORMS)
    if detected is None:
        if explicit is None:
            raise InitError(
                "this host is neither a Jetson nor an RK3588, so init can't tell which runner "
                f"the project's CI workflow is for; pass --platform {names.replace(' or ', '|')}"
            )
        return explicit
    if explicit is not None and explicit != detected:
        raise InitError(
            f"--platform {explicit} doesn't match this host, which is a {detected}; leave "
            "--platform out here, or run init on the target it names"
        )
    return detected


def rknn_artifact_path(model: Path, runtime_label: str) -> Path:
    """The converted model the RKNN adapter loads for this .onnx (its sibling cache file)."""
    precision = runtime_label.split("-", 1)[1] if "-" in runtime_label else "fp16"
    return Path(f"{model}.rknn.{precision}.default")


def _check_rknn_artifact(model: Path, model_rel: str, runtime_label: str) -> str:
    """The .rknn and its build record must be beside the model: the board can't convert."""
    from calbredge_protocol import rknn_artifact

    artifact = rknn_artifact_path(model, runtime_label)
    record = rknn_artifact.record_path(artifact)
    precision = artifact.name.split(".")[-2]
    convert = (
        f"On an x86 host, run: calbredge convert --framework rknn --model {model.name} "
        f"--precision {precision} --output {artifact.name}\n"
        f"then copy {artifact.name} and {record.name} beside {model_rel}."
    )
    if not artifact.is_file():
        raise InitError(
            f"{model_rel} has no converted model beside it ({artifact.name}), and this board "
            f"can't convert one. Nothing was written.\n{convert}"
        )
    if not record.is_file():
        raise InitError(
            f"{artifact.name} has no build record ({record.name}), so nothing shows which "
            "model it was built from. "
            f"Nothing was written.\n{convert}"
        )
    return artifact.name


def prepare(
    *,
    cwd: Path,
    model: Path | None,
    example: bool,
    protocol: Path | None,
    runtime: str | None,
    branch: str | None,
    force: bool,
    fetch_example: Callable[[Path], None],
    platform: str | None = None,
) -> tuple[Outcome, Path, str]:
    """Every check, then every file. Returns (outcome, model path, requested runtime)."""
    from calbredge.inspect_v2 import predict_runtime_label

    hooks = load_hooks()
    # Only a hook needs to be told the target; with none, the host's own is enough.
    platform = resolve_platform(platform) if hooks else (platform or detected_platform())
    if example and platform == "rk3588":
        raise InitError(
            "--example downloads only the .onnx, and an RK3588 can't convert it to a .rknn. "
            "Convert your model on an x86 host with `calbredge convert --framework rknn`, copy "
            "the .onnx, the .rknn and its .json here, and run init with --model."
        )
    project_dir = cwd.resolve()
    root = repo_root(project_dir)
    project_rel = "" if project_dir == root else project_dir.relative_to(root).as_posix()

    project_file = project_dir / PROJECT_DIR / PROJECT_FILE
    if project_file.exists() and not force:
        replaced = ", ".join(hook.replaced_by_force for hook in hooks)
        also = f" ({replaced} are replaced too)" if replaced else ""
        raise InitError(f"{project_file} already exists; pass --force to write it again{also}")
    outer = enclosing_project(project_dir, root)
    if outer is not None and not force:
        raise InitError(
            f"this directory is inside the project in {outer.parent.parent}; a project can't "
            "hold another one. Run init in a sibling directory, or pass --force."
        )
    inner = enclosed_projects(project_dir)
    if inner and not force:
        raise InitError(
            f"this directory already holds project(s) below it ({', '.join(inner)}); a project "
            "can't hold another one. Run init in a sibling directory, or pass --force."
        )
    request = InitRequest(
        project_dir=project_dir,
        repo_root=root,
        project_rel=project_rel,
        platform=platform or "",
        branch=branch,
        force=force,
    )
    for hook in hooks:
        hook.check(request)

    clocks = clock_check()
    # Only a fail refuses: on a host with no pinning rule (x86) the clocks are a warning, and the
    # run records clocks_pinned as null.
    if clocks.status == "fail":
        fix = pin_fix() if "jetson" in clocks.detail else (clocks.fix or _PIN_FIX)
        raise NotReadyError(f"clocks pinned: {clocks.detail}. Nothing was written; {fix}")

    if example:
        model = project_dir / EXAMPLE_MODEL
        fetch_example(model)
    if model is None:
        raise click.UsageError("give --model <onnx>, or --example")
    model_rel = _relative(model, project_dir)
    dims = model_input_dims(model)

    outcome = Outcome(
        root=project_dir,
        repo_root=root,
        project_rel=project_rel,
        model=model_rel,
        hooks=hooks,
    )
    if protocol is not None:
        outcome.input_shape = _check_given_protocol(protocol, dims)
        outcome.protocol_name = protocol.name
        protocol_text = protocol.read_text(encoding="utf-8")
    else:
        outcome.protocol_name, outcome.input_shape = choose_protocol(dims)
        protocol_text = (files("calbredge_protocol.protocols") / outcome.protocol_name).read_text(
            encoding="utf-8"
        )
    # A first line naming where the copy came from; the run header shows it. A comment, so
    # the protocol's digest (over its parsed values) is unchanged.
    protocol_text = f"{ORIGIN_PREFIX}{outcome.protocol_name}.\n{protocol_text}"
    existing_protocol = project_dir / PROTOCOL_FILE
    if (
        existing_protocol.exists()
        and not force
        and existing_protocol.read_text(encoding="utf-8") != protocol_text
    ):
        raise InitError(
            f"{PROTOCOL_FILE} exists and differs from the protocol for this model; "
            "pass --force to replace it"
        )

    requested = runtime or default_runtime()
    outcome.requested_runtime = requested
    try:
        outcome.runtime = predict_runtime_label(requested)[0]
    except ValueError as exc:
        raise InitError(str(exc)) from exc
    if outcome.runtime.split("-", 1)[0] == "rknn":
        outcome.converted_model = _check_rknn_artifact(model, model_rel, outcome.runtime)

    # Everything checked: write. Shown paths are relative to the repository root.
    def shown(rel: str) -> str:
        return f"{project_rel}/{rel}" if project_rel else rel

    _put(
        project_dir,
        PROTOCOL_FILE,
        protocol_text,
        force=True,
        outcome=outcome,
        shown=shown(PROTOCOL_FILE),
        note=f"{outcome.protocol_name}, input {outcome.input_shape}",
    )
    project = Project(
        root=project_dir,
        model=model_rel,
        protocol=PROTOCOL_FILE,
        policy=None,
        targets=(Target(runtime=outcome.runtime),),
    )
    # A hook's files go between the protocol and the project file, which names them.
    for hook in hooks:
        project = replace(project, **hook.write(request, outcome))
    write_project(project, overwrite=force)
    outcome.written.insert(0, shown(f"{PROJECT_DIR}/{PROJECT_FILE}"))
    outcome.notes[shown(f"{PROJECT_DIR}/{PROJECT_FILE}")] = f"runtime {outcome.runtime}"
    return outcome, project_dir / model_rel, requested


def discovery_run(model: Path, runtime: str, protocol: Path):
    """Measure the model once, like `calbredge run --offline`, into ~/.calbredge/runs."""
    from calbredge import cli

    state: dict = {}
    log_lines: list[str] = []

    def echo(message: str) -> None:
        log_lines.append(message)
        click.echo(message, err=True)

    result = None
    try:
        result = cli._execute_benchmark(
            model=model,
            runtime=runtime,
            protocol_path=protocol,
            model_label=None,
            cpu_affinity="big",
            state=state,
            echo=echo,
        )
    finally:
        adapter = state.get("adapter")
        if adapter is not None:
            # Best-effort teardown: must not mask the original exception.
            with contextlib.suppress(Exception):
                adapter.unload()
        if result is not None:
            cli._write_run_bundle(
                state=state, result=result, model=model, bundle_dir=None, log_lines=log_lines
            )
    return result


def rerun_command(outcome: Outcome) -> str:
    """The command that measures the project's model again, from its directory."""
    return (
        f"calbredge run --model {outcome.model} --runtime {outcome.requested_runtime} "
        f"--protocol {PROTOCOL_FILE} --offline"
    )


def summary(outcome: Outcome, result, style: Style) -> str:
    """What was written, how the discovery run went, and what comes next."""
    where = f"the project in {outcome.project_rel}" if outcome.project_rel else "the project"
    lines = ["", f"Wrote {where} (paths from the repository root):"]
    prefix = f"{outcome.project_rel}/" if outcome.project_rel else ""
    notes = {
        **outcome.notes,
        f"{prefix}{GITIGNORE}": f"{outcome.gitignore}: engines, contract caches and CI output "
        "are never committed",
    }
    lines += [f"  {rel}" + (f"  ({notes[rel]})" if rel in notes else "") for rel in outcome.written]
    if outcome.kept:
        lines += ["Kept (already there; --force replaces them):"]
        lines += [f"  {rel}" for rel in outcome.kept]
    lines.append("")
    stable = result.run_status == "stable"
    status = "stable" if stable else f"{result.run_status} ({result.run_status_reason})"
    lines.append(
        f"{style.label('OK' if stable else 'WARN')} discovery run {result.run_id}: {status}, "
        f"p50 {result.latency_p50_ms:.3f} ms, runtime {result.runtime}"
    )
    if result.runtime != outcome.runtime:
        lines.append(
            f"{style.label('WARN')} the run recorded runtime {result.runtime}, but "
            f"project.yaml says {outcome.runtime}: measure the project with the project's runtime"
        )
    if outcome.converted_model:
        folder = str(Path(outcome.model).parent)
        beside = f"in {folder}/" if folder != "." else "beside the model"
        lines.append(
            f"Commit {outcome.converted_model} and {outcome.converted_model}.json with "
            f"{Path(outcome.model).name} ({beside}): the board can't convert, and the record "
            "names this exact .onnx, so the three travel together."
        )
    if not outcome.hooks:
        lines.append(f"The project is measured. Measure it again with: {rerun_command(outcome)}")
    for hook in outcome.hooks:
        lines += hook.closing(outcome, result)
    return "\n".join(lines)
