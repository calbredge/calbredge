"""The project file, .calbredge/project.yaml: what one project measures and gates.

Open code. A project is a directory holding .calbredge/; a repository can hold several side
by side (never one inside another), one model each. Its committed file names the model, the
protocol and the targets it is measured on, and optionally a policy and quality evidence.
Each target has a runtime and a pointer to its approved baseline declaration (null until one
is approved). Open code never reads the policy, the quality evidence or a baseline: they are
paths the release gate fills in and interprets. `calbredge init` writes the file, `calbredge
baseline approve` updates a target's pointer, and `calbredge gate` reads it; each finds the
nearest project above the current directory.

Schema version "1":

    schema_version: "1"
    model: models/m.onnx                 # every path is relative to the project directory,
    protocol: .calbredge/protocol.yaml   # the one holding .calbredge/
    policy: .calbredge/policy.yaml       # optional: null until the gate's init hook sets it
    quality_evidence: null               # optional
    targets:                             # exactly one entry until multi-target support
      - runtime: tensorrt-fp16           # what is measured; the gate derives the runtime
        baseline: null                   # from the declaration and refuses a mismatch

Strict: an unknown key is refused (a typo would otherwise be silently ignored), and a newer
schema_version is refused with "upgrade calbredge". Later versions may add keys (target
selection by hardware, several candidates) but never change the meaning of these.

The file is machine-written: updating it rewrites it, so comments are not kept.
"""

import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from calbredge_protocol.canonical import is_runtime_label

PROJECT_DIR = ".calbredge"
PROJECT_FILE = "project.yaml"
SCHEMA_VERSION = "1"
MAX_TARGETS = 1  # until multi-target support

_KEYS = ("schema_version", "model", "protocol", "policy", "quality_evidence", "targets")
_REQUIRED = ("schema_version", "model", "protocol", "targets")
_TARGET_KEYS = ("runtime", "baseline")


class ProjectError(ValueError):
    """The project file is missing, unreadable or invalid."""


@dataclass(frozen=True)
class Target:
    runtime: str
    baseline: str | None = None


@dataclass(frozen=True)
class Project:
    """A loaded project file. Paths are as written, relative to root."""

    root: Path  # the project directory, the one holding .calbredge/
    model: str
    protocol: str
    policy: str | None
    targets: tuple[Target, ...]
    quality_evidence: str | None = None

    @property
    def file(self) -> Path:
        return self.root / PROJECT_DIR / PROJECT_FILE

    @property
    def target(self) -> Target:
        """The one target (a project has exactly one until multi-target support)."""
        return self.targets[0]

    def path(self, relative: str) -> Path:
        """A path from the file, resolved against the project directory."""
        return self.root / relative

    def to_dict(self) -> dict[str, Any]:
        """The file's contents, in its fixed key order."""
        return {
            "schema_version": SCHEMA_VERSION,
            "model": self.model,
            "protocol": self.protocol,
            "policy": self.policy,
            "quality_evidence": self.quality_evidence,
            "targets": [{"runtime": t.runtime, "baseline": t.baseline} for t in self.targets],
        }


def find_project_file(start: Path | None = None) -> Path:
    """.calbredge/project.yaml in start (default: the current directory) or its nearest parent."""
    start = (start or Path.cwd()).resolve()
    for directory in (start, *start.parents):
        candidate = directory / PROJECT_DIR / PROJECT_FILE
        if candidate.is_file():
            return candidate
    raise ProjectError(
        f"no {PROJECT_DIR}/{PROJECT_FILE} in {start} or any parent directory; "
        "run `calbredge init` in your repository"
    )


def load_project(start: Path | None = None) -> Project:
    """Find and load the project file above start (default: the current directory)."""
    return read_project(find_project_file(start))


def read_project(file: Path) -> Project:
    """Load and validate one project file; its project directory is the parent of .calbredge/."""
    try:
        data = yaml.safe_load(file.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProjectError(f"could not read {file}: {exc}") from exc
    return _parse(data, root=file.parent.parent, source=file)


def _parse(data: Any, *, root: Path, source: Path) -> Project:
    if not isinstance(data, dict):
        raise ProjectError(f"{source}: must be a YAML mapping")
    _check_version(data.get("schema_version"), source)
    unknown = sorted(str(k) for k in data if k not in _KEYS)
    if unknown:
        raise ProjectError(f"{source}: unknown key(s) {', '.join(unknown)}")
    missing = [k for k in _REQUIRED if k not in data]
    if missing:
        raise ProjectError(f"{source}: missing key(s) {', '.join(missing)}")
    policy = data.get("policy")
    evidence = data.get("quality_evidence")
    return Project(
        root=root,
        model=_path(data["model"], "model", source),
        protocol=_path(data["protocol"], "protocol", source),
        policy=None if policy is None else _path(policy, "policy", source),
        quality_evidence=None if evidence is None else _path(evidence, "quality_evidence", source),
        targets=_targets(data["targets"], source),
    )


def _check_version(version: Any, source: Path) -> None:
    if version is None:
        raise ProjectError(f"{source}: missing key(s) schema_version")
    if not isinstance(version, str) or not version.isdigit():
        raise ProjectError(f'{source}: schema_version must be a quoted whole number, e.g. "1"')
    if int(version) > int(SCHEMA_VERSION):
        raise ProjectError(
            f"{source}: written for project schema {version}, but this calbredge reads "
            f"schema {SCHEMA_VERSION}; upgrade calbredge"
        )
    if version != SCHEMA_VERSION:
        raise ProjectError(f"{source}: unsupported schema_version {version!r}")


def _path(value: Any, key: str, source: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProjectError(f"{source}: {key} must be a path (a non-empty string)")
    if PurePosixPath(value).is_absolute() or Path(value).is_absolute():
        raise ProjectError(
            f"{source}: {key} must be relative to the project directory, not {value!r}"
        )
    return value


def _targets(value: Any, source: Path) -> tuple[Target, ...]:
    if not isinstance(value, list) or not value:
        raise ProjectError(f"{source}: targets must be a list with one entry")
    if len(value) > MAX_TARGETS:
        raise ProjectError(
            f"{source}: {len(value)} targets, but this calbredge supports {MAX_TARGETS}; "
            "upgrade calbredge for multi-target projects"
        )
    targets = []
    for i, entry in enumerate(value):
        where = f"targets[{i}]"
        if not isinstance(entry, dict):
            raise ProjectError(f"{source}: {where} must be a mapping")
        unknown = sorted(str(k) for k in entry if k not in _TARGET_KEYS)
        if unknown:
            raise ProjectError(f"{source}: unknown key(s) in {where}: {', '.join(unknown)}")
        runtime = entry.get("runtime")
        if not isinstance(runtime, str) or not is_runtime_label(runtime):
            raise ProjectError(
                f"{source}: {where}.runtime must be a runtime label, e.g. tensorrt-fp16"
            )
        baseline = entry.get("baseline")
        targets.append(
            Target(
                runtime=runtime,
                baseline=None if baseline is None else _path(baseline, f"{where}.baseline", source),
            )
        )
    return tuple(targets)


def _write_atomic(file: Path, text: str) -> None:
    """Write via a temporary file in the same directory and a rename: never half-written."""
    fd, tmp = tempfile.mkstemp(dir=file.parent, prefix=f".{file.name}.", suffix=".partial")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, file)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def render_project(project: Project) -> str:
    """The file's text: YAML with every string quoted as YAML needs, in a fixed key order."""
    return yaml.safe_dump(project.to_dict(), sort_keys=False, default_flow_style=False)


def write_project(project: Project, *, overwrite: bool = False) -> Path:
    """Write the project file under project.root/.calbredge/, atomically; return its path."""
    # Validate what is written exactly as a reader would.
    _parse(yaml.safe_load(render_project(project)), root=project.root, source=project.file)
    if project.file.exists() and not overwrite:
        raise ProjectError(f"{project.file} already exists")
    project.file.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(project.file, render_project(project))
    return project.file


def set_baseline(file: Path, baseline: str, *, target: int = 0) -> Project:
    """Point one target at a new baseline declaration, atomically; return the updated project.

    The file is re-read first, so a change made since it was loaded is kept, and the
    pointer is the only thing that changes. The previous declaration is untouched.
    """
    project = read_project(file)
    _path(baseline, f"targets[{target}].baseline", file)
    targets = list(project.targets)
    targets[target] = replace(targets[target], baseline=baseline)
    updated = replace(project, targets=tuple(targets))
    _write_atomic(file, render_project(updated))
    return updated
