"""The supported interface for the gate package.

Gate code reaches the open agent only through this module; tests/open/test_open_gate_boundary.py
fails on any other import. Every name is a plain re-export of open code under a public name
and adds no behaviour, except ``default_bundle_dir``, which reads the directory when called.
"""

from pathlib import Path

import calbredge as _calbredge
from calbredge import bundle as _bundle
from calbredge import cli as _cli
from calbredge import clock_state as _clock_state
from calbredge import doctor as _doctor
from calbredge import hasher as _hasher
from calbredge import host_v2 as _host_v2
from calbredge import init as _init
from calbredge import inspect_v2 as _inspect_v2
from calbredge import project as _project
from calbredge import runtime_probe as _runtime_probe
from calbredge import term as _term

__all__ = [
    "BUNDLE_DIR_HELP",
    "DEFAULT_PROTOCOL_NAME",
    "PROJECT_DIR",
    "InitError",
    "InitOutcome",
    "InitRequest",
    "ProbeResult",
    "ProjectError",
    "SpacedCommand",
    "Style",
    "__version__",
    "build_run_v2",
    "classify_host",
    "compute_model_hash",
    "default_branch",
    "default_bundle_dir",
    "detect_platform",
    "execute_benchmark",
    "find_project_file",
    "git_toplevel",
    "git_tracked",
    "git_user_name",
    "init_rerun_command",
    "inspect_model",
    "jetson_pin_command",
    "load_project",
    "load_protocol_or_default",
    "needs_probe",
    "probe",
    "put_init_file",
    "read_project",
    "result_json_text",
    "set_baseline",
    "write_run_bundle",
]

#: Supported interface for the gate package: the installed calbredge version.
__version__ = _calbredge.__version__

# --- the benchmark core (calbredge/cli.py) ----------------------------------------------------

#: Supported interface for the gate package: run one full benchmark pass.
execute_benchmark = _cli._execute_benchmark
#: Supported interface for the gate package: a finished run's v2 result and its errors.
build_run_v2 = _cli._build_run_v2
#: Supported interface for the gate package: write a finished run's bundle.
write_run_bundle = _cli._write_run_bundle
#: Supported interface for the gate package: load a protocol file, or the bundled default.
load_protocol_or_default = _cli._load_protocol_or_default
#: Supported interface for the gate package: the click command class the CLI's commands use.
SpacedCommand = _cli._SpacedCommand
#: Supported interface for the gate package: the help text of the --bundle-dir option.
BUNDLE_DIR_HELP = _cli._BUNDLE_DIR_HELP
#: Supported interface for the gate package: the bundled default protocol's file name.
DEFAULT_PROTOCOL_NAME = _cli._DEFAULT_PROTOCOL_NAME


def default_bundle_dir() -> Path:
    """Supported interface for the gate package: where run bundles go by default, read now."""
    return _cli.DEFAULT_BUNDLE_DIR


# --- run bundles, model identity, terminal output ---------------------------------------------

#: Supported interface for the gate package: a bundle's result.json text.
result_json_text = _bundle.result_json_text
#: Supported interface for the gate package: a model file's hash.
compute_model_hash = _hasher.compute_model_hash
#: Supported interface for the gate package: terminal styling of statuses and verdicts.
Style = _term.Style

# --- the host ---------------------------------------------------------------------------------

#: Supported interface for the gate package: which supported platform this host is.
detect_platform = _clock_state.detect_platform
#: Supported interface for the gate package: the command that pins a Jetson's clocks.
jetson_pin_command = _doctor.jetson_pin_command
#: Supported interface for the gate package: the host's support-matrix classification.
classify_host = _host_v2.classify_host
#: Supported interface for the gate package: the git repository root of a path.
git_toplevel = _host_v2.git_toplevel
#: Supported interface for the gate package: whether git tracks a path.
git_tracked = _host_v2.git_tracked
#: Supported interface for the gate package: git's configured user name.
git_user_name = _host_v2.git_user_name
#: Supported interface for the gate package: inspect a model against a protocol.
inspect_model = _inspect_v2.inspect_model

# --- the runtime probe ------------------------------------------------------------------------

#: Supported interface for the gate package: whether a runtime is probed before a benchmark.
needs_probe = _runtime_probe.needs_probe
#: Supported interface for the gate package: run one inference in a child process.
probe = _runtime_probe.probe
#: Supported interface for the gate package: what a probe's child printed and how it ended.
ProbeResult = _runtime_probe.ProbeResult

# --- init hooks (calbredge/init.py) -----------------------------------------------------------

#: Supported interface for the gate package: the error that makes init refuse.
InitError = _init.InitError
#: Supported interface for the gate package: what an init hook is asked to set up.
InitRequest = _init.InitRequest
#: Supported interface for the gate package: what init did, for its summary.
InitOutcome = _init.Outcome
#: Supported interface for the gate package: the branch a repository's pushes go to.
default_branch = _init.default_branch
#: Supported interface for the gate package: write one of init's files and record it.
put_init_file = _init._put
#: Supported interface for the gate package: the command that measures a project again.
init_rerun_command = _init.rerun_command

# --- the project file -------------------------------------------------------------------------

#: Supported interface for the gate package: the project directory's name.
PROJECT_DIR = _project.PROJECT_DIR
#: Supported interface for the gate package: the error a project file raises.
ProjectError = _project.ProjectError
#: Supported interface for the gate package: find the nearest project file.
find_project_file = _project.find_project_file
#: Supported interface for the gate package: load the project a path belongs to.
load_project = _project.load_project
#: Supported interface for the gate package: read one project file.
read_project = _project.read_project
#: Supported interface for the gate package: move a target's baseline pointer.
set_baseline = _project.set_baseline
