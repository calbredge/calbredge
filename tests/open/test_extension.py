"""``calbredge.extension``: the supported interface for the gate package.

Every name is open code re-exported unchanged under a public name; the bundle directory is
the one function, read when called. The gate package's own tests check that it uses nothing
else from the open agent.
"""

import importlib
import types

import pytest
from calbredge import extension

from tests.open.test_open_gate_boundary import OPEN

pytestmark = pytest.mark.unit


def test_every_name_the_extension_exports_resolves():
    # ruff keeps __all__ in its own order (RUF022); here, only that nothing is listed twice.
    assert len(set(extension.__all__)) == len(extension.__all__)
    for name in extension.__all__:
        assert getattr(extension, name) is not None, name


def test_the_extension_exports_everything_public_it_defines():
    public = {
        name
        for name, value in vars(extension).items()
        if not name.startswith("_") and not isinstance(value, types.ModuleType) and name != "Path"
    }
    assert public | {"__version__"} == set(extension.__all__)


def test_the_extension_re_exports_open_code_unchanged():
    sources = {
        "execute_benchmark": ("calbredge.cli", "_execute_benchmark"),
        "build_run_v2": ("calbredge.cli", "_build_run_v2"),
        "write_run_bundle": ("calbredge.cli", "_write_run_bundle"),
        "load_protocol_or_default": ("calbredge.cli", "_load_protocol_or_default"),
        "SpacedCommand": ("calbredge.cli", "_SpacedCommand"),
        "BUNDLE_DIR_HELP": ("calbredge.cli", "_BUNDLE_DIR_HELP"),
        "DEFAULT_PROTOCOL_NAME": ("calbredge.cli", "_DEFAULT_PROTOCOL_NAME"),
        "__version__": ("calbredge", "__version__"),
        "result_json_text": ("calbredge.bundle", "result_json_text"),
        "compute_model_hash": ("calbredge.hasher", "compute_model_hash"),
        "Style": ("calbredge.term", "Style"),
        "detect_platform": ("calbredge.clock_state", "detect_platform"),
        "jetson_pin_command": ("calbredge.doctor", "jetson_pin_command"),
        "classify_host": ("calbredge.host_v2", "classify_host"),
        "git_toplevel": ("calbredge.host_v2", "git_toplevel"),
        "git_tracked": ("calbredge.host_v2", "git_tracked"),
        "git_user_name": ("calbredge.host_v2", "git_user_name"),
        "inspect_model": ("calbredge.inspect_v2", "inspect_model"),
        "needs_probe": ("calbredge.runtime_probe", "needs_probe"),
        "probe": ("calbredge.runtime_probe", "probe"),
        "ProbeResult": ("calbredge.runtime_probe", "ProbeResult"),
        "InitError": ("calbredge.init", "InitError"),
        "InitRequest": ("calbredge.init", "InitRequest"),
        "InitOutcome": ("calbredge.init", "Outcome"),
        "default_branch": ("calbredge.init", "default_branch"),
        "put_init_file": ("calbredge.init", "_put"),
        "init_rerun_command": ("calbredge.init", "rerun_command"),
        "PROJECT_DIR": ("calbredge.project", "PROJECT_DIR"),
        "ProjectError": ("calbredge.project", "ProjectError"),
        "find_project_file": ("calbredge.project", "find_project_file"),
        "load_project": ("calbredge.project", "load_project"),
        "read_project": ("calbredge.project", "read_project"),
        "set_baseline": ("calbredge.project", "set_baseline"),
    }
    assert set(sources) | {"default_bundle_dir"} == set(extension.__all__)
    for name, (module, attribute) in sources.items():
        assert module in OPEN, module
        assert getattr(extension, name) is getattr(importlib.import_module(module), attribute), name


def test_the_default_bundle_directory_is_read_when_called(monkeypatch, tmp_path):
    # conftest patches calbredge.cli.DEFAULT_BUNDLE_DIR once for the session; the gate must see it.
    monkeypatch.setattr("calbredge.cli.DEFAULT_BUNDLE_DIR", tmp_path / "elsewhere")
    assert extension.default_bundle_dir() == tmp_path / "elsewhere"
