"""Open code never imports gate code.

The gate is its own package, calbredge_gate; agent/calbredge and
protocol/calbredge_protocol are open. No module of the open packages may import anything
under the calbredge_gate name, whatever the submodule. Gate modules may import open ones,
never the reverse. The gate's commands reach the open CLI through the
``calbredge.commands`` entry-point group, which only calbredge-gate registers, not an import.
"""

import ast
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
OPEN_PACKAGES = {
    "calbredge": ROOT / "agent" / "calbredge",
    "calbredge_protocol": ROOT / "protocol" / "calbredge_protocol",
}
# The gate package's import name: everything under it is gate code.
GATE_PACKAGE = "calbredge_gate"

OPEN = frozenset(
    {
        "calbredge",
        "calbredge.affinity",
        "calbredge.bundle",
        "calbredge.cli",
        "calbredge.clock_state",
        "calbredge.collector",
        "calbredge.converter",
        "calbredge.debug",
        "calbredge.example_model",
        "calbredge.extension",  # the only way gate code reaches the open agent
        "calbredge.doctor",
        "calbredge.fingerprint",
        "calbredge.hasher",
        "calbredge.host_v2",
        "calbredge.init",
        "calbredge.inspect_v2",
        "calbredge.onnx_contract",
        "calbredge.project",
        "calbredge.result_builder",
        "calbredge.runtime_probe",
        "calbredge.term",
        "calbredge.uploader",
        "calbredge.validate_rknn",
        "calbredge_protocol",
        "calbredge_protocol.adapters",
        "calbredge_protocol.autodetect",
        "calbredge_protocol.bundled",
        "calbredge_protocol.canonical",  # results are digested with it; so will signed decisions be
        "calbredge_protocol.derive_v2",
        "calbredge_protocol.engine_cache",
        "calbredge_protocol.graph_variant",
        "calbredge_protocol.iterative_policy",
        "calbredge_protocol.loader",
        "calbredge_protocol.mock_adapter",
        "calbredge_protocol.models",
        "calbredge_protocol.onnx_adapter",
        "calbredge_protocol.power_meter",
        "calbredge_protocol.power_window",
        "calbredge_protocol.protocols",
        "calbredge_protocol.result_v2",
        "calbredge_protocol.rknn_adapter",
        "calbredge_protocol.rknn_artifact",
        "calbredge_protocol.stats",
        # which measurements are trustworthy; the gate enforces it
        "calbredge_protocol.support_matrix",
        "calbredge_protocol.tensorrt_adapter",
        "calbredge_protocol.tflite_adapter",
        "calbredge_protocol.throughput",
    }
)


def _modules(packages: dict[str, Path] = OPEN_PACKAGES) -> dict[str, Path]:
    found = {}
    for package, root in packages.items():
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            parts = [package, *path.relative_to(root).with_suffix("").parts]
            if parts[-1] == "__init__":
                parts.pop()
            found[".".join(parts)] = path
    return found


def _imported_modules(name: str, path: Path) -> Iterator[tuple[int, str]]:
    """Every module a file can import, including imports inside functions.

    ``from x import y`` yields both ``x`` and ``x.y``, since y may be a
    submodule (``from calbredge_gate import some_module``). ``importlib.import_module``
    with a literal name counts too.
    """
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")[: len(package.split(".")) - (node.level - 1)]
                base = ".".join([*anchor, base] if base else anchor)
            yield node.lineno, base
            for alias in node.names:
                yield node.lineno, f"{base}.{alias.name}"
        elif (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", getattr(node.func, "id", None)) == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            yield node.lineno, node.args[0].value


def _is_gate(target: str) -> bool:
    return target.split(".")[0] == GATE_PACKAGE


def test_open_modules_never_import_gate_modules():
    violations = [
        f"{path.relative_to(ROOT)}:{line}  {name} -> {target}"
        for name, path in _modules().items()
        for line, target in _imported_modules(name, path)
        if _is_gate(target)
    ]
    assert violations == [], (
        "open module imports gate code (move the code, or reclassify deliberately):\n  "
        + "\n  ".join(violations)
    )


def test_the_rule_catches_a_violation(tmp_path):
    bad = tmp_path / "cli.py"
    # Invented submodule names: the rule is the package prefix, not a list of modules.
    bad.write_text(
        "from calbredge_gate.invented_a import evaluate\n"
        "def f():\n    from calbredge_gate import invented_b\n"
        "import importlib\nimportlib.import_module('calbredge_gate.invented_c')\n"
    )
    targets = {target for _, target in _imported_modules("calbredge.cli", bad)}
    assert {
        "calbredge_gate",
        "calbredge_gate.invented_a",
        "calbredge_gate.invented_b",
        "calbredge_gate.invented_c",
    } <= {target for target in targets if _is_gate(target)}


def test_importing_the_open_cli_loads_no_gate_module():
    # A fresh interpreter: this test process has already imported gate code.
    code = (
        "import sys, calbredge.cli, calbredge.init; "
        f"print(sorted(m for m in sys.modules if m.split('.')[0] == {GATE_PACKAGE!r}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "[]"


def test_no_open_test_imports_or_waits_for_the_gate():
    """tests/open runs without calbredge-gate: nothing here may import it or skip for it.

    A test that needs the gate, or that reads this repository's own tooling, belongs outside
    tests/open. The open job (tests/open/conftest.py)
    fails a run in which gate code was loaded; this holds the source to the same rule, so a
    guarded import (``pytest.importorskip``, a ``try``) is caught where it is written.
    """
    here = Path(__file__).parent
    offenders = [
        f"{path.relative_to(ROOT)}:{line}  {target}"
        for path in sorted(here.rglob("*.py"))
        for line, target in _imported_modules(f"tests.open.{path.stem}", path)
        if _is_gate(target) or target.startswith(("tests.gate", "tests.repo", "tests.api"))
    ]
    offenders += [
        f"{path.relative_to(ROOT)}  importorskip"
        for path in sorted(here.rglob("*.py"))
        if path != Path(__file__) and 'importorskip("calbredge_gate' in path.read_text()
    ]
    assert offenders == []
