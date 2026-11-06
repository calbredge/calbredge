"""The tier tables in docs/SUPPORTED_PLATFORMS.md and docs/QUICKSTART.md agree with
protocol/calbredge_protocol/support_matrix.py, and with each other.

Each doc row names a target, runtime and precision; ROWS maps it to the matrix lookup it
stands for. A doc row the map doesn't know, a tier that disagrees with the matrix, or a matrix
row no doc row covers fails, so neither partner-facing table can drift from what the gate does.
"""

from pathlib import Path

import pytest
from calbredge_protocol.support_matrix import ANY, SUPPORT_MATRIX, lookup, status_of

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "SUPPORTED_PLATFORMS.md"
QUICKSTART = ROOT / "docs" / "QUICKSTART.md"
# Each document and the heading its tier table sits under.
TABLES = {DOC: "## The tiers", QUICKSTART: "## What's supported today"}

# (target, runtime) in the doc -> (runtime_family, execution_provider, device_class)
ROWS = {
    ("NVIDIA Jetson", "TensorRT"): ("tensorrt", None, "jetson"),
    ("NVIDIA Jetson", "ONNX Runtime (CUDA)"): ("onnxruntime", "cuda", "jetson"),
    ("x86_64 Linux", "ONNX Runtime (CPU)"): ("onnxruntime", "cpu", "x86_64"),
    ("x86_64 Linux", "ONNX Runtime (CUDA)"): ("onnxruntime", "cuda", "x86_64"),
    ("x86_64 Linux", "TensorRT"): ("tensorrt", None, "x86_64"),
    ("Rockchip RK3588", "RKNN"): ("rknn", None, "rk3588"),
    ("Hailo-8", "HailoRT"): ("hailort", None, "hailo8"),
}
# The matrix has no "planned" status: a planned target is unsupported, or has no row yet.
TIER = {"Supported": {"supported"}, "Results only": {"results_only"}, "Planned": {"unsupported"}}
PRECISIONS = ("fp32", "fp16", "int8")
# Matrix rows the doc deliberately leaves out: not a target (a workload type), or not planned.
NOT_LISTED = {("tflite", None)}


def _doc_rows(doc: Path = DOC) -> list[tuple[str, str, str, str]]:
    section = doc.read_text().split(TABLES[doc], 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0] not in ("Target", "") and not set(cells[0]) <= set("-"):
            rows.append(tuple(cells))
    return rows


def _precisions(cell: str) -> tuple[str, ...]:
    return PRECISIONS if cell == "any" else tuple(p.strip().lower() for p in cell.split(","))


def test_the_doc_has_a_tier_table():
    assert len(_doc_rows()) >= len(ROWS)


def test_quickstarts_table_has_the_same_rows():
    assert _doc_rows(QUICKSTART) == _doc_rows(DOC)


@pytest.mark.parametrize(
    "target,runtime,precision,tier", [row for doc in TABLES for row in _doc_rows(doc)]
)
def test_each_doc_row_matches_the_matrix(target, runtime, precision, tier):
    assert (target, runtime) in ROWS, f"add ({target!r}, {runtime!r}) to ROWS"
    assert tier in TIER, tier
    family, provider, device = ROWS[(target, runtime)]
    for p in _precisions(precision):
        row = lookup(family, provider, "linux", device, precision=p)
        assert status_of(row) in TIER[tier], (target, runtime, p, status_of(row))


@pytest.mark.parametrize("doc", list(TABLES), ids=lambda d: d.name)
def test_every_matrix_row_is_in_the_doc(doc):
    covered = {
        (ROWS[(t, r)], p) for t, r, precision, _ in _doc_rows(doc) for p in _precisions(precision)
    }
    for row in SUPPORT_MATRIX:
        if row.workload_type != ANY or (row.runtime_family, row.execution_provider) in NOT_LISTED:
            continue
        key = (row.runtime_family, row.execution_provider, row.device_class)
        if row.precision == ANY:  # listed at any precision (RK3588 lists the planned ones)
            assert any(k == key for k, _ in covered), f"{doc.name} has no {key} row"
        else:
            assert (key, row.precision) in covered, (
                f"{doc.name} has no row for {key} {row.precision}"
            )
