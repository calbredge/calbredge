"""The v2 result schema's doc and its models must describe the same fields.

docs/RESULT_SCHEMA.md is the public contract; protocol/calbredge_protocol/result_v2.py
implements it. A field added to one but not the other fails here, in both directions.
"""

import re
from pathlib import Path

import pytest
from calbredge_protocol.result_v2 import (
    Candidate,
    Capabilities,
    Metrics,
    ProtocolInfo,
    Quality,
    ResultV2,
    Target,
    Workload,
)

pytestmark = pytest.mark.unit

DOCS = Path(__file__).resolve().parents[2] / "docs"
RESULT_SCHEMA = (DOCS / "RESULT_SCHEMA.md").read_text()


def _section(text: str, heading_prefix: str) -> str:
    """Body of the '### <prefix>...' section, up to the next '### ' or '## '."""
    match = re.search(
        rf"^### {re.escape(heading_prefix)}.*?$(.*?)(?=^### |^## |\Z)", text, re.M | re.S
    )
    assert match, f"no section starting '### {heading_prefix}'"
    return match.group(1)


def _table_names(section: str, header_first_cell: str) -> set[str]:
    """Backticked names in the first column of tables whose header starts with
    '| <header_first_cell> |'. Other tables in the section are ignored."""
    names: set[str] = set()
    in_table = False
    for line in section.splitlines():
        if line.startswith(f"| {header_first_cell} |"):
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("|"):
            names.update(re.findall(r"`([a-z0-9_]+)`", line.split("|")[1]))
        else:
            in_table = False
    return names


SECTION_MODELS = [
    ("3.2 `workload`", Workload),
    ("3.3 `candidate`", Candidate),
    ("3.4 `target`", Target),
    ("3.5 `protocol`", ProtocolInfo),
    ("3.6 `capabilities`", Capabilities),
    ("3.7 `metrics`", Metrics),
    ("3.9 `quality`", Quality),
]


@pytest.mark.parametrize(("heading", "model"), SECTION_MODELS)
def test_result_schema_sections_match_models(heading, model):
    documented = _table_names(_section(RESULT_SCHEMA, heading), "Field")
    assert documented == set(model.model_fields)


def test_result_schema_top_level_matches_model():
    documented = _table_names(_section(RESULT_SCHEMA, "3.1 Top level"), "Field")
    section_fields = {
        "workload",
        "candidate",
        "target",
        "protocol",
        "capabilities",
        "metrics",
        "quality",
    }
    scalar_fields = set(ResultV2.model_fields) - section_fields - {"workload_metrics"}
    assert documented == scalar_fields
    assert "### 3.8 `workload_metrics`" in RESULT_SCHEMA
